// Dynamic mesh-strategy experiment for the adaptive Hartree treecode.
//
// This executable deliberately includes the validated implementation instead
// of copying its quadrature, hierarchy, MAC, multipole, and evaluation code.
// TREECODE_NO_MAIN keeps the normal tree_exact entry point unchanged.
#define TREECODE_REUSE_EXPERIMENT 1
#define TREECODE_NO_MAIN 1
#define TREECODE_DYNAMIC_MESH_MODES 1
#include "compare_exact.cpp"

#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace mesh_modes {

enum class Mode
{
  frozen,
  every_h,
  periodic_h,
  residual_h,
  moving_r,
  hybrid_hr
};

const char* mode_name(Mode mode)
{
  switch (mode) {
    case Mode::frozen: return "FROZEN";
    case Mode::every_h: return "EVERY_H";
    case Mode::periodic_h: return "PERIODIC_H";
    case Mode::residual_h: return "RESIDUAL_H";
    case Mode::moving_r: return "MOVING_R";
    case Mode::hybrid_hr: return "HYBRID_HR";
  }
  return "UNKNOWN";
}

std::string upper(std::string value)
{
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  return value;
}

Mode parse_mode(const std::string& value)
{
  const std::string normalized = upper(value);
  if (normalized == "FROZEN") return Mode::frozen;
  if (normalized == "EVERY_H") return Mode::every_h;
  if (normalized == "PERIODIC_H") return Mode::periodic_h;
  if (normalized == "RESIDUAL_H") return Mode::residual_h;
  if (normalized == "MOVING_R") return Mode::moving_r;
  if (normalized == "HYBRID_HR") return Mode::hybrid_hr;
  throw std::invalid_argument(
      "Invalid --mode: " + value
      + " (use FROZEN, EVERY_H, PERIODIC_H, RESIDUAL_H, MOVING_R, "
        "or HYBRID_HR)");
}

bool parse_bool(const std::string& value, const char* name)
{
  const std::string normalized = upper(value);
  if (normalized == "1" || normalized == "TRUE"
      || normalized == "YES" || normalized == "ON") {
    return true;
  }
  if (normalized == "0" || normalized == "FALSE"
      || normalized == "NO" || normalized == "OFF") {
    return false;
  }
  throw std::invalid_argument(
      std::string("Invalid ") + name + ": " + value
      + " (use 0/1 or false/true)");
}

struct Options
{
  std::string mesh_file = "D.mesh";
  Mode mode = Mode::frozen;
  int initial_refinements = 2;
  int quadrature_accuracy = 6;
  int expansion_order = 4;
  double mac_theta = 0.5;

  int sequence_steps = 6;
  double sequence_decay = 0.5;
  double initial_amplitude = 0.5;
  double initial_center_shift = 0.8;

  int update_interval = 2;
  int max_reuse_steps = 4;
  double residual_trigger = 2.0e-2;
  double adapt_tolerance = 1.0e-6;
  int refine_steps = 1;
  bool refine_only = false;
  bool base_radial_adaptation = true;

  double monitor_strength = 10.0;
  double monitor_exponent = 0.5;
  double monitor_floor = 0.1;
  int monitor_smooth_steps = 3;
  double move_relaxation = 0.2;
  int move_iterations = 5;
  double max_displacement_ratio = 0.2;
  double minimum_quality = 0.05;

  std::string output_file = "mesh_mode_results.tsv";
};

void print_usage(std::ostream& output)
{
  output
      << "Usage: tree_mesh_modes MESH_FILE [options]\n\n"
      << "Core options:\n"
      << "  --mode MODE                 FROZEN, EVERY_H, PERIODIC_H, "
         "RESIDUAL_H, MOVING_R, HYBRID_HR\n"
      << "  --initial-refinements N     Global refinements (default 2)\n"
      << "  --p N                       Quadrature accuracy (default 6)\n"
      << "  --P N                       Multipole order (default 4)\n"
      << "  --theta X                   MAC theta (default 0.5)\n"
      << "Sequence options:\n"
      << "  --steps N --decay X --initial-amplitude X --initial-shift X\n"
      << "h-adaptation options:\n"
      << "  --period N --max-reuse N --residual-trigger X\n"
      << "  --adapt-tol X --refine-steps N --refine-only 0|1\n"
      << "Moving-mesh options:\n"
      << "  --monitor-strength X --monitor-exponent X --monitor-floor X\n"
      << "  --monitor-smooth N --move-relax X --move-iters N\n"
      << "  --max-displacement X --min-quality X\n"
      << "Other:\n"
      << "  --base-adapt 0|1 --output FILE\n";
}

Options parse_options(int argc, char* argv[])
{
  if (argc < 2) {
    print_usage(std::cerr);
    throw std::invalid_argument("MESH_FILE is required");
  }
  if (std::string(argv[1]) == "--help") {
    print_usage(std::cout);
    std::exit(0);
  }

  Options options;
  options.mesh_file = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string key = argv[index];
    if (key == "--help") {
      print_usage(std::cout);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("Missing value after " + key);
    }
    const std::string value = argv[++index];
    if (key == "--mode") options.mode = parse_mode(value);
    else if (key == "--initial-refinements")
      options.initial_refinements = parse_int(value.c_str(), key.c_str());
    else if (key == "--p")
      options.quadrature_accuracy = parse_int(value.c_str(), key.c_str());
    else if (key == "--P")
      options.expansion_order = parse_int(value.c_str(), key.c_str());
    else if (key == "--theta")
      options.mac_theta = parse_double(value.c_str(), key.c_str());
    else if (key == "--steps")
      options.sequence_steps = parse_int(value.c_str(), key.c_str());
    else if (key == "--decay")
      options.sequence_decay = parse_double(value.c_str(), key.c_str());
    else if (key == "--initial-amplitude")
      options.initial_amplitude = parse_double(value.c_str(), key.c_str());
    else if (key == "--initial-shift")
      options.initial_center_shift = parse_double(value.c_str(), key.c_str());
    else if (key == "--period")
      options.update_interval = parse_int(value.c_str(), key.c_str());
    else if (key == "--max-reuse")
      options.max_reuse_steps = parse_int(value.c_str(), key.c_str());
    else if (key == "--residual-trigger")
      options.residual_trigger = parse_double(value.c_str(), key.c_str());
    else if (key == "--adapt-tol")
      options.adapt_tolerance = parse_double(value.c_str(), key.c_str());
    else if (key == "--refine-steps")
      options.refine_steps = parse_int(value.c_str(), key.c_str());
    else if (key == "--refine-only")
      options.refine_only = parse_bool(value, key.c_str());
    else if (key == "--base-adapt")
      options.base_radial_adaptation = parse_bool(value, key.c_str());
    else if (key == "--monitor-strength")
      options.monitor_strength = parse_double(value.c_str(), key.c_str());
    else if (key == "--monitor-exponent")
      options.monitor_exponent = parse_double(value.c_str(), key.c_str());
    else if (key == "--monitor-floor")
      options.monitor_floor = parse_double(value.c_str(), key.c_str());
    else if (key == "--monitor-smooth")
      options.monitor_smooth_steps = parse_int(value.c_str(), key.c_str());
    else if (key == "--move-relax")
      options.move_relaxation = parse_double(value.c_str(), key.c_str());
    else if (key == "--move-iters")
      options.move_iterations = parse_int(value.c_str(), key.c_str());
    else if (key == "--max-displacement")
      options.max_displacement_ratio = parse_double(value.c_str(), key.c_str());
    else if (key == "--min-quality")
      options.minimum_quality = parse_double(value.c_str(), key.c_str());
    else if (key == "--output") options.output_file = value;
    else throw std::invalid_argument("Unknown option: " + key);
  }

  if (options.initial_refinements < 0 || options.initial_refinements > 5)
    throw std::invalid_argument("--initial-refinements must be in [0,5]");
  if (options.quadrature_accuracy < 1
      || options.quadrature_accuracy
          > treecode_quadrature::maximum_algebraic_accuracy)
    throw std::invalid_argument("--p is outside the supported range");
  if (options.expansion_order < 1
      || options.expansion_order > treecode_multipole::maximum_order)
    throw std::invalid_argument("--P is outside the supported range");
  if (!(options.mac_theta > 0.0 && options.mac_theta < 1.0))
    throw std::invalid_argument("--theta must lie in (0,1)");
  if (options.sequence_steps < 1 || options.sequence_steps > 1000)
    throw std::invalid_argument("--steps must lie in [1,1000]");
  if (!(options.sequence_decay > 0.0 && options.sequence_decay <= 1.0))
    throw std::invalid_argument("--decay must lie in (0,1]");
  if (options.update_interval < 1 || options.max_reuse_steps < 1)
    throw std::invalid_argument("--period and --max-reuse must be positive");
  if (!(options.residual_trigger > 0.0)
      || !(options.adapt_tolerance > 0.0))
    throw std::invalid_argument(
        "--residual-trigger and --adapt-tol must be positive");
  if (options.refine_steps < 1 || options.refine_steps > 8)
    throw std::invalid_argument("--refine-steps must lie in [1,8]");
  if (!(options.monitor_strength >= 0.0)
      || !(options.monitor_exponent > 0.0)
      || !(options.monitor_floor > 0.0))
    throw std::invalid_argument("Moving-mesh monitor parameters are invalid");
  if (options.monitor_smooth_steps < 0 || options.move_iterations < 1)
    throw std::invalid_argument("Moving-mesh iteration counts are invalid");
  if (!(options.move_relaxation > 0.0
        && options.move_relaxation <= 1.0)
      || !(options.max_displacement_ratio > 0.0
           && options.max_displacement_ratio <= 1.0)
      || !(options.minimum_quality > 0.0
           && options.minimum_quality < 1.0))
    throw std::invalid_argument("Moving-mesh safety parameters are invalid");
  if (options.output_file.empty())
    throw std::invalid_argument("--output must not be empty");
  return options;
}

void initialize_fem_space(
    FEMSpace<double, DIM>& fem_space,
    RegularMesh<DIM>& mesh,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements)
{
  fem_space.reinit(mesh, template_elements);
  const int element_count = mesh.n_geometry(DIM);
  fem_space.element().resize(element_count);
  for (int index = 0; index < element_count; ++index) {
    const int vertex_count = mesh.geometry(DIM, index).n_vertex();
    int template_index = -1;
    if (vertex_count == 4) template_index = 0;
    else if (vertex_count == 5) template_index = 1;
    else if (vertex_count == 7) template_index = 2;
    else throw std::runtime_error("Unsupported regular macroelement");
    fem_space.element(index).reinit(fem_space, index, template_index);
  }
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();
}

double element_diameter(
    const std::vector<AFEPack::Point<DIM>>& vertices)
{
  double diameter = 0.0;
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    for (std::size_t j = i + 1; j < vertices.size(); ++j) {
      diameter = std::max(diameter, (vertices[i] - vertices[j]).length());
    }
  }
  return diameter;
}

// This is a source-projection residual for the integral treecode:
// eta_K^2 = h_K^2 ||rho - mean_K(rho)||^2_K.  It measures whether the
// current source mesh resolves rho.  It is intentionally not advertised as
// the finite-element Poisson jump estimator, because the treecode currently
// stores no conforming nodal approximation of V_H.
double build_source_residual_indicator(
    Indicator<DIM>& indicator,
    RegularMesh<DIM>& mesh,
    FEMSpace<double, DIM>& fem_space,
    int quadrature_accuracy)
{
  indicator.reinit(mesh);
  const int element_count = fem_space.n_element();
  std::vector<double> local_residual(
      static_cast<std::size_t>(element_count), 0.0);
  std::vector<double> local_source_norm(
      static_cast<std::size_t>(element_count), 0.0);

  #pragma omp parallel for schedule(static) default(none) \
      shared(local_residual, local_source_norm, fem_space, mesh) \
      firstprivate(element_count, quadrature_accuracy)
  for (int i = 0; i < element_count; ++i) {
    const Element<double, DIM>& element = fem_space.element(i);
    const GeometryBM& geometry = element.geometry();
    std::vector<AFEPack::Point<DIM>> vertices(
        static_cast<std::size_t>(geometry.n_vertex()));
    for (int vertex = 0; vertex < geometry.n_vertex(); ++vertex) {
      vertices[static_cast<std::size_t>(vertex)] =
          mesh.point()[geometry.vertex()[vertex]];
    }
    const auto rule = treecode_quadrature::build_physical_rule(
        vertices, quadrature_accuracy);
    double volume = 0.0;
    double source_integral = 0.0;
    for (const auto& entry : rule) {
      const double source = f(entry.point);
      volume += entry.weight;
      source_integral += entry.weight * source;
      local_source_norm[static_cast<std::size_t>(i)] +=
          entry.weight * source * source;
    }
    const double mean = source_integral / std::max(
        volume, std::numeric_limits<double>::min());
    double variance = 0.0;
    for (const auto& entry : rule) {
      const double difference = f(entry.point) - mean;
      variance += entry.weight * difference * difference;
    }
    const double diameter = element_diameter(vertices);
    local_residual[static_cast<std::size_t>(i)] =
        diameter * diameter * variance;
  }

  const double residual_sum = std::accumulate(
      local_residual.begin(), local_residual.end(), 0.0);
  const double source_norm_sum = std::accumulate(
      local_source_norm.begin(), local_source_norm.end(), 0.0);
  const double normalization = std::max(
      source_norm_sum, std::numeric_limits<double>::min());
  for (int i = 0; i < element_count; ++i) {
    const int geometry_index = fem_space.element(i).geometry().index();
    indicator[geometry_index] =
        local_residual[static_cast<std::size_t>(i)] / normalization;
  }
  return std::sqrt(residual_sum / normalization);
}

double apply_h_adaptation(
    std::unique_ptr<IrregularMesh<DIM>>& irregular_mesh,
    FEMSpace<double, DIM>& fem_space,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements,
    const Options& options,
    bool use_source_residual,
    double tolerance,
    int refine_steps,
    double* relative_residual,
    bool force_refine_only = false)
{
  RegularMesh<DIM>& mesh = irregular_mesh->regularMesh();
  Indicator<DIM> indicator(mesh);
  const double start = omp_get_wtime();
  double residual = 0.0;
  if (use_source_residual) {
    residual = build_source_residual_indicator(
        indicator, mesh, fem_space, options.quadrature_accuracy);
  } else {
    get_indicator(
        indicator, mesh, fem_space, options.quadrature_accuracy, false);
  }
  if (relative_residual != nullptr) *relative_residual = residual;

  MeshAdaptor<DIM> adaptor(*irregular_mesh);
  adaptor.setIndicator(indicator);
  adaptor.tolerence() = tolerance;
  adaptor.convergenceOrder() = 0.0;
  adaptor.refineStep() = refine_steps;
  adaptor.is_refine_only() = options.refine_only || force_refine_only;
  adaptor.adapt();
  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);
  initialize_fem_space(
      fem_space, irregular_mesh->regularMesh(), template_elements);
  return omp_get_wtime() - start;
}

struct MovingStats
{
  bool moved = false;
  double seconds = 0.0;
  double maximum_displacement = 0.0;
  double minimum_quality = 1.0;
  int accepted_iterations = 0;
  std::size_t fixed_vertices = 0;
  std::size_t movable_vertices = 0;
};

struct MeshGeometryCheckpoint
{
  using Vertex = HGeometry<0, DIM>;
  std::vector<Vertex*> vertices;
  std::vector<AFEPack::Point<DIM>> positions;
};

MeshGeometryCheckpoint capture_mesh_geometry(
    IrregularMesh<DIM>& irregular_mesh)
{
  MeshGeometryCheckpoint checkpoint;
  std::unordered_map<MeshGeometryCheckpoint::Vertex*, std::size_t> seen;
  IrregularMesh<DIM>::ActiveIterator element =
      irregular_mesh.beginActiveElement();
  IrregularMesh<DIM>::ActiveIterator end_element =
      irregular_mesh.endActiveElement();
  for (; element != end_element; ++element) {
    for (MeshGeometryCheckpoint::Vertex* vertex :
         element->h_element->vertex) {
      if (seen.emplace(vertex, checkpoint.vertices.size()).second) {
        checkpoint.vertices.push_back(vertex);
        checkpoint.positions.push_back(*vertex);
      }
    }
  }
  return checkpoint;
}

double restore_mesh_geometry(
    IrregularMesh<DIM>& irregular_mesh,
    const MeshGeometryCheckpoint& checkpoint)
{
  const double start = omp_get_wtime();
  if (checkpoint.vertices.size() != checkpoint.positions.size()) {
    throw std::runtime_error("Invalid moving-mesh geometry checkpoint");
  }
  RegularMesh<DIM>& regular_mesh = irregular_mesh.regularMesh();
  for (std::size_t i = 0; i < checkpoint.vertices.size(); ++i) {
    MeshGeometryCheckpoint::Vertex* vertex = checkpoint.vertices[i];
    if (vertex == nullptr || vertex->index < 0
        || static_cast<unsigned int>(vertex->index)
               >= regular_mesh.n_geometry(0)) {
      throw std::runtime_error(
          "Cannot restore an inactive moving-mesh vertex");
    }
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      (*vertex)[coordinate] = checkpoint.positions[i][coordinate];
      regular_mesh.point()[static_cast<std::size_t>(vertex->index)]
                          [coordinate] = checkpoint.positions[i][coordinate];
    }
  }
  return omp_get_wtime() - start;
}

double signed_six_volume(
    const AFEPack::Point<DIM>& a,
    const AFEPack::Point<DIM>& b,
    const AFEPack::Point<DIM>& c,
    const AFEPack::Point<DIM>& d)
{
  const AFEPack::Point<DIM> ab = b - a;
  const AFEPack::Point<DIM> ac = c - a;
  const AFEPack::Point<DIM> ad = d - a;
  return det3(ab, ac, ad);
}

double tetrahedron_quality(
    const AFEPack::Point<DIM>& a,
    const AFEPack::Point<DIM>& b,
    const AFEPack::Point<DIM>& c,
    const AFEPack::Point<DIM>& d)
{
  const std::array<AFEPack::Point<DIM>, 4> point{{a, b, c, d}};
  double edge_squared_sum = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      const double length = (point[i] - point[j]).length();
      edge_squared_sum += length * length;
    }
  }
  const double volume = std::abs(signed_six_volume(a, b, c, d)) / 6.0;
  if (!(volume > 0.0) || !(edge_squared_sum > 0.0)) return 0.0;
  return 12.0 * std::pow(3.0 * volume, 2.0 / 3.0)
      / edge_squared_sum;
}

MovingStats move_mesh(
    IrregularMesh<DIM>& irregular_mesh,
    FEMSpace<double, DIM>& fem_space,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements,
    const Options& options)
{
  const double start = omp_get_wtime();
  using Vertex = HGeometry<0, DIM>;
  std::vector<HElement<DIM>*> leaves;
  std::vector<Vertex*> vertices;
  std::unordered_map<Vertex*, std::size_t> vertex_index;
  IrregularMesh<DIM>::ActiveIterator element =
      irregular_mesh.beginActiveElement();
  IrregularMesh<DIM>::ActiveIterator end_element =
      irregular_mesh.endActiveElement();
  for (; element != end_element; ++element) {
    leaves.push_back(&(*element));
    if (element->h_element == nullptr
        || element->h_element->vertex.size() != 4) {
      throw std::runtime_error(
          "Moving mesh requires tetrahedral hierarchy leaves");
    }
    for (Vertex* vertex : element->h_element->vertex) {
      if (vertex_index.emplace(vertex, vertices.size()).second) {
        vertices.push_back(vertex);
      }
    }
  }
  std::vector<std::set<std::size_t>> adjacency(vertices.size());
  std::vector<std::vector<HElement<DIM>*>> incident_elements(
      vertices.size());
  for (HElement<DIM>* leaf : leaves) {
    const auto& element_vertices = leaf->h_element->vertex;
    for (Vertex* vertex : element_vertices) {
      incident_elements[vertex_index.at(vertex)].push_back(leaf);
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        const std::size_t a = vertex_index.at(element_vertices[i]);
        const std::size_t b = vertex_index.at(element_vertices[j]);
        adjacency[a].insert(b);
        adjacency[b].insert(a);
      }
    }
  }

  std::array<double, DIM> coordinate_minimum;
  std::array<double, DIM> coordinate_maximum;
  coordinate_minimum.fill(std::numeric_limits<double>::infinity());
  coordinate_maximum.fill(-std::numeric_limits<double>::infinity());
  for (const Vertex* vertex : vertices) {
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      coordinate_minimum[coordinate] = std::min(
          coordinate_minimum[coordinate], (*vertex)[coordinate]);
      coordinate_maximum[coordinate] = std::max(
          coordinate_maximum[coordinate], (*vertex)[coordinate]);
    }
  }
  std::vector<bool> fixed_vertex(vertices.size(), false);
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    fixed_vertex[i] = vertices[i]->bmark != 0;
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      const double extent = coordinate_maximum[coordinate]
          - coordinate_minimum[coordinate];
      const double boundary_tolerance =
          1.0e-12 * std::max(1.0, extent);
      fixed_vertex[i] = fixed_vertex[i]
          || std::abs((*vertices[i])[coordinate]
                      - coordinate_minimum[coordinate])
                 <= boundary_tolerance
          || std::abs((*vertices[i])[coordinate]
                      - coordinate_maximum[coordinate])
                 <= boundary_tolerance;
    }

    if (fixed_vertex[i]) continue;
    // A vertex on a hanging interface has an open incident-tetrahedron star:
    // moving it independently changes the total physical volume.  Detect
    // that constraint from the gradient of its incident volume and freeze
    // it.  For a conforming interior star the gradients cancel to roundoff.
    double local_volume = 0.0;
    double local_length = 0.0;
    for (HElement<DIM>* leaf : incident_elements[i]) {
      const auto& v = leaf->h_element->vertex;
      local_volume += std::abs(signed_six_volume(
          *v[0], *v[1], *v[2], *v[3])) / 6.0;
      for (Vertex* other : v) {
        if (other != vertices[i]) {
          local_length = std::max(
              local_length, (*other - *vertices[i]).length());
        }
      }
    }
    const double derivative_scale = local_volume
        / std::max(local_length, std::numeric_limits<double>::min());
    double gradient_squared = 0.0;
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      const double extent = coordinate_maximum[coordinate]
          - coordinate_minimum[coordinate];
      const double epsilon = 1.0e-7 * std::max(1.0, extent);
      const double original = (*vertices[i])[coordinate];
      (*vertices[i])[coordinate] = original + epsilon;
      double perturbed_volume = 0.0;
      for (HElement<DIM>* leaf : incident_elements[i]) {
        const auto& v = leaf->h_element->vertex;
        perturbed_volume += std::abs(signed_six_volume(
            *v[0], *v[1], *v[2], *v[3])) / 6.0;
      }
      (*vertices[i])[coordinate] = original;
      const double derivative =
          (perturbed_volume - local_volume) / epsilon;
      gradient_squared += derivative * derivative;
    }
    const double constraint_tolerance =
        1.0e-8 * std::max(derivative_scale, 1.0e-12);
    fixed_vertex[i] =
        std::sqrt(gradient_squared) > constraint_tolerance;
  }

  MovingStats stats;
  stats.fixed_vertices = static_cast<std::size_t>(std::count(
      fixed_vertex.begin(), fixed_vertex.end(), true));
  stats.movable_vertices = vertices.size() - stats.fixed_vertices;

  std::vector<double> monitor(vertices.size(), options.monitor_floor);
  double maximum_source = 0.0;
  for (Vertex* vertex : vertices) maximum_source = std::max(
      maximum_source, std::abs(f(*vertex)));
  maximum_source = std::max(
      maximum_source, std::numeric_limits<double>::min());
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    monitor[i] = std::pow(
        options.monitor_floor
            + options.monitor_strength
                * std::abs(f(*vertices[i])) / maximum_source,
        options.monitor_exponent);
  }
  for (int step = 0; step < options.monitor_smooth_steps; ++step) {
    std::vector<double> smoothed = monitor;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      if (adjacency[i].empty()) continue;
      double neighbor_sum = 0.0;
      for (std::size_t neighbor : adjacency[i]) {
        neighbor_sum += monitor[neighbor];
      }
      smoothed[i] = 0.5 * monitor[i]
          + 0.5 * neighbor_sum / static_cast<double>(adjacency[i].size());
    }
    monitor.swap(smoothed);
  }

  for (int iteration = 0; iteration < options.move_iterations; ++iteration) {
    std::vector<AFEPack::Point<DIM>> old_position(vertices.size());
    std::vector<AFEPack::Point<DIM>> displacement(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      old_position[i] = *vertices[i];
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        displacement[i][coordinate] = 0.0;
      }
      if (fixed_vertex[i] || adjacency[i].empty()) continue;

      AFEPack::Point<DIM> weighted_average;
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        weighted_average[coordinate] = 0.0;
      }
      double weight_sum = 0.0;
      double local_edge_scale = std::numeric_limits<double>::infinity();
      for (std::size_t neighbor : adjacency[i]) {
        const double distance =
            (*vertices[i] - *vertices[neighbor]).length();
        local_edge_scale = std::min(local_edge_scale, distance);
        const double weight = (monitor[i] + monitor[neighbor])
            / std::max(distance, 1.0e-14);
        for (int coordinate = 0; coordinate < DIM; ++coordinate) {
          weighted_average[coordinate] +=
              weight * (*vertices[neighbor])[coordinate];
        }
        weight_sum += weight;
      }
      const double inverse_weight_sum = 1.0 / std::max(
          weight_sum, std::numeric_limits<double>::min());
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        weighted_average[coordinate] *= inverse_weight_sum;
        displacement[i][coordinate] = options.move_relaxation
            * (weighted_average[coordinate]
               - (*vertices[i])[coordinate]);
      }
      const double displacement_length = displacement[i].length();
      const double displacement_limit =
          options.max_displacement_ratio * local_edge_scale;
      if (displacement_length > displacement_limit) {
        const double scale = displacement_limit / displacement_length;
        for (int coordinate = 0; coordinate < DIM; ++coordinate) {
          displacement[i][coordinate] *= scale;
        }
      }
    }

    double line_search = 1.0;
    bool accepted = false;
    double accepted_minimum_quality = 1.0;
    for (int trial = 0; trial < 12; ++trial) {
      for (std::size_t i = 0; i < vertices.size(); ++i) {
        for (int coordinate = 0; coordinate < DIM; ++coordinate) {
          (*vertices[i])[coordinate] = old_position[i][coordinate]
              + line_search * displacement[i][coordinate];
        }
      }
      bool valid = true;
      double minimum_quality = 1.0;
      for (HElement<DIM>* leaf : leaves) {
        const auto& v = leaf->h_element->vertex;
        const double old_orientation = signed_six_volume(
            old_position[vertex_index.at(v[0])],
            old_position[vertex_index.at(v[1])],
            old_position[vertex_index.at(v[2])],
            old_position[vertex_index.at(v[3])]);
        const double new_orientation = signed_six_volume(
            *v[0], *v[1], *v[2], *v[3]);
        const double quality = tetrahedron_quality(
            *v[0], *v[1], *v[2], *v[3]);
        minimum_quality = std::min(minimum_quality, quality);
        if (old_orientation * new_orientation <= 0.0
            || quality < options.minimum_quality) {
          valid = false;
          break;
        }
      }
      if (valid) {
        accepted = true;
        accepted_minimum_quality = minimum_quality;
        break;
      }
      line_search *= 0.5;
    }
    if (!accepted) {
      for (std::size_t i = 0; i < vertices.size(); ++i) {
        for (int coordinate = 0; coordinate < DIM; ++coordinate) {
          (*vertices[i])[coordinate] = old_position[i][coordinate];
        }
      }
      break;
    }
    ++stats.accepted_iterations;
    stats.minimum_quality = std::min(
        stats.minimum_quality, accepted_minimum_quality);
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      stats.maximum_displacement = std::max(
          stats.maximum_displacement,
          (*vertices[i] - old_position[i]).length());
    }
  }
  double domain_diagonal_squared = 0.0;
  for (int coordinate = 0; coordinate < DIM; ++coordinate) {
    const double extent = coordinate_maximum[coordinate]
        - coordinate_minimum[coordinate];
    domain_diagonal_squared += extent * extent;
  }
  const double movement_tolerance = 1.0e-14
      * std::max(1.0, std::sqrt(domain_diagonal_squared));
  stats.moved = stats.accepted_iterations > 0
      && stats.maximum_displacement > movement_tolerance;
  if (stats.moved) {
    // r-adaptation changes coordinates but not connectivity.  Rebuilding the
    // RegularMesh here is both unnecessary and unsafe while FEMSpace still
    // references it.  Synchronize the active hierarchy vertices into the
    // existing regular mesh in place instead.
    RegularMesh<DIM>& regular_mesh = irregular_mesh.regularMesh();
    for (Vertex* vertex : vertices) {
      if (vertex->index < 0
          || static_cast<unsigned int>(vertex->index)
                 >= regular_mesh.n_geometry(0)) {
        throw std::runtime_error(
            "Moving mesh encountered an invalid active-vertex index");
      }
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        regular_mesh.point()[static_cast<std::size_t>(vertex->index)]
                            [coordinate] = (*vertex)[coordinate];
      }
    }
  }
  stats.seconds = omp_get_wtime() - start;
  return stats;
}

struct PreparedState
{
  std::vector<hElementCache> hierarchy_cache;
  std::map<HElement<DIM>*, int> hierarchy_index;
  std::vector<HElement<DIM>*> hierarchy_elements;
  std::vector<ElementCache> element_cache;
  double mapping_seconds = 0.0;
  double validation_seconds = 0.0;
  double list_seconds = 0.0;
  double quadrature_seconds = 0.0;
  double moment_seconds = 0.0;
  int threads = 1;
};

void update_moments(PreparedState& state, int expansion_order)
{
  const double start = omp_get_wtime();
  const std::size_t hierarchy_count = state.hierarchy_elements.size();
  #pragma omp parallel for schedule(dynamic, 16) default(none) \
      shared(state) firstprivate(hierarchy_count, expansion_order)
  for (std::size_t i = 0; i < hierarchy_count; ++i) {
    update_multipoleInfo_hElementCache(
        state.hierarchy_elements[i], state.hierarchy_cache,
        state.hierarchy_index, state.element_cache,
        expansion_order, false);
  }
  state.moment_seconds = omp_get_wtime() - start;
}

PreparedState prepare_state(
    IrregularMesh<DIM>& irregular_mesh,
    FEMSpace<double, DIM>& fem_space,
    const Options& options)
{
  PreparedState state;
  state.list_seconds = update_hElementCache(
      irregular_mesh, state.hierarchy_cache, state.hierarchy_index,
      options.mac_theta, true, &state.mapping_seconds,
      &state.validation_seconds, &state.threads);

  const double quadrature_start = omp_get_wtime();
  updateElementCache(
      state.element_cache, fem_space, options.quadrature_accuracy);
  IrregularMesh<DIM>::ActiveIterator element =
      irregular_mesh.beginActiveElement();
  IrregularMesh<DIM>::ActiveIterator end_element =
      irregular_mesh.endActiveElement();
  for (; element != end_element; ++element) {
    const int leaf_index = element->index;
    if (leaf_index < 0
        || static_cast<std::size_t>(leaf_index)
            >= state.element_cache.size()) {
      throw std::runtime_error("Active leaf index is outside element cache");
    }
    ElementCache& cache =
        state.element_cache[static_cast<std::size_t>(leaf_index)];
    cache.the_hElement = &(*element);
    cache.hierarchy_index = state.hierarchy_index.at(&(*element));
  }
  state.quadrature_seconds = omp_get_wtime() - quadrature_start;

  state.hierarchy_elements.assign(
      state.hierarchy_cache.size(), nullptr);
  for (const auto& entry : state.hierarchy_index) {
    state.hierarchy_elements[static_cast<std::size_t>(entry.second)] =
        entry.first;
  }
  if (std::find(
          state.hierarchy_elements.begin(),
          state.hierarchy_elements.end(), nullptr)
      != state.hierarchy_elements.end()) {
    throw std::runtime_error("Incomplete hierarchy mapping");
  }
  update_moments(state, options.expansion_order);
  return state;
}

double refresh_density(PreparedState& state)
{
  const double start = omp_get_wtime();
  const std::size_t element_count = state.element_cache.size();
  #pragma omp parallel for schedule(static) default(none) \
      shared(state) firstprivate(element_count)
  for (std::size_t element_index = 0;
       element_index < element_count; ++element_index) {
    ElementCache& element = state.element_cache[element_index];
    for (std::size_t q = 0; q < element.q_point.size(); ++q) {
      element.source_charge[q] = element.Jxw[q] * f(element.q_point[q]);
    }
  }
  return omp_get_wtime() - start;
}

double perturbation_source_at(
    const AFEPack::Point<DIM>& point,
    double center_x,
    double center_y,
    double center_z)
{
  constexpr double a = 6.0 * PI;
  constexpr double b = 5.0 * PI;
  constexpr double c = 9.0 * PI;
  const double x = point[0] - center_x;
  const double y = point[1] - center_y;
  const double z = point[2] - center_z;
  const double potential = std::exp(-a*x*x - b*y*y - c*z*z);
  const double quadratic = a*a*x*x + b*b*y*y + c*c*z*z;
  return (2.0*(a+b+c) - 4.0*quadratic) * potential / (4.0*PI);
}

double relative_density_change(
    const FEMSpace<double, DIM>& fem_space,
    int quadrature_accuracy,
    double current_amplitude,
    double current_center_x,
    double previous_amplitude,
    double previous_center_x)
{
  const Mesh<DIM>& mesh = fem_space.mesh();
  const int element_count = fem_space.n_element();
  const double center_y = reuse_sequence_center_y;
  const double center_z = reuse_sequence_center_z;
  double difference_norm = 0.0;
  double density_norm = 0.0;
  #pragma omp parallel for schedule(static) default(none) \
      shared(fem_space, mesh) firstprivate(element_count, quadrature_accuracy, \
          current_amplitude, current_center_x, previous_amplitude, \
          previous_center_x, center_y, center_z) \
      reduction(+:difference_norm,density_norm)
  for (int i = 0; i < element_count; ++i) {
    const GeometryBM& geometry = fem_space.element(i).geometry();
    std::vector<AFEPack::Point<DIM>> vertices(
        static_cast<std::size_t>(geometry.n_vertex()));
    for (int vertex = 0; vertex < geometry.n_vertex(); ++vertex) {
      vertices[static_cast<std::size_t>(vertex)] =
          mesh.point()[geometry.vertex()[vertex]];
    }
    const auto rule = treecode_quadrature::build_physical_rule(
        vertices, quadrature_accuracy);
    for (const auto& entry : rule) {
      const double base = treecode_test_problem::source_value(
          entry.point[0], entry.point[1], entry.point[2]);
      const double current = base + current_amplitude
          * perturbation_source_at(
              entry.point, current_center_x,
              center_y, center_z);
      const double previous = base + previous_amplitude
          * perturbation_source_at(
              entry.point, previous_center_x,
              center_y, center_z);
      const double difference = current - previous;
      difference_norm += entry.weight * difference * difference;
      density_norm += entry.weight * current * current;
    }
  }
  return std::sqrt(difference_norm)
      / std::max(
          std::sqrt(density_norm), std::numeric_limits<double>::min());
}

struct EvaluationResult
{
  double seconds = 0.0;
  double l2_error = 0.0;
  double relative_l2_error = 0.0;
};

EvaluationResult evaluate_potential(
    const PreparedState& state,
    const Options& options)
{
  const double start = omp_get_wtime();
  const std::size_t element_count = state.element_cache.size();
  double error_squared = 0.0;
  double exact_norm_squared = 0.0;
  std::size_t invalid_count = 0;
  #pragma omp parallel for schedule(dynamic, 16) default(none) \
      shared(state) firstprivate(element_count, options) \
      reduction(+:error_squared,exact_norm_squared,invalid_count)
  for (std::size_t element_index = 0;
       element_index < element_count; ++element_index) {
    const ElementCache& target = state.element_cache[element_index];
    const hElementCache& hierarchy = state.hierarchy_cache[
        static_cast<std::size_t>(target.hierarchy_index)];
    std::array<double,
               treecode_multipole::maximum_coefficient_count> workspace{};
    const double approximation = multipole_approximation(
        hierarchy.ref_point, target.the_hElement, hierarchy,
        state.hierarchy_cache, state.element_cache,
        options.expansion_order, options.quadrature_accuracy,
        workspace.data());
    const double exact = u(hierarchy.ref_point);
    if (!std::isfinite(approximation) || !std::isfinite(exact)
        || !std::isfinite(hierarchy.volume)) {
      ++invalid_count;
      continue;
    }
    const double difference = approximation - exact;
    error_squared += hierarchy.volume * difference * difference;
    exact_norm_squared += hierarchy.volume * exact * exact;
  }
  if (invalid_count != 0) {
    throw std::runtime_error("Non-finite potential evaluation");
  }
  EvaluationResult result;
  result.seconds = omp_get_wtime() - start;
  result.l2_error = std::sqrt(error_squared);
  result.relative_l2_error = result.l2_error
      / std::max(
          std::sqrt(exact_norm_squared),
          std::numeric_limits<double>::min());
  return result;
}

void perform_base_adaptation(
    std::unique_ptr<IrregularMesh<DIM>>& irregular_mesh,
    FEMSpace<double, DIM>& fem_space,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements,
    const Options& options)
{
  if (!options.base_radial_adaptation) return;
  double unused = 0.0;
  apply_h_adaptation(
      irregular_mesh, fem_space, template_elements, options,
      false, 1.0e-2, 5, &unused);
  apply_h_adaptation(
      irregular_mesh, fem_space, template_elements, options,
      false, 1.0e-3, 2, &unused);
}

}  // namespace mesh_modes

int main(int argc, char* argv[])
{
  using namespace mesh_modes;
  const double program_start = omp_get_wtime();
  Options options;
  try {
    options = mesh_modes::parse_options(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  double common_setup_seconds = 0.0;

  std::cout << "Mesh strategy configuration: mode="
            << mode_name(options.mode)
            << ", p=" << options.quadrature_accuracy
            << ", P=" << options.expansion_order
            << ", theta=" << options.mac_theta
            << ", steps=" << options.sequence_steps
            << ", decay=" << options.sequence_decay
            << ", initial_shift=" << options.initial_center_shift
            << '\n';

  reuse_sequence_amplitude = options.initial_amplitude;
  reuse_sequence_center_x = 0.70 + options.initial_center_shift;

  HGeometryTree<DIM> hierarchy_tree;
  hierarchy_tree.readMesh(options.mesh_file);
  auto irregular_mesh = std::make_unique<IrregularMesh<DIM>>();
  irregular_mesh->reinit(hierarchy_tree);
  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);
  irregular_mesh->globalRefine(options.initial_refinements);
  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);

  TemplateGeometry<DIM> tetrahedron_geometry;
  tetrahedron_geometry.readData("tetrahedron.tmp_geo");
  TemplateDOF<DIM> tetrahedron_dof(tetrahedron_geometry);
  tetrahedron_dof.readData("tetrahedron.1.tmp_dof");
  CoordTransform<DIM> tetrahedron_transform;
  tetrahedron_transform.readData("tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> tetrahedron_basis(tetrahedron_dof);
  tetrahedron_basis.readData("tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> tetrahedron_normal;
  tetrahedron_normal.readData("tetrahedron.out_nrm");

  TemplateGeometry<DIM> twin_geometry;
  twin_geometry.readData("twin_tetrahedron.tmp_geo");
  TemplateDOF<DIM> twin_dof(twin_geometry);
  twin_dof.readData("twin_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> twin_transform;
  twin_transform.readData("twin_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> twin_basis(twin_dof);
  twin_basis.readData("twin_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> twin_normal;
  twin_normal.readData("twin_tetrahedron.out_nrm");

  TemplateGeometry<DIM> four_geometry;
  four_geometry.readData("four_tetrahedron.tmp_geo");
  TemplateDOF<DIM> four_dof(four_geometry);
  four_dof.readData("four_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> four_transform;
  four_transform.readData("four_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> four_basis(four_dof);
  four_basis.readData("four_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> four_normal;
  four_normal.readData("four_tetrahedron.out_nrm");

  std::vector<TemplateElement<double, DIM, DIM>> template_elements(3);
  template_elements[0].reinit(
      tetrahedron_geometry, tetrahedron_dof, tetrahedron_transform,
      tetrahedron_basis, tetrahedron_normal);
  template_elements[1].reinit(
      twin_geometry, twin_dof, twin_transform,
      twin_basis, twin_normal);
  template_elements[2].reinit(
      four_geometry, four_dof, four_transform,
      four_basis, four_normal);

  FEMSpace<double, DIM> fem_space;
  initialize_fem_space(
      fem_space, irregular_mesh->regularMesh(), template_elements);
  perform_base_adaptation(
      irregular_mesh, fem_space, template_elements, options);

  PreparedState state;
  try {
    state = prepare_state(*irregular_mesh, fem_space, options);
  } catch (const std::exception& error) {
    std::cerr << "Initial treecode preparation failed: "
              << error.what() << '\n';
    return 1;
  }
  common_setup_seconds = omp_get_wtime() - program_start;

  std::ofstream output(options.output_file, std::ios::trunc);
  if (!output.is_open()) {
    std::cerr << "Cannot write " << options.output_file << '\n';
    return 1;
  }
  output << std::setprecision(15)
      << "mode\tstep\tp\tP\ttheta\tomp_threads"
      << "\tamplitude\tcenter_x\trelative_density_change"
      << "\tsource_residual\tmesh_updated\tupdate_kind"
      << "\tn_elements\thierarchy_nodes\tmesh_update_time_s"
      << "\tmapping_time_s\thierarchy_validation_time_s\tlist_time_s"
      << "\tquadrature_cache_time_s\tdensity_update_time_s"
      << "\tmoment_time_s\tpotential_time_s\tstep_time_s"
      << "\tL2_error\trelative_L2_error\tmax_vertex_displacement"
      << "\tmin_tetra_quality\tfixed_vertices\tmovable_vertices\n";

  double previous_amplitude = options.initial_amplitude;
  double previous_center_x = 0.70 + options.initial_center_shift;
  int steps_since_h_update = 0;
  bool r_motion_since_h_update = false;
  MeshGeometryCheckpoint h_geometry_checkpoint =
      capture_mesh_geometry(*irregular_mesh);
  double total_seconds = 0.0;
  int h_update_count = 0;
  int move_update_count = 0;

  for (int step = 0; step < options.sequence_steps; ++step) {
    const double step_start = omp_get_wtime();
    const double decay = std::pow(options.sequence_decay, step);
    const double amplitude = options.initial_amplitude * decay;
    const double center_x = 0.70 + options.initial_center_shift * decay;
    reuse_sequence_amplitude = amplitude;
    reuse_sequence_center_x = center_x;

    const double density_change = step == 0 ? 0.0
        : relative_density_change(
            fem_space, options.quadrature_accuracy,
            amplitude, center_x,
            previous_amplitude, previous_center_x);
    Indicator<DIM> residual_indicator(irregular_mesh->regularMesh());
    const double source_residual = build_source_residual_indicator(
        residual_indicator, irregular_mesh->regularMesh(), fem_space,
        options.quadrature_accuracy);

    bool request_h_update = false;
    bool request_move = false;
    if (step > 0) {
      switch (options.mode) {
        case Mode::frozen:
          break;
        case Mode::every_h:
          request_h_update = true;
          break;
        case Mode::periodic_h:
          request_h_update = step % options.update_interval == 0;
          break;
        case Mode::residual_h:
          request_h_update = source_residual > options.residual_trigger
              || steps_since_h_update >= options.max_reuse_steps;
          break;
        case Mode::moving_r:
          request_move = true;
          break;
        case Mode::hybrid_hr:
          request_h_update = source_residual > options.residual_trigger
              || steps_since_h_update >= options.max_reuse_steps;
          request_move = true;
          break;
      }
    }

    double mesh_update_seconds = 0.0;
    MovingStats moving_stats;
    std::string update_kind = "REUSE";
    if (request_h_update) {
      double measured_residual = 0.0;
      // AFEPack h-coarsening can reactivate parent tetrahedra.  Restore the
      // last conforming post-h geometry before changing topology, then take a
      // new checkpoint for the following r-only interval.
      if (options.mode == Mode::hybrid_hr && r_motion_since_h_update) {
        mesh_update_seconds += restore_mesh_geometry(
            *irregular_mesh, h_geometry_checkpoint);
      }
      mesh_update_seconds += apply_h_adaptation(
          irregular_mesh, fem_space, template_elements, options,
          true, options.adapt_tolerance, options.refine_steps,
          &measured_residual);
      h_geometry_checkpoint = capture_mesh_geometry(*irregular_mesh);
      ++h_update_count;
      steps_since_h_update = 0;
      r_motion_since_h_update = false;
      update_kind = "H";
    } else {
      ++steps_since_h_update;
    }
    if (request_move) {
      moving_stats = move_mesh(
          *irregular_mesh, fem_space, template_elements, options);
      mesh_update_seconds += moving_stats.seconds;
      if (moving_stats.moved) {
        ++move_update_count;
        r_motion_since_h_update = true;
        update_kind = request_h_update ? "HR" : "R";
      }
    }
    const bool mesh_updated = request_h_update || moving_stats.moved;

    double density_seconds = 0.0;
    if (mesh_updated) {
      try {
        state = prepare_state(*irregular_mesh, fem_space, options);
      } catch (const std::exception& error) {
        std::cerr << "Step " << step
                  << " mesh/tree rebuild failed: " << error.what() << '\n';
        return 1;
      }
    } else {
      density_seconds = refresh_density(state);
      update_moments(state, options.expansion_order);
    }

    EvaluationResult evaluation;
    try {
      evaluation = evaluate_potential(state, options);
    } catch (const std::exception& error) {
      std::cerr << "Step " << step
                << " potential evaluation failed: " << error.what() << '\n';
      return 1;
    }
    const double step_seconds = omp_get_wtime() - step_start;
    total_seconds += step_seconds;

    output << mode_name(options.mode) << '\t'
           << step << '\t'
           << options.quadrature_accuracy << '\t'
           << options.expansion_order << '\t'
           << options.mac_theta << '\t'
           << state.threads << '\t'
           << amplitude << '\t' << center_x << '\t'
           << density_change << '\t' << source_residual << '\t'
           << (mesh_updated ? 1 : 0) << '\t' << update_kind << '\t'
           << state.element_cache.size() << '\t'
           << state.hierarchy_cache.size() << '\t'
           << mesh_update_seconds << '\t'
           << (mesh_updated ? state.mapping_seconds : 0.0) << '\t'
           << (mesh_updated ? state.validation_seconds : 0.0) << '\t'
           << (mesh_updated ? state.list_seconds : 0.0) << '\t'
           << (mesh_updated ? state.quadrature_seconds : 0.0) << '\t'
           << density_seconds << '\t' << state.moment_seconds << '\t'
           << evaluation.seconds << '\t' << step_seconds << '\t'
           << evaluation.l2_error << '\t'
           << evaluation.relative_l2_error << '\t'
           << moving_stats.maximum_displacement << '\t'
           << moving_stats.minimum_quality << '\t'
           << moving_stats.fixed_vertices << '\t'
           << moving_stats.movable_vertices << '\n';

    std::cout << std::setprecision(8)
              << "step " << step
              << " mode=" << mode_name(options.mode)
              << " update=" << update_kind
              << " n=" << state.element_cache.size()
              << " r_rho=" << density_change
              << " eta_source=" << source_residual
              << " T_step=" << step_seconds
              << " L2=" << evaluation.l2_error
              << " relL2=" << evaluation.relative_l2_error << '\n';

    previous_amplitude = amplitude;
    previous_center_x = center_x;
  }
  const double full_wall_seconds = omp_get_wtime() - program_start;
  output << "# common_initial_setup_time_s\t"
         << common_setup_seconds << '\n'
         << "# sequence_time_s\t" << total_seconds << '\n'
         << "# wall_time_including_initial_setup_s\t"
         << full_wall_seconds << '\n'
         << "# h_update_count\t" << h_update_count << '\n'
         << "# move_update_count\t" << move_update_count << '\n';
  output.close();

  std::cout << "Mesh strategy summary: mode=" << mode_name(options.mode)
            << ", sequence=" << total_seconds
            << " s, including setup=" << full_wall_seconds
            << " s, h_updates=" << h_update_count
            << ", move_updates=" << move_update_count
            << ", output=" << options.output_file << '\n';
  return 0;
}
