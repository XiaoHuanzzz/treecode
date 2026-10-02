// Generate a direct truncated-domain reference solution on a globally
// refined tetrahedral mesh. One barycenter/value record is written per cell.

#include <AFEPack/DGFEMSpace.h>
#include <AFEPack/HGeometry.h>
#include <AFEPack/TemplateElement.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int DIM = 3;
constexpr double PI = 3.14159265358979323846;
constexpr int DEFAULT_REFINEMENT_LEVEL = 5;
constexpr int DEFAULT_QUADRATURE_ACCURACY = 6;
constexpr int DEFAULT_SOURCE_BLOCK_SIZE = 4096;

struct Options
{
  std::string mesh_file;
  int refinement_level = DEFAULT_REFINEMENT_LEVEL;
  int quadrature_accuracy = DEFAULT_QUADRATURE_ACCURACY;
  std::string output_file = "results_u1.bin";
  int source_block_size = DEFAULT_SOURCE_BLOCK_SIZE;
};

int parse_integer(const char* text, const char* name)
{
  try {
    const std::string value(text);
    std::size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return result;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
  }
}

Options parse_options(int argc, char* argv[])
{
  if (argc < 2 || argc > 6) {
    throw std::invalid_argument(
        "Usage: uni2 MESH_FILE [REFINEMENT_LEVEL] "
        "[QUADRATURE_ACCURACY] [OUTPUT_FILE] [SOURCE_BLOCK_SIZE]");
  }

  Options options;
  options.mesh_file = argv[1];
  if (argc >= 3) {
    options.refinement_level =
        parse_integer(argv[2], "refinement level");
  }
  if (argc >= 4) {
    options.quadrature_accuracy =
        parse_integer(argv[3], "quadrature accuracy");
  }
  if (argc >= 5) {
    options.output_file = argv[4];
  }
  if (argc >= 6) {
    options.source_block_size =
        parse_integer(argv[5], "source block size");
  }

  if (options.refinement_level < 0 || options.refinement_level > 7) {
    throw std::invalid_argument("REFINEMENT_LEVEL must be between 0 and 7");
  }
  const std::array<int, 6> supported_orders{{1, 2, 3, 4, 6, 8}};
  if (std::find(
          supported_orders.begin(), supported_orders.end(),
          options.quadrature_accuracy) == supported_orders.end()) {
    throw std::invalid_argument(
        "QUADRATURE_ACCURACY must be one of 1, 2, 3, 4, 6, or 8");
  }
  if (options.source_block_size <= 0) {
    throw std::invalid_argument("SOURCE_BLOCK_SIZE must be positive");
  }
  return options;
}

double whole_space_solution(const double* point)
{
  return std::exp(
      -PI * point[0] * point[0]
      -2.0 * PI * point[1] * point[1]
      -3.0 * PI * point[2] * point[2]);
}

double density(const double* point)
{
  const double laplacian_factor =
      4.0 * PI * PI * point[0] * point[0]
      +16.0 * PI * PI * point[1] * point[1]
      +36.0 * PI * PI * point[2] * point[2]
      -12.0 * PI;
  return -laplacian_factor * whole_space_solution(point) / (4.0 * PI);
}

double determinant(
    const AFEPack::Point<DIM>& a,
    const AFEPack::Point<DIM>& b,
    const AFEPack::Point<DIM>& c)
{
  return
      a[0] * (b[1] * c[2] - b[2] * c[1])
      -a[1] * (b[0] * c[2] - b[2] * c[0])
      +a[2] * (b[0] * c[1] - b[1] * c[0]);
}

double tetrahedron_volume(
    const std::array<AFEPack::Point<DIM>, 4>& vertices)
{
  return std::abs(determinant(
      vertices[1] - vertices[0],
      vertices[2] - vertices[0],
      vertices[3] - vertices[0])) / 6.0;
}

AFEPack::Point<DIM> tetrahedron_barycenter(
    const std::array<AFEPack::Point<DIM>, 4>& vertices)
{
  AFEPack::Point<DIM> result =
      vertices[0] + vertices[1] + vertices[2] + vertices[3];
  result /= 4.0;
  return result;
}

std::array<AFEPack::Point<DIM>, 4> element_vertices(
    const AFEPack::Element<double, DIM>& element,
    const std::vector<AFEPack::Point<DIM>>& mesh_points)
{
  std::array<AFEPack::Point<DIM>, 4> vertices;
  const AFEPack::GeometryBM& geometry = element.geometry();
  for (int vertex = 0; vertex < 4; ++vertex) {
    vertices[vertex] = mesh_points[geometry.vertex()[vertex]];
  }
  return vertices;
}

double singular_self_integral(
    const AFEPack::Point<DIM>& target,
    const std::array<AFEPack::Point<DIM>, 4>& vertices)
{
  // Eight-point Gauss-Legendre rule mapped to [0,1].
  static constexpr double nodes[8] = {
      0.019855071751231884, 0.10166676129318663,
      0.23723379504183550,  0.40828267875217510,
      0.59171732124782490,  0.76276620495816450,
      0.89833323870681337,  0.98014492824876812};
  static constexpr double weights[8] = {
      0.05061426814518813, 0.11119051722668724,
      0.15685332293894364, 0.18134189168918099,
      0.18134189168918099, 0.15685332293894364,
      0.11119051722668724, 0.05061426814518813};
  static constexpr int faces[4][3] = {
      {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

  double integral = 0.0;
  for (const auto& face : faces) {
    const AFEPack::Point<DIM>& a = vertices[face[0]];
    const AFEPack::Point<DIM>& b = vertices[face[1]];
    const AFEPack::Point<DIM>& c = vertices[face[2]];
    const double det =
        std::abs(determinant(a - target, b - target, c - target));
    if (det <= std::numeric_limits<double>::epsilon()) {
      continue;
    }

    for (int it = 0; it < 8; ++it) {
      const double t = nodes[it];
      for (int is = 0; is < 8; ++is) {
        const double s = nodes[is];
        for (int ir = 0; ir < 8; ++ir) {
          const double r = nodes[ir];

          AFEPack::Point<DIM> face_point = a;
          face_point *= 1.0 - s;
          AFEPack::Point<DIM> contribution = b;
          contribution *= s * (1.0 - r);
          face_point += contribution;
          contribution = c;
          contribution *= s * r;
          face_point += contribution;

          const AFEPack::Point<DIM> radial = face_point - target;
          const double radial_length = radial.length();
          AFEPack::Point<DIM> radial_step = radial;
          radial_step *= t;
          AFEPack::Point<DIM> point = target + radial_step;

          // The Duffy Jacobian is det*t^2*s and |point-target|=t|radial|.
          integral +=
              weights[it] * weights[is] * weights[ir]
              * det * t * s * density(point) / radial_length;
        }
      }
    }
  }
  return integral;
}

void initialize_tetrahedron_template(
    AFEPack::TemplateElement<double, DIM, DIM>& element,
    AFEPack::TemplateGeometry<DIM>& geometry,
    AFEPack::TemplateDOF<DIM>& dof,
    AFEPack::CoordTransform<DIM>& transform,
    AFEPack::BasisFunctionAdmin<double, DIM>& basis,
    AFEPack::UnitOutNormal<DIM>& normal)
{
  geometry.readData("tetrahedron.tmp_geo");
  dof.reinit(geometry);
  dof.readData("tetrahedron.1.tmp_dof");
  transform.readData("tetrahedron.crd_trs");
  basis.reinit(dof);
  basis.readData("tetrahedron.1.bas_fun");
  normal.readData("tetrahedron.out_nrm");
  element.reinit(geometry, dof, transform, basis, normal);
}

} // namespace

int main(int argc, char* argv[])
{
  try {
    const Options options = parse_options(argc, argv);
    std::cout
        << "Uniform reference configuration: level="
        << options.refinement_level
        << ", source quadrature=" << options.quadrature_accuracy
        << ", source block=" << options.source_block_size
        << ", output=" << options.output_file << '\n';

    AFEPack::HGeometryTree<DIM> hierarchy;
    hierarchy.readMesh(options.mesh_file);

    AFEPack::IrregularMesh<DIM> irregular_mesh;
    irregular_mesh.reinit(hierarchy);
    irregular_mesh.globalRefine(options.refinement_level);
    irregular_mesh.semiregularize();
    irregular_mesh.regularize(false);
    AFEPack::RegularMesh<DIM>& mesh = irregular_mesh.regularMesh();

    AFEPack::TemplateGeometry<DIM> tetrahedron_geometry;
    AFEPack::TemplateDOF<DIM> tetrahedron_dof;
    AFEPack::CoordTransform<DIM> tetrahedron_transform;
    AFEPack::BasisFunctionAdmin<double, DIM> tetrahedron_basis;
    AFEPack::UnitOutNormal<DIM> tetrahedron_normal;
    std::vector<AFEPack::TemplateElement<double, DIM, DIM>>
        template_elements(1);
    initialize_tetrahedron_template(
        template_elements[0],
        tetrahedron_geometry,
        tetrahedron_dof,
        tetrahedron_transform,
        tetrahedron_basis,
        tetrahedron_normal);

    AFEPack::FEMSpace<double, DIM> fem_space;
    fem_space.reinit(mesh, template_elements);
    const int element_count = mesh.n_geometry(DIM);
    fem_space.element().resize(element_count);
    for (int i = 0; i < element_count; ++i) {
      if (mesh.geometry(DIM, i).n_vertex() != 4) {
        throw std::runtime_error(
            "Global refinement produced a non-tetrahedral regular element");
      }
      fem_space.element(i).reinit(fem_space, i, 0);
    }
    fem_space.buildElement();

    const auto& mesh_points = mesh.point();
    if (element_count == 0) {
      throw std::runtime_error("Uniform mesh contains no elements");
    }

    const auto first_vertices =
        element_vertices(fem_space.element(0), mesh_points);
    const double cell_volume = tetrahedron_volume(first_vertices);
    double maximum_volume_difference = 0.0;
    #pragma omp parallel for schedule(static) default(none) \
        shared(fem_space, mesh_points) \
        firstprivate(element_count, cell_volume) \
        reduction(max:maximum_volume_difference)
    for (int i = 0; i < element_count; ++i) {
      const AFEPack::Element<double, DIM>& element = fem_space.element(i);
      const auto vertices = element_vertices(element, mesh_points);
      maximum_volume_difference =
          std::max(
              maximum_volume_difference,
              std::abs(tetrahedron_volume(vertices) - cell_volume));
    }
    if (maximum_volume_difference >
        1.0e-12 * std::max(1.0, cell_volume)) {
      throw std::runtime_error(
          "Uniform refinement produced unequal cell volumes");
    }
    const double domain_volume =
        cell_volume * static_cast<double>(element_count);

    const long double pair_count =
        static_cast<long double>(element_count)
        * static_cast<long double>(element_count);
    const AFEPack::QuadratureInfo<DIM>& source_quadrature =
        fem_space.element(0).findQuadratureInfo(
            options.quadrature_accuracy);
    const int source_quadrature_count =
        source_quadrature.n_quadraturePoint();
    const int source_block_size =
        std::min(options.source_block_size, element_count);
    const std::size_t persistent_bytes =
        static_cast<std::size_t>(element_count) * sizeof(double);
    const std::size_t block_bytes =
        static_cast<std::size_t>(source_block_size)
        * static_cast<std::size_t>(source_quadrature_count)
        * (sizeof(AFEPack::Point<DIM>) + sizeof(double));

    std::cout << "Elements: " << element_count
              << ", domain volume: " << std::setprecision(15)
              << domain_volume << '\n'
              << "Direct source-target pairs: "
              << std::setprecision(6) << std::scientific
              << pair_count << std::defaultfloat << '\n'
              << "Persistent calculation storage: "
              << persistent_bytes / (1024.0 * 1024.0) << " MiB\n"
              << "Reusable source block storage: "
              << block_bytes / (1024.0 * 1024.0) << " MiB\n";
    if (element_count > 100000) {
      std::cout
          << "Warning: this is an O(N^2) direct reference calculation; "
             "deep uniform levels are intentionally expensive in time.\n";
    }

    std::vector<double> potentials(
        static_cast<std::size_t>(element_count), 0.0);
    std::vector<AFEPack::Point<DIM>> source_points;
    std::vector<double> source_charges;
    source_points.reserve(
        static_cast<std::size_t>(source_block_size)
        * static_cast<std::size_t>(source_quadrature_count));
    source_charges.reserve(source_points.capacity());
    const double start_time = omp_get_wtime();

    for (int block_begin = 0;
         block_begin < element_count;
         block_begin += source_block_size) {
      const int block_count =
          std::min(source_block_size, element_count - block_begin);
      const std::size_t block_entry_count =
          static_cast<std::size_t>(block_count)
          * static_cast<std::size_t>(source_quadrature_count);
      source_points.resize(block_entry_count);
      source_charges.resize(block_entry_count);

      #pragma omp parallel for schedule(static) default(none) \
          shared(source_points, source_charges, fem_space, source_quadrature, \
                 mesh_points) \
          firstprivate(block_begin, block_count, source_quadrature_count)
      for (int local_source = 0;
           local_source < block_count;
           ++local_source) {
        const AFEPack::Element<double, DIM>& source_element =
            fem_space.element(block_begin + local_source);
        const auto vertices =
            element_vertices(source_element, mesh_points);
        const double source_volume = tetrahedron_volume(vertices);
        const auto& local_points = source_quadrature.quadraturePoint();
        const std::size_t offset =
            static_cast<std::size_t>(local_source)
            * static_cast<std::size_t>(source_quadrature_count);
        for (int q = 0; q < source_quadrature_count; ++q) {
          const std::size_t index = offset + static_cast<std::size_t>(q);
          const AFEPack::Point<DIM>& local =
              local_points[static_cast<std::size_t>(q)];
          for (int d = 0; d < DIM; ++d) {
            source_points[index][d] =
                vertices[0][d]
                + local[0] * (vertices[1][d] - vertices[0][d])
                + local[1] * (vertices[2][d] - vertices[0][d])
                + local[2] * (vertices[3][d] - vertices[0][d]);
          }
          const double jxw = source_quadrature.weight(q) * source_volume;
          source_charges[index] =
              jxw * density(source_points[index]);
        }
      }

      #pragma omp parallel for schedule(static) default(none) \
          shared(potentials, source_points, source_charges, \
                 fem_space, mesh_points) \
          firstprivate(element_count, block_begin, block_count, \
                       source_quadrature_count)
      for (int target_index = 0;
           target_index < element_count;
           ++target_index) {
        const auto target_vertices = element_vertices(
            fem_space.element(target_index), mesh_points);
        const AFEPack::Point<DIM> target =
            tetrahedron_barycenter(target_vertices);
        double potential =
            potentials[static_cast<std::size_t>(target_index)];

        for (int local_source = 0;
             local_source < block_count;
             ++local_source) {
          const int source_index = block_begin + local_source;
          if (source_index == target_index) {
            potential +=
                singular_self_integral(target, target_vertices);
            continue;
          }

          const std::size_t offset =
              static_cast<std::size_t>(local_source)
              * static_cast<std::size_t>(source_quadrature_count);
          for (int q = 0; q < source_quadrature_count; ++q) {
            const std::size_t index =
                offset + static_cast<std::size_t>(q);
            const AFEPack::Point<DIM> displacement =
                source_points[index] - target;
            potential +=
                source_charges[index] / displacement.length();
          }
        }
        potentials[static_cast<std::size_t>(target_index)] = potential;
      }
    }

    double whole_space_error_squared = 0.0;
    #pragma omp parallel for schedule(static) default(none) \
        shared(potentials, fem_space, mesh_points) \
        firstprivate(element_count, cell_volume) \
        reduction(+:whole_space_error_squared)
    for (int i = 0; i < element_count; ++i) {
      const AFEPack::Point<DIM> target = tetrahedron_barycenter(
          element_vertices(fem_space.element(i), mesh_points));
      const double difference =
          potentials[static_cast<std::size_t>(i)]
          - whole_space_solution(target);
      whole_space_error_squared += cell_volume * difference * difference;
    }

    std::ofstream output(
        options.output_file,
        std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output) {
      throw std::runtime_error(
          "Cannot open output file: " + options.output_file);
    }
    constexpr int output_chunk_size = 65536;
    std::vector<double> output_buffer(
        static_cast<std::size_t>(DIM + 1)
        * static_cast<std::size_t>(
            std::min(output_chunk_size, element_count)));
    for (int chunk_begin = 0;
         chunk_begin < element_count;
         chunk_begin += output_chunk_size) {
      const int chunk_count =
          std::min(output_chunk_size, element_count - chunk_begin);
      for (int local_index = 0; local_index < chunk_count; ++local_index) {
        const int global_index = chunk_begin + local_index;
        const AFEPack::Point<DIM> target = tetrahedron_barycenter(
            element_vertices(fem_space.element(global_index), mesh_points));
        double* record =
            output_buffer.data()
            + static_cast<std::size_t>(DIM + 1)
              * static_cast<std::size_t>(local_index);
        for (int d = 0; d < DIM; ++d) {
          record[d] = target[d];
        }
        record[DIM] =
            potentials[static_cast<std::size_t>(global_index)];
      }
      output.write(
          reinterpret_cast<const char*>(output_buffer.data()),
          static_cast<std::streamsize>(
              static_cast<std::size_t>(DIM + 1)
              * static_cast<std::size_t>(chunk_count)
              * sizeof(double)));
      if (!output) {
        throw std::runtime_error(
            "Failed while writing output file: " + options.output_file);
      }
    }

    const double elapsed = omp_get_wtime() - start_time;
    std::cout
        << "Wrote " << element_count << " records in "
        << elapsed << " seconds\n"
        << "L2 difference from whole-space manufactured solution: "
        << std::sqrt(whole_space_error_squared) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
