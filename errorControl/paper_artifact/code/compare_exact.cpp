// Adaptive treecode with direct comparison against the analytic solution.
//
// This is the complete implementation. Compiling this file directly selects
// exact-solution comparison. compare_parallel.cpp defines
// TREECODE_COMPARE_REFERENCE and includes this file to retain the older
// binary-reference comparison without duplicating the mesh/treecode code.
#ifndef TREECODE_COMPARE_REFERENCE
#define TREECODE_COMPARE_EXACT 1
#endif

// parallel fill neighbor and compare part.
#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>

/// for mesh
#include <AFEPack/HGeometry.h>
/// for template element
#include <AFEPack/TemplateElement.h>
/// for space
#include <AFEPack/DGFEMSpace.h>

#include "local_quadrature.hpp"
#include "cartesian_multipole.hpp"
#include "gaussian_test_problem.hpp"

#ifdef TREECODE_USE_CUDA
#include "compare_gpu.hpp"
#endif


constexpr int DIM = 3;
constexpr double PI = treecode_test_problem::pi;
constexpr int DEFAULT_INITIAL_REFINEMENTS = 2;
constexpr int DEFAULT_QUADRATURE_ACCURACY = 8;
constexpr int DEFAULT_EXPANSION_ORDER = 4;
constexpr double DEFAULT_MAC_THETA = 0.5;
constexpr double DEFAULT_ADAPT_TOLERANCE_STAGE1 = 1.0e-2;
constexpr double DEFAULT_ADAPT_TOLERANCE_STAGE2 = 1.0e-3;

enum class ComputeBackend
{
  cpu,
  gpu
};

const char* backend_name(ComputeBackend backend)
{
  return backend == ComputeBackend::cpu ? "CPU" : "GPU";
}

const char* timing_results_filename()
{
#ifdef TREECODE_UNIFORM_MESH
  return "results_uniform.txt";
#else
  return "results.txt";
#endif
}

struct ProgramOptions
{
  std::string mesh_file;
  int initial_refinements = DEFAULT_INITIAL_REFINEMENTS;
  int quadrature_accuracy = DEFAULT_QUADRATURE_ACCURACY;
  int expansion_order = DEFAULT_EXPANSION_ORDER;
  double mac_theta = DEFAULT_MAC_THETA;
  std::string reference_file = "results_u1.bin";
  bool compare_solution = true;
  ComputeBackend backend = ComputeBackend::cpu;
  double adaptive_tolerance_stage1 = DEFAULT_ADAPT_TOLERANCE_STAGE1;
  double adaptive_tolerance_stage2 = DEFAULT_ADAPT_TOLERANCE_STAGE2;
#ifdef TREECODE_REUSE_EXPERIMENT
  int sequence_steps = 6;
  double sequence_decay = 0.5;
  double sequence_initial_amplitude = 0.5;
  std::string sequence_output_file = "multirate_reuse_results.tsv";
#endif
};

bool append_timing_result(
    const ProgramOptions& options,
    std::size_t element_count,
    double mapping_geometry_seconds,
    double hierarchy_validation_seconds,
    bool has_neighbor_fill_time,
    double neighbor_fill_seconds,
    double total_execution_seconds,
    bool has_potential_evaluation_time,
    double potential_evaluation_seconds,
    bool has_comparison_time,
    double comparison_seconds,
    bool has_l2_error,
    double l2_error,
    int omp_threads_used)
{
  const char* results_filename = timing_results_filename();
  bool header_required = true;
  bool leading_newline_required = false;
  {
    std::ifstream existing_results(
        results_filename, std::ios::binary | std::ios::ate);
    if (existing_results.is_open() && existing_results.tellg() > 0) {
      header_required = false;
      existing_results.seekg(-1, std::ios::end);
      char final_character = '\0';
      existing_results.get(final_character);
      leading_newline_required = final_character != '\n';
    }
  }

  std::ofstream results_file(results_filename, std::ios::app);
  if (!results_file.is_open()) {
    std::cerr << "Error: cannot open " << results_filename
              << " for appending\n";
    return false;
  }

  auto write_field = [&results_file](const auto& value, int width) {
    results_file << std::left << std::setw(width) << value;
  };

  if (header_required) {
    write_field("p", 4);
    write_field("P", 4);
    write_field("theta", 8);
    write_field("n_elements", 12);
    write_field("mapping_geometry_time_s", 26);
    write_field("hierarchy_validation_time_s", 30);
    write_field("filling_neighbor_time_s", 26);
    write_field("execution_time_s", 20);
    write_field("L2_error_exact", 24);
    write_field("backend", 12);
    write_field("omp_max_threads", 18);
    write_field("potential_evaluation_time_s", 30);
    write_field("comparison_time_s", 26);
    results_file << "omp_threads_used\n";
  } else if (leading_newline_required) {
    results_file << '\n';
  }

  results_file << std::setprecision(15);
  write_field(options.quadrature_accuracy, 4);
  write_field(options.expansion_order, 4);
  write_field(options.mac_theta, 8);
  write_field(element_count, 12);
  write_field(mapping_geometry_seconds, 26);
  write_field(hierarchy_validation_seconds, 30);
  if (has_neighbor_fill_time) {
    write_field(neighbor_fill_seconds, 26);
  } else {
    write_field("NA", 26);
  }
  write_field(total_execution_seconds, 20);
  if (has_l2_error) {
    write_field(l2_error, 24);
  } else {
    write_field("NA", 24);
  }
  write_field(backend_name(options.backend), 12);
  write_field(omp_get_max_threads(), 18);
  if (has_potential_evaluation_time) {
    write_field(potential_evaluation_seconds, 30);
  } else {
    write_field("NA", 30);
  }
  if (has_comparison_time) {
    write_field(comparison_seconds, 26);
  } else {
    write_field("NA", 26);
  }
  results_file << omp_threads_used;
  results_file << '\n';

  if (!results_file) {
    std::cerr << "Error: failed while appending to "
              << results_filename << '\n';
    return false;
  }
  return true;
}

int parse_int(const char* text, const char* name)
{
  try {
    std::size_t parsed = 0;
    const std::string value(text);
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return result;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
  }
}

double parse_double(const char* text, const char* name)
{
  try {
    std::size_t parsed = 0;
    const std::string value(text);
    const double result = std::stod(value, &parsed);
    if (parsed != value.size() || !std::isfinite(result)) {
      throw std::invalid_argument("invalid value");
    }
    return result;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
  }
}

bool parse_comparison_flag(const char* text)
{
  const std::string value(text);
  if (value == "1" || value == "true"
      || value == "on" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false"
      || value == "off" || value == "no") {
    return false;
  }
  throw std::invalid_argument(
      "Invalid comparison flag: " + value
      + " (use 0/1, false/true, off/on, or no/yes)");
}

bool is_backend_name(const char* text)
{
  std::string value(text);
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  return value == "CPU" || value == "GPU";
}

ComputeBackend parse_backend(const char* text)
{
  std::string value(text);
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  if (value == "CPU") return ComputeBackend::cpu;
  if (value == "GPU") return ComputeBackend::gpu;
  throw std::invalid_argument(
      "Invalid backend: " + value + " (use CPU or GPU)");
}

ProgramOptions parse_options(int argc, char* argv[])
{
#ifdef TREECODE_COMPARE_EXACT
#ifdef TREECODE_REUSE_EXPERIMENT
  if (argc < 2 || argc > 12) {
    throw std::invalid_argument(
        "Usage: tree_reuse MESH_FILE [INITIAL_REFINEMENTS] "
        "[QUADRATURE_ACCURACY] [EXPANSION_ORDER] [MAC_THETA] "
        "[COMPARE] [BACKEND] [SEQUENCE_STEPS] [DECAY] "
        "[INITIAL_PERTURBATION_AMPLITUDE] [OUTPUT_TSV]");
  }
#else
  if (argc < 2 || argc > 10) {
    throw std::invalid_argument(
        "Usage: tree_exact MESH_FILE [INITIAL_REFINEMENTS] "
        "[QUADRATURE_ACCURACY] [EXPANSION_ORDER] [MAC_THETA] "
        "[COMPARE] [BACKEND] [ADAPT_TOLERANCE_STAGE1] "
        "[ADAPT_TOLERANCE_STAGE2]\n"
        "       BACKEND is CPU or GPU; COMPARE and BACKEND may be swapped");
  }
#endif
#else
  if (argc < 2 || argc > 11) {
    throw std::invalid_argument(
        "Usage: tree MESH_FILE [INITIAL_REFINEMENTS] [QUADRATURE_ACCURACY] "
        "[EXPANSION_ORDER] [MAC_THETA] [REFERENCE_FILE] [COMPARE] "
        "[BACKEND] [ADAPT_TOLERANCE_STAGE1] "
        "[ADAPT_TOLERANCE_STAGE2]");
  }
#endif

  ProgramOptions options;
  options.mesh_file = argv[1];
  if (argc >= 3) options.initial_refinements =
      parse_int(argv[2], "initial refinement count");
  if (argc >= 4) options.quadrature_accuracy =
      parse_int(argv[3], "quadrature accuracy");
  if (argc >= 5) options.expansion_order =
      parse_int(argv[4], "expansion order");
  if (argc >= 6) options.mac_theta = parse_double(argv[5], "MAC theta");
#ifndef TREECODE_COMPARE_EXACT
  if (argc >= 7) options.reference_file = argv[6];
  if (argc >= 8) options.compare_solution =
      parse_comparison_flag(argv[7]);
  if (argc >= 9) options.backend = parse_backend(argv[8]);
  if (argc >= 10) {
    options.adaptive_tolerance_stage1 =
        parse_double(argv[9], "stage-1 adaptive tolerance");
  }
  if (argc >= 11) {
    options.adaptive_tolerance_stage2 =
        parse_double(argv[10], "stage-2 adaptive tolerance");
  }
#else
  if (argc >= 7) {
    if (is_backend_name(argv[6])) {
      options.backend = parse_backend(argv[6]);
    } else {
      options.compare_solution = parse_comparison_flag(argv[6]);
    }
  }
  if (argc >= 8) {
    if (is_backend_name(argv[6])) {
      options.compare_solution = parse_comparison_flag(argv[7]);
    } else {
      options.backend = parse_backend(argv[7]);
    }
  }
#endif

#if defined(TREECODE_COMPARE_EXACT) && !defined(TREECODE_REUSE_EXPERIMENT)
  if (argc >= 9) {
    options.adaptive_tolerance_stage1 =
        parse_double(argv[8], "stage-1 adaptive tolerance");
  }
  if (argc >= 10) {
    options.adaptive_tolerance_stage2 =
        parse_double(argv[9], "stage-2 adaptive tolerance");
  }
#endif

#ifdef TREECODE_REUSE_EXPERIMENT
  if (argc >= 9) {
    options.sequence_steps = parse_int(argv[8], "sequence step count");
  }
  if (argc >= 10) {
    options.sequence_decay = parse_double(argv[9], "sequence decay");
  }
  if (argc >= 11) {
    options.sequence_initial_amplitude =
        parse_double(argv[10], "initial perturbation amplitude");
  }
  if (argc >= 12) {
    options.sequence_output_file = argv[11];
    if (options.sequence_output_file.empty()) {
      throw std::invalid_argument("OUTPUT_TSV must not be empty");
    }
  }
#endif

  if (options.initial_refinements < 0 || options.initial_refinements > 5) {
    throw std::invalid_argument("INITIAL_REFINEMENTS must be between 0 and 5");
  }
  if (options.quadrature_accuracy < 1 ||
      options.quadrature_accuracy >
          treecode_quadrature::maximum_algebraic_accuracy) {
    throw std::invalid_argument(
        "QUADRATURE_ACCURACY must be between 1 and "
        + std::to_string(
            treecode_quadrature::maximum_algebraic_accuracy));
  }
  if (options.expansion_order < 1 ||
      options.expansion_order > treecode_multipole::maximum_order) {
    throw std::invalid_argument(
        "EXPANSION_ORDER must be between 1 and "
        + std::to_string(treecode_multipole::maximum_order));
  }
  if (!(options.mac_theta > 0.0 && options.mac_theta < 1.0)) {
    throw std::invalid_argument("MAC_THETA must lie strictly between 0 and 1");
  }
#ifndef TREECODE_REUSE_EXPERIMENT
  if (!(options.adaptive_tolerance_stage1 > 0.0) ||
      !(options.adaptive_tolerance_stage2 > 0.0)) {
    throw std::invalid_argument("Adaptive tolerances must be positive");
  }
#endif
#ifdef TREECODE_REUSE_EXPERIMENT
  if (options.sequence_steps < 1 || options.sequence_steps > 1000) {
    throw std::invalid_argument("SEQUENCE_STEPS must be between 1 and 1000");
  }
  if (!(options.sequence_decay > 0.0 && options.sequence_decay < 1.0)) {
    throw std::invalid_argument("DECAY must lie strictly between 0 and 1");
  }
  if (!(options.sequence_initial_amplitude >= 0.0)) {
    throw std::invalid_argument(
        "INITIAL_PERTURBATION_AMPLITUDE must be non-negative");
  }
  if (options.backend != ComputeBackend::cpu) {
    throw std::invalid_argument(
        "The preliminary reuse experiment currently supports CPU only");
  }
  if (!options.compare_solution) {
    throw std::invalid_argument(
        "The preliminary reuse experiment requires COMPARE=1");
  }
#endif
#ifndef TREECODE_USE_CUDA
  if (options.backend == ComputeBackend::gpu) {
    throw std::invalid_argument(
        "GPU backend requires a CUDA-enabled executable; use tree_exact_gpu "
        "or build the exact-gpu target");
  }
#endif
  return options;
}

void writeNeighborElementsToFile(const std::string& filename,
                                 const std::set<HElement<DIM>*>& neighbor_of_the_elements,
                                 const RegularMesh<DIM>& r_mesh) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write the number of elements
    file << neighbor_of_the_elements.size() << std::endl;

    // Iterate over the neighbor elements
    for (const auto& hElement : neighbor_of_the_elements) {
        HGeometry<DIM>* current_geometry = hElement->h_element;

        // Write the vertices
        for (int j = 0; j <= DIM; ++j) {
            for (int k = 0; k < DIM; ++k) {
                file << r_mesh.point(current_geometry->vertex[j]->index)[k] << " ";
            }
            // file << std::endl;
        }
        file << std::endl;
    }

    file.close();
    // std::cout << "Neighbor elements data written to " << filename << std::endl;
}

#ifdef TREECODE_REUSE_EXPERIMENT
// A smooth, shifted perturbation supplies a nontrivial sequence of changing
// densities while retaining an analytic potential at every sequence step.
// The value is changed only between OpenMP regions and is read-only inside
// every parallel calculation.
double reuse_sequence_amplitude = 0.5;
double reuse_sequence_center_x = 0.70;
double reuse_sequence_center_y = -0.35;
double reuse_sequence_center_z = 0.20;

double reuse_perturbation_exact_value(const double* p)
{
  constexpr double a = 6.0 * PI;
  constexpr double b = 5.0 * PI;
  constexpr double c = 9.0 * PI;
  const double x = p[0] - reuse_sequence_center_x;
  const double y = p[1] - reuse_sequence_center_y;
  const double z = p[2] - reuse_sequence_center_z;
  return std::exp(-a * x * x - b * y * y - c * z * z);
}

double reuse_perturbation_source_value(const double* p)
{
  constexpr double a = 6.0 * PI;
  constexpr double b = 5.0 * PI;
  constexpr double c = 9.0 * PI;
  const double x = p[0] - reuse_sequence_center_x;
  const double y = p[1] - reuse_sequence_center_y;
  const double z = p[2] - reuse_sequence_center_z;
  const double solution = reuse_perturbation_exact_value(p);
  const double quadratic =
      a * a * x * x + b * b * y * y + c * c * z * z;
  return (2.0 * (a + b + c) - 4.0 * quadratic)
      * solution / (4.0 * PI);
}
#endif

double u(const double * p)
{
  double value = treecode_test_problem::exact_value(p[0], p[1], p[2]);
#ifdef TREECODE_REUSE_EXPERIMENT
  value += reuse_sequence_amplitude * reuse_perturbation_exact_value(p);
#endif
  return value;
};

double f(const double * p)
{
  double value = treecode_test_problem::source_value(p[0], p[1], p[2]);
#ifdef TREECODE_REUSE_EXPERIMENT
  value += reuse_sequence_amplitude * reuse_perturbation_source_value(p);
#endif
  return value;
};


double selected_source_value(const double* p, int source_component)
{
  (void)source_component;
  return f(p);
}

double tetrahedronVolume(const AFEPack::Point<3>& A, 
  const AFEPack::Point<3>& B, 
  const AFEPack::Point<3>& C, 
  const AFEPack::Point<3>& D) {
// Compute vectors
  AFEPack::Point<3> AB = { B[0] - A[0], B[1] - A[1], B[2] - A[2] };
  AFEPack::Point<3> AC = { C[0] - A[0], C[1] - A[1], C[2] - A[2] };
  AFEPack::Point<3> AD = { D[0] - A[0], D[1] - A[1], D[2] - A[2] };

  // Compute cross product AC × AD
  AFEPack::Point<3> crossProduct = {
    AC[1] * AD[2] - AC[2] * AD[1],
    AC[2] * AD[0] - AC[0] * AD[2],
    AC[0] * AD[1] - AC[1] * AD[0]
  };

  // Compute dot product (AB · (AC × AD))
  double dotProduct = AB[0] * crossProduct[0] + AB[1] * crossProduct[1] + AB[2] * crossProduct[2];

  // Compute volume
  return std::abs(dotProduct) / 6.0;
}

struct hElementCache
{
  // The MAC traversal visits every source subtree at most once, so these
  // interaction lists contain no duplicates by construction.  Flat vectors
  // avoid the allocation and O(log m) insertion cost of std::set.
  std::vector<HElement<DIM>*> neighbor_elements;
  std::vector<HElement<DIM>*> nonNeighbor_elements;

  // The current MAC traversal stores the lists used by the evaluator as
  // compact integer indices.  This halves list storage relative to pointers
  // on a 64-bit host and removes a std::map lookup from every far interaction.
  // The pointer lists above are retained only for the legacy geometric-layer
  // visualization helpers.
  std::vector<int> neighbor_leaf_indices;
  std::vector<int> far_hierarchy_indices;

  int level = 0;
  // Number of active (leaf) elements represented by this hierarchy node.
  // It is computed once in a bottom-up hierarchy pass and lets interaction
  // lists be checked without expanding every cluster to all descendants.
  std::size_t subtree_leaf_count = 0;
  std::size_t subtree_leaf_begin = 0;
  std::size_t subtree_leaf_end = 0;
  int parent_index = -1;
  std::vector<int> child_indices;
  double volume = 0.0;
  AFEPack::Point<DIM> ref_point;/// by default, we choose barycenter of each hElement
  double radius = 0.0;
  // Symmetric raw Cartesian moments M_(a,b,c), a+b+c <= expansion order.
  std::vector<double> moments;
};

struct ElementCache : public ElementAdditionalData<double, DIM>
{
  HElement<DIM>* the_hElement = nullptr;
  int hierarchy_index = -1;

  // JxW[q] * rho(q) is independent of the target.  Cache it once, as the GPU
  // path does, instead of reevaluating the Gaussian for every near-field pair.
  std::vector<double> source_charge;

  // Retain the 4/5/7 physical vertices so a target-containing twin or
  // four-tetrahedron macroelement can be decomposed consistently for the
  // analytic singular self-interaction.
  std::vector<AFEPack::Point<DIM>> physical_vertices;

};


#include <omp.h>
void build_element_mapping(
    const std::vector<HElement<DIM>*>& root_element_list,
    std::map<HElement<DIM>*, int>& map_HElement_to_int,
    std::vector<hElementCache>& hElement_cache,
    std::vector<int>& element_levels)
{
    int idx = 0;
    map_HElement_to_int.clear();
    element_levels.clear();
    
    // 遍历所有根元素
    for(auto root : root_element_list) {
        std::stack<std::pair<HElement<DIM>*, int>> stack;
        stack.push({root, 0});
        
        while(!stack.empty()) {
            HElement<DIM>* current = stack.top().first;
            const int current_level = stack.top().second;
            stack.pop();
            
            // 建立映射
            map_HElement_to_int[current] = idx++;
            element_levels.push_back(current_level);
            
            // 如果有子元素，压入栈中
            // HElement keeps allocated children after coarsening.  `value`
            // describes the active irregular-mesh subtree: 1 is an active
            // parent, 0 is an active leaf, and negative values are inactive.
            if(current->value == 1) {
                // 为了保持顺序，反向压栈
                for(int i = static_cast<int>(current->child.size()) - 1; i >= 0; --i) {
                    stack.push({current->child[i], current_level + 1});
                }
            }
        }
    }
    
    // 预分配cache空间
    hElement_cache.resize(idx);
    
    std::cout << "Mapping built: " << idx << " elements" << std::endl;
}

void initialize_hierarchy_geometry(
    HElement<DIM>* element,
    hElementCache& cache,
    int level)
{
    cache.level = level;
    cache.neighbor_elements.clear();
    cache.nonNeighbor_elements.clear();
    cache.neighbor_leaf_indices.clear();
    cache.far_hierarchy_indices.clear();

    const auto& vertices = element->h_element->vertex;
    cache.ref_point = *vertices[0] + *vertices[1] + *vertices[2] + *vertices[3];
    cache.ref_point /= 4.0;

    cache.radius = 0.0;
    for (const auto* vertex : vertices) {
      cache.radius =
          std::max(cache.radius, (*vertex - cache.ref_point).length());
    }
    cache.volume = tetrahedronVolume(
        *vertices[0], *vertices[1], *vertices[2], *vertices[3]);
}

bool is_ancestor_of(const HElement<DIM>* possible_ancestor,
                    const HElement<DIM>* element)
{
    for (const HElement<DIM>* current = element;
         current != nullptr;
         current = current->parent) {
      if (current == possible_ancestor) {
        return true;
      }
    }
    return false;
}

struct InteractionListStats
{
  std::size_t represented_leaf_count = 0;
  std::size_t near_entries = 0;
  std::size_t far_entries = 0;
  std::size_t visited_hierarchy_nodes = 0;
  std::size_t hnl_candidate_regions = 0;
  std::size_t hnl_cover_regions = 0;
  std::size_t hnl_star_queries = 0;
  std::size_t hnl_raw_candidate_visits = 0;
  std::size_t hnl_unique_level_candidates = 0;
  std::size_t hnl_completed_sibling_entries = 0;
  std::size_t hnl_materialization_visits = 0;
  long double represented_volume = 0.0L;
  long double near_volume = 0.0L;
  double maximum_mac_ratio = 0.0;
  bool contains_target_in_near_field = false;
  bool unique_entries = true;
};

// The production HNL path keeps the three logical states requested in the
// algorithm, but stores integer IDs in flat vectors.  This avoids allocating
// red-black tree nodes for every target leaf.
using VertexLevelIndex =
    std::vector<std::vector<std::vector<int>>>;

struct HNLState
{
  std::vector<int> frontier_vertices;
  std::vector<int> covered_source_regions;
  std::vector<int> candidate_regions;
  std::vector<int> current_layer;
  std::vector<int> next_layer;
  std::vector<int> level_candidates;
  std::vector<int> completed_siblings;
  std::vector<unsigned int> vertex_marks;
  unsigned int vertex_epoch = 0;
  const std::vector<HElement<DIM>*>* hierarchy_elements = nullptr;
  const std::vector<hElementCache>* hierarchy_cache = nullptr;
  std::size_t star_queries = 0;
  std::size_t raw_candidate_visits = 0;
  std::size_t unique_level_candidates = 0;
  std::size_t completed_sibling_entries = 0;
  std::size_t materialization_visits = 0;
};

void hnl_begin_target(
    HNLState& state,
    std::size_t vertex_count,
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    const std::vector<hElementCache>& hierarchy_cache)
{
  state.frontier_vertices.clear();
  state.covered_source_regions.clear();
  state.candidate_regions.clear();
  state.current_layer.clear();
  state.next_layer.clear();
  state.level_candidates.clear();
  state.completed_siblings.clear();
  state.hierarchy_elements = &hierarchy_elements;
  state.hierarchy_cache = &hierarchy_cache;
  state.star_queries = 0;
  state.raw_candidate_visits = 0;
  state.unique_level_candidates = 0;
  state.completed_sibling_entries = 0;
  state.materialization_visits = 0;
  if (state.vertex_marks.size() != vertex_count) {
    state.vertex_marks.assign(vertex_count, 0U);
    state.vertex_epoch = 0;
  }
}

void hnl_sort_unique(std::vector<int>& values)
{
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

void hnl_collect_frontier_vertices(
    const std::vector<int>& current_layer,
    HNLState& state)
{
  state.frontier_vertices.clear();
  // Stamps deduplicate vertices within this level pass.  The epoch advances
  // at every level because the same geometric vertex may occur at coarser
  // levels with a different level-indexed incidence list.
  ++state.vertex_epoch;
  if (state.vertex_epoch == 0U) {
    std::fill(state.vertex_marks.begin(), state.vertex_marks.end(), 0U);
    state.vertex_epoch = 1U;
  }
  const auto& elements = *state.hierarchy_elements;
  for (int element_index : current_layer) {
    if (element_index < 0 ||
        static_cast<std::size_t>(element_index) >= elements.size()) {
      continue;
    }
    HElement<DIM>* element = elements[static_cast<std::size_t>(element_index)];
    if (element == nullptr || element->h_element == nullptr) {
      continue;
    }
    for (HGeometry<0, DIM>* vertex : element->h_element->vertex) {
      if (vertex == nullptr || vertex->index < 0 ||
          static_cast<std::size_t>(vertex->index) >= state.vertex_marks.size()) {
        continue;
      }
      const std::size_t vertex_index =
          static_cast<std::size_t>(vertex->index);
      if (state.vertex_marks[vertex_index] != state.vertex_epoch) {
        state.vertex_marks[vertex_index] = state.vertex_epoch;
        state.frontier_vertices.push_back(vertex->index);
      }
    }
  }
}

// Record a candidate source region.  The final disjoint cover is formed after
// all levels have been expanded, so this hot path only appends an integer.
void hnl_add_candidate_source_region(int region_index, HNLState& state)
{
  if (region_index >= 0 && state.hierarchy_elements != nullptr &&
      static_cast<std::size_t>(region_index) <
          state.hierarchy_elements->size()) {
    state.candidate_regions.push_back(region_index);
  }
}

std::pair<bool, bool> hnl_candidate_flags(
    int region_index,
    const std::vector<int>& sorted_candidates,
    const std::vector<hElementCache>& hierarchy_cache)
{
  const hElementCache& region =
      hierarchy_cache[static_cast<std::size_t>(region_index)];
  const std::size_t begin = region.subtree_leaf_begin;
  const std::size_t end = region.subtree_leaf_end;
  auto it = std::lower_bound(
      sorted_candidates.begin(), sorted_candidates.end(), begin,
      [&hierarchy_cache](int candidate, std::size_t value) {
        return hierarchy_cache[static_cast<std::size_t>(candidate)]
                   .subtree_leaf_begin < value;
      });
  bool is_candidate = false;
  bool has_descendant = false;
  for (; it != sorted_candidates.end(); ++it) {
    const hElementCache& candidate =
        hierarchy_cache[static_cast<std::size_t>(*it)];
    if (candidate.subtree_leaf_begin >= end) {
      break;
    }
    if (*it == region_index) {
      is_candidate = true;
    } else if (candidate.subtree_leaf_begin >= begin &&
               candidate.subtree_leaf_end <= end) {
      has_descendant = true;
    }
  }
  return {is_candidate, has_descendant};
}

// Convert candidate regions into a disjoint source cover.  Split a region
// exactly when it has a candidate descendant; otherwise retain the whole
// region.  Every child of a split region is visited, including children with
// no candidate descendants, so coverage does not depend on sibling closure.
void hnl_materialize_source_cover(
    int region_index,
    const std::vector<int>& sorted_candidates,
    HNLState& state)
{
  ++state.materialization_visits;
  const auto& elements = *state.hierarchy_elements;
  const auto& cache = *state.hierarchy_cache;
  if (region_index < 0 ||
      static_cast<std::size_t>(region_index) >= elements.size()) {
    return;
  }
  const auto flags = hnl_candidate_flags(region_index, sorted_candidates, cache);
  const HElement<DIM>* region =
      elements[static_cast<std::size_t>(region_index)];
  if (region == nullptr) {
    return;
  }
  if (region->value == 0 || !flags.second) {
    state.covered_source_regions.push_back(region_index);
    return;
  }
  for (int child_index : cache[static_cast<std::size_t>(region_index)].child_indices) {
    hnl_materialize_source_cover(child_index, sorted_candidates, state);
  }
}

void build_hnl_source_cover(
    int target_index,
    const std::vector<int>& root_indices,
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    const std::vector<hElementCache>& hierarchy_cache,
    const VertexLevelIndex& node_neighbors,
    HNLState& state)
{
  const hElementCache& target_cache =
      hierarchy_cache[static_cast<std::size_t>(target_index)];
  hnl_begin_target(
      state, node_neighbors.size(), hierarchy_elements, hierarchy_cache);
  state.current_layer.push_back(target_index);
  hnl_add_candidate_source_region(target_index, state);

  for (int level = target_cache.level; level > 0; --level) {
    hnl_collect_frontier_vertices(state.current_layer, state);
    state.level_candidates.clear();
    for (int vertex_index : state.frontier_vertices) {
      if (vertex_index < 0 ||
          static_cast<std::size_t>(vertex_index) >= node_neighbors.size() ||
          static_cast<std::size_t>(level) >=
              node_neighbors[static_cast<std::size_t>(vertex_index)].size()) {
        continue;
      }
      ++state.star_queries;
      const auto& star = node_neighbors[static_cast<std::size_t>(vertex_index)]
                                      [static_cast<std::size_t>(level)];
      state.raw_candidate_visits += star.size();
      state.level_candidates.insert(
          state.level_candidates.end(), star.begin(), star.end());
    }
    hnl_sort_unique(state.level_candidates);
    state.unique_level_candidates += state.level_candidates.size();

    // Complete sibling families from the full vertex-star union first.  Only
    // then remove regions already present in the current layer.
    state.completed_siblings.clear();
    for (int element_index : state.level_candidates) {
      const hElementCache& element_cache =
          hierarchy_cache[static_cast<std::size_t>(element_index)];
      if (element_cache.parent_index >= 0) {
        const auto& siblings = hierarchy_cache[
            static_cast<std::size_t>(element_cache.parent_index)].child_indices;
        state.completed_siblings.insert(
            state.completed_siblings.end(), siblings.begin(), siblings.end());
      } else {
        state.completed_siblings.push_back(element_index);
      }
    }
    hnl_sort_unique(state.completed_siblings);
    state.completed_siblings.erase(
        std::remove_if(
            state.completed_siblings.begin(), state.completed_siblings.end(),
            [&state](int value) {
              return std::binary_search(
                  state.current_layer.begin(), state.current_layer.end(), value);
            }),
        state.completed_siblings.end());
    state.completed_sibling_entries += state.completed_siblings.size();
    for (int region_index : state.completed_siblings) {
      hnl_add_candidate_source_region(region_index, state);
    }

    state.next_layer.clear();
    for (int element_index : state.current_layer) {
      const int parent_index = hierarchy_cache[
          static_cast<std::size_t>(element_index)].parent_index;
      if (parent_index >= 0) {
        state.next_layer.push_back(parent_index);
      }
    }
    for (int element_index : state.completed_siblings) {
      const int parent_index = hierarchy_cache[
          static_cast<std::size_t>(element_index)].parent_index;
      if (parent_index >= 0) {
        state.next_layer.push_back(parent_index);
      }
    }
    hnl_sort_unique(state.next_layer);
    state.current_layer.swap(state.next_layer);
    if (state.current_layer.empty()) {
      break;
    }
  }

  for (int root_index : root_indices) {
    hnl_add_candidate_source_region(root_index, state);
  }

  hnl_sort_unique(state.candidate_regions);
  std::sort(
      state.candidate_regions.begin(), state.candidate_regions.end(),
      [&hierarchy_cache](int left, int right) {
        const hElementCache& lhs =
            hierarchy_cache[static_cast<std::size_t>(left)];
        const hElementCache& rhs =
            hierarchy_cache[static_cast<std::size_t>(right)];
        if (lhs.subtree_leaf_begin != rhs.subtree_leaf_begin) {
          return lhs.subtree_leaf_begin < rhs.subtree_leaf_begin;
        }
        return lhs.subtree_leaf_end > rhs.subtree_leaf_end;
      });

  for (int root_index : root_indices) {
    hnl_materialize_source_cover(root_index, state.candidate_regions, state);
  }
}

void classify_hnl_region(
    int target_index,
    int source_index,
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    const std::vector<hElementCache>& hierarchy_cache,
    hElementCache& target_cache,
    double theta,
    InteractionListStats& stats)
{
  if (source_index < 0 ||
      static_cast<std::size_t>(source_index) >= hierarchy_elements.size()) {
    stats.unique_entries = false;
    return;
  }
  HElement<DIM>* source = hierarchy_elements[static_cast<std::size_t>(source_index)];
  if (source == nullptr) {
    stats.unique_entries = false;
    return;
  }
  ++stats.visited_hierarchy_nodes;
  const hElementCache& source_cache =
      hierarchy_cache[static_cast<std::size_t>(source_index)];
  const double center_distance =
      (source_cache.ref_point - target_cache.ref_point).length();
  const double minimum_target_distance =
      center_distance - target_cache.radius;
  const hElementCache& target_hierarchy_cache =
      hierarchy_cache[static_cast<std::size_t>(target_index)];
  const bool contains_target =
      source_cache.subtree_leaf_begin <= target_hierarchy_cache.subtree_leaf_begin &&
      target_hierarchy_cache.subtree_leaf_begin < source_cache.subtree_leaf_end;
  const bool mac_accepted =
      !contains_target && minimum_target_distance > 0.0 &&
      source_cache.radius / minimum_target_distance <= theta;

  if (mac_accepted) {
    target_cache.far_hierarchy_indices.push_back(source_index);
    stats.represented_leaf_count += source_cache.subtree_leaf_count;
    stats.represented_volume += source_cache.volume;
    ++stats.far_entries;
    stats.maximum_mac_ratio = std::max(
        stats.maximum_mac_ratio,
        source_cache.radius / minimum_target_distance);
    return;
  }
  if (source->value == 0) {
    target_cache.neighbor_leaf_indices.push_back(source->index);
    ++stats.represented_leaf_count;
    stats.represented_volume += source_cache.volume;
    stats.near_volume += source_cache.volume;
    ++stats.near_entries;
    stats.contains_target_in_near_field =
        stats.contains_target_in_near_field || source ==
            hierarchy_elements[static_cast<std::size_t>(target_index)];
    return;
  }
  for (int child_index : source_cache.child_indices) {
    classify_hnl_region(
        target_index, child_index, hierarchy_elements, hierarchy_cache,
        target_cache, theta, stats);
  }
}

void build_hnl_mac_interaction_lists(
    int target_index,
    const std::vector<int>& root_indices,
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    const std::vector<hElementCache>& hierarchy_cache,
    const VertexLevelIndex& node_neighbors,
    hElementCache& target_cache,
    double theta,
    HNLState& state,
    InteractionListStats& stats)
{
  target_cache.neighbor_elements.clear();
  target_cache.nonNeighbor_elements.clear();
  target_cache.neighbor_leaf_indices.clear();
  target_cache.far_hierarchy_indices.clear();
  stats = InteractionListStats{};

  build_hnl_source_cover(
      target_index, root_indices, hierarchy_elements, hierarchy_cache,
      node_neighbors, state);
  stats.hnl_candidate_regions = state.candidate_regions.size();
  stats.hnl_cover_regions = state.covered_source_regions.size();
  stats.hnl_star_queries = state.star_queries;
  stats.hnl_raw_candidate_visits = state.raw_candidate_visits;
  stats.hnl_unique_level_candidates = state.unique_level_candidates;
  stats.hnl_completed_sibling_entries = state.completed_sibling_entries;
  stats.hnl_materialization_visits = state.materialization_visits;
  for (int source_index : state.covered_source_regions) {
    classify_hnl_region(
        target_index, source_index, hierarchy_elements, hierarchy_cache,
        target_cache, theta, stats);
  }
}

// Matched ablation baseline: start MAC recursion directly from every source
// root.  It deliberately shares classify_hnl_region(), list storage, target
// ordering, scheduling, and validation with the HNL path.  Consequently, a
// TREECODE_ROOT_BASED_LISTS build changes only the source-search strategy.
void build_root_mac_interaction_lists(
    int target_index,
    const std::vector<int>& root_indices,
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    const std::vector<hElementCache>& hierarchy_cache,
    hElementCache& target_cache,
    double theta,
    InteractionListStats& stats)
{
  target_cache.neighbor_elements.clear();
  target_cache.nonNeighbor_elements.clear();
  target_cache.neighbor_leaf_indices.clear();
  target_cache.far_hierarchy_indices.clear();
  stats = InteractionListStats{};
  for (int root_index : root_indices) {
    classify_hnl_region(
        target_index, root_index, hierarchy_elements, hierarchy_cache,
        target_cache, theta, stats);
  }
}

struct HierarchySummary
{
  std::size_t leaf_count = 0;
  double domain_volume = 0.0;
};

std::uint64_t mix_interaction_id(std::uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

void save_interaction_fingerprints(
    const std::vector<HElement<DIM>*>& elements,
    const std::vector<hElementCache>& cache,
    std::map<std::string, double>& diagnostics)
{
  std::uint64_t near_xor = 0;
  std::uint64_t near_sum = 0;
  std::uint64_t far_xor = 0;
  std::uint64_t far_sum = 0;
  for (std::size_t target = 0; target < elements.size(); ++target) {
    if (elements[target] == nullptr || elements[target]->value != 0) {
      continue;
    }
    for (int source : cache[target].neighbor_leaf_indices) {
      const std::uint64_t key =
          (static_cast<std::uint64_t>(target) << 32U)
          ^ static_cast<std::uint32_t>(source);
      const std::uint64_t mixed = mix_interaction_id(key ^ 0x4e454152ULL);
      near_xor ^= mixed;
      near_sum += mixed;
    }
    for (int source : cache[target].far_hierarchy_indices) {
      const std::uint64_t key =
          (static_cast<std::uint64_t>(target) << 32U)
          ^ static_cast<std::uint32_t>(source);
      const std::uint64_t mixed = mix_interaction_id(key ^ 0x464152ULL);
      far_xor ^= mixed;
      far_sum += mixed;
    }
  }
  const auto save_u64 = [&diagnostics](const std::string& name,
                                       std::uint64_t value) {
    diagnostics[name + "_hi"] = static_cast<double>(value >> 32U);
    diagnostics[name + "_lo"] =
        static_cast<double>(value & 0xffffffffULL);
  };
  save_u64("near_fingerprint_xor", near_xor);
  save_u64("near_fingerprint_sum", near_sum);
  save_u64("far_fingerprint_xor", far_xor);
  save_u64("far_fingerprint_sum", far_sum);
}

HierarchySummary prepare_hierarchy_aggregates(
    const std::vector<HElement<DIM>*>& elements,
    const std::vector<HElement<DIM>*>& roots,
    std::vector<hElementCache>& hierarchy_cache,
    const std::map<HElement<DIM>*, int>& element_to_index)
{
    std::size_t leaf_count = 0;
    for (HElement<DIM>* element : elements) {
      if (element != nullptr && element->value == 0) {
        ++leaf_count;
      }
    }
    std::vector<HElement<DIM>*> leaves(leaf_count, nullptr);
    std::vector<double> leaf_volumes(leaf_count, 0.0);
    std::size_t next_leaf_ordinal = 0;
    for (HElement<DIM>* element : elements) {
      if (element == nullptr) {
        throw std::runtime_error("Null element in hierarchy mapping");
      }
      hElementCache& element_cache =
          hierarchy_cache[element_to_index.at(element)];
      element_cache.child_indices.clear();
      if (element->value == 1) {
        element_cache.child_indices.reserve(element->child.size());
        for (HElement<DIM>* child : element->child) {
          element_cache.child_indices.push_back(element_to_index.at(child));
        }
      }
      if (element->value == 0) {
        const int leaf_index = element->index;
        if (leaf_index < 0
            || static_cast<std::size_t>(leaf_index) >= leaf_count
            || leaves[static_cast<std::size_t>(leaf_index)] != nullptr) {
          throw std::runtime_error(
              "Leaf indices are not a unique zero-based permutation");
        }
        leaves[static_cast<std::size_t>(leaf_index)] = element;
        leaf_volumes[static_cast<std::size_t>(leaf_index)] =
            element_cache.volume;
        element_cache.subtree_leaf_begin = next_leaf_ordinal;
        element_cache.subtree_leaf_end = ++next_leaf_ordinal;
      }
    }
    if (std::find(leaves.begin(), leaves.end(), nullptr) != leaves.end()) {
      throw std::runtime_error("Leaf index coverage contains a gap");
    }

    double domain_volume = 0.0;
    for (double volume : leaf_volumes) {
      if (!(volume > 0.0) || !std::isfinite(volume)) {
        throw std::runtime_error("A leaf has invalid physical volume");
      }
      domain_volume += volume;
    }
    const double hierarchy_tolerance =
        1.0e-10 * std::max(1.0, domain_volume);
#ifndef TREECODE_DYNAMIC_MESH_MODES
    double root_volume = 0.0;
    for (HElement<DIM>* root : roots) {
      if (root == nullptr || root->parent != nullptr) {
        throw std::runtime_error("Invalid root in hierarchy");
      }
      root_volume += hierarchy_cache[element_to_index.at(root)].volume;
    }
    if (std::abs(root_volume - domain_volume) > hierarchy_tolerance) {
      throw std::runtime_error(
          "Root volume does not equal the complete leaf-mesh volume");
    }
#endif
    // build_element_mapping stores every parent before its children.  The
    // reverse pass is therefore a postorder pass.  Cache each subtree's leaf
    // count and verify the hierarchy once, in O(number of hierarchy nodes).
    for (std::size_t reverse_index = elements.size();
         reverse_index > 0;
         --reverse_index) {
      HElement<DIM>* element = elements[reverse_index - 1];
      hElementCache& element_cache =
          hierarchy_cache[element_to_index.at(element)];
      element_cache.parent_index =
          element->parent == nullptr
              ? -1
              : element_to_index.at(element->parent);
      if (element->value == 0) {
        element_cache.subtree_leaf_count = 1;
        continue;
      }
      if (element->child.empty()) {
        throw std::runtime_error(
            "A refined hierarchy element has no children");
      }
      double child_volume = 0.0;
      std::size_t child_leaf_count = 0;
      std::size_t expected_child_begin = 0;
      bool first_child = true;
      for (HElement<DIM>* child : element->child) {
        if (child == nullptr || child->parent != element) {
          throw std::runtime_error(
              "Invalid parent-child link in mesh hierarchy");
        }
        const hElementCache& child_cache =
            hierarchy_cache[element_to_index.at(child)];
        if (child_cache.subtree_leaf_count == 0) {
          throw std::runtime_error(
              "Hierarchy order is not parent-before-child");
        }
        child_volume += child_cache.volume;
        child_leaf_count += child_cache.subtree_leaf_count;
        if (first_child) {
          element_cache.subtree_leaf_begin =
              child_cache.subtree_leaf_begin;
          expected_child_begin = child_cache.subtree_leaf_begin;
          first_child = false;
        }
        if (child_cache.subtree_leaf_begin != expected_child_begin) {
          throw std::runtime_error(
              "Child subtrees are not contiguous in hierarchy order");
        }
        expected_child_begin = child_cache.subtree_leaf_end;
      }
#ifdef TREECODE_DYNAMIC_MESH_MODES
      // After r-adaptation, a hierarchy parent is a cluster rather than a
      // physical straight-sided tetrahedron.  Rebuild its aggregate volume
      // and a conservative bounding sphere from its active children.
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        element_cache.ref_point[coordinate] = 0.0;
      }
      for (HElement<DIM>* child : element->child) {
        const hElementCache& child_cache =
            hierarchy_cache[element_to_index.at(child)];
        for (int coordinate = 0; coordinate < DIM; ++coordinate) {
          element_cache.ref_point[coordinate] +=
              child_cache.volume * child_cache.ref_point[coordinate];
        }
      }
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        element_cache.ref_point[coordinate] /= child_volume;
      }
      element_cache.radius = 0.0;
      for (HElement<DIM>* child : element->child) {
        const hElementCache& child_cache =
            hierarchy_cache[element_to_index.at(child)];
        element_cache.radius = std::max(
            element_cache.radius,
            (child_cache.ref_point - element_cache.ref_point).length()
                + child_cache.radius);
      }
      element_cache.volume = child_volume;
#else
      const double tolerance =
          1.0e-10 * std::max(1.0, element_cache.volume);
      if (std::abs(child_volume - element_cache.volume) > tolerance) {
        throw std::runtime_error(
            "Children do not cover their parent element volume");
      }
#endif
      element_cache.subtree_leaf_count = child_leaf_count;
      element_cache.subtree_leaf_end = expected_child_begin;
    }

#ifdef TREECODE_DYNAMIC_MESH_MODES
    double root_volume = 0.0;
    for (HElement<DIM>* root : roots) {
      if (root == nullptr || root->parent != nullptr) {
        throw std::runtime_error("Invalid root in hierarchy");
      }
      root_volume += hierarchy_cache[element_to_index.at(root)].volume;
    }
    if (std::abs(root_volume - domain_volume) > hierarchy_tolerance) {
      throw std::runtime_error(
          "Root clusters do not cover the complete leaf-mesh volume");
    }
#endif

    std::size_t root_leaf_count = 0;
    for (HElement<DIM>* root : roots) {
      const hElementCache& root_cache =
          hierarchy_cache[element_to_index.at(root)];
      if (root_cache.subtree_leaf_begin != root_leaf_count) {
        throw std::runtime_error(
            "Root subtrees are not a contiguous domain partition");
      }
      root_leaf_count += root_cache.subtree_leaf_count;
      if (root_cache.subtree_leaf_end != root_leaf_count) {
        throw std::runtime_error(
            "A root leaf interval has an invalid extent");
      }
    }
    if (root_leaf_count != leaf_count) {
      throw std::runtime_error(
          "Root subtrees do not cover every active element");
    }
    return {leaf_count, domain_volume};
}

void validate_interaction_stats(
    const std::vector<HElement<DIM>*>& elements,
    const std::vector<InteractionListStats>& interaction_stats,
    const HierarchySummary& hierarchy,
    double theta,
    std::map<std::string, double>* diagnostics)
{
    struct RangeAccumulator {
      std::size_t minimum = std::numeric_limits<std::size_t>::max();
      std::size_t maximum = 0;
      long double total = 0.0L;
      void add(std::size_t value) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        total += static_cast<long double>(value);
      }
    };
    std::size_t validated_target_count = 0;
    std::size_t total_near = 0;
    std::size_t total_far = 0;
    std::size_t total_visited = 0;
    std::size_t minimum_near = std::numeric_limits<std::size_t>::max();
    std::size_t maximum_near = 0;
    std::size_t minimum_far = std::numeric_limits<std::size_t>::max();
    std::size_t maximum_far = 0;
    long double minimum_near_volume =
        std::numeric_limits<long double>::infinity();
    long double maximum_near_volume = 0.0L;
    long double total_near_volume = 0.0L;
    long double maximum_partition_volume_error = 0.0L;
    const long double domain_volume = hierarchy.domain_volume;
    const long double volume_tolerance =
        1.0e-10L * std::max(1.0L, domain_volume);
    double largest_ratio = 0.0;
    RangeAccumulator list_entries_range;
    RangeAccumulator visited_range;
    RangeAccumulator candidate_range;
    RangeAccumulator cover_range;
    RangeAccumulator star_query_range;
    RangeAccumulator raw_candidate_range;
    RangeAccumulator unique_candidate_range;
    RangeAccumulator sibling_range;
    RangeAccumulator materialization_range;

    for (std::size_t i = 0; i < elements.size(); ++i) {
      if (elements[i]->value == 1) {
        continue;
      }
      ++validated_target_count;
      const InteractionListStats& stats = interaction_stats[i];
      if (!stats.unique_entries) {
        throw std::runtime_error(
            "MAC traversal generated a null or duplicate interaction at target "
            + std::to_string(i));
      }
      if (!stats.contains_target_in_near_field) {
        throw std::runtime_error(
            "MAC traversal failed to include the target in its near field at target "
            + std::to_string(i));
      }
      if (stats.represented_leaf_count != hierarchy.leaf_count) {
        throw std::runtime_error(
            "Near/far interaction lists do not represent every source leaf at target "
            + std::to_string(i) + ": represented="
            + std::to_string(stats.represented_leaf_count)
            + ", expected=" + std::to_string(hierarchy.leaf_count));
      }
      const long double partition_volume_error =
          std::abs(stats.represented_volume - domain_volume);
      if (partition_volume_error > volume_tolerance) {
        throw std::runtime_error(
            "Near/far interaction lists do not cover the truncated domain");
      }
      if (stats.maximum_mac_ratio > theta * (1.0 + 1.0e-12)) {
        throw std::runtime_error("A far-field interaction violates the MAC");
      }

      largest_ratio = std::max(largest_ratio, stats.maximum_mac_ratio);
      maximum_partition_volume_error =
          std::max(maximum_partition_volume_error, partition_volume_error);
      total_near += stats.near_entries;
      total_far += stats.far_entries;
      total_visited += stats.visited_hierarchy_nodes;
      minimum_near = std::min(minimum_near, stats.near_entries);
      maximum_near = std::max(maximum_near, stats.near_entries);
      minimum_far = std::min(minimum_far, stats.far_entries);
      maximum_far = std::max(maximum_far, stats.far_entries);
      minimum_near_volume =
          std::min(minimum_near_volume, stats.near_volume);
      maximum_near_volume =
          std::max(maximum_near_volume, stats.near_volume);
      total_near_volume += stats.near_volume;
      list_entries_range.add(stats.near_entries + stats.far_entries);
      visited_range.add(stats.visited_hierarchy_nodes);
      candidate_range.add(stats.hnl_candidate_regions);
      cover_range.add(stats.hnl_cover_regions);
      star_query_range.add(stats.hnl_star_queries);
      raw_candidate_range.add(stats.hnl_raw_candidate_visits);
      unique_candidate_range.add(stats.hnl_unique_level_candidates);
      sibling_range.add(stats.hnl_completed_sibling_entries);
      materialization_range.add(stats.hnl_materialization_visits);
    }

    if (validated_target_count != hierarchy.leaf_count) {
      throw std::runtime_error(
          "Not every physical leaf received an interaction list");
    }
    const double target_denominator =
        hierarchy.leaf_count == 0
            ? 1.0
            : static_cast<double>(hierarchy.leaf_count);
    if (diagnostics != nullptr) {
      auto save_range = [diagnostics, target_denominator](
          const std::string& name, const RangeAccumulator& range) {
        (*diagnostics)[name + "_min"] = static_cast<double>(range.minimum);
        (*diagnostics)[name + "_avg"] =
            static_cast<double>(range.total) / target_denominator;
        (*diagnostics)[name + "_max"] = static_cast<double>(range.maximum);
      };
      (*diagnostics)["interaction_targets"] =
          static_cast<double>(validated_target_count);
      (*diagnostics)["near_entries_total"] = static_cast<double>(total_near);
      (*diagnostics)["far_entries_total"] = static_cast<double>(total_far);
      (*diagnostics)["near_entries_min"] = static_cast<double>(minimum_near);
      (*diagnostics)["near_entries_avg"] =
          static_cast<double>(total_near) / target_denominator;
      (*diagnostics)["near_entries_max"] = static_cast<double>(maximum_near);
      (*diagnostics)["far_entries_min"] = static_cast<double>(minimum_far);
      (*diagnostics)["far_entries_avg"] =
          static_cast<double>(total_far) / target_denominator;
      (*diagnostics)["far_entries_max"] = static_cast<double>(maximum_far);
      save_range("list_entries", list_entries_range);
      save_range("visited_nodes", visited_range);
      save_range("hnl_candidate_regions", candidate_range);
      save_range("hnl_cover_regions", cover_range);
      save_range("hnl_star_queries", star_query_range);
      save_range("hnl_raw_candidate_visits", raw_candidate_range);
      save_range("hnl_unique_level_candidates", unique_candidate_range);
      save_range("hnl_completed_sibling_entries", sibling_range);
      save_range("hnl_materialization_visits", materialization_range);
      (*diagnostics)["maximum_mac_ratio"] = largest_ratio;
      (*diagnostics)["maximum_partition_volume_error"] =
          static_cast<double>(maximum_partition_volume_error);
    }
    std::cout << "Validated " << validated_target_count
              << " disjoint source-tree partitions over truncated-domain "
                 "volume "
              << std::setprecision(15) << hierarchy.domain_volume
              << "; max MAC ratio=" << largest_ratio
              << ", near entries min/avg/max="
              << minimum_near << "/"
              << static_cast<double>(total_near) / target_denominator
              << "/" << maximum_near
              << ", far entries min/avg/max="
              << minimum_far << "/"
              << static_cast<double>(total_far) / target_denominator
              << "/" << maximum_far
              << ", visited nodes avg="
              << static_cast<double>(total_visited) / target_denominator
              << ", near volume min/avg/max="
              << static_cast<double>(minimum_near_volume) << "/"
              << static_cast<double>(
                     total_near_volume / target_denominator)
              << "/" << static_cast<double>(maximum_near_volume)
              << "; max partition-volume error="
              << static_cast<double>(maximum_partition_volume_error)
              << '\n';
}

std::vector<std::vector<std::set<HElement<DIM>* > > >
build_vertex_level_index(
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    const std::vector<int>& element_levels)
{
  int maximum_vertex_index = -1;
  for (HElement<DIM>* element : hierarchy_elements) {
    if (element == nullptr || element->h_element == nullptr) {
      continue;
    }
    for (HGeometry<0, DIM>* vertex : element->h_element->vertex) {
      if (vertex != nullptr) {
        maximum_vertex_index = std::max(maximum_vertex_index, vertex->index);
      }
    }
  }
  if (maximum_vertex_index < 0) {
    return {};
  }

  std::vector<std::vector<std::set<HElement<DIM>* > > > node_neighbors(
      static_cast<std::size_t>(maximum_vertex_index + 1));
  for (std::size_t i = 0; i < hierarchy_elements.size(); ++i) {
    HElement<DIM>* element = hierarchy_elements[i];
    if (element == nullptr || element->h_element == nullptr ||
        i >= element_levels.size() || element_levels[i] < 0) {
      continue;
    }
    const std::size_t level = static_cast<std::size_t>(element_levels[i]);
    for (HGeometry<0, DIM>* vertex : element->h_element->vertex) {
      if (vertex == nullptr || vertex->index < 0) {
        continue;
      }
      auto& levels = node_neighbors[static_cast<std::size_t>(vertex->index)];
      if (levels.size() <= level) {
        levels.resize(level + 1);
      }
      levels[level].insert(element);
    }
  }
  return node_neighbors;
}

VertexLevelIndex build_vertex_level_index_ids(
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    const std::vector<int>& element_levels,
    const std::map<HElement<DIM>*, int>& element_to_index)
{
  int maximum_vertex_index = -1;
  for (HElement<DIM>* element : hierarchy_elements) {
    if (element == nullptr || element->h_element == nullptr) {
      continue;
    }
    for (HGeometry<0, DIM>* vertex : element->h_element->vertex) {
      if (vertex != nullptr) {
        maximum_vertex_index = std::max(maximum_vertex_index, vertex->index);
      }
    }
  }
  if (maximum_vertex_index < 0) {
    return {};
  }

  VertexLevelIndex node_neighbors(
      static_cast<std::size_t>(maximum_vertex_index + 1));
  for (std::size_t element_index = 0;
       element_index < hierarchy_elements.size(); ++element_index) {
    HElement<DIM>* element = hierarchy_elements[element_index];
    if (element == nullptr || element->h_element == nullptr ||
        element_index >= element_levels.size() ||
        element_levels[element_index] < 0) {
      continue;
    }
    const int level = element_levels[element_index];
    const int compact_index = element_to_index.at(element);
    for (HGeometry<0, DIM>* vertex : element->h_element->vertex) {
      if (vertex == nullptr || vertex->index < 0) {
        continue;
      }
      auto& levels = node_neighbors[static_cast<std::size_t>(vertex->index)];
      if (levels.size() <= static_cast<std::size_t>(level)) {
        levels.resize(static_cast<std::size_t>(level + 1));
      }
      levels[static_cast<std::size_t>(level)].push_back(compact_index);
    }
  }
  for (auto& levels : node_neighbors) {
    for (auto& elements_at_level : levels) {
      hnl_sort_unique(elements_at_level);
    }
  }
  return node_neighbors;
}

// Build geometry first, then construct each leaf's HNL source cover and
// classify that cover with the MAC.  The aggregate validator checks that the
// resulting near/far lists represent every source leaf exactly once.
double parallel_fill_neighbors(
    const std::vector<HElement<DIM>*>& root_element_list,
    std::vector<hElementCache>& hElement_cache,
    std::map<HElement<DIM>*, int>& map_HElement_to_int,
    const std::vector<std::vector<std::set<HElement<DIM>* > > >&,
    double theta,
    bool construct_interactions,
    double* mapping_geometry_seconds,
    double* hierarchy_validation_time_seconds,
    int* omp_threads_used,
    std::map<std::string, double>* interaction_diagnostics)
{
    const double preprocessing_start = omp_get_wtime();
    std::vector<int> element_levels;
    build_element_mapping(root_element_list, map_HElement_to_int, hElement_cache,
                          element_levels);
    const std::vector<int> read_only_element_levels = std::move(element_levels);
    
    std::vector<HElement<DIM>*> all_elements(hElement_cache.size());
    for(auto& pair : map_HElement_to_int) {
        all_elements[pair.second] = pair.first;
    }
    const std::vector<HElement<DIM>*> read_only_elements =
        std::move(all_elements);
    const auto& read_only_map = map_HElement_to_int;

    for (const HElement<DIM>* element : read_only_elements) {
      if (element == nullptr || element->h_element == nullptr ||
          element->h_element->vertex.size() != 4) {
        throw std::runtime_error("Invalid tetrahedral hierarchy element");
      }
    }
    
    std::cout << "Starting parallel processing with at most "
              << omp_get_max_threads()
              << " OpenMP threads for " << read_only_elements.size()
              << " elements..." << std::endl;
#ifdef TREECODE_ROOT_BASED_LISTS
    std::cout << "Interaction search: ROOT -> recursive MAC" << std::endl;
#else
    std::cout << "Interaction search: HNL frontier_vertices -> "
                 "sibling completion -> covered_source_regions -> MAC"
              << std::endl;
#endif

    #pragma omp parallel default(none) \
        shared(read_only_elements, read_only_element_levels, hElement_cache, \
               omp_threads_used)
    {
      #pragma omp single
      {
        if (omp_threads_used != nullptr) {
          *omp_threads_used = omp_get_num_threads();
        }
      }
      #pragma omp for schedule(static)
      for(std::size_t i = 0; i < read_only_elements.size(); ++i) {
          initialize_hierarchy_geometry(
              read_only_elements[i], hElement_cache[i],
              read_only_element_levels[i]);
      }
    }
    if (omp_threads_used != nullptr) {
      std::cout << "OpenMP threads actually used: "
                << *omp_threads_used << std::endl;
    }
    const double geometry_seconds = omp_get_wtime() - preprocessing_start;

    const double hierarchy_validation_start = omp_get_wtime();
    const HierarchySummary hierarchy = prepare_hierarchy_aggregates(
        read_only_elements, root_element_list, hElement_cache, read_only_map);
    const double hierarchy_validation_seconds =
        omp_get_wtime() - hierarchy_validation_start;
    if (mapping_geometry_seconds != nullptr) {
      *mapping_geometry_seconds = geometry_seconds;
    }
    if (hierarchy_validation_time_seconds != nullptr) {
      *hierarchy_validation_time_seconds = hierarchy_validation_seconds;
    }

    if (!construct_interactions) {
      std::cout
          << "Neighbor preprocessing timing: mapping/geometry="
          << geometry_seconds
          << " seconds, hierarchy validation="
          << hierarchy_validation_seconds
          << " seconds; MAC lists and coverage validation deferred to CUDA\n"
          << "Parallel hierarchy preprocessing completed!" << std::endl;
      return 0.0;
    }

    std::vector<int> root_indices;
    root_indices.reserve(root_element_list.size());
    for (HElement<DIM>* root : root_element_list) {
      root_indices.push_back(read_only_map.at(root));
    }
    const auto& read_only_roots = root_indices;
    const auto& read_only_cache = hElement_cache;
    std::vector<InteractionListStats> interaction_stats(
        read_only_elements.size());
#ifndef TREECODE_ROOT_BASED_LISTS
    const double hnl_index_start = omp_get_wtime();
    const VertexLevelIndex node_neighbors = build_vertex_level_index_ids(
        read_only_elements, read_only_element_levels, read_only_map);
    const double hnl_index_seconds = omp_get_wtime() - hnl_index_start;
    const auto& read_only_node_neighbors = node_neighbors;
    std::vector<HNLState> hnl_states(
        static_cast<std::size_t>(std::max(1, omp_get_max_threads())));
    for (HNLState& state : hnl_states) {
      state.vertex_marks.assign(read_only_node_neighbors.size(), 0U);
    }
#else
    const double hnl_index_seconds = 0.0;
#endif
    const double interaction_start = omp_get_wtime();
#ifdef TREECODE_ROOT_BASED_LISTS
    #pragma omp parallel for schedule(dynamic, 16) default(none) \
        shared(read_only_elements, read_only_roots, read_only_cache, \
               hElement_cache, interaction_stats) firstprivate(theta)
    for(std::size_t i = 0; i < read_only_elements.size(); ++i) {
      if (read_only_elements[i]->value == 0) {
        build_root_mac_interaction_lists(
            static_cast<int>(i), read_only_roots, read_only_elements,
            read_only_cache, hElement_cache[i], theta, interaction_stats[i]);
      }
    }
#else
    #pragma omp parallel default(none) \
        shared(read_only_elements, read_only_roots, read_only_cache, \
               read_only_map, read_only_node_neighbors, hElement_cache, \
               interaction_stats, hnl_states, theta)
    {
      HNLState& state = hnl_states[static_cast<std::size_t>(omp_get_thread_num())];
      #pragma omp for schedule(dynamic, 16)
      for(std::size_t i = 0; i < read_only_elements.size(); ++i) {
        if (read_only_elements[i]->value == 0) {
          build_hnl_mac_interaction_lists(
              static_cast<int>(i), read_only_roots, read_only_elements,
              read_only_cache, read_only_node_neighbors, hElement_cache[i],
              theta, state, interaction_stats[i]);
        }
      }
    }
#endif
    const double interaction_seconds = omp_get_wtime() - interaction_start;

    const double validation_start = omp_get_wtime();
    validate_interaction_stats(
        read_only_elements, interaction_stats, hierarchy, theta,
        interaction_diagnostics);
    if (interaction_diagnostics != nullptr) {
      save_interaction_fingerprints(
          read_only_elements, hElement_cache, *interaction_diagnostics);
      const int maximum_level = read_only_element_levels.empty()
          ? 0
          : *std::max_element(
                read_only_element_levels.begin(), read_only_element_levels.end());
      (*interaction_diagnostics)["hierarchy_depth"] =
          static_cast<double>(maximum_level);
      (*interaction_diagnostics)["hierarchy_nodes"] =
          static_cast<double>(read_only_elements.size());
      (*interaction_diagnostics)["root_regions"] =
          static_cast<double>(read_only_roots.size());
      (*interaction_diagnostics)["hnl_index_s"] = hnl_index_seconds;
      (*interaction_diagnostics)["list_total_s"] =
          hnl_index_seconds + interaction_seconds;
#ifdef TREECODE_ROOT_BASED_LISTS
      (*interaction_diagnostics)["search_method_root"] = 1.0;
#else
      (*interaction_diagnostics)["search_method_root"] = 0.0;
#endif
    }
    const double validation_seconds = omp_get_wtime() - validation_start;
    
    std::cout << "Neighbor preprocessing timing: mapping/geometry="
              << geometry_seconds
              << " seconds, hierarchy validation="
              << hierarchy_validation_seconds
              << " seconds, HNL index=" << hnl_index_seconds
              << " seconds, MAC construction=" << interaction_seconds
              << " seconds, total list preparation="
              << hnl_index_seconds + interaction_seconds
              << " seconds, aggregate validation=" << validation_seconds
              << " seconds\n"
              << "Parallel processing completed!" << std::endl;
    return interaction_seconds;
}
void write_current_hElement_neighbor(std::fstream& file,
            std::fstream& mpFile,
            HElement<DIM>* the_hElement,
            std::map<HElement<DIM>*, int>& map_HElement_to_int,
            std::vector<hElementCache>& hElement_cache,
            RegularMesh<DIM>& r_mesh
            )
{
  std::map<HElement<DIM>* , int>::iterator the_map_it = map_HElement_to_int.find(the_hElement);
  const int& idx_hElementCache = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[idx_hElementCache];

  std::vector<HElement<DIM>*>::iterator
    the_set_it = the_hElementCache.neighbor_elements.begin(),
    end_set_it = the_hElementCache.neighbor_elements.end();      
  for(;the_set_it != end_set_it;++ the_set_it){
    HGeometry<DIM>* current_geometry = (*the_set_it)->h_element;
    const int& n_vtx = current_geometry->vertex.size();
    mpFile << "draw ";
    for(int j = 0;j < n_vtx;++ j){
      if(j != n_vtx - 1){
  mpFile << "(" << 10. *r_mesh.point(current_geometry->vertex[j]->index)[0] << "cm, " 
        << 10.* r_mesh.point(current_geometry->vertex[j]->index)[1] << "cm)--";
      }
      else{
  mpFile << "(" << 10.* r_mesh.point(current_geometry->vertex[j]->index)[0] << "cm, " 
        << 10.* r_mesh.point(current_geometry->vertex[j]->index)[1] << "cm)--cycle;" << std::endl;
      }
      
      for(int k = 0;k < DIM;++ k){
  file << r_mesh.point(current_geometry->vertex[j]->index)[k] << " ";
      }
      file << std::endl;
    }
  }

  if(the_hElement->parent == NULL) return;

  write_current_hElement_neighbor(file, mpFile, the_hElement->parent, map_HElement_to_int, hElement_cache, r_mesh);
}



double update_hElementCache(IrregularMesh<DIM>& ir_mesh,
        std::vector<hElementCache>& hElement_cache,
        std::map<HElement<DIM>*, int>& map_HElement_to_int,
        double theta,
        bool construct_interactions,
        double* mapping_geometry_seconds,
        double* hierarchy_validation_seconds,
        int* omp_threads_used,
        std::map<std::string, double>* interaction_diagnostics = nullptr)
{
  RegularMesh<DIM>& r_mesh = ir_mesh.regularMesh();
  const int n_node = r_mesh.n_geometry(0);
  std::cout << "There are total " << n_node << " nodes" << std::endl;

  std::vector<HElement<DIM> *> root_element_list;
  IrregularMesh<DIM>::RootIterator
    the_root = ir_mesh.beginRootElement(),
    end_root = ir_mesh.endRootElement();
  for(;the_root != end_root;++ the_root){
    root_element_list.push_back(&(*the_root));
  }

  std::cout << "OpenMP maximum threads: "
            << omp_get_max_threads() << std::endl;
  const std::vector<std::vector<std::set<HElement<DIM>* > > > unused_neighbors;
  const double neighbor_fill_seconds = parallel_fill_neighbors(
      root_element_list,
      hElement_cache,
      map_HElement_to_int,
      unused_neighbors,
      theta,
      construct_interactions,
      mapping_geometry_seconds,
      hierarchy_validation_seconds,
      omp_threads_used,
      interaction_diagnostics
  );

  std::cout << "There are total " << hElement_cache.size()
            << " hElements" << std::endl;
  if (construct_interactions) {
    std::cout << "Filling neighbor_elements uses: "
              << neighbor_fill_seconds << " seconds" << std::endl;
  } else {
    std::cout << "Filling neighbor_elements uses: CUDA deferred\n";
  }
  return neighbor_fill_seconds;
}


void updateElementCache(std::vector<ElementCache > & element_cache,
      const FEMSpace<double, DIM>& fem_space,
      int quadrature_accuracy)
{
  element_cache.clear();
  
  const int n_element = fem_space.n_element();
  element_cache.resize(n_element);

  const Mesh<DIM>&  r_mesh = fem_space.mesh();
  const std::vector<int> element_indices = [&fem_space, n_element]() {
    std::vector<int> indices(static_cast<std::size_t>(n_element));
    std::vector<bool> index_seen(static_cast<std::size_t>(n_element), false);
    for (int i = 0; i < n_element; ++i) {
      const int index = fem_space.element(i).index();
      if (index < 0 || index >= n_element || index_seen[index]) {
        throw std::runtime_error(
            "Element indices must be a unique permutation for parallel "
            "cache updates.");
      }
      indices[i] = index;
      index_seen[index] = true;
    }
    return indices;
  }();

  #pragma omp parallel for default(none) \
      shared(element_cache, element_indices, fem_space, r_mesh) \
      firstprivate(n_element, quadrature_accuracy)
  for(int i = 0;i < n_element;++ i){
    const Element<double, DIM>& the_ele = fem_space.element(i);
    const int ele_idx = element_indices[i];

    ElementCache& the_ec = element_cache[ele_idx];
    const GeometryBM& the_geo = the_ele.geometry();

    const int& n_vtx = the_geo.n_vertex();
    std::vector<AFEPack::Point<DIM> > vertex(n_vtx);
    for(int j = 0;j < n_vtx;++ j){
      vertex[j] = r_mesh.point()[the_geo.vertex()[j]]; //
    }
    the_ec.physical_vertices = vertex;
    
    int& n_quadrature_point = the_ec.n_quadrature_point;
    double& volume = the_ec.volume;
    AFEPack::Point<DIM>& bc = the_ec.bc;
    std::vector<double>& Jxw = the_ec.Jxw;
    std::vector<AFEPack::Point<DIM> >& q_point = the_ec.q_point;
    std::vector<double>& source_charge = the_ec.source_charge;

    const std::vector<treecode_quadrature::QuadraturePoint> rule =
        treecode_quadrature::build_physical_rule(
            vertex, quadrature_accuracy);
    n_quadrature_point = static_cast<int>(rule.size());
    q_point.resize(rule.size());
    Jxw.resize(rule.size());
    source_charge.resize(rule.size());
    volume = 0.0;
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      bc[coordinate] = 0.0;
    }
    for (std::size_t q = 0; q < rule.size(); ++q) {
      q_point[q] = rule[q].point;
      Jxw[q] = rule[q].weight;
      source_charge[q] = Jxw[q] * f(q_point[q]);
      volume += Jxw[q];
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        bc[coordinate] += Jxw[q] * q_point[q][coordinate];
      }
    }
    if (!(volume > 0.0)) {
      throw std::runtime_error(
          "Custom local quadrature produced non-positive volume");
    }
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      bc[coordinate] /= volume;
    }
    
  }

  for (const ElementCache& cache : element_cache) {
    if (!(cache.volume > 0.0) || !std::isfinite(cache.volume)) {
      throw std::runtime_error("Invalid element volume from Gaussian quadrature");
    }
    for (double weight : cache.Jxw) {
      if (!std::isfinite(weight)) {
        throw std::runtime_error("Invalid Gaussian quadrature weight");
      }
    }
    if (cache.source_charge.size() != cache.Jxw.size()
        || std::any_of(
            cache.source_charge.begin(),
            cache.source_charge.end(),
            [](double charge) { return !std::isfinite(charge); })) {
      throw std::runtime_error("Invalid cached quadrature source charge");
    }
  }

  std::size_t tetrahedron_count = 0;
  std::size_t twin_tetrahedron_count = 0;
  std::size_t four_tetrahedron_count = 0;
  std::size_t total_quadrature_points = 0;
  for (int i = 0; i < n_element; ++i) {
    const int vertex_count = fem_space.element(i).geometry().n_vertex();
    if (vertex_count == 4) {
      ++tetrahedron_count;
    } else if (vertex_count == 5) {
      ++twin_tetrahedron_count;
    } else if (vertex_count == 7) {
      ++four_tetrahedron_count;
    } else {
      throw std::runtime_error(
          "Unexpected element type after custom quadrature construction");
    }
    total_quadrature_points +=
        static_cast<std::size_t>(
            element_cache[static_cast<std::size_t>(
                fem_space.element(i).index())].n_quadrature_point);
  }
  std::cout
      << "Code-owned local quadrature: algebraic accuracy="
      << quadrature_accuracy
      << ", elements(tetra/twin/four)="
      << tetrahedron_count << "/"
      << twin_tetrahedron_count << "/"
      << four_tetrahedron_count
      << ", total points=" << total_quadrature_points << '\n';
}


// --- helper: compute signed determinant of 3x3 matrix [a b c] (columns)
double det3(const AFEPack::Point<DIM>& a,
            const AFEPack::Point<DIM>& b,
            const AFEPack::Point<DIM>& c) {
    // treat points as vectors (x,y,z). Using column form
    double A00 = a[0], A10 = a[1], A20 = a[2];
    double A01 = b[0], A11 = b[1], A21 = b[2];
    double A02 = c[0], A12 = c[1], A22 = c[2];
    return A00*(A11*A22 - A12*A21)
         - A01*(A10*A22 - A12*A20)
         + A02*(A10*A21 - A11*A20);
}

void collect_multipole_from_leaf_hElement(
    std::vector<double>& moments,
    int expansion_order,
    const AFEPack::Point<DIM>& ref_point,
    const std::vector<ElementCache>& element_cache,
    HElement<DIM>* the_hElement)
{
  if (the_hElement->value == 0) {
    const ElementCache& element =
        element_cache[static_cast<std::size_t>(the_hElement->index)];
    for (int q = 0; q < element.n_quadrature_point; ++q) {
      const AFEPack::Point<DIM>& point = element.q_point[q];
      treecode_multipole::accumulate_raw_moments(
          moments,
          expansion_order,
          point[0] - ref_point[0],
          point[1] - ref_point[1],
          point[2] - ref_point[2],
          element.source_charge[static_cast<std::size_t>(q)]);
    }
    return;
  }

  for (HElement<DIM>* child : the_hElement->child) {
    if (child != nullptr) {
      collect_multipole_from_leaf_hElement(
          moments,
          expansion_order,
          ref_point,
          element_cache,
          child);
    }
  }
}


void update_multipoleInfo_hElementCache(HElement<DIM>* the_hElement,
          std::vector<hElementCache>& hElement_cache,
          const std::map<HElement<DIM>*, int>& map_HElement_to_int,
          const std::vector<ElementCache>& element_cache,
          int expansion_order,
          bool recurse_children = true)
{
  /// 1. prepare the hElementCache
  auto the_map_it = map_HElement_to_int.find(the_hElement);
  const int& idx_hElementCache = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[idx_hElementCache];

  /// 2. update hElementCache.ref_point
  AFEPack::Point<DIM>& ref_point = the_hElementCache.ref_point;
#ifndef TREECODE_DYNAMIC_MESH_MODES
  for(int j = 0;j < DIM;++ j){
    ref_point[j] = 0.;
  }
  HGeometry<DIM>* the_hGeometry = the_hElement->h_element;
  const int& n_vtx = the_hGeometry->vertex.size();
  for(int i = 0;i < n_vtx;++ i){
    const AFEPack::Point<DIM>& the_vtx = *the_hGeometry->vertex[i];
    for(int j = 0;j < DIM;++ j){
      ref_point[j] += the_vtx[j];
    }
  }
  for(int j = 0;j < DIM;++ j){
    ref_point[j] /= n_vtx;
  }
#endif

  /// 3. tracing back to all leaf hElement, and collect the useful
  /// information from those leaf hElement.

  the_hElementCache.moments =
      treecode_multipole::zero_moments(expansion_order);
  collect_multipole_from_leaf_hElement(
      the_hElementCache.moments,
      expansion_order,
      ref_point,
      element_cache,
      the_hElement);
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if(!recurse_children || the_hElement->value == 0){
    return;
  }
  else{
    for(HElement<DIM>* child : the_hElement->child){
      if (child != nullptr) {
        update_multipoleInfo_hElementCache(child,
          hElement_cache,
          map_HElement_to_int,
          element_cache,
          expansion_order,
          true);
      }
    }
  }
}

bool pointInTetra(const AFEPack::Point<DIM>& p,
                  const AFEPack::Point<DIM>& A,
                  const AFEPack::Point<DIM>& B,
                  const AFEPack::Point<DIM>& C,
                  const AFEPack::Point<DIM>& D);

void calculate_far_field_multipole_contribution(
    double& approximation,
    const double* target_point,
    const hElementCache& target_cache,
    const std::vector<hElementCache>& hElement_cache,
    int expansion_order,
    double* coefficient_workspace)
{
    const int coefficient_count =
        treecode_multipole::coefficient_count(expansion_order);
    for (int source_index : target_cache.far_hierarchy_indices)
    {
        const hElementCache& nn_cache =
            hElement_cache[static_cast<std::size_t>(source_index)];

        const AFEPack::Point<DIM>& center =
            nn_cache.ref_point;

        //------------------------------------------------------------------
        // R = target - expansion_center
        //------------------------------------------------------------------

        double Rx = target_point[0] - center[0];
        double Ry = target_point[1] - center[1];
        double Rz = target_point[2] - center[2];

        double R2 =
            Rx * Rx +
            Ry * Ry +
            Rz * Rz;

        if (R2 < 1e-30)
            continue;

        if (!treecode_multipole::build_taylor_coefficients(
                Rx, Ry, Rz, expansion_order, coefficient_workspace)) {
          continue;
        }
        for (int coefficient = 0;
             coefficient < coefficient_count;
             ++coefficient) {
          approximation +=
              nn_cache.moments[static_cast<std::size_t>(coefficient)]
              * coefficient_workspace[static_cast<std::size_t>(coefficient)];
        }
    }
}

double singular_tetrahedron_integral(
    const AFEPack::Point<DIM>& target,
    const std::array<AFEPack::Point<DIM>, 4>& vertices,
    int algebraic_accuracy,
    int source_component = -1)
{
  const auto& rule =
      treecode_quadrature::singular_duffy_axis_rule(
          algebraic_accuracy);
  const double target_density =
      selected_source_value(target, source_component);
  const double analytic_kernel =
      treecode_quadrature::
          constant_density_tetrahedron_coulomb_integral(
              target, vertices);
  const bool use_analytic_subtraction =
      std::isfinite(target_density) && std::isfinite(analytic_kernel);

  // Connecting an interior target to each face partitions the tetrahedron.
  static constexpr int faces[4][3] = {
      {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

  // The constant-density singular part is exact.  Duffy quadrature is used
  // only for (rho(y)-rho(x))/|x-y|, which is bounded at the target.  For a
  // degenerate configuration, retain the original full Duffy evaluation.
  double integral = use_analytic_subtraction
      ? target_density * analytic_kernel : 0.0;
  for (const auto& face : faces) {
    const AFEPack::Point<DIM>& a =
        vertices[static_cast<std::size_t>(face[0])];
    const AFEPack::Point<DIM>& b =
        vertices[static_cast<std::size_t>(face[1])];
    const AFEPack::Point<DIM>& c =
        vertices[static_cast<std::size_t>(face[2])];
    const AFEPack::Point<DIM> ea = a - target;
    const AFEPack::Point<DIM> eb = b - target;
    const AFEPack::Point<DIM> ec = c - target;
    const double determinant = std::abs(det3(ea, eb, ec));
    if (determinant <= std::numeric_limits<double>::epsilon()) {
      continue;
    }

    // For fixed (face,r,s), the face point, radial direction and distance are
    // independent of t.  This is the same geometry reuse used by CUDA.
    for (const auto& s_entry : rule) {
      const double s = s_entry.first;
      for (const auto& r_entry : rule) {
        const double r = r_entry.first;
        AFEPack::Point<DIM> face_point = a;
        face_point *= (1.0 - s);
        AFEPack::Point<DIM> contribution = b;
        contribution *= s * (1.0 - r);
        face_point += contribution;
        contribution = c;
        contribution *= s * r;
        face_point += contribution;
        const AFEPack::Point<DIM> radial = face_point - target;
        const double radial_length = radial.length();
        if (radial_length <= std::numeric_limits<double>::min()) {
          continue;
        }
        const double reusable_weight =
            s_entry.second * r_entry.second
            * determinant * s / radial_length;
        for (const auto& t_entry : rule) {
          const double t = t_entry.first;
          AFEPack::Point<DIM> radial_step = radial;
          radial_step *= t;
          const AFEPack::Point<DIM> quadrature_point =
              target + radial_step;
          double density = selected_source_value(
              quadrature_point, source_component);
          if (use_analytic_subtraction) {
            density -= target_density;
          }
          // det*t^2*s / (t*|radial|) is the transformed 1/r Jacobian.
          integral += t_entry.second * t * reusable_weight
              * density;
        }
      }
    }
  }
  return integral;
}

double regular_quadrature_range(
    const AFEPack::Point<DIM>& target,
    const ElementCache& source,
    std::size_t begin,
    std::size_t end)
{
  double integral = 0.0;
  for (std::size_t q = begin; q < end; ++q) {
    const double distance = (source.q_point[q] - target).length();
    if (!(distance > 1.0e-14)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    integral += source.source_charge[q] / distance;
  }
  return integral;
}

double singular_element_integral(
    const AFEPack::Point<DIM>& target,
    const ElementCache& source,
    int algebraic_accuracy,
    int source_component = -1)
{
  const auto decomposition = treecode_quadrature::sub_tetrahedron_indices(
      source.physical_vertices.size());
  const std::size_t points_per_subtetrahedron =
      treecode_quadrature::quadrature_point_count(4, algebraic_accuracy);
  if (source.q_point.size()
      != decomposition.size() * points_per_subtetrahedron
      || source.source_charge.size() != source.q_point.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double integral = 0.0;
  bool target_found = false;
  for (std::size_t subcell = 0; subcell < decomposition.size(); ++subcell) {
    std::array<AFEPack::Point<DIM>, 4> vertices;
    for (int vertex = 0; vertex < 4; ++vertex) {
      vertices[static_cast<std::size_t>(vertex)] =
          source.physical_vertices[static_cast<std::size_t>(
              decomposition[subcell][static_cast<std::size_t>(vertex)])];
    }
    if (treecode_quadrature::point_in_tetrahedron(target, vertices)) {
      target_found = true;
      integral += singular_tetrahedron_integral(
          target, vertices, algebraic_accuracy, source_component);
    } else {
      const std::size_t begin = subcell * points_per_subtetrahedron;
      integral += regular_quadrature_range(
          target, source, begin, begin + points_per_subtetrahedron);
    }
  }
  return target_found
      ? integral : std::numeric_limits<double>::quiet_NaN();
}


double multipole_approximation(const double * target_point,
            HElement<DIM>* the_hElement,
            const hElementCache& target_hierarchy_cache,
            const std::vector<hElementCache>& hElement_cache,
            const std::vector<ElementCache>& element_cache,
            int expansion_order,
            int quadrature_accuracy,
            double* coefficient_workspace,
            int source_component = -1)
{
  double approximation = 0.;
  for (int ele_idx : target_hierarchy_cache.neighbor_leaf_indices) {
      const ElementCache& the_ec =
          element_cache[static_cast<std::size_t>(ele_idx)];
      HElement<DIM>* source_element = the_ec.the_hElement;
      const std::vector<AFEPack::Point<DIM> >& q_point = the_ec.q_point;
      const std::vector<double>& source_charge = the_ec.source_charge;
      const int& n_quadrature_point = the_ec.n_quadrature_point;

      if (source_element == the_hElement) {
          AFEPack::Point<DIM> target;
          for (int d = 0; d < DIM; ++d) {
            target[d] = target_point[d];
          }
          approximation +=
              singular_element_integral(
                  target, the_ec, quadrature_accuracy,
                  source_component);
          continue;
      }

      double near_contribution = 0.0;
      for(int l = 0;l < n_quadrature_point;++ l){
          const double dx = q_point[l][0] - target_point[0];
          const double dy = q_point[l][1] - target_point[1];
          const double dz = q_point[l][2] - target_point[2];
          const double distance_squared = dx * dx + dy * dy + dz * dz;

          if (distance_squared <= 1.0e-28) {
              AFEPack::Point<DIM> target;
              for (int d = 0; d < DIM; ++d) {
                target[d] = target_point[d];
              }
              near_contribution =
                  singular_element_integral(
                      target, the_ec, quadrature_accuracy,
                      source_component);
              break;
          }

          near_contribution +=
              source_charge[static_cast<std::size_t>(l)]
              / std::sqrt(distance_squared);
         }
      approximation += near_contribution;
    }
    calculate_far_field_multipole_contribution(
        approximation,
        target_point,
        target_hierarchy_cache,
        hElement_cache,
        expansion_order,
        coefficient_workspace);
  return approximation;
}

#ifdef TREECODE_REUSE_EXPERIMENT
int run_reuse_sequence_experiment(
    const ProgramOptions& options,
    double initial_list_construction_seconds,
    IrregularMesh<DIM>& ir_mesh,
    std::vector<ElementCache>& element_cache,
    std::vector<hElementCache>& hierarchy_cache,
    const std::map<HElement<DIM>*, int>& element_to_hierarchy_index)
{
  const std::size_t element_count = element_cache.size();
  const std::size_t hierarchy_count = hierarchy_cache.size();
  const int expansion_order = options.expansion_order;
  const int quadrature_accuracy = options.quadrature_accuracy;

  std::vector<HElement<DIM>*> hierarchy_elements(
      hierarchy_count, nullptr);
  for (const auto& entry : element_to_hierarchy_index) {
    if (entry.second < 0
        || static_cast<std::size_t>(entry.second) >= hierarchy_count) {
      throw std::runtime_error(
          "Invalid hierarchy index in reuse experiment");
    }
    hierarchy_elements[static_cast<std::size_t>(entry.second)] = entry.first;
  }
  if (std::find(
          hierarchy_elements.begin(), hierarchy_elements.end(), nullptr)
      != hierarchy_elements.end()) {
    throw std::runtime_error(
        "Incomplete hierarchy mapping in reuse experiment");
  }

  unsigned long long near_entries = 0;
  unsigned long long far_entries = 0;
  for (const ElementCache& element : element_cache) {
    if (element.the_hElement == nullptr || element.hierarchy_index < 0) {
      throw std::runtime_error(
          "Leaf cache is incomplete in reuse experiment");
    }
    const hElementCache& target = hierarchy_cache[
        static_cast<std::size_t>(element.hierarchy_index)];
    near_entries += target.neighbor_leaf_indices.size();
    far_entries += target.far_hierarchy_indices.size();
  }

  const std::string& output_filename = options.sequence_output_file;
  std::ofstream output(output_filename, std::ios::trunc);
  if (!output.is_open()) {
    std::cerr << "Error: cannot write " << output_filename << '\n';
    return 1;
  }
  output
      << "step\tperturbation_amplitude\trelative_density_change"
      << "\tn_elements\tp\tP\ttheta"
      << "\tlist_rebuild_time_s"
      << "\tdensity_update_time_s\tmoment_update_time_s"
      << "\tpotential_evaluation_time_s\tstep_time_s"
      << "\tL2_error\trelative_L2_error\n";
  output << std::setprecision(15);

  std::cout
      << "\nPreliminary multirate/reuse experiment\n"
      << "  Manufactured sequence: V_k = V_0 + beta_k V_1, "
         "beta_k = beta_0 * decay^k\n"
      << "  V_1 is a shifted anisotropic Gaussian centered at "
         "(0.70,-0.35,0.20)\n"
      << "  Reused data: adaptive mesh, quadrature geometry, hierarchy, "
         "and MAC near/far lists\n"
      << "  Updated data: source charges, Cartesian moments, and potential\n"
      << "  Steps=" << options.sequence_steps
      << ", beta_0=" << options.sequence_initial_amplitude
      << ", decay=" << options.sequence_decay
      << ", leaves=" << element_count
      << ", near/far entries=" << near_entries << '/' << far_entries
      << '\n';

  double sequence_update_seconds = 0.0;
  double measured_rebuild_seconds = 0.0;
  double previous_amplitude = options.sequence_initial_amplitude;
  for (int step = 0; step < options.sequence_steps; ++step) {
    const double amplitude =
        options.sequence_initial_amplitude
        * std::pow(options.sequence_decay, step);
    reuse_sequence_amplitude = amplitude;

    // Measure the non-reuse baseline on exactly the same mesh.  The rebuilt
    // lists are discarded; density, moments, and potential evaluation below
    // use the original cached lists in both timing models.
    std::vector<hElementCache> rebuilt_hierarchy_cache;
    std::map<HElement<DIM>*, int> rebuilt_element_to_hierarchy_index;
    double rebuilt_mapping_seconds = 0.0;
    double rebuilt_validation_seconds = 0.0;
    int rebuilt_threads = 1;
    const double list_rebuild_start = omp_get_wtime();
    update_hElementCache(
        ir_mesh,
        rebuilt_hierarchy_cache,
        rebuilt_element_to_hierarchy_index,
        options.mac_theta,
        true,
        &rebuilt_mapping_seconds,
        &rebuilt_validation_seconds,
        &rebuilt_threads);
    const double list_rebuild_seconds =
        omp_get_wtime() - list_rebuild_start;
    measured_rebuild_seconds += list_rebuild_seconds;

    unsigned long long rebuilt_near_entries = 0;
    unsigned long long rebuilt_far_entries = 0;
    for (const ElementCache& element : element_cache) {
      const auto rebuilt_iterator =
          rebuilt_element_to_hierarchy_index.find(element.the_hElement);
      if (rebuilt_iterator == rebuilt_element_to_hierarchy_index.end()) {
        throw std::runtime_error(
            "Rebuilt interaction hierarchy is missing an active leaf");
      }
      const hElementCache& rebuilt_target = rebuilt_hierarchy_cache[
          static_cast<std::size_t>(rebuilt_iterator->second)];
      const hElementCache& cached_target = hierarchy_cache[
          static_cast<std::size_t>(element.hierarchy_index)];
      if (rebuilt_target.neighbor_leaf_indices !=
              cached_target.neighbor_leaf_indices ||
          rebuilt_target.far_hierarchy_indices !=
              cached_target.far_hierarchy_indices) {
        throw std::runtime_error(
            "Repeated HNL--MAC construction changed an interaction list");
      }
      rebuilt_near_entries += rebuilt_target.neighbor_leaf_indices.size();
      rebuilt_far_entries += rebuilt_target.far_hierarchy_indices.size();
    }
    if (rebuilt_near_entries != near_entries ||
        rebuilt_far_entries != far_entries) {
      throw std::runtime_error(
          "Repeated HNL--MAC construction changed the interaction counts");
    }

    double density_change_squared = 0.0;
    double density_norm_squared = 0.0;
    const double density_start = omp_get_wtime();
    #pragma omp parallel for schedule(static) default(none) \
        shared(element_cache) firstprivate(element_count, amplitude, \
                                            previous_amplitude) \
        reduction(+:density_change_squared,density_norm_squared)
    for (std::size_t element_index = 0;
         element_index < element_count;
         ++element_index) {
      ElementCache& element = element_cache[element_index];
      for (std::size_t q = 0; q < element.q_point.size(); ++q) {
        const AFEPack::Point<DIM>& point = element.q_point[q];
        const double base_source = treecode_test_problem::source_value(
            point[0], point[1], point[2]);
        const double perturbation_source =
            reuse_perturbation_source_value(point);
        const double source = base_source + amplitude * perturbation_source;
        const double change =
            (amplitude - previous_amplitude) * perturbation_source;
        const double weight = element.Jxw[q];
        element.source_charge[q] = weight * source;
        density_change_squared += weight * change * change;
        density_norm_squared += weight * source * source;
      }
    }
    const double density_seconds = omp_get_wtime() - density_start;
    const double relative_density_change = step == 0
        ? 0.0
        : std::sqrt(density_change_squared)
            / std::max(std::sqrt(density_norm_squared),
                       std::numeric_limits<double>::min());

    const double moment_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 16) default(none) \
        shared(hierarchy_elements, hierarchy_cache, \
               element_to_hierarchy_index, element_cache) \
        firstprivate(hierarchy_count, expansion_order)
    for (std::size_t hierarchy_index = 0;
         hierarchy_index < hierarchy_count;
         ++hierarchy_index) {
      update_multipoleInfo_hElementCache(
          hierarchy_elements[hierarchy_index],
          hierarchy_cache,
          element_to_hierarchy_index,
          element_cache,
          expansion_order,
          false);
    }
    const double moment_seconds = omp_get_wtime() - moment_start;

    double error_squared = 0.0;
    double exact_norm_squared = 0.0;
    std::size_t evaluation_count = 0;
    std::size_t nonfinite_count = 0;
    const double evaluation_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 16) default(none) \
        shared(element_cache, hierarchy_cache) \
        firstprivate(element_count, expansion_order, quadrature_accuracy) \
        reduction(+:error_squared,exact_norm_squared,evaluation_count, \
                    nonfinite_count)
    for (std::size_t element_index = 0;
         element_index < element_count;
         ++element_index) {
      const ElementCache& target_element = element_cache[element_index];
      const hElementCache& target_hierarchy = hierarchy_cache[
          static_cast<std::size_t>(target_element.hierarchy_index)];
      std::array<double,
                 treecode_multipole::maximum_coefficient_count>
          coefficient_workspace{};
      const AFEPack::Point<DIM>& target = target_hierarchy.ref_point;
      const double approximation = multipole_approximation(
          target,
          target_element.the_hElement,
          target_hierarchy,
          hierarchy_cache,
          element_cache,
          expansion_order,
          quadrature_accuracy,
          coefficient_workspace.data());
      const double exact = u(target);
      const double weight = target_hierarchy.volume;
      if (!std::isfinite(approximation)
          || !std::isfinite(exact)
          || !std::isfinite(weight)) {
        ++nonfinite_count;
        continue;
      }
      const double difference = approximation - exact;
      error_squared += weight * difference * difference;
      exact_norm_squared += weight * exact * exact;
      ++evaluation_count;
    }
    const double evaluation_seconds = omp_get_wtime() - evaluation_start;
    if (nonfinite_count != 0 || evaluation_count != element_count) {
      std::cerr << "Error: invalid evaluation in reuse step " << step
                << " (evaluated=" << evaluation_count
                << ", invalid=" << nonfinite_count << ")\n";
      return 1;
    }

    const double error = std::sqrt(error_squared);
    const double relative_error = error
        / std::max(std::sqrt(exact_norm_squared),
                   std::numeric_limits<double>::min());
    const double step_seconds =
        density_seconds + moment_seconds + evaluation_seconds;
    sequence_update_seconds += step_seconds;

    output << step << '\t'
           << amplitude << '\t'
           << relative_density_change << '\t'
           << element_count << '\t'
           << quadrature_accuracy << '\t'
           << expansion_order << '\t'
           << options.mac_theta << '\t'
           << list_rebuild_seconds << '\t'
           << density_seconds << '\t'
           << moment_seconds << '\t'
           << evaluation_seconds << '\t'
           << step_seconds << '\t'
           << error << '\t'
           << relative_error << '\n';
    std::cout << std::setprecision(9)
              << "  step " << step
              << ": beta=" << amplitude
              << ", r_rho=" << relative_density_change
              << ", T_list_rebuild=" << list_rebuild_seconds
              << ", T_update=" << density_seconds
              << " s, T_moment=" << moment_seconds
              << " s, T_eval=" << evaluation_seconds
              << " s, L2_error=" << error
              << ", relative_error=" << relative_error << '\n';
    previous_amplitude = amplitude;
  }

  const double reused_total_seconds =
      initial_list_construction_seconds + sequence_update_seconds;
  const double rebuild_every_step_seconds =
      measured_rebuild_seconds + sequence_update_seconds;
  const double saved_seconds =
      rebuild_every_step_seconds - reused_total_seconds;
  const double speedup = rebuild_every_step_seconds
      / std::max(reused_total_seconds,
                 std::numeric_limits<double>::min());
  output << "# initial_list_construction_time_s\t"
         << initial_list_construction_seconds << '\n'
         << "# measured_rebuild_sum_s\t" << measured_rebuild_seconds << '\n'
         << "# reused_total_time_s\t" << reused_total_seconds << '\n'
         << "# rebuild_every_step_measured_time_s\t"
         << rebuild_every_step_seconds << '\n'
         << "# saved_time_s\t" << saved_seconds << '\n'
         << "# estimated_speedup\t" << speedup << '\n';
  output.close();

  std::cout << std::setprecision(9)
            << "\nReuse experiment summary\n"
            << "  One-time interaction-list construction: "
            << initial_list_construction_seconds << " s\n"
            << "  Sum of measured per-step list rebuilds: "
            << measured_rebuild_seconds << " s\n"
            << "  Sum of density/moment/evaluation updates: "
            << sequence_update_seconds << " s\n"
            << "  Reuse strategy total: " << reused_total_seconds << " s\n"
            << "  Rebuild-every-step measured total: "
            << rebuild_every_step_seconds << " s\n"
            << "  Measured saved time: " << saved_seconds << " s\n"
            << "  Measured speedup: " << speedup << "x\n"
            << "  Results written to " << output_filename << '\n'
            << "  Note: this is a manufactured-density proof of concept, "
               "not a KS-SCF calculation.\n";
  return 0;
}
#endif

bool pointInTetra(const AFEPack::Point<DIM>& p,
                  const AFEPack::Point<DIM>& A,
                  const AFEPack::Point<DIM>& B,
                  const AFEPack::Point<DIM>& C,
                  const AFEPack::Point<DIM>& D)
{
    double V  = (tetrahedronVolume(A,B,C,D));
    double V1 = (tetrahedronVolume(p,B,C,D));
    double V2 = (tetrahedronVolume(A,p,C,D));
    double V3 = (tetrahedronVolume(A,B,p,D));
    double V4 = (tetrahedronVolume(A,B,C,p));
    double diff = fabs(V - (V1+V2+V3+V4));
    return diff <= 1.0e-12 + 1.0e-8 * V;
}


HElement<DIM>* findLeafContainingPoint(
                            const std::vector<HElement<DIM>*>& root_element_list,
                            const AFEPack::Point<DIM>& point)
{
    HElement<DIM>* current = nullptr;

    for(HElement<DIM>* root : root_element_list){
      HGeometry<DIM>* geom = root->h_element;
      const auto& vtx = geom->vertex;

      // Hierarchy vertices carry their coordinates directly.  Their `index`
      // is only valid for vertices present in the current RegularMesh, so it
      // must not be used while descending through inactive hierarchy cells.
      if (pointInTetra(point, *vtx[0], *vtx[1], *vtx[2], *vtx[3])) {
        current = root;
        break;
      }
    }

    if (current == nullptr) {
      return nullptr;
    }

    // HElement preallocates eight null child slots even for a leaf.  Use the
    // library's refinement predicate rather than vector::empty().
    while (current->value == 1) {
      HElement<DIM>* matching_child = nullptr;

      for (HElement<DIM>* child : current->child) {
        if (child == nullptr) continue;

        HGeometry<DIM>* geom = child->h_element;
        const auto& vtx = geom->vertex;

        if (pointInTetra(point, *vtx[0], *vtx[1], *vtx[2], *vtx[3])) {
          matching_child = child;
          break;
        }
      }

      if (matching_child == nullptr) {
        return nullptr;
      }

      current = matching_child;
    }

    return current;
}


void get_indicator(Indicator<DIM>& ind,
        RegularMesh<DIM>& mesh,
        FEMSpace<double,DIM>& fem_space,
        int quadrature_accuracy,
        bool use_gpu_backend)
  {
    ind.reinit(mesh);
    const int n_element = mesh.n_geometry(DIM);
#ifdef TREECODE_USE_CUDA
    if (use_gpu_backend) {
    std::vector<GpuQuadratureElement> gpu_elements(
        static_cast<std::size_t>(n_element));
    for (int i = 0; i < n_element; ++i) {
      Element<double, DIM>& element = fem_space.element(i);
      GeometryBM& geometry = element.geometry();
      const int vertex_count = geometry.n_vertex();
      if (vertex_count != 4 && vertex_count != 5 && vertex_count != 7) {
        throw std::runtime_error(
            "CUDA indicator supports only tetrahedron/twin/four elements");
      }
      GpuQuadratureElement& gpu_element =
          gpu_elements[static_cast<std::size_t>(i)];
      gpu_element.vertex_count = vertex_count;
      gpu_element.sample_count = static_cast<int>(
          treecode_quadrature::quadrature_point_count(
              static_cast<std::size_t>(vertex_count),
              quadrature_accuracy));
      for (int vertex = 0; vertex < vertex_count; ++vertex) {
        const AFEPack::Point<DIM>& point =
            mesh.point()[geometry.vertex()[vertex]];
        gpu_element.vertices[vertex] =
            {point[0], point[1], point[2]};
      }
    }
    double indicator_seconds = 0.0;
    const std::vector<double> indicator_values =
        evaluate_refinement_indicator_gpu(
            gpu_elements, quadrature_accuracy, &indicator_seconds);
    for (int i = 0; i < n_element; ++i) {
      const int geometry_index =
          fem_space.element(i).geometry().index();
      ind[geometry_index] =
          indicator_values[static_cast<std::size_t>(i)];
    }
    std::cout << "CUDA refinement indicator time: "
              << indicator_seconds << " seconds\n";
    return;
    }
#else
    if (use_gpu_backend) {
      throw std::runtime_error(
          "GPU refinement indicator requested in a CPU-only executable");
    }
#endif

    // The CPU backend keeps the complete refinement path on OpenMP/CPU.
    #pragma omp parallel for schedule(static) default(none) \
        shared(ind, mesh, fem_space) firstprivate(n_element, quadrature_accuracy)
    for(int i = 0;i < n_element;++ i){
      Element<double, DIM>& the_ele = fem_space.element(i);
      GeometryBM& the_geo = the_ele.geometry();
      const int& the_geo_idx = the_geo.index();

      std::vector<AFEPack::Point<DIM>> vertices(
          static_cast<std::size_t>(the_geo.n_vertex()));
      for (int vertex = 0; vertex < the_geo.n_vertex(); ++vertex) {
        vertices[static_cast<std::size_t>(vertex)] =
            mesh.point()[the_geo.vertex()[vertex]];
      }
      const auto rule = treecode_quadrature::build_physical_rule(
          vertices, quadrature_accuracy);
      for (const auto& entry : rule) {
        const double radius_squared =
            entry.point[0] * entry.point[0] +
            entry.point[1] * entry.point[1] +
            entry.point[2] * entry.point[2];
        ind[the_geo_idx] +=
            entry.weight
            / std::sqrt(std::max(radius_squared, 1.0e-30));
      }
    }
  }

#ifdef TREECODE_USE_CUDA
GpuTreecodeProblem flatten_treecode_for_gpu(
    std::vector<GpuPoint3> targets,
    std::vector<int> target_hierarchy_indices,
    std::vector<int> target_leaf_indices,
    const FEMSpace<double, DIM>& fem_space,
    const std::vector<hElementCache>& hierarchy_cache,
    const std::map<HElement<DIM>*, int>& element_to_hierarchy_index,
    int quadrature_accuracy,
    double mac_theta,
    bool targets_match_source_samples,
    bool generate_exact_targets)
{
  if ((!generate_exact_targets
       && (targets.size() != target_hierarchy_indices.size()
           || targets.size() != target_leaf_indices.size()))
      || (generate_exact_targets
          && (!targets.empty()
              || !target_hierarchy_indices.empty()
              || !target_leaf_indices.empty()))) {
    throw std::runtime_error("Inconsistent CUDA target arrays");
  }
  const auto checked_gpu_index = [](std::size_t value, const char* name) {
    if (value >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          std::string(name) + " exceeds CUDA's 32-bit index range");
    }
    return static_cast<int>(value);
  };

  GpuTreecodeProblem problem;
  problem.targets = std::move(targets);
  problem.target_hierarchy_indices =
      std::move(target_hierarchy_indices);
  problem.target_leaf_indices = std::move(target_leaf_indices);
  problem.mac_theta = mac_theta;
  problem.targets_match_source_samples = targets_match_source_samples;
  problem.generate_samples_on_gpu = true;
  problem.generate_exact_targets_on_gpu = generate_exact_targets;

  const int leaf_count = fem_space.n_element();
  problem.leaves.resize(static_cast<std::size_t>(leaf_count));
  problem.quadrature_elements.resize(
      static_cast<std::size_t>(leaf_count));
  problem.leaf_hierarchy_indices.resize(
      static_cast<std::size_t>(leaf_count), -1);
  std::vector<HElement<DIM>*> leaf_elements(
      static_cast<std::size_t>(leaf_count), nullptr);
  for (const auto& entry : element_to_hierarchy_index) {
    HElement<DIM>* element = entry.first;
    if (element->value == 0) {
      if (element->index < 0 || element->index >= leaf_count
          || leaf_elements[static_cast<std::size_t>(element->index)]
              != nullptr) {
        throw std::runtime_error(
            "CUDA leaf indices are not a unique zero-based permutation");
      }
      leaf_elements[static_cast<std::size_t>(element->index)] = element;
      problem.leaf_hierarchy_indices[
          static_cast<std::size_t>(element->index)] = entry.second;
    }
  }
  if (std::find(
          leaf_elements.begin(), leaf_elements.end(), nullptr)
      != leaf_elements.end()) {
    throw std::runtime_error("CUDA leaf hierarchy mapping contains a gap");
  }

  const Mesh<DIM>& mesh = fem_space.mesh();
  std::size_t tetrahedron_count = 0;
  std::size_t twin_tetrahedron_count = 0;
  std::size_t four_tetrahedron_count = 0;
  for (int element_position = 0;
       element_position < leaf_count;
       ++element_position) {
    const Element<double, DIM>& element =
        fem_space.element(element_position);
    const int leaf_index = element.index();
    if (leaf_index < 0 || leaf_index >= leaf_count) {
      throw std::runtime_error("CUDA FEM element index is invalid");
    }
    const GeometryBM& geometry = element.geometry();
    const int vertex_count = geometry.n_vertex();
    if (vertex_count == 4) {
      ++tetrahedron_count;
    } else if (vertex_count == 5) {
      ++twin_tetrahedron_count;
    } else if (vertex_count == 7) {
      ++four_tetrahedron_count;
    } else {
      throw std::runtime_error(
          "CUDA quadrature supports only tetrahedron/twin/four elements");
    }
    GpuQuadratureElement& quadrature_element =
        problem.quadrature_elements[
            static_cast<std::size_t>(leaf_index)];
    quadrature_element.vertex_count = vertex_count;
    quadrature_element.leaf_index = leaf_index;
    quadrature_element.hierarchy_index =
        problem.leaf_hierarchy_indices[
            static_cast<std::size_t>(leaf_index)];
    for (int vertex = 0; vertex < vertex_count; ++vertex) {
      const AFEPack::Point<DIM>& point =
          mesh.point()[geometry.vertex()[vertex]];
      quadrature_element.vertices[vertex] =
          {point[0], point[1], point[2]};
    }
    quadrature_element.sample_count = checked_gpu_index(
        treecode_quadrature::quadrature_point_count(
            static_cast<std::size_t>(vertex_count),
            quadrature_accuracy),
        "Element quadrature point count");
  }

  std::size_t sample_count = 0;
  for (int leaf_index = 0; leaf_index < leaf_count; ++leaf_index) {
    HElement<DIM>* source =
        leaf_elements[static_cast<std::size_t>(leaf_index)];
    GpuLeaf& gpu_leaf =
        problem.leaves[static_cast<std::size_t>(leaf_index)];
    GpuQuadratureElement& quadrature_element =
        problem.quadrature_elements[
            static_cast<std::size_t>(leaf_index)];
    quadrature_element.sample_offset =
        checked_gpu_index(sample_count, "Quadrature sample count");
    gpu_leaf.sample_offset = quadrature_element.sample_offset;
    gpu_leaf.sample_count = quadrature_element.sample_count;
    sample_count +=
        static_cast<std::size_t>(quadrature_element.sample_count);
    const auto& vertices = source->h_element->vertex;
    if (vertices.size() != 4) {
      throw std::runtime_error(
          "CUDA near-field integration requires tetrahedral leaves");
    }
    for (int vertex = 0; vertex < 4; ++vertex) {
      gpu_leaf.vertices[vertex] = {
          (*vertices[vertex])[0],
          (*vertices[vertex])[1],
          (*vertices[vertex])[2]};
    }
    problem.domain_volume += hierarchy_cache[
        static_cast<std::size_t>(
            problem.leaf_hierarchy_indices[
                static_cast<std::size_t>(leaf_index)])].volume;
  }
  problem.generated_sample_count = sample_count;
  checked_gpu_index(sample_count, "Quadrature sample count");
  std::cout
      << "GPU local quadrature: algebraic accuracy="
      << quadrature_accuracy
      << ", elements(tetra/twin/four)="
      << tetrahedron_count << "/"
      << twin_tetrahedron_count << "/"
      << four_tetrahedron_count
      << ", total points=" << sample_count << '\n'
      << "GPU quadrature mesh volume: "
      << std::setprecision(15) << problem.domain_volume << '\n';

  // Pack the raw hierarchy. CUDA constructs the MAC lists and aggregates
  // moments bottom-up, so neither large interaction vectors nor moment arrays
  // are copied from the CPU.
  problem.multipoles.resize(hierarchy_cache.size());
  problem.child_offsets.resize(hierarchy_cache.size() + 1);
  for (std::size_t hierarchy_index = 0;
       hierarchy_index < hierarchy_cache.size();
       ++hierarchy_index) {
    const hElementCache& source = hierarchy_cache[hierarchy_index];
    GpuMultipole& multipole = problem.multipoles[hierarchy_index];
    multipole.center = {
        source.ref_point[0], source.ref_point[1], source.ref_point[2]};
    multipole.radius = source.radius;
    multipole.volume = source.volume;
    multipole.level = source.level;
    multipole.leaf_begin =
        checked_gpu_index(source.subtree_leaf_begin, "Subtree leaf begin");
    multipole.leaf_end =
        checked_gpu_index(source.subtree_leaf_end, "Subtree leaf end");
    problem.child_offsets[hierarchy_index] =
        checked_gpu_index(
            problem.child_hierarchy_indices.size(),
            "Hierarchy child count");
    problem.child_hierarchy_indices.insert(
        problem.child_hierarchy_indices.end(),
        source.child_indices.begin(),
        source.child_indices.end());
  }
  problem.child_offsets.back() =
      checked_gpu_index(
          problem.child_hierarchy_indices.size(),
          "Hierarchy child count");
  for (const auto& entry : element_to_hierarchy_index) {
    HElement<DIM>* element = entry.first;
    const int hierarchy_index = entry.second;
    if (element->parent == nullptr) {
      problem.root_hierarchy_indices.push_back(hierarchy_index);
    }
    if (element->value == 0) {
      if (element->index < 0
          || static_cast<std::size_t>(element->index)
              >= problem.leaves.size()
          || problem.leaf_hierarchy_indices[
                 static_cast<std::size_t>(element->index)]
              != hierarchy_index) {
        throw std::runtime_error(
            "CUDA leaf-to-hierarchy mapping is inconsistent");
      }
      problem.multipoles[
          static_cast<std::size_t>(hierarchy_index)].leaf_index =
          element->index;
    }
  }
  return problem;
}
#endif



#ifndef TREECODE_NO_MAIN
int main(int argc, char* argv[])
{
  const double program_start_time = omp_get_wtime();
  ProgramOptions options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  const char* list_diagnostics_path =
      std::getenv("TREECODE_LIST_DIAGNOSTICS_JSON");
  const bool list_diagnostics_only =
      list_diagnostics_path != nullptr && list_diagnostics_path[0] != '\0';
  const char* mesh_only_flag = std::getenv("TREECODE_MESH_ONLY");
  const bool mesh_only =
      mesh_only_flag != nullptr && std::string(mesh_only_flag) == "1";
  const bool use_gpu_backend = options.backend == ComputeBackend::gpu;
#ifdef TREECODE_REUSE_EXPERIMENT
  reuse_sequence_amplitude = options.sequence_initial_amplitude;
#endif

  std::cout << "Configuration: initial_refinements="
            << options.initial_refinements
            << ", quadrature_accuracy=" << options.quadrature_accuracy
            << ", expansion_order=" << options.expansion_order
            << ", theta=" << options.mac_theta
            << ", backend=" << backend_name(options.backend)
#ifdef TREECODE_UNIFORM_MESH
            << ", mesh=uniform"
#else
            << ", mesh=adaptive"
#ifndef TREECODE_REUSE_EXPERIMENT
            << ", adapt_tolerance_stage1="
            << options.adaptive_tolerance_stage1
            << ", adapt_tolerance_stage2="
            << options.adaptive_tolerance_stage2
#endif
#endif
#ifdef TREECODE_COMPARE_EXACT
            << ", comparison="
            << (options.compare_solution
                    ? "analytic exact solution" : "disabled")
            << '\n';
#else
            << ", reference=" << options.reference_file
            << ", comparison="
            << (options.compare_solution ? "enabled" : "disabled")
            << '\n';
#endif

#ifdef TREECODE_REUSE_EXPERIMENT
  std::cout << "Reuse sequence: steps=" << options.sequence_steps
            << ", decay=" << options.sequence_decay
            << ", initial perturbation amplitude="
            << options.sequence_initial_amplitude << '\n';
#endif


  HGeometryTree<DIM> h_tree;
  h_tree.readMesh(options.mesh_file);

  std::unique_ptr<IrregularMesh<DIM>> ir_mesh = std::make_unique<IrregularMesh<DIM>>();

  ir_mesh->reinit(h_tree);
  ir_mesh->semiregularize();
  ir_mesh->regularize(false);
  ir_mesh->globalRefine(options.initial_refinements);
  FEMSpace<double,DIM> fem_space;
    std::vector<TemplateElement<double,DIM,DIM> > template_elements;
    
    /// 对非正则网格做半正则化和正则化
    ir_mesh->semiregularize();
    ir_mesh->regularize(false);

    /// 这就是通过正则化得到的网格
    RegularMesh<DIM>& _r_mesh = ir_mesh->regularMesh();
    // _r_mesh.writeOpenDXData("D.dx");
    
    template_elements.resize(3);
    // std::vector<TemplateElement<double,DIM,DIM> > template_elements(3);
   /// for tetrahedron
  TemplateGeometry<DIM> tet_geo;
  tet_geo.readData("tetrahedron.tmp_geo");
  TemplateDOF<DIM> tet_template_dof(tet_geo);
  tet_template_dof.readData("tetrahedron.1.tmp_dof");
  CoordTransform<DIM> tet_crd_trs;
  tet_crd_trs.readData("tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> tet_basFunctionAdmin(tet_template_dof);
  tet_basFunctionAdmin.readData("tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> tet_unit_outNormal;
  tet_unit_outNormal.readData("tetrahedron.out_nrm");

  template_elements[0].reinit(tet_geo,
			      tet_template_dof,
			      tet_crd_trs,
			      tet_basFunctionAdmin,
			      tet_unit_outNormal);

  /// for twin_tetrahedron
  TemplateGeometry<DIM> twin_tet_geo;
  twin_tet_geo.readData("twin_tetrahedron.tmp_geo");
  TemplateDOF<DIM> twin_tet_template_dof(twin_tet_geo);
  twin_tet_template_dof.readData("twin_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> twin_tet_crd_trs;
  twin_tet_crd_trs.readData("twin_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> twin_tet_basFunctionAdmin(twin_tet_template_dof);
  twin_tet_basFunctionAdmin.readData("twin_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> twin_tet_unit_outNormal;
  twin_tet_unit_outNormal.readData("twin_tetrahedron.out_nrm");

  template_elements[1].reinit(twin_tet_geo,
			      twin_tet_template_dof,
			      twin_tet_crd_trs,
			      twin_tet_basFunctionAdmin,
			      twin_tet_unit_outNormal);

  /// for four_tetrahedron
  TemplateGeometry<DIM> four_tet_geo;
  four_tet_geo.readData("four_tetrahedron.tmp_geo");
  TemplateDOF<DIM> four_tet_template_dof(four_tet_geo);
  four_tet_template_dof.readData("four_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> four_tet_crd_trs;
  four_tet_crd_trs.readData("four_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> four_tet_basFunctionAdmin(four_tet_template_dof);
  four_tet_basFunctionAdmin.readData("four_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> four_tet_unit_outNormal;
  four_tet_unit_outNormal.readData("four_tetrahedron.out_nrm");

  template_elements[2].reinit(four_tet_geo,
			      four_tet_template_dof,
			      four_tet_crd_trs,
			      four_tet_basFunctionAdmin,
			      four_tet_unit_outNormal);
 
   //FEMSpace<double,DIM> fem_space(mesh, template_elements);
   
    fem_space.reinit(_r_mesh, template_elements);
   int _n_element = _r_mesh.n_geometry(DIM);
   fem_space.element().resize(_n_element);
   for (int i = 0;i < _n_element;i ++) {
     if (_r_mesh.geometry(DIM,i).n_vertex() == 4) {
       fem_space.element(i).reinit(fem_space,i,0);
     } else if (_r_mesh.geometry(DIM,i).n_vertex() == 5){
       fem_space.element(i).reinit(fem_space,i,1);      
     } else if (_r_mesh.geometry(DIM,i).n_vertex() == 7){
       fem_space.element(i).reinit(fem_space,i,2); 
     } else{} //{std::cout << ""}
   }


   fem_space.buildElement();
   fem_space.buildDof();
   fem_space.buildDofBoundaryMark();

   
#ifndef TREECODE_UNIFORM_MESH
    /// 下面一段计算用于加密的指示子。
    Indicator<DIM> indicator(_r_mesh);
    get_indicator(
        indicator, _r_mesh, fem_space, options.quadrature_accuracy,
        use_gpu_backend);

    /// 下面的几行调用进行自适应的函数，都是套话。
    std::unique_ptr<IrregularMesh<DIM>> old_ir_mesh = std::make_unique<IrregularMesh<DIM>>(*ir_mesh);
  ir_mesh = std::make_unique<IrregularMesh<DIM>>(); 

    
  MeshAdaptor<DIM> mesh_adaptor(*old_ir_mesh, *ir_mesh);
  mesh_adaptor.setIndicator(indicator);
  mesh_adaptor.tolerence() = options.adaptive_tolerance_stage1;
  mesh_adaptor.convergenceOrder() = 0.;
  mesh_adaptor.refineStep() = 5;
  mesh_adaptor.adapt();

  // } while(i < atoi(argv[2]));   
  ir_mesh->semiregularize();
  ir_mesh->regularize(false);
  RegularMesh<DIM> &r_mesh1 = ir_mesh->regularMesh();
  if (!list_diagnostics_only) r_mesh1.writeOpenDXData("D.dx");
  fem_space.reinit(r_mesh1, template_elements);
   fem_space.element().resize(r_mesh1.n_geometry(DIM));
   const int n_element1 = static_cast<int>(r_mesh1.n_geometry(DIM));
   for (int i = 0; i < n_element1; ++i) {
     if (r_mesh1.geometry(DIM,i).n_vertex() == 4) {
       fem_space.element(i).reinit(fem_space,i,0);
     } else if (r_mesh1.geometry(DIM,i).n_vertex() == 5){
       fem_space.element(i).reinit(fem_space,i,1);      
     } else if (r_mesh1.geometry(DIM,i).n_vertex() == 7){
       fem_space.element(i).reinit(fem_space,i,2); 
     } else{} //{std::cout << ""}
   }


   fem_space.buildElement();
   fem_space.buildDof();
   fem_space.buildDofBoundaryMark();

   
  //     // 下面一段计算用于加密的指示子。
    Indicator<DIM> indicator1(r_mesh1);
    get_indicator(
        indicator1, r_mesh1, fem_space, options.quadrature_accuracy,
        use_gpu_backend);

    /// 下面的几行调用进行自适应的函数，都是套话。
    std::unique_ptr<IrregularMesh<DIM>> old_ir_mesh1 = std::make_unique<IrregularMesh<DIM>>(*ir_mesh);
  ir_mesh = std::make_unique<IrregularMesh<DIM>>(); 

    
  MeshAdaptor<DIM> mesh_adaptor1(*old_ir_mesh1, *ir_mesh);
  mesh_adaptor1.setIndicator(indicator1);
  mesh_adaptor1.tolerence() = options.adaptive_tolerance_stage2;
  mesh_adaptor1.convergenceOrder() = 0.;
  mesh_adaptor1.refineStep() = 2;
  mesh_adaptor1.adapt();
#endif

  //   } while(i < atoi(argv[2]));   
  ir_mesh->semiregularize();
  ir_mesh->regularize(false);
  RegularMesh<DIM>& r_mesh2 = ir_mesh->regularMesh();
#ifdef TREECODE_UNIFORM_MESH
  if (!list_diagnostics_only && !mesh_only)
    r_mesh2.writeOpenDXData("D_uniform.dx");
#else
  if (!list_diagnostics_only && !mesh_only)
    r_mesh2.writeOpenDXData("D.dx");
#endif
    fem_space.reinit(r_mesh2, template_elements);
   int _n_element2 = r_mesh2.n_geometry(DIM);
   fem_space.element().resize(_n_element2);
   for (int i = 0;i < _n_element2;i ++) {
     if (r_mesh2.geometry(DIM,i).n_vertex() == 4) {
       fem_space.element(i).reinit(fem_space,i,0);
     } else if (r_mesh2.geometry(DIM,i).n_vertex() == 5){
       fem_space.element(i).reinit(fem_space,i,1);      
     } else if (r_mesh2.geometry(DIM,i).n_vertex() == 7){
       fem_space.element(i).reinit(fem_space,i,2); 
     } else{} //{std::cout << ""}
   }


   fem_space.buildElement();
   fem_space.buildDof();
   fem_space.buildDofBoundaryMark();

  if (mesh_only) {
    std::cout << "Mesh-only leaf count: "
              << r_mesh2.n_geometry(DIM) << '\n';
    return 0;
  }

  std::vector<hElementCache> hElement_cache;
  std::map<HElement<DIM>*, int> map_HElement_to_int;  
  const bool construct_cpu_interactions = !use_gpu_backend;
  double mapping_geometry_seconds = 0.0;
  double hierarchy_validation_seconds = 0.0;
  int omp_threads_used = 1;
  std::map<std::string, double> interaction_diagnostics;
  const double initial_list_construction_start = omp_get_wtime();
  double neighbor_fill_seconds = update_hElementCache(
      *ir_mesh,
      hElement_cache,
      map_HElement_to_int,
      options.mac_theta,
      construct_cpu_interactions,
      &mapping_geometry_seconds,
      &hierarchy_validation_seconds,
      &omp_threads_used,
      list_diagnostics_only ? &interaction_diagnostics : nullptr);
  const double initial_list_construction_seconds =
      omp_get_wtime() - initial_list_construction_start;
#ifndef TREECODE_REUSE_EXPERIMENT
  (void) initial_list_construction_seconds;
#endif
#ifndef TREECODE_COMPARE_EXACT
  (void) neighbor_fill_seconds;
#endif

  if (list_diagnostics_only) {
    if (use_gpu_backend) {
      throw std::runtime_error(
          "TREECODE_LIST_DIAGNOSTICS_JSON requires the CPU backend");
    }
    std::ofstream diagnostics_file(list_diagnostics_path);
    if (!diagnostics_file.is_open()) {
      throw std::runtime_error(
          std::string("Cannot open list-diagnostics output: ")
          + list_diagnostics_path);
    }
    diagnostics_file << "{\n"
                     << "  \"mesh\": \"adaptive\",\n"
                     << "  \"N\": " << r_mesh2.n_geometry(DIM) << ",\n"
                     << "  \"p\": " << options.quadrature_accuracy << ",\n"
                     << "  \"P\": " << options.expansion_order << ",\n"
                     << "  \"theta\": " << std::setprecision(17)
                     << options.mac_theta << ",\n"
                     << "  \"tolerance_stage1\": "
                     << options.adaptive_tolerance_stage1 << ",\n"
                     << "  \"tolerance_stage2\": "
                     << options.adaptive_tolerance_stage2 << ",\n"
                     << "  \"threads\": " << omp_threads_used << ",\n"
                     << "  \"mapping_geometry_s\": "
                     << mapping_geometry_seconds << ",\n"
                     << "  \"hierarchy_validation_s\": "
                     << hierarchy_validation_seconds << ",\n"
                     << "  \"list_kernel_s\": "
                     << neighbor_fill_seconds;
    for (auto iterator = interaction_diagnostics.begin();
         iterator != interaction_diagnostics.end(); ++iterator) {
      diagnostics_file << ",\n  \"" << iterator->first << "\": "
                       << iterator->second;
    }
    diagnostics_file << "\n}\n";
    if (!diagnostics_file) {
      throw std::runtime_error(
          std::string("Failed to write list diagnostics: ")
          + list_diagnostics_path);
    }
    std::cout << "List-only diagnostics written to "
              << list_diagnostics_path << '\n';
    return 0;
  }
  
  std::vector<ElementCache> element_cache;
  if (!use_gpu_backend) {
  updateElementCache(
      element_cache, fem_space, options.quadrature_accuracy);
  IrregularMesh<DIM>::ActiveIterator
    the_activeElement = ir_mesh->beginActiveElement(),
    end_activeElement = ir_mesh->endActiveElement();      
  for(;the_activeElement != end_activeElement;++ the_activeElement){
    const int& hElement_idx = the_activeElement->index;
    if (hElement_idx < 0 ||
        static_cast<std::size_t>(hElement_idx) >= element_cache.size()) {
      throw std::runtime_error("Active element index is outside element cache");
    }
    ElementCache& the_ec = element_cache[hElement_idx];
    the_ec.the_hElement = &(*the_activeElement);
    the_ec.hierarchy_index =
        map_HElement_to_int.at(the_ec.the_hElement);
  }

  double quadrature_volume = 0.0;
  for (const ElementCache& cache : element_cache) {
    if (cache.the_hElement == nullptr) {
      throw std::runtime_error("Element cache is not connected to a leaf");
    }
    quadrature_volume += cache.volume;
    const hElementCache& hierarchy_entry =
        hElement_cache[map_HElement_to_int.at(cache.the_hElement)];
    const double volume_tolerance =
        1.0e-10 * std::max(1.0, hierarchy_entry.volume);
    if (std::abs(cache.volume - hierarchy_entry.volume) > volume_tolerance) {
      throw std::runtime_error(
          "Regular-element and hierarchy-element volumes disagree");
    }
  }
  std::cout << "Gaussian quadrature mesh volume: "
            << std::setprecision(15) << quadrature_volume << '\n';
  std::cout
      << "Target-element singular integration: closed-form constant-density "
         "tetrahedron potential plus Duffy quadrature of the density "
         "remainder\n";
  } else {
  std::cout
      << "Local quadrature point generation deferred to CUDA\n";
  }

#ifdef TREECODE_REUSE_EXPERIMENT
  return run_reuse_sequence_experiment(
      options,
      initial_list_construction_seconds,
      *ir_mesh,
      element_cache,
      hElement_cache,
      map_HElement_to_int);
#endif
  
  std::vector<HElement<DIM> *> root_element_list;
  IrregularMesh<DIM>::RootIterator
    the_root = ir_mesh->beginRootElement(),
    end_root = ir_mesh->endRootElement();
  for(;the_root != end_root;++ the_root){
    root_element_list.push_back(&(*the_root));
  }
  double b_time = omp_get_wtime();
  const auto& read_only_map = map_HElement_to_int;
  const auto& read_only_element_cache = element_cache;
  const int multipole_order = options.expansion_order;
  if (!use_gpu_backend) {
  {
  const std::vector<HElement<DIM>*> multipole_elements =
      [&map_HElement_to_int, &hElement_cache]() {
        std::vector<HElement<DIM>*> elements(
            hElement_cache.size(), nullptr);
        for (const auto& entry : map_HElement_to_int) {
          elements[entry.second] = entry.first;
        }
        if (std::find(elements.begin(), elements.end(), nullptr) !=
            elements.end()) {
          throw std::runtime_error(
              "Incomplete hierarchy mapping before multipole preprocessing.");
        }
        return elements;
      }();
  #pragma omp parallel for schedule(dynamic, 16) default(none) \
      shared(multipole_elements, hElement_cache, read_only_map, \
             read_only_element_cache) firstprivate(multipole_order)
  for(std::size_t i = 0; i < multipole_elements.size(); ++i){
    update_multipoleInfo_hElementCache(
        multipole_elements[i], hElement_cache, read_only_map,
        read_only_element_cache, multipole_order, false);
  }
  std::cout << "Updating multipoleInfo uses "
            << omp_get_wtime() - b_time << " seconds\n";
  }
  } else {
  std::cout
      << "Multipole moment calculation deferred to CUDA\n";
  }
  const std::size_t moments_per_cluster =
      static_cast<std::size_t>(
          treecode_multipole::coefficient_count(multipole_order));
  const double moment_storage_mib =
      static_cast<double>(
          moments_per_cluster * hElement_cache.size() * sizeof(double))
      / (1024.0 * 1024.0);
  std::cout << "Cartesian multipole order " << multipole_order
            << ": " << moments_per_cluster
            << " symmetric coefficients/cluster, "
            << moment_storage_mib << " MiB of moment values\n";

  if (!options.compare_solution) {
    bool has_neighbor_fill_time = construct_cpu_interactions;
#ifdef TREECODE_USE_CUDA
    if (use_gpu_backend) {
      GpuTreecodeProblem gpu_problem;
      try {
        gpu_problem = flatten_treecode_for_gpu(
            std::vector<GpuPoint3>{},
            std::vector<int>{},
            std::vector<int>{},
            fem_space,
            hElement_cache,
            read_only_map,
            options.quadrature_accuracy,
            options.mac_theta,
            true,
            true);
      } catch (const std::exception& error) {
        std::cerr << "CUDA data error: " << error.what() << '\n';
        return 1;
      }

      try {
        const GpuTreecodeResult gpu_result = evaluate_treecode_gpu(
            gpu_problem,
            options.expansion_order,
            options.quadrature_accuracy,
            false,
            false);
        neighbor_fill_seconds = gpu_result.interaction_seconds;
        has_neighbor_fill_time = true;
        std::cout
            << "CUDA device: " << gpu_result.device_name << '\n'
            << "CUDA quadrature generation time: "
            << gpu_result.quadrature_seconds << " seconds\n"
            << "CUDA source-density preparation time: "
            << gpu_result.source_preparation_seconds << " seconds\n"
            << "CUDA multipole moment time: "
            << gpu_result.moment_seconds << " seconds\n"
            << "CUDA MAC list construction and validation time: "
            << gpu_result.interaction_seconds << " seconds\n"
            << "CUDA interactions: "
            << gpu_result.near_interaction_count << " near, "
            << gpu_result.far_interaction_count << " far\n"
            << "Final potential evaluation disabled\n";
      } catch (const std::exception& error) {
        std::cerr << "CUDA error: " << error.what() << '\n';
        return 1;
      }
    }
#endif
    const double calculation_without_comparison_seconds =
        omp_get_wtime() - program_start_time;
    if (!append_timing_result(
            options,
            static_cast<std::size_t>(fem_space.n_element()),
            mapping_geometry_seconds,
            hierarchy_validation_seconds,
            has_neighbor_fill_time,
            neighbor_fill_seconds,
            calculation_without_comparison_seconds,
            false,
            0.0,
            false,
            0.0,
            false,
            0.0,
            omp_threads_used)) {
      return 1;
    }
    std::cout
        << "Comparison disabled by command-line flag; "
           "preprocessing completed without evaluating comparison targets "
           "and timing was appended to "
        << timing_results_filename() << '\n'
        << "Calculation time without comparison: "
        << calculation_without_comparison_seconds << " seconds\n";
    return 0;
  }

#ifdef TREECODE_COMPARE_EXACT
  // Evaluate once at the volume barycenter of every adaptive leaf.  Gaussian
  // points remain source-integration data; they are deliberately not reused as
  // targets.  Element volumes weight the resulting leaf-center discrete L2
  // comparison.
  b_time = omp_get_wtime();
  double exact_error_squared = 0.0;
  double exact_norm_squared = 0.0;
  double approximation_norm_squared = 0.0;
  double exact_error = 0.0;
  double exact_norm = 0.0;
  double approximation_norm = 0.0;
  double relative_exact_error = 0.0;
  double potential_evaluation_seconds = 0.0;
  std::size_t evaluation_count = 0;
  std::size_t nonfinite_count = 0;
  const int expansion_order = options.expansion_order;
  const int quadrature_accuracy = options.quadrature_accuracy;
  const auto& read_only_hierarchy_cache = hElement_cache;
  std::vector<int> leaf_hierarchy_indices(
      static_cast<std::size_t>(fem_space.n_element()), -1);
  for (const auto& entry : read_only_map) {
    const HElement<DIM>* element = entry.first;
    if (element->value == 0) {
      if (element->index < 0
          || static_cast<std::size_t>(element->index)
              >= leaf_hierarchy_indices.size()
          || leaf_hierarchy_indices[
                 static_cast<std::size_t>(element->index)] != -1) {
        std::cerr << "Error: invalid leaf index for barycenter targets\n";
        return 1;
      }
      leaf_hierarchy_indices[static_cast<std::size_t>(element->index)] =
          entry.second;
    }
  }
  double leaf_target_weight_sum = 0.0;
  for (int hierarchy_index : leaf_hierarchy_indices) {
    if (hierarchy_index < 0
        || static_cast<std::size_t>(hierarchy_index)
            >= read_only_hierarchy_cache.size()) {
      std::cerr << "Error: invalid leaf-to-hierarchy target mapping\n";
      return 1;
    }
    leaf_target_weight_sum += read_only_hierarchy_cache[
        static_cast<std::size_t>(hierarchy_index)].volume;
  }
  if (!std::isfinite(leaf_target_weight_sum)
      || !(leaf_target_weight_sum > 0.0)) {
    std::cerr << "Error: invalid total leaf volume for barycenter targets\n";
    return 1;
  }

#ifdef TREECODE_USE_CUDA
  if (use_gpu_backend) {
  GpuTreecodeProblem gpu_problem;
  try {
    const std::size_t target_count = leaf_hierarchy_indices.size();
    std::vector<GpuPoint3> targets(target_count);
    std::vector<int> target_hierarchy_indices(target_count, -1);
    std::vector<int> target_leaf_indices(target_count, -1);
    std::vector<double> target_weights(target_count, 0.0);
    for (std::size_t element_index = 0;
         element_index < target_count;
         ++element_index) {
      const int hierarchy_index = leaf_hierarchy_indices[element_index];
      if (hierarchy_index < 0
          || static_cast<std::size_t>(hierarchy_index)
              >= read_only_hierarchy_cache.size()) {
        throw std::runtime_error(
            "Invalid leaf cache while constructing barycenter targets");
      }
      const hElementCache& target_hierarchy_cache =
          read_only_hierarchy_cache[
              static_cast<std::size_t>(hierarchy_index)];
      targets[element_index] = {
          target_hierarchy_cache.ref_point[0],
          target_hierarchy_cache.ref_point[1],
          target_hierarchy_cache.ref_point[2]};
      target_hierarchy_indices[element_index] =
          hierarchy_index;
      target_leaf_indices[element_index] =
          static_cast<int>(element_index);
      target_weights[element_index] = target_hierarchy_cache.volume;
    }
    gpu_problem = flatten_treecode_for_gpu(
        std::move(targets),
        std::move(target_hierarchy_indices),
        std::move(target_leaf_indices),
        fem_space,
        read_only_hierarchy_cache,
        read_only_map,
        quadrature_accuracy,
        options.mac_theta,
        false,
        false);
    gpu_problem.target_weights = std::move(target_weights);
  } catch (const std::exception& error) {
    std::cerr << "CUDA data error: " << error.what() << '\n';
    return 1;
  }
  std::cout << "CUDA exact data: "
            << gpu_problem.targets.size()
            << " leaf-barycenter targets, "
            << gpu_problem.generated_sample_count
            << " generated source samples, "
            << gpu_problem.multipoles.size() << " hierarchy nodes\n";

  GpuTreecodeResult gpu_result;
  try {
    gpu_result = evaluate_treecode_gpu(
        gpu_problem,
        expansion_order,
        quadrature_accuracy,
        true);
  } catch (const std::exception& error) {
    std::cerr << "CUDA error: " << error.what() << '\n';
    return 1;
  }
  if (!gpu_result.has_l2_norms) {
    std::cerr << "CUDA error: device L2 reduction did not return norms\n";
    return 1;
  }
  evaluation_count = gpu_result.evaluation_count;
  nonfinite_count += gpu_result.nonfinite_count;
  exact_error_squared = gpu_result.exact_error_squared;
  exact_norm_squared = gpu_result.exact_norm_squared;
  approximation_norm_squared = gpu_result.approximation_norm_squared;
  exact_error = gpu_result.exact_error;
  exact_norm = gpu_result.exact_norm;
  approximation_norm = gpu_result.approximation_norm;
  relative_exact_error = gpu_result.relative_exact_error;
  potential_evaluation_seconds = gpu_result.kernel_seconds;
  neighbor_fill_seconds = gpu_result.interaction_seconds;
  std::cout << "CUDA device: " << gpu_result.device_name << '\n'
            << "CUDA potential evaluator: multi-warp, "
            << gpu_result.evaluation_threads_per_target
            << " threads per target\n"
            << "CUDA quadrature generation time: "
            << gpu_result.quadrature_seconds << " seconds\n"
            << "CUDA source-density preparation time: "
            << gpu_result.source_preparation_seconds << " seconds\n"
            << "CUDA multipole moment time: "
            << gpu_result.moment_seconds << " seconds\n"
            << "CUDA MAC list construction and validation time: "
            << gpu_result.interaction_seconds << " seconds\n"
            << "CUDA interactions: "
            << gpu_result.near_interaction_count << " near, "
            << gpu_result.far_interaction_count << " far\n"
            << "CUDA treecode kernel time: "
            << gpu_result.kernel_seconds << " seconds\n"
            << "CUDA L2 norm reduction time: "
            << gpu_result.norm_reduction_seconds << " seconds\n"
            << "CUDA exact comparison copied only final norm scalars to CPU\n";
  } else
#endif
  {
  const std::size_t adaptive_element_count = element_cache.size();
  const double potential_evaluation_start = omp_get_wtime();
  #pragma omp parallel for schedule(dynamic, 16) default(none) \
      shared(read_only_element_cache, read_only_hierarchy_cache) \
      firstprivate(adaptive_element_count, expansion_order, \
                   quadrature_accuracy) \
      reduction(+:exact_error_squared,exact_norm_squared, \
                  approximation_norm_squared,evaluation_count, \
                  nonfinite_count)
  for (std::size_t element_index = 0;
       element_index < adaptive_element_count;
       ++element_index) {
    const ElementCache& target_cache =
        read_only_element_cache[element_index];
    HElement<DIM>* target_element = target_cache.the_hElement;
    if (target_element == nullptr
        || target_cache.hierarchy_index < 0
        || static_cast<std::size_t>(target_cache.hierarchy_index)
            >= read_only_hierarchy_cache.size()) {
      ++nonfinite_count;
      continue;
    }
    const hElementCache& target_hierarchy_cache =
        read_only_hierarchy_cache[
            static_cast<std::size_t>(target_cache.hierarchy_index)];
    std::array<double, treecode_multipole::maximum_coefficient_count>
        coefficient_workspace{};

    const AFEPack::Point<DIM>& target = target_hierarchy_cache.ref_point;
    const double approximation = multipole_approximation(
        target,
        target_element,
        target_hierarchy_cache,
        read_only_hierarchy_cache,
        read_only_element_cache,
        expansion_order,
        quadrature_accuracy,
        coefficient_workspace.data());
    const double exact_value = u(target);
    const double weight = target_hierarchy_cache.volume;
    if (!std::isfinite(approximation)
        || !std::isfinite(exact_value)
        || !std::isfinite(weight)) {
      ++nonfinite_count;
      continue;
    }

    const double difference = approximation - exact_value;
    exact_error_squared += weight * difference * difference;
    exact_norm_squared += weight * exact_value * exact_value;
    approximation_norm_squared +=
        weight * approximation * approximation;
    ++evaluation_count;
  }
  potential_evaluation_seconds =
      omp_get_wtime() - potential_evaluation_start;
  }

  if (nonfinite_count != 0) {
    std::cerr << "Error: " << nonfinite_count
              << " invalid adaptive leaf-barycenter evaluations\n";
    return 1;
  }
  if (evaluation_count == 0) {
    std::cerr << "Error: no adaptive leaf barycenters were evaluated\n";
    return 1;
  }
  if (exact_error_squared < 0.0
      || exact_norm_squared < 0.0
      || approximation_norm_squared < 0.0) {
    std::cerr
        << "Error: leaf-volume weighting produced a negative squared norm\n";
    return 1;
  }

  if (!use_gpu_backend) {
    exact_error = std::sqrt(exact_error_squared);
    exact_norm = std::sqrt(exact_norm_squared);
    approximation_norm = std::sqrt(approximation_norm_squared);
    relative_exact_error =
        exact_norm > 0.0 ? exact_error / exact_norm : 0.0;
  }
  const double exact_evaluation_seconds = omp_get_wtime() - b_time;
  std::cout << std::setprecision(12)
            << "Exact-comparison leaf barycenters: "
            << evaluation_count << '\n'
            << "Leaf-barycenter target volume: "
            << leaf_target_weight_sum << '\n'
            << "Exact solution: exp(-4*pi*x^2-8*pi*y^2-12*pi*z^2)\n"
            << "L2 norm of exact solution: " << exact_norm << '\n'
            << "L2 norm of treecode solution: "
            << approximation_norm << '\n'
            << "L2 error against exact solution: "
            << exact_error << '\n'
            << "Relative L2 error against exact solution: "
            << relative_exact_error << '\n'
            << "Exact-comparison evaluation time: "
            << exact_evaluation_seconds << " seconds\n"
            << "Potential evaluation time: "
            << potential_evaluation_seconds << " seconds\n";

  // Record the production time before any optional mesh-independent
  // validation.  The latter is a diagnostic and is not part of the solver
  // timing reported in the article.
  const double total_execution_seconds =
      omp_get_wtime() - program_start_time;

  const char* common_target_file =
      std::getenv("TREECODE_COMMON_TARGET_FILE");
  if (common_target_file != nullptr && common_target_file[0] != '\0') {
    struct WeightedTarget {
      AFEPack::Point<DIM> point;
      double weight = 0.0;
    };
    std::ifstream input(common_target_file);
    if (!input.is_open()) {
      std::cerr << "Error: cannot open common-target file "
                << common_target_file << '\n';
      return 1;
    }
    std::vector<WeightedTarget> targets;
    WeightedTarget target;
    while (input >> target.point[0] >> target.point[1]
                 >> target.point[2] >> target.weight) {
      if (!(target.weight > 0.0) || !std::isfinite(target.weight)) {
        std::cerr << "Error: invalid common-target weight\n";
        return 1;
      }
      targets.push_back(target);
    }
    if (!input.eof() || targets.empty()) {
      std::cerr << "Error: malformed or empty common-target file "
                << common_target_file << '\n';
      return 1;
    }

    double common_error_squared = 0.0;
    double common_exact_norm_squared = 0.0;
    double common_weight_sum = 0.0;
    std::size_t common_missing = 0;
    std::size_t common_nonfinite = 0;
    const std::size_t common_target_count = targets.size();
    const double common_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 16) default(none) \
        shared(targets, root_element_list, read_only_map, \
               read_only_hierarchy_cache, read_only_element_cache) \
        firstprivate(common_target_count, expansion_order, \
                     quadrature_accuracy) \
        reduction(+:common_error_squared,common_exact_norm_squared, \
                    common_weight_sum,common_missing,common_nonfinite)
    for (std::size_t target_index = 0;
         target_index < common_target_count;
         ++target_index) {
      const WeightedTarget& sample = targets[target_index];
      HElement<DIM>* target_leaf =
          findLeafContainingPoint(root_element_list, sample.point);
      if (target_leaf == nullptr) {
        ++common_missing;
        continue;
      }
      const auto hierarchy_iterator = read_only_map.find(target_leaf);
      if (hierarchy_iterator == read_only_map.end()) {
        ++common_missing;
        continue;
      }
      const hElementCache& target_cache = read_only_hierarchy_cache[
          static_cast<std::size_t>(hierarchy_iterator->second)];
      std::array<double,
                 treecode_multipole::maximum_coefficient_count>
          coefficient_workspace{};
      const double approximation = multipole_approximation(
          sample.point,
          target_leaf,
          target_cache,
          read_only_hierarchy_cache,
          read_only_element_cache,
          expansion_order,
          quadrature_accuracy,
          coefficient_workspace.data());
      const double exact_value = u(sample.point);
      if (!std::isfinite(approximation) || !std::isfinite(exact_value)) {
        ++common_nonfinite;
        continue;
      }
      const double difference = approximation - exact_value;
      common_error_squared +=
          sample.weight * difference * difference;
      common_exact_norm_squared +=
          sample.weight * exact_value * exact_value;
      common_weight_sum += sample.weight;
    }
    const double common_seconds = omp_get_wtime() - common_start;
    if (common_missing != 0 || common_nonfinite != 0) {
      std::cerr << "Error: common-target validation has "
                << common_missing << " missing and "
                << common_nonfinite << " non-finite targets\n";
      return 1;
    }
    const double common_error = std::sqrt(common_error_squared);
    const double common_exact_norm =
        std::sqrt(common_exact_norm_squared);
    std::cout << std::setprecision(12)
              << "Common-target count: " << common_target_count << '\n'
              << "Common-target weight sum: " << common_weight_sum << '\n'
              << "Common-target L2 norm of exact solution: "
              << common_exact_norm << '\n'
              << "Common-target L2 error against exact solution: "
              << common_error << '\n'
              << "Common-target relative L2 error: "
              << (common_exact_norm > 0.0
                      ? common_error / common_exact_norm : 0.0)
              << '\n'
              << "Common-target validation time: "
              << common_seconds << " seconds\n";
  }

  if (!append_timing_result(
          options,
          static_cast<std::size_t>(fem_space.n_element()),
          mapping_geometry_seconds,
          hierarchy_validation_seconds,
          true,
          neighbor_fill_seconds,
          total_execution_seconds,
          true,
          potential_evaluation_seconds,
          true,
          exact_evaluation_seconds,
          true,
          exact_error,
          omp_threads_used)) {
    return 1;
  }
  std::cout << "Appended execution timings to "
            << timing_results_filename() << '\n'
            << "Total execution time: "
            << total_execution_seconds << " seconds\n";
  return 0;
#else
  // ============ 1. 串行读取所有数据 ============
b_time = omp_get_wtime();
auto start = std::chrono::high_resolution_clock::now();

std::ifstream infile(options.reference_file, std::ios::binary);
if (!infile.is_open()) {
    std::cerr << "Error: cannot open for reading.\n";
    return 1;
}

const int record_size = DIM + 1; // DIM coords + 1 value
std::vector<double> buffer(record_size);

// 存储所有读取的数据
std::vector<AFEPack::Point<DIM>> all_q_points;
std::vector<double> all_approximations_u;

// 预分配内存提高性能
all_q_points.reserve(1000000);
all_approximations_u.reserve(1000000);

// 串行读取文件
while (infile.read(reinterpret_cast<char*>(buffer.data()),
                   record_size * sizeof(double))) 
{
    AFEPack::Point<DIM> q_point_u;
    for (int d = 0; d < DIM; ++d) {
        if (!std::isfinite(buffer[d])) {
            std::cerr << "Error: non-finite reference coordinate\n";
            return 1;
        }
        q_point_u[d] = buffer[d];
    }
    if (!std::isfinite(buffer[DIM])) {
        std::cerr << "Error: non-finite reference value\n";
        return 1;
    }
    all_q_points.push_back(q_point_u);
    // buffer[DIM] = u(q_point_u); // 
    all_approximations_u.push_back(buffer[DIM]);
}
if (infile.gcount() != 0) {
    std::cerr << "Error: truncated record in " << options.reference_file
              << '\n';
    return 1;
}
infile.close();

std::cout << "Read " << all_q_points.size() << " reference points\n";

// ============ 2. 并行计算所有点的多极展开 ============
double l2_error = 0.0;
double exact_l2_error = 0.0;
double reference_exact_l2_error = 0.0;
double reference_l2_norm = 0.0;
double exact_l2_norm = 0.0;
std::size_t missing_points = 0;
const size_t n_points = all_q_points.size();
if (n_points == 0) {
    std::cerr << "Error: reference file contains no complete records\n";
    return 1;
}

double domain_volume = 0.0;
double first_root_volume = 0.0;
for (const HElement<DIM>* root : root_element_list) {
    const auto& vertices = root->h_element->vertex;
    const double root_volume = tetrahedronVolume(
        *vertices[0], *vertices[1], *vertices[2], *vertices[3]);
    if (first_root_volume == 0.0) {
      first_root_volume = root_volume;
    } else if (std::abs(root_volume - first_root_volume) >
               1.0e-12 * std::max(1.0, first_root_volume)) {
      std::cerr
          << "Error: equal reference weights require equal root volumes\n";
      return 1;
    }
    domain_volume += root_volume;
}
if (n_points % root_element_list.size() != 0) {
    std::cerr << "Error: reference record count is not a uniform refinement\n";
    return 1;
}
std::size_t cells_per_root = n_points / root_element_list.size();
while (cells_per_root > 1 && cells_per_root % 8 == 0) {
    cells_per_root /= 8;
}
if (cells_per_root != 1) {
    std::cerr << "Error: reference record count is not 8^level per root\n";
    return 1;
}
const double scale_factor = domain_volume / static_cast<double>(n_points);
const int expansion_order = options.expansion_order;
const int quadrature_accuracy = options.quadrature_accuracy;
const auto& reference_points = all_q_points;
const auto& reference_values = all_approximations_u;
const auto& read_only_roots = root_element_list;
const auto& read_only_hierarchy_cache = hElement_cache;

  #ifdef TREECODE_USE_CUDA
  std::vector<GpuPoint3> gpu_targets(n_points);
  std::vector<int> gpu_target_hierarchy_indices(n_points, -1);
  std::vector<int> gpu_target_leaf_indices(n_points, -1);

// AFEPack's hierarchy is pointer based. Locate targets on the host once and
// pass stable integer indices to CUDA; the GPU never dereferences host mesh
  // objects.
  #pragma omp parallel for schedule(static) default(none) \
      shared(gpu_targets, gpu_target_hierarchy_indices, \
             gpu_target_leaf_indices, reference_points, \
             read_only_roots, read_only_map) \
      firstprivate(n_points) reduction(+:missing_points)
  for (size_t i = 0; i < n_points; ++i) {
    const AFEPack::Point<DIM>& point = reference_points[i];
    gpu_targets[i] = {point[0], point[1], point[2]};
  HElement<DIM>* target_leaf =
      findLeafContainingPoint(read_only_roots, point);
  if (target_leaf == nullptr) {
    ++missing_points;
    continue;
  }
    gpu_target_hierarchy_indices[i] =
        read_only_map.at(target_leaf);
    gpu_target_leaf_indices[i] = target_leaf->index;
}

if (missing_points != 0) {
  std::cerr << "Error: " << missing_points
            << " reference points were outside the adaptive mesh.\n";
  return 1;
}

  GpuTreecodeProblem gpu_problem;
  try {
    gpu_problem = flatten_treecode_for_gpu(
        std::move(gpu_targets),
        std::move(gpu_target_hierarchy_indices),
        std::move(gpu_target_leaf_indices),
        fem_space,
        read_only_hierarchy_cache,
        read_only_map,
        quadrature_accuracy,
        options.mac_theta,
        false,
        false);
  } catch (const std::exception& error) {
    std::cerr << "CUDA data error: " << error.what() << '\n';
    return 1;
  }

  std::cout << "CUDA data: " << gpu_problem.targets.size() << " targets, "
            << gpu_problem.generated_sample_count
            << " generated quadrature samples, "
            << gpu_problem.multipoles.size() << " hierarchy nodes\n";

GpuTreecodeResult gpu_result;
try {
  gpu_result = evaluate_treecode_gpu(
      gpu_problem, expansion_order, quadrature_accuracy);
} catch (const std::exception& error) {
  std::cerr << "CUDA error: " << error.what() << '\n';
  return 1;
}
const std::vector<double>& gpu_approximations = gpu_result.potentials;

// Error norms are inexpensive and remain on the CPU. This also checks every
// value copied back from the device before it enters a reported norm.
#pragma omp parallel for schedule(static) default(none) \
    shared(reference_points, reference_values, gpu_approximations) \
    firstprivate(n_points, scale_factor) \
    reduction(+:l2_error,exact_l2_error,reference_exact_l2_error, \
                reference_l2_norm,exact_l2_norm)
for (size_t i = 0; i < n_points; ++i) {
  const AFEPack::Point<DIM>& q_point = reference_points[i];
  const double approximation_u = reference_values[i];
  const double approximation = gpu_approximations[i];
  if (!std::isfinite(approximation)) {
    l2_error = std::numeric_limits<double>::quiet_NaN();
    continue;
  }
  const double exact_value = u(q_point);
  const double reference_diff = approximation_u - approximation;
  const double exact_diff = exact_value - approximation;
  const double reference_exact_diff = approximation_u - exact_value;
  l2_error += scale_factor * reference_diff * reference_diff;
  exact_l2_error += scale_factor * exact_diff * exact_diff;
  reference_exact_l2_error +=
      scale_factor * reference_exact_diff * reference_exact_diff;
  reference_l2_norm +=
      scale_factor * approximation_u * approximation_u;
  exact_l2_norm += scale_factor * exact_value * exact_value;
}
if (!std::isfinite(l2_error)) {
  std::cerr << "Error: CUDA returned a non-finite approximation\n";
  return 1;
}
std::cout << "CUDA device: " << gpu_result.device_name << '\n'
          << "CUDA quadrature generation time: "
          << gpu_result.quadrature_seconds << " seconds\n"
          << "CUDA source-density preparation time: "
          << gpu_result.source_preparation_seconds << " seconds\n"
          << "CUDA multipole moment time: "
          << gpu_result.moment_seconds << " seconds\n"
          << "CUDA MAC list construction and validation time: "
          << gpu_result.interaction_seconds << " seconds\n"
          << "CUDA interactions: "
          << gpu_result.near_interaction_count << " near, "
          << gpu_result.far_interaction_count << " far\n"
          << "CUDA treecode kernel time: "
          << gpu_result.kernel_seconds << " seconds\n";
#else
// Meshes, maps, caches, and reference samples are read-only here. Worker
// threads update only private temporaries and OpenMP reduction variables.
#pragma omp parallel for schedule(dynamic, 100) default(none) \
    shared(reference_points, reference_values, read_only_roots, \
           read_only_map, read_only_hierarchy_cache, read_only_element_cache) \
    firstprivate(n_points, scale_factor, expansion_order, \
                 quadrature_accuracy) \
    reduction(+:l2_error,exact_l2_error,reference_exact_l2_error, \
                reference_l2_norm,exact_l2_norm,missing_points)
for (size_t i = 0; i < n_points; ++i) {
        
        const AFEPack::Point<DIM>& q_point = reference_points[i];
        const double approximation_u = reference_values[i];
        
        // Point location is read-only and can run concurrently.
        HElement<DIM>* the_hElement =
            findLeafContainingPoint(read_only_roots, q_point);
        
        if (!the_hElement) {
            ++missing_points;
            continue;
        }
        const hElementCache& target_hierarchy_cache =
            read_only_hierarchy_cache[
                static_cast<std::size_t>(read_only_map.at(the_hElement))];
        std::array<double, treecode_multipole::maximum_coefficient_count>
            coefficient_workspace{};
        
        // 2. 计算多极展开近似
        double approximation = multipole_approximation(q_point,
                                                      the_hElement,
                                                      target_hierarchy_cache,
                                                      read_only_hierarchy_cache,
                                                      read_only_element_cache,
                                                      expansion_order,
                                                      quadrature_accuracy,
                                                      coefficient_workspace.data());
        
        const double exact_value = u(q_point);
        const double reference_diff = approximation_u - approximation;
        const double exact_diff = exact_value - approximation;
        const double reference_exact_diff = approximation_u - exact_value;
        l2_error += scale_factor * reference_diff * reference_diff;
        exact_l2_error += scale_factor * exact_diff * exact_diff;
        reference_exact_l2_error +=
            scale_factor * reference_exact_diff * reference_exact_diff;
        reference_l2_norm +=
            scale_factor * approximation_u * approximation_u;
        exact_l2_norm += scale_factor * exact_value * exact_value;
}
#endif

// ============ 3. 输出结果 ============
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;

std::cout << "Execution time: " << elapsed.count() << " seconds\n";
std::cout << "Doing the calculation uses " << omp_get_wtime() - b_time << " seconds " << std::endl;
if (missing_points != 0) {
    std::cerr << "Error: " << missing_points
              << " reference points were outside the adaptive mesh.\n";
    return 1;
}
const double reference_error = std::sqrt(l2_error);
const double reference_norm = std::sqrt(reference_l2_norm);
const double exact_error = std::sqrt(exact_l2_error);
const double exact_norm = std::sqrt(exact_l2_norm);
const double reference_exact_error = std::sqrt(reference_exact_l2_error);
std::cout << std::setprecision(12)
          << "Domain volume represented by reference points: "
          << domain_volume << '\n'
          << "L2 error against uniform reference: " << reference_error << '\n'
          << "Relative L2 error against uniform reference: "
          << (reference_norm > 0.0 ? reference_error / reference_norm : 0.0)
          << '\n'
          << "L2 error against analytic solution: " << exact_error << '\n'
          << "Relative L2 error against analytic solution: "
          << (exact_norm > 0.0 ? exact_error / exact_norm : 0.0) << '\n'
          << "Uniform-reference error against analytic solution: "
          << reference_exact_error << '\n';
if (reference_exact_error > exact_error) {
  std::cout
      << "Warning: the uniform reference is less accurate than the adaptive "
         "result; use the analytic error for convergence studies.\n";
}

return 0;
#endif
}
#endif
