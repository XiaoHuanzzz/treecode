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
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <memory>

/// for mesh
#include <AFEPack/HGeometry.h>
/// for template element
#include <AFEPack/TemplateElement.h>
/// for space
#include <AFEPack/DGFEMSpace.h>

#include "local_quadrature.hpp"
#include "cartesian_multipole.hpp"
#ifdef TREECODE_WATER10_MODEL
#include "water10_test_problem.hpp"
#else
#include "gaussian_test_problem.hpp"
#endif
#include "hnl_candidates.hpp"
#include "hnl_family_mac.hpp"
#include "mac_geometry.hpp"

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
constexpr int DEFAULT_INDICATOR_ACCURACY = 8;

enum class ComputeBackend
{
  cpu,
  gpu
};

enum class IndicatorMode
{
  radial,
  gaussian_radial,
  density_variance,
  hartree_curvature
};

const char* backend_name(ComputeBackend backend)
{
  return backend == ComputeBackend::cpu ? "CPU" : "GPU";
}

const char* indicator_mode_name(IndicatorMode mode)
{
  switch (mode) {
    case IndicatorMode::radial:
      return "RADIAL";
    case IndicatorMode::gaussian_radial:
      return "GAUSSIAN_RADIAL";
    case IndicatorMode::density_variance:
      return "DENSITY_VARIANCE";
    case IndicatorMode::hartree_curvature:
      return "HARTREE_CURVATURE";
  }
  return "UNKNOWN";
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
  IndicatorMode indicator_mode = IndicatorMode::radial;
  int indicator_accuracy = DEFAULT_INDICATOR_ACCURACY;
  bool report_mesh_estimator = false;
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

IndicatorMode parse_indicator_mode(const char* text)
{
  std::string value(text);
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  if (value == "RADIAL") return IndicatorMode::radial;
  if (value == "GAUSSIAN_RADIAL" || value == "ELLIPSOIDAL_RADIAL")
    return IndicatorMode::gaussian_radial;
  if (value == "DENSITY_VARIANCE" || value == "VARIANCE")
    return IndicatorMode::density_variance;
  if (value == "HARTREE_CURVATURE" || value == "CURVATURE")
    return IndicatorMode::hartree_curvature;
  throw std::invalid_argument(
      "Invalid indicator mode: " + value
      + " (use RADIAL, GAUSSIAN_RADIAL, DENSITY_VARIANCE, "
        "or HARTREE_CURVATURE)");
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
  if (argc < 2 || argc > 13) {
    throw std::invalid_argument(
        "Usage: tree_exact MESH_FILE [INITIAL_REFINEMENTS] "
        "[QUADRATURE_ACCURACY] [EXPANSION_ORDER] [MAC_THETA] "
        "[COMPARE] [BACKEND] [ADAPT_TOLERANCE_STAGE1] "
        "[ADAPT_TOLERANCE_STAGE2] [INDICATOR_MODE] "
        "[INDICATOR_ORDER] [REPORT_MESH_ESTIMATOR]\n"
        "       BACKEND is CPU or GPU; COMPARE and BACKEND may be swapped");
  }
#endif
#else
  if (argc < 2 || argc > 14) {
    throw std::invalid_argument(
        "Usage: tree MESH_FILE [INITIAL_REFINEMENTS] [QUADRATURE_ACCURACY] "
        "[EXPANSION_ORDER] [MAC_THETA] [REFERENCE_FILE] [COMPARE] "
        "[BACKEND] [ADAPT_TOLERANCE_STAGE1] "
        "[ADAPT_TOLERANCE_STAGE2] [INDICATOR_MODE] "
        "[INDICATOR_ORDER] [REPORT_MESH_ESTIMATOR]");
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
  if (argc >= 12) {
    options.indicator_mode = parse_indicator_mode(argv[11]);
  }
  if (argc >= 13) {
    options.indicator_accuracy =
        parse_int(argv[12], "indicator quadrature accuracy");
  }
  if (argc >= 14) {
    options.report_mesh_estimator = parse_comparison_flag(argv[13]);
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
  if (argc >= 11) {
    options.indicator_mode = parse_indicator_mode(argv[10]);
  }
  if (argc >= 12) {
    options.indicator_accuracy =
        parse_int(argv[11], "indicator quadrature accuracy");
  }
  if (argc >= 13) {
    options.report_mesh_estimator = parse_comparison_flag(argv[12]);
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
  if (options.indicator_accuracy < 1 ||
      options.indicator_accuracy >
          treecode_quadrature::maximum_algebraic_accuracy) {
    throw std::invalid_argument(
        "INDICATOR_ORDER must be between 1 and "
        + std::to_string(
            treecode_quadrature::maximum_algebraic_accuracy));
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
  // Legacy visualization lists; evaluation uses the compact lists below.
  std::vector<HElement<DIM>*> neighbor_elements;
  std::vector<HElement<DIM>*> nonNeighbor_elements;

  // HNL followed by MAC stores the lists used by the evaluator as
  // compact integer indices.  This halves list storage relative to pointers
  // on a 64-bit host and removes a std::map lookup from every far interaction.
  // The pointer lists above are retained only for the legacy geometric-layer
  // visualization helpers.
  std::vector<int> neighbor_leaf_indices;
  std::vector<int> far_hierarchy_indices;
  // Only retained when CUDA will apply MAC to the CPU-generated HNL cover.
  std::vector<int> hnl_candidate_indices;

  int level = 0;
  // Number of active (leaf) elements represented by this hierarchy node.
  // It is computed once in a bottom-up hierarchy pass and lets interaction
  // lists be checked without expanding every cluster to all descendants.
  std::size_t subtree_leaf_count = 0;
  std::size_t subtree_leaf_begin = 0;
  std::size_t subtree_leaf_end = 0;
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


void construct_neighbor_layers(
    const std::set<HGeometry<0, DIM>*>& initial_vertices,
    std::set<HGeometry<0, DIM>*>& outer_vertices_set,
    std::set<HGeometry<0, DIM>*>& vertices_set,
    const std::set<HElement<DIM>*>& initial_layer,
    std::set<HElement<DIM>*>& outer_layer,  
    const int initial_level,
    std::set<HElement<DIM>*>& neighbor_of_the_elements,
    const std::vector<std::vector<std::set<HElement<DIM>* > > >& node_neighbors,
    const int flag
) { 
    outer_layer.erase(outer_layer.begin(), outer_layer.end());
    outer_vertices_set.erase(outer_vertices_set.begin(), outer_vertices_set.end());
    neighbor_of_the_elements.erase(neighbor_of_the_elements.begin(), neighbor_of_the_elements.end());
    
    if(flag == 1){
      
      // Find neighbors of initial vertices
      for (const auto& vertex : initial_vertices) {
          if (vertex != nullptr) {  // Check if vertex is valid
              int idx_vtx = vertex->index;  // Get index of the vertex
    
              // Ensure idx_vtx is valid before accessing node_neighbors
              if (idx_vtx >= 0 &&
                  static_cast<std::size_t>(idx_vtx) < node_neighbors.size()) {
                  if (initial_level >= 0 &&
                      static_cast<std::size_t>(initial_level) <
                          node_neighbors[idx_vtx].size()) {
                      neighbor_of_the_elements.insert(
                          node_neighbors[idx_vtx][initial_level].begin(),
                          node_neighbors[idx_vtx][initial_level].end()
                      );
                  }
              }
              vertices_set.insert(vertex);
          }
      }
  
      //Erase the last layer
      for (auto element : initial_layer) {
          if (element != nullptr) {  // Ensure element is valid
              neighbor_of_the_elements.erase(const_cast<HElement<DIM>*>(element));
          }
      }
  
      std::set<HElement<DIM>*> elements_to_add;  // Temporary container for new elements
    
      // Store siblings in elements_to_add, store parents in outer_layers
      for (auto element : neighbor_of_the_elements) {
          if (element != nullptr && element->parent != nullptr) {
              // outer_layer.insert(element->parent);  // Insert the parent directly
              for (auto child : element->parent->child) {
                  if (child != nullptr) {  // Ensure child is valid
                      elements_to_add.insert(child);  // Add children to temporary set
                  }
              }
          }
      }
  
      for (auto element : initial_layer) {
          if (element != nullptr) {  // Ensure element is valid
              elements_to_add.erase(const_cast<HElement<DIM>*>(element));
          }
      }
  
      // Now merge the elements from `elements_to_add` into `neighbor_of_the_elements`
      neighbor_of_the_elements.insert(elements_to_add.begin(), elements_to_add.end());
    
      
  
      for (auto element : neighbor_of_the_elements) {
          if (element != nullptr && element->parent != nullptr) {
              outer_layer.insert(element->parent); 
          }
      }
  
      // Store outer_layer's vertices in outer_vertices_set
      for (const auto& element : outer_layer) {
          if (element != nullptr) {  // Ensure element is valid
              std::vector<HGeometry<0, DIM>*> vertices = element->h_element->vertex;
              if (vertices.size() > 0) {
                  outer_vertices_set.insert(vertices.begin(), vertices.end());
              }
          }
      }
  
      // Erase initial_vertices in outer_vertices_set
      for (const auto& vertex : vertices_set) {
          if (vertex != nullptr) {  // Ensure vertex is valid
              outer_vertices_set.erase(const_cast<HGeometry<0, DIM>*>(vertex));  // Erase one by one
          }
      }
  }
  else if (flag == 0){
    // Find neighbors of initial vertices
    // initial_level--;
      for (const auto& vertex : initial_vertices) {
          if (vertex != nullptr) {  // Check if vertex is valid
              int idx_vtx = vertex->index;  // Get index of the vertex
    
              // Ensure idx_vtx is valid before accessing node_neighbors
              if (idx_vtx >= 0 &&
                  static_cast<std::size_t>(idx_vtx) < node_neighbors.size()) {
                  if (initial_level >= 0 &&
                      static_cast<std::size_t>(initial_level) <
                          node_neighbors[idx_vtx].size()) {
                      neighbor_of_the_elements.insert(
                          node_neighbors[idx_vtx][initial_level].begin(),
                          node_neighbors[idx_vtx][initial_level].end()
                      );
                  }
              }
              vertices_set.insert(vertex);
          }
      }

      //Erase the last layer
      for (auto element : initial_layer) {
        const int& n_child = element->child.size();
      for(int i = 0;i < n_child;++ i){
          if (element->child[i] != nullptr) {  // Ensure element is valid
              neighbor_of_the_elements.erase(const_cast<HElement<DIM>*>(element->child[i]));
          }
      }
    }

      std::set<HElement<DIM>*> elements_to_add;  // Temporary container for new elements
    
      // Store siblings in elements_to_add, store parents in outer_layers
      for (auto element : neighbor_of_the_elements) {
          if (element != nullptr && element->parent != nullptr) {
              // outer_layer.insert(element->parent);  // Insert the parent directly
              for (auto child : element->parent->child) {
                  if (child != nullptr) {  // Ensure child is valid
                      elements_to_add.insert(child);  // Add children to temporary set
                  }

              }
          }
    }

    for (auto element : initial_layer) {
        const int& n_child = element->child.size();
      for(int i = 0;i < n_child;++ i){
          if (element->child[i] != nullptr) {  // Ensure element is valid
            elements_to_add.erase(const_cast<HElement<DIM>*>(element->child[i]));
          }
      }
    }

    // Now merge the elements from `elements_to_add` into `neighbor_of_the_elements`
    neighbor_of_the_elements.insert(elements_to_add.begin(), elements_to_add.end());
  
    

    for (auto element : neighbor_of_the_elements) {
        if (element != nullptr && element->parent != nullptr) {
            outer_layer.insert(element->parent); 
        }
    }

    // Store outer_layer's vertices in outer_vertices_set
    for (const auto& element : outer_layer) {
        if (element != nullptr) {  // Ensure element is valid
            std::vector<HGeometry<0, DIM>*> vertices = element->h_element->vertex;
            if (vertices.size() > 0) {
                outer_vertices_set.insert(vertices.begin(), vertices.end());
            }
        }
    }

    // Erase initial_vertices in outer_vertices_set
    for (const auto& vertex : vertices_set) {
        if (vertex != nullptr) {  // Ensure vertex is valid
            outer_vertices_set.erase(const_cast<HGeometry<0, DIM>*>(vertex));  // Erase one by one
        }
    }
  }
}


void record_current_hElement_node_neighbors(HElement<DIM>& the_HElement,
              int level,
              std::vector<std::vector<std::set<HElement<DIM>* > > >& node_neighbors,
              int& n_hElement)
{
  n_hElement ++;
  HGeometry<DIM>* the_geometry = the_HElement.h_element;
  std::vector<HGeometry<0, DIM>* > the_vertex = the_geometry->vertex;

  const int& n_vtx = the_vertex.size();
  for(int i = 0;i < n_vtx;++ i){
    auto& neighbors_by_level = node_neighbors[the_vertex[i]->index];
    if (level >= static_cast<int>(neighbors_by_level.size())) {
      neighbors_by_level.resize(static_cast<std::size_t>(level + 1));
    }
    neighbors_by_level[level].insert(&the_HElement);
  }

  if(the_HElement.value == 0) return;
  else{
    // leaf_elements.push_back(&the_HElement);
    const int& n_child = the_HElement.child.size();
    for(int i = 0;i < n_child;++ i){
      record_current_hElement_node_neighbors(
          *the_HElement.child[i], level + 1, node_neighbors, n_hElement);
    }
  }
  
}

void find_all_leaf_elements(HElement<DIM>* current,
                            std::vector<HElement<DIM>*>& leaf_elements)
{
    // Base case: if current is a leaf node (has no children)
    if (current->value == 0) {
        leaf_elements.push_back(current);
        return;
    }

    // Recursive case: iterate through the children
    const std::vector<HElement<DIM>*>& children = current->child;
    for (HElement<DIM>* child : children) {
        find_all_leaf_elements(child, leaf_elements);
    }
}

void find_elements_MAC(RegularMesh<DIM>& r_mesh, 
  HElement<DIM>& the_hElement,
  hElementCache& the_hElement_cache,
  std::vector<hElementCache>& hElement_cache,
  const std::map<AFEPack::HElement<DIM>*, int>& map_HElement_to_int,
  int& level,
  AFEPack::Point<DIM> ref_point, int& flag)
{
  HGeometry<DIM>* the_hGeometry = the_hElement.h_element;
  std::vector<HGeometry<0, DIM> *> the_vtx = the_hGeometry->vertex;

  AFEPack::Point<DIM> A = *the_vtx[0];
  AFEPack::Point<DIM> B = *the_vtx[1];
  AFEPack::Point<DIM> C = *the_vtx[2];
  AFEPack::Point<DIM> D = *the_vtx[3];

  AFEPack::Point<DIM> current_point = A+B+C+D;
  current_point /= 4.0;

  AFEPack::Point<DIM> diff = current_point - ref_point;
  double distance = diff.length();

  AFEPack::Point<DIM> diffA = current_point - A;
  double distanceA = diffA.length();
  AFEPack::Point<DIM> diffB = current_point - B;
  double distanceB = diffB.length();
  AFEPack::Point<DIM> diffC = current_point - C;
  double distanceC = diffC.length();
  AFEPack::Point<DIM> diffD = current_point - D;
  double distanceD = diffD.length();

  double max_distance = std::max({distanceA, distanceB, distanceC, distanceD});
  if (max_distance/distance > 1.0/1.0){
    // down to the next level
    flag = 0;
    const int& n_child = the_hElement.child.size(); // cautious!
    if(the_hElement.value == 0){
      the_hElement_cache.neighbor_elements.push_back(&the_hElement);
      return;
    }
    else{
      for(int i = 0;i < n_child;++ i){
          find_elements_MAC(r_mesh, *(the_hElement.child[i]), the_hElement_cache, hElement_cache, map_HElement_to_int, level, ref_point, flag);
      }
    }
  }
  else{
    // add it to nonNeighbor
    the_hElement_cache.nonNeighbor_elements.push_back(&the_hElement);
    return;
  }
}

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

// ==================== 第二部分：填充邻居信息（无idx参数） ====================

void fill_neighbor_for_element(
    const std::vector<HElement<DIM>*>& root_element_list,
    HElement<DIM>& the_hElement,
    std::vector<hElementCache>& hElement_cache,
    const std::map<HElement<DIM>*, int>& map_HElement_to_int,
    int level,
    const std::vector<std::vector<std::set<HElement<DIM>* > > >& node_neighbors)
{
    // 从map获取索引，不再需要idx参数
    const int current_idx = map_HElement_to_int.at(&the_hElement);
    hElementCache& the_hElement_cache = hElement_cache[current_idx];
    
    // 计算几何属性
    HGeometry<DIM>* the_hGeometry = the_hElement.h_element;
    std::vector<HGeometry<0, DIM> *> the_vtx = the_hGeometry->vertex;
    
    AFEPack::Point<DIM> A = *the_vtx[0];
    AFEPack::Point<DIM> B = *the_vtx[1];
    AFEPack::Point<DIM> C = *the_vtx[2];
    AFEPack::Point<DIM> D = *the_vtx[3];
    
    AFEPack::Point<DIM> ref_point = A + B + C + D;
    ref_point /= 4.0;
    
    AFEPack::Point<DIM> dA = A - ref_point;
    AFEPack::Point<DIM> dB = B - ref_point;
    AFEPack::Point<DIM> dC = C - ref_point;
    AFEPack::Point<DIM> dD = D - ref_point;
    
    double da = dA.length();
    double db = dB.length();
    double dc = dC.length();
    double dd = dD.length();
    
    double length = std::max({da, db, dc, dd});
    the_hElement_cache.radius = length;
    
    double area = tetrahedronVolume(A, B, C, D);
    the_hElement_cache.volume = area;
  
    // 顶点集合
    std::set<HGeometry<0, DIM>*> initial_vertices;
    std::vector<HGeometry<0, DIM>*> vertices = the_hElement.h_element->vertex;
    initial_vertices.insert(vertices.begin(), vertices.end());
    
    // 邻层构建
    std::set<HGeometry<0, DIM>*> outer_vertices_set;
    std::set<HGeometry<0, DIM>*> vertices_set;
    std::set<HElement<DIM>*> neighbor_of_the_elements;
    std::set<HElement<DIM>*> nonneighbor_of_the_elements;
    std::set<HElement<DIM>*> outer_layer;
    std::set<HElement<DIM>*> initial_layer;
    initial_layer.insert(&the_hElement);
    
    // 第一次构建邻层
    construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, 
                             initial_layer, outer_layer, level, 
                             neighbor_of_the_elements, node_neighbors, 1);
    
    // 添加邻居元素
    for(auto element : neighbor_of_the_elements) {
        std::vector<HElement<DIM>*> leaf_elements;
        find_all_leaf_elements(element, leaf_elements);
        for(auto ele : leaf_elements) {
            the_hElement_cache.neighbor_elements.push_back(ele);
        }
    }
    
    the_hElement_cache.nonNeighbor_elements.insert(
        the_hElement_cache.nonNeighbor_elements.end(),
        root_element_list.begin(), root_element_list.end());
    the_hElement_cache.neighbor_elements.push_back(&the_hElement);
    
    int _level = level;
    
    // 第二次构建邻层
    initial_vertices.clear();
    for(auto vertex : outer_vertices_set) {
        initial_vertices.insert(vertex);
    }
    initial_layer.clear();
    for(auto element : outer_layer) {
        initial_layer.insert(element);
    }
    construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, 
                             initial_layer, outer_layer, _level, 
                             neighbor_of_the_elements, node_neighbors, 1);
    
    for(auto element : neighbor_of_the_elements) {
        std::vector<HElement<DIM>*> leaf_elements;
        find_all_leaf_elements(element, leaf_elements);
        for(auto ele : leaf_elements) {
            the_hElement_cache.neighbor_elements.push_back(ele);
        }
    }
    
    // 第三次构建邻层
    initial_vertices.clear();
    for(auto vertex : outer_vertices_set) {
        initial_vertices.insert(vertex);
    }
    initial_layer.clear();
    for(auto element : outer_layer) {
        initial_layer.insert(element);
    }
    construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, 
                             initial_layer, outer_layer, _level, 
                             neighbor_of_the_elements, node_neighbors, 1);
    
    for(auto element : neighbor_of_the_elements) {
        std::vector<HElement<DIM>*> leaf_elements;
        find_all_leaf_elements(element, leaf_elements);
        for(auto ele : leaf_elements) {
            the_hElement_cache.neighbor_elements.push_back(ele);
        }
    }
    
    // while循环处理剩余的层
    int flag = 1;
    while(_level > 0) {
        _level--;
        
        initial_vertices.clear();
        for(auto vertex : outer_vertices_set) {
            initial_vertices.insert(vertex);
        }
        initial_layer.clear();
        for(auto element : outer_layer) {
            initial_layer.insert(element);
        }
        
        construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, 
                                 initial_layer, outer_layer, _level, 
                                 neighbor_of_the_elements, node_neighbors, flag);
        
        for(auto element : neighbor_of_the_elements) {
            flag = 1;
            the_hElement_cache.nonNeighbor_elements.push_back(element);
        }
        
        if(flag == 0) {
            _level++;
        }
    }
    
    // 清理nonNeighbor_elements中的父元素
    std::set<HElement<DIM>*> elements_to_remove;
    
    for(auto element : the_hElement_cache.nonNeighbor_elements) {
        HElement<DIM>* element_parent = element;
        if(element_parent->parent == nullptr) {
            continue;
        }
        while(element_parent->parent != nullptr) {
            element_parent = element_parent->parent;
        }
        elements_to_remove.insert(element_parent);
    }
    
    for(auto elem : elements_to_remove) {
        the_hElement_cache.nonNeighbor_elements.erase(
            std::remove(
                the_hElement_cache.nonNeighbor_elements.begin(),
                the_hElement_cache.nonNeighbor_elements.end(), elem),
            the_hElement_cache.nonNeighbor_elements.end());
    }
    
    // 从nonNeighbor中移除neighbor中的元素
    elements_to_remove.clear();
    for(auto element : the_hElement_cache.neighbor_elements) {
        HElement<DIM>* element_parent = element;
        if(element_parent->parent == nullptr) {
            continue;
        }
        while(element_parent->parent != nullptr) {
            element_parent = element_parent->parent;
        }
        elements_to_remove.insert(element_parent);
    }
    
    for(auto elem : elements_to_remove) {
        the_hElement_cache.nonNeighbor_elements.erase(
            std::remove(
                the_hElement_cache.nonNeighbor_elements.begin(),
                the_hElement_cache.nonNeighbor_elements.end(), elem),
            the_hElement_cache.nonNeighbor_elements.end());
    }

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
    cache.hnl_candidate_indices.clear();

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

struct HierarchySummary
{
  std::size_t leaf_count = 0;
  double domain_volume = 0.0;
};

struct InteractionListStats
{
  std::size_t represented_leaf_count = 0;
  std::size_t near_entries = 0;
  std::size_t far_entries = 0;
  std::size_t visited_hierarchy_nodes = 0;
  std::size_t hnl_candidates = 0;
  std::size_t hnl_layers = 0;
  std::size_t hnl_star_entries = 0;
  long double represented_volume = 0.0L;
  long double near_volume = 0.0L;
  double maximum_mac_ratio = 0.0;
  bool contains_target_in_near_field = false;
  bool unique_entries = true;
};

struct InteractionWorkspace {
  std::vector<int> stack, near, far;
};

void classify_source_cover_to_workspace(
    int target_index,
    const std::vector<int>& candidate_indices,
    const treecode_mac::Geometry& geometry,
    double theta,
    InteractionListStats& stats,
    InteractionWorkspace& workspace)
{
    auto& stack = workspace.stack;
    auto& near = workspace.near;
    auto& far = workspace.far;
    near.clear();
    far.clear();
    std::size_t visited = 0;
    double maximum_ratio = 0.0;
    const auto& target = geometry.nodes[static_cast<std::size_t>(target_index)];
    // Refine a disjoint source cover using MAC. The production HNL path
    // supplies its geometric candidates; the separate root-walk benchmark
    // supplies the roots. Both use exactly the same acceptance and evaluator.
    // Reuse a worker-local descendant stack. Stream candidates in the same
    // reverse order as the former full-copy stack; no per-target allocation
    // or copy of the complete HNL cover is required.
    stack.clear();
    auto candidate = candidate_indices.rbegin();
    while (candidate != candidate_indices.rend() || !stack.empty()) {
      if (stack.empty()) stack.push_back(*candidate++);
      const int source_index = stack.back();
      stack.pop_back();
      ++visited;
      if (source_index < 0
          || static_cast<std::size_t>(source_index)
              >= geometry.nodes.size()) {
        stats.unique_entries = false;
        continue;
      }

      const auto& source = geometry.nodes[static_cast<std::size_t>(source_index)];
      const bool contains_target =
          source.leaf_begin <= target.leaf_begin
          && target.leaf_begin < source.leaf_end;
      const double minimum_target_distance = contains_target ? 0.0 :
          treecode_mac::center_distance(source, target) - target.radius;
      const double ratio = minimum_target_distance > 0.0
          ? source.radius / minimum_target_distance
          : std::numeric_limits<double>::infinity();
      const bool mac_accepted =
          !contains_target &&
          minimum_target_distance > 0.0 &&
          ratio <= theta;

      if (mac_accepted) {
        far.push_back(source_index);
#ifdef TREECODE_HNL_VERIFY
        stats.represented_leaf_count += source.leaf_count;
        stats.represented_volume += source.volume;
#endif
        maximum_ratio = std::max(maximum_ratio, ratio);
      } else if (source.leaf_index >= 0) {
        near.push_back(source.leaf_index);
#ifdef TREECODE_HNL_VERIFY
        ++stats.represented_leaf_count;
        stats.represented_volume += source.volume;
        stats.near_volume += source.volume;
        stats.contains_target_in_near_field =
            stats.contains_target_in_near_field || source_index == target_index;
#endif
      } else {
        const std::size_t end = source.child_begin + source.child_count;
        for (std::size_t child = source.child_begin; child < end; ++child) {
          stack.push_back(geometry.children[child]);
        }
      }
    }
    stats.visited_hierarchy_nodes = visited;
    stats.near_entries = near.size();
    stats.far_entries = far.size();
    stats.maximum_mac_ratio = maximum_ratio;
}

void classify_source_cover_by_mac(
    int target_index,
    const std::vector<int>& candidate_indices,
    hElementCache& target_cache,
    const treecode_mac::Geometry& geometry,
    double theta,
    InteractionListStats& stats,
    InteractionWorkspace& workspace)
{
    classify_source_cover_to_workspace(target_index, candidate_indices,
                                       geometry, theta, stats, workspace);
    target_cache.neighbor_elements.clear();
    target_cache.nonNeighbor_elements.clear();
    const auto& near = workspace.near;
    const auto& far = workspace.far;
    // Cached mode retains lists for repeated use. Streaming evaluates the
    // workspace directly and never performs these all-target list copies.
    target_cache.neighbor_leaf_indices.assign(near.begin(), near.end());
    target_cache.far_hierarchy_indices.assign(far.begin(), far.end());
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
    double theta, bool verify_partition)
{
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

    for (std::size_t i = 0; i < elements.size(); ++i) {
      if (elements[i]->value == 1) {
        continue;
      }
      ++validated_target_count;
      const InteractionListStats& stats = interaction_stats[i];
      if (verify_partition && !stats.unique_entries) {
        throw std::runtime_error(
            "MAC traversal generated a null or duplicate interaction");
      }
      if (verify_partition && !stats.contains_target_in_near_field) {
        throw std::runtime_error(
            "MAC traversal failed to include the target in its near field");
      }
      if (verify_partition && stats.represented_leaf_count != hierarchy.leaf_count) {
        throw std::runtime_error(
            "Near/far interaction lists do not represent every source leaf");
      }
      const long double partition_volume_error =
          std::abs(stats.represented_volume - domain_volume);
      if (verify_partition && partition_volume_error > volume_tolerance) {
        throw std::runtime_error(
            "Near/far interaction lists do not cover the truncated domain");
      }
      if (verify_partition && stats.maximum_mac_ratio > theta * (1.0 + 1.0e-12)) {
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
    }

    if (verify_partition && validated_target_count != hierarchy.leaf_count) {
      throw std::runtime_error(
          "Not every physical leaf received an interaction list");
    }
    const double target_denominator =
        hierarchy.leaf_count == 0
            ? 1.0
            : static_cast<double>(hierarchy.leaf_count);
    std::cout << (verify_partition ? "Validated " : "Constructed ") << validated_target_count
              << (verify_partition ? " disjoint source-tree partitions over truncated-domain "
                                   : " interaction lists over truncated-domain ")
              << "volume "
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
              << static_cast<double>(total_visited) / target_denominator;
    if (verify_partition)
      std::cout << ", near volume min/avg/max="
              << static_cast<double>(minimum_near_volume) << "/"
              << static_cast<double>(
                     total_near_volume / target_denominator)
              << "/" << static_cast<double>(maximum_near_volume)
              << "; max partition-volume error="
              << static_cast<double>(maximum_partition_volume_error);
    else
      std::cout << "; per-target coverage volumes not collected (verification build only)";
    std::cout << '\n';
}

treecode_hnl::Topology build_hnl_topology(
    const std::vector<HElement<DIM>*>& elements,
    const std::vector<hElementCache>& cache,
    const std::map<HElement<DIM>*, int>& indices,
    const std::vector<int>& roots)
{
  treecode_hnl::Topology topology;
  topology.roots = roots;
  topology.parents.assign(elements.size(), -1);
  topology.levels.resize(elements.size());
  topology.leaf_begin.resize(elements.size());
  topology.leaf_end.resize(elements.size());
  topology.children.resize(elements.size());
  topology.vertex_stars.resize(elements.size());
  // Child IDs were already validated and indexed by prepare_hierarchy_aggregates.
  // Reuse them to form parent links instead of a map lookup for every node.
  #pragma omp parallel for schedule(static) if(elements.size() > 10000)
  for (std::size_t i = 0; i < elements.size(); ++i) {
    topology.levels[i] = cache[i].level;
    topology.leaf_begin[i] = cache[i].subtree_leaf_begin;
    topology.leaf_end[i] = cache[i].subtree_leaf_end;
    topology.children[i] = cache[i].child_indices;
    for (int child : cache[i].child_indices)
      topology.parents[static_cast<std::size_t>(child)] = static_cast<int>(i);
  }
  std::unordered_map<const HGeometry<0, DIM>*, int> vertices;
  vertices.reserve(elements.size() / 4 + 1);
  for (std::size_t i = 0; i < elements.size(); ++i) {
    const auto* element = elements[i];
    for (int v = 0; v < 4; ++v) {
      const auto* vertex = element->h_element->vertex[v];
      // Unlike emplace, try_emplace avoids constructing an insertion node on
      // repeated vertices (most of the four-per-tetrahedron lookups).
      const int vertex_id = vertices.try_emplace(vertex, vertices.size()).first->second;
      if (topology.vertex_level_stars.size() <= static_cast<std::size_t>(vertex_id))
        topology.vertex_level_stars.resize(vertex_id + 1);
      auto& by_level = topology.vertex_level_stars[vertex_id];
      if (by_level.size() <= static_cast<std::size_t>(cache[i].level))
        by_level.resize(cache[i].level + 1, -1);
      // by_level is already the vertex/level lookup: no second hash table.
      int star = by_level[cache[i].level];
      if (star < 0) {
        star = static_cast<int>(topology.stars.size());
        topology.stars.emplace_back();
        topology.star_vertices.push_back(vertex_id);
        by_level[cache[i].level] = star;
      }
      topology.stars[star].push_back(static_cast<int>(i));
      topology.vertex_stars[i][v] = star;
    }
  }
  topology.vertex_count = vertices.size();
  // A single hierarchy pass avoids repeatedly walking to the root for each
  // emitted candidate in every target's HNL construction.
  topology.root_of_node.resize(elements.size(), -1);
  std::vector<int> pending;
  for (int root : roots) {
    pending.push_back(root);
    while (!pending.empty()) {
      const int node = pending.back();
      pending.pop_back();
      topology.root_of_node[node] = root;
      pending.insert(pending.end(), topology.children[node].begin(),
                     topology.children[node].end());
    }
  }
  topology.prepare_root_slots();
  return topology;
}

// Independent TEST-only audit. This never constructs, filters or repairs a
// layer; production list construction does not call it.
bool validate_hnl_cover(const std::vector<int>& candidates,
                        const std::vector<hElementCache>& cache,
                        std::size_t leaf_count)
{
  std::vector<std::pair<std::size_t, std::size_t>> intervals;
  intervals.reserve(candidates.size());
  for (int index : candidates)
    intervals.emplace_back(cache[index].subtree_leaf_begin,
                           cache[index].subtree_leaf_end);
  std::sort(intervals.begin(), intervals.end());
  std::size_t cursor = 0;
  for (const auto& interval : intervals) {
    if (interval.first != cursor || interval.second <= cursor) return false;
    cursor = interval.second;
  }
  return cursor == leaf_count;
}

#ifdef TREECODE_ROOT_BASED_LISTS
#include "root_based_neighbors.hpp"
#endif

#ifdef TREECODE_HNL_VISUALIZE
#include "hnl_visualization.hpp"
#endif

// Geometry is captured BEFORE moment preparation, which can rewrite ref_point.
// Thus streaming uses precisely the same MAC as the retained-list path.
struct StreamingInteractionContext {
  treecode_mac::Geometry geometry;
  treecode_hnl::Topology topology;
  std::vector<HElement<DIM>*> elements;
  std::vector<int> roots, leaf_hierarchy;
  std::vector<InteractionListStats> statistics;
  HierarchySummary hierarchy;
  double preparation_seconds = 0.0;
  bool family_mac = false;
  bool shared_targets = false;
  treecode_hnl::SharedTargetGroups target_groups;
  std::vector<std::vector<int>> shared_candidates;
  std::vector<treecode_hnl::LayerCounters> shared_counters;
  std::vector<std::size_t> shared_layer_counts;
};

// Geometry -> vertex/level incidence -> sibling-completed HNL layers -> MAC.
// CUDA receives the same HNL candidates and applies MAC on the device.
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
    StreamingInteractionContext* streaming = nullptr,
    bool family_mac = false)
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

    std::vector<int> root_indices;
    root_indices.reserve(root_element_list.size());
    for (HElement<DIM>* root : root_element_list) {
      root_indices.push_back(read_only_map.at(root));
    }
    const std::vector<int> read_only_root_indices = std::move(root_indices);
    if (streaming != nullptr) {
      streaming->elements = read_only_elements;
      streaming->roots = read_only_root_indices;
      streaming->hierarchy = hierarchy;
      streaming->statistics.resize(read_only_elements.size());
      streaming->family_mac = family_mac;
      const double start = omp_get_wtime();
      streaming->geometry = treecode_mac::Geometry(read_only_elements, hElement_cache);
#ifndef TREECODE_ROOT_BASED_LISTS
      streaming->topology = build_hnl_topology(read_only_elements, hElement_cache,
                                              read_only_map, read_only_root_indices);
#endif
#ifdef TREECODE_HNL_VERIFY
      streaming->leaf_hierarchy.resize(hierarchy.leaf_count, -1);
      for (std::size_t i = 0; i < read_only_elements.size(); ++i)
        if (read_only_elements[i]->value == 0)
          streaming->leaf_hierarchy[read_only_elements[i]->index] = static_cast<int>(i);
#endif
      streaming->preparation_seconds = omp_get_wtime() - start;
      std::cout << "CPU streaming geometry/topology preparation time: "
                << streaming->preparation_seconds << " seconds\n"
                << "CPU MAC geometry record bytes: " << sizeof(treecode_mac::Node)
                << "\nInteraction lists are temporary per worker; no all-target retention\n";
      return streaming->preparation_seconds;
    }
#ifndef TREECODE_ROOT_BASED_LISTS
    const auto& read_only_cache = hElement_cache;
#endif
    std::vector<InteractionListStats> interaction_stats(
        read_only_elements.size());
    const double interaction_start = omp_get_wtime();
    // Immutable compact data for the CPU classifier, shared by HNL and ROOT.
    // Include this one-time preparation in T_list; CUDA keeps its own packing.
    treecode_mac::Geometry mac_geometry;
    if (construct_interactions)
      mac_geometry = treecode_mac::Geometry(read_only_elements, hElement_cache);
    const double mac_geometry_seconds = omp_get_wtime() - interaction_start;
#ifdef TREECODE_ROOT_BASED_LISTS
    if (!construct_interactions)
      throw std::runtime_error("The root-based comparison is CPU-only");
    build_root_based_interactions(read_only_elements, read_only_root_indices, mac_geometry,
                                 hElement_cache, interaction_stats,
                                 hierarchy.leaf_count, theta);
    const double interaction_seconds = omp_get_wtime() - interaction_start;
    std::cout << "Interaction algorithm: root-based MAC tree walk\n";
#else
    const double topology_start = omp_get_wtime();
    const auto topology = build_hnl_topology(
        read_only_elements, read_only_cache, read_only_map,
        read_only_root_indices);
#ifdef TREECODE_HNL_VISUALIZE
    // Dedicated visualization executable only: export ONE actual target's
    // production HNL layers, then unwind before all-target lists/evaluation.
    export_hnl_visualization(topology, read_only_elements, read_only_cache,
                             hierarchy);
    throw HnlVisualizationComplete{};
#endif
#ifdef TREECODE_HNL_VERIFY
    constexpr bool audit_cover = true;
#else
    constexpr bool audit_cover = false;
#endif
    std::vector<int> leaf_hierarchy;
    if (audit_cover) {
      leaf_hierarchy.resize(hierarchy.leaf_count, -1);
      for (std::size_t i = 0; i < read_only_elements.size(); ++i)
        if (read_only_elements[i]->value == 0)
          leaf_hierarchy[read_only_elements[i]->index] = static_cast<int>(i);
    }
    const double topology_seconds = omp_get_wtime() - topology_start;
    std::size_t invalid_hnl = 0, total_candidates = 0, total_layers = 0;
    std::size_t total_star_entries = 0;
    std::size_t total_sibling_seeds = 0, total_sibling_groups = 0;
    std::size_t total_sibling_entries = 0;
    #pragma omp parallel default(none) \
        shared(read_only_elements, hElement_cache, topology, leaf_hierarchy, mac_geometry, \
               read_only_cache, interaction_stats) \
        firstprivate(theta, construct_interactions, audit_cover, hierarchy, family_mac) \
        reduction(+:invalid_hnl,total_candidates,total_layers,total_star_entries, \
                    total_sibling_seeds,total_sibling_groups,total_sibling_entries)
    {
      treecode_hnl::Workspace workspace(topology);
      treecode_hnl::FamilyPackingWorkspace family_workspace;
      InteractionWorkspace mac_workspace;
      #pragma omp for schedule(dynamic, 16)
      for(std::size_t i = 0; i < read_only_elements.size(); ++i) {
        if (read_only_elements[i]->value == 0) {
          std::size_t represented = 0;
          long double volume = 0.0L;
          // The constructor itself emits disjoint layers. Only the separate
          // verification executable checks them; it never repairs them.
          const auto& candidates = treecode_hnl::build_candidates(
              topology, static_cast<int>(i), workspace, [&](int index) {
                if (audit_cover) {
                  represented += read_only_cache[index].subtree_leaf_count;
                  volume += read_only_cache[index].volume;
                }
              });
          if (audit_cover && (represented != hierarchy.leaf_count ||
              std::abs(volume - hierarchy.domain_volume) >
                  1.0e-10L * std::max(1.0, hierarchy.domain_volume) ||
              !validate_hnl_cover(candidates, read_only_cache, hierarchy.leaf_count))) {
            ++invalid_hnl;
#ifdef TREECODE_HNL_REQUESTED_LAYERWISE
            interaction_stats[i].unique_entries = false;
            interaction_stats[i].represented_leaf_count = represented;
            interaction_stats[i].represented_volume = volume;
#endif
            continue;
          }
          auto& stats = interaction_stats[i];
          stats.hnl_candidates = candidates.size();
          stats.hnl_layers = workspace.layers;
          stats.hnl_star_entries = workspace.star_entries;
          total_candidates += candidates.size();
          total_layers += workspace.layers;
          total_star_entries += workspace.star_entries;
          total_sibling_seeds += workspace.sibling_seed_visits;
          total_sibling_groups += workspace.sibling_groups;
          total_sibling_entries += workspace.sibling_entries;
          if (construct_interactions) {
            const auto& mac_candidates = family_mac
                ? treecode_hnl::pack_complete_sibling_families(topology, workspace, family_workspace)
                : candidates;
            classify_source_cover_by_mac(
                static_cast<int>(i), mac_candidates,
                hElement_cache[i], mac_geometry,
                theta, stats, mac_workspace);
            if (audit_cover) {
              std::vector<int> final_cover = hElement_cache[i].far_hierarchy_indices;
              for (int leaf : hElement_cache[i].neighbor_leaf_indices)
                final_cover.push_back(leaf_hierarchy[leaf]);
              stats.unique_entries = validate_hnl_cover(
                  final_cover, read_only_cache, hierarchy.leaf_count);
            }
          } else {
            hElement_cache[i].hnl_candidate_indices = candidates;
          }
        }
      }
    }
    const double interaction_seconds = omp_get_wtime() - interaction_start;
    if (invalid_hnl != 0) {
#ifdef TREECODE_HNL_REQUESTED_LAYERWISE
      for (std::size_t i = 0; i < interaction_stats.size(); ++i) {
        if (interaction_stats[i].unique_entries) continue;
        std::cerr << std::setprecision(17)
                  << "Requested layerwise test: first failed target leaf="
                  << read_only_elements[i]->index << ", level=" << topology.levels[i]
                  << ", represented leaves=" << interaction_stats[i].represented_leaf_count
                  << "/" << hierarchy.leaf_count << ", represented volume="
                  << interaction_stats[i].represented_volume << "/" << hierarchy.domain_volume
                  << '\n';
        break;
      }
#endif
      throw std::runtime_error("HNL cover has gaps/overlaps for " +
                               std::to_string(invalid_hnl) + " targets");
    }
    const double target_count = static_cast<double>(hierarchy.leaf_count);
    std::cout << "Interaction algorithm: geometric HNL -> MAC\n"
              << "HNL topology time: " << topology_seconds << " seconds; "
              << "candidates avg=" << total_candidates / target_count
              << ", layers avg=" << total_layers / target_count
              << ", vertex-star entries avg=" << total_star_entries / target_count
              << "; exhaustive interval audit=" << (audit_cover ? "ON (test build)" : "NOT COMPILED")
              << (audit_cover ? "; invalid covers=" : "; construction failures=") << invalid_hnl << '\n';
    std::cout << "HNL constructor: original treecode.cpp set semantics; "
              << "one expansion per level; direct outer layer/vertices handoff; no repair adapter; "
              << "reused buffers and touched-byte membership; "
              << "sibling seeds/scans/entries avg="
              << total_sibling_seeds / target_count << "/"
              << total_sibling_groups / target_count << "/"
              << total_sibling_entries / target_count << '\n';
#endif

    if (construct_interactions)
      std::cout << "Compact MAC geometry preparation: " << mac_geometry_seconds
                << " seconds; " << sizeof(treecode_mac::Node) << " bytes/node\n";
    const double validation_start = omp_get_wtime();
#ifdef TREECODE_HNL_VERIFY
    constexpr bool verify_partition = true;
#else
    constexpr bool verify_partition = false;
#endif
    if (construct_interactions) validate_interaction_stats(
        read_only_elements, interaction_stats, hierarchy, theta, verify_partition);
    const double validation_seconds = omp_get_wtime() - validation_start;
    
    std::cout << "Neighbor preprocessing timing: mapping/geometry="
              << geometry_seconds
              << " seconds, hierarchy validation="
              << hierarchy_validation_seconds
              << " seconds, "
#ifndef TREECODE_ROOT_BASED_LISTS
              << "HNL and "
#endif
              << (construct_interactions ? "MAC construction=" : "CPU candidate construction (MAC deferred to CUDA)=")
              << interaction_seconds
              << " seconds, list statistics=" << validation_seconds
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
        StreamingInteractionContext* streaming = nullptr,
        bool family_mac = false)
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
      streaming,
      family_mac
  );

  std::cout << "There are total " << hElement_cache.size()
            << " hElements" << std::endl;
  if (streaming != nullptr) {
    std::cout << "HNL/MAC target traversal deferred until source data are ready\n";
  } else if (construct_interactions) {
    std::cout << "Filling neighbor_elements uses: "
              << neighbor_fill_seconds << " seconds" << std::endl;
  } else {
    std::cout << "HNL candidate preprocessing uses: " << neighbor_fill_seconds
              << " seconds; MAC classification deferred to CUDA\n";
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
    const std::vector<int>& far_indices,
    const std::vector<hElementCache>& hElement_cache,
    int expansion_order,
    double* coefficient_workspace)
{
    const int coefficient_count =
        treecode_multipole::coefficient_count(expansion_order);
    for (int source_index : far_indices)
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
            int source_component = -1,
            const InteractionWorkspace* temporary_interactions = nullptr)
{
  double approximation = 0.;
  const auto& near_indices = temporary_interactions
      ? temporary_interactions->near : target_hierarchy_cache.neighbor_leaf_indices;
  const auto& far_indices = temporary_interactions
      ? temporary_interactions->far : target_hierarchy_cache.far_hierarchy_indices;
  for (int ele_idx : near_indices) {
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
        far_indices,
        hElement_cache,
        expansion_order,
        coefficient_workspace);
  return approximation;
}

#ifdef TREECODE_REUSE_EXPERIMENT
int run_reuse_sequence_experiment(
    const ProgramOptions& options,
    double reusable_structure_seconds,
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
  double previous_amplitude = options.sequence_initial_amplitude;
  for (int step = 0; step < options.sequence_steps; ++step) {
    const double amplitude =
        options.sequence_initial_amplitude
        * std::pow(options.sequence_decay, step);
    reuse_sequence_amplitude = amplitude;

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
              << ", T_update=" << density_seconds
              << " s, T_moment=" << moment_seconds
              << " s, T_eval=" << evaluation_seconds
              << " s, L2_error=" << error
              << ", relative_error=" << relative_error << '\n';
    previous_amplitude = amplitude;
  }

  const double reused_total_seconds =
      reusable_structure_seconds + sequence_update_seconds;
  const double rebuild_every_step_estimate =
      options.sequence_steps * reusable_structure_seconds
      + sequence_update_seconds;
  const double saved_seconds =
      rebuild_every_step_estimate - reused_total_seconds;
  const double speedup = rebuild_every_step_estimate
      / std::max(reused_total_seconds,
                 std::numeric_limits<double>::min());
  output << "# reusable_structure_time_s\t"
         << reusable_structure_seconds << '\n'
         << "# reused_total_time_s\t" << reused_total_seconds << '\n'
         << "# rebuild_every_step_estimated_time_s\t"
         << rebuild_every_step_estimate << '\n'
         << "# saved_time_s\t" << saved_seconds << '\n'
         << "# estimated_speedup\t" << speedup << '\n';
  output.close();

  std::cout << std::setprecision(9)
            << "\nReuse experiment summary\n"
            << "  One-time reusable structure preparation: "
            << reusable_structure_seconds << " s\n"
            << "  Sum of density/moment/evaluation updates: "
            << sequence_update_seconds << " s\n"
            << "  Reuse strategy total: " << reused_total_seconds << " s\n"
            << "  Rebuild-every-step estimate: "
            << rebuild_every_step_estimate << " s\n"
            << "  Estimated saved time: " << saved_seconds << " s\n"
            << "  Estimated speedup: " << speedup << "x\n"
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
        IndicatorMode indicator_mode,
        bool use_gpu_backend)
  {
    ind.reinit(mesh);
    const int n_element = mesh.n_geometry(DIM);
#ifdef TREECODE_USE_CUDA
    if (use_gpu_backend && indicator_mode == IndicatorMode::radial) {
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
    if (use_gpu_backend) {
      std::cout
          << indicator_mode_name(indicator_mode)
          << " refinement indicator is evaluated on the CPU; "
             "the requested GPU backend is retained for treecode evaluation\n";
    }
#else
    if (use_gpu_backend && indicator_mode == IndicatorMode::radial) {
      throw std::runtime_error(
          "GPU refinement indicator requested in a CPU-only executable");
    }
#endif

    // The CPU backend keeps the complete refinement path on OpenMP/CPU.
    #pragma omp parallel for schedule(static) default(none) \
        shared(ind, mesh, fem_space) \
        firstprivate(n_element, quadrature_accuracy, indicator_mode)
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
      if (indicator_mode == IndicatorMode::radial ||
          indicator_mode == IndicatorMode::gaussian_radial) {
        double radial_indicator = 0.0;
        for (const auto& entry : rule) {
#ifndef TREECODE_WATER10_MODEL
          const double x_squared = entry.point[0] * entry.point[0];
          const double y_squared = entry.point[1] * entry.point[1];
          const double z_squared = entry.point[2] * entry.point[2];
          const double radius_squared =
              indicator_mode == IndicatorMode::gaussian_radial
                  ? x_squared + 2.0 * y_squared + 3.0 * z_squared
                  : x_squared + y_squared + z_squared;
#endif
#ifdef TREECODE_WATER10_MODEL
          radial_indicator += entry.weight * treecode_test_problem::radial_monitor(
              entry.point[0], entry.point[1], entry.point[2]);
#else
          radial_indicator +=
              entry.weight
              / std::sqrt(std::max(radius_squared, 1.0e-30));
#endif
        }
        ind[the_geo_idx] = radial_indicator;
      } else if (indicator_mode == IndicatorMode::density_variance) {
        double volume = 0.0;
        double density_integral = 0.0;
        for (const auto& entry : rule) {
          const double density = treecode_test_problem::source_value(
              entry.point[0], entry.point[1], entry.point[2]);
          volume += entry.weight;
          density_integral += entry.weight * density;
        }
        if (!(volume > 0.0)) {
          ind[the_geo_idx] = 0.0;
          continue;
        }
        const double density_mean = density_integral / volume;
        double variance_integral = 0.0;
        for (const auto& entry : rule) {
          const double density = treecode_test_problem::source_value(
              entry.point[0], entry.point[1], entry.point[2]);
          const double difference = density - density_mean;
          variance_integral += entry.weight * difference * difference;
        }
        ind[the_geo_idx] =
            std::sqrt(std::max(0.0, variance_integral));
      } else {
        double diameter_squared = 0.0;
        for (std::size_t first = 0; first < vertices.size(); ++first) {
          for (std::size_t second = first + 1;
               second < vertices.size(); ++second) {
            double distance_squared = 0.0;
            for (int dimension = 0; dimension < DIM; ++dimension) {
              const double difference =
                  vertices[first][dimension] - vertices[second][dimension];
              distance_squared += difference * difference;
            }
            diameter_squared = std::max(diameter_squared, distance_squared);
          }
        }
        double curvature_squared_integral = 0.0;
        for (const auto& entry : rule) {
          curvature_squared_integral +=
              entry.weight
              * treecode_test_problem::exact_hessian_frobenius_squared(
                    entry.point[0], entry.point[1], entry.point[2]);
        }
        ind[the_geo_idx] =
            diameter_squared
            * std::sqrt(std::max(0.0, curvature_squared_integral));
      }
    }
  }

double exact_gaussian_mesh_estimator(
    RegularMesh<DIM>& mesh,
    FEMSpace<double, DIM>& fem_space,
    int quadrature_accuracy)
{
  const int n_element = mesh.n_geometry(DIM);
  double estimator_squared = 0.0;

  #pragma omp parallel for schedule(static) reduction(+:estimator_squared) \
      default(none) shared(mesh, fem_space) \
      firstprivate(n_element, quadrature_accuracy)
  for (int i = 0; i < n_element; ++i) {
    Element<double, DIM>& element = fem_space.element(i);
    GeometryBM& geometry = element.geometry();
    std::vector<AFEPack::Point<DIM>> vertices(
        static_cast<std::size_t>(geometry.n_vertex()));
    for (int vertex = 0; vertex < geometry.n_vertex(); ++vertex) {
      vertices[static_cast<std::size_t>(vertex)] =
          mesh.point()[geometry.vertex()[vertex]];
    }
    const auto rule = treecode_quadrature::build_physical_rule(
        vertices, quadrature_accuracy);

    double volume = 0.0;
    double exact_integral = 0.0;
    for (const auto& entry : rule) {
      const double exact = treecode_test_problem::exact_value(
          entry.point[0], entry.point[1], entry.point[2]);
      volume += entry.weight;
      exact_integral += entry.weight * exact;
    }
    if (!(volume > 0.0)) continue;
    const double exact_mean = exact_integral / volume;
    double element_estimator_squared = 0.0;
    for (const auto& entry : rule) {
      const double exact = treecode_test_problem::exact_value(
          entry.point[0], entry.point[1], entry.point[2]);
      const double difference = exact - exact_mean;
      element_estimator_squared +=
          entry.weight * difference * difference;
    }
    estimator_squared += std::max(0.0, element_estimator_squared);
  }
  return std::sqrt(std::max(0.0, estimator_squared));
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

  // Pack the hierarchy and the HNL cover. CUDA classifies those candidates
  // with MAC and aggregates moments bottom-up.
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
  problem.hnl_candidate_offsets.reserve(static_cast<std::size_t>(leaf_count) + 1);
  for (int leaf = 0; leaf < leaf_count; ++leaf) {
    problem.hnl_candidate_offsets.push_back(checked_gpu_index(
        problem.hnl_candidate_indices.size(), "HNL candidate count"));
    const auto& candidates = hierarchy_cache[
        problem.leaf_hierarchy_indices[leaf]].hnl_candidate_indices;
    if (candidates.empty())
      throw std::runtime_error("CUDA requires geometric HNL candidates before MAC");
    problem.hnl_candidate_indices.insert(problem.hnl_candidate_indices.end(),
                                         candidates.begin(), candidates.end());
  }
  problem.hnl_candidate_offsets.push_back(checked_gpu_index(
      problem.hnl_candidate_indices.size(), "HNL candidate count"));
  return problem;
}
#endif



#ifdef TREECODE_ERROR_VALIDATION
#include "error_validation.hpp"
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
  const bool use_gpu_backend = options.backend == ComputeBackend::gpu;
  bool stream_cpu_interactions = false;
  bool family_mac = false;
  bool shared_hnl_targets = false;
  if (const char* flag = std::getenv("TREECODE_HNL_MAC_FAMILIES")) {
    const std::string value(flag);
    if (value != "0" && value != "1" && !value.empty()) {
      std::cerr << "TREECODE_HNL_MAC_FAMILIES must be 0 or 1\n";
      return 1;
    }
#ifndef TREECODE_ROOT_BASED_LISTS
    family_mac = value == "1";
#endif
  }
  if (family_mac && use_gpu_backend) {
    std::cerr << "Experimental HNL family MAC currently supports CPU only\n";
    return 1;
  }
  if (const char* flag = std::getenv("TREECODE_HNL_SHARED_TARGETS")) {
    const std::string value(flag);
    if (value != "0" && value != "1" && !value.empty()) {
      std::cerr << "TREECODE_HNL_SHARED_TARGETS must be 0 or 1\n";
      return 1;
    }
    shared_hnl_targets = value == "1";
  }
#ifdef TREECODE_ROOT_BASED_LISTS
  if (shared_hnl_targets) {
    std::cerr << "TREECODE_HNL_SHARED_TARGETS is available only for HNL\n";
    return 1;
  }
#endif
  if (shared_hnl_targets && (family_mac || use_gpu_backend)) {
    std::cerr << "Shared-target HNL requires CPU and family packing OFF\n";
    return 1;
  }
  std::cout << "HNL MAC family packing: " << (family_mac ? "ON" : "OFF") << '\n';
  if (const char* mode = std::getenv("TREECODE_CPU_INTERACTIONS")) {
    const std::string value(mode);
    if (value == "STREAM") stream_cpu_interactions = true;
    else if (!value.empty() && value != "CACHED") {
      std::cerr << "TREECODE_CPU_INTERACTIONS must be CACHED or STREAM\n";
      return 1;
    }
  }
  if (stream_cpu_interactions) {
#if !defined(TREECODE_COMPARE_EXACT) || defined(TREECODE_REUSE_EXPERIMENT) || defined(TREECODE_HNL_VISUALIZE)
    std::cerr << "STREAM currently supports the standalone exact-comparison driver only\n";
    return 1;
#endif
    if (use_gpu_backend || !options.compare_solution) {
      std::cerr << "STREAM requires CPU and compare=1 (one potential per active leaf)\n";
      return 1;
    }
  }
  if (shared_hnl_targets && !stream_cpu_interactions) {
    std::cerr << "Shared-target HNL currently requires TREECODE_CPU_INTERACTIONS=STREAM\n";
    return 1;
  }
  std::cout << "HNL target sharing: "
            << (shared_hnl_targets ? "SIBLING_PARENT" : "OFF") << '\n';
  if (!use_gpu_backend)
    std::cout << "CPU interaction mode: "
              << (stream_cpu_interactions ? "STREAM" : "CACHED") << '\n';
#ifdef TREECODE_INDICATOR_ACCURACY
  static_assert(TREECODE_INDICATOR_ACCURACY >= 1 &&
                    TREECODE_INDICATOR_ACCURACY <=
                        treecode_quadrature::maximum_algebraic_accuracy,
                "Frozen indicator accuracy is outside the supported range");
  const int indicator_quadrature_accuracy = TREECODE_INDICATOR_ACCURACY;
#else
  const int indicator_quadrature_accuracy = options.indicator_accuracy;
#endif
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
            << ", indicator_mode="
            << indicator_mode_name(options.indicator_mode)
            << ", indicator_order="
            << indicator_quadrature_accuracy
            << ", mesh_estimator="
            << (options.report_mesh_estimator ? "enabled" : "disabled")
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

  std::cout << "Refinement indicator: "
            << indicator_mode_name(options.indicator_mode)
            << ", fixed algebraic accuracy="
            << indicator_quadrature_accuracy << '\n';

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
        indicator, _r_mesh, fem_space, indicator_quadrature_accuracy,
        options.indicator_mode,
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
  r_mesh1.writeOpenDXData("D.dx");
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
        indicator1, r_mesh1, fem_space, indicator_quadrature_accuracy,
        options.indicator_mode,
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
  r_mesh2.writeOpenDXData("D_uniform.dx");
#else
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

#ifndef TREECODE_UNIFORM_MESH
  if (options.report_mesh_estimator) {
    const double estimator_start = omp_get_wtime();
    const double mesh_estimator = exact_gaussian_mesh_estimator(
        r_mesh2, fem_space, indicator_quadrature_accuracy);
    std::cout << "Exact Gaussian mesh-resolution estimator: "
              << std::setprecision(15) << mesh_estimator
              << " (indicator order=" << indicator_quadrature_accuracy
              << ", time=" << (omp_get_wtime() - estimator_start)
              << " seconds)\n";
  }
#endif

  std::vector<hElementCache> hElement_cache;
  std::map<HElement<DIM>*, int> map_HElement_to_int;  
  StreamingInteractionContext streaming_context;
  streaming_context.shared_targets = shared_hnl_targets;
  const bool construct_cpu_interactions = !use_gpu_backend;
  double mapping_geometry_seconds = 0.0;
  double hierarchy_validation_seconds = 0.0;
  int omp_threads_used = 1;
  double neighbor_fill_seconds = update_hElementCache(
      *ir_mesh,
      hElement_cache,
      map_HElement_to_int,
      options.mac_theta,
      construct_cpu_interactions,
      &mapping_geometry_seconds,
      &hierarchy_validation_seconds,
      &omp_threads_used,
      stream_cpu_interactions ? &streaming_context : nullptr,
      family_mac);
#ifndef TREECODE_COMPARE_EXACT
  (void) neighbor_fill_seconds;
#endif
  
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
  const double reusable_structure_seconds =
      omp_get_wtime() - program_start_time;
  return run_reuse_sequence_experiment(
      options,
      reusable_structure_seconds,
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
  if (!use_gpu_backend) {
    double total_source_charge = 0.0;
    for (HElement<DIM>* root_element : root_element_list) {
      const auto root_entry = map_HElement_to_int.find(root_element);
      if (root_entry == map_HElement_to_int.end()) {
        throw std::runtime_error(
            "Root element missing from hierarchy map while summing charge");
      }
      const auto& moments = hElement_cache[root_entry->second].moments;
      if (moments.empty()) {
        throw std::runtime_error(
            "Root multipole moment is empty while summing charge");
      }
      total_source_charge += moments[0];
    }
    std::cout << "Integrated source charge: "
              << std::setprecision(15) << total_source_charge << '\n';
  }

#ifdef TREECODE_ERROR_VALIDATION
  if (use_gpu_backend || stream_cpu_interactions || family_mac || shared_hnl_targets)
    throw std::runtime_error("Error validation requires CPU CACHED per-leaf HNL");
  return run_article_error_validation(
      options, root_element_list, hElement_cache, element_cache,
      map_HElement_to_int);
#endif

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
        neighbor_fill_seconds += gpu_result.interaction_seconds;
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
  neighbor_fill_seconds += gpu_result.interaction_seconds;
  std::cout << "Filling neighbor_elements uses: " << neighbor_fill_seconds
            << " seconds (CPU HNL + GPU MAC)\n";
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
  const double mac_theta = options.mac_theta;
  const double potential_evaluation_start = omp_get_wtime();
#ifndef TREECODE_ROOT_BASED_LISTS
  if (stream_cpu_interactions && streaming_context.shared_targets) {
    const double shared_cover_start = omp_get_wtime();
    streaming_context.target_groups = treecode_hnl::group_targets_by_ancestor(
        streaming_context.topology, leaf_hierarchy_indices, 1);
    const std::size_t group_count =
        streaming_context.target_groups.anchors.size();
    if (streaming_context.target_groups.group_of_target.size()
            != adaptive_element_count
        || std::find(streaming_context.target_groups.group_of_target.begin(),
                     streaming_context.target_groups.group_of_target.end(), -1)
            != streaming_context.target_groups.group_of_target.end()) {
      std::cerr << "Error: failed to assign every active target to a shared HNL group\n";
      return 1;
    }
    streaming_context.shared_candidates.resize(group_count);
    streaming_context.shared_counters.resize(group_count);
    streaming_context.shared_layer_counts.resize(group_count);
    #pragma omp parallel
    {
      treecode_hnl::Workspace workspace(streaming_context.topology);
      #pragma omp for schedule(dynamic, 16)
      for (std::size_t group = 0; group < group_count; ++group) {
        const auto& candidates = treecode_hnl::build_candidates(
            streaming_context.topology,
            streaming_context.target_groups.anchors[group], workspace);
        streaming_context.shared_candidates[group] = candidates;
        streaming_context.shared_counters[group] =
            static_cast<const treecode_hnl::LayerCounters&>(workspace);
        streaming_context.shared_layer_counts[group] = workspace.layers;
      }
    }
    std::size_t maximum_group_size = 0;
    for (const auto& targets : streaming_context.target_groups.target_positions)
      maximum_group_size = std::max(maximum_group_size, targets.size());
    const double shared_cover_seconds = omp_get_wtime() - shared_cover_start;
    std::cout << "Shared-HNL target groups: " << group_count
              << " for " << adaptive_element_count
              << " leaves, average targets/group="
              << (group_count ? static_cast<double>(adaptive_element_count) / group_count : 0.0)
              << ", maximum=" << maximum_group_size << '\n'
              << "Shared-HNL cover construction time: " << shared_cover_seconds
              << " seconds\n";
  }
#endif
  #pragma omp parallel default(none) \
      shared(read_only_element_cache, read_only_hierarchy_cache, streaming_context) \
      firstprivate(adaptive_element_count, expansion_order, \
                   quadrature_accuracy, stream_cpu_interactions, mac_theta) \
      reduction(+:exact_error_squared,exact_norm_squared, \
                  approximation_norm_squared,evaluation_count, \
                  nonfinite_count)
  {
  InteractionWorkspace interaction_workspace;
#ifndef TREECODE_ROOT_BASED_LISTS
  std::unique_ptr<treecode_hnl::Workspace> hnl_workspace;
  treecode_hnl::FamilyPackingWorkspace family_workspace;
  if (stream_cpu_interactions && !streaming_context.shared_targets)
    hnl_workspace = std::make_unique<treecode_hnl::Workspace>(streaming_context.topology);
#endif
  // build_taylor_coefficients overwrites every coefficient it uses. Reuse one
  // workspace per worker instead of zeroing the maximum-order array per leaf.
  std::array<double, treecode_multipole::maximum_coefficient_count> coefficient_workspace;
  #pragma omp for schedule(dynamic, 16)
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
    if (stream_cpu_interactions) {
      const int target_index = target_cache.hierarchy_index;
      auto& stats = streaming_context.statistics[static_cast<std::size_t>(target_index)];
#ifdef TREECODE_ROOT_BASED_LISTS
      const auto& candidates = streaming_context.roots;
#else
      const std::vector<int>* original_candidate_pointer = nullptr;
      if (streaming_context.shared_targets) {
        const int group = streaming_context.target_groups.group_of_target[element_index];
        if (group < 0 || static_cast<std::size_t>(group)
                            >= streaming_context.shared_candidates.size()) {
          stats.unique_entries = false;
          ++nonfinite_count;
          continue;
        }
        original_candidate_pointer =
            &streaming_context.shared_candidates[static_cast<std::size_t>(group)];
        const auto& counters =
            streaming_context.shared_counters[static_cast<std::size_t>(group)];
        stats.hnl_candidates = original_candidate_pointer->size();
        stats.hnl_layers =
            streaming_context.shared_layer_counts[static_cast<std::size_t>(group)];
        stats.hnl_star_entries = counters.star_entries;
      } else {
        original_candidate_pointer = &treecode_hnl::build_candidates(
            streaming_context.topology, target_index, *hnl_workspace);
        stats.hnl_candidates = original_candidate_pointer->size();
        stats.hnl_layers = hnl_workspace->layers;
        stats.hnl_star_entries = hnl_workspace->star_entries;
      }
      const auto& original_candidates = *original_candidate_pointer;
#ifdef TREECODE_HNL_VERIFY
      if (!validate_hnl_cover(original_candidates, read_only_hierarchy_cache,
                             streaming_context.hierarchy.leaf_count)) {
        stats.unique_entries = false;
        ++nonfinite_count;
        continue;
      }
#endif
      const auto& candidates = streaming_context.family_mac
          ? treecode_hnl::pack_complete_sibling_families(
                streaming_context.topology, *hnl_workspace, family_workspace)
          : original_candidates;
#endif
#ifdef TREECODE_HNL_VERIFY
      if (!validate_hnl_cover(candidates, read_only_hierarchy_cache,
                              streaming_context.hierarchy.leaf_count)) {
        stats.unique_entries = false;
        ++nonfinite_count;
        continue;
      }
#endif
      classify_source_cover_to_workspace(target_index, candidates,
          streaming_context.geometry, mac_theta, stats, interaction_workspace);
#ifdef TREECODE_HNL_VERIFY
      std::vector<int> final_cover = interaction_workspace.far;
      for (int leaf : interaction_workspace.near)
        final_cover.push_back(streaming_context.leaf_hierarchy[leaf]);
      stats.unique_entries = validate_hnl_cover(final_cover, read_only_hierarchy_cache,
                                               streaming_context.hierarchy.leaf_count);
#endif
    }
    const AFEPack::Point<DIM>& target = target_hierarchy_cache.ref_point;
    const double approximation = multipole_approximation(
        target,
        target_element,
        target_hierarchy_cache,
        read_only_hierarchy_cache,
        read_only_element_cache,
        expansion_order,
        quadrature_accuracy,
        coefficient_workspace.data(), -1,
        stream_cpu_interactions ? &interaction_workspace : nullptr);
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
  }
  potential_evaluation_seconds =
      omp_get_wtime() - potential_evaluation_start;
  if (stream_cpu_interactions) {
#ifdef TREECODE_HNL_VERIFY
    constexpr bool verify_stream = true;
#else
    constexpr bool verify_stream = false;
#endif
    validate_interaction_stats(streaming_context.elements, streaming_context.statistics,
                               streaming_context.hierarchy, mac_theta, verify_stream);
#ifdef TREECODE_HNL_VERIFY
    std::cout << "Streaming raw-cover and final-partition audits: invalid covers=0\n";
#endif
    std::cout << "CPU streamed target-loop time: " << potential_evaluation_seconds
              << " seconds\n"
              << "CPU interaction construction and evaluation time: "
              << streaming_context.preparation_seconds + potential_evaluation_seconds
              << " seconds\n";
  }
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
#ifdef TREECODE_WATER10_MODEL
            << "Reference: full-space sum q*erf(sqrt(alpha)*r)/r; numerical source truncated to mesh box\n"
#elif defined(TREECODE_POSITIVE_GAUSSIAN)
            << "Exact solution: erf(sqrt(alpha)*r)/r, alpha="
            << treecode_test_problem::gaussian_alpha << "\n"
#else
            << "Exact solution: exp(-" << treecode_test_problem::gaussian_a
            << "*x^2-" << treecode_test_problem::gaussian_b
            << "*y^2-" << treecode_test_problem::gaussian_c << "*z^2)\n"
#endif
            << "L2 norm of exact solution: " << exact_norm << '\n'
            << "L2 norm of treecode solution: "
            << approximation_norm << '\n'
            << "L2 error against exact solution: "
            << exact_error << '\n'
            << "Relative L2 error against exact solution: "
            << relative_exact_error << '\n'
            << "Exact-comparison evaluation time: "
            << exact_evaluation_seconds << " seconds\n";
  if (stream_cpu_interactions)
    std::cout << "Potential evaluation time: not isolated (STREAM includes HNL/MAC)\n";
  else
    std::cout << "Potential evaluation time: "
              << potential_evaluation_seconds << " seconds\n";

  const double total_execution_seconds =
      omp_get_wtime() - program_start_time;
  if (!append_timing_result(
          options,
          static_cast<std::size_t>(fem_space.n_element()),
          mapping_geometry_seconds,
          hierarchy_validation_seconds,
          !stream_cpu_interactions,
          neighbor_fill_seconds,
          total_execution_seconds,
          !stream_cpu_interactions,
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
#ifdef TREECODE_WATER10_MODEL
  // Optional independent validation, outside all reported solver timings.
  // Cached MAC lists are conservative over the entire target leaf.
  if (const char* grid_env = std::getenv("WATER10_VALIDATION_GRID")) {
    const int m = parse_int(grid_env, "WATER10_VALIDATION_GRID");
    if (m < 2 || m > 128 || stream_cpu_interactions)
      throw std::runtime_error("Common validation requires CACHED mode and grid 2..128");
    const char* padding_env = std::getenv("WATER10_PADDING");
    const double padding = padding_env ? parse_double(padding_env,"padding") : 4.0;
    const double validation_start = omp_get_wtime();
    double err = 0.0, norm = 0.0;
    int failures = 0;
    const int count = m*m*m;
    #pragma omp parallel for schedule(dynamic,32) reduction(+:err,norm,failures)
    for (int k=0; k<count; ++k) {
      AFEPack::Point<DIM> point;
      const int cell[3] = {k%m,(k/m)%m,k/(m*m)};
      // Fixed stratified jitter avoids mesh faces; independent of source mesh.
      unsigned int seed = static_cast<unsigned int>(k)+7351u;
      for(int d=0;d<3;++d) {
        seed=1664525u*seed+1013904223u;
        const double offset=(static_cast<double>(seed)+.5)/4294967296.0;
        const double half=treecode_test_problem::halfwidth[d]+padding;
        point[d]=-half+2*half*(cell[d]+offset)/m;
      }
      HElement<DIM>* leaf=findLeafContainingPoint(root_element_list,point);
      if(!leaf) { ++failures; continue; }
      const auto& cache=hElement_cache[map_HElement_to_int.at(leaf)];
      std::array<double,treecode_multipole::maximum_coefficient_count> workspace{};
      const double value=multipole_approximation(point,leaf,cache,hElement_cache,
          element_cache,multipole_order,options.quadrature_accuracy,workspace.data());
      const double exact=u(point);
      if(!std::isfinite(value)) { ++failures; continue; }
      err+=(value-exact)*(value-exact); norm+=exact*exact;
    }
    if(failures) throw std::runtime_error("Common validation contains missing/nonfinite targets");
    std::cout << "Common validation targets: " << count << '\n'
              << "Common validation relative L2 error: " << std::sqrt(err/norm) << '\n'
              << "Common validation time (excluded): " << omp_get_wtime()-validation_start << '\n';
  }
#endif
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
