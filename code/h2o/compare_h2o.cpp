// Standalone H2O three-center Gaussian treecode.  This translation unit intentionally
// does not include compare_exact.cpp, so molecule-specific mesh adaptation,
// interaction lists, and comparison changes remain isolated here.
#define TREECODE_H2O_THREE_CENTER_MODEL 1

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
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include "hnl_candidates.hpp"

/// for mesh
#include <AFEPack/HGeometry.h>
/// for template element
#include <AFEPack/TemplateElement.h>
/// for space
#include <AFEPack/DGFEMSpace.h>

#include "local_quadrature.hpp"
#include "cartesian_multipole.hpp"
#include "gaussian_test_problem.hpp"
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
#include "h2o_three_center_test_problem.hpp"
#endif

#ifdef TREECODE_USE_CUDA
#include "compare_gpu.hpp"
#endif


constexpr int DIM = 3;
constexpr double PI = treecode_test_problem::pi;
constexpr int DEFAULT_INITIAL_REFINEMENTS = 2;
constexpr int DEFAULT_QUADRATURE_ACCURACY = 8;
constexpr int DEFAULT_EXPANSION_ORDER = 4;
constexpr double DEFAULT_MAC_THETA = 0.5;

enum class ComputeBackend
{
  cpu,
  gpu
};

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
enum class H2OMeshMode
{
  three,
  single,
  both
};

enum class H2ONearFieldMode
{
  quadrature,
  one_point
};

enum class H2OIndicatorMode
{
  radial,
  gaussian_radial,
  moment_defect
};

enum class H2OSingularMode
{
  duffy,
  core_analytic,
  analytic_all
};
#endif

const char* backend_name(ComputeBackend backend)
{
  return backend == ComputeBackend::cpu ? "CPU" : "GPU";
}

const char* timing_results_filename()
{
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  return "results_h2o.txt";
#else
  return "results.txt";
#endif
}

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
const char* h2o_near_field_mode_name(H2ONearFieldMode mode)
{
  return mode == H2ONearFieldMode::quadrature
      ? "QUADRATURE" : "ONE_POINT";
}

const char* h2o_mesh_mode_name(H2OMeshMode mode)
{
  if (mode == H2OMeshMode::single) return "SINGLE";
  if (mode == H2OMeshMode::both) return "BOTH";
  return "THREE";
}

const char* h2o_indicator_mode_name(H2OIndicatorMode mode)
{
  if (mode == H2OIndicatorMode::moment_defect) return "MOMENT_DEFECT";
  if (mode == H2OIndicatorMode::gaussian_radial) {
    return "GAUSSIAN_RADIAL";
  }
  return "RADIAL";
}

const char* h2o_singular_mode_name(H2OSingularMode mode)
{
  if (mode == H2OSingularMode::duffy) return "DUFFY";
  if (mode == H2OSingularMode::analytic_all) return "ANALYTIC_ALL";
  return "CORE_ANALYTIC";
}
#endif

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
void print_h2o_timing_summary(
    const char* mesh_mode,
    double list_seconds,
    double preparation_seconds,
    double evaluation_seconds,
    double total_seconds)
{
  std::cout << std::setprecision(12)
            << "\n============================================================\n"
            << "H2O TREECODE TIMING SUMMARY [" << mesh_mode << "]\n"
            << "1. Neighbor/non-neighbor list preparation time : "
            << list_seconds << " s\n"
            << "2. Multipole information and other preparation : "
            << preparation_seconds << " s\n"
            << "3. Final near-field and far-field evaluation    : "
            << evaluation_seconds << " s\n"
            << "4. Total algorithm time                         : "
            << total_seconds << " s\n"
            << "   Check (1 + 2 + 3)                            : "
            << list_seconds + preparation_seconds + evaluation_seconds
            << " s\n"
            << "============================================================\n";
}
#endif

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
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  H2OMeshMode h2o_mesh_mode = H2OMeshMode::three;
  H2ONearFieldMode h2o_near_field_mode = H2ONearFieldMode::quadrature;
  double adaptation_tolerance = 1.0e-3;
  H2OIndicatorMode h2o_indicator_mode = H2OIndicatorMode::moment_defect;
  H2OSingularMode h2o_singular_mode = H2OSingularMode::core_analytic;
  double h2o_core_radius = 0.25;
#endif
};

bool append_timing_result(
    const ProgramOptions& options,
    int mesh_count,
    std::size_t element_count,
    double list_preparation_seconds,
    double multipole_other_preparation_seconds,
    double near_far_evaluation_seconds,
    double total_seconds,
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
    write_field("mesh_count", 12);
    write_field("near_field_mode", 18);
    write_field("n_elements", 12);
    write_field("list_preparation_time_s", 26);
    write_field("multipole_other_preparation_time_s", 38);
    write_field("near_far_evaluation_time_s", 30);
    write_field("total_time_s", 20);
    write_field("L2_error_exact", 24);
    write_field("backend", 12);
    write_field("omp_threads_used", 20);
    write_field("adapt_tolerance", 20);
    results_file << "source_partition\n";
  } else if (leading_newline_required) {
    results_file << '\n';
  }

  results_file << std::setprecision(15);
  write_field(options.quadrature_accuracy, 4);
  write_field(options.expansion_order, 4);
  write_field(options.mac_theta, 8);
  write_field(mesh_count, 12);
  write_field(h2o_near_field_mode_name(options.h2o_near_field_mode), 18);
  write_field(element_count, 12);
  write_field(list_preparation_seconds, 26);
  write_field(multipole_other_preparation_seconds, 38);
  write_field(near_far_evaluation_seconds, 30);
  write_field(total_seconds, 20);
  if (has_l2_error) {
    write_field(l2_error, 24);
  } else {
    write_field("NA", 24);
  }
  write_field(backend_name(options.backend), 12);
  write_field(omp_threads_used, 20);
  write_field(options.adaptation_tolerance, 20);
  results_file << (mesh_count == 3 ? "BECKE" : "TOTAL_SOURCE")
               << '_' << h2o_indicator_mode_name(options.h2o_indicator_mode)
               << '_' << h2o_singular_mode_name(options.h2o_singular_mode);
  if (options.h2o_singular_mode == H2OSingularMode::core_analytic) {
    results_file << "_R" << options.h2o_core_radius;
  }
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

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
H2OMeshMode parse_h2o_mesh_mode(const char* text)
{
  std::string value(text);
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  if (value == "THREE") return H2OMeshMode::three;
  if (value == "SINGLE") return H2OMeshMode::single;
  if (value == "BOTH") return H2OMeshMode::both;
  throw std::invalid_argument(
      "Invalid H2O mesh mode: " + value
      + " (use THREE, SINGLE, or BOTH)");
}

H2ONearFieldMode parse_h2o_near_field_mode(const char* text)
{
  std::string value(text);
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  if (value == "QUADRATURE") return H2ONearFieldMode::quadrature;
  if (value == "ONE_POINT") return H2ONearFieldMode::one_point;
  throw std::invalid_argument(
      "Invalid H2O near-field mode: " + value
      + " (use QUADRATURE or ONE_POINT)");
}

H2OIndicatorMode parse_h2o_indicator_mode(const char* text)
{
  std::string value(text);
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  if (value == "RADIAL") return H2OIndicatorMode::radial;
  if (value == "GAUSSIAN_RADIAL" || value == "ELLIPSOIDAL_RADIAL") {
    return H2OIndicatorMode::gaussian_radial;
  }
  if (value == "MOMENT" || value == "MOMENT_DEFECT") {
    return H2OIndicatorMode::moment_defect;
  }
  throw std::invalid_argument(
      "Invalid H2O indicator mode: " + value
      + " (use RADIAL, GAUSSIAN_RADIAL, or MOMENT_DEFECT)");
}

H2OSingularMode parse_h2o_singular_mode(const char* text)
{
  std::string value(text);
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
  if (value == "DUFFY") return H2OSingularMode::duffy;
  if (value == "CORE" || value == "CORE_ANALYTIC") {
    return H2OSingularMode::core_analytic;
  }
  if (value == "ALL" || value == "ANALYTIC_ALL") {
    return H2OSingularMode::analytic_all;
  }
  throw std::invalid_argument(
      "Invalid H2O singular mode: " + value
      + " (use DUFFY, CORE_ANALYTIC, or ANALYTIC_ALL)");
}
#endif

ProgramOptions parse_options(int argc, char* argv[])
{
#ifdef TREECODE_COMPARE_EXACT
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  if (argc < 2 || argc > 14) {
    throw std::invalid_argument(
        "Usage: tree_h2o MESH_FILE [INITIAL_REFINEMENTS] "
        "[QUADRATURE_ACCURACY] [EXPANSION_ORDER] [MAC_THETA] "
        "[COMPARE] [BACKEND] [MESH_MODE] [NEAR_FIELD_MODE] "
        "[ADAPT_TOLERANCE] [INDICATOR_MODE] [SINGULAR_MODE] "
        "[CORE_RADIUS]\n"
        "       MESH_MODE is THREE, SINGLE, or BOTH; default: THREE\n"
        "       NEAR_FIELD_MODE is QUADRATURE or ONE_POINT; "
        "default: QUADRATURE\n"
        "       INDICATOR_MODE is RADIAL, GAUSSIAN_RADIAL, "
        "or MOMENT_DEFECT; "
        "default: MOMENT_DEFECT\n"
        "       SINGULAR_MODE is DUFFY, CORE_ANALYTIC, or ANALYTIC_ALL; "
        "default: CORE_ANALYTIC\n"
        "       CORE_RADIUS is measured in bohr; default: 0.25");
  }
#else
  if (argc < 2 || argc > 8) {
    throw std::invalid_argument(
        "Usage: tree_exact MESH_FILE [INITIAL_REFINEMENTS] "
        "[QUADRATURE_ACCURACY] [EXPANSION_ORDER] [MAC_THETA] "
        "[COMPARE] [BACKEND]\n"
        "       BACKEND is CPU or GPU; COMPARE and BACKEND may be swapped");
  }
#endif
#else
  if (argc < 2 || argc > 9) {
    throw std::invalid_argument(
        "Usage: tree MESH_FILE [INITIAL_REFINEMENTS] [QUADRATURE_ACCURACY] "
        "[EXPANSION_ORDER] [MAC_THETA] [REFERENCE_FILE] [COMPARE] "
        "[BACKEND]");
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
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  if (argc >= 9) {
    options.h2o_mesh_mode = parse_h2o_mesh_mode(argv[8]);
  }
  if (argc >= 10) {
    options.h2o_near_field_mode = parse_h2o_near_field_mode(argv[9]);
  }
  if (argc >= 11) {
    options.adaptation_tolerance =
        parse_double(argv[10], "adaptive-refinement tolerance");
  }
  if (argc >= 12) {
    options.h2o_indicator_mode = parse_h2o_indicator_mode(argv[11]);
  }
  if (argc >= 13) {
    options.h2o_singular_mode = parse_h2o_singular_mode(argv[12]);
  }
  if (argc >= 14) {
    options.h2o_core_radius =
        parse_double(argv[13], "atom-core radius");
  }
#endif
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
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  if (!(options.adaptation_tolerance > 0.0)) {
    throw std::invalid_argument("ADAPT_TOLERANCE must be positive");
  }
  if (!(options.h2o_core_radius >= 0.0)) {
    throw std::invalid_argument("CORE_RADIUS must be nonnegative");
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

double u(const double * p)
{
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  return treecode_h2o_three_center_problem::exact_value(p[0], p[1], p[2]);
#else
  return treecode_test_problem::exact_value(p[0], p[1], p[2]);
#endif
};

double f(const double * p)
{
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  return treecode_h2o_three_center_problem::source_value(p[0], p[1], p[2]);
#else
  return treecode_test_problem::source_value(p[0], p[1], p[2]);
#endif
};

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
double h2o_weighted_source(const double* p, std::size_t atom_index)
{
  return treecode_h2o_three_center_problem::becke_weighted_source_value(
      p[0], p[1], p[2], atom_index);
}
#endif

double selected_source_value(const double* p, int source_component)
{
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  if (source_component >= 0) {
    return h2o_weighted_source(
        p, static_cast<std::size_t>(source_component));
  }
#else
  (void)source_component;
#endif
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

  // Preserve the regular-element geometry.  In a semiregular AFEPack mesh a
  // leaf can be a tetrahedron, twin tetrahedron, or four-tetrahedron macro
  // element.  Cross-mesh H2O targets need these vertices to identify and
  // accurately integrate the source cell containing the target.
  std::vector<AFEPack::Point<DIM>> physical_vertices;

  // H2O one-point finite-volume representation.  Each leaf is represented
  // by its volume-weighted barycenter and one charge |K| rho(x_K).
  double one_point_radius = 0.0;

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
            if(current->value != 0) {  // 非叶子节点
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
  long double represented_volume = 0.0L;
  long double near_volume = 0.0L;
  double maximum_mac_ratio = 0.0;
  bool contains_target_in_near_field = false;
  bool unique_entries = true;
};

void build_mac_interaction_lists(
    int target_index,
    const std::vector<int>& source_candidates,
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    hElementCache& target_cache,
    const std::vector<hElementCache>& hierarchy_cache,
    double theta,
    InteractionListStats& stats)
{
    HElement<DIM>* target =
        hierarchy_elements[static_cast<std::size_t>(target_index)];
    target_cache.neighbor_elements.clear();
    target_cache.nonNeighbor_elements.clear();
    target_cache.neighbor_leaf_indices.clear();
    target_cache.far_hierarchy_indices.clear();
    stats = InteractionListStats{};

    // Classify the disjoint HNL cover, descending only rejected candidates.
    // The caller supplies geometric layers, not the list of tree roots.
    std::vector<int> stack(source_candidates.begin(), source_candidates.end());
    while (!stack.empty()) {
      const int source_index = stack.back();
      stack.pop_back();
      ++stats.visited_hierarchy_nodes;
      if (source_index < 0
          || static_cast<std::size_t>(source_index)
              >= hierarchy_elements.size()) {
        stats.unique_entries = false;
        continue;
      }

      HElement<DIM>* source =
          hierarchy_elements[static_cast<std::size_t>(source_index)];
      const hElementCache& source_cache =
          hierarchy_cache[static_cast<std::size_t>(source_index)];
      const double center_distance =
          (source_cache.ref_point - target_cache.ref_point).length();
      const double minimum_target_distance =
          center_distance - target_cache.radius;
      const bool contains_target =
          source_cache.subtree_leaf_begin
              <= target_cache.subtree_leaf_begin
          && target_cache.subtree_leaf_begin
              < source_cache.subtree_leaf_end;
      const bool mac_accepted =
          !contains_target &&
          minimum_target_distance > 0.0 &&
          source_cache.radius / minimum_target_distance <= theta;

      if (mac_accepted) {
        target_cache.far_hierarchy_indices.push_back(source_index);
        stats.represented_leaf_count += source_cache.subtree_leaf_count;
        stats.represented_volume += source_cache.volume;
        ++stats.far_entries;
        stats.maximum_mac_ratio =
            std::max(
                stats.maximum_mac_ratio,
                source_cache.radius / minimum_target_distance);
      } else if (!source->isRefined()) {
        target_cache.neighbor_leaf_indices.push_back(source->index);
        ++stats.represented_leaf_count;
        stats.represented_volume += source_cache.volume;
        stats.near_volume += source_cache.volume;
        ++stats.near_entries;
        stats.contains_target_in_near_field =
            stats.contains_target_in_near_field || source == target;
      } else {
        for (int child_index : source_cache.child_indices) {
          stack.push_back(child_index);
        }
      }
    }

}

HierarchySummary prepare_hierarchy_aggregates(
    const std::vector<HElement<DIM>*>& elements,
    const std::vector<HElement<DIM>*>& roots,
    std::vector<hElementCache>& hierarchy_cache,
    const std::map<HElement<DIM>*, int>& element_to_index)
{
    std::size_t leaf_count = 0;
    for (HElement<DIM>* element : elements) {
      if (element != nullptr && !element->isRefined()) {
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
      if (element->isRefined()) {
        element_cache.child_indices.reserve(element->child.size());
        for (HElement<DIM>* child : element->child) {
          element_cache.child_indices.push_back(element_to_index.at(child));
        }
      }
      if (!element->isRefined()) {
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

    double root_volume = 0.0;
    for (HElement<DIM>* root : roots) {
      if (root == nullptr || root->parent != nullptr) {
        throw std::runtime_error("Invalid root in hierarchy");
      }
      root_volume += hierarchy_cache[element_to_index.at(root)].volume;
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
    if (std::abs(root_volume - domain_volume) > hierarchy_tolerance) {
      throw std::runtime_error(
          "Root volume does not equal the complete leaf-mesh volume");
    }
    // build_element_mapping stores every parent before its children.  The
    // reverse pass is therefore a postorder pass.  Cache each subtree's leaf
    // count and verify the hierarchy once, in O(number of hierarchy nodes).
    for (std::size_t reverse_index = elements.size();
         reverse_index > 0;
         --reverse_index) {
      HElement<DIM>* element = elements[reverse_index - 1];
      hElementCache& element_cache =
          hierarchy_cache[element_to_index.at(element)];
      if (!element->isRefined()) {
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
      const double tolerance =
          1.0e-10 * std::max(1.0, element_cache.volume);
      if (std::abs(child_volume - element_cache.volume) > tolerance) {
        throw std::runtime_error(
            "Children do not cover their parent element volume");
      }
      element_cache.subtree_leaf_count = child_leaf_count;
      element_cache.subtree_leaf_end = expected_child_begin;
    }

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
    double theta)
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
      if (elements[i]->isRefined()) {
        continue;
      }
      ++validated_target_count;
      const InteractionListStats& stats = interaction_stats[i];
      if (!stats.unique_entries) {
        throw std::runtime_error(
            "MAC traversal generated a null or duplicate interaction");
      }
      if (!stats.contains_target_in_near_field) {
        throw std::runtime_error(
            "MAC traversal failed to include the target in its near field");
      }
      if (stats.represented_leaf_count != hierarchy.leaf_count) {
        throw std::runtime_error(
            "Near/far interaction lists do not represent every source leaf");
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
    }

    if (validated_target_count != hierarchy.leaf_count) {
      throw std::runtime_error(
          "Not every physical leaf received an interaction list");
    }
    const double target_denominator =
        hierarchy.leaf_count == 0
            ? 1.0
            : static_cast<double>(hierarchy.leaf_count);
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

// Build geometry first, then construct each leaf's disjoint near/far
// partition.  At every visited source node, build_mac_interaction_lists
// performs exactly one of two operations:
//   (1) accept that complete subtree once, or
//   (2) replace it by every child.
// Starting from the HNL cover preserves a disjoint cover of the complete
// truncated domain by induction.  The aggregate validator below checks the
// resulting leaf count and physical volume without expanding clusters.
// AFEPack-to-HNL adapter matching compare_exact.cpp; the layer algorithm
// itself is shared through hnl_candidates.hpp.
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

double parallel_fill_neighbors(
    const std::vector<HElement<DIM>*>& root_element_list,
    std::vector<hElementCache>& hElement_cache,
    std::map<HElement<DIM>*, int>& map_HElement_to_int,
    const std::vector<std::vector<std::set<HElement<DIM>* > > >&,
    double theta,
    bool construct_interactions,
    double* mapping_geometry_seconds,
    double* hierarchy_validation_time_seconds,
    int* omp_threads_used)
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

    if (!construct_interactions) {
      std::cout
          << "Neighbor preprocessing timing: mapping/geometry="
          << geometry_seconds
          << " seconds, hierarchy validation="
          << hierarchy_validation_seconds
          << " seconds; interaction lists not requested for this mesh\n"
          << "Parallel hierarchy preprocessing completed!" << std::endl;
      return 0.0;
    }

    std::vector<int> root_indices;
    root_indices.reserve(root_element_list.size());
    for (HElement<DIM>* root : root_element_list) {
      root_indices.push_back(read_only_map.at(root));
    }
    const std::vector<int> read_only_root_indices = std::move(root_indices);
    const auto& read_only_cache = hElement_cache;
    std::vector<InteractionListStats> interaction_stats(
        read_only_elements.size());
    const double interaction_start = omp_get_wtime();
    const auto topology = build_hnl_topology(
        read_only_elements, read_only_cache, read_only_map, read_only_root_indices);
    std::vector<int> leaf_hierarchy;
#ifdef TREECODE_HNL_VERIFY
    leaf_hierarchy.resize(hierarchy.leaf_count, -1);
    for (std::size_t i = 0; i < read_only_elements.size(); ++i)
      if (!read_only_elements[i]->isRefined())
        leaf_hierarchy[read_only_elements[i]->index] = static_cast<int>(i);
#endif
    std::size_t target_count = 0, candidate_count = 0, layer_count = 0;
    std::size_t invalid_cover_count = 0;
    #pragma omp parallel default(none) \
        shared(read_only_elements, topology, hElement_cache, \
               read_only_cache, interaction_stats, hierarchy, leaf_hierarchy) \
        firstprivate(theta) \
        reduction(+:target_count,candidate_count,layer_count,invalid_cover_count)
    {
      treecode_hnl::Workspace workspace(topology);
      #pragma omp for schedule(dynamic, 16)
      for(std::size_t i = 0; i < read_only_elements.size(); ++i) {
        if (!read_only_elements[i]->isRefined()) {
          const auto& candidates = treecode_hnl::build_candidates(
              topology, static_cast<int>(i), workspace);
#ifdef TREECODE_HNL_VERIFY
          // Test-only coverage audit; never repair or filter HNL output.
          if (!validate_hnl_cover(candidates, read_only_cache, hierarchy.leaf_count)) {
            ++invalid_cover_count;
            continue;
          }
#endif
          ++target_count;
          candidate_count += candidates.size();
          layer_count += workspace.layers;
          build_mac_interaction_lists(
              static_cast<int>(i), candidates,
              read_only_elements, hElement_cache[i], read_only_cache,
              theta, interaction_stats[i]);
#ifdef TREECODE_HNL_VERIFY
          auto final_cover = hElement_cache[i].far_hierarchy_indices;
          for (int leaf : hElement_cache[i].neighbor_leaf_indices)
            final_cover.push_back(leaf_hierarchy[leaf]);
          if (!validate_hnl_cover(final_cover, read_only_cache, hierarchy.leaf_count))
            ++invalid_cover_count;
#endif
        }
      }
    }
    if (invalid_cover_count != 0)
      throw std::runtime_error("H2O HNL cover contains overlapping or missing source regions");
    const double interaction_seconds = omp_get_wtime() - interaction_start;
    std::cout << "Interaction algorithm: HNL geometric layers -> target-specific MAC (CACHED)\n"
              << "HNL targets: " << target_count
              << ", candidates avg=" << (target_count ? double(candidate_count)/target_count : 0.0)
              << ", layers avg=" << (target_count ? double(layer_count)/target_count : 0.0)
#ifdef TREECODE_HNL_VERIFY
              << "; disjoint-cover verification passed for every target"
#endif
              << '\n';

    const double validation_start = omp_get_wtime();
    validate_interaction_stats(
        read_only_elements, interaction_stats, hierarchy, theta);
    const double validation_seconds = omp_get_wtime() - validation_start;
    
    std::cout << "Neighbor preprocessing timing: mapping/geometry="
              << geometry_seconds
              << " seconds, hierarchy validation="
              << hierarchy_validation_seconds
              << " seconds, HNL + MAC construction=" << interaction_seconds
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
        int* omp_threads_used)
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
      omp_threads_used
  );

  std::cout << "There are total " << hElement_cache.size()
            << " hElements" << std::endl;
  if (construct_interactions) {
    std::cout << "Filling neighbor_elements uses: "
              << neighbor_fill_seconds << " seconds" << std::endl;
  } else {
    std::cout
        << "Interaction-list construction deferred; "
           "hierarchy geometry only\n";
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

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
void select_h2o_becke_source(
    std::vector<ElementCache>& element_cache,
    std::size_t atom_index)
{
  if (atom_index >= treecode_h2o_three_center_problem::atom_count) {
    throw std::out_of_range("H2O source atom index is out of range");
  }
  const std::size_t element_count = element_cache.size();
  double maximum_partition_error = 0.0;
  double minimum_weight = 1.0;
  double maximum_weight = 0.0;
  #pragma omp parallel for schedule(static) default(none) \
      shared(element_cache) firstprivate(element_count, atom_index) \
      reduction(max:maximum_partition_error, maximum_weight) \
      reduction(min:minimum_weight)
  for (std::size_t element_index = 0;
       element_index < element_count;
       ++element_index) {
    ElementCache& element = element_cache[element_index];
    for (std::size_t q = 0; q < element.q_point.size(); ++q) {
      const AFEPack::Point<DIM>& point = element.q_point[q];
      const auto weights =
          treecode_h2o_three_center_problem::becke_weights(
              point[0], point[1], point[2]);
      double weight_sum = 0.0;
      for (double weight : weights) weight_sum += weight;
      maximum_partition_error = std::max(
          maximum_partition_error, std::abs(weight_sum - 1.0));
      minimum_weight = std::min(minimum_weight, weights[atom_index]);
      maximum_weight = std::max(maximum_weight, weights[atom_index]);
      element.source_charge[q] = element.Jxw[q]
          * weights[atom_index] * f(point);
    }
  }
  std::cout << "Becke partition on atom-centered mesh " << atom_index
            << ": max|sum_A w_A-1|=" << maximum_partition_error
            << ", local weight range=[" << minimum_weight
            << ", " << maximum_weight << "]\n";
}

void update_h2o_one_point_element_cache(
    std::vector<ElementCache>& element_cache,
    const FEMSpace<double, DIM>& fem_space,
    int source_component,
    int quadrature_accuracy)
{
  if (source_component < -1
      || source_component
          >= static_cast<int>(
              treecode_h2o_three_center_problem::atom_count)) {
    throw std::out_of_range("H2O source atom index is out of range");
  }
  const int element_count = fem_space.n_element();
  element_cache.clear();
  element_cache.resize(static_cast<std::size_t>(element_count));
  const Mesh<DIM>& regular_mesh = fem_space.mesh();
  const double four_pi = 4.0 * std::acos(-1.0);

  #pragma omp parallel for schedule(static) default(none) \
      shared(element_cache, fem_space, regular_mesh) \
      firstprivate(element_count, source_component, quadrature_accuracy, \
                   four_pi)
  for (int element_number = 0;
       element_number < element_count;
       ++element_number) {
    const Element<double, DIM>& element = fem_space.element(element_number);
    const int leaf_index = element.index();
    if (leaf_index < 0 || leaf_index >= element_count) {
      throw std::runtime_error("Invalid H2O leaf index");
    }
    ElementCache& cache =
        element_cache[static_cast<std::size_t>(leaf_index)];
    const GeometryBM& geometry = element.geometry();
    cache.physical_vertices.resize(
        static_cast<std::size_t>(geometry.n_vertex()));
    for (int vertex = 0; vertex < geometry.n_vertex(); ++vertex) {
      cache.physical_vertices[static_cast<std::size_t>(vertex)] =
          regular_mesh.point()[geometry.vertex()[vertex]];
    }

    const auto integration_rule =
        treecode_quadrature::build_physical_rule(
            cache.physical_vertices, quadrature_accuracy);
    cache.volume = 0.0;
    double integrated_charge = 0.0;
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      cache.bc[coordinate] = 0.0;
    }
    for (const auto& sample : integration_rule) {
      cache.volume += sample.weight;
      const double source_value = source_component >= 0
          ? h2o_weighted_source(
                sample.point,
                static_cast<std::size_t>(source_component))
          : f(sample.point);
      integrated_charge += sample.weight * source_value;
      for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        cache.bc[coordinate] += sample.weight * sample.point[coordinate];
      }
    }
    if (!(cache.volume > 0.0) || !std::isfinite(cache.volume)) {
      throw std::runtime_error("Invalid H2O one-point leaf volume");
    }
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
      cache.bc[coordinate] /= cache.volume;
    }
    cache.n_quadrature_point = 1;
    cache.q_point.assign(1, cache.bc);
    cache.Jxw.assign(1, cache.volume);
    cache.source_charge.assign(1, integrated_charge);
    cache.one_point_radius =
        std::cbrt(3.0 * cache.volume / four_pi);
  }

  std::cout
      << "H2O one-point leaf representation: source="
      << (source_component >= 0
              ? std::to_string(source_component)
              : std::string("all atoms"))
      << ", leaves=" << element_count
      << ", potential source points=" << element_count
      << ", charge-integration accuracy=" << quadrature_accuracy << '\n';
}
#endif

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
  if (!the_hElement->isRefined()) {
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

  if(!recurse_children || !the_hElement->isRefined()){
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
    const HElement<DIM>& source,
    int algebraic_accuracy,
    int source_component = -1)
{
  const auto& rule =
      treecode_quadrature::singular_duffy_axis_rule(
          algebraic_accuracy);

  const auto& vertices = source.h_element->vertex;

  // Connecting an interior target to each face partitions the tetrahedron.
  static constexpr int faces[4][3] = {
      {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

  double integral = 0.0;
  for (const auto& face : faces) {
    const AFEPack::Point<DIM>& a = *vertices[face[0]];
    const AFEPack::Point<DIM>& b = *vertices[face[1]];
    const AFEPack::Point<DIM>& c = *vertices[face[2]];
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
          // det*t^2*s / (t*|radial|) is the transformed 1/r Jacobian.
          integral += t_entry.second * t * reusable_weight
              * selected_source_value(
                  quadrature_point, source_component);
        }
      }
    }
  }
  return integral;
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
              singular_tetrahedron_integral(
                  target, *source_element, quadrature_accuracy,
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
                  singular_tetrahedron_integral(
                      target, *source_element, quadrature_accuracy,
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

bool pointInTetra(const AFEPack::Point<DIM>& p,
                  const AFEPack::Point<DIM>& A,
                  const AFEPack::Point<DIM>& B,
                  const AFEPack::Point<DIM>& C,
                  const AFEPack::Point<DIM>& D)
{
    const AFEPack::Point<DIM> e1 = B - A;
    const AFEPack::Point<DIM> e2 = C - A;
    const AFEPack::Point<DIM> e3 = D - A;
    const double determinant = det3(e1, e2, e3);
    const double determinant_scale = e1.length() * e2.length() * e3.length();
    if (std::abs(determinant)
        <= 64.0 * std::numeric_limits<double>::epsilon()
           * determinant_scale) {
      return false;
    }

    const AFEPack::Point<DIM> relative = p - A;
    const double lambda1 = det3(relative, e2, e3) / determinant;
    const double lambda2 = det3(e1, relative, e3) / determinant;
    const double lambda3 = det3(e1, e2, relative) / determinant;
    const double lambda0 = 1.0 - lambda1 - lambda2 - lambda3;
    const double condition_estimate = std::max(
        1.0, determinant_scale / std::abs(determinant));
    const double tolerance = std::min(
        1.0e-8,
        std::max(
            1.0e-12,
            256.0 * std::numeric_limits<double>::epsilon()
                * condition_estimate));
    return lambda0 >= -tolerance && lambda0 <= 1.0 + tolerance
        && lambda1 >= -tolerance && lambda1 <= 1.0 + tolerance
        && lambda2 >= -tolerance && lambda2 <= 1.0 + tolerance
        && lambda3 >= -tolerance && lambda3 <= 1.0 + tolerance;
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
    while (current->isRefined()) {
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
        bool use_gpu_backend,
        int h2o_atom_index = -1,
        H2OIndicatorMode indicator_mode = H2OIndicatorMode::radial)
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
        shared(ind, mesh, fem_space) \
        firstprivate(n_element, quadrature_accuracy, h2o_atom_index, \
                     indicator_mode)
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
#ifdef TREECODE_H2O_THREE_CENTER_MODEL
        const bool gaussian_radial =
            indicator_mode == H2OIndicatorMode::gaussian_radial;
        const double indicator_value = h2o_atom_index >= 0
            ? (gaussian_radial
                ? treecode_h2o_three_center_problem::
                    atomic_gaussian_radial_refinement_indicator(
                        entry.point[0], entry.point[1], entry.point[2],
                        static_cast<std::size_t>(h2o_atom_index))
                : treecode_h2o_three_center_problem::
                    atomic_radial_refinement_indicator(
                        entry.point[0], entry.point[1], entry.point[2],
                        static_cast<std::size_t>(h2o_atom_index)))
            : (gaussian_radial
                ? treecode_h2o_three_center_problem::
                    gaussian_radial_refinement_indicator_value(
                        entry.point[0], entry.point[1], entry.point[2])
                : treecode_h2o_three_center_problem::
                    refinement_indicator_value(
                        entry.point[0], entry.point[1], entry.point[2]));
        ind[the_geo_idx] += entry.weight * indicator_value;
#else
        (void)h2o_atom_index;
        const double radius_squared =
            entry.point[0] * entry.point[0] +
            entry.point[1] * entry.point[1] +
            entry.point[2] * entry.point[2];
        ind[the_geo_idx] +=
            entry.weight
            / std::sqrt(std::max(radius_squared, 1.0e-30));
#endif
      }
    }
  }

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
struct MomentDefectIndicatorStats
{
  double estimator = 0.0;
  double maximum = 0.0;
  std::size_t nonzero_elements = 0;
};

MomentDefectIndicatorStats get_moment_defect_indicator(
    Indicator<DIM>& ind,
    RegularMesh<DIM>& mesh,
    FEMSpace<double, DIM>& fem_space,
    int quadrature_accuracy,
    int source_component)
{
  ind.reinit(mesh);
  const int embedded_accuracy = std::min(
      quadrature_accuracy + 2,
      treecode_quadrature::maximum_algebraic_accuracy);
  if (embedded_accuracy == quadrature_accuracy) {
    throw std::invalid_argument(
        "Moment-defect refinement requires quadrature accuracy at most 18");
  }

  const int element_count = mesh.n_geometry(DIM);
  double estimator_squared = 0.0;
  double maximum = 0.0;
  std::size_t nonzero_elements = 0;

  #pragma omp parallel for schedule(static) default(none) \
      shared(ind, mesh, fem_space) \
      firstprivate(element_count, quadrature_accuracy, embedded_accuracy, \
                   source_component) \
      reduction(+:estimator_squared, nonzero_elements) reduction(max:maximum)
  for (int element_index = 0;
       element_index < element_count;
       ++element_index) {
    Element<double, DIM>& element = fem_space.element(element_index);
    GeometryBM& geometry = element.geometry();
    const int geometry_index = geometry.index();

    std::vector<AFEPack::Point<DIM>> vertices(
        static_cast<std::size_t>(geometry.n_vertex()));
    for (int vertex = 0; vertex < geometry.n_vertex(); ++vertex) {
      vertices[static_cast<std::size_t>(vertex)] =
          mesh.point()[geometry.vertex()[vertex]];
    }

    const auto coarse_rule = treecode_quadrature::build_physical_rule(
        vertices, quadrature_accuracy);
    const auto embedded_rule = treecode_quadrature::build_physical_rule(
        vertices, embedded_accuracy);

    AFEPack::Point<DIM> center;
    double volume = 0.0;
    for (const auto& entry : embedded_rule) {
      AFEPack::Point<DIM> weighted_point = entry.point;
      weighted_point *= entry.weight;
      center += weighted_point;
      volume += entry.weight;
    }
    if (!(volume > 0.0)) {
      ind[geometry_index] = 0.0;
      continue;
    }
    center /= volume;

    double radius = 0.0;
    for (const auto& vertex : vertices) {
      radius = std::max(radius, (vertex - center).length());
    }
    radius = std::max(radius, 1.0e-14);

    std::array<double, 10> coarse_moments{};
    std::array<double, 10> embedded_moments{};
    const auto accumulate = [source_component, &center](
        const std::vector<treecode_quadrature::QuadraturePoint>& rule,
        std::array<double, 10>& moments) {
      for (const auto& entry : rule) {
        const double point[DIM] = {
            entry.point[0], entry.point[1], entry.point[2]};
        const double weighted_source =
            entry.weight * selected_source_value(point, source_component);
        const double displacement[DIM] = {
            entry.point[0] - center[0],
            entry.point[1] - center[1],
            entry.point[2] - center[2]};
        const double powers[DIM][3] = {
            {1.0, displacement[0], displacement[0] * displacement[0]},
            {1.0, displacement[1], displacement[1] * displacement[1]},
            {1.0, displacement[2], displacement[2] * displacement[2]}};
        for (int degree = 0; degree <= 2; ++degree) {
          for (int x_degree = 0; x_degree <= degree; ++x_degree) {
            for (int y_degree = 0;
                 y_degree <= degree - x_degree;
                 ++y_degree) {
              const int z_degree = degree - x_degree - y_degree;
              const int coefficient = treecode_multipole::coefficient_index(
                  x_degree, y_degree, z_degree);
              moments[static_cast<std::size_t>(coefficient)] +=
                  weighted_source
                  * powers[0][x_degree]
                  * powers[1][y_degree]
                  * powers[2][z_degree];
            }
          }
        }
      }
    };
    accumulate(coarse_rule, coarse_moments);
    accumulate(embedded_rule, embedded_moments);

    double indicator_squared = 0.0;
    for (int degree = 0; degree <= 2; ++degree) {
      const double inverse_radius_power =
          degree == 0 ? 1.0
          : degree == 1 ? 1.0 / radius
                        : 1.0 / (radius * radius);
      for (int x_degree = 0; x_degree <= degree; ++x_degree) {
        for (int y_degree = 0;
             y_degree <= degree - x_degree;
             ++y_degree) {
          const int z_degree = degree - x_degree - y_degree;
          const int coefficient = treecode_multipole::coefficient_index(
              x_degree, y_degree, z_degree);
          const double defect = inverse_radius_power
              * (embedded_moments[static_cast<std::size_t>(coefficient)]
                 - coarse_moments[static_cast<std::size_t>(coefficient)]);
          indicator_squared += defect * defect;
        }
      }
    }

    const double indicator_value = std::sqrt(indicator_squared);
    ind[geometry_index] = indicator_value;
    estimator_squared += indicator_squared;
    maximum = std::max(maximum, indicator_value);
    if (indicator_value > 0.0) ++nonzero_elements;
  }

  MomentDefectIndicatorStats stats;
  stats.estimator = std::sqrt(estimator_squared);
  stats.maximum = maximum;
  stats.nonzero_elements = nonzero_elements;
  std::cout << std::setprecision(12)
            << "Embedded moment-defect indicator: p="
            << quadrature_accuracy
            << ", embedded p=" << embedded_accuracy
            << ", estimator=" << stats.estimator
            << ", maximum element indicator=" << stats.maximum
            << ", nonzero elements=" << stats.nonzero_elements << '\n';
  return stats;
}

std::size_t mark_moment_defect_dorfler(
    Indicator<DIM>& indicator,
    RegularMesh<DIM>& mesh,
    double marking_fraction)
{
  struct RankedIndicator
  {
    double squared_value;
    int geometry_index;
  };
  std::vector<RankedIndicator> ranking;
  ranking.reserve(static_cast<std::size_t>(mesh.n_geometry(DIM)));
  double total_squared = 0.0;
  const int element_count = static_cast<int>(mesh.n_geometry(DIM));
  for (int element_index = 0;
       element_index < element_count;
       ++element_index) {
    const int geometry_index = mesh.geometry(DIM, element_index).index();
    const double value = indicator[geometry_index];
    const double squared_value = value * value;
    ranking.push_back({squared_value, geometry_index});
    total_squared += squared_value;
  }
  std::sort(
      ranking.begin(), ranking.end(),
      [](const RankedIndicator& left, const RankedIndicator& right) {
        return left.squared_value > right.squared_value;
      });

  const double target = marking_fraction * total_squared;
  double marked_squared = 0.0;
  std::size_t marked_count = 0;
  for (const RankedIndicator& entry : ranking) {
    indicator[entry.geometry_index] = 0.0;
  }
  for (const RankedIndicator& entry : ranking) {
    if (marked_squared >= target || !(entry.squared_value > 0.0)) break;
    indicator[entry.geometry_index] = 1.0;
    marked_squared += entry.squared_value;
    ++marked_count;
  }
  std::cout << "Dorfler marking: fraction=" << marking_fraction
            << ", marked=" << marked_count << '/' << ranking.size()
            << ", captured="
            << (total_squared > 0.0 ? marked_squared / total_squared : 0.0)
            << " of squared estimator\n";
  return marked_count;
}
#endif

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
    if (!element->isRefined()) {
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
    if (!element->isRefined()) {
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

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
void initialize_fem_space(
    FEMSpace<double, DIM>& fem_space,
    RegularMesh<DIM>& mesh,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements)
{
  fem_space.reinit(mesh, template_elements);
  const int element_count = mesh.n_geometry(DIM);
  fem_space.element().resize(element_count);
  for (int element_index = 0;
       element_index < element_count;
       ++element_index) {
    const int vertex_count =
        mesh.geometry(DIM, element_index).n_vertex();
    int template_index = -1;
    if (vertex_count == 4) template_index = 0;
    if (vertex_count == 5) template_index = 1;
    if (vertex_count == 7) template_index = 2;
    if (template_index < 0) {
      throw std::runtime_error(
          "Unexpected AFEPack regular-element vertex count");
    }
    fem_space.element(element_index).reinit(
        fem_space, element_index, template_index);
  }
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();
}

std::unique_ptr<IrregularMesh<DIM>> build_h2o_refined_mesh(
    HGeometryTree<DIM>& h_tree,
    int initial_refinements,
    int quadrature_accuracy,
    double adaptation_tolerance,
    H2OIndicatorMode indicator_mode,
    int indicator_center,
    const std::string& output_filename,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements,
    std::unique_ptr<FEMSpace<double, DIM>>& fem_space)
{
  auto mesh = std::make_unique<IrregularMesh<DIM>>();
  mesh->reinit(h_tree);
  mesh->semiregularize();
  mesh->regularize(false);
  mesh->globalRefine(initial_refinements);
  mesh->semiregularize();
  mesh->regularize(false);

  fem_space = std::make_unique<FEMSpace<double, DIM>>();
  initialize_fem_space(
      *fem_space, mesh->regularMesh(), template_elements);
  Indicator<DIM> first_indicator(mesh->regularMesh());
  get_indicator(
      first_indicator,
      mesh->regularMesh(),
      *fem_space,
      quadrature_accuracy,
      false,
      indicator_center);

  auto old_mesh = std::make_unique<IrregularMesh<DIM>>(*mesh);
  mesh = std::make_unique<IrregularMesh<DIM>>();
  MeshAdaptor<DIM> first_adaptor(*old_mesh, *mesh);
  first_adaptor.setIndicator(first_indicator);
  first_adaptor.tolerence() = 1.0e-2;
  first_adaptor.convergenceOrder() = 0.0;
  first_adaptor.refineStep() = 5;
  first_adaptor.adapt();
  mesh->semiregularize();
  mesh->regularize(false);

  initialize_fem_space(
      *fem_space, mesh->regularMesh(), template_elements);
  if (indicator_mode == H2OIndicatorMode::moment_defect) {
    constexpr int maximum_adaptation_rounds = 12;
    bool converged = false;
    for (int round = 0; round < maximum_adaptation_rounds; ++round) {
      Indicator<DIM> moment_indicator(mesh->regularMesh());
      const MomentDefectIndicatorStats moment_stats =
          get_moment_defect_indicator(
              moment_indicator,
              mesh->regularMesh(),
              *fem_space,
              quadrature_accuracy,
              indicator_center);
      std::cout << "Moment-defect adaptation round " << round + 1
                << ": regular elements="
                << mesh->regularMesh().n_geometry(DIM)
                << ", target estimator=" << adaptation_tolerance << '\n';
      if (moment_stats.estimator <= adaptation_tolerance) {
        converged = true;
        break;
      }
      if (mark_moment_defect_dorfler(
              moment_indicator, mesh->regularMesh(), 0.5) == 0) {
        break;
      }

      auto previous_mesh = std::make_unique<IrregularMesh<DIM>>(*mesh);
      mesh = std::make_unique<IrregularMesh<DIM>>();
      MeshAdaptor<DIM> adaptor(*previous_mesh, *mesh);
      adaptor.setIndicator(moment_indicator);
      // Marked entries are one and unmarked entries are zero.  With the
      // AFEPack threshold and DIM=3, 1e-2 refines exactly the marked set.
      adaptor.tolerence() = 1.0e-2;
      adaptor.convergenceOrder() = 0.0;
      adaptor.refineStep() = 1;
      adaptor.is_refine_only() = true;
      adaptor.adapt();
      mesh->semiregularize();
      mesh->regularize(false);
      initialize_fem_space(
          *fem_space, mesh->regularMesh(), template_elements);
    }
    if (!converged) {
      std::cout
          << "Moment-defect adaptation reached its round limit before "
             "the requested estimator tolerance.\n";
    }
  } else {
    Indicator<DIM> second_indicator(mesh->regularMesh());
    get_indicator(
        second_indicator,
        mesh->regularMesh(),
        *fem_space,
        quadrature_accuracy,
        false,
        indicator_center,
        indicator_mode);
    auto old_mesh_second = std::make_unique<IrregularMesh<DIM>>(*mesh);
    mesh = std::make_unique<IrregularMesh<DIM>>();
    MeshAdaptor<DIM> second_adaptor(*old_mesh_second, *mesh);
    second_adaptor.setIndicator(second_indicator);
    second_adaptor.tolerence() = adaptation_tolerance;
    second_adaptor.convergenceOrder() = 0.0;
    second_adaptor.refineStep() = 2;
    second_adaptor.adapt();
    mesh->semiregularize();
    mesh->regularize(false);
    initialize_fem_space(
        *fem_space, mesh->regularMesh(), template_elements);
  }

  mesh->regularMesh().writeOpenDXData(output_filename);
  return mesh;
}

std::unique_ptr<IrregularMesh<DIM>> build_h2o_atomic_mesh(
    HGeometryTree<DIM>& h_tree,
    int initial_refinements,
    int quadrature_accuracy,
    double adaptation_tolerance,
    H2OIndicatorMode indicator_mode,
    std::size_t atom_index,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements,
    std::unique_ptr<FEMSpace<double, DIM>>& fem_space)
{
  const std::array<std::string,
                   treecode_h2o_three_center_problem::atom_count> filenames{{
      "h2o_O_afepack.dx",
      "h2o_H1_afepack.dx",
      "h2o_H2_afepack.dx"}};
  return build_h2o_refined_mesh(
      h_tree,
      initial_refinements,
      quadrature_accuracy,
      adaptation_tolerance,
      indicator_mode,
      static_cast<int>(atom_index),
      filenames.at(atom_index),
      template_elements,
      fem_space);
}

std::unique_ptr<IrregularMesh<DIM>> build_h2o_all_atoms_mesh(
    HGeometryTree<DIM>& h_tree,
    int initial_refinements,
    int quadrature_accuracy,
    double adaptation_tolerance,
    H2OIndicatorMode indicator_mode,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements,
    std::unique_ptr<FEMSpace<double, DIM>>& fem_space)
{
  return build_h2o_refined_mesh(
      h_tree,
      initial_refinements,
      quadrature_accuracy,
      adaptation_tolerance,
      indicator_mode,
      -1,
      "h2o_all_atoms_afepack.dx",
      template_elements,
      fem_space);
}

struct H2OSourceTree
{
  std::vector<hElementCache> hierarchy_cache;
  std::map<HElement<DIM>*, int> hierarchy_map;
  std::vector<HElement<DIM>*> hierarchy_elements;
  std::vector<HElement<DIM>*> root_elements;
  std::vector<ElementCache> element_cache;
};

using H2OBinomialTable = std::array<
    std::array<double, treecode_multipole::maximum_order + 1>,
    treecode_multipole::maximum_order + 1>;

void translate_and_accumulate_child_moments(
    hElementCache& parent,
    const hElementCache& child,
    int expansion_order,
    const H2OBinomialTable& binomial,
    std::array<double,
               treecode_multipole::maximum_coefficient_count>& x_translated,
    std::array<double,
               treecode_multipole::maximum_coefficient_count>& xy_translated)
{
  std::array<double, treecode_multipole::maximum_order + 1> x_power{};
  std::array<double, treecode_multipole::maximum_order + 1> y_power{};
  std::array<double, treecode_multipole::maximum_order + 1> z_power{};
  x_power[0] = y_power[0] = z_power[0] = 1.0;
  const AFEPack::Point<DIM> shift = child.ref_point - parent.ref_point;
  for (int degree = 1; degree <= expansion_order; ++degree) {
    x_power[static_cast<std::size_t>(degree)] =
        x_power[static_cast<std::size_t>(degree - 1)] * shift[0];
    y_power[static_cast<std::size_t>(degree)] =
        y_power[static_cast<std::size_t>(degree - 1)] * shift[1];
    z_power[static_cast<std::size_t>(degree)] =
        z_power[static_cast<std::size_t>(degree - 1)] * shift[2];
  }

  // Translate one Cartesian direction at a time.  This is the CPU analogue
  // of the separable GPU translation and costs O(P C_P), not O(C_P^2).
  for (int degree = 0; degree <= expansion_order; ++degree) {
    for (int x = 0; x <= degree; ++x) {
      for (int y = 0; y <= degree - x; ++y) {
        const int z = degree - x - y;
        double value = 0.0;
        for (int child_x = 0; child_x <= x; ++child_x) {
          value += binomial[static_cast<std::size_t>(x)]
                           [static_cast<std::size_t>(child_x)]
              * x_power[static_cast<std::size_t>(x - child_x)]
              * child.moments[static_cast<std::size_t>(
                    treecode_multipole::coefficient_index(
                        child_x, y, z))];
        }
        x_translated[static_cast<std::size_t>(
            treecode_multipole::coefficient_index(x, y, z))] = value;
      }
    }
  }
  for (int degree = 0; degree <= expansion_order; ++degree) {
    for (int x = 0; x <= degree; ++x) {
      for (int y = 0; y <= degree - x; ++y) {
        const int z = degree - x - y;
        double value = 0.0;
        for (int child_y = 0; child_y <= y; ++child_y) {
          value += binomial[static_cast<std::size_t>(y)]
                           [static_cast<std::size_t>(child_y)]
              * y_power[static_cast<std::size_t>(y - child_y)]
              * x_translated[static_cast<std::size_t>(
                    treecode_multipole::coefficient_index(
                        x, child_y, z))];
        }
        xy_translated[static_cast<std::size_t>(
            treecode_multipole::coefficient_index(x, y, z))] = value;
      }
    }
  }
  for (int degree = 0; degree <= expansion_order; ++degree) {
    for (int x = 0; x <= degree; ++x) {
      for (int y = 0; y <= degree - x; ++y) {
        const int z = degree - x - y;
        double value = 0.0;
        for (int child_z = 0; child_z <= z; ++child_z) {
          value += binomial[static_cast<std::size_t>(z)]
                           [static_cast<std::size_t>(child_z)]
              * z_power[static_cast<std::size_t>(z - child_z)]
              * xy_translated[static_cast<std::size_t>(
                    treecode_multipole::coefficient_index(
                        x, y, child_z))];
        }
        parent.moments[static_cast<std::size_t>(
            treecode_multipole::coefficient_index(x, y, z))] += value;
      }
    }
  }
}

void build_h2o_multipoles_bottom_up(
    H2OSourceTree& source,
    int expansion_order)
{
  const double start = omp_get_wtime();
  const std::size_t hierarchy_count = source.hierarchy_elements.size();
  const std::size_t coefficient_count = static_cast<std::size_t>(
      treecode_multipole::coefficient_count(expansion_order));
  int maximum_level = 0;
  for (const hElementCache& cache : source.hierarchy_cache) {
    maximum_level = std::max(maximum_level, cache.level);
  }
  H2OBinomialTable binomial{};
  for (int degree = 0; degree <= expansion_order; ++degree) {
    binomial[static_cast<std::size_t>(degree)][0] = 1.0;
    binomial[static_cast<std::size_t>(degree)]
            [static_cast<std::size_t>(degree)] = 1.0;
    for (int index = 1; index < degree; ++index) {
      binomial[static_cast<std::size_t>(degree)]
              [static_cast<std::size_t>(index)] =
          binomial[static_cast<std::size_t>(degree - 1)]
                  [static_cast<std::size_t>(index - 1)]
          + binomial[static_cast<std::size_t>(degree - 1)]
                    [static_cast<std::size_t>(index)];
    }
  }
  std::vector<std::vector<std::size_t>> hierarchy_indices_by_level(
      static_cast<std::size_t>(maximum_level + 1));
  for (std::size_t hierarchy_index = 0;
       hierarchy_index < hierarchy_count;
       ++hierarchy_index) {
    hierarchy_indices_by_level[static_cast<std::size_t>(
        source.hierarchy_cache[hierarchy_index].level)].push_back(
            hierarchy_index);
  }

  #pragma omp parallel for schedule(static)
  for (std::size_t hierarchy_index = 0;
       hierarchy_index < hierarchy_count;
       ++hierarchy_index) {
    hElementCache& hierarchy = source.hierarchy_cache[hierarchy_index];
    hierarchy.moments.assign(coefficient_count, 0.0);
    HElement<DIM>* element = source.hierarchy_elements[hierarchy_index];
    if (element->isRefined()) continue;
    const ElementCache& leaf = source.element_cache[
        static_cast<std::size_t>(element->index)];
    for (int q = 0; q < leaf.n_quadrature_point; ++q) {
      const AFEPack::Point<DIM>& point =
          leaf.q_point[static_cast<std::size_t>(q)];
      treecode_multipole::accumulate_raw_moments(
          hierarchy.moments,
          expansion_order,
          point[0] - hierarchy.ref_point[0],
          point[1] - hierarchy.ref_point[1],
          point[2] - hierarchy.ref_point[2],
          leaf.source_charge[static_cast<std::size_t>(q)]);
    }
  }

  for (int level = maximum_level - 1; level >= 0; --level) {
    const std::vector<std::size_t>& level_indices =
        hierarchy_indices_by_level[static_cast<std::size_t>(level)];
    const std::size_t level_count = level_indices.size();
    #pragma omp parallel for schedule(static)
    for (std::size_t level_position = 0;
         level_position < level_count;
         ++level_position) {
      const std::size_t hierarchy_index = level_indices[level_position];
      hElementCache& parent = source.hierarchy_cache[hierarchy_index];
      if (parent.child_indices.empty()) continue;
      std::array<double,
                 treecode_multipole::maximum_coefficient_count> x_translated;
      std::array<double,
                 treecode_multipole::maximum_coefficient_count> xy_translated;
      for (int child_index : parent.child_indices) {
        translate_and_accumulate_child_moments(
            parent,
            source.hierarchy_cache[static_cast<std::size_t>(child_index)],
            expansion_order,
            binomial,
            x_translated,
            xy_translated);
      }
    }
  }

  long double leaf_charge = 0.0L;
  long double absolute_leaf_charge = 0.0L;
  for (const ElementCache& leaf : source.element_cache) {
    for (double charge : leaf.source_charge) {
      leaf_charge += static_cast<long double>(charge);
      absolute_leaf_charge += std::abs(static_cast<long double>(charge));
    }
  }
  long double root_monopole = 0.0L;
  for (HElement<DIM>* root : source.root_elements) {
    const int root_index = source.hierarchy_map.at(root);
    root_monopole += static_cast<long double>(
        source.hierarchy_cache[static_cast<std::size_t>(root_index)]
            .moments[0]);
  }
  const long double monopole_tolerance =
      2.0e-12L * std::max(1.0L, absolute_leaf_charge);
  if (std::abs(root_monopole - leaf_charge) > monopole_tolerance) {
    throw std::runtime_error(
        "Bottom-up H2O root monopoles do not match the leaf charges");
  }

  std::cout << "H2O bottom-up multipole construction: order="
            << expansion_order << ", coefficients=" << coefficient_count
            << ", root-charge error="
            << static_cast<double>(std::abs(root_monopole - leaf_charge))
            << ", time=" << omp_get_wtime() - start << " seconds\n";
}

void collect_leaf_descendants(
    HElement<DIM>* element,
    std::set<HElement<DIM>*>& leaves)
{
  if (element == nullptr) return;
  if (!element->isRefined()) {
    leaves.insert(element);
    return;
  }
  for (HElement<DIM>* child : element->child) {
    if (child != nullptr) collect_leaf_descendants(child, leaves);
  }
}

// Legacy unused three-expansion implementation. The active path above uses
// hnl_candidates.hpp, one geometric expansion per decreasing refinement level.
void build_layered_mac_interaction_list(
    int target_index,
    const std::vector<int>& root_indices,
    const std::vector<HElement<DIM>*>& hierarchy_elements,
    std::vector<hElementCache>& hierarchy_cache,
    const std::map<HElement<DIM>*, int>& hierarchy_map,
    const std::vector<std::vector<std::set<HElement<DIM>*>>>& node_neighbors,
    double theta,
    InteractionListStats& stats)
{
  HElement<DIM>* target =
      hierarchy_elements[static_cast<std::size_t>(target_index)];
  hElementCache& target_cache =
      hierarchy_cache[static_cast<std::size_t>(target_index)];
  target_cache.neighbor_elements.clear();
  target_cache.nonNeighbor_elements.clear();
  target_cache.neighbor_leaf_indices.clear();
  target_cache.far_hierarchy_indices.clear();
  stats = InteractionListStats{};

  // Reproduce the three explicit geometric expansions used by the original
  // layer construction.  Keep the resulting hierarchy elements as MAC
  // candidates; do not flatten refined candidates into all descendants.
  std::set<HElement<DIM>*> geometric_candidates;
  geometric_candidates.insert(target);
  std::set<HGeometry<0, DIM>*> initial_vertices;
  initial_vertices.insert(
      target->h_element->vertex.begin(), target->h_element->vertex.end());
  std::set<HGeometry<0, DIM>*> outer_vertices;
  std::set<HGeometry<0, DIM>*> visited_vertices;
  std::set<HElement<DIM>*> initial_layer;
  std::set<HElement<DIM>*> outer_layer;
  std::set<HElement<DIM>*> geometric_layer;
  initial_layer.insert(target);
  for (int layer = 0; layer < 3 && !initial_layer.empty(); ++layer) {
    construct_neighbor_layers(
        initial_vertices,
        outer_vertices,
        visited_vertices,
        initial_layer,
        outer_layer,
        target_cache.level,
        geometric_layer,
        node_neighbors,
        1);
    for (HElement<DIM>* candidate : geometric_layer) {
      if (candidate != nullptr) geometric_candidates.insert(candidate);
    }
    initial_vertices = outer_vertices;
    initial_layer = outer_layer;
  }

  // Descend to each geometric candidate, then let the MAC classify the
  // candidate itself.  Only its strict ancestors are forced to descend.
  std::set<int> geometric_candidate_paths;
  for (HElement<DIM>* candidate : geometric_candidates) {
    for (HElement<DIM>* current = candidate->parent;
         current != nullptr;
         current = current->parent) {
      geometric_candidate_paths.insert(hierarchy_map.at(current));
    }
  }

  std::vector<int> stack(root_indices.begin(), root_indices.end());
  while (!stack.empty()) {
    const int source_index = stack.back();
    stack.pop_back();
    ++stats.visited_hierarchy_nodes;
    HElement<DIM>* source =
        hierarchy_elements[static_cast<std::size_t>(source_index)];
    const hElementCache& source_cache =
        hierarchy_cache[static_cast<std::size_t>(source_index)];
    const bool geometric_path =
        geometric_candidate_paths.find(source_index)
        != geometric_candidate_paths.end();
    const double center_distance =
        (source_cache.ref_point - target_cache.ref_point).length();
    const double minimum_target_distance =
        center_distance - target_cache.radius;
    const bool contains_target =
        source_cache.subtree_leaf_begin
            <= target_cache.subtree_leaf_begin
        && target_cache.subtree_leaf_begin
            < source_cache.subtree_leaf_end;
    const bool mac_accepted =
        !geometric_path && !contains_target
        && minimum_target_distance > 0.0
        && source_cache.radius / minimum_target_distance <= theta;

    if (mac_accepted) {
      target_cache.far_hierarchy_indices.push_back(source_index);
      stats.represented_leaf_count += source_cache.subtree_leaf_count;
      stats.represented_volume += source_cache.volume;
      ++stats.far_entries;
      stats.maximum_mac_ratio = std::max(
          stats.maximum_mac_ratio,
          source_cache.radius / minimum_target_distance);
    } else if (!source->isRefined()) {
      target_cache.neighbor_leaf_indices.push_back(source->index);
      ++stats.represented_leaf_count;
      stats.represented_volume += source_cache.volume;
      stats.near_volume += source_cache.volume;
      ++stats.near_entries;
      stats.contains_target_in_near_field =
          stats.contains_target_in_near_field || source == target;
    } else {
      for (int child_index : source_cache.child_indices) {
        stack.push_back(child_index);
      }
    }
  }
}

double fill_layered_mac_lists_for_source(
    H2OSourceTree& source,
    double theta,
    int* omp_threads_used)
{
  const double start = omp_get_wtime();
  std::size_t vertex_table_size = 0;
  for (HElement<DIM>* element : source.hierarchy_elements) {
    for (HGeometry<0, DIM>* vertex : element->h_element->vertex) {
      if (vertex->index >= 0) {
        vertex_table_size = std::max(
            vertex_table_size,
            static_cast<std::size_t>(vertex->index + 1));
      }
    }
  }
  std::vector<std::vector<std::set<HElement<DIM>*>>> node_neighbors(
      vertex_table_size);
  for (std::size_t hierarchy_index = 0;
       hierarchy_index < source.hierarchy_elements.size();
       ++hierarchy_index) {
    HElement<DIM>* element = source.hierarchy_elements[hierarchy_index];
    const int level = source.hierarchy_cache[hierarchy_index].level;
    for (HGeometry<0, DIM>* vertex : element->h_element->vertex) {
      if (vertex->index < 0) continue;
      auto& by_level =
          node_neighbors[static_cast<std::size_t>(vertex->index)];
      if (static_cast<std::size_t>(level) >= by_level.size()) {
        by_level.resize(static_cast<std::size_t>(level + 1));
      }
      by_level[static_cast<std::size_t>(level)].insert(element);
    }
  }

  std::vector<int> root_indices;
  root_indices.reserve(source.root_elements.size());
  for (HElement<DIM>* root : source.root_elements) {
    root_indices.push_back(source.hierarchy_map.at(root));
  }
  const HierarchySummary hierarchy = prepare_hierarchy_aggregates(
      source.hierarchy_elements,
      source.root_elements,
      source.hierarchy_cache,
      source.hierarchy_map);
  std::vector<InteractionListStats> stats(
      source.hierarchy_elements.size());
  const std::size_t hierarchy_count = source.hierarchy_elements.size();
  #pragma omp parallel default(none) \
      shared(source, root_indices, node_neighbors, stats, omp_threads_used) \
      firstprivate(theta, hierarchy_count)
  {
    #pragma omp single
    {
      if (omp_threads_used != nullptr) {
        *omp_threads_used = std::max(
            *omp_threads_used, omp_get_num_threads());
      }
    }
    #pragma omp for schedule(dynamic, 16)
    for (std::size_t target_index = 0;
         target_index < hierarchy_count;
         ++target_index) {
      if (!source.hierarchy_elements[target_index]->isRefined()) {
        build_layered_mac_interaction_list(
            static_cast<int>(target_index),
            root_indices,
            source.hierarchy_elements,
            source.hierarchy_cache,
            source.hierarchy_map,
            node_neighbors,
            theta,
            stats[target_index]);
      }
    }
  }
  validate_interaction_stats(
      source.hierarchy_elements, stats, hierarchy, theta);
  const double seconds = omp_get_wtime() - start;
  std::cout
      << "Layered geometric construction plus MAC uses: "
      << seconds << " seconds\n";
  return seconds;
}

void prepare_h2o_source_tree(
    H2OSourceTree& source,
    IrregularMesh<DIM>& mesh,
    const FEMSpace<double, DIM>& fem_space,
    int source_component,
    const ProgramOptions& options,
    double& mapping_seconds,
    double& hierarchy_seconds,
    double& interaction_seconds,
    int& omp_threads_used)
{
  double local_mapping_seconds = 0.0;
  double local_hierarchy_seconds = 0.0;
  int local_threads_used = 1;
  const double local_interaction_seconds = update_hElementCache(
      mesh,
      source.hierarchy_cache,
      source.hierarchy_map,
      options.mac_theta,
      true,
      &local_mapping_seconds,
      &local_hierarchy_seconds,
      &local_threads_used);
  mapping_seconds += local_mapping_seconds;
  hierarchy_seconds += local_hierarchy_seconds;
  interaction_seconds += local_interaction_seconds;
  omp_threads_used = std::max(omp_threads_used, local_threads_used);

  if (options.h2o_near_field_mode == H2ONearFieldMode::quadrature) {
    updateElementCache(
        source.element_cache,
        fem_space,
        options.quadrature_accuracy);
    std::cout << "Target-element singular integration: ";
    if (options.h2o_singular_mode == H2OSingularMode::duffy) {
      std::cout << "original collapsed Duffy quadrature\n";
    } else if (options.h2o_singular_mode
               == H2OSingularMode::analytic_all) {
      std::cout
          << "analytic constant-density term plus Duffy remainder "
             "for every target\n";
    } else {
      std::cout
          << "analytic constant-density term plus Duffy remainder only "
             "within " << options.h2o_core_radius
          << " bohr of an atom; original Duffy quadrature elsewhere\n";
    }
    if (source_component >= 0) {
      select_h2o_becke_source(
          source.element_cache,
          static_cast<std::size_t>(source_component));
    }
  } else {
    update_h2o_one_point_element_cache(
        source.element_cache,
        fem_space,
        source_component,
        options.quadrature_accuracy);
  }
  IrregularMesh<DIM>::ActiveIterator element = mesh.beginActiveElement();
  IrregularMesh<DIM>::ActiveIterator end_element = mesh.endActiveElement();
  for (; element != end_element; ++element) {
    if (element->index < 0
        || static_cast<std::size_t>(element->index)
            >= source.element_cache.size()) {
      throw std::runtime_error("Invalid H2O source leaf index");
    }
    ElementCache& cache =
        source.element_cache[static_cast<std::size_t>(element->index)];
    cache.the_hElement = &(*element);
    cache.hierarchy_index = source.hierarchy_map.at(&(*element));
  }
  source.hierarchy_elements.assign(
      source.hierarchy_cache.size(), nullptr);
  for (const auto& entry : source.hierarchy_map) {
    source.hierarchy_elements[static_cast<std::size_t>(entry.second)] =
        entry.first;
    if (entry.first->parent == nullptr) {
      source.root_elements.push_back(entry.first);
    }
  }
  if (std::find(source.hierarchy_elements.begin(),
                source.hierarchy_elements.end(),
                nullptr) != source.hierarchy_elements.end()) {
    throw std::runtime_error("Incomplete H2O source hierarchy mapping");
  }
  std::sort(
      source.root_elements.begin(),
      source.root_elements.end(),
      [&source](HElement<DIM>* left, HElement<DIM>* right) {
        return source.hierarchy_cache[static_cast<std::size_t>(
                   source.hierarchy_map.at(left))].subtree_leaf_begin
             < source.hierarchy_cache[static_cast<std::size_t>(
                   source.hierarchy_map.at(right))].subtree_leaf_begin;
      });

  build_h2o_multipoles_bottom_up(source, options.expansion_order);
}

double singular_subtetrahedron_integral(
    const AFEPack::Point<DIM>& target,
    const std::array<AFEPack::Point<DIM>, 4>& vertices,
    int algebraic_accuracy,
    int source_component,
    bool enable_analytic_subtraction)
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
      enable_analytic_subtraction
      && std::isfinite(target_density) && std::isfinite(analytic_kernel);
  static constexpr int faces[4][3] = {
      {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  // Split rho(y) = rho(x) + (rho(y)-rho(x)).  The singular constant part is
  // integrated in closed form; Duffy quadrature sees only the regularized
  // remainder.  Fall back to the original full Duffy rule for a degenerate
  // geometric configuration.
  double integral = use_analytic_subtraction
      ? target_density * analytic_kernel : 0.0;
  for (const auto& face : faces) {
    const AFEPack::Point<DIM>& a = vertices[face[0]];
    const AFEPack::Point<DIM>& b = vertices[face[1]];
    const AFEPack::Point<DIM>& c = vertices[face[2]];
    const AFEPack::Point<DIM> ea = a - target;
    const AFEPack::Point<DIM> eb = b - target;
    const AFEPack::Point<DIM> ec = c - target;
    const double determinant = std::abs(det3(ea, eb, ec));
    if (determinant <= std::numeric_limits<double>::epsilon()) continue;
    for (const auto& s_entry : rule) {
      const double s = s_entry.first;
      for (const auto& r_entry : rule) {
        const double r = r_entry.first;
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
        if (radial_length <= std::numeric_limits<double>::min()) continue;
        const double reusable_weight =
            s_entry.second * r_entry.second
            * determinant * s / radial_length;
        for (const auto& t_entry : rule) {
          AFEPack::Point<DIM> step = radial;
          step *= t_entry.first;
          const AFEPack::Point<DIM> point = target + step;
          double density = selected_source_value(point, source_component);
          if (use_analytic_subtraction) {
            density -= target_density;
          }
          integral += t_entry.second * t_entry.first * reusable_weight
              * density;
        }
      }
    }
  }
  return integral;
}

double integrate_h2o_source_leaf_one_point(
    const AFEPack::Point<DIM>& target,
    const ElementCache& source)
{
  const double charge = source.source_charge[0];
  const double distance = (source.q_point[0] - target).length();
  const double radius = source.one_point_radius;
  if (distance >= radius) return charge / distance;

  // A point charge is singular when a target and a source barycenter
  // coincide.  Represent the finite leaf by an equal-volume uniform sphere
  // only inside its effective radius; outside, retain the point-charge kernel.
  const double radius_squared = radius * radius;
  return charge
      * (3.0 * radius_squared - distance * distance)
      / (2.0 * radius_squared * radius);
}

double integrate_h2o_regular_quadrature_range(
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

double integrate_h2o_source_leaf_quadrature(
    const AFEPack::Point<DIM>& target,
    const ElementCache& source,
    bool contains_target,
    int algebraic_accuracy,
    int source_component,
    bool enable_analytic_subtraction)
{
  if (source.q_point.size() != source.source_charge.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!contains_target) {
    return integrate_h2o_regular_quadrature_range(
        target, source, 0, source.q_point.size());
  }

  const auto decomposition = treecode_quadrature::sub_tetrahedron_indices(
      source.physical_vertices.size());
  const std::size_t points_per_subtetrahedron =
      treecode_quadrature::quadrature_point_count(4, algebraic_accuracy);
  if (source.q_point.size()
      != decomposition.size() * points_per_subtetrahedron) {
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
    if (pointInTetra(
            target, vertices[0], vertices[1], vertices[2], vertices[3])) {
      target_found = true;
      integral += singular_subtetrahedron_integral(
          target,
          vertices,
          algebraic_accuracy,
          source_component,
          enable_analytic_subtraction);
    } else {
      const std::size_t begin = subcell * points_per_subtetrahedron;
      integral += integrate_h2o_regular_quadrature_range(
          target,
          source,
          begin,
          begin + points_per_subtetrahedron);
    }
  }
  if (!target_found) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return integral;
}

bool target_uses_analytic_singular_treatment(
    const AFEPack::Point<DIM>& target,
    const ProgramOptions& options)
{
  if (options.h2o_near_field_mode != H2ONearFieldMode::quadrature
      || options.h2o_singular_mode == H2OSingularMode::duffy) {
    return false;
  }
  if (options.h2o_singular_mode == H2OSingularMode::analytic_all) {
    return true;
  }
  const double radius_squared =
      options.h2o_core_radius * options.h2o_core_radius;
  for (std::size_t atom = 0;
       atom < treecode_h2o_three_center_problem::atom_count;
       ++atom) {
    if (treecode_h2o_three_center_problem::squared_distance(
            target[0], target[1], target[2], atom)
        <= radius_squared) {
      return true;
    }
  }
  return false;
}

double evaluate_h2o_source_tree(
    const AFEPack::Point<DIM>& target,
    const H2OSourceTree& source,
    const hElementCache& interactions,
    const ProgramOptions& options,
    int containing_source_leaf_index,
    int source_component)
{
  std::array<double, treecode_multipole::maximum_coefficient_count>
      coefficient_workspace{};
  const int coefficient_count =
      treecode_multipole::coefficient_count(options.expansion_order);
  const bool enable_analytic_subtraction =
      target_uses_analytic_singular_treatment(target, options);
  double potential = 0.0;
  for (int leaf_index : interactions.neighbor_leaf_indices) {
    const ElementCache& leaf =
        source.element_cache[static_cast<std::size_t>(leaf_index)];
    if (options.h2o_near_field_mode == H2ONearFieldMode::quadrature) {
      potential += integrate_h2o_source_leaf_quadrature(
          target,
          leaf,
          leaf_index == containing_source_leaf_index,
          options.quadrature_accuracy,
          source_component,
          enable_analytic_subtraction);
    } else {
      potential += integrate_h2o_source_leaf_one_point(target, leaf);
    }
  }
  for (int hierarchy_index : interactions.far_hierarchy_indices) {
    const hElementCache& node =
        source.hierarchy_cache[static_cast<std::size_t>(hierarchy_index)];
    const AFEPack::Point<DIM> displacement = target - node.ref_point;
    if (treecode_multipole::build_taylor_coefficients(
            displacement[0], displacement[1], displacement[2],
            options.expansion_order, coefficient_workspace.data())) {
      for (int coefficient = 0;
           coefficient < coefficient_count;
           ++coefficient) {
        potential += node.moments[static_cast<std::size_t>(coefficient)]
            * coefficient_workspace[static_cast<std::size_t>(coefficient)];
      }
    }
  }
  return potential;
}

// Optional independent targets; does not change the production targets or timers.
// Cached MAC lists use distance minus the target-leaf radius, so they remain
// admissible for any point inside that leaf, not only its barycenter.
#ifndef H2O_VALIDATION_MAX_GRID
#define H2O_VALIDATION_MAX_GRID 128
#endif
void validate_h2o_common_targets(const H2OSourceTree& source,
                                const ProgramOptions& options)
{
  const char* requested = std::getenv("H2O_VALIDATION_GRID");
  if (!requested) return;
  const int m = parse_int(requested, "H2O_VALIDATION_GRID");
  if (m < 2 || m > H2O_VALIDATION_MAX_GRID || !options.compare_solution)
    throw std::runtime_error("H2O common validation requires compare=1 and grid 2.." +
                             std::to_string(H2O_VALIDATION_MAX_GRID));
  const bool focused = std::getenv("H2O_VALIDATION_FOCUSED") != nullptr;
  // A fixed tensor partition resolves the narrow Gaussian peaks without
  // depending on any adaptive source mesh. Four intervals on each outer
  // side retain coverage of the full cube; m intervals cover the inner box.
  const int side = focused ? m+8 : m;
  const int target_count = side*side*side;
  const double inner_lower[3] = {-2.5, -1.0, -1.0};
  const double inner_upper[3] = {2.5, 1.0, 2.5};
  const double start = omp_get_wtime();
  double error_squared = 0.0, norm_squared = 0.0;
  int failures = 0;
  #pragma omp parallel for schedule(dynamic,16) \
      reduction(+:error_squared,norm_squared,failures)
  for (int k = 0; k < target_count; ++k) {
    AFEPack::Point<DIM> point;
    const int cell[3] = {k%side, (k/side)%side, k/(side*side)};
    double weight = 1.0;
    unsigned int seed = static_cast<unsigned int>(k)+7351u;
    for (int d = 0; d < 3; ++d) {
      seed = 1664525u*seed+1013904223u;
      const double offset = (static_cast<double>(seed)+0.5)/4294967296.0;
      double lower = -5.0, width = 10.0/m;
      int local_cell = cell[d];
      if (focused) {
        if (cell[d] < 4) {
          width = (inner_lower[d]+5.0)/4;
        } else if (cell[d] < m+4) {
          lower = inner_lower[d]; width = (inner_upper[d]-lower)/m;
          local_cell -= 4;
        } else {
          lower = inner_upper[d]; width = (5.0-lower)/4;
          local_cell -= m+4;
        }
      }
      point[d] = lower + width*(local_cell+offset);
      weight *= width;
    }
    HElement<DIM>* leaf = findLeafContainingPoint(source.root_elements, point);
    if (!leaf) { ++failures; continue; }
    const hElementCache& interactions = source.hierarchy_cache[
        static_cast<std::size_t>(source.hierarchy_map.at(leaf))];
    const double value = evaluate_h2o_source_tree(
        point, source, interactions, options, leaf->index, -1);
    const double exact = u(point);
    if (!std::isfinite(value) || !std::isfinite(exact)) {
      ++failures; continue;
    }
    error_squared += weight*(value-exact)*(value-exact);
    norm_squared += weight*exact*exact;
  }
  if (failures || !(norm_squared > 0.0))
    throw std::runtime_error("Missing/nonfinite common H2O targets or zero reference norm");
  std::cout << std::setprecision(15)
            << "Common validation sampling: " << (focused ? "focused" : "uniform") << '\n'
            << "Common validation targets: " << target_count << '\n'
            << "Common validation absolute L2 error: " << std::sqrt(error_squared) << '\n'
            << "Common validation relative L2 error: " << std::sqrt(error_squared/norm_squared) << '\n'
            << "Common validation exact L2 norm: " << std::sqrt(norm_squared) << '\n'
            << "Common validation time (excluded): " << omp_get_wtime()-start << '\n';
}

int run_h2o_single_all_atoms_mesh(
    const ProgramOptions& options,
    double program_start_time,
    HGeometryTree<DIM>& h_tree,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements)
{
  std::cout
      << "\nSingle-mesh baseline: refining one mesh around O, H1, and H2\n";
  std::unique_ptr<FEMSpace<double, DIM>> fem_space;
  std::unique_ptr<IrregularMesh<DIM>> mesh = build_h2o_all_atoms_mesh(
      h_tree,
      options.initial_refinements,
      options.quadrature_accuracy,
      options.adaptation_tolerance,
      options.h2o_indicator_mode,
      template_elements,
      fem_space);

  H2OSourceTree source;
  double mapping_seconds = 0.0;
  double hierarchy_seconds = 0.0;
  double list_seconds = 0.0;
  int omp_threads_used = 1;
  prepare_h2o_source_tree(
      source,
      *mesh,
      *fem_space,
      -1,
      options,
      mapping_seconds,
      hierarchy_seconds,
      list_seconds,
      omp_threads_used);

  const std::size_t target_count = source.element_cache.size();
  std::size_t analytic_target_count = 0;
  for (const ElementCache& target_leaf : source.element_cache) {
    const hElementCache& target_cache =
        source.hierarchy_cache[static_cast<std::size_t>(
            target_leaf.hierarchy_index)];
    if (target_uses_analytic_singular_treatment(
            target_cache.ref_point, options)) {
      ++analytic_target_count;
    }
  }
  double source_integral = 0.0;
  for (const ElementCache& leaf : source.element_cache) {
    for (double charge : leaf.source_charge) {
      source_integral += charge;
    }
  }

  std::vector<double> potential(target_count, 0.0);
  double evaluation_seconds = 0.0;
  if (options.compare_solution) {
    const double evaluation_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 16) default(none) \
        shared(source, potential, options) firstprivate(target_count)
    for (std::size_t target_index = 0;
         target_index < target_count;
         ++target_index) {
      const ElementCache& target_leaf = source.element_cache[target_index];
      const hElementCache& target_interactions =
          source.hierarchy_cache[static_cast<std::size_t>(
              target_leaf.hierarchy_index)];
      potential[target_index] = evaluate_h2o_source_tree(
          target_interactions.ref_point,
          source,
          target_interactions,
          options,
          static_cast<int>(target_index),
          -1);
    }
    evaluation_seconds = omp_get_wtime() - evaluation_start;
  }

  const double algorithm_total_seconds =
      omp_get_wtime() - program_start_time;
  const double multipole_other_preparation_seconds = std::max(
      0.0,
      algorithm_total_seconds - list_seconds - evaluation_seconds);

  double exact_error_squared = 0.0;
  double exact_norm_squared = 0.0;
  double approximation_norm_squared = 0.0;
  std::size_t nonfinite_count = 0;
  if (options.compare_solution) {
    #pragma omp parallel for schedule(static) default(none) \
        shared(source, potential) firstprivate(target_count) \
        reduction(+:exact_error_squared, exact_norm_squared, \
                    approximation_norm_squared, nonfinite_count)
    for (std::size_t target_index = 0;
         target_index < target_count;
         ++target_index) {
      const ElementCache& target_leaf = source.element_cache[target_index];
      const hElementCache& target_cache =
          source.hierarchy_cache[static_cast<std::size_t>(
              target_leaf.hierarchy_index)];
      const double exact = u(target_cache.ref_point);
      const double approximation = potential[target_index];
      const double weight = target_cache.volume;
      if (!std::isfinite(exact)
          || !std::isfinite(approximation)
          || !std::isfinite(weight)) {
        ++nonfinite_count;
        continue;
      }
      const double error = approximation - exact;
      exact_error_squared += weight * error * error;
      exact_norm_squared += weight * exact * exact;
      approximation_norm_squared += weight * approximation * approximation;
    }
  }
  if (nonfinite_count != 0) {
    std::cerr << "H2O single-mesh comparison produced "
              << nonfinite_count << " nonfinite targets\n";
    return 1;
  }

  const double exact_error = std::sqrt(exact_error_squared);
  const double exact_norm = std::sqrt(exact_norm_squared);
  const double approximation_norm =
      std::sqrt(approximation_norm_squared);
  const double relative_error = exact_norm > 0.0
      ? exact_error / exact_norm : 0.0;
  std::cout << std::setprecision(12)
            << "Single all-atoms mesh: elements=" << target_count
            << ", integral(f)=" << source_integral << '\n'
            << "Analytic singular targets: " << analytic_target_count
            << '/' << target_count << " ("
            << (target_count == 0 ? 0.0
                : 100.0 * static_cast<double>(analytic_target_count)
                    / static_cast<double>(target_count))
            << "%)\n";
  if (options.compare_solution) {
    std::cout << "Single-mesh L2 norm of exact solution: "
              << exact_norm << '\n'
              << "Single-mesh L2 norm of treecode solution: "
              << approximation_norm << '\n'
              << "Single-mesh L2 error: " << exact_error << '\n'
              << "Single-mesh relative L2 error: "
              << relative_error << '\n';
  }
  print_h2o_timing_summary(
      "ONE ALL-ATOM MESH",
      list_seconds,
      multipole_other_preparation_seconds,
      evaluation_seconds,
      algorithm_total_seconds);

  if (!append_timing_result(
          options,
          1,
          target_count,
          list_seconds,
          multipole_other_preparation_seconds,
          evaluation_seconds,
          algorithm_total_seconds,
          options.compare_solution,
          exact_error,
          omp_threads_used)) {
    return 1;
  }
  std::cout << "Appended single-mesh result to "
            << timing_results_filename() << '\n';
  validate_h2o_common_targets(source, options);
  return 0;
}

int run_h2o_three_center_multimesh(
    const ProgramOptions& options,
    double program_start_time,
    HGeometryTree<DIM>& h_tree,
    IrregularMesh<DIM>& target_mesh,
    FEMSpace<double, DIM>& target_fem_space,
    std::vector<TemplateElement<double, DIM, DIM>>& template_elements)
{
  if (options.backend != ComputeBackend::cpu) {
    std::cerr
        << "The AFEPack H2O multi-mesh path currently requires CPU; "
           "select CPU for BACKEND.\n";
    return 1;
  }

  std::vector<hElementCache> target_hierarchy;
  std::map<HElement<DIM>*, int> target_map;
  double mapping_seconds = 0.0;
  double hierarchy_seconds = 0.0;
  int omp_threads_used = 1;
  update_hElementCache(
      target_mesh,
      target_hierarchy,
      target_map,
      options.mac_theta,
      false,
      &mapping_seconds,
      &hierarchy_seconds,
      &omp_threads_used);

  struct Target
  {
    AFEPack::Point<DIM> point;
    double weight = 0.0;
  };
  std::vector<Target> targets(
      static_cast<std::size_t>(target_fem_space.n_element()));
  for (const auto& entry : target_map) {
    if (!entry.first->isRefined()) {
      const int leaf_index = entry.first->index;
      if (leaf_index < 0
          || static_cast<std::size_t>(leaf_index) >= targets.size()) {
        throw std::runtime_error("Invalid H2O target leaf index");
      }
      targets[static_cast<std::size_t>(leaf_index)].point =
          target_hierarchy[static_cast<std::size_t>(entry.second)].ref_point;
      targets[static_cast<std::size_t>(leaf_index)].weight =
          target_hierarchy[static_cast<std::size_t>(entry.second)].volume;
    }
  }
  double target_volume = 0.0;
  for (const Target& target : targets) target_volume += target.weight;
  std::size_t analytic_target_count = 0;
  for (const Target& target : targets) {
    if (target_uses_analytic_singular_treatment(
            target.point, options)) {
      ++analytic_target_count;
    }
  }

  std::vector<double> summed_potential(targets.size(), 0.0);
  std::array<std::size_t,
             treecode_h2o_three_center_problem::atom_count> source_elements{};
  std::array<double,
             treecode_h2o_three_center_problem::atom_count> source_integrals{};
  double potential_evaluation_seconds = 0.0;
  double interaction_list_seconds = 0.0;
  for (std::size_t atom_index = 0;
       atom_index < treecode_h2o_three_center_problem::atom_count;
       ++atom_index) {
    std::cout << "Building AFEPack atom-centered mesh "
              << (atom_index == 0 ? "O" :
                  std::string("H") + std::to_string(atom_index))
              << " with radial indicator centered at ("
              << treecode_h2o_three_center_problem::centers[atom_index].x << ", "
              << treecode_h2o_three_center_problem::centers[atom_index].y << ", "
              << treecode_h2o_three_center_problem::centers[atom_index].z << ")\n";
    std::unique_ptr<FEMSpace<double, DIM>> owned_source_fem_space;
    std::unique_ptr<IrregularMesh<DIM>> owned_source_mesh;
    FEMSpace<double, DIM>* source_fem_space = nullptr;
    IrregularMesh<DIM>* source_mesh = nullptr;
    if (atom_index == 0) {
      // The common comparison grid is the oxygen-centered grid itself.  Reuse
      // it as the first source mesh so the calculation owns exactly three
      // atom-centered meshes rather than creating a fourth union mesh.
      target_mesh.regularMesh().writeOpenDXData("h2o_O_afepack.dx");
      source_mesh = &target_mesh;
      source_fem_space = &target_fem_space;
    } else {
      owned_source_mesh = build_h2o_atomic_mesh(
          h_tree,
          options.initial_refinements,
          options.quadrature_accuracy,
          options.adaptation_tolerance,
          options.h2o_indicator_mode,
          atom_index,
          template_elements,
          owned_source_fem_space);
      source_mesh = owned_source_mesh.get();
      source_fem_space = owned_source_fem_space.get();
    }
    H2OSourceTree source;
    double current_list_seconds = 0.0;
    prepare_h2o_source_tree(
        source,
        *source_mesh,
        *source_fem_space,
        static_cast<int>(atom_index),
        options,
        mapping_seconds,
        hierarchy_seconds,
        current_list_seconds,
        omp_threads_used);
    interaction_list_seconds += current_list_seconds;
    source_elements[atom_index] = source.element_cache.size();
    for (const ElementCache& element : source.element_cache) {
      for (std::size_t q = 0; q < element.source_charge.size(); ++q) {
        source_integrals[atom_index] += element.source_charge[q];
      }
    }
    // Each source mesh uses the same validated MAC interaction-list builder
    // as compare_exact.  A common H2O target locates its containing source
    // leaf and reuses that leaf's near/far partition.
    std::vector<int> target_source_hierarchy_indices(
        targets.size(), -1);
    std::vector<int> target_source_leaf_indices(
        targets.size(), -1);
    unsigned long long near_entry_count = 0;
    unsigned long long far_entry_count = 0;
    std::size_t missing_source_leaf_count = 0;
    const double target_mapping_start = omp_get_wtime();
    const std::size_t target_count = targets.size();
    #pragma omp parallel for schedule(dynamic, 32) default(none) \
        shared(targets, target_source_hierarchy_indices, \
               target_source_leaf_indices, source) \
        firstprivate(target_count) \
        reduction(+:near_entry_count, far_entry_count, \
                    missing_source_leaf_count)
    for (std::size_t target_index = 0;
         target_index < target_count;
         ++target_index) {
      HElement<DIM>* containing_leaf = findLeafContainingPoint(
          source.root_elements, targets[target_index].point);
      if (containing_leaf == nullptr) {
        ++missing_source_leaf_count;
        continue;
      }
      const int hierarchy_index = source.hierarchy_map.at(containing_leaf);
      target_source_hierarchy_indices[target_index] = hierarchy_index;
      target_source_leaf_indices[target_index] = containing_leaf->index;
      const hElementCache& lists = source.hierarchy_cache[
          static_cast<std::size_t>(hierarchy_index)];
      near_entry_count += lists.neighbor_leaf_indices.size();
      far_entry_count += lists.far_hierarchy_indices.size();
    }
    const double target_mapping_seconds =
        omp_get_wtime() - target_mapping_start;
    if (missing_source_leaf_count != 0) {
      throw std::runtime_error(
          "A common H2O target was not contained in a source-grid leaf");
    }

    double evaluation_seconds = 0.0;
    if (options.compare_solution) {
      const double evaluation_start = omp_get_wtime();
      const int source_component = static_cast<int>(atom_index);
      #pragma omp parallel for schedule(dynamic, 16) default(none) \
          shared(targets, summed_potential, source, \
                 target_source_hierarchy_indices, \
                 target_source_leaf_indices, options) \
          firstprivate(target_count, source_component)
      for (std::size_t target_index = 0;
           target_index < target_count;
           ++target_index) {
        summed_potential[target_index] += evaluate_h2o_source_tree(
            targets[target_index].point,
            source,
            source.hierarchy_cache[static_cast<std::size_t>(
                target_source_hierarchy_indices[target_index])],
            options,
            target_source_leaf_indices[target_index],
            source_component);
      }
      evaluation_seconds = omp_get_wtime() - evaluation_start;
      potential_evaluation_seconds += evaluation_seconds;
    }
    std::cout << "Three-center source mesh "
              << (atom_index == 0 ? "O" :
                  std::string("H") + std::to_string(atom_index))
              << ": elements=" << source_elements[atom_index]
              << ", integral(w_A f)=" << source_integrals[atom_index]
              << ", compare_exact MAC list construction="
              << current_list_seconds
              << " seconds, target-to-source-leaf mapping="
              << target_mapping_seconds
              << " seconds, evaluation=" << evaluation_seconds
              << " seconds, neighbor/non-neighbor entries="
              << near_entry_count << "/" << far_entry_count
              << ", coverage=validated over the complete source domain\n";
  }

  const double total_source_integral =
      source_integrals[0] + source_integrals[1] + source_integrals[2];
  const double exact_truncated_source_integral =
      treecode_h2o_three_center_problem::truncated_source_integral();
  const double fuzzy_charge_conservation_error = std::abs(
      total_source_integral - exact_truncated_source_integral);
  const double algorithm_total_seconds =
      omp_get_wtime() - program_start_time;
  const double multipole_other_preparation_seconds = std::max(
      0.0,
      algorithm_total_seconds
          - interaction_list_seconds
          - potential_evaluation_seconds);
  std::cout << "AFEPack atom-centered meshes: O/H1/H2 elements="
            << source_elements[0] << "/"
            << source_elements[1] << "/"
            << source_elements[2] << '\n'
            << "Becke-weighted source integrals O/H1/H2="
            << source_integrals[0] << "/"
            << source_integrals[1] << "/"
            << source_integrals[2] << ", sum=" << total_source_integral << '\n'
            << "Exact truncated-domain integral(f)="
            << exact_truncated_source_integral
            << ", fuzzy-cell quadrature charge error="
            << fuzzy_charge_conservation_error << '\n'
            << "Common target mesh (reused O grid): " << targets.size()
            << " leaf barycenters, volume=" << target_volume << '\n'
            << "Analytic singular targets: " << analytic_target_count
            << '/' << targets.size() << " ("
            << (targets.empty() ? 0.0
                : 100.0 * static_cast<double>(analytic_target_count)
                    / static_cast<double>(targets.size()))
            << "%), applied once per source mesh\n";

  double exact_error_squared = 0.0;
  double exact_norm_squared = 0.0;
  double approximation_norm_squared = 0.0;
  std::size_t nonfinite_count = 0;
  if (options.compare_solution) {
    const std::size_t target_count = targets.size();
    #pragma omp parallel for schedule(static) default(none) \
        shared(targets, summed_potential) firstprivate(target_count) \
        reduction(+:exact_error_squared, exact_norm_squared, \
                    approximation_norm_squared, nonfinite_count)
    for (std::size_t target_index = 0;
         target_index < target_count;
         ++target_index) {
      const double exact = u(targets[target_index].point);
      const double approximation = summed_potential[target_index];
      const double weight = targets[target_index].weight;
      if (!std::isfinite(exact)
          || !std::isfinite(approximation)
          || !std::isfinite(weight)) {
        ++nonfinite_count;
        continue;
      }
      const double error = approximation - exact;
      exact_error_squared += weight * error * error;
      exact_norm_squared += weight * exact * exact;
      approximation_norm_squared +=
          weight * approximation * approximation;
    }
  }
  if (nonfinite_count != 0) {
    std::cerr << "H2O multi-mesh comparison produced "
              << nonfinite_count << " nonfinite targets\n";
    return 1;
  }

  const double exact_error = std::sqrt(exact_error_squared);
  const double exact_norm = std::sqrt(exact_norm_squared);
  const double approximation_norm =
      std::sqrt(approximation_norm_squared);
  const double relative_error = exact_norm > 0.0
      ? exact_error / exact_norm : 0.0;
  if (options.compare_solution) {
    std::cout << std::setprecision(12)
              << "Three-center potential assembled as sum_A V[w_A f]\n"
              << "L2 norm of exact solution: " << exact_norm << '\n'
              << "L2 norm of treecode solution: "
              << approximation_norm << '\n'
              << "L2 error against exact solution: "
              << exact_error << '\n'
              << "Relative L2 error against exact solution: "
              << relative_error << '\n';
  } else {
    std::cout << "Comparison disabled; three AFEPack atom-centered source "
                 "meshes and multipoles were generated.\n";
  }
  print_h2o_timing_summary(
      "THREE ATOM-CENTERED MESHES",
      interaction_list_seconds,
      multipole_other_preparation_seconds,
      potential_evaluation_seconds,
      algorithm_total_seconds);
  if (!append_timing_result(
          options,
          3,
          targets.size(),
          interaction_list_seconds,
          multipole_other_preparation_seconds,
          potential_evaluation_seconds,
          algorithm_total_seconds,
          options.compare_solution,
          exact_error,
          omp_threads_used)) {
    return 1;
  }
  std::cout << "Appended H2O multi-mesh timings to "
            << timing_results_filename() << '\n';
  return 0;
}
#endif


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

  if (use_gpu_backend) {
    std::cerr
        << "The standalone three-grid H2O test currently supports CPU "
           "only; select CPU for BACKEND.\n";
    return 1;
  }

  std::cout << "Configuration: initial_refinements="
            << options.initial_refinements
            << ", quadrature_accuracy=" << options.quadrature_accuracy
            << ", expansion_order=" << options.expansion_order
            << ", theta=" << options.mac_theta
            << ", backend=" << backend_name(options.backend)
            << ", mesh_mode="
            << h2o_mesh_mode_name(options.h2o_mesh_mode)
            << ", near_field_mode="
            << h2o_near_field_mode_name(options.h2o_near_field_mode)
            << ", adaptation_tolerance="
            << options.adaptation_tolerance
            << ", refinement_indicator="
            << h2o_indicator_mode_name(options.h2o_indicator_mode)
            << ", singular_mode="
            << h2o_singular_mode_name(options.h2o_singular_mode)
            << ", atom_core_radius=" << options.h2o_core_radius
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

#ifdef TREECODE_H2O_THREE_CENTER_MODEL
  std::cout << "Density model: "
            << treecode_h2o_three_center_problem::description() << '\n';
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

  // SINGLE does not need the separately constructed O-centered target mesh.
  // Dispatch here so its total excludes that unused three-mesh preparation.
  if (options.h2o_mesh_mode == H2OMeshMode::single) {
    return run_h2o_single_all_atoms_mesh(
        options, program_start_time, h_tree, template_elements);
  }
 
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

   
    /// 下面一段计算用于加密的指示子。
    Indicator<DIM> indicator(_r_mesh);
    get_indicator(
        indicator, _r_mesh, fem_space, options.quadrature_accuracy,
        use_gpu_backend, 0);

    /// 下面的几行调用进行自适应的函数，都是套话。
    std::unique_ptr<IrregularMesh<DIM>> old_ir_mesh = std::make_unique<IrregularMesh<DIM>>(*ir_mesh);
  ir_mesh = std::make_unique<IrregularMesh<DIM>>(); 

    
  MeshAdaptor<DIM> mesh_adaptor(*old_ir_mesh, *ir_mesh);
  mesh_adaptor.setIndicator(indicator);
  mesh_adaptor.tolerence() = 10.0e-03;
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
        indicator1, r_mesh1, fem_space, options.quadrature_accuracy,
        use_gpu_backend, 0);

    /// 下面的几行调用进行自适应的函数，都是套话。
    std::unique_ptr<IrregularMesh<DIM>> old_ir_mesh1 = std::make_unique<IrregularMesh<DIM>>(*ir_mesh);
  ir_mesh = std::make_unique<IrregularMesh<DIM>>(); 

    
  MeshAdaptor<DIM> mesh_adaptor1(*old_ir_mesh1, *ir_mesh);
  mesh_adaptor1.setIndicator(indicator1);
  mesh_adaptor1.tolerence() = options.adaptation_tolerance;
  mesh_adaptor1.convergenceOrder() = 0.;
  mesh_adaptor1.refineStep() = 2;
  mesh_adaptor1.adapt();

  //   } while(i < atoi(argv[2]));   
  ir_mesh->semiregularize();
  ir_mesh->regularize(false);
  RegularMesh<DIM>& r_mesh2 = ir_mesh->regularMesh();
  r_mesh2.writeOpenDXData("D.dx");
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

  // The oxygen-centered mesh above is reused as the common target mesh and
  // as the first of the three independently centered source meshes.
  const int three_mesh_status = run_h2o_three_center_multimesh(
      options,
      program_start_time,
      h_tree,
      *ir_mesh,
      fem_space,
      template_elements);
  if (three_mesh_status != 0
      || options.h2o_mesh_mode == H2OMeshMode::three) {
    return three_mesh_status;
  }
  const double single_mesh_start_time = omp_get_wtime();
  return run_h2o_single_all_atoms_mesh(
      options, single_mesh_start_time, h_tree, template_elements);
}
