#ifndef TREECODE_COMPARE_GPU_HPP
#define TREECODE_COMPARE_GPU_HPP

#include <cstddef>
#include <string>
#include <vector>

struct GpuPoint3
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct GpuQuadratureSample
{
  GpuPoint3 point;
  double weight = 0.0;
};

struct GpuLeaf
{
  int sample_offset = 0;
  int sample_count = 0;
  GpuPoint3 vertices[4]{};
};

struct GpuQuadratureElement
{
  GpuPoint3 vertices[7]{};
  int vertex_count = 0;
  int sample_offset = 0;
  int sample_count = 0;
  int leaf_index = -1;
  int hierarchy_index = -1;
};

struct GpuMultipole
{
  GpuPoint3 center;
  double radius = 0.0;
  double volume = 0.0;
  int moment_offset = 0;
  int level = 0;
  int leaf_begin = 0;
  int leaf_end = 0;
  int leaf_index = -1;
};

struct GpuTreecodeProblem
{
  std::vector<GpuPoint3> targets;
  std::vector<int> target_hierarchy_indices;
  std::vector<int> target_leaf_indices;
  // Integration weights associated with target points.  These are distinct
  // from source quadrature weights when one barycenter is evaluated per leaf.
  std::vector<double> target_weights;

  std::vector<GpuLeaf> leaves;
  std::vector<GpuQuadratureSample> samples;
  std::vector<GpuQuadratureElement> quadrature_elements;
  std::size_t generated_sample_count = 0;
  std::vector<GpuMultipole> multipoles;

  // Raw hierarchy data used to construct near/far CSR lists and multipole
  // moments directly on the GPU.
  std::vector<int> leaf_hierarchy_indices;
  std::vector<int> root_hierarchy_indices;
  std::vector<int> child_offsets;
  std::vector<int> child_hierarchy_indices;
  double mac_theta = 0.5;
  double domain_volume = 0.0;
  bool targets_match_source_samples = false;
  bool generate_samples_on_gpu = false;
  bool generate_exact_targets_on_gpu = false;
};

struct GpuTreecodeResult
{
  std::vector<double> potentials;
  std::string device_name;
  double quadrature_seconds = 0.0;
  double source_preparation_seconds = 0.0;
  double moment_seconds = 0.0;
  double interaction_seconds = 0.0;
  double kernel_seconds = 0.0;
  double norm_reduction_seconds = 0.0;
  double exact_error_squared = 0.0;
  double exact_norm_squared = 0.0;
  double approximation_norm_squared = 0.0;
  double exact_error = 0.0;
  double exact_norm = 0.0;
  double approximation_norm = 0.0;
  double relative_exact_error = 0.0;
  std::size_t evaluation_count = 0;
  std::size_t nonfinite_count = 0;
  std::size_t near_interaction_count = 0;
  std::size_t far_interaction_count = 0;
  int evaluation_threads_per_target = 0;
  bool has_l2_norms = false;
};

GpuTreecodeResult evaluate_treecode_gpu(
    const GpuTreecodeProblem& problem,
    int expansion_order,
    int quadrature_accuracy,
    bool calculate_exact_l2_norms = false,
    bool evaluate_potentials = true);

std::vector<double> evaluate_refinement_indicator_gpu(
    const std::vector<GpuQuadratureElement>& elements,
    int quadrature_accuracy,
    double* kernel_seconds = nullptr);

#endif
