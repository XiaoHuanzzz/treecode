#include "compare_gpu.hpp"
#include "cartesian_multipole.hpp"
#include "gaussian_test_problem.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double PI = treecode_test_problem::pi;

void check_cuda(cudaError_t status, const char* operation)
{
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

std::vector<std::pair<double, double>>
gauss_legendre_on_unit_interval(int point_count)
{
  std::vector<std::pair<double, double>> rule(
      static_cast<std::size_t>(point_count));
  const int root_count = (point_count + 1) / 2;
  for (int root = 0; root < root_count; ++root) {
    double x = std::cos(
        PI * (static_cast<double>(root) + 0.75)
        / (static_cast<double>(point_count) + 0.5));
    double derivative = 0.0;
    for (int iteration = 0; iteration < 100; ++iteration) {
      double previous = 1.0;
      double current = x;
      for (int degree = 2; degree <= point_count; ++degree) {
        const double next =
            ((2.0 * degree - 1.0) * x * current
             - (degree - 1.0) * previous) / degree;
        previous = current;
        current = next;
      }
      derivative =
          point_count * (x * current - previous) / (x * x - 1.0);
      const double step = current / derivative;
      x -= step;
      if (std::abs(step)
          <= 8.0 * std::numeric_limits<double>::epsilon()
             * std::max(1.0, std::abs(x))) {
        break;
      }
    }
    double previous = 1.0;
    double current = x;
    for (int degree = 2; degree <= point_count; ++degree) {
      const double next =
          ((2.0 * degree - 1.0) * x * current
           - (degree - 1.0) * previous) / degree;
      previous = current;
      current = next;
    }
    derivative =
        point_count * (x * current - previous) / (x * x - 1.0);
    const double weight =
        1.0 / ((1.0 - x * x) * derivative * derivative);
    rule[static_cast<std::size_t>(root)] =
        {0.5 * (1.0 - x), weight};
    rule[static_cast<std::size_t>(point_count - 1 - root)] =
        {0.5 * (1.0 + x), weight};
  }
  return rule;
}

template <class T>
class DeviceBuffer
{
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count)
      : count_(count)
  {
    if (count_ != 0) {
      check_cuda(
          cudaMalloc(
              reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
          "cudaMalloc");
    }
  }

  explicit DeviceBuffer(const std::vector<T>& values)
      : DeviceBuffer(values.size())
  {
    if (!values.empty()) {
      check_cuda(
          cudaMemcpy(
              data_, values.data(), values.size() * sizeof(T),
              cudaMemcpyHostToDevice),
          "cudaMemcpy host to device");
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0))
  {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
  {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  ~DeviceBuffer()
  {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }

  void copy_to(std::vector<T>& values) const
  {
    if (values.size() != count_) {
      throw std::invalid_argument(
          "CUDA destination vector has the wrong size");
    }
    if (count_ != 0) {
      check_cuda(
          cudaMemcpy(
              values.data(), data_, count_ * sizeof(T),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy device to host");
    }
  }

private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class CudaEvent
{
public:
  CudaEvent()
  {
    check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  ~CudaEvent()
  {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

__device__ double density_device(const GpuPoint3& point)
{
  return treecode_test_problem::source_value(
      point.x, point.y, point.z);
}

__device__ double quiet_nan_device()
{
  return __longlong_as_double(0x7ff8000000000000ULL);
}

__device__ GpuPoint3 subtract(
    const GpuPoint3& left,
    const GpuPoint3& right)
{
  return {
      left.x - right.x,
      left.y - right.y,
      left.z - right.z};
}

__device__ double determinant(
    const GpuPoint3& a,
    const GpuPoint3& b,
    const GpuPoint3& c)
{
  return
      a.x * (b.y * c.z - b.z * c.y)
      - a.y * (b.x * c.z - b.z * c.x)
      + a.z * (b.x * c.y - b.y * c.x);
}

__device__ double point_dot(
    const GpuPoint3& left,
    const GpuPoint3& right)
{
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

__device__ GpuPoint3 point_cross(
    const GpuPoint3& left,
    const GpuPoint3& right)
{
  return {
      left.y * right.z - left.z * right.y,
      left.z * right.x - left.x * right.z,
      left.x * right.y - left.y * right.x};
}

__device__ GpuPoint3 point_scale(const GpuPoint3& point, double factor)
{
  return {factor * point.x, factor * point.y, factor * point.z};
}

__device__ double point_norm(const GpuPoint3& point)
{
  return sqrt(point_dot(point, point));
}

__device__ double constant_density_tetrahedron_coulomb_integral_device(
    const GpuPoint3& target,
    const GpuLeaf& source)
{
  constexpr int faces[4][4] = {
      {1, 2, 3, 0},
      {0, 3, 2, 1},
      {0, 1, 3, 2},
      {0, 2, 1, 3}};
  double coordinate_scale = 1.0;
  for (int vertex = 0; vertex < 4; ++vertex) {
    coordinate_scale = fmax(
        coordinate_scale,
        fmax(
            fabs(source.vertices[vertex].x),
            fmax(
                fabs(source.vertices[vertex].y),
                fabs(source.vertices[vertex].z))));
  }
  const double tolerance =
      256.0 * 2.2204460492503131e-16 * coordinate_scale;
  double volume_integral = 0.0;
  for (int face_index = 0; face_index < 4; ++face_index) {
    GpuPoint3 triangle[3] = {
        source.vertices[faces[face_index][0]],
        source.vertices[faces[face_index][1]],
        source.vertices[faces[face_index][2]]};
    const GpuPoint3 opposite =
        source.vertices[faces[face_index][3]];
    GpuPoint3 edge01 = subtract(triangle[1], triangle[0]);
    GpuPoint3 edge02 = subtract(triangle[2], triangle[0]);
    GpuPoint3 normal = point_cross(edge01, edge02);
    double normal_length = point_norm(normal);
    if (!(normal_length > tolerance)) return quiet_nan_device();
    if (point_dot(subtract(opposite, triangle[0]), normal) > 0.0) {
      const GpuPoint3 swap = triangle[1];
      triangle[1] = triangle[2];
      triangle[2] = swap;
      edge01 = subtract(triangle[1], triangle[0]);
      edge02 = subtract(triangle[2], triangle[0]);
      normal = point_cross(edge01, edge02);
      normal_length = point_norm(normal);
    }
    normal = point_scale(normal, 1.0 / normal_length);
    GpuPoint3 relative[3] = {
        subtract(triangle[0], target),
        subtract(triangle[1], target),
        subtract(triangle[2], target)};
    const double radius[3] = {
        point_norm(relative[0]),
        point_norm(relative[1]),
        point_norm(relative[2])};
    double height = point_dot(relative[0], normal);
    if (height < -tolerance) return quiet_nan_device();
    height = fmax(0.0, height);
    if (height <= tolerance) continue;

    const double solid_angle_numerator = fabs(point_dot(
        relative[0], point_cross(relative[1], relative[2])));
    const double solid_angle_denominator =
        radius[0] * radius[1] * radius[2]
        + radius[0] * point_dot(relative[1], relative[2])
        + radius[1] * point_dot(relative[2], relative[0])
        + radius[2] * point_dot(relative[0], relative[1]);
    const double solid_angle = 2.0 * atan2(
        solid_angle_numerator, solid_angle_denominator);
    double surface_integral = -height * solid_angle;
    for (int edge = 0; edge < 3; ++edge) {
      const int next = (edge + 1) % 3;
      const GpuPoint3 edge_vector = subtract(
          triangle[next], triangle[edge]);
      const double edge_length = point_norm(edge_vector);
      if (!(edge_length > tolerance)) return quiet_nan_device();
      const GpuPoint3 edge_direction =
          point_scale(edge_vector, 1.0 / edge_length);
      const GpuPoint3 inward_normal =
          point_cross(normal, edge_direction);
      const double signed_distance =
          -point_dot(relative[edge], inward_normal);
      const double radial_sum = radius[edge] + radius[next];
      const double denominator = radial_sum - edge_length;
      if (!(denominator > tolerance)) {
        if (fabs(signed_distance) <= tolerance) continue;
        return quiet_nan_device();
      }
      surface_integral += signed_distance
          * log1p(2.0 * edge_length / denominator);
    }
    volume_integral += 0.5 * height * surface_integral;
  }
  return volume_integral;
}

__device__ double block_sum(double value, double* workspace);

__device__ inline void evaluation_sync()
{
  if (blockDim.x == 32) {
    __syncwarp();
  } else {
    __syncthreads();
  }
}

__device__ double singular_tetrahedron_integral_device(
    const GpuPoint3& target,
    const GpuLeaf& source,
    const double* nodes,
    const double* weights,
    int axis_point_count,
    double* reduction_workspace)
{
  constexpr int faces[4][3] = {
      {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

  // For fixed (face,r,s), the face geometry, radial direction and distance
  // do not depend on the radial Duffy coordinate t.  Reuse those quantities
  // for all t points instead of rebuilding them axis_point_count times.
  if (static_cast<int>(threadIdx.x) == 0) {
    reduction_workspace[0] = density_device(target);
    reduction_workspace[1] =
        constant_density_tetrahedron_coulomb_integral_device(
            target, source);
    reduction_workspace[2] =
        isfinite(reduction_workspace[0])
            && isfinite(reduction_workspace[1]) ? 1.0 : 0.0;
  }
  evaluation_sync();
  const double target_density = reduction_workspace[0];
  const double analytic_kernel = reduction_workspace[1];
  const bool use_analytic_subtraction = reduction_workspace[2] != 0.0;

  const int points_per_face = axis_point_count * axis_point_count;
  const int total_point_count = 4 * points_per_face;
  double local_integral =
      static_cast<int>(threadIdx.x) == 0 && use_analytic_subtraction
          ? target_density * analytic_kernel : 0.0;
  for (int linear_index = static_cast<int>(threadIdx.x);
       linear_index < total_point_count;
       linear_index += static_cast<int>(blockDim.x)) {
    const int face_index = linear_index / points_per_face;
    int axis_index = linear_index % points_per_face;
    const int ir = axis_index % axis_point_count;
    const int is = axis_index / axis_point_count;
    const GpuPoint3 a = source.vertices[faces[face_index][0]];
    const GpuPoint3 b = source.vertices[faces[face_index][1]];
    const GpuPoint3 c = source.vertices[faces[face_index][2]];
    const double det = fabs(determinant(
        subtract(a, target), subtract(b, target), subtract(c, target)));
    if (det <= 2.2204460492503131e-16) {
      continue;
    }

    const double s = nodes[is];
    const double r = nodes[ir];
    const GpuPoint3 face_point{
        (1.0 - s) * a.x
            + s * (1.0 - r) * b.x + s * r * c.x,
        (1.0 - s) * a.y
            + s * (1.0 - r) * b.y + s * r * c.y,
        (1.0 - s) * a.z
            + s * (1.0 - r) * b.z + s * r * c.z};
    const GpuPoint3 radial = subtract(face_point, target);
    const double radial_length = sqrt(
        radial.x * radial.x
        + radial.y * radial.y
        + radial.z * radial.z);
    if (radial_length <= 2.2250738585072014e-308) {
      continue;
    }
    const double reusable_weight =
        weights[is] * weights[ir] * det * s / radial_length;
    for (int it = 0; it < axis_point_count; ++it) {
      const double t = nodes[it];
      const GpuPoint3 point{
          target.x + t * radial.x,
          target.y + t * radial.y,
          target.z + t * radial.z};

      // det*t^2*s / (t*|radial|) is the transformed 1/r Jacobian.
      double density = density_device(point);
      if (use_analytic_subtraction) density -= target_density;
      local_integral += weights[it] * t * reusable_weight * density;
    }
  }
  return block_sum(local_integral, reduction_workspace);
}

__device__ double block_sum(double value, double* workspace)
{
  const int lane = static_cast<int>(threadIdx.x);
  if (blockDim.x == 32) {
    for (int offset = 16; offset > 0; offset /= 2) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return __shfl_sync(0xffffffffU, value, 0);
  }
  workspace[lane] = value;
  evaluation_sync();
  for (int stride = static_cast<int>(blockDim.x) / 2;
       stride > 0;
       stride /= 2) {
    if (lane < stride) {
      workspace[lane] += workspace[lane + stride];
    }
    evaluation_sync();
  }
  return workspace[0];
}

[[maybe_unused]] __global__ void evaluate_treecode_kernel(
    const GpuPoint3* targets,
    const int* target_hierarchy_indices,
    const int* target_leaf_indices,
    int target_count,
    const GpuLeaf* leaves,
    int leaf_count,
    const GpuQuadratureSample* samples,
    const double* sample_charges,
    const GpuMultipole* multipoles,
    const double* moments,
    int moment_count,
    int hierarchy_count,
    const int* near_offsets,
    const int* near_leaf_indices,
    const int* far_offsets,
    const int* far_hierarchy_indices,
    int expansion_order,
    const double* singular_nodes,
    const double* singular_weights,
    int singular_axis_point_count,
    double* potentials)
{
  const int target_index = static_cast<int>(blockIdx.x);
  if (target_index >= target_count) {
    return;
  }

  // Size this storage for the requested expansion order at launch time.
  // The former maximum-order allocation reserved 1771 doubles even for the
  // common P=4 case, which needs only 35 and severely limited occupancy.
  extern __shared__ double shared_values[];
  const int coefficient_count =
      treecode_multipole::coefficient_count(expansion_order);
  double* coefficients = shared_values;
  double* reduction_workspace = shared_values + coefficient_count;
  __shared__ double approximation;
  __shared__ int singular_hit;

  const int lane = static_cast<int>(threadIdx.x);
  const int hierarchy_index = target_hierarchy_indices[target_index];
  const int target_leaf_index = target_leaf_indices[target_index];
  const bool target_valid =
      hierarchy_index >= 0 && hierarchy_index < hierarchy_count
      && target_leaf_index >= 0 && target_leaf_index < leaf_count;
  if (!target_valid) {
    if (lane == 0) {
      potentials[target_index] = quiet_nan_device();
    }
    return;
  }
  if (lane == 0) {
    approximation = 0.0;
  }

  const GpuPoint3 target = targets[target_index];

  for (int entry = near_offsets[target_leaf_index];
       entry < near_offsets[target_leaf_index + 1];
       ++entry) {
    const int source_leaf_index = near_leaf_indices[entry];
    if (source_leaf_index < 0 || source_leaf_index >= leaf_count) {
      if (lane == 0) {
        potentials[target_index] = quiet_nan_device();
      }
      return;
    }
    const GpuLeaf& source_leaf = leaves[source_leaf_index];
    if (source_leaf_index == target_leaf_index) {
      const double singular_value =
          singular_tetrahedron_integral_device(
              target,
              source_leaf,
              singular_nodes,
              singular_weights,
              singular_axis_point_count,
              reduction_workspace);
      if (lane == 0) {
        approximation += singular_value;
      }
      continue;
    }

    if (lane == 0) {
      singular_hit = 0;
    }
    evaluation_sync();
    double near_value = 0.0;
    for (int sample_index = source_leaf.sample_offset + lane;
         sample_index <
             source_leaf.sample_offset + source_leaf.sample_count;
         sample_index += static_cast<int>(blockDim.x)) {
      const GpuQuadratureSample& sample = samples[sample_index];
      const double dx = sample.point.x - target.x;
      const double dy = sample.point.y - target.y;
      const double dz = sample.point.z - target.z;
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (distance_squared <= 1.0e-28) {
        atomicExch(&singular_hit, 1);
      } else {
        near_value += sample_charges[sample_index]
            / sqrt(distance_squared);
      }
    }
    const double near_sum = block_sum(near_value, reduction_workspace);
    if (singular_hit) {
      const double singular_value =
          singular_tetrahedron_integral_device(
              target,
              source_leaf,
              singular_nodes,
              singular_weights,
              singular_axis_point_count,
              reduction_workspace);
      if (lane == 0) {
        approximation += singular_value;
      }
    } else if (lane == 0) {
      approximation += near_sum;
    }
  }

  for (int entry = far_offsets[target_leaf_index];
       entry < far_offsets[target_leaf_index + 1];
       ++entry) {
    const int source_hierarchy_index = far_hierarchy_indices[entry];
    if (source_hierarchy_index < 0
        || source_hierarchy_index >= hierarchy_count) {
      if (lane == 0) {
        potentials[target_index] = quiet_nan_device();
      }
      return;
    }
    const GpuMultipole& source = multipoles[source_hierarchy_index];
    if (source.moment_offset < 0
        || source.moment_offset + coefficient_count > moment_count) {
      if (lane == 0) {
        potentials[target_index] = quiet_nan_device();
      }
      return;
    }

    const double rx = target.x - source.center.x;
    const double ry = target.y - source.center.y;
    const double rz = target.z - source.center.z;
    const double radius_squared = rx * rx + ry * ry + rz * rz;
    if (lane == 0) {
      coefficients[0] =
          radius_squared >= 1.0e-30
              ? 1.0 / sqrt(radius_squared) : 0.0;
    }
    evaluation_sync();
    if (radius_squared < 1.0e-30) {
      continue;
    }

    for (int degree = 1; degree <= expansion_order; ++degree) {
      const double first_factor =
          (2.0 * degree - 1.0) / (degree * radius_squared);
      const double second_factor =
          (degree - 1.0) / (degree * radius_squared);
      int degree_index = 0;
      for (int x = 0; x <= degree; ++x) {
        for (int y = 0; y <= degree - x; ++y, ++degree_index) {
          if (degree_index % static_cast<int>(blockDim.x) != lane) {
            continue;
          }
          const int z = degree - x - y;
          double first = 0.0;
          double second = 0.0;
          if (x >= 1) {
            first += rx * coefficients[
                treecode_multipole::coefficient_index(x - 1, y, z)];
          }
          if (y >= 1) {
            first += ry * coefficients[
                treecode_multipole::coefficient_index(x, y - 1, z)];
          }
          if (z >= 1) {
            first += rz * coefficients[
                treecode_multipole::coefficient_index(x, y, z - 1)];
          }
          if (x >= 2) {
            second += coefficients[
                treecode_multipole::coefficient_index(x - 2, y, z)];
          }
          if (y >= 2) {
            second += coefficients[
                treecode_multipole::coefficient_index(x, y - 2, z)];
          }
          if (z >= 2) {
            second += coefficients[
                treecode_multipole::coefficient_index(x, y, z - 2)];
          }
          coefficients[treecode_multipole::coefficient_index(x, y, z)] =
              first_factor * first - second_factor * second;
        }
      }
      evaluation_sync();
    }

    double far_value = 0.0;
    for (int index = lane;
         index < coefficient_count;
         index += static_cast<int>(blockDim.x)) {
      far_value +=
          moments[source.moment_offset + index] * coefficients[index];
    }
    const double far_sum = block_sum(far_value, reduction_workspace);
    if (lane == 0) {
      approximation += far_sum;
    }
  }

  if (lane == 0) {
    potentials[target_index] = approximation;
  }
}

// A100-oriented evaluator.  A complete warp owns one interaction at a time,
// while several warps cooperate on the same target.  Near and far partial
// sums are retained in registers and reduced only once per target; the former
// kernel performed a warp reduction for every near leaf and every far node.
template <int fixed_expansion_order>
__global__ void evaluate_treecode_multiwarp_kernel(
    const GpuPoint3* targets,
    const int* target_hierarchy_indices,
    const int* target_leaf_indices,
    int target_count,
    const GpuLeaf* leaves,
    int leaf_count,
    const GpuQuadratureSample* samples,
    const double* sample_charges,
    const GpuMultipole* multipoles,
    const double* moments,
    int moment_count,
    int hierarchy_count,
    const int* near_offsets,
    const int* near_leaf_indices,
    const int* far_offsets,
    const int* far_hierarchy_indices,
    int expansion_order,
    const double* singular_nodes,
    const double* singular_weights,
    int singular_axis_point_count,
    double* potentials)
{
  constexpr int warp_size = 32;
  const int target_index = static_cast<int>(blockIdx.x);
  if (target_index >= target_count) {
    return;
  }

  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & (warp_size - 1);
  const int warp = thread / warp_size;
  const int warp_count = static_cast<int>(blockDim.x) / warp_size;
  const int effective_expansion_order =
      fixed_expansion_order > 0
          ? fixed_expansion_order : expansion_order;
  const int coefficient_count = treecode_multipole::coefficient_count(
      effective_expansion_order);

  // Layout: one independent coefficient workspace per warp, followed by a
  // block-wide singular-integral reduction area and two warp-result arrays.
  extern __shared__ double shared_values[];
  double* warp_coefficients =
      shared_values + warp * coefficient_count;
  double* singular_workspace =
      shared_values + warp_count * coefficient_count;
  double* near_warp_sums = singular_workspace + blockDim.x;
  double* far_warp_sums = near_warp_sums + warp_count;
  __shared__ int invalid_target;

  const int hierarchy_index = target_hierarchy_indices[target_index];
  const int target_leaf_index = target_leaf_indices[target_index];
  const bool target_valid =
      hierarchy_index >= 0 && hierarchy_index < hierarchy_count
      && target_leaf_index >= 0 && target_leaf_index < leaf_count;
  if (!target_valid) {
    if (thread == 0) {
      potentials[target_index] = quiet_nan_device();
    }
    return;
  }
  if (thread == 0) {
    invalid_target = 0;
  }
  __syncthreads();

  const GpuPoint3 target = targets[target_index];

  // The target-element integral is the only singular interaction for an
  // interior tetrahedron barycenter.  Compute it once with the full block so
  // the remaining near work needs no branches or atomics.
  const double singular_value = singular_tetrahedron_integral_device(
      target,
      leaves[target_leaf_index],
      singular_nodes,
      singular_weights,
      singular_axis_point_count,
      singular_workspace);

  double near_value = 0.0;
  const int near_begin = near_offsets[target_leaf_index];
  const int near_end = near_offsets[target_leaf_index + 1];
  for (int entry = near_begin + warp;
       entry < near_end;
       entry += warp_count) {
    const int source_leaf_index = near_leaf_indices[entry];
    if (source_leaf_index < 0 || source_leaf_index >= leaf_count) {
      if (lane == 0) {
        atomicExch(&invalid_target, 1);
      }
      continue;
    }
    if (source_leaf_index == target_leaf_index) {
      continue;
    }
    const GpuLeaf source_leaf = leaves[source_leaf_index];
    const int sample_end =
        source_leaf.sample_offset + source_leaf.sample_count;
    for (int sample_index = source_leaf.sample_offset + lane;
         sample_index < sample_end;
         sample_index += warp_size) {
      const GpuQuadratureSample sample = samples[sample_index];
      const double dx = sample.point.x - target.x;
      const double dy = sample.point.y - target.y;
      const double dz = sample.point.z - target.z;
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (distance_squared <= 1.0e-28) {
        atomicExch(&invalid_target, 1);
      } else {
        near_value += sample_charges[sample_index]
            / sqrt(distance_squared);
      }
    }
  }
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    near_value += __shfl_down_sync(0xffffffffU, near_value, offset);
  }
  if (lane == 0) {
    near_warp_sums[warp] = near_value;
  }

  double far_value = 0.0;
  const int far_begin = far_offsets[target_leaf_index];
  const int far_end = far_offsets[target_leaf_index + 1];
  for (int entry = far_begin + warp;
       entry < far_end;
       entry += warp_count) {
    const int source_hierarchy_index = far_hierarchy_indices[entry];
    if (source_hierarchy_index < 0
        || source_hierarchy_index >= hierarchy_count) {
      if (lane == 0) {
        atomicExch(&invalid_target, 1);
      }
      continue;
    }
    const GpuMultipole source = multipoles[source_hierarchy_index];
    if (source.moment_offset < 0
        || source.moment_offset + coefficient_count > moment_count) {
      if (lane == 0) {
        atomicExch(&invalid_target, 1);
      }
      continue;
    }

    const double rx = target.x - source.center.x;
    const double ry = target.y - source.center.y;
    const double rz = target.z - source.center.z;
    const double radius_squared = rx * rx + ry * ry + rz * rz;
    if (lane == 0) {
      warp_coefficients[0] =
          radius_squared >= 1.0e-30
              ? 1.0 / sqrt(radius_squared) : 0.0;
    }
    __syncwarp();
    if (radius_squared < 1.0e-30) {
      if (lane == 0) {
        atomicExch(&invalid_target, 1);
      }
      continue;
    }

    for (int degree = 1;
         degree <= effective_expansion_order;
         ++degree) {
      const double first_factor =
          (2.0 * degree - 1.0) / (degree * radius_squared);
      const double second_factor =
          (degree - 1.0) / (degree * radius_squared);
      int degree_index = 0;
      for (int x = 0; x <= degree; ++x) {
        for (int y = 0; y <= degree - x; ++y, ++degree_index) {
          if (degree_index % warp_size != lane) {
            continue;
          }
          const int z = degree - x - y;
          double first = 0.0;
          double second = 0.0;
          if (x >= 1) {
            first += rx * warp_coefficients[
                treecode_multipole::coefficient_index(x - 1, y, z)];
          }
          if (y >= 1) {
            first += ry * warp_coefficients[
                treecode_multipole::coefficient_index(x, y - 1, z)];
          }
          if (z >= 1) {
            first += rz * warp_coefficients[
                treecode_multipole::coefficient_index(x, y, z - 1)];
          }
          if (x >= 2) {
            second += warp_coefficients[
                treecode_multipole::coefficient_index(x - 2, y, z)];
          }
          if (y >= 2) {
            second += warp_coefficients[
                treecode_multipole::coefficient_index(x, y - 2, z)];
          }
          if (z >= 2) {
            second += warp_coefficients[
                treecode_multipole::coefficient_index(x, y, z - 2)];
          }
          warp_coefficients[
              treecode_multipole::coefficient_index(x, y, z)] =
              first_factor * first - second_factor * second;
        }
      }
      __syncwarp();
    }

    for (int coefficient = lane;
         coefficient < coefficient_count;
         coefficient += warp_size) {
      far_value += moments[source.moment_offset + coefficient]
          * warp_coefficients[coefficient];
    }
    // All lanes must finish reading this interaction's coefficients before
    // lane zero overwrites coefficient zero for the next source node.
    __syncwarp();
  }
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    far_value += __shfl_down_sync(0xffffffffU, far_value, offset);
  }
  if (lane == 0) {
    far_warp_sums[warp] = far_value;
  }
  __syncthreads();

  if (thread == 0) {
    if (invalid_target != 0) {
      potentials[target_index] = quiet_nan_device();
      return;
    }
    double approximation = singular_value;
    for (int source_warp = 0;
         source_warp < warp_count;
         ++source_warp) {
      approximation += near_warp_sums[source_warp]
          + far_warp_sums[source_warp];
    }
    potentials[target_index] = approximation;
  }
}

struct MomentExponent
{
  int x = 0;
  int y = 0;
  int z = 0;
};

__device__ void quadrature_sub_tetrahedron(
    int vertex_count,
    int sub_tetrahedron,
    int* indices)
{
  if (vertex_count == 4) {
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
    indices[3] = 3;
  } else if (vertex_count == 5) {
    constexpr int twin[2][4] = {
        {0, 1, 2, 4}, {0, 2, 3, 4}};
    for (int vertex = 0; vertex < 4; ++vertex) {
      indices[vertex] = twin[sub_tetrahedron][vertex];
    }
  } else {
    constexpr int four[4][4] = {
        {0, 1, 5, 6},
        {0, 2, 4, 6},
        {0, 3, 4, 5},
        {0, 4, 5, 6}};
    for (int vertex = 0; vertex < 4; ++vertex) {
      indices[vertex] = four[sub_tetrahedron][vertex];
    }
  }
}

__global__ void generate_quadrature_kernel(
    const GpuQuadratureElement* elements,
    int element_count,
    const double* radial_nodes,
    const double* radial_weights,
    int radial_count,
    const double* face_nodes,
    const double* face_weights,
    int face_count,
    const double* edge_nodes,
    const double* edge_weights,
    int edge_count,
    GpuQuadratureSample* samples,
    GpuPoint3* targets,
    int* target_hierarchy_indices,
    int* target_leaf_indices)
{
  const int element_index = static_cast<int>(blockIdx.x);
  if (element_index >= element_count) {
    return;
  }
  const GpuQuadratureElement element = elements[element_index];
  const int points_per_sub_tetrahedron =
      radial_count * face_count * edge_count;

  for (int local_index = static_cast<int>(threadIdx.x);
       local_index < element.sample_count;
       local_index += static_cast<int>(blockDim.x)) {
    const int sub_tetrahedron =
        local_index / points_per_sub_tetrahedron;
    int axis_index = local_index % points_per_sub_tetrahedron;
    const int edge_index = axis_index % edge_count;
    axis_index /= edge_count;
    const int face_index = axis_index % face_count;
    const int radial_index = axis_index / face_count;
    int vertex_indices[4];
    quadrature_sub_tetrahedron(
        element.vertex_count, sub_tetrahedron, vertex_indices);

    const GpuPoint3 v0 = element.vertices[vertex_indices[0]];
    const GpuPoint3 v1 = element.vertices[vertex_indices[1]];
    const GpuPoint3 v2 = element.vertices[vertex_indices[2]];
    const GpuPoint3 v3 = element.vertices[vertex_indices[3]];
    const GpuPoint3 e1 = subtract(v1, v0);
    const GpuPoint3 e2 = subtract(v2, v0);
    const GpuPoint3 e3 = subtract(v3, v0);
    const double absolute_determinant =
        fabs(determinant(e1, e2, e3));
    const double r = radial_nodes[radial_index];
    const double s = face_nodes[face_index];
    const double t = edge_nodes[edge_index];
    const double lambda0 = 1.0 - r;
    const double lambda1 = r * (1.0 - s);
    const double lambda2 = r * s * (1.0 - t);
    const double lambda3 = r * s * t;
    const GpuPoint3 point{
        lambda0 * v0.x + lambda1 * v1.x
            + lambda2 * v2.x + lambda3 * v3.x,
        lambda0 * v0.y + lambda1 * v1.y
            + lambda2 * v2.y + lambda3 * v3.y,
        lambda0 * v0.z + lambda1 * v1.z
            + lambda2 * v2.z + lambda3 * v3.z};
    const double weight =
        radial_weights[radial_index]
        * face_weights[face_index]
        * edge_weights[edge_index]
        * r * r * s * absolute_determinant;
    const int sample_index = element.sample_offset + local_index;
    samples[sample_index] = {point, weight};
    if (targets != nullptr) {
      targets[sample_index] = point;
      target_hierarchy_indices[sample_index] =
          element.hierarchy_index;
      target_leaf_indices[sample_index] = element.leaf_index;
    }
  }
}

__global__ void refinement_indicator_kernel(
    const GpuQuadratureElement* elements,
    int element_count,
    const double* radial_nodes,
    const double* radial_weights,
    int radial_count,
    const double* face_nodes,
    const double* face_weights,
    int face_count,
    const double* edge_nodes,
    const double* edge_weights,
    int edge_count,
    double* indicators)
{
  const int element_index = static_cast<int>(blockIdx.x);
  if (element_index >= element_count) {
    return;
  }
  constexpr int indicator_block_size = 256;
  __shared__ double partial_sums[indicator_block_size];
  const int lane = static_cast<int>(threadIdx.x);
  const GpuQuadratureElement element = elements[element_index];
  const int points_per_sub_tetrahedron =
      radial_count * face_count * edge_count;
  double value = 0.0;

  for (int local_index = lane;
       local_index < element.sample_count;
       local_index += static_cast<int>(blockDim.x)) {
    const int sub_tetrahedron =
        local_index / points_per_sub_tetrahedron;
    int axis_index = local_index % points_per_sub_tetrahedron;
    const int edge_index = axis_index % edge_count;
    axis_index /= edge_count;
    const int face_index = axis_index % face_count;
    const int radial_index = axis_index / face_count;
    int vertex_indices[4];
    quadrature_sub_tetrahedron(
        element.vertex_count, sub_tetrahedron, vertex_indices);
    const GpuPoint3 v0 = element.vertices[vertex_indices[0]];
    const GpuPoint3 v1 = element.vertices[vertex_indices[1]];
    const GpuPoint3 v2 = element.vertices[vertex_indices[2]];
    const GpuPoint3 v3 = element.vertices[vertex_indices[3]];
    const GpuPoint3 e1 = subtract(v1, v0);
    const GpuPoint3 e2 = subtract(v2, v0);
    const GpuPoint3 e3 = subtract(v3, v0);
    const double absolute_determinant =
        fabs(determinant(e1, e2, e3));
    const double r = radial_nodes[radial_index];
    const double s = face_nodes[face_index];
    const double t = edge_nodes[edge_index];
    const double lambda0 = 1.0 - r;
    const double lambda1 = r * (1.0 - s);
    const double lambda2 = r * s * (1.0 - t);
    const double lambda3 = r * s * t;
    const GpuPoint3 point{
        lambda0 * v0.x + lambda1 * v1.x
            + lambda2 * v2.x + lambda3 * v3.x,
        lambda0 * v0.y + lambda1 * v1.y
            + lambda2 * v2.y + lambda3 * v3.y,
        lambda0 * v0.z + lambda1 * v1.z
            + lambda2 * v2.z + lambda3 * v3.z};
    const double weight =
        radial_weights[radial_index]
        * face_weights[face_index]
        * edge_weights[edge_index]
        * r * r * s * absolute_determinant;
    const double radius_squared =
        point.x * point.x + point.y * point.y + point.z * point.z;
    value += weight / sqrt(fmax(radius_squared, 1.0e-30));
  }

  partial_sums[lane] = value;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2;
       stride > 0;
       stride /= 2) {
    if (lane < stride) {
      partial_sums[lane] += partial_sums[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    indicators[element_index] = partial_sums[0];
  }
}

__global__ void prepare_source_charges_kernel(
    const GpuQuadratureSample* samples,
    int sample_count,
    double* sample_charges)
{
  const int index =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
      + static_cast<int>(threadIdx.x);
  if (index >= sample_count) {
    return;
  }
  const GpuQuadratureSample& sample = samples[index];
  sample_charges[index] =
      sample.weight * density_device(sample.point);
}

__device__ double integer_power(double value, int exponent)
{
  double result = 1.0;
  for (int i = 0; i < exponent; ++i) {
    result *= value;
  }
  return result;
}

__device__ double binomial_coefficient(int degree, int index)
{
  index = min(index, degree - index);
  double result = 1.0;
  for (int i = 1; i <= index; ++i) {
    result *=
        static_cast<double>(degree - index + i)
        / static_cast<double>(i);
  }
  return result;
}

__global__ void calculate_leaf_moments_kernel(
    const GpuLeaf* leaves,
    const int* leaf_hierarchy_indices,
    int leaf_count,
    const GpuQuadratureSample* samples,
    const double* sample_charges,
    const GpuMultipole* multipoles,
    const MomentExponent* exponents,
    int coefficient_count,
    double* moments)
{
  const int leaf_index = static_cast<int>(blockIdx.x);
  if (leaf_index >= leaf_count) {
    return;
  }
  const int hierarchy_index = leaf_hierarchy_indices[leaf_index];
  const GpuLeaf& leaf = leaves[leaf_index];
  const GpuPoint3 center = multipoles[hierarchy_index].center;
  const int moment_offset = multipoles[hierarchy_index].moment_offset;

  for (int coefficient = static_cast<int>(threadIdx.x);
       coefficient < coefficient_count;
       coefficient += static_cast<int>(blockDim.x)) {
    const MomentExponent exponent = exponents[coefficient];
    double value = 0.0;
    for (int sample_index = leaf.sample_offset;
         sample_index < leaf.sample_offset + leaf.sample_count;
         ++sample_index) {
      const GpuPoint3 point = samples[sample_index].point;
      value +=
          sample_charges[sample_index]
          * integer_power(point.x - center.x, exponent.x)
          * integer_power(point.y - center.y, exponent.y)
          * integer_power(point.z - center.z, exponent.z);
    }
    moments[moment_offset + coefficient] = value;
  }
}

__global__ void aggregate_moments_at_level_kernel(
    const GpuMultipole* multipoles,
    int hierarchy_count,
    const int* child_offsets,
    const int* child_indices,
    const MomentExponent* exponents,
    int coefficient_count,
    int level,
    double* moments)
{
  const int hierarchy_index = static_cast<int>(blockIdx.x);
  if (hierarchy_index >= hierarchy_count
      || multipoles[hierarchy_index].level != level
      || child_offsets[hierarchy_index]
          == child_offsets[hierarchy_index + 1]) {
    return;
  }

  extern __shared__ double translation_workspace[];
  double* accumulated = translation_workspace;
  double* x_translated = accumulated + coefficient_count;
  double* xy_translated = x_translated + coefficient_count;
  const int lane = static_cast<int>(threadIdx.x);
  for (int coefficient = lane;
       coefficient < coefficient_count;
       coefficient += static_cast<int>(blockDim.x)) {
    accumulated[coefficient] = 0.0;
  }
  __syncthreads();

  const GpuMultipole parent = multipoles[hierarchy_index];
  for (int child_entry = child_offsets[hierarchy_index];
       child_entry < child_offsets[hierarchy_index + 1];
       ++child_entry) {
    const int child_index = child_indices[child_entry];
    const GpuMultipole child = multipoles[child_index];
    const double shift_x = child.center.x - parent.center.x;
    const double shift_y = child.center.y - parent.center.y;
    const double shift_z = child.center.z - parent.center.z;

    for (int coefficient = lane;
         coefficient < coefficient_count;
         coefficient += static_cast<int>(blockDim.x)) {
      const MomentExponent exponent = exponents[coefficient];
      double value = 0.0;
      for (int child_x = 0; child_x <= exponent.x; ++child_x) {
        const int child_coefficient =
            treecode_multipole::coefficient_index(
                child_x, exponent.y, exponent.z);
        value +=
            binomial_coefficient(exponent.x, child_x)
            * integer_power(shift_x, exponent.x - child_x)
            * moments[child.moment_offset + child_coefficient];
      }
      x_translated[coefficient] = value;
    }
    __syncthreads();

    for (int coefficient = lane;
         coefficient < coefficient_count;
         coefficient += static_cast<int>(blockDim.x)) {
      const MomentExponent exponent = exponents[coefficient];
      double value = 0.0;
      for (int child_y = 0; child_y <= exponent.y; ++child_y) {
        value +=
            binomial_coefficient(exponent.y, child_y)
            * integer_power(shift_y, exponent.y - child_y)
            * x_translated[
                treecode_multipole::coefficient_index(
                    exponent.x, child_y, exponent.z)];
      }
      xy_translated[coefficient] = value;
    }
    __syncthreads();

    for (int coefficient = lane;
         coefficient < coefficient_count;
         coefficient += static_cast<int>(blockDim.x)) {
      const MomentExponent exponent = exponents[coefficient];
      double value = 0.0;
      for (int child_z = 0; child_z <= exponent.z; ++child_z) {
        value +=
            binomial_coefficient(exponent.z, child_z)
            * integer_power(shift_z, exponent.z - child_z)
            * xy_translated[
                treecode_multipole::coefficient_index(
                    exponent.x, exponent.y, child_z)];
      }
      accumulated[coefficient] += value;
    }
    __syncthreads();
  }

  for (int coefficient = lane;
       coefficient < coefficient_count;
       coefficient += static_cast<int>(blockDim.x)) {
    moments[parent.moment_offset + coefficient] =
        accumulated[coefficient];
  }
}

constexpr int interaction_stack_capacity = 256;

__global__ void construct_interaction_lists_kernel(
    const GpuMultipole* hierarchy,
    int hierarchy_count,
    const int* child_offsets,
    const int* child_indices,
    const int* root_indices,
    int root_count,
    const int* leaf_hierarchy_indices,
    int leaf_count,
    double theta,
    double domain_volume,
    int* near_counts,
    int* far_counts,
    const int* near_offsets,
    const int* far_offsets,
    int* near_leaf_indices,
    int* far_hierarchy_indices,
    unsigned long long int* invalid_target_count)
{
  const int leaf_index =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
      + static_cast<int>(threadIdx.x);
  if (leaf_index >= leaf_count) {
    return;
  }
  const bool fill_lists =
      near_leaf_indices != nullptr && far_hierarchy_indices != nullptr;
  const int target_hierarchy_index =
      leaf_hierarchy_indices[leaf_index];
  if (target_hierarchy_index < 0
      || target_hierarchy_index >= hierarchy_count) {
    atomicAdd(invalid_target_count, 1ULL);
    return;
  }
  const GpuMultipole target = hierarchy[target_hierarchy_index];

  int stack[interaction_stack_capacity];
  int stack_size = 0;
  bool valid = true;
  for (int root = 0; root < root_count; ++root) {
    if (stack_size >= interaction_stack_capacity) {
      valid = false;
      break;
    }
    stack[stack_size++] = root_indices[root];
  }

  int near_count = 0;
  int far_count = 0;
  int represented_leaf_count = 0;
  double represented_volume = 0.0;
  bool contains_target = false;
  while (valid && stack_size > 0) {
    const int source_index = stack[--stack_size];
    if (source_index < 0 || source_index >= hierarchy_count) {
      valid = false;
      break;
    }
    const GpuMultipole source = hierarchy[source_index];
    const double dx = source.center.x - target.center.x;
    const double dy = source.center.y - target.center.y;
    const double dz = source.center.z - target.center.z;
    const double center_distance = sqrt(dx * dx + dy * dy + dz * dz);
    const double minimum_target_distance =
        center_distance - target.radius;
    const bool source_contains_target =
        source.leaf_begin <= target.leaf_begin
        && target.leaf_begin < source.leaf_end;
    const bool mac_accepted =
        !source_contains_target
        && minimum_target_distance > 0.0
        && source.radius / minimum_target_distance <= theta;
    const int child_count =
        child_offsets[source_index + 1] - child_offsets[source_index];

    if (mac_accepted) {
      if (fill_lists) {
        far_hierarchy_indices[
            far_offsets[leaf_index] + far_count] = source_index;
      }
      ++far_count;
      represented_leaf_count += source.leaf_end - source.leaf_begin;
      represented_volume += source.volume;
    } else if (child_count == 0) {
      if (source.leaf_index < 0 || source.leaf_index >= leaf_count) {
        valid = false;
        break;
      }
      if (fill_lists) {
        near_leaf_indices[
            near_offsets[leaf_index] + near_count] = source.leaf_index;
      }
      ++near_count;
      ++represented_leaf_count;
      represented_volume += source.volume;
      contains_target =
          contains_target || source_index == target_hierarchy_index;
    } else {
      if (stack_size + child_count > interaction_stack_capacity) {
        valid = false;
        break;
      }
      for (int child = child_offsets[source_index];
           child < child_offsets[source_index + 1];
           ++child) {
        stack[stack_size++] = child_indices[child];
      }
    }
  }

  if (!fill_lists) {
    near_counts[leaf_index] = near_count;
    far_counts[leaf_index] = far_count;
    const double volume_tolerance =
        1.0e-10 * fmax(1.0, domain_volume);
    if (!valid
        || !contains_target
        || represented_leaf_count != leaf_count
        || fabs(represented_volume - domain_volume) > volume_tolerance) {
      atomicAdd(invalid_target_count, 1ULL);
    }
  } else if (near_count
                 != near_offsets[leaf_index + 1]
                    - near_offsets[leaf_index]
             || far_count
                 != far_offsets[leaf_index + 1]
                    - far_offsets[leaf_index]) {
    atomicAdd(invalid_target_count, 1ULL);
  }
}

__device__ double atomic_add_double(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* integer_address =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *integer_address;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(
        integer_address,
        assumed,
        __double_as_longlong(
            value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

__global__ void reduce_weighted_l2_kernel(
    const double* potentials,
    const GpuPoint3* targets,
    const GpuQuadratureSample* samples,
    const double* target_weights,
    int target_count,
    double* norm_sums,
    unsigned long long int* nonfinite_count)
{
  constexpr int reduction_block_size = 256;
  __shared__ double error_sums[reduction_block_size];
  __shared__ double exact_sums[reduction_block_size];
  __shared__ double approximation_sums[reduction_block_size];
  __shared__ unsigned long long int invalid_counts[reduction_block_size];

  const int lane = static_cast<int>(threadIdx.x);
  const int global_index =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + lane;
  const int grid_stride =
      static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
  double local_error = 0.0;
  double local_exact = 0.0;
  double local_approximation = 0.0;
  unsigned long long int local_invalid = 0;

  for (int index = global_index;
       index < target_count;
       index += grid_stride) {
    const double approximation = potentials[index];
    const GpuPoint3 target = targets[index];
    const double exact_value = treecode_test_problem::exact_value(
        target.x, target.y, target.z);
    const double weight = target_weights != nullptr
        ? target_weights[index]
        : samples[index].weight;
    if (!isfinite(approximation)
        || !isfinite(exact_value)
        || !isfinite(weight)) {
      ++local_invalid;
      continue;
    }
    const double difference = approximation - exact_value;
    local_error += weight * difference * difference;
    local_exact += weight * exact_value * exact_value;
    local_approximation += weight * approximation * approximation;
  }

  error_sums[lane] = local_error;
  exact_sums[lane] = local_exact;
  approximation_sums[lane] = local_approximation;
  invalid_counts[lane] = local_invalid;
  __syncthreads();

  for (int stride = static_cast<int>(blockDim.x) / 2;
       stride > 0;
       stride /= 2) {
    if (lane < stride) {
      error_sums[lane] += error_sums[lane + stride];
      exact_sums[lane] += exact_sums[lane + stride];
      approximation_sums[lane] += approximation_sums[lane + stride];
      invalid_counts[lane] += invalid_counts[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    atomic_add_double(&norm_sums[0], error_sums[0]);
    atomic_add_double(&norm_sums[1], exact_sums[0]);
    atomic_add_double(&norm_sums[2], approximation_sums[0]);
    atomicAdd(nonfinite_count, invalid_counts[0]);
  }
}

__global__ void finalize_weighted_l2_kernel(double* norm_values)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const double exact_error = sqrt(norm_values[0]);
  const double exact_norm = sqrt(norm_values[1]);
  const double approximation_norm = sqrt(norm_values[2]);
  norm_values[3] = exact_error;
  norm_values[4] = exact_norm;
  norm_values[5] = approximation_norm;
  norm_values[6] =
      exact_norm > 0.0 ? exact_error / exact_norm : 0.0;
}

void validate_problem(
    const GpuTreecodeProblem& problem,
    int expansion_order,
    int quadrature_accuracy,
    bool calculate_exact_l2_norms)
{
  const auto require_int_index = [](std::size_t size, const char* name) {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          std::string(name) + " exceeds CUDA's 32-bit index range");
    }
  };

  if (expansion_order < 1 ||
      expansion_order > treecode_multipole::maximum_order) {
    throw std::invalid_argument(
        "GPU expansion order must be between 1 and "
        + std::to_string(treecode_multipole::maximum_order));
  }
  if (quadrature_accuracy < 1 || quadrature_accuracy > 64) {
    throw std::invalid_argument(
        "GPU quadrature accuracy must be between 1 and 64");
  }
  const std::size_t target_count =
      problem.generate_exact_targets_on_gpu
          ? problem.generated_sample_count
          : problem.targets.size();
  const std::size_t sample_count =
      problem.generate_samples_on_gpu
          ? problem.generated_sample_count
          : problem.samples.size();
  if (target_count == 0) {
    throw std::invalid_argument("GPU problem contains no target points");
  }
  if (!problem.generate_exact_targets_on_gpu
      && (problem.target_hierarchy_indices.size() != target_count
          || problem.target_leaf_indices.size() != target_count)) {
    throw std::invalid_argument("GPU target arrays have inconsistent sizes");
  }
  if (problem.leaves.empty() || sample_count == 0
      || problem.multipoles.empty()) {
    throw std::invalid_argument(
        "GPU source mesh is incomplete");
  }
  if (problem.leaf_hierarchy_indices.size() != problem.leaves.size()) {
    throw std::invalid_argument(
        "GPU leaf-to-hierarchy mapping has the wrong size");
  }
  if (problem.child_offsets.size() != problem.multipoles.size() + 1
      || problem.child_offsets.back()
          != static_cast<int>(problem.child_hierarchy_indices.size())) {
    throw std::invalid_argument(
        "GPU hierarchy child offsets are inconsistent");
  }
  if (problem.root_hierarchy_indices.empty()) {
    throw std::invalid_argument("GPU hierarchy contains no roots");
  }
  if (!(problem.mac_theta > 0.0) || !(problem.domain_volume > 0.0)) {
    throw std::invalid_argument(
        "GPU MAC parameter or domain volume is invalid");
  }
  if (calculate_exact_l2_norms
      && problem.target_weights.size() != target_count
      && (!problem.targets_match_source_samples
          || target_count != sample_count)) {
    throw std::invalid_argument(
        "GPU exact L2 reduction requires one weight per target or targets "
        "that match source samples");
  }
  if (problem.generate_exact_targets_on_gpu
      && (!problem.generate_samples_on_gpu
          || !problem.targets.empty()
          || !problem.target_hierarchy_indices.empty()
          || !problem.target_leaf_indices.empty())) {
    throw std::invalid_argument(
        "GPU-generated exact targets have inconsistent host arrays");
  }
  if (problem.generate_samples_on_gpu) {
    if (problem.quadrature_elements.size() != problem.leaves.size()
        || !problem.samples.empty()) {
      throw std::invalid_argument(
          "GPU quadrature element data is inconsistent");
    }
  }
  require_int_index(target_count, "GPU target count");
  require_int_index(problem.target_weights.size(), "GPU target weight count");
  require_int_index(problem.leaves.size(), "GPU leaf count");
  require_int_index(sample_count, "GPU sample count");
  require_int_index(problem.multipoles.size(), "GPU hierarchy count");
  require_int_index(
      problem.child_hierarchy_indices.size(), "GPU hierarchy child count");
  const std::size_t moment_count =
      problem.multipoles.size()
      * static_cast<std::size_t>(
          treecode_multipole::coefficient_count(expansion_order));
  require_int_index(moment_count, "GPU multipole moment count");

  for (std::size_t leaf_index = 0;
       leaf_index < problem.leaves.size();
       ++leaf_index) {
    const GpuLeaf& leaf = problem.leaves[leaf_index];
    if (leaf.sample_offset < 0 || leaf.sample_count <= 0
        || static_cast<std::size_t>(leaf.sample_offset)
               + static_cast<std::size_t>(leaf.sample_count)
            > sample_count) {
      throw std::invalid_argument(
          "GPU leaf quadrature ranges are inconsistent");
    }
    const int hierarchy_index =
        problem.leaf_hierarchy_indices[leaf_index];
    if (hierarchy_index < 0
        || static_cast<std::size_t>(hierarchy_index)
            >= problem.multipoles.size()
        || problem.multipoles[
               static_cast<std::size_t>(hierarchy_index)].leaf_index
            != static_cast<int>(leaf_index)) {
      throw std::invalid_argument(
          "GPU leaf hierarchy mapping is inconsistent");
    }
  }
  if (problem.generate_samples_on_gpu) {
    for (std::size_t element_index = 0;
         element_index < problem.quadrature_elements.size();
         ++element_index) {
      const GpuQuadratureElement& element =
          problem.quadrature_elements[element_index];
      if ((element.vertex_count != 4
           && element.vertex_count != 5
           && element.vertex_count != 7)
          || element.sample_offset
              != problem.leaves[element_index].sample_offset
          || element.sample_count
              != problem.leaves[element_index].sample_count
          || element.leaf_index != static_cast<int>(element_index)
          || element.hierarchy_index
              != problem.leaf_hierarchy_indices[element_index]) {
        throw std::invalid_argument(
            "GPU quadrature element mapping is inconsistent");
      }
    }
  }

  for (std::size_t hierarchy_index = 0;
       hierarchy_index < problem.multipoles.size();
       ++hierarchy_index) {
    const GpuMultipole& node = problem.multipoles[hierarchy_index];
    if (!(node.radius >= 0.0) || !(node.volume > 0.0)
        || node.leaf_begin < 0 || node.leaf_end <= node.leaf_begin
        || static_cast<std::size_t>(node.leaf_end)
            > problem.leaves.size()) {
      throw std::invalid_argument(
          "GPU hierarchy geometry is inconsistent");
    }
    for (int child_entry = problem.child_offsets[hierarchy_index];
         child_entry < problem.child_offsets[hierarchy_index + 1];
         ++child_entry) {
      const int child_index =
          problem.child_hierarchy_indices[
              static_cast<std::size_t>(child_entry)];
      if (child_index < 0
          || static_cast<std::size_t>(child_index)
              >= problem.multipoles.size()
          || problem.multipoles[
                 static_cast<std::size_t>(child_index)].level
              != node.level + 1) {
        throw std::invalid_argument(
            "GPU hierarchy parent-child data is inconsistent");
      }
    }
  }
  for (int root_index : problem.root_hierarchy_indices) {
    if (root_index < 0
        || static_cast<std::size_t>(root_index)
            >= problem.multipoles.size()
        || problem.multipoles[
               static_cast<std::size_t>(root_index)].level != 0) {
      throw std::invalid_argument("GPU root hierarchy index is invalid");
    }
  }
}

} // namespace

GpuTreecodeResult evaluate_treecode_gpu(
    const GpuTreecodeProblem& problem,
    int expansion_order,
    int quadrature_accuracy,
    bool calculate_exact_l2_norms,
    bool evaluate_potentials)
{
  if (calculate_exact_l2_norms && !evaluate_potentials) {
    throw std::invalid_argument(
        "GPU exact L2 norms require potential evaluation");
  }
  validate_problem(
      problem,
      expansion_order,
      quadrature_accuracy,
      calculate_exact_l2_norms);

  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    throw std::runtime_error("No CUDA-capable GPU is visible");
  }

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(
      cudaGetDeviceProperties(&properties, device),
      "cudaGetDeviceProperties");

  const std::size_t target_count_size =
      problem.generate_exact_targets_on_gpu
          ? problem.generated_sample_count
          : problem.targets.size();
  const std::size_t sample_count_size =
      problem.generate_samples_on_gpu
          ? problem.generated_sample_count
          : problem.samples.size();
  const int target_count = static_cast<int>(target_count_size);
  const int sample_count = static_cast<int>(sample_count_size);
  DeviceBuffer<GpuPoint3> targets;
  DeviceBuffer<int> target_hierarchy_indices;
  DeviceBuffer<int> target_leaf_indices;
  if (problem.generate_exact_targets_on_gpu) {
    targets = DeviceBuffer<GpuPoint3>(target_count_size);
    target_hierarchy_indices = DeviceBuffer<int>(target_count_size);
    target_leaf_indices = DeviceBuffer<int>(target_count_size);
  } else {
    targets = DeviceBuffer<GpuPoint3>(problem.targets);
    target_hierarchy_indices =
        DeviceBuffer<int>(problem.target_hierarchy_indices);
    target_leaf_indices =
        DeviceBuffer<int>(problem.target_leaf_indices);
  }
  DeviceBuffer<GpuLeaf> leaves(problem.leaves);
  DeviceBuffer<GpuQuadratureSample> samples;
  float quadrature_milliseconds = 0.0F;
  if (problem.generate_samples_on_gpu) {
    samples = DeviceBuffer<GpuQuadratureSample>(sample_count_size);
    DeviceBuffer<GpuQuadratureElement> quadrature_elements(
        problem.quadrature_elements);
    const int radial_count = (quadrature_accuracy + 4) / 2;
    const int face_count = (quadrature_accuracy + 3) / 2;
    const int edge_count = (quadrature_accuracy + 2) / 2;
    const auto radial_rule =
        gauss_legendre_on_unit_interval(radial_count);
    const auto face_rule =
        gauss_legendre_on_unit_interval(face_count);
    const auto edge_rule =
        gauss_legendre_on_unit_interval(edge_count);
    const auto split_rule = [](
        const std::vector<std::pair<double, double>>& rule) {
      std::pair<std::vector<double>, std::vector<double>> result;
      result.first.resize(rule.size());
      result.second.resize(rule.size());
      for (std::size_t i = 0; i < rule.size(); ++i) {
        result.first[i] = rule[i].first;
        result.second[i] = rule[i].second;
      }
      return result;
    };
    const auto radial_values = split_rule(radial_rule);
    const auto face_values = split_rule(face_rule);
    const auto edge_values = split_rule(edge_rule);
    DeviceBuffer<double> radial_nodes(radial_values.first);
    DeviceBuffer<double> radial_weights(radial_values.second);
    DeviceBuffer<double> face_nodes(face_values.first);
    DeviceBuffer<double> face_weights(face_values.second);
    DeviceBuffer<double> edge_nodes(edge_values.first);
    DeviceBuffer<double> edge_weights(edge_values.second);
    CudaEvent quadrature_start;
    CudaEvent quadrature_finish;
    check_cuda(
        cudaEventRecord(quadrature_start.get()),
        "cudaEventRecord quadrature start");
    constexpr int quadrature_block_size = 256;
    generate_quadrature_kernel
        <<<static_cast<int>(problem.quadrature_elements.size()),
           quadrature_block_size>>>(
            quadrature_elements.data(),
            static_cast<int>(problem.quadrature_elements.size()),
            radial_nodes.data(),
            radial_weights.data(),
            radial_count,
            face_nodes.data(),
            face_weights.data(),
            face_count,
            edge_nodes.data(),
            edge_weights.data(),
            edge_count,
            samples.data(),
            problem.generate_exact_targets_on_gpu
                ? targets.data() : nullptr,
            problem.generate_exact_targets_on_gpu
                ? target_hierarchy_indices.data() : nullptr,
            problem.generate_exact_targets_on_gpu
                ? target_leaf_indices.data() : nullptr);
    check_cuda(
        cudaGetLastError(),
        "generate_quadrature_kernel launch");
    check_cuda(
        cudaEventRecord(quadrature_finish.get()),
        "cudaEventRecord quadrature finish");
    check_cuda(
        cudaEventSynchronize(quadrature_finish.get()),
        "CUDA quadrature generation");
    check_cuda(
        cudaEventElapsedTime(
            &quadrature_milliseconds,
            quadrature_start.get(),
            quadrature_finish.get()),
        "cudaEventElapsedTime quadrature");
  } else {
    samples = DeviceBuffer<GpuQuadratureSample>(problem.samples);
  }
  std::vector<GpuMultipole> host_multipoles = problem.multipoles;
  const int coefficient_count =
      treecode_multipole::coefficient_count(expansion_order);
  for (std::size_t hierarchy_index = 0;
       hierarchy_index < host_multipoles.size();
       ++hierarchy_index) {
    host_multipoles[hierarchy_index].moment_offset =
        static_cast<int>(
            hierarchy_index
            * static_cast<std::size_t>(coefficient_count));
  }
  DeviceBuffer<GpuMultipole> multipoles(host_multipoles);
  DeviceBuffer<int> leaf_hierarchy_indices(
      problem.leaf_hierarchy_indices);
  DeviceBuffer<int> root_hierarchy_indices(
      problem.root_hierarchy_indices);
  DeviceBuffer<int> child_offsets(problem.child_offsets);
  DeviceBuffer<int> child_hierarchy_indices(
      problem.child_hierarchy_indices);

  const int leaf_count = static_cast<int>(problem.leaves.size());
  const int hierarchy_count =
      static_cast<int>(problem.multipoles.size());
  DeviceBuffer<double> sample_charges(sample_count_size);
  CudaEvent source_start;
  CudaEvent source_finish;
  check_cuda(
      cudaEventRecord(source_start.get()),
      "cudaEventRecord source preparation start");
  constexpr int preparation_block_size = 256;
  prepare_source_charges_kernel
      <<<(sample_count + preparation_block_size - 1)
             / preparation_block_size,
         preparation_block_size>>>(
          samples.data(), sample_count, sample_charges.data());
  check_cuda(
      cudaGetLastError(),
      "prepare_source_charges_kernel launch");
  check_cuda(
      cudaEventRecord(source_finish.get()),
      "cudaEventRecord source preparation finish");
  check_cuda(
      cudaEventSynchronize(source_finish.get()),
      "CUDA source preparation");
  float source_milliseconds = 0.0F;
  check_cuda(
      cudaEventElapsedTime(
          &source_milliseconds,
          source_start.get(),
          source_finish.get()),
      "cudaEventElapsedTime source preparation");

  std::vector<MomentExponent> host_exponents(
      static_cast<std::size_t>(coefficient_count));
  for (int degree = 0; degree <= expansion_order; ++degree) {
    for (int x = 0; x <= degree; ++x) {
      for (int y = 0; y <= degree - x; ++y) {
        const int z = degree - x - y;
        host_exponents[static_cast<std::size_t>(
            treecode_multipole::coefficient_index(x, y, z))] =
            {x, y, z};
      }
    }
  }
  DeviceBuffer<MomentExponent> exponents(host_exponents);
  const std::size_t moment_count =
      problem.multipoles.size()
      * static_cast<std::size_t>(coefficient_count);
  DeviceBuffer<double> moments(moment_count);
  check_cuda(
      cudaMemset(
          moments.data(), 0, moment_count * sizeof(double)),
      "cudaMemset multipole moments");
  int maximum_level = 0;
  for (const GpuMultipole& multipole : host_multipoles) {
    maximum_level = std::max(maximum_level, multipole.level);
  }
  CudaEvent moment_start;
  CudaEvent moment_finish;
  check_cuda(
      cudaEventRecord(moment_start.get()),
      "cudaEventRecord moment start");
  constexpr int moment_block_size = 128;
  calculate_leaf_moments_kernel<<<leaf_count, moment_block_size>>>(
      leaves.data(),
      leaf_hierarchy_indices.data(),
      leaf_count,
      samples.data(),
      sample_charges.data(),
      multipoles.data(),
      exponents.data(),
      coefficient_count,
      moments.data());
  check_cuda(
      cudaGetLastError(),
      "calculate_leaf_moments_kernel launch");
  const std::size_t translation_shared_bytes =
      3 * static_cast<std::size_t>(coefficient_count) * sizeof(double);
  for (int level = maximum_level - 1; level >= 0; --level) {
    aggregate_moments_at_level_kernel
        <<<hierarchy_count,
           moment_block_size,
           translation_shared_bytes>>>(
            multipoles.data(),
            hierarchy_count,
            child_offsets.data(),
            child_hierarchy_indices.data(),
            exponents.data(),
            coefficient_count,
            level,
            moments.data());
    check_cuda(
        cudaGetLastError(),
        "aggregate_moments_at_level_kernel launch");
  }
  check_cuda(
      cudaEventRecord(moment_finish.get()),
      "cudaEventRecord moment finish");
  check_cuda(
      cudaEventSynchronize(moment_finish.get()),
      "CUDA multipole moment calculation");
  float moment_milliseconds = 0.0F;
  check_cuda(
      cudaEventElapsedTime(
          &moment_milliseconds,
          moment_start.get(),
          moment_finish.get()),
      "cudaEventElapsedTime moments");

  DeviceBuffer<int> near_counts(
      static_cast<std::size_t>(leaf_count) + 1);
  DeviceBuffer<int> far_counts(
      static_cast<std::size_t>(leaf_count) + 1);
  DeviceBuffer<int> near_offsets(
      static_cast<std::size_t>(leaf_count) + 1);
  DeviceBuffer<int> far_offsets(
      static_cast<std::size_t>(leaf_count) + 1);
  DeviceBuffer<unsigned long long int> interaction_invalid_count(1);
  check_cuda(
      cudaMemset(
          near_counts.data(),
          0,
          (static_cast<std::size_t>(leaf_count) + 1) * sizeof(int)),
      "cudaMemset near counts");
  check_cuda(
      cudaMemset(
          far_counts.data(),
          0,
          (static_cast<std::size_t>(leaf_count) + 1) * sizeof(int)),
      "cudaMemset far counts");
  check_cuda(
      cudaMemset(
          interaction_invalid_count.data(),
          0,
          sizeof(unsigned long long int)),
      "cudaMemset interaction validation");

  constexpr int interaction_block_size = 128;
  const int interaction_block_count =
      (leaf_count + interaction_block_size - 1)
      / interaction_block_size;
  CudaEvent interaction_start;
  CudaEvent interaction_finish;
  check_cuda(
      cudaEventRecord(interaction_start.get()),
      "cudaEventRecord interaction start");
  construct_interaction_lists_kernel
      <<<interaction_block_count, interaction_block_size>>>(
          multipoles.data(),
          hierarchy_count,
          child_offsets.data(),
          child_hierarchy_indices.data(),
          root_hierarchy_indices.data(),
          static_cast<int>(problem.root_hierarchy_indices.size()),
          leaf_hierarchy_indices.data(),
          leaf_count,
          problem.mac_theta,
          problem.domain_volume,
          near_counts.data(),
          far_counts.data(),
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          interaction_invalid_count.data());
  check_cuda(
      cudaGetLastError(),
      "construct_interaction_lists count launch");
  thrust::device_ptr<int> near_count_pointer(near_counts.data());
  thrust::device_ptr<int> far_count_pointer(far_counts.data());
  thrust::device_ptr<int> near_offset_pointer(near_offsets.data());
  thrust::device_ptr<int> far_offset_pointer(far_offsets.data());
  thrust::exclusive_scan(
      near_count_pointer,
      near_count_pointer + leaf_count + 1,
      near_offset_pointer);
  thrust::exclusive_scan(
      far_count_pointer,
      far_count_pointer + leaf_count + 1,
      far_offset_pointer);

  int near_interaction_count = 0;
  int far_interaction_count = 0;
  check_cuda(
      cudaMemcpy(
          &near_interaction_count,
          near_offsets.data() + leaf_count,
          sizeof(int),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy near interaction count");
  check_cuda(
      cudaMemcpy(
          &far_interaction_count,
          far_offsets.data() + leaf_count,
          sizeof(int),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy far interaction count");
  std::vector<unsigned long long int> host_interaction_invalid_count(1);
  interaction_invalid_count.copy_to(host_interaction_invalid_count);
  if (host_interaction_invalid_count[0] != 0) {
    throw std::runtime_error(
        "GPU MAC counting failed truncated-domain validation for "
        + std::to_string(host_interaction_invalid_count[0])
        + " targets");
  }
  if (near_interaction_count < 0 || far_interaction_count < 0) {
    throw std::runtime_error("GPU interaction count overflow");
  }
  DeviceBuffer<int> near_leaf_indices(
      static_cast<std::size_t>(near_interaction_count));
  DeviceBuffer<int> far_hierarchy_indices(
      static_cast<std::size_t>(far_interaction_count));

  check_cuda(
      cudaMemset(
          interaction_invalid_count.data(),
          0,
          sizeof(unsigned long long int)),
      "cudaMemset interaction fill validation");
  construct_interaction_lists_kernel
      <<<interaction_block_count, interaction_block_size>>>(
          multipoles.data(),
          hierarchy_count,
          child_offsets.data(),
          child_hierarchy_indices.data(),
          root_hierarchy_indices.data(),
          static_cast<int>(problem.root_hierarchy_indices.size()),
          leaf_hierarchy_indices.data(),
          leaf_count,
          problem.mac_theta,
          problem.domain_volume,
          nullptr,
          nullptr,
          near_offsets.data(),
          far_offsets.data(),
          near_leaf_indices.data(),
          far_hierarchy_indices.data(),
          interaction_invalid_count.data());
  check_cuda(
      cudaGetLastError(),
      "construct_interaction_lists fill launch");
  check_cuda(
      cudaEventRecord(interaction_finish.get()),
      "cudaEventRecord interaction finish");
  check_cuda(
      cudaEventSynchronize(interaction_finish.get()),
      "CUDA interaction-list construction");
  float interaction_milliseconds = 0.0F;
  check_cuda(
      cudaEventElapsedTime(
          &interaction_milliseconds,
          interaction_start.get(),
          interaction_finish.get()),
      "cudaEventElapsedTime interactions");
  interaction_invalid_count.copy_to(host_interaction_invalid_count);
  if (host_interaction_invalid_count[0] != 0) {
    throw std::runtime_error(
        "GPU MAC construction failed truncated-domain validation for "
        + std::to_string(host_interaction_invalid_count[0])
        + " targets");
  }

  GpuTreecodeResult result;
  result.device_name = properties.name;
  result.source_preparation_seconds =
      static_cast<double>(source_milliseconds) * 1.0e-3;
  result.moment_seconds =
      static_cast<double>(moment_milliseconds) * 1.0e-3;
  result.interaction_seconds =
      static_cast<double>(interaction_milliseconds) * 1.0e-3;
  result.quadrature_seconds =
      static_cast<double>(quadrature_milliseconds) * 1.0e-3;
  result.near_interaction_count =
      static_cast<std::size_t>(near_interaction_count);
  result.far_interaction_count =
      static_cast<std::size_t>(far_interaction_count);
  if (!evaluate_potentials) {
    return result;
  }

  const int singular_axis_point_count =
      std::max(4, (quadrature_accuracy + 4) / 2);
  const auto singular_rule =
      gauss_legendre_on_unit_interval(singular_axis_point_count);
  std::vector<double> singular_nodes_host(
      static_cast<std::size_t>(singular_axis_point_count));
  std::vector<double> singular_weights_host(
      static_cast<std::size_t>(singular_axis_point_count));
  for (int i = 0; i < singular_axis_point_count; ++i) {
    singular_nodes_host[static_cast<std::size_t>(i)] =
        singular_rule[static_cast<std::size_t>(i)].first;
    singular_weights_host[static_cast<std::size_t>(i)] =
        singular_rule[static_cast<std::size_t>(i)].second;
  }
  DeviceBuffer<double> singular_nodes(singular_nodes_host);
  DeviceBuffer<double> singular_weights(singular_weights_host);
  DeviceBuffer<double> potentials(target_count_size);

  // Several warps share a target so the A100 can reach full warp occupancy
  // despite its resident-block limit.  Each warp processes an independent
  // stream of source interactions and owns a private coefficient workspace.
  const int block_size =
      expansion_order <= 6 ? 128
      : expansion_order <= 12 ? 128
      : 64;
  const int block_count = target_count;
  const int warp_count = block_size / 32;
  result.evaluation_threads_per_target = block_size;
  const std::size_t evaluation_shared_bytes =
      static_cast<std::size_t>(
          warp_count * coefficient_count + block_size + 2 * warp_count)
      * sizeof(double);
  CudaEvent start;
  CudaEvent finish;
  check_cuda(cudaEventRecord(start.get()), "cudaEventRecord start");
#define TREECODE_LAUNCH_EVALUATOR(FIXED_ORDER)                            \
  evaluate_treecode_multiwarp_kernel<FIXED_ORDER>                        \
      <<<block_count, block_size, evaluation_shared_bytes>>>(            \
          targets.data(),                                                \
          target_hierarchy_indices.data(),                               \
          target_leaf_indices.data(),                                    \
          target_count,                                                  \
          leaves.data(),                                                 \
          static_cast<int>(problem.leaves.size()),                       \
          samples.data(),                                                \
          sample_charges.data(),                                         \
          multipoles.data(),                                             \
          moments.data(),                                                \
          static_cast<int>(moment_count),                                \
          hierarchy_count,                                               \
          near_offsets.data(),                                           \
          near_leaf_indices.data(),                                      \
          far_offsets.data(),                                            \
          far_hierarchy_indices.data(),                                  \
          expansion_order,                                               \
          singular_nodes.data(),                                         \
          singular_weights.data(),                                       \
          singular_axis_point_count,                                     \
          potentials.data())
  switch (expansion_order) {
    case 2:
      TREECODE_LAUNCH_EVALUATOR(2);
      break;
    case 4:
      TREECODE_LAUNCH_EVALUATOR(4);
      break;
    case 6:
      TREECODE_LAUNCH_EVALUATOR(6);
      break;
    case 8:
      TREECODE_LAUNCH_EVALUATOR(8);
      break;
    case 12:
      TREECODE_LAUNCH_EVALUATOR(12);
      break;
    default:
      TREECODE_LAUNCH_EVALUATOR(0);
      break;
  }
#undef TREECODE_LAUNCH_EVALUATOR
  check_cuda(cudaGetLastError(), "evaluate_treecode_kernel launch");
  check_cuda(cudaEventRecord(finish.get()), "cudaEventRecord finish");
  check_cuda(cudaEventSynchronize(finish.get()), "CUDA kernel execution");

  float elapsed_milliseconds = 0.0F;
  check_cuda(
      cudaEventElapsedTime(
          &elapsed_milliseconds, start.get(), finish.get()),
      "cudaEventElapsedTime");

  result.kernel_seconds =
      static_cast<double>(elapsed_milliseconds) * 1.0e-3;
  result.evaluation_count = target_count_size;

  if (calculate_exact_l2_norms) {
    DeviceBuffer<double> target_weights(problem.target_weights);
    DeviceBuffer<double> norm_sums(7);
    DeviceBuffer<unsigned long long int> invalid_count(1);
    check_cuda(
        cudaMemset(norm_sums.data(), 0, 7 * sizeof(double)),
        "cudaMemset L2 norm sums");
    check_cuda(
        cudaMemset(
            invalid_count.data(), 0, sizeof(unsigned long long int)),
        "cudaMemset L2 invalid count");

    constexpr int reduction_block_size = 256;
    const int reduction_block_count = std::min(
        1024,
        (target_count + reduction_block_size - 1)
            / reduction_block_size);
    CudaEvent reduction_start;
    CudaEvent reduction_finish;
    check_cuda(
        cudaEventRecord(reduction_start.get()),
        "cudaEventRecord L2 reduction start");
    reduce_weighted_l2_kernel
        <<<reduction_block_count, reduction_block_size>>>(
            potentials.data(),
            targets.data(),
            samples.data(),
            target_weights.data(),
            target_count,
            norm_sums.data(),
            invalid_count.data());
    check_cuda(
        cudaGetLastError(),
        "reduce_weighted_l2_kernel launch");
    finalize_weighted_l2_kernel<<<1, 1>>>(norm_sums.data());
    check_cuda(
        cudaGetLastError(),
        "finalize_weighted_l2_kernel launch");
    check_cuda(
        cudaEventRecord(reduction_finish.get()),
        "cudaEventRecord L2 reduction finish");
    check_cuda(
        cudaEventSynchronize(reduction_finish.get()),
        "CUDA L2 reduction");

    float reduction_milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(
            &reduction_milliseconds,
            reduction_start.get(),
            reduction_finish.get()),
        "cudaEventElapsedTime L2 reduction");
    std::vector<double> host_norm_sums(7);
    std::vector<unsigned long long int> host_invalid_count(1);
    norm_sums.copy_to(host_norm_sums);
    invalid_count.copy_to(host_invalid_count);

    result.norm_reduction_seconds =
        static_cast<double>(reduction_milliseconds) * 1.0e-3;
    result.exact_error_squared = host_norm_sums[0];
    result.exact_norm_squared = host_norm_sums[1];
    result.approximation_norm_squared = host_norm_sums[2];
    result.exact_error = host_norm_sums[3];
    result.exact_norm = host_norm_sums[4];
    result.approximation_norm = host_norm_sums[5];
    result.relative_exact_error = host_norm_sums[6];
    result.nonfinite_count =
        static_cast<std::size_t>(host_invalid_count[0]);
    result.has_l2_norms = true;
  } else {
    result.potentials.resize(target_count_size);
    potentials.copy_to(result.potentials);
  }
  return result;
}

std::vector<double> evaluate_refinement_indicator_gpu(
    const std::vector<GpuQuadratureElement>& elements,
    int quadrature_accuracy,
    double* kernel_seconds)
{
  if (elements.empty()) {
    throw std::invalid_argument(
        "GPU refinement indicator contains no elements");
  }
  if (quadrature_accuracy < 1 || quadrature_accuracy > 20) {
    throw std::invalid_argument(
        "GPU refinement indicator accuracy must be between 1 and 20");
  }
  if (elements.size()
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "GPU refinement indicator element count is too large");
  }

  const int radial_count = (quadrature_accuracy + 4) / 2;
  const int face_count = (quadrature_accuracy + 3) / 2;
  const int edge_count = (quadrature_accuracy + 2) / 2;
  const int points_per_sub =
      radial_count * face_count * edge_count;
  for (const GpuQuadratureElement& element : elements) {
    const int sub_tetrahedron_count =
        element.vertex_count == 4 ? 1
        : element.vertex_count == 5 ? 2
        : element.vertex_count == 7 ? 4
        : 0;
    if (sub_tetrahedron_count == 0
        || element.sample_count
            != sub_tetrahedron_count * points_per_sub) {
      throw std::invalid_argument(
          "GPU refinement indicator element geometry is invalid");
    }
  }

  const auto split_rule = [](
      const std::vector<std::pair<double, double>>& rule) {
    std::pair<std::vector<double>, std::vector<double>> result;
    result.first.resize(rule.size());
    result.second.resize(rule.size());
    for (std::size_t i = 0; i < rule.size(); ++i) {
      result.first[i] = rule[i].first;
      result.second[i] = rule[i].second;
    }
    return result;
  };
  const auto radial_values = split_rule(
      gauss_legendre_on_unit_interval(radial_count));
  const auto face_values = split_rule(
      gauss_legendre_on_unit_interval(face_count));
  const auto edge_values = split_rule(
      gauss_legendre_on_unit_interval(edge_count));
  DeviceBuffer<GpuQuadratureElement> device_elements(elements);
  DeviceBuffer<double> radial_nodes(radial_values.first);
  DeviceBuffer<double> radial_weights(radial_values.second);
  DeviceBuffer<double> face_nodes(face_values.first);
  DeviceBuffer<double> face_weights(face_values.second);
  DeviceBuffer<double> edge_nodes(edge_values.first);
  DeviceBuffer<double> edge_weights(edge_values.second);
  DeviceBuffer<double> device_indicators(elements.size());

  CudaEvent start;
  CudaEvent finish;
  check_cuda(
      cudaEventRecord(start.get()),
      "cudaEventRecord indicator start");
  constexpr int indicator_block_size = 256;
  refinement_indicator_kernel
      <<<static_cast<int>(elements.size()), indicator_block_size>>>(
          device_elements.data(),
          static_cast<int>(elements.size()),
          radial_nodes.data(),
          radial_weights.data(),
          radial_count,
          face_nodes.data(),
          face_weights.data(),
          face_count,
          edge_nodes.data(),
          edge_weights.data(),
          edge_count,
          device_indicators.data());
  check_cuda(
      cudaGetLastError(),
      "refinement_indicator_kernel launch");
  check_cuda(
      cudaEventRecord(finish.get()),
      "cudaEventRecord indicator finish");
  check_cuda(
      cudaEventSynchronize(finish.get()),
      "CUDA refinement indicator");
  float elapsed_milliseconds = 0.0F;
  check_cuda(
      cudaEventElapsedTime(
          &elapsed_milliseconds, start.get(), finish.get()),
      "cudaEventElapsedTime refinement indicator");

  std::vector<double> indicators(elements.size());
  device_indicators.copy_to(indicators);
  if (kernel_seconds != nullptr) {
    *kernel_seconds =
        static_cast<double>(elapsed_milliseconds) * 1.0e-3;
  }
  return indicators;
}
