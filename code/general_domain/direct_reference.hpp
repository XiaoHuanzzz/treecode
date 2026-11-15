#ifndef GENERAL_DOMAIN_DIRECT_REFERENCE_HPP
#define GENERAL_DOMAIN_DIRECT_REFERENCE_HPP

// Include after compare_exact.cpp: reuse only its source function, affine
// quadrature and singular tetrahedron integration, never its HNL/MAC/tree sums.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gd {

using Tet = std::array<AFEPack::Point<3>, 4>;

struct DirectResult {
  std::vector<double> values;
  double charge = 0.0;
  double volume = 0.0;
  double preparation_s = 0.0;
  double evaluation_s = 0.0;
  // Base quadrature samples cached across tiles; locally subdivided samples
  // used by near corrections are not included in this diagnostic count.
  std::size_t quadrature_points = 0;
  // Number of source-tetrahedron/target pairs replacing the base quadrature.
  std::size_t near_corrections = 0;
};

namespace direct_detail {

using Point = AFEPack::Point<3>;

struct Geometry {
  double lower[3];
  double upper[3];
  double diameter_squared = 0.0;
};

inline Geometry geometry(const Tet& tet)
{
  Geometry result{};
  for (int d = 0; d < 3; ++d) {
    result.lower[d] = result.upper[d] = tet[0][d];
    for (int v = 1; v < 4; ++v) {
      result.lower[d] = std::min(result.lower[d], tet[v][d]);
      result.upper[d] = std::max(result.upper[d], tet[v][d]);
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = a + 1; b < 4; ++b) {
      double distance_squared = 0.0;
      for (int d = 0; d < 3; ++d) {
        const double delta = tet[a][d] - tet[b][d];
        distance_squared += delta * delta;
      }
      result.diameter_squared =
          std::max(result.diameter_squared, distance_squared);
    }
  }
  return result;
}

inline bool near(const Point& target, const Geometry& cell)
{
  double distance_squared = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double delta = std::max(
        0.0, std::max(cell.lower[d] - target[d], target[d] - cell.upper[d]));
    distance_squared += delta * delta;
  }
  // Conservative bounding-box test: a distance below half a cell diameter
  // warrants local quadrature refinement. This is NOT a multipole MAC.
  return distance_squared <= 0.25 * cell.diameter_squared;
}

inline std::array<Tet, 8> children(const Tet& v)
{
  Point a = v[0] + v[1]; a *= 0.5;
  Point b = v[0] + v[2]; b *= 0.5;
  Point c = v[0] + v[3]; c *= 0.5;
  Point d = v[1] + v[2]; d *= 0.5;
  Point e = v[1] + v[3]; e *= 0.5;
  Point f = v[2] + v[3]; f *= 0.5;
  // Four corner tetrahedra and the middle octahedron split along a--f.
  return {{
      {{v[0], a, b, c}}, {{v[1], a, d, e}},
      {{v[2], b, d, f}}, {{v[3], c, e, f}},
      {{a, b, c, f}}, {{a, b, d, f}},
      {{a, c, e, f}}, {{a, d, e, f}}
  }};
}

inline double regular(const Point& target, const Tet& tet, int p)
{
  const auto& rules = treecode_quadrature::polynomial_duffy_axis_rules(p);
  const Point edge1 = tet[1] - tet[0];
  const Point edge2 = tet[2] - tet[0];
  const Point edge3 = tet[3] - tet[0];
  const double jacobian = std::abs(
      treecode_quadrature::determinant(edge1, edge2, edge3));
  double value = 0.0;
  for (const auto& radial : rules.radial) {
    const double r = radial.first;
    for (const auto& face : rules.face) {
      const double s = face.first;
      for (const auto& edge : rules.edge) {
        const double t = edge.first;
        Point q;
        double distance_squared = 0.0;
        for (int dimension = 0; dimension < 3; ++dimension) {
          q[dimension] = tet[0][dimension]
              + r * (1.0 - s) * edge1[dimension]
              + r * s * (1.0 - t) * edge2[dimension]
              + r * s * t * edge3[dimension];
          const double delta = q[dimension] - target[dimension];
          distance_squared += delta * delta;
        }
        if (!(distance_squared > 1.0e-28)) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        const double weight = radial.second * face.second * edge.second
            * r * r * s * jacobian;
        value += weight * ::f(q) / std::sqrt(distance_squared);
      }
    }
  }
  return value;
}

inline double corrected(const Point& target, const Tet& tet, int p, int depth)
{
  if (treecode_quadrature::point_in_tetrahedron(target, tet)) {
    return ::singular_tetrahedron_integral(target, tet, p);
  }
  if (depth > 0 && near(target, geometry(tet))) {
    double value = 0.0;
    for (const Tet& child : children(tet)) {
      value += corrected(target, child, p, depth - 1);
    }
    return value;
  }
  return regular(target, tet, p);
}

struct Sample {
  double x, y, z, charge;
};

} // namespace direct_detail

inline DirectResult direct_reference(
    const std::vector<Tet>& tetrahedra,
    const std::vector<AFEPack::Point<3>>& targets,
    int p,
    int near_depth = 3)
{
  if (p < 1 || p > treecode_quadrature::maximum_algebraic_accuracy) {
    throw std::invalid_argument("Direct-reference quadrature p must be 1..20");
  }
  if (near_depth < 0 || near_depth > 8) {
    throw std::invalid_argument("Direct-reference near depth must be 0..8");
  }
  DirectResult result;
  result.values.assign(targets.size(), 0.0);
  // Initialize immutable rule caches before entering OpenMP regions.
  (void)treecode_quadrature::polynomial_duffy_axis_rules(p);
  (void)treecode_quadrature::singular_duffy_axis_rule(p);
  const std::size_t samples_per_tet =
      treecode_quadrature::quadrature_point_count(4, p);
  // At most ~8 MiB of source sample data, independently of the mesh size.
  const std::size_t tile_size = std::max<std::size_t>(
      1, 262144 / samples_per_tet);

  for (std::size_t begin = 0; begin < tetrahedra.size(); begin += tile_size) {
    const double preparation_start = omp_get_wtime();
    const std::size_t count = std::min(tile_size, tetrahedra.size() - begin);
    std::vector<direct_detail::Sample> samples(count * samples_per_tet);
    std::vector<direct_detail::Geometry> geometry(count);
    double tile_charge = 0.0;
    double tile_volume = 0.0;
    int invalid_geometry = 0;
    #pragma omp parallel for schedule(static) reduction(+:tile_charge,tile_volume,invalid_geometry)
    for (std::size_t local = 0; local < count; ++local) {
      const Tet& tet = tetrahedra[begin + local];
      geometry[local] = direct_detail::geometry(tet);
      const double volume = std::abs(treecode_quadrature::determinant(
          tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0])) / 6.0;
      if (!(volume > 0.0) || !std::isfinite(volume)) {
        ++invalid_geometry;
        continue;
      }
      std::vector<treecode_quadrature::QuadraturePoint> rule;
      rule.reserve(samples_per_tet);
      treecode_quadrature::append_affine_tetrahedron_rule(tet, p, rule);
      double charge = 0.0;
      for (std::size_t q = 0; q < samples_per_tet; ++q) {
        const auto& entry = rule[q];
        const double weighted_density = entry.weight * ::f(entry.point);
        samples[local * samples_per_tet + q] = {
            entry.point[0], entry.point[1], entry.point[2], weighted_density};
        charge += weighted_density;
      }
      tile_charge += charge;
      tile_volume += volume;
    }
    if (invalid_geometry != 0) {
      throw std::runtime_error("Non-positive tetrahedron in direct-reference mesh");
    }
    result.charge += tile_charge;
    result.volume += tile_volume;
    result.quadrature_points += samples.size();
    result.preparation_s += omp_get_wtime() - preparation_start;

    const double evaluation_start = omp_get_wtime();
    std::size_t corrections = 0;
    #pragma omp parallel for schedule(dynamic,1) reduction(+:corrections)
    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
      const auto& target = targets[target_index];
      double sum = 0.0;
      double compensation = 0.0;
      for (std::size_t local = 0; local < count; ++local) {
        double contribution = 0.0;
        if (direct_detail::near(target, geometry[local])) {
          ++corrections;
          contribution = direct_detail::corrected(
              target, tetrahedra[begin + local], p, near_depth);
        } else {
          const std::size_t offset = local * samples_per_tet;
          for (std::size_t q = 0; q < samples_per_tet; ++q) {
            const auto& sample = samples[offset + q];
            const double dx = sample.x - target[0];
            const double dy = sample.y - target[1];
            const double dz = sample.z - target[2];
            contribution += sample.charge / std::sqrt(dx * dx + dy * dy + dz * dz);
          }
        }
        const double corrected_value = contribution - compensation;
        const double next = sum + corrected_value;
        compensation = (next - sum) - corrected_value;
        sum = next;
      }
      result.values[target_index] += sum;
    }
    result.near_corrections += corrections;
    result.evaluation_s += omp_get_wtime() - evaluation_start;
  }

  for (double value : result.values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("Direct reference produced a nonfinite potential");
    }
  }
  return result;
}

} // namespace gd
#endif
