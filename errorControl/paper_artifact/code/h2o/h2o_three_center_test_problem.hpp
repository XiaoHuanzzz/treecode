#ifndef TREECODE_H2O_THREE_CENTER_TEST_PROBLEM_HPP
#define TREECODE_H2O_THREE_CENTER_TEST_PROBLEM_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "gaussian_test_problem.hpp"

namespace treecode_h2o_three_center_problem {

constexpr double angstrom_to_bohr = 1.8897261246257702;
constexpr std::size_t atom_count = 3;

struct Center
{
  const char* symbol;
  double x;
  double y;
  double z;
};

// Gas-phase H2O geometry in Bohr.  These coordinates define only the three
// centers of the manufactured test; no physical atomic density is assumed.
constexpr std::array<Center, atom_count> centers{{
    {"O", 0.0, 0.0, 0.0},
    {"H",  0.756950327 * angstrom_to_bohr,
           0.0,
           0.585882276 * angstrom_to_bohr},
    {"H", -0.756950327 * angstrom_to_bohr,
           0.0,
           0.585882276 * angstrom_to_bohr}}};

inline double squared_distance(
    double x,
    double y,
    double z,
    std::size_t center_index)
{
  const Center& center = centers[center_index];
  const double dx = x - center.x;
  const double dy = y - center.y;
  const double dz = z - center.z;
  return dx * dx + dy * dy + dz * dz;
}

// This is exactly the compare_exact manufactured solution translated from
// the origin to center A.  The coefficients a, b, and c are unchanged.
inline double shifted_exact_value(
    double x,
    double y,
    double z,
    std::size_t center_index)
{
  if (center_index >= atom_count) return 0.0;
  const Center& center = centers[center_index];
  const double dx = x - center.x;
  const double dy = y - center.y;
  const double dz = z - center.z;
  return std::exp(
      -treecode_test_problem::gaussian_a * dx * dx
      -treecode_test_problem::gaussian_b * dy * dy
      -treecode_test_problem::gaussian_c * dz * dz);
}

// The translated density satisfying -Delta(u_A) = 4*pi*f_A.
inline double shifted_source_value(
    double x,
    double y,
    double z,
    std::size_t center_index)
{
  if (center_index >= atom_count) return 0.0;
  const Center& center = centers[center_index];
  const double dx = x - center.x;
  const double dy = y - center.y;
  const double dz = z - center.z;
  const double solution = shifted_exact_value(x, y, z, center_index);
  const double quadratic =
      treecode_test_problem::gaussian_a
          * treecode_test_problem::gaussian_a * dx * dx
      + treecode_test_problem::gaussian_b
          * treecode_test_problem::gaussian_b * dy * dy
      + treecode_test_problem::gaussian_c
          * treecode_test_problem::gaussian_c * dz * dz;
  return
      (2.0 * (treecode_test_problem::gaussian_a
              + treecode_test_problem::gaussian_b
              + treecode_test_problem::gaussian_c)
       - 4.0 * quadratic)
      * solution / (4.0 * treecode_test_problem::pi);
}

inline double source_value(double x, double y, double z)
{
  double value = 0.0;
  for (std::size_t center = 0; center < atom_count; ++center) {
    value += shifted_source_value(x, y, z, center);
  }
  return value;
}

inline double becke_switch(double mu)
{
  double value = std::max(-1.0, std::min(1.0, mu));
  for (int iteration = 0; iteration < 3; ++iteration) {
    value = 1.5 * value - 0.5 * value * value * value;
  }
  return value;
}

inline std::array<double, atom_count> becke_weights(
    double x,
    double y,
    double z)
{
  std::array<double, atom_count> distances{};
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    distances[atom] = std::sqrt(squared_distance(x, y, z, atom));
  }

  std::array<double, atom_count> cell_functions{};
  cell_functions.fill(1.0);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    for (std::size_t other = 0; other < atom_count; ++other) {
      if (atom == other) continue;
      const double dx = centers[atom].x - centers[other].x;
      const double dy = centers[atom].y - centers[other].y;
      const double dz = centers[atom].z - centers[other].z;
      const double separation = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double mu =
          (distances[atom] - distances[other]) / separation;
      cell_functions[atom] *= 0.5 * (1.0 - becke_switch(mu));
    }
  }

  double normalization = 0.0;
  for (double value : cell_functions) normalization += value;
  if (!(normalization > 0.0) || !std::isfinite(normalization)) {
    // This should be unreachable for distinct centers, but the fallback
    // keeps the partition finite at a numerically degenerate point.
    cell_functions.fill(1.0 / static_cast<double>(atom_count));
    return cell_functions;
  }
  for (double& value : cell_functions) value /= normalization;
  return cell_functions;
}

inline double becke_weighted_source_value(
    double x,
    double y,
    double z,
    std::size_t atom_index)
{
  if (atom_index >= atom_count) return 0.0;
  return becke_weights(x, y, z)[atom_index] * source_value(x, y, z);
}

inline long double gaussian_interval_integral(
    long double coefficient,
    long double center,
    long double lower,
    long double upper)
{
  const long double root = std::sqrt(coefficient);
  return std::sqrt(static_cast<long double>(treecode_test_problem::pi))
      * (std::erf(root * (upper - center))
         - std::erf(root * (lower - center)))
      / (2.0L * root);
}

// Since f=-Delta(u)/(4*pi), its exact integral over the truncated cube is
// obtained stably from the outward Gaussian flux on the six boundary faces.
inline double truncated_source_integral(
    long double lower = -5.0L,
    long double upper = 5.0L)
{
  const std::array<long double, 3> coefficients{{
      static_cast<long double>(treecode_test_problem::gaussian_a),
      static_cast<long double>(treecode_test_problem::gaussian_b),
      static_cast<long double>(treecode_test_problem::gaussian_c)}};
  long double total = 0.0L;
  for (const Center& atom : centers) {
    const std::array<long double, 3> center{{
        static_cast<long double>(atom.x),
        static_cast<long double>(atom.y),
        static_cast<long double>(atom.z)}};
    std::array<long double, 3> interval_integrals{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      interval_integrals[axis] = gaussian_interval_integral(
          coefficients[axis], center[axis], lower, upper);
    }
    long double flux = 0.0L;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const long double upper_offset = upper - center[axis];
      const long double lower_offset = lower - center[axis];
      const long double upper_derivative =
          -2.0L * coefficients[axis] * upper_offset
          * std::exp(-coefficients[axis]
                     * upper_offset * upper_offset);
      const long double lower_derivative =
          -2.0L * coefficients[axis] * lower_offset
          * std::exp(-coefficients[axis]
                     * lower_offset * lower_offset);
      long double transverse_integral = 1.0L;
      for (std::size_t other = 0; other < 3; ++other) {
        if (other != axis) {
          transverse_integral *= interval_integrals[other];
        }
      }
      flux += (upper_derivative - lower_derivative)
          * transverse_integral;
    }
    total -= flux
        / (4.0L * static_cast<long double>(treecode_test_problem::pi));
  }
  return static_cast<double>(total);
}

inline double exact_value(double x, double y, double z)
{
  double value = 0.0;
  for (std::size_t center = 0; center < atom_count; ++center) {
    value += shifted_exact_value(x, y, z, center);
  }
  return value;
}

// Same indicator as compare_exact, translated to center A.
inline double atomic_radial_refinement_indicator(
    double x,
    double y,
    double z,
    std::size_t center_index)
{
  return 1.0 / std::sqrt(std::max(
      squared_distance(x, y, z, center_index), 1.0e-30));
}

inline double atomic_gaussian_radial_refinement_indicator(
    double x,
    double y,
    double z,
    std::size_t center_index)
{
  const Center& center = centers[center_index];
  const double dx = x - center.x;
  const double dy = y - center.y;
  const double dz = z - center.z;
  const double ellipsoidal_radius_squared =
      dx * dx + 2.0 * dy * dy + 3.0 * dz * dz;
  return 1.0 / std::sqrt(std::max(
      ellipsoidal_radius_squared, 1.0e-30));
}

inline double gaussian_radial_refinement_indicator_value(
    double x,
    double y,
    double z)
{
  double value = 0.0;
  for (std::size_t center = 0; center < atom_count; ++center) {
    value = std::max(
        value,
        atomic_gaussian_radial_refinement_indicator(x, y, z, center));
  }
  return value;
}

inline double refinement_indicator_value(double x, double y, double z)
{
  double value = 0.0;
  for (std::size_t center = 0; center < atom_count; ++center) {
    value = std::max(
        value,
        atomic_radial_refinement_indicator(x, y, z, center));
  }
  return value;
}

inline const char* description()
{
  return "Becke-weighted partition of the three-center Gaussian source";
}

} // namespace treecode_h2o_three_center_problem

#endif
