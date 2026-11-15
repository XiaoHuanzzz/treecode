#ifndef TREECODE_CARTESIAN_MULTIPOLE_HPP
#define TREECODE_CARTESIAN_MULTIPOLE_HPP

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace treecode_multipole {

constexpr int maximum_order = 20;
constexpr int maximum_coefficient_count = 1771;

#if defined(__CUDACC__)
#define TREECODE_HD __host__ __device__
#else
#define TREECODE_HD
#endif

TREECODE_HD constexpr int degree_offset(int degree)
{
  return degree * (degree + 1) * (degree + 2) / 6;
}

TREECODE_HD constexpr int coefficient_count(int order)
{
  return (order + 1) * (order + 2) * (order + 3) / 6;
}

// Coefficients of the same total degree are ordered by increasing x exponent,
// then increasing y exponent; z is determined by degree - x - y.
TREECODE_HD constexpr int coefficient_index(int x, int y, int z)
{
  const int degree = x + y + z;
  return degree_offset(degree)
      + x * (degree + 1) - x * (x - 1) / 2 + y;
}

inline void validate_order(int order)
{
  if (order < 1 || order > maximum_order) {
    throw std::invalid_argument(
        "Multipole expansion order must be between 1 and "
        + std::to_string(maximum_order));
  }
}

inline std::vector<double> zero_moments(int order)
{
  validate_order(order);
  return std::vector<double>(
      static_cast<std::size_t>(coefficient_count(order)), 0.0);
}

inline void accumulate_raw_moments(
    std::vector<double>& moments,
    int order,
    double dx,
    double dy,
    double dz,
    double weighted_density)
{
  validate_order(order);
  if (moments.size()
      != static_cast<std::size_t>(coefficient_count(order))) {
    throw std::invalid_argument("Multipole moment vector has the wrong size");
  }

  std::array<double, maximum_order + 1> x_power{};
  std::array<double, maximum_order + 1> y_power{};
  std::array<double, maximum_order + 1> z_power{};
  x_power[0] = y_power[0] = z_power[0] = 1.0;
  for (int degree = 1; degree <= order; ++degree) {
    x_power[degree] = x_power[degree - 1] * dx;
    y_power[degree] = y_power[degree - 1] * dy;
    z_power[degree] = z_power[degree - 1] * dz;
  }

  for (int degree = 0; degree <= order; ++degree) {
    for (int x = 0; x <= degree; ++x) {
      for (int y = 0; y <= degree - x; ++y) {
        const int z = degree - x - y;
        moments[static_cast<std::size_t>(
            coefficient_index(x, y, z))] +=
            weighted_density * x_power[x] * y_power[y] * z_power[z];
      }
    }
  }
}

// Build the homogeneous Taylor coefficients C_(a,b,c)(R) in
//
//   1 / |R-d| = sum C_(a,b,c)(R) d_x^a d_y^b d_z^c.
//
// The recurrence is the coefficient form of the Legendre recurrence.
// `coefficients` must provide coefficient_count(order) entries.
TREECODE_HD inline bool build_taylor_coefficients(
    double rx,
    double ry,
    double rz,
    int order,
    double* coefficients)
{
  const double radius_squared = rx * rx + ry * ry + rz * rz;
  if (radius_squared < 1.0e-30) {
    return false;
  }

  coefficients[0] = 1.0 / sqrt(radius_squared);
  for (int degree = 1; degree <= order; ++degree) {
    const double first_factor =
        (2.0 * degree - 1.0) / (degree * radius_squared);
    const double second_factor =
        (degree - 1.0) / (degree * radius_squared);

    for (int x = 0; x <= degree; ++x) {
      for (int y = 0; y <= degree - x; ++y) {
        const int z = degree - x - y;
        double first = 0.0;
        double second = 0.0;
        if (x >= 1) {
          first += rx * coefficients[coefficient_index(x - 1, y, z)];
        }
        if (y >= 1) {
          first += ry * coefficients[coefficient_index(x, y - 1, z)];
        }
        if (z >= 1) {
          first += rz * coefficients[coefficient_index(x, y, z - 1)];
        }
        if (x >= 2) {
          second += coefficients[coefficient_index(x - 2, y, z)];
        }
        if (y >= 2) {
          second += coefficients[coefficient_index(x, y - 2, z)];
        }
        if (z >= 2) {
          second += coefficients[coefficient_index(x, y, z - 2)];
        }
        coefficients[coefficient_index(x, y, z)] =
            first_factor * first - second_factor * second;
      }
    }
  }
  return true;
}

inline double evaluate(
    const std::vector<double>& moments,
    int order,
    double rx,
    double ry,
    double rz)
{
  validate_order(order);
  const int count = coefficient_count(order);
  if (moments.size() != static_cast<std::size_t>(count)) {
    throw std::invalid_argument("Multipole moment vector has the wrong size");
  }
  std::array<double, maximum_coefficient_count> coefficients;
  if (!build_taylor_coefficients(
          rx, ry, rz, order, coefficients.data())) {
    return 0.0;
  }
  double value = 0.0;
  for (int index = 0; index < count; ++index) {
    value += moments[static_cast<std::size_t>(index)]
        * coefficients[static_cast<std::size_t>(index)];
  }
  return value;
}

#undef TREECODE_HD

} // namespace treecode_multipole

#endif
