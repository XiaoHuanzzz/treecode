#include "cartesian_multipole.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Source
{
  double x;
  double y;
  double z;
  double charge;
};

double direct_value(
    const std::vector<Source>& sources,
    const std::array<double, 3>& target)
{
  double value = 0.0;
  for (const Source& source : sources) {
    const double dx = target[0] - source.x;
    const double dy = target[1] - source.y;
    const double dz = target[2] - source.z;
    value += source.charge / std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  return value;
}

double pointwise_legendre_expansion(
    const std::vector<Source>& sources,
    const std::array<double, 3>& target,
    int order)
{
  const double radius_squared =
      target[0] * target[0]
      + target[1] * target[1]
      + target[2] * target[2];
  const double radius = std::sqrt(radius_squared);
  double value = 0.0;
  for (const Source& source : sources) {
    const double source_squared =
        source.x * source.x + source.y * source.y + source.z * source.z;
    const double dot =
        target[0] * source.x
        + target[1] * source.y
        + target[2] * source.z;
    double previous_previous = 1.0 / radius;
    value += source.charge * previous_previous;
    if (order == 0) {
      continue;
    }
    double previous = dot / (radius_squared * radius);
    value += source.charge * previous;
    for (int degree = 2; degree <= order; ++degree) {
      const double current =
          (2.0 * degree - 1.0) / degree
              * (dot / radius_squared) * previous
          - (degree - 1.0) / degree
              * (source_squared / radius_squared) * previous_previous;
      value += source.charge * current;
      previous_previous = previous;
      previous = current;
    }
  }
  return value;
}

} // namespace

int main()
{
  const std::vector<Source> sources = {
      {0.20, -0.15, 0.10, 1.20},
      {-0.30, 0.25, -0.05, -0.35},
      {0.12, 0.18, 0.22, 0.70},
      {-0.08, -0.21, 0.16, 0.45}};
  const std::array<double, 3> target = {3.1, -2.4, 4.0};
  const double exact = direct_value(sources, target);

  double previous_error = 1.0e100;
  for (int order = 1;
       order <= treecode_multipole::maximum_order;
       ++order) {
    std::vector<double> moments =
        treecode_multipole::zero_moments(order);
    for (const Source& source : sources) {
      treecode_multipole::accumulate_raw_moments(
          moments, order, source.x, source.y, source.z, source.charge);
    }
    const double computed = treecode_multipole::evaluate(
        moments, order, target[0], target[1], target[2]);
    const double reference =
        pointwise_legendre_expansion(sources, target, order);
    const double identity_error = std::abs(computed - reference);
    const double direct_error = std::abs(computed - exact);
    const double tolerance =
        2.0e-13 * std::max(1.0, std::abs(reference));
    if (identity_error > tolerance) {
      throw std::runtime_error(
          "Cartesian multipole recurrence failed at order "
          + std::to_string(order));
    }
    if (order >= 3 && direct_error > 1.05 * previous_error) {
      throw std::runtime_error(
          "Multipole error failed to decrease at order "
          + std::to_string(order));
    }
    previous_error = direct_error;
    std::cout << "order=" << std::setw(2) << order
              << ", coefficients="
              << std::setw(4)
              << treecode_multipole::coefficient_count(order)
              << ", direct error=" << std::setprecision(8)
              << std::scientific << direct_error << '\n';
  }
  return 0;
}
