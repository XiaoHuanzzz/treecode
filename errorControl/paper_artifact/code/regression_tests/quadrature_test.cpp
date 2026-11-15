#include "local_quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using treecode_quadrature::Point;
using treecode_quadrature::QuadraturePoint;

double factorial(int value)
{
  double result = 1.0;
  for (int i = 2; i <= value; ++i) {
    result *= i;
  }
  return result;
}

double exact_unit_tetrahedron_monomial(int x, int y, int z)
{
  return
      factorial(x) * factorial(y) * factorial(z)
      / factorial(x + y + z + 3);
}

std::vector<Point> reference_vertices(int vertex_count)
{
  std::vector<Point> vertices(static_cast<std::size_t>(vertex_count));
  vertices[0] = {0.0, 0.0, 0.0};
  vertices[1] = {1.0, 0.0, 0.0};
  if (vertex_count == 4) {
    vertices[2] = {0.0, 1.0, 0.0};
    vertices[3] = {0.0, 0.0, 1.0};
  } else if (vertex_count == 5) {
    vertices[2] = {0.5, 0.5, 0.0};
    vertices[3] = {0.0, 1.0, 0.0};
    vertices[4] = {0.0, 0.0, 1.0};
  } else if (vertex_count == 7) {
    vertices[2] = {0.0, 1.0, 0.0};
    vertices[3] = {0.0, 0.0, 1.0};
    vertices[4] = {0.0, 0.5, 0.5};
    vertices[5] = {0.5, 0.0, 0.5};
    vertices[6] = {0.5, 0.5, 0.0};
  } else {
    throw std::invalid_argument("Unsupported reference vertex count");
  }
  return vertices;
}

double integer_power(double value, int exponent)
{
  double result = 1.0;
  for (int i = 0; i < exponent; ++i) {
    result *= value;
  }
  return result;
}

double duffy_constant_density_reference(
    const Point& target,
    const std::array<Point, 4>& vertices)
{
  static constexpr int faces[4][3] = {
      {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  const auto rule =
      treecode_quadrature::gauss_legendre_on_unit_interval(64);
  double integral = 0.0;
  for (const auto& face : faces) {
    const Point& a = vertices[static_cast<std::size_t>(face[0])];
    const Point& b = vertices[static_cast<std::size_t>(face[1])];
    const Point& c = vertices[static_cast<std::size_t>(face[2])];
    const Point ea = a - target;
    const Point eb = b - target;
    const Point ec = c - target;
    const double determinant = std::abs(
        treecode_quadrature::determinant(ea, eb, ec));
    for (const auto& s_entry : rule) {
      const double s = s_entry.first;
      for (const auto& r_entry : rule) {
        const double r = r_entry.first;
        Point face_point = a;
        face_point *= 1.0 - s;
        Point contribution = b;
        contribution *= s * (1.0 - r);
        face_point += contribution;
        contribution = c;
        contribution *= s * r;
        face_point += contribution;
        const double radial_length = (face_point - target).length();
        // The radial Duffy integral is exact for unit density:
        // integral_0^1 t dt = 1/2.
        integral += 0.5 * s_entry.second * r_entry.second
            * determinant * s / radial_length;
      }
    }
  }
  return integral;
}

} // namespace

int main(int argc, char* argv[])
{
  try {
    const int maximum_accuracy =
        argc >= 2 ? std::stoi(argv[1]) : 12;
    if (maximum_accuracy < 1
        || maximum_accuracy
            > treecode_quadrature::maximum_algebraic_accuracy) {
      throw std::invalid_argument("Invalid maximum test accuracy");
    }

    double maximum_absolute_error = 0.0;
    double maximum_scaled_error = 0.0;
    for (const int vertex_count : {4, 5, 7}) {
      const auto vertices = reference_vertices(vertex_count);
      for (int accuracy = 1; accuracy <= maximum_accuracy; ++accuracy) {
        const std::vector<QuadraturePoint> rule =
            treecode_quadrature::build_physical_rule(vertices, accuracy);
        for (int x_degree = 0;
             x_degree <= accuracy;
             ++x_degree) {
          for (int y_degree = 0;
               y_degree <= accuracy - x_degree;
               ++y_degree) {
            for (int z_degree = 0;
                 z_degree <= accuracy - x_degree - y_degree;
                 ++z_degree) {
              double approximation = 0.0;
              for (const QuadraturePoint& point : rule) {
                approximation +=
                    point.weight
                    * integer_power(point.point[0], x_degree)
                    * integer_power(point.point[1], y_degree)
                    * integer_power(point.point[2], z_degree);
              }
              const double exact = exact_unit_tetrahedron_monomial(
                  x_degree, y_degree, z_degree);
              const double error = std::abs(approximation - exact);
              maximum_absolute_error =
                  std::max(maximum_absolute_error, error);
              maximum_scaled_error =
                  std::max(
                      maximum_scaled_error,
                      error
                          / std::max(
                              std::abs(exact),
                              100.0
                                  * std::numeric_limits<double>::epsilon()));
              if (error
                  > 2.0e-12
                    * std::max(1.0, std::abs(exact))) {
                throw std::runtime_error(
                    "Monomial exactness test failed for "
                    + std::to_string(vertex_count)
                    + " vertices at algebraic accuracy "
                    + std::to_string(accuracy));
              }
            }
          }
        }
        std::cout << "vertices=" << vertex_count
                  << ", accuracy=" << accuracy
                  << ", points=" << rule.size() << '\n';
      }
    }

    using TestPoint = treecode_quadrature::Point;
    const std::array<TestPoint, 4> unit_tetrahedron{{
        TestPoint{0.0, 0.0, 0.0},
        TestPoint{1.0, 0.0, 0.0},
        TestPoint{0.0, 1.0, 0.0},
        TestPoint{0.0, 0.0, 1.0}}};
    TestPoint unit_target{0.25, 0.25, 0.25};
    const double unit_analytic =
        treecode_quadrature::
            constant_density_tetrahedron_coulomb_integral(
                unit_target, unit_tetrahedron);
    const double unit_reference =
        duffy_constant_density_reference(unit_target, unit_tetrahedron);

    const std::array<TestPoint, 4> skew_tetrahedron{{
        TestPoint{0.0, 0.0, 0.0},
        TestPoint{2.0, 0.1, 0.0},
        TestPoint{-0.2, 1.3, 0.1},
        TestPoint{0.1, -0.1, 1.7}}};
    TestPoint skew_target{0.475, 0.325, 0.45};
    const double skew_analytic =
        treecode_quadrature::
            constant_density_tetrahedron_coulomb_integral(
                skew_target, skew_tetrahedron);
    const double skew_reference =
        duffy_constant_density_reference(skew_target, skew_tetrahedron);

    std::cout << std::setprecision(16)
              << "Unit Coulomb analytic/reference: "
              << unit_analytic << " / " << unit_reference << '\n'
              << "Skew Coulomb analytic/reference: "
              << skew_analytic << " / " << skew_reference << '\n';

    for (const auto& comparison : {
             std::pair<double, double>{unit_analytic, unit_reference},
             std::pair<double, double>{skew_analytic, skew_reference}}) {
      if (!std::isfinite(comparison.first)
          || std::abs(comparison.first - comparison.second)
              > 2.0e-11 * std::max(1.0, std::abs(comparison.second))) {
        throw std::runtime_error(
            "Closed-form tetrahedron Coulomb integral test failed");
      }
    }
    std::cout << std::setprecision(16)
              << "Maximum absolute monomial error: "
              << maximum_absolute_error << '\n'
              << "Maximum scaled monomial error: "
              << maximum_scaled_error << '\n'
              << "Closed-form unit-tetrahedron Coulomb integral: "
              << unit_analytic << '\n'
              << "Closed-form skew-tetrahedron Coulomb integral: "
              << skew_analytic << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Quadrature test error: " << error.what() << '\n';
    return 1;
  }
}
