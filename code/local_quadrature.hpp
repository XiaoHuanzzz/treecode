#ifndef TREECODE_LOCAL_QUADRATURE_HPP
#define TREECODE_LOCAL_QUADRATURE_HPP

#include <AFEPack/Geometry.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace treecode_quadrature {

constexpr int dimension = 3;
constexpr int maximum_algebraic_accuracy = 20;
using Point = AFEPack::Point<dimension>;

struct QuadraturePoint
{
  Point point;
  double weight = 0.0;
};

inline std::vector<std::pair<double, double>>
gauss_legendre_on_unit_interval(int point_count)
{
  if (point_count <= 0) {
    throw std::invalid_argument(
        "Gauss-Legendre point count must be positive");
  }

  std::vector<std::pair<double, double>> rule(
      static_cast<std::size_t>(point_count));
  constexpr double pi = 3.141592653589793238462643383279502884;
  const int root_count = (point_count + 1) / 2;
  for (int root = 0; root < root_count; ++root) {
    double x = std::cos(
        pi * (static_cast<double>(root) + 0.75)
        / (static_cast<double>(point_count) + 0.5));
    double derivative = 0.0;
    bool converged = false;
    for (int iteration = 0; iteration < 100; ++iteration) {
      double p_previous = 1.0;
      double p_current = x;
      if (point_count == 1) {
        p_current = x;
      } else {
        for (int degree = 2; degree <= point_count; ++degree) {
          const double p_next =
              ((2.0 * degree - 1.0) * x * p_current
               - (degree - 1.0) * p_previous) / degree;
          p_previous = p_current;
          p_current = p_next;
        }
      }
      derivative =
          point_count * (x * p_current - p_previous) / (x * x - 1.0);
      const double step = p_current / derivative;
      x -= step;
      if (std::abs(step)
          <= 8.0 * std::numeric_limits<double>::epsilon()
             * std::max(1.0, std::abs(x))) {
        converged = true;
        break;
      }
    }
    if (!converged) {
      throw std::runtime_error(
          "Gauss-Legendre root iteration did not converge");
    }

    // Re-evaluate the derivative at the converged root.
    double p_previous = 1.0;
    double p_current = x;
    for (int degree = 2; degree <= point_count; ++degree) {
      const double p_next =
          ((2.0 * degree - 1.0) * x * p_current
           - (degree - 1.0) * p_previous) / degree;
      p_previous = p_current;
      p_current = p_next;
    }
    derivative =
        point_count * (x * p_current - p_previous) / (x * x - 1.0);
    const double weight =
        1.0 / ((1.0 - x * x) * derivative * derivative);

    // Map the symmetric rule from [-1,1] to [0,1].
    rule[static_cast<std::size_t>(root)] =
        {0.5 * (1.0 - x), weight};
    rule[static_cast<std::size_t>(point_count - 1 - root)] =
        {0.5 * (1.0 + x), weight};
  }
  return rule;
}

inline const std::vector<std::pair<double, double>>&
singular_duffy_axis_rule(int algebraic_accuracy)
{
  if (algebraic_accuracy < 1
      || algebraic_accuracy > maximum_algebraic_accuracy) {
    throw std::invalid_argument(
        "Unsupported singular quadrature accuracy");
  }
  static const auto rules = [] {
    std::array<std::vector<std::pair<double, double>>,
               maximum_algebraic_accuracy + 1> result;
    for (int accuracy = 1;
         accuracy <= maximum_algebraic_accuracy;
         ++accuracy) {
      const int order = std::max(4, (accuracy + 4) / 2);
      result[static_cast<std::size_t>(accuracy)] =
          gauss_legendre_on_unit_interval(order);
    }
    return result;
  }();
  return rules[static_cast<std::size_t>(algebraic_accuracy)];
}

struct DuffyAxisRules
{
  std::vector<std::pair<double, double>> radial;
  std::vector<std::pair<double, double>> face;
  std::vector<std::pair<double, double>> edge;
};

inline const DuffyAxisRules&
polynomial_duffy_axis_rules(int algebraic_accuracy)
{
  if (algebraic_accuracy < 0
      || algebraic_accuracy > maximum_algebraic_accuracy) {
    throw std::invalid_argument(
        "Unsupported polynomial quadrature accuracy");
  }
  static const auto rules = [] {
    std::array<DuffyAxisRules, maximum_algebraic_accuracy + 1> result;
    for (int accuracy = 0;
         accuracy <= maximum_algebraic_accuracy;
         ++accuracy) {
      result[static_cast<std::size_t>(accuracy)].radial =
          gauss_legendre_on_unit_interval((accuracy + 4) / 2);
      result[static_cast<std::size_t>(accuracy)].face =
          gauss_legendre_on_unit_interval((accuracy + 3) / 2);
      result[static_cast<std::size_t>(accuracy)].edge =
          gauss_legendre_on_unit_interval((accuracy + 2) / 2);
    }
    return result;
  }();
  return rules[static_cast<std::size_t>(algebraic_accuracy)];
}

inline double determinant(
    const Point& a,
    const Point& b,
    const Point& c)
{
  return
      a[0] * (b[1] * c[2] - b[2] * c[1])
      - a[1] * (b[0] * c[2] - b[2] * c[0])
      + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

inline void append_affine_tetrahedron_rule(
    const std::array<Point, 4>& vertices,
    int algebraic_accuracy,
    std::vector<QuadraturePoint>& result)
{
  if (algebraic_accuracy < 0
      || algebraic_accuracy > maximum_algebraic_accuracy) {
    throw std::invalid_argument(
        "Algebraic accuracy must be between 0 and "
        + std::to_string(maximum_algebraic_accuracy));
  }

  // Under the collapsed-coordinate map below, a polynomial of total degree p
  // has degrees at most p+2, p+1, and p in r, s, and t after multiplication
  // by the r^2*s Duffy Jacobian.
  const DuffyAxisRules& rules =
      polynomial_duffy_axis_rules(algebraic_accuracy);

  const Point e1 = vertices[1] - vertices[0];
  const Point e2 = vertices[2] - vertices[0];
  const Point e3 = vertices[3] - vertices[0];
  const double absolute_determinant =
      std::abs(determinant(e1, e2, e3));
  if (!(absolute_determinant
        > 64.0 * std::numeric_limits<double>::epsilon())) {
    throw std::runtime_error(
        "Degenerate sub-tetrahedron in local quadrature mapping");
  }

  for (const auto& r_entry : rules.radial) {
    const double r = r_entry.first;
    for (const auto& s_entry : rules.face) {
      const double s = s_entry.first;
      for (const auto& t_entry : rules.edge) {
        const double t = t_entry.first;
        const double lambda[4] = {
            1.0 - r,
            r * (1.0 - s),
            r * s * (1.0 - t),
            r * s * t};

        QuadraturePoint entry;
        for (int coordinate = 0; coordinate < dimension; ++coordinate) {
          entry.point[coordinate] =
              lambda[0] * vertices[0][coordinate]
              + lambda[1] * vertices[1][coordinate]
              + lambda[2] * vertices[2][coordinate]
              + lambda[3] * vertices[3][coordinate];
        }
        entry.weight =
            r_entry.second * s_entry.second * t_entry.second
            * r * r * s * absolute_determinant;
        result.push_back(entry);
      }
    }
  }
}

inline std::vector<std::array<int, 4>>
sub_tetrahedron_indices(std::size_t vertex_count)
{
  if (vertex_count == 4) {
    return {{{0, 1, 2, 3}}};
  }
  if (vertex_count == 5) {
    // twin_tetrahedron: vertex 2 is the midpoint of edge (1,3).
    return {{{0, 1, 2, 4}}, {{0, 2, 3, 4}}};
  }
  if (vertex_count == 7) {
    // four_tetrahedron: vertices 4,5,6 are midpoints of the three
    // edges on the face opposite vertex 0.
    return {
        {{0, 1, 5, 6}},
        {{0, 2, 4, 6}},
        {{0, 3, 4, 5}},
        {{0, 4, 5, 6}}};
  }
  throw std::invalid_argument(
      "Local quadrature supports only 4-, 5-, or 7-vertex elements");
}

inline double point_dot(const Point& left, const Point& right)
{
  return left[0] * right[0]
      + left[1] * right[1]
      + left[2] * right[2];
}

inline Point point_cross(const Point& left, const Point& right)
{
  Point result;
  result[0] = left[1] * right[2] - left[2] * right[1];
  result[1] = left[2] * right[0] - left[0] * right[2];
  result[2] = left[0] * right[1] - left[1] * right[0];
  return result;
}

inline double point_norm(const Point& point)
{
  return std::sqrt(point_dot(point, point));
}

inline bool point_in_tetrahedron(
    const Point& point,
    const std::array<Point, 4>& vertices)
{
  const Point edge1 = vertices[1] - vertices[0];
  const Point edge2 = vertices[2] - vertices[0];
  const Point edge3 = vertices[3] - vertices[0];
  const double jacobian = determinant(edge1, edge2, edge3);
  const double jacobian_scale =
      point_norm(edge1) * point_norm(edge2) * point_norm(edge3);
  if (std::abs(jacobian)
      <= 64.0 * std::numeric_limits<double>::epsilon()
          * jacobian_scale) {
    return false;
  }
  const Point relative = point - vertices[0];
  const double lambda1 = determinant(relative, edge2, edge3) / jacobian;
  const double lambda2 = determinant(edge1, relative, edge3) / jacobian;
  const double lambda3 = determinant(edge1, edge2, relative) / jacobian;
  const double lambda0 = 1.0 - lambda1 - lambda2 - lambda3;
  const double condition_estimate = std::max(
      1.0, jacobian_scale / std::abs(jacobian));
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

// Closed-form Newton/Coulomb potential of a unit-density tetrahedron at a
// target on or inside it:
//
//     A_K(x) = integral_K 1 / |x-y| dy.
//
// The divergence identity div((y-x)/|y-x|) = 2/|y-x| reduces the volume
// integral to four triangular-face integrals.  Each face integral is then
// expressed by its three logarithmic edge terms and its signed solid angle.
// This removes the 1/r singularity analytically; no volume quadrature is used
// in this function.
inline double constant_density_tetrahedron_coulomb_integral(
    const Point& target,
    const std::array<Point, 4>& vertices)
{
  static constexpr int faces[4][4] = {
      {1, 2, 3, 0},
      {0, 3, 2, 1},
      {0, 1, 3, 2},
      {0, 2, 1, 3}};

  double coordinate_scale = 1.0;
  for (const Point& vertex : vertices) {
    coordinate_scale = std::max(
        coordinate_scale,
        std::max(
            std::abs(vertex[0]),
            std::max(std::abs(vertex[1]), std::abs(vertex[2]))));
  }
  const double geometric_tolerance =
      256.0 * std::numeric_limits<double>::epsilon()
      * coordinate_scale;

  double volume_integral = 0.0;
  for (const auto& face : faces) {
    std::array<Point, 3> triangle{{
        vertices[static_cast<std::size_t>(face[0])],
        vertices[static_cast<std::size_t>(face[1])],
        vertices[static_cast<std::size_t>(face[2])]}};
    const Point& opposite =
        vertices[static_cast<std::size_t>(face[3])];

    Point edge01 = triangle[1] - triangle[0];
    Point edge02 = triangle[2] - triangle[0];
    Point normal = point_cross(edge01, edge02);
    double normal_length = point_norm(normal);
    if (!(normal_length > geometric_tolerance)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    // Ensure counterclockwise face vertices as seen from outside the
    // tetrahedron.  The target is then on the nonnegative-distance side of
    // every outward face.
    if (point_dot(opposite - triangle[0], normal) > 0.0) {
      std::swap(triangle[1], triangle[2]);
      edge01 = triangle[1] - triangle[0];
      edge02 = triangle[2] - triangle[0];
      normal = point_cross(edge01, edge02);
      normal_length = point_norm(normal);
    }
    normal *= 1.0 / normal_length;

    std::array<Point, 3> relative{{
        triangle[0] - target,
        triangle[1] - target,
        triangle[2] - target}};
    std::array<double, 3> radius{{
        point_norm(relative[0]),
        point_norm(relative[1]),
        point_norm(relative[2])}};
    double height = point_dot(relative[0], normal);
    if (height < -geometric_tolerance) {
      // The closed form below is intentionally restricted to a target on or
      // inside the tetrahedron, which is the treecode self-interaction case.
      return std::numeric_limits<double>::quiet_NaN();
    }
    height = std::max(0.0, height);
    if (height <= geometric_tolerance) {
      // This face has zero coefficient in the divergence-theorem sum.  The
      // skip also avoids an indeterminate 0*log(infinity) for edge targets.
      continue;
    }

    const double solid_angle_numerator = std::abs(
        point_dot(
            relative[0], point_cross(relative[1], relative[2])));
    const double solid_angle_denominator =
        radius[0] * radius[1] * radius[2]
        + radius[0] * point_dot(relative[1], relative[2])
        + radius[1] * point_dot(relative[2], relative[0])
        + radius[2] * point_dot(relative[0], relative[1]);
    const double solid_angle = 2.0 * std::atan2(
        solid_angle_numerator, solid_angle_denominator);

    double surface_integral = -height * solid_angle;
    for (int edge = 0; edge < 3; ++edge) {
      const int next = (edge + 1) % 3;
      const Point edge_vector =
          triangle[static_cast<std::size_t>(next)]
          - triangle[static_cast<std::size_t>(edge)];
      const double edge_length = point_norm(edge_vector);
      if (!(edge_length > geometric_tolerance)) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      Point edge_direction = edge_vector;
      edge_direction *= 1.0 / edge_length;
      const Point inward_normal = point_cross(normal, edge_direction);
      const double signed_distance = -point_dot(
          relative[static_cast<std::size_t>(edge)], inward_normal);
      const double radial_sum =
          radius[static_cast<std::size_t>(edge)]
          + radius[static_cast<std::size_t>(next)];
      const double logarithm_denominator = radial_sum - edge_length;
      if (!(logarithm_denominator > geometric_tolerance)) {
        if (std::abs(signed_distance) <= geometric_tolerance) {
          continue;
        }
        return std::numeric_limits<double>::quiet_NaN();
      }
      const double edge_logarithm = std::log1p(
          2.0 * edge_length / logarithm_denominator);
      surface_integral += signed_distance * edge_logarithm;
    }

    volume_integral += 0.5 * height * surface_integral;
  }

  return volume_integral;
}

inline std::size_t quadrature_point_count(
    std::size_t vertex_count,
    int algebraic_accuracy)
{
  if (algebraic_accuracy < 0
      || algebraic_accuracy > maximum_algebraic_accuracy) {
    throw std::invalid_argument("Unsupported algebraic accuracy");
  }
  const std::size_t sub_tetrahedron_count =
      sub_tetrahedron_indices(vertex_count).size();
  const std::size_t r_order =
      static_cast<std::size_t>((algebraic_accuracy + 4) / 2);
  const std::size_t s_order =
      static_cast<std::size_t>((algebraic_accuracy + 3) / 2);
  const std::size_t t_order =
      static_cast<std::size_t>((algebraic_accuracy + 2) / 2);
  return sub_tetrahedron_count * r_order * s_order * t_order;
}

inline std::vector<QuadraturePoint> build_physical_rule(
    const std::vector<Point>& vertices,
    int algebraic_accuracy)
{
  const auto decomposition = sub_tetrahedron_indices(vertices.size());
  std::vector<QuadraturePoint> result;
  result.reserve(
      quadrature_point_count(vertices.size(), algebraic_accuracy));
  for (const auto& indices : decomposition) {
    std::array<Point, 4> tetrahedron;
    for (int vertex = 0; vertex < 4; ++vertex) {
      tetrahedron[vertex] =
          vertices[static_cast<std::size_t>(indices[vertex])];
    }
    append_affine_tetrahedron_rule(
        tetrahedron, algebraic_accuracy, result);
  }
  return result;
}

} // namespace treecode_quadrature

#endif
