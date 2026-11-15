#ifndef TREECODE_GAUSSIAN_TEST_PROBLEM_HPP
#define TREECODE_GAUSSIAN_TEST_PROBLEM_HPP

#include <cmath>

namespace treecode_test_problem {

constexpr double pi = 3.141592653589793238462643383279502884;

// u(x,y,z) = exp(-a*x^2 - b*y^2 - c*z^2).  A compile-time scale permits
// controlled mesh-efficiency studies without changing the solver path.
#ifndef TREECODE_GAUSSIAN_SCALE
#define TREECODE_GAUSSIAN_SCALE 4.0
#endif
constexpr double gaussian_scale = TREECODE_GAUSSIAN_SCALE;
constexpr double gaussian_a = gaussian_scale * pi;
constexpr double gaussian_b = 2.0 * gaussian_scale * pi;
constexpr double gaussian_c = 3.0 * gaussian_scale * pi;

#ifndef TREECODE_GAUSSIAN_ALPHA
#define TREECODE_GAUSSIAN_ALPHA 4.0
#endif
constexpr double gaussian_alpha = TREECODE_GAUSSIAN_ALPHA;

#if defined(__CUDACC__)
#define TREECODE_TEST_HD __host__ __device__
#else
#define TREECODE_TEST_HD
#endif

TREECODE_TEST_HD inline double exact_value(
    double x,
    double y,
    double z)
{
#ifdef TREECODE_POSITIVE_GAUSSIAN
  const double radius = sqrt(x * x + y * y + z * z);
  const double root_alpha = sqrt(gaussian_alpha);
  if (radius < 1.0e-8) {
    return 2.0 * root_alpha / sqrt(pi);
  }
  return erf(root_alpha * radius) / radius;
#else
  return exp(
      -gaussian_a * x * x
      -gaussian_b * y * y
      -gaussian_c * z * z);
#endif
}

// Newton-potential density satisfying -Delta(u) = 4*pi*f.
TREECODE_TEST_HD inline double source_value(
    double x,
    double y,
    double z)
{
#ifdef TREECODE_POSITIVE_GAUSSIAN
  const double normalization =
      gaussian_alpha * sqrt(gaussian_alpha) / (pi * sqrt(pi));
  return normalization
      * exp(-gaussian_alpha * (x * x + y * y + z * z));
#else
  const double solution = exact_value(x, y, z);
  const double quadratic =
      gaussian_a * gaussian_a * x * x
      + gaussian_b * gaussian_b * y * y
      + gaussian_c * gaussian_c * z * z;
  return
      (2.0 * (gaussian_a + gaussian_b + gaussian_c)
       - 4.0 * quadratic)
      * solution / (4.0 * pi);
#endif
}

// Frobenius norm squared of the exact Hessian.  This is used only as an
// oracle mesh indicator in manufactured-solution experiments.
TREECODE_TEST_HD inline double exact_hessian_frobenius_squared(
    double x,
    double y,
    double z)
{
#ifdef TREECODE_POSITIVE_GAUSSIAN
  const double radius = sqrt(x * x + y * y + z * z);
  const double root_alpha = sqrt(gaussian_alpha);
  if (radius < 1.0e-5) {
    const double eigenvalue =
        -4.0 * gaussian_alpha * root_alpha / (3.0 * sqrt(pi));
    return 3.0 * eigenvalue * eigenvalue;
  }
  const double exponential = exp(-gaussian_alpha * radius * radius);
  const double g = erf(root_alpha * radius);
  const double g_prime = 2.0 * root_alpha * exponential / sqrt(pi);
  const double g_second =
      -4.0 * gaussian_alpha * root_alpha * radius
      * exponential / sqrt(pi);
  const double radial_first =
      g_prime / radius - g / (radius * radius);
  const double radial_second =
      g_second / radius
      - 2.0 * g_prime / (radius * radius)
      + 2.0 * g / (radius * radius * radius);
  const double tangential_eigenvalue = radial_first / radius;
  return radial_second * radial_second
      + 2.0 * tangential_eigenvalue * tangential_eigenvalue;
#else
  const double solution = exact_value(x, y, z);
  const double u_xx =
      (4.0 * gaussian_a * gaussian_a * x * x - 2.0 * gaussian_a)
      * solution;
  const double u_yy =
      (4.0 * gaussian_b * gaussian_b * y * y - 2.0 * gaussian_b)
      * solution;
  const double u_zz =
      (4.0 * gaussian_c * gaussian_c * z * z - 2.0 * gaussian_c)
      * solution;
  const double u_xy = 4.0 * gaussian_a * gaussian_b * x * y * solution;
  const double u_xz = 4.0 * gaussian_a * gaussian_c * x * z * solution;
  const double u_yz = 4.0 * gaussian_b * gaussian_c * y * z * solution;
  return u_xx * u_xx + u_yy * u_yy + u_zz * u_zz
      + 2.0 * (u_xy * u_xy + u_xz * u_xz + u_yz * u_yz);
#endif
}

#undef TREECODE_TEST_HD

} // namespace treecode_test_problem

#endif
