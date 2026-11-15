#ifndef TREECODE_GAUSSIAN_TEST_PROBLEM_HPP
#define TREECODE_GAUSSIAN_TEST_PROBLEM_HPP

#include <cmath>

namespace treecode_test_problem {

constexpr double pi = 3.141592653589793238462643383279502884;

// u(x,y,z) = exp(-a*x^2 - b*y^2 - c*z^2).
// These coefficients are four times the former (pi, 2*pi, 3*pi) values.
constexpr double gaussian_a = 4.0 * pi;
constexpr double gaussian_b = 8.0 * pi;
constexpr double gaussian_c = 12.0 * pi;

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
  return exp(
      -gaussian_a * x * x
      -gaussian_b * y * y
      -gaussian_c * z * z);
}

// Newton-potential density satisfying -Delta(u) = 4*pi*f.
TREECODE_TEST_HD inline double source_value(
    double x,
    double y,
    double z)
{
  const double solution = exact_value(x, y, z);
  const double quadratic =
      gaussian_a * gaussian_a * x * x
      + gaussian_b * gaussian_b * y * y
      + gaussian_c * gaussian_c * z * z;
  return
      (2.0 * (gaussian_a + gaussian_b + gaussian_c)
       - 4.0 * quadratic)
      * solution / (4.0 * pi);
}

#undef TREECODE_TEST_HD

} // namespace treecode_test_problem

#endif
