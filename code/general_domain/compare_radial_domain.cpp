// Radially symmetric source used only by the separate general-domain study.
#include <cmath>
#include <stdexcept>

#define TREECODE_GENERAL_DOMAIN_SOURCE_DEFINED
#define TREECODE_GENERAL_DOMAIN_SOURCE_DESCRIPTION \
  "exp(-4*((x+2)^2+(y+2)^2+z^2)), truncated to the mesh domain"
#define TREECODE_GAUSSIAN_TEST_PROBLEM_HPP
namespace treecode_test_problem {
constexpr double pi = 3.141592653589793238462643383279502884;
inline double source_value(double x, double y, double z) {
  x += 2.0;
  y += 2.0;
  return std::exp(-4.0*(x*x+y*y+z*z));
}
inline double exact_value(double, double, double) {
  throw std::logic_error(
      "The radial general-domain study uses a numerical direct reference");
}
}

#include "compare_general_domain.cpp"
