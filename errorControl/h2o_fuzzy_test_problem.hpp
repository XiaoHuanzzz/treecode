#ifndef TREECODE_H2O_FUZZY_TEST_PROBLEM_HPP
#define TREECODE_H2O_FUZZY_TEST_PROBLEM_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "gaussian_test_problem.hpp"

namespace treecode_h2o_fuzzy_problem {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double angstrom_to_bohr = 1.8897261246257702;
constexpr std::size_t atom_count = 3;

struct Atom
{
  const char* symbol;
  double x;
  double y;
  double z;
  double electron_count;
  double gaussian_exponent;
};

// Experimental gas-phase H2O geometry.  Oxygen is at the origin and the
// molecule lies in the x-z plane.  Coordinates are stored in Bohr.
constexpr std::array<Atom, atom_count> atoms{{
    {"O", 0.0, 0.0, 0.0, 8.0, 8.0},
    {"H",  0.756950327 * angstrom_to_bohr,
           0.0,
           0.585882276 * angstrom_to_bohr,
           1.0,
           2.0},
    {"H", -0.756950327 * angstrom_to_bohr,
           0.0,
           0.585882276 * angstrom_to_bohr,
           1.0,
           2.0}}};

inline double squared_distance(
    double x,
    double y,
    double z,
    const Atom& atom)
{
  const double dx = x - atom.x;
  const double dy = y - atom.y;
  const double dz = z - atom.z;
  return dx * dx + dy * dy + dz * dz;
}

inline double atomic_density(
    double x,
    double y,
    double z,
    const Atom& atom)
{
  // Normalized three-dimensional Gaussian containing atom.electron_count
  // electrons on the full space.
  const double normalization =
      atom.electron_count
      * std::pow(atom.gaussian_exponent / pi, 1.5);
  return normalization
      * std::exp(-atom.gaussian_exponent
                 * squared_distance(x, y, z, atom));
}

inline double electron_density(double x, double y, double z)
{
  double density = 0.0;
  for (const Atom& atom : atoms) {
    density += atomic_density(x, y, z, atom);
  }
  return density;
}

inline double atomic_source_value(
    double x,
    double y,
    double z,
    std::size_t atom_index)
{
  return atomic_density(x, y, z, atoms[atom_index]);
}

inline double becke_polynomial(double value)
{
  return 1.5 * value - 0.5 * value * value * value;
}

inline std::array<double, atom_count> fuzzy_weights(
    double x,
    double y,
    double z)
{
  std::array<double, atom_count> distances{};
  std::array<double, atom_count> unnormalized{};
  for (std::size_t atom_index = 0;
       atom_index < atom_count;
       ++atom_index) {
    distances[atom_index] = std::sqrt(
        squared_distance(x, y, z, atoms[atom_index]));
    unnormalized[atom_index] = 1.0;
  }

  // Equal-size Becke switching is deliberately used in this first prototype.
  // An O-H atomic-size correction can be added after this path is validated.
  for (std::size_t atom_a = 0; atom_a < atom_count; ++atom_a) {
    for (std::size_t atom_b = 0; atom_b < atom_count; ++atom_b) {
      if (atom_a == atom_b) continue;
      const double dx = atoms[atom_a].x - atoms[atom_b].x;
      const double dy = atoms[atom_a].y - atoms[atom_b].y;
      const double dz = atoms[atom_a].z - atoms[atom_b].z;
      const double nuclear_distance =
          std::sqrt(dx * dx + dy * dy + dz * dz);
      double mu =
          (distances[atom_a] - distances[atom_b]) / nuclear_distance;
      mu = std::max(-1.0, std::min(1.0, mu));
      // Three Becke iterations give a smooth transition with flat endpoints.
      for (int iteration = 0; iteration < 3; ++iteration) {
        mu = becke_polynomial(mu);
      }
      unnormalized[atom_a] *= 0.5 * (1.0 - mu);
    }
  }

  double denominator = 0.0;
  for (double value : unnormalized) denominator += value;
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    // This is only a roundoff safeguard; positive Becke products should
    // normally make the denominator strictly positive.
    return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  }
  for (double& value : unnormalized) value /= denominator;
  return unnormalized;
}

inline double shifted_source_value(
    double x,
    double y,
    double z,
    std::size_t atom_index)
{
  if (atom_index >= atom_count) return 0.0;
  const Atom& atom = atoms[atom_index];
  const double dx = x - atom.x;
  const double dy = y - atom.y;
  const double dz = z - atom.z;
  const double solution = std::exp(
      -treecode_test_problem::gaussian_a * dx * dx
      -treecode_test_problem::gaussian_b * dy * dy
      -treecode_test_problem::gaussian_c * dz * dz);
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
  double source = 0.0;
  for (std::size_t atom_index = 0; atom_index < atom_count; ++atom_index) {
    source += shifted_source_value(x, y, z, atom_index);
  }
  return source;
}

inline double shifted_exact_value(
    double x,
    double y,
    double z,
    std::size_t atom_index)
{
  if (atom_index >= atom_count) return 0.0;
  const Atom& atom = atoms[atom_index];
  const double dx = x - atom.x;
  const double dy = y - atom.y;
  const double dz = z - atom.z;
  return std::exp(
      -treecode_test_problem::gaussian_a * dx * dx
      -treecode_test_problem::gaussian_b * dy * dy
      -treecode_test_problem::gaussian_c * dz * dz);
}

inline double gaussian_coulomb_potential(
    double x,
    double y,
    double z,
    const Atom& atom)
{
  const double radius =
      std::sqrt(squared_distance(x, y, z, atom));
  if (radius < 1.0e-12) {
    return 2.0 * atom.electron_count
        * std::sqrt(atom.gaussian_exponent / pi);
  }
  return atom.electron_count
      * std::erf(std::sqrt(atom.gaussian_exponent) * radius)
      / radius;
}

inline double exact_value(double x, double y, double z)
{
  double potential = 0.0;
  for (std::size_t atom_index = 0; atom_index < atom_count; ++atom_index) {
    potential += shifted_exact_value(x, y, z, atom_index);
  }
  return potential;
}

// One radial refinement field is defined for each atomic center.  Keeping
// these fields separate makes it possible to inspect or tune the O, H1, and
// H2 neighborhoods independently.  The softening removes the singular value
// at a nucleus while retaining a monotonically decreasing radial profile.
inline double atomic_radial_refinement_indicator(
    double x,
    double y,
    double z,
    std::size_t atom_index)
{
  const double radius_squared =
      squared_distance(x, y, z, atoms[atom_index]);
  return 1.0 / std::sqrt(std::max(radius_squared, 1.0e-30));
}

inline std::array<double, atom_count> radial_refinement_indicators(
    double x,
    double y,
    double z)
{
  return {{
      atomic_radial_refinement_indicator(x, y, z, 0), // O
      atomic_radial_refinement_indicator(x, y, z, 1), // H1
      atomic_radial_refinement_indicator(x, y, z, 2)  // H2
  }};
}

inline double oxygen_refinement_indicator(double x, double y, double z)
{
  return atomic_radial_refinement_indicator(x, y, z, 0);
}

inline double hydrogen1_refinement_indicator(double x, double y, double z)
{
  return atomic_radial_refinement_indicator(x, y, z, 1);
}

inline double hydrogen2_refinement_indicator(double x, double y, double z)
{
  return atomic_radial_refinement_indicator(x, y, z, 2);
}

inline double refinement_indicator_value(double x, double y, double z)
{
  const std::array<double, atom_count> indicators =
      radial_refinement_indicators(x, y, z);

  // AFEPack accepts one scalar indicator.  The maximum forms the union of
  // the three atom-centered refinement regions without double-counting the
  // overlap between them.
  return *std::max_element(indicators.begin(), indicators.end());
}

inline const char* description()
{
  return
      "three-center manufactured Gaussian test with shifted copies of "
      "the compare_exact solution and source";
}

} // namespace treecode_h2o_fuzzy_problem

#endif
