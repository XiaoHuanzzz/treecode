#include "mac_geometry.hpp"
#include <AFEPack/Geometry.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef NDEBUG
#error "MAC geometry parity tests require enabled assertions"
#endif

namespace {
using Coordinates = std::array<double,3>;
constexpr std::array<double,5> thetas{{0.1,0.3,0.5,0.7,0.9}};
std::size_t distance_checks = 0, decision_checks = 0;

std::uint64_t bits(double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t), "Requires 64-bit double");
  static_assert(std::numeric_limits<double>::is_iec559, "Requires IEEE 754 double");
  std::uint64_t result;
  std::memcpy(&result,&value,sizeof(result));
  return result;
}

// Calls the installed AFEPack implementation, not a test reimplementation of
// its arithmetic. Link this executable with the same AFEPack library as the
// solver and compile with the same floating-point flags (no fast-math).
double original_distance(const Coordinates& source, const Coordinates& target) {
  const Point<3> source_point(source.data()), target_point(target.data());
  return (source_point-target_point).length();
}

bool original_mac(const treecode_mac::Node& source,
                  const treecode_mac::Node& target, double theta) {
  const double distance = original_distance(source.center,target.center);
  const double minimum_target_distance = distance-target.radius;
  const bool contains_target = source.leaf_begin <= target.leaf_begin
      && target.leaf_begin < source.leaf_end;
  return !contains_target && minimum_target_distance > 0.0
      && source.radius/minimum_target_distance <= theta;
}

bool compact_mac(const treecode_mac::Node& source,
                 const treecode_mac::Node& target, double theta) {
  const double distance = treecode_mac::center_distance(source,target);
  const double minimum_target_distance = distance-target.radius;
  const bool contains_target = source.leaf_begin <= target.leaf_begin
      && target.leaf_begin < source.leaf_end;
  return !contains_target && minimum_target_distance > 0.0
      && source.radius/minimum_target_distance <= theta;
}

bool check_decision(const treecode_mac::Node& source,
                    const treecode_mac::Node& target, double theta) {
  const bool expected = original_mac(source,target,theta);
  assert(compact_mac(source,target,theta) == expected);
  ++decision_checks;
  return expected;
}

void check_pair(const Coordinates& source_center, const Coordinates& target_center,
                bool expect_infinity = false) {
  treecode_mac::Node source, target;
  source.center = source_center; target.center = target_center;
  const double expected = original_distance(source_center,target_center);
  const double actual = treecode_mac::center_distance(source,target);
  assert(bits(actual) == bits(expected));
  assert(expect_infinity ? std::isinf(expected) : std::isfinite(expected));
  ++distance_checks;
  if (!(expected > 0.0) || !std::isfinite(expected)) return;

  source.leaf_begin = 10; source.leaf_end = 20;
  target.leaf_begin = 20; target.leaf_end = 21; // Half-open source boundary.
  // Include both substantial clearance and cancellation leaving one distance
  // ulp. Target radii and resulting clearances must remain strictly positive.
  for (double target_radius : {0.25*expected,std::nextafter(expected,0.0)}) {
    target.radius = target_radius;
    const double clearance = expected-target_radius;
    assert(target.radius > 0.0 && clearance > 0.0);
    for (double theta : thetas) {
      const double boundary_radius = theta*clearance;
      assert(boundary_radius > 0.0 && std::isfinite(boundary_radius));
      for (double source_radius : {std::nextafter(boundary_radius,0.0),
                                   boundary_radius,
                                   std::nextafter(boundary_radius,
                                       std::numeric_limits<double>::infinity())}) {
        source.radius = source_radius;
        check_decision(source,target,theta);
        target.leaf_begin = source.leaf_begin;
        assert(!check_decision(source,target,theta)); // Containment overrides MAC.
        target.leaf_begin = source.leaf_end;
      }
    }
  }
  // The positivity guard, including exactly tangent bounding spheres.
  source.radius = 0.0;
  for (double target_radius : {expected,
          std::nextafter(expected,std::numeric_limits<double>::infinity())}) {
    target.radius = target_radius;
    for (double theta : thetas) assert(!check_decision(source,target,theta));
  }
}

void check_distance_and_mac() {
  const double tiny = std::numeric_limits<double>::denorm_min();
  check_pair({{0.0,0.0,0.0}},{{-0.0,0.0,-0.0}});
  check_pair({{3.0,4.0,0.0}},{{0.0,0.0,0.0}});
  check_pair({{1.0,-2.0,3.0}},{{1.0,-2.0,3.0}});
  check_pair({{tiny,-tiny,tiny}},{{0.0,0.0,0.0}}); // Squared underflow to zero.
  check_pair({{1e-161,-2e-161,3e-161}},{{0.0,0.0,0.0}});
  check_pair({{1e-150,-2e-150,3e-150}},{{-1e-150,2e-150,-3e-150}});
  check_pair({{1e150,-2e150,3e150}},{{-1e150,2e150,-3e150}});
  const double small_normal = std::sqrt(std::numeric_limits<double>::min());
  check_pair({{small_normal,small_normal,-small_normal}},{{0.0,0.0,0.0}});
  const double large_safe = std::sqrt(std::numeric_limits<double>::max())/4.0;
  check_pair({{large_safe,-large_safe,large_safe}},
             {{-large_safe,large_safe,-large_safe}});
  const double large = std::ldexp(1.0,500);
  check_pair({{std::nextafter(large,0.0),large,-large}},
             {{large,std::nextafter(large,0.0),-large}});
  // Permuting unequal squared terms detects accidental reassociation.
  Coordinates unequal{{1.0,std::ldexp(1.0,-26),std::ldexp(1.0,-27)}};
  std::sort(unequal.begin(),unequal.end());
  do check_pair(unequal,{{0.0,0.0,0.0}});
  while (std::next_permutation(unequal.begin(),unequal.end()));
  // Explicit infinity cases are separate from all finite-distance samples.
  check_pair({{1e200,0.0,0.0}},{{0.0,0.0,0.0}},true);
  check_pair({{std::numeric_limits<double>::max(),0.0,0.0}},
             {{-std::numeric_limits<double>::max(),0.0,0.0}},true);

  std::mt19937_64 random(0x92cd35ab71e0468fULL);
  const auto unit = [&]() { return double(random() >> 11)*0x1.0p-53; };
  for (int sample = 0; sample < 100000; ++sample) {
    Coordinates source{}, target{};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      source[coordinate] = 20.0*unit()-10.0;
      target[coordinate] = 20.0*unit()-10.0;
    }
    check_pair(source,target);
  }
  // Independent exponents create widely separated finite coordinate scales;
  // [-500,500] keeps squared differences safely below overflow.
  for (int sample = 0; sample < 8192; ++sample) {
    Coordinates source{}, target{};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      source[coordinate] = std::ldexp(2.0*unit()-1.0,int(random()%1001)-500);
      target[coordinate] = std::ldexp(2.0*unit()-1.0,int(random()%1001)-500);
    }
    check_pair(source,target);
  }

  // With clearance exactly one, division is exact: equality MUST be accepted,
  // while the very next representable source radius MUST be rejected.
  treecode_mac::Node source, target;
  source.center = {{2.0,0.0,0.0}};
  source.leaf_begin = 0; source.leaf_end = 1;
  target.leaf_begin = 1; target.leaf_end = 2; target.radius = 1.0;
  for (double theta : thetas) {
    source.radius = std::nextafter(theta,0.0);
    assert(check_decision(source,target,theta));
    source.radius = theta;
    assert(check_decision(source,target,theta));
    source.radius = std::nextafter(theta,std::numeric_limits<double>::infinity());
    assert(!check_decision(source,target,theta));
  }
}

struct Element { int value = 0, index = -1; };
struct Cache {
  Point<3> ref_point;
  double radius = 0.0, volume = 0.0;
  std::size_t subtree_leaf_begin = 0, subtree_leaf_end = 0, subtree_leaf_count = 0;
  std::vector<int> child_indices;
};

void check_geometry() {
  static_assert(sizeof(treecode_mac::Node) == 64, "Expected packed MAC record");
  std::vector<Element> storage{{1,41},{0,7},{2,13},{0,2},{0,8},{0,9}};
  std::vector<Element*> elements;
  for (auto& element : storage) elements.push_back(&element);
  std::vector<Cache> cache(elements.size());
  for (std::size_t i = 0; i < cache.size(); ++i) {
    cache[i].ref_point[0] = i == 0 ? -0.0 : double(i)*0.125;
    cache[i].ref_point[1] = -double(i)-0.25;
    cache[i].ref_point[2] = double(i)+0.75;
    cache[i].radius = double(i)+0.5;
    cache[i].volume = double(i)+1.5;
    cache[i].subtree_leaf_begin = i*11;
    cache[i].subtree_leaf_end = i*11+7;
    cache[i].subtree_leaf_count = i+1;
  }
  cache[0].child_indices = {4,2,1}; // Deliberately unsorted stable IDs.
  cache[2].child_indices = {5,3};
  const treecode_mac::Geometry geometry(elements,cache);
  assert(geometry.nodes.size() == cache.size());
  assert((geometry.children == std::vector<int>{4,2,1,5,3}));
  std::size_t offset = 0;
  for (std::size_t i = 0; i < cache.size(); ++i) {
    const auto& node = geometry.nodes[i];
    const auto& reference = cache[i];
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      assert(bits(node.center[coordinate]) == bits(reference.ref_point[coordinate]));
    assert(bits(node.radius) == bits(reference.radius));
    assert(bits(node.volume) == bits(reference.volume));
    assert(node.leaf_begin == reference.subtree_leaf_begin);
    assert(node.leaf_end == reference.subtree_leaf_end);
    assert(node.leaf_count == reference.subtree_leaf_count);
    assert(node.child_begin == offset);
    assert(node.child_count == reference.child_indices.size());
    assert(node.leaf_index == (elements[i]->value == 0 ? elements[i]->index : -1));
    for (std::size_t child = 0; child < node.child_count; ++child)
      assert(geometry.children[node.child_begin+child] == reference.child_indices[child]);
    offset += node.child_count;
  }
  // Preserve the classifier's stack/LIFO child traversal order exactly.
  std::vector<int> original_stack{0}, compact_stack{0}, original_order, compact_order;
  while (!original_stack.empty()) {
    const int node = original_stack.back(); original_stack.pop_back();
    original_order.push_back(node);
    for (int child : cache[node].child_indices) original_stack.push_back(child);
  }
  while (!compact_stack.empty()) {
    const int index = compact_stack.back(); compact_stack.pop_back();
    compact_order.push_back(index);
    const auto& node = geometry.nodes[index];
    for (std::size_t child = 0; child < node.child_count; ++child)
      compact_stack.push_back(geometry.children[node.child_begin+child]);
  }
  assert(original_order == compact_order);
  assert((compact_order == std::vector<int>{0,1,2,3,5,4}));

  cache[0].ref_point[0] = 123.0; cache[0].child_indices.clear();
  storage[1].index = 987;
  assert(bits(geometry.nodes[0].center[0]) == bits(-0.0));
  assert(geometry.nodes[1].leaf_index == 7);
  assert((geometry.children == std::vector<int>{4,2,1,5,3}));

  const treecode_mac::Geometry empty(std::vector<Element*>{},std::vector<Cache>{});
  assert(empty.nodes.empty() && empty.children.empty());
  bool mismatch_rejected = false, null_rejected = false;
  try { treecode_mac::Geometry invalid(elements,std::vector<Cache>{}); }
  catch (const std::invalid_argument&) { mismatch_rejected = true; }
  elements[0] = nullptr;
  try { treecode_mac::Geometry invalid(elements,cache); }
  catch (const std::invalid_argument&) { null_rejected = true; }
  assert(mismatch_rejected && null_rejected);
}

void check_packed_bounds() {
  const std::size_t maximum = std::numeric_limits<std::uint32_t>::max();
  assert(treecode_mac::detail::packed_index(0, "test") == 0);
  assert(treecode_mac::detail::packed_index(maximum - 1, "test") == maximum - 1);
  assert(treecode_mac::detail::packed_index(maximum, "test") == maximum);
  assert(treecode_mac::detail::packed_child_total(maximum, 0) == maximum);
  assert(treecode_mac::detail::packed_child_total(maximum - 7, 7) == maximum);
  const auto expect_length_error = [](auto operation) {
    bool rejected = false;
    try { operation(); }
    catch (const std::length_error&) { rejected = true; }
    assert(rejected);
  };
  expect_length_error([&] { treecode_mac::detail::packed_child_total(maximum, 1); });
  expect_length_error([&] { treecode_mac::detail::packed_child_total(maximum - 7, 8); });
  if (std::numeric_limits<std::size_t>::max() > maximum) {
    expect_length_error([&] { treecode_mac::detail::packed_index(maximum + 1, "test"); });
    expect_length_error([&] { treecode_mac::detail::packed_index(
        std::numeric_limits<std::size_t>::max(), "test"); });
    expect_length_error([&] { treecode_mac::detail::packed_child_total(maximum + 1, 0); });
    expect_length_error([&] { treecode_mac::detail::packed_child_total(0, maximum + 1); });
  }

  // Exercise constructor narrowing at the inclusive limit without allocating
  // a gigantic mesh. These synthetic metadata values need not form a tree.
  Element element{0, 0};
  std::vector<Element*> elements{&element};
  std::vector<Cache> cache(1);
  cache[0].subtree_leaf_begin = maximum;
  cache[0].subtree_leaf_end = maximum;
  cache[0].subtree_leaf_count = maximum;
  const treecode_mac::Geometry geometry(elements, cache);
  assert(geometry.nodes[0].leaf_begin == maximum);
  assert(geometry.nodes[0].leaf_end == maximum);
  assert(geometry.nodes[0].leaf_count == maximum);
  if (std::numeric_limits<std::size_t>::max() > maximum) {
    for (std::size_t Cache::*field : {&Cache::subtree_leaf_begin,
             &Cache::subtree_leaf_end, &Cache::subtree_leaf_count}) {
      cache[0].*field = maximum + 1;
      expect_length_error([&] { treecode_mac::Geometry invalid(elements, cache); });
      cache[0].*field = maximum;
    }
  }
}
} // namespace

int main() {
  check_geometry();
  check_packed_bounds();
  check_distance_and_mac();
  std::cout << "MAC geometry tests passed: " << distance_checks
            << " bitwise AFEPack distance comparisons; " << decision_checks
            << " identical MAC decisions, including threshold-adjacent radii.\n"
            << "64-byte records, metadata copies, packed child order, traversal order, "
               "and uint32 input/overflow guards passed.\n";
}
