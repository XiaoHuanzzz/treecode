#ifndef TREECODE_MAC_GEOMETRY_HPP
#define TREECODE_MAC_GEOMETRY_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace treecode_mac {

// Only the fields used by MAC classification. Checked 32-bit integer metadata
// keeps each record to one 64-byte cache-line's worth of storage. IDs used by
// the classifier are already signed int; no geometric value is compressed.
struct Node {
  std::array<double, 3> center{};
  double radius = 0.0;
  double volume = 0.0;
  std::uint32_t leaf_begin = 0;
  std::uint32_t leaf_end = 0;
  std::uint32_t leaf_count = 0;
  std::uint32_t child_begin = 0;
  int leaf_index = -1;
  std::uint32_t child_count = 0;
};

static_assert(sizeof(Node) == 64, "MAC geometry must remain a 64-byte record");

namespace detail {
inline std::uint32_t packed_index(std::size_t value, const char* error) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error(error);
  return static_cast<std::uint32_t>(value);
}

// Guard before addition: this also prevents size_t overflow on a 32-bit host.
inline std::size_t packed_child_total(std::size_t current, std::size_t added) {
  const std::size_t maximum = std::numeric_limits<std::uint32_t>::max();
  if (current > maximum || added > maximum - current)
    throw std::length_error("MAC geometry child storage exceeds uint32_t");
  return current + added;
}
} // namespace detail

struct Geometry {
  std::vector<Node> nodes;
  std::vector<int> children;

  Geometry() = default;

  // One-time immutable geometry copy and child packing before the parallel
  // target loop. Preserve existing IDs, center values, and child order exactly;
  // this does not traverse, filter, repair, or otherwise change the hierarchy.
  template<class Element, class Cache>
  Geometry(const std::vector<Element*>& elements,
           const std::vector<Cache>& cache) {
    if (elements.size() != cache.size())
      throw std::invalid_argument("MAC geometry element/cache size mismatch");
    if (elements.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::length_error("MAC geometry node count exceeds signed int IDs");
    std::size_t total_children = 0;
    for (std::size_t i = 0; i < elements.size(); ++i) {
      if (elements[i] == nullptr)
        throw std::invalid_argument("MAC geometry contains a null element");
      const std::size_t count = cache[i].child_indices.size();
      total_children = detail::packed_child_total(total_children, count);
    }
    nodes.resize(elements.size());
    children.reserve(total_children);
    for (std::size_t i = 0; i < elements.size(); ++i) {
      const auto& source = cache[i];
      Node& node = nodes[i];
      for (int coordinate = 0; coordinate < 3; ++coordinate)
        node.center[coordinate] = source.ref_point[coordinate];
      node.radius = source.radius;
      node.volume = source.volume;
      node.leaf_begin = detail::packed_index(source.subtree_leaf_begin,
          "MAC geometry leaf begin exceeds uint32_t");
      node.leaf_end = detail::packed_index(source.subtree_leaf_end,
          "MAC geometry leaf end exceeds uint32_t");
      node.leaf_count = detail::packed_index(source.subtree_leaf_count,
          "MAC geometry leaf count exceeds uint32_t");
      // The total child storage was checked before any allocation/copy.
      node.child_begin = static_cast<std::uint32_t>(children.size());
      node.child_count = static_cast<std::uint32_t>(source.child_indices.size());
      node.leaf_index = elements[i]->value == 0 ? elements[i]->index : -1;
      children.insert(children.end(), source.child_indices.begin(),
                      source.child_indices.end());
    }
  }
};

// Same coordinate subtraction, left-to-right accumulation, and final sqrt as
// AFEPack Point<3>::length(). Build without reassociation/fast-math flags.
inline double center_distance(const Node& source, const Node& target) {
  double squared = 0.0;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    const double delta = source.center[coordinate] - target.center[coordinate];
    squared += delta * delta;
  }
  return std::sqrt(squared);
}

}  // namespace treecode_mac
#endif
