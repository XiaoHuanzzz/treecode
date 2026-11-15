#ifndef TREECODE_HNL_FAMILY_MAC_HPP
#define TREECODE_HNL_FAMILY_MAC_HPP

#include "hnl_candidates.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace treecode_hnl {

// OPTIONAL EXPERIMENT: unchanged geometric HNL cover -> immediate sibling
// family packing -> the ordinary MAC classifier. This is not an exact-cut
// optimization: accepting a packed parent replaces its children's multipole
// contributions, and can change the numerical error under the same MAC.
// A rejected parent descends in the original child order. No root traversal,
// recursive coarsening, cross-layer union or coverage repair occurs here.
struct FamilyPackingWorkspace {
  std::vector<int> candidates;
  std::size_t complete_families = 0;
  std::size_t replaced_children = 0;
  std::size_t saved_candidates = 0;
  bool used_ordered_path = false;
};

// Input is the original, valid output of build_candidates, including its
// separately emitted target at position zero and its original layer offsets.
// Keep that target untouched, even if its siblings occur immediately after it.
// The topology's once-prepared ordering witness proves child families have
// disjoint ascending ID ranges at each level. Matching the local sequence to
// the direct children witnesses completeness, including exclusion of partial
// families. Only these already-emitted candidates are inspected.
inline const std::vector<int>& pack_complete_sibling_families(
    const Topology& topology, const std::vector<int>& original,
    const std::vector<std::size_t>& layer_offsets,
    FamilyPackingWorkspace& workspace) {
  workspace.candidates.clear();
  workspace.complete_families = 0;
  workspace.replaced_children = 0;
  workspace.saved_candidates = 0;
  workspace.used_ordered_path = false;
  if (!topology.ordered_child_families) return original;
  if (layer_offsets.empty() || layer_offsets.front() != 0
      || layer_offsets.back() != original.size()
      || !std::is_sorted(layer_offsets.begin(), layer_offsets.end()))
    throw std::invalid_argument("Invalid original HNL layer offsets for family packing");
  workspace.used_ordered_path = true;
  workspace.candidates.reserve(original.size());
  for (std::size_t layer = 0; layer + 1 < layer_offsets.size(); ++layer) {
    std::size_t position = layer_offsets[layer];
    const std::size_t end = layer_offsets[layer + 1];
    if (position == 0 && end != 0) {
      workspace.candidates.push_back(original[position++]);
    }
    while (position < end) {
      const int node = original[position];
      const int parent = topology.parents[node];
      if (parent >= 0) {
        const auto& children = topology.children[parent];
        // Never consume nodes from a different emitted layer. In particular,
        // children excluded by the preceding layer cannot be silently merged.
        if (children.size() > 1 && children.size() <= end - position
            && children.front() == node
            && std::equal(children.begin(), children.end(),
                          original.begin() + position)) {
          workspace.candidates.push_back(parent);
          ++workspace.complete_families;
          workspace.replaced_children += children.size();
          workspace.saved_candidates += children.size() - 1;
          position += children.size();
          continue;
        }
      }
      workspace.candidates.push_back(node);
      ++position;
    }
  }
  return workspace.candidates;
}

inline const std::vector<int>& pack_complete_sibling_families(
    const Topology& topology, const Workspace& original,
    FamilyPackingWorkspace& workspace) {
  return pack_complete_sibling_families(topology, original.candidates,
                                       original.layer_offsets, workspace);
}

} // namespace treecode_hnl
#endif
