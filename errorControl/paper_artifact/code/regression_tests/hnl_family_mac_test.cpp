#include "hnl_family_mac.hpp"
#include "hnl_tetra_test_fixture.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <set>
#include <vector>

#ifdef NDEBUG
#error "HNL family packing tests require enabled assertions"
#endif

namespace {
std::size_t cover_checks = 0, family_checks = 0, fallback_checks = 0;

void check_cover(const treecode_hnl::Topology& topology,
                 const std::vector<int>& cover, std::size_t leaf_count) {
  std::vector<unsigned char> visits(leaf_count, 0);
  for (int node : cover) {
    assert(node >= 0 && static_cast<std::size_t>(node) < topology.parents.size());
    for (std::size_t leaf = topology.leaf_begin[node];
         leaf < topology.leaf_end[node]; ++leaf) {
      assert(leaf < leaf_count && visits[leaf] == 0);
      visits[leaf] = 1;
    }
  }
  assert(std::all_of(visits.begin(), visits.end(),
                     [](unsigned char value) { return value == 1; }));
  ++cover_checks;
}

// Independent original set-constructor replay; no family helper is used.
std::vector<int> reference_cover(const treecode_hnl::Topology& topology,
                                 int target) {
  std::set<int> initial_vertices, outer_vertices, visited_vertices;
  std::set<int> initial_layer{target}, outer_layer, neighbors;
  for (int star : topology.vertex_stars[target])
    initial_vertices.insert(topology.star_vertices[star]);
  std::vector<int> result{target};
  std::set<int> remaining_roots(topology.roots.begin(), topology.roots.end());
  remaining_roots.erase(topology.root_of_node[target]);
  for (int level = topology.levels[target]; level >= 0; --level) {
    treecode_hnl::construct_neighbor_layers(initial_vertices, outer_vertices,
        visited_vertices, initial_layer, outer_layer, level, neighbors,
        topology, 1);
    for (int node : neighbors) {
      result.push_back(node);
      remaining_roots.erase(topology.root_of_node[node]);
    }
    initial_vertices = outer_vertices;
    initial_layer = outer_layer;
  }
  result.insert(result.end(), remaining_roots.begin(), remaining_roots.end());
  return result;
}

std::vector<int> rejected_parent_order(const treecode_hnl::Topology& topology,
                                      std::vector<int> stack) {
  std::vector<int> leaves;
  while (!stack.empty()) {
    const int node = stack.back();
    stack.pop_back();
    if (topology.children[node].empty()) leaves.push_back(node);
    else for (int child : topology.children[node]) stack.push_back(child);
  }
  return leaves;
}

void check_local_replacements(const treecode_hnl::Topology& topology,
    const treecode_hnl::Workspace& original,
    const treecode_hnl::FamilyPackingWorkspace& packed) {
  std::size_t position = 0, families = 0, replaced = 0;
  assert(!packed.candidates.empty());
  assert(packed.candidates.front() == original.candidates.front());
  for (int node : packed.candidates) {
    assert(position < original.candidates.size());
    if (node == original.candidates[position]) { ++position; continue; }
    const auto& children = topology.children[node];
    assert(children.size() > 1);
    assert(position + children.size() <= original.candidates.size());
    assert(std::equal(children.begin(), children.end(),
                      original.candidates.begin() + position));
    const auto layer_end = std::upper_bound(original.layer_offsets.begin(),
                                            original.layer_offsets.end(), position);
    assert(layer_end != original.layer_offsets.end());
    assert(position + children.size() <= *layer_end);
    position += children.size();
    ++families;
    replaced += children.size();
  }
  assert(position == original.candidates.size());
  assert(families == packed.complete_families);
  assert(replaced == packed.replaced_children);
  assert(replaced - families == packed.saved_candidates);
  assert(packed.saved_candidates == original.candidates.size() - packed.candidates.size());
  family_checks += families;
}

void check_mesh(const TetraTestMesh& mesh) {
  const auto& topology = mesh.topology;
  assert(topology.ordered_child_families);
  treecode_hnl::Workspace original(topology);
  treecode_hnl::FamilyPackingWorkspace packed;
  for (std::size_t node = 0; node < topology.parents.size(); ++node) {
    if (!topology.children[node].empty()) continue;
    const auto& cover = treecode_hnl::build_candidates(topology, static_cast<int>(node), original);
    const auto original_copy = cover;
    const auto offsets_copy = original.layer_offsets;
    assert(cover == reference_cover(topology, static_cast<int>(node)));
    const auto& result = treecode_hnl::pack_complete_sibling_families(topology, original, packed);
    assert(&result == &packed.candidates && packed.used_ordered_path);
    assert(original.candidates == original_copy && original.layer_offsets == offsets_copy);
    check_cover(topology, original.candidates, mesh.leaves);
    check_cover(topology, result, mesh.leaves);
    check_local_replacements(topology, original, packed);
    assert(rejected_parent_order(topology, original.candidates)
        == rejected_parent_order(topology, result));
  }
}

void check_boundaries() {
  const TetraTestMesh mesh(2, 0, 2);
  const auto& topology = mesh.topology;
  const int target = topology.roots[0];
  const int root = topology.roots[1];
  const auto& children = topology.children[root];
  treecode_hnl::FamilyPackingWorkspace packed;
  std::vector<int> input{target};
  input.insert(input.end(), children.begin(), children.end());
  const auto& full = treecode_hnl::pack_complete_sibling_families(
      topology, input, {0, 1, input.size()}, packed);
  assert((full == std::vector<int>{target, root}));
  assert(packed.complete_families == 1 && packed.replaced_children == children.size());
  check_cover(topology, full, mesh.leaves);

  // Remove each possible child: partial families must remain byte-for-byte.
  for (std::size_t missing = 0; missing < children.size(); ++missing) {
    auto partial = input;
    partial.erase(partial.begin() + 1 + missing);
    const auto& result = treecode_hnl::pack_complete_sibling_families(
        topology, partial, {0, 1, partial.size()}, packed);
    assert(result == partial && packed.complete_families == 0);
  }
  // Even a complete union across two layers must not merge.
  const auto& split = treecode_hnl::pack_complete_sibling_families(
      topology, input, {0, 1, 5, input.size()}, packed);
  assert(split == input && packed.complete_families == 0);

  // Exactly ONE merge level: grandchildren -> children, never -> root.
  std::vector<int> grandchildren{target};
  for (int child : children)
    grandchildren.insert(grandchildren.end(), topology.children[child].begin(),
                         topology.children[child].end());
  const auto& once = treecode_hnl::pack_complete_sibling_families(
      topology, grandchildren, {0, 1, grandchildren.size()}, packed);
  assert(once == input && packed.complete_families == children.size());
  check_cover(topology, once, mesh.leaves);

  // The separately emitted target is never absorbed into its family.
  std::vector<int> target_family = children;
  const auto& untouched_target = treecode_hnl::pack_complete_sibling_families(
      topology, target_family, {0, target_family.size()}, packed);
  assert(untouched_target == target_family && packed.complete_families == 0);

  bool invalid_offsets_rejected = false;
  try { treecode_hnl::pack_complete_sibling_families(topology, input, {0, input.size()-1}, packed); }
  catch (const std::invalid_argument&) { invalid_offsets_rejected = true; }
  assert(invalid_offsets_rejected);
}

treecode_hnl::Topology reverse_ids(const treecode_hnl::Topology& original) {
  auto result = original;
  const int count = static_cast<int>(original.parents.size());
  const auto remap = [count](int node) { return node < 0 ? node : count - 1 - node; };
  for (int old = 0; old < count; ++old) {
    const int node = remap(old);
    result.parents[node] = remap(original.parents[old]);
    result.levels[node] = original.levels[old];
    result.root_of_node[node] = remap(original.root_of_node[old]);
    result.vertex_stars[node] = original.vertex_stars[old];
    result.leaf_begin[node] = original.leaf_begin[old];
    result.leaf_end[node] = original.leaf_end[old];
    result.children[node].clear();
    for (int child : original.children[old]) result.children[node].push_back(remap(child));
  }
  for (std::size_t i = 0; i < result.stars.size(); ++i) {
    result.stars[i].clear();
    for (int node : original.stars[i]) result.stars[i].push_back(remap(node));
    std::sort(result.stars[i].begin(), result.stars[i].end());
  }
  for (auto& root : result.roots) root = remap(root);
  result.prepare_root_slots();
  assert(!result.ordered_child_families);
  return result;
}

void check_shuffled_fallback() {
  const TetraTestMesh mesh(2, 0, 1);
  const auto topology = reverse_ids(mesh.topology);
  treecode_hnl::Workspace original(topology);
  treecode_hnl::FamilyPackingWorkspace packed;
  for (std::size_t node = 0; node < topology.parents.size(); ++node) {
    if (!topology.children[node].empty()) continue;
    treecode_hnl::build_candidates(topology, static_cast<int>(node), original);
    packed.candidates = {-1};
    packed.complete_families = packed.replaced_children = packed.saved_candidates = 999;
    const auto& result = treecode_hnl::pack_complete_sibling_families(topology, original, packed);
    assert(&result == &original.candidates && !packed.used_ordered_path);
    assert(packed.candidates.empty() && packed.complete_families == 0);
    assert(packed.replaced_children == 0 && packed.saved_candidates == 0);
    assert(result == reference_cover(topology, static_cast<int>(node)));
    check_cover(topology, result, mesh.leaves);
    ++fallback_checks;
  }
}
} // namespace

int main() {
  for (int roots = 0; roots != 4; ++roots) {
    for (int depth = 0; depth <= 3; ++depth)
      check_mesh(TetraTestMesh(depth, 0, roots));
    for (unsigned mask : {0x01u, 0x03u, 0x55u, 0xaau, 0xffu})
      check_mesh(TetraTestMesh(1, mask, roots));
  }
  check_boundaries();
  check_shuffled_fallback();
  assert(family_checks > 0);
  std::cout << "Experimental HNL family packing tests passed: " << cover_checks
            << " genuine tetrahedral cover checks; " << family_checks
            << " complete-family witnesses; " << fallback_checks
            << " shuffled-ID fallbacks.\n"
            << "Original constructor/candidate order unchanged; partial and split-layer "
               "families retained; one-level-only replacement and rejected-parent "
               "traversal order passed.\n";
}
