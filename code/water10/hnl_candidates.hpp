#ifndef TREECODE_HNL_CANDIDATES_HPP
#define TREECODE_HNL_CANDIDATES_HPP
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace treecode_hnl {
struct Topology {
  std::vector<int> parents, levels;
  std::vector<std::vector<int>> children;
  std::vector<std::array<int, 4>> vertex_stars;
  std::vector<int> star_vertices;
  std::vector<std::vector<int>> stars;
  // Original node_neighbors[vertex][level], using stable IDs.
  std::vector<std::vector<int>> vertex_level_stars;
  std::vector<std::size_t> leaf_begin, leaf_end;
  std::vector<int> roots, root_of_node;
  std::vector<int> sorted_roots, root_slot_of_node;
  std::size_t vertex_count = 0;
  // For preorder IDs, same-level sibling families occupy disjoint ascending
  // ID ranges. Validate this layout once; generic ID layouts use the fallback.
  bool ordered_child_families = false;

  void prepare_child_family_order() {
    ordered_child_families = false;
    if (children.size() != parents.size() || levels.size() != parents.size()) return;
    int maximum_level = 0;
    std::size_t nonroot_count = 0;
    for (std::size_t node = 0; node < parents.size(); ++node) {
      if (levels[node] < 0 || static_cast<std::size_t>(levels[node]) > parents.size()) return;
      maximum_level = std::max(maximum_level, levels[node]);
      if (parents[node] >= 0) ++nonroot_count;
    }
    std::vector<int> previous_child(static_cast<std::size_t>(maximum_level) + 1, -1);
    std::size_t child_count = 0;
    for (std::size_t parent = 0; parent < children.size(); ++parent) {
      if (children[parent].empty()) continue;
      if (levels[parent] >= maximum_level) return;
      const int child_level = levels[parent] + 1;
      for (int child : children[parent]) {
        if (child < 0 || static_cast<std::size_t>(child) >= parents.size()
            || parents[child] != static_cast<int>(parent)
            || levels[child] != child_level
            || child <= previous_child[child_level]) return;
        previous_child[child_level] = child;
        ++child_count;
      }
    }
    // Parent consistency and strict ordering already exclude duplicate child
    // IDs. This count also proves every non-root seed belongs to its family.
    ordered_child_families = child_count == nonroot_count;
  }

  // Once per topology, not per target. Root flags then need only roots.size()
  // bytes per worker, instead of another node-sized membership array.
  void prepare_root_slots() {
    sorted_roots = roots;
    std::sort(sorted_roots.begin(), sorted_roots.end());
    root_slot_of_node.assign(parents.size(), -1);
    for (std::size_t slot = 0; slot < sorted_roots.size(); ++slot)
      root_slot_of_node[sorted_roots[slot]] = static_cast<int>(slot);
    for (std::size_t node = 0; node < parents.size(); ++node)
      root_slot_of_node[node] = root_slot_of_node[root_of_node[node]];
    prepare_child_family_order();
  }
};
struct LayerCounters {
  std::size_t vertex_queries = 0, star_entries = 0;
  std::size_t sibling_seed_visits = 0, sibling_groups = 0, sibling_entries = 0;
};

// Set-for-set port of treecode.cpp:125, commit 7038a6c.
// Pointers become stable integer IDs; identical sibling families are scanned
// once per call. Output sets retain the original constructor's semantics.
// Retains BOTH flag branches and the CUMULATIVE vertices_set.
// The independent regression test executes the verbatim original function.
inline void construct_neighbor_layers(
    const std::set<int>& initial_vertices,
    std::set<int>& outer_vertices_set,
    std::set<int>& vertices_set,
    const std::set<int>& initial_layer,
    std::set<int>& outer_layer,
    int initial_level,
    std::set<int>& neighbor_of_the_elements,
    const Topology& topology,
    int flag,
    LayerCounters* counters = nullptr) {
  outer_layer.clear();
  outer_vertices_set.clear();
  neighbor_of_the_elements.clear();
  if (flag != 0 && flag != 1) return;
  for (int vertex : initial_vertices) {
    if (vertex >= 0 && static_cast<std::size_t>(vertex) < topology.vertex_count) {
      const auto& by_level = topology.vertex_level_stars[vertex];
      if (initial_level >= 0 && static_cast<std::size_t>(initial_level) < by_level.size()) {
        const int star = by_level[initial_level];
        if (star >= 0) {
          const auto& incident = topology.stars[star];
          neighbor_of_the_elements.insert(incident.begin(), incident.end());
          if (counters) {
            ++counters->vertex_queries;
            counters->star_entries += incident.size();
          }
        }
      }
      vertices_set.insert(vertex);
    }
  }
  const auto erase_previous = [&](std::set<int>& elements) {
    for (int element : initial_layer) {
      if (flag == 1) elements.erase(element);
      else for (int child : topology.children[element]) elements.erase(child);
    }
  };
  erase_previous(neighbor_of_the_elements);
  std::set<int> elements_to_add;
  for (int element : neighbor_of_the_elements) {
    const int parent = topology.parents[element];
    if (counters) ++counters->sibling_seed_visits;
    if (parent >= 0 && outer_layer.insert(parent).second) {
      // Every seed keeps its place in the output. Complete its family once;
      // this is set-equivalent to the original repeated sibling loop.
      for (int child : topology.children[parent]) elements_to_add.insert(child);
      if (counters) {
        ++counters->sibling_groups;
        counters->sibling_entries += topology.children[parent].size();
      }
    }
  }
  erase_previous(elements_to_add);
  neighbor_of_the_elements.insert(elements_to_add.begin(), elements_to_add.end());
  for (int element : outer_layer)
    for (int star : topology.vertex_stars[element])
      outer_vertices_set.insert(topology.star_vertices[star]);
  for (int vertex : vertices_set) outer_vertices_set.erase(vertex);
}

struct Workspace : LayerCounters {
  std::vector<int> initial_vertices, outer_vertices_set, vertices_set;
  std::vector<int> initial_layer, outer_layer, neighbor_of_the_elements;
  // Worker-private byte membership. Only touched entries are reset: never
  // clear a node/vertex-sized array per target or per layer. No epoch wrap.
  std::vector<std::uint8_t> node_state, vertex_visited, frontier_seen, root_seen;
  std::vector<int> touched_nodes;
  std::vector<int> candidates;
  std::vector<std::size_t> layer_offsets;
  std::vector<int> layer_levels;
  std::size_t layers = 0;
  explicit Workspace(const Topology& t)
      : node_state(t.parents.size(), 0), vertex_visited(t.vertex_count, 0),
        frontier_seen(t.vertex_count, 0), root_seen(t.roots.size(), 0) {
    initial_vertices.reserve(64); outer_vertices_set.reserve(64);
    vertices_set.reserve(256); initial_layer.reserve(64); outer_layer.reserve(64);
    neighbor_of_the_elements.reserve(256); touched_nodes.reserve(512);
    candidates.reserve(1024); layer_offsets.reserve(16); layer_levels.reserve(16);
  }
  void clear_layer_marks() {
    for (int node : touched_nodes) node_state[node] = 0;
    touched_nodes.clear();
  }
  bool mark_node(int node, std::uint8_t bit) {
    auto& state = node_state[node];
    if (state & bit) return false;
    if (state == 0) touched_nodes.push_back(node);
    state |= bit;
    return true;
  }
};

// Group leaf targets by a common ancestor. With ancestor_steps=1, active
// siblings share one geometric HNL construction, while the later MAC remains
// target-specific. A root leaf is its own anchor. The target positions are
// caller IDs (for example ElementCache indices), not hierarchy IDs.
struct SharedTargetGroups {
  std::vector<int> anchors;
  std::vector<std::vector<std::size_t>> target_positions;
  std::vector<int> group_of_target;
};

inline SharedTargetGroups group_targets_by_ancestor(
    const Topology& topology, const std::vector<int>& target_nodes,
    int ancestor_steps = 1) {
  SharedTargetGroups result;
  result.group_of_target.assign(target_nodes.size(), -1);
  if (ancestor_steps < 0) return result;
  std::vector<int> group_of_anchor(topology.parents.size(), -1);
  for (std::size_t position = 0; position < target_nodes.size(); ++position) {
    const int target = target_nodes[position];
    if (target < 0 || static_cast<std::size_t>(target) >= topology.parents.size())
      continue;
    int anchor = target;
    for (int step = 0; step < ancestor_steps && topology.parents[anchor] >= 0;
         ++step)
      anchor = topology.parents[anchor];
    int& group = group_of_anchor[anchor];
    if (group < 0) {
      group = static_cast<int>(result.anchors.size());
      result.anchors.push_back(anchor);
      result.target_positions.emplace_back();
    }
    result.group_of_target[position] = group;
    result.target_positions[static_cast<std::size_t>(group)].push_back(position);
  }
  return result;
}

// Allocation-free steady-state port of the SAME constructor above. The
// original set overload remains an independent reference for tests. Roles
// have distinct bits, so exclusion, neighbor membership and parent membership
// never interfere, including in the flag=0 branch.
inline void construct_neighbor_layers(const Topology& t, int level,
                                     Workspace& w, int flag = 1) {
  constexpr std::uint8_t excluded = 1, present = 2, parent_seen = 4;
  w.clear_layer_marks();
  w.outer_layer.clear(); w.outer_vertices_set.clear();
  w.neighbor_of_the_elements.clear();
  if (flag != 0 && flag != 1) return;
  for (int node : w.initial_layer) {
    if (flag == 1) w.mark_node(node, excluded);
    else for (int child : t.children[node]) w.mark_node(child, excluded);
  }
  for (int vertex : w.initial_vertices) {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= t.vertex_count) continue;
    if (!w.vertex_visited[vertex]) {
      w.vertex_visited[vertex] = 1;
      w.vertices_set.push_back(vertex);
    }
    const auto& by_level = t.vertex_level_stars[vertex];
    if (level < 0 || static_cast<std::size_t>(level) >= by_level.size()) continue;
    const int star = by_level[level];
    if (star < 0) continue;
    ++w.vertex_queries;
    w.star_entries += t.stars[star].size();
    for (int node : t.stars[star])
      if (!(w.node_state[node] & excluded) && w.mark_node(node, present))
        w.neighbor_of_the_elements.push_back(node);
  }
  // Snapshot the seed count: siblings appended below must NOT become seeds.
  const std::size_t seed_count = w.neighbor_of_the_elements.size();
  w.sibling_seed_visits += seed_count;
  bool ordered_completion = t.ordered_child_families;
  for (std::size_t i = 0; i < seed_count; ++i) {
    const int parent = t.parents[w.neighbor_of_the_elements[i]];
    if (parent < 0) { ordered_completion = false; continue; }
    if (!w.mark_node(parent, parent_seen)) continue;
    w.outer_layer.push_back(parent);
    ++w.sibling_groups;
    w.sibling_entries += t.children[parent].size();
    if (t.levels[parent] != level - 1) ordered_completion = false;
  }
  std::sort(w.outer_layer.begin(), w.outer_layer.end());
  if (ordered_completion) {
    // This is the identical sibling-completed set, enumerated directly in
    // sorted ID order. Distinct validated families cannot share children;
    // every original seed is already among these children. No coverage or
    // cumulative-region reconstruction is involved, and exclusions are exact.
    w.neighbor_of_the_elements.clear();
    for (int parent : w.outer_layer)
      for (int child : t.children[parent])
        if (!(w.node_state[child] & excluded))
          w.neighbor_of_the_elements.push_back(child);
  } else {
    // Root seeds and arbitrary stable-ID layouts retain the generic union.
    for (int parent : w.outer_layer)
      for (int child : t.children[parent])
        if (!(w.node_state[child] & excluded) && w.mark_node(child, present))
          w.neighbor_of_the_elements.push_back(child);
    std::sort(w.neighbor_of_the_elements.begin(), w.neighbor_of_the_elements.end());
  }
  for (int parent : w.outer_layer)
    for (int star : t.vertex_stars[parent]) {
      const int vertex = t.star_vertices[star];
      if (!w.vertex_visited[vertex] && !w.frontier_seen[vertex]) {
        w.frontier_seen[vertex] = 1;
        w.outer_vertices_set.push_back(vertex);
      }
    }
  for (int vertex : w.outer_vertices_set) w.frontier_seen[vertex] = 0;
  // Match std::set iteration exactly, including the MAC accumulation order.
  std::sort(w.outer_vertices_set.begin(), w.outer_vertices_set.end());
}

// One geometric expansion per refinement level. Only the preceding outer
// layer/frontier is forwarded; never rebuild a cumulative covered region.
template <class OnCandidate>
inline const std::vector<int>& build_candidates(
    const Topology& t, int target, Workspace& w, OnCandidate on_candidate) {
  static_cast<LayerCounters&>(w) = LayerCounters{};
  w.initial_vertices.clear();
  for (int star : t.vertex_stars[target])
    w.initial_vertices.push_back(t.star_vertices[star]);
  std::sort(w.initial_vertices.begin(), w.initial_vertices.end());
  w.initial_vertices.erase(std::unique(w.initial_vertices.begin(),
                                      w.initial_vertices.end()), w.initial_vertices.end());
  w.initial_layer = {target};
  for (int vertex : w.vertices_set) w.vertex_visited[vertex] = 0;
  w.vertices_set.clear();
  w.candidates.clear();
  w.layer_offsets = {0}; w.layer_levels.clear(); w.layers = 0;
  std::fill(w.root_seen.begin(), w.root_seen.end(), 0);
  const auto emit = [&](int node) {
    w.candidates.push_back(node);
    on_candidate(node);
    // Root identity is prepared once with topology, not searched per target.
    w.root_seen[t.root_slot_of_node[node]] = 1;
  };
  emit(target);
  int level = t.levels[target];
  for (;;) {
    construct_neighbor_layers(t, level, w, 1);
    for (int node : w.neighbor_of_the_elements) emit(node);
    w.layer_levels.push_back(level);
    w.layer_offsets.push_back(w.candidates.size());
    ++w.layers;
    if (level == 0) break;
    // Consume exactly the constructor's outputs at the next parent level.
    w.initial_vertices.swap(w.outer_vertices_set);
    w.initial_layer.swap(w.outer_layer);
    --level;
  }
  // Coarsest remainder, including disconnected roots. No descendant search,
  // overlap filter, interval splitting, or post-construction repair.
  for (std::size_t slot = 0; slot < t.sorted_roots.size(); ++slot) {
    if (w.root_seen[slot]) continue;
    w.candidates.push_back(t.sorted_roots[slot]);
    on_candidate(t.sorted_roots[slot]);
  }
  w.layer_levels.push_back(0);
  w.layer_offsets.push_back(w.candidates.size());
  ++w.layers;
  return w.candidates;
}
inline const std::vector<int>& build_candidates(
    const Topology& t, int target, Workspace& w) {
  return build_candidates(t, target, w, [](int) {});
}
} // namespace treecode_hnl
#endif
