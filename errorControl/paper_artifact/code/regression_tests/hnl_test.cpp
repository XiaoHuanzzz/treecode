#include "hnl_candidates.hpp"
#include "hnl_tetra_test_fixture.hpp"
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

namespace original {
constexpr int DIM = 3;
template<int D, int Space> struct HGeometry {
  int index = -1;
  std::vector<HGeometry<0, Space>*> vertex;
};
template<int Space> struct HElement {
  int id = -1;
  HElement* parent = nullptr;
  std::vector<HElement*> child;
  HGeometry<Space, Space>* h_element = nullptr;
};
#include "hnl_original_constructor_reference.hpp"
}

// Run the verbatim historical constructor on pointer views of the SAME valid
// tetrahedral hierarchy, independently of its integer-ID port.
struct OriginalMesh {
  std::vector<original::HGeometry<0,3>> vertices;
  std::vector<original::HGeometry<3,3>> geometries;
  std::vector<original::HElement<3>> elements;
  std::vector<std::vector<std::set<original::HElement<3>*>>> neighbors;
  explicit OriginalMesh(const treecode_hnl::Topology& t)
      : vertices(t.vertex_count), geometries(t.parents.size()),
        elements(t.parents.size()), neighbors(t.vertex_count) {
    for (std::size_t v = 0; v < vertices.size(); ++v) {
      vertices[v].index = static_cast<int>(v);
      neighbors[v].resize(t.vertex_level_stars[v].size());
    }
    for (std::size_t i = 0; i < elements.size(); ++i) {
      elements[i].id = static_cast<int>(i);
      elements[i].h_element = &geometries[i];
      if (t.parents[i] >= 0) elements[i].parent = &elements[t.parents[i]];
      for (int child : t.children[i]) elements[i].child.push_back(&elements[child]);
      for (int star : t.vertex_stars[i]) {
        const int v = t.star_vertices[star];
        geometries[i].vertex.push_back(&vertices[v]);
        neighbors[v][t.levels[i]].insert(&elements[i]);
      }
    }
  }
};
using ESet = std::set<original::HElement<3>*>;
using VSet = std::set<original::HGeometry<0,3>*>;
static std::set<int> ids(const ESet& input) {
  std::set<int> result;
  for (auto* element : input) result.insert(element->id);
  return result;
}
static std::set<int> ids(const VSet& input) {
  std::set<int> result;
  for (auto* vertex : input) result.insert(vertex->index);
  return result;
}
template<class Container>
static std::set<int> unique_ids(const Container& input) {
  const std::set<int> result(input.begin(),input.end());
  // Normalizing vector-backed frontiers must not hide duplicate entries.
  assert(result.size() == input.size());
  return result;
}

// Set-for-set comparison of BOTH flag branches, including cumulative vertices.
static void compare_call(const treecode_hnl::Topology& t, OriginalMesh& reference,
    VSet& initial_v, VSet& outer_v, VSet& visited_v,
    ESet& initial_e, ESet& outer_e, int level, ESet& neighbors, int flag) {
  auto vi = ids(initial_v), vo = ids(outer_v), vs = ids(visited_v);
  auto ei = ids(initial_e), eo = ids(outer_e), ne = ids(neighbors);
  treecode_hnl::construct_neighbor_layers(vi,vo,vs,ei,eo,level,ne,t,flag);
  original::construct_neighbor_layers(initial_v,outer_v,visited_v,
      initial_e,outer_e,level,neighbors,reference.neighbors,flag);
  assert(vi == ids(initial_v) && ei == ids(initial_e));
  assert(vo == ids(outer_v) && vs == ids(visited_v));
  assert(eo == ids(outer_e) && ne == ids(neighbors));
}

static void check_frontier_replay(const treecode_hnl::Topology& t,
    OriginalMesh& reference, int target, const treecode_hnl::Workspace& w,
    std::size_t& flag_one_calls, std::size_t& flag_zero_calls) {
  VSet initial_v, outer_v, visited_v;
  ESet initial_e{&reference.elements[target]}, outer_e, neighbors;
  for (auto* vertex : reference.geometries[target].vertex) initial_v.insert(vertex);
  std::set<int> remaining_roots(t.roots.begin(),t.roots.end());
  remaining_roots.erase(t.root_of_node[target]);
  for (int level = t.levels[target]; level >= 0; --level) {
    const std::size_t layer = static_cast<std::size_t>(t.levels[target]-level);
    for (auto* element : initial_e) assert(t.levels[element->id] == level);
    compare_call(t,reference,initial_v,outer_v,visited_v,
                 initial_e,outer_e,level,neighbors,1);
    ++flag_one_calls;
    std::vector<int> expected;
    if (layer == 0) expected.push_back(target);
    const auto neighbor_ids = ids(neighbors);
    expected.insert(expected.end(),neighbor_ids.begin(),neighbor_ids.end());
    const std::vector<int> actual(w.candidates.begin()+w.layer_offsets[layer],
                                 w.candidates.begin()+w.layer_offsets[layer+1]);
    assert(actual == expected);
    for (int node : neighbor_ids) remaining_roots.erase(t.root_of_node[node]);
    if (level > 0) {
      // EXACT preceding outputs, never a reconstructed accumulated region.
      initial_v = outer_v;
      initial_e = outer_e;
    }
  }
  assert(ids(initial_v) == unique_ids(w.initial_vertices));
  assert(ids(initial_e) == unique_ids(w.initial_layer));
  assert(ids(outer_v) == unique_ids(w.outer_vertices_set));
  assert(ids(outer_e) == unique_ids(w.outer_layer));
  assert(ids(visited_v) == unique_ids(w.vertices_set));
  assert(ids(neighbors) == unique_ids(w.neighbor_of_the_elements));
  const std::vector<int> expected_roots(remaining_roots.begin(),remaining_roots.end());
  const std::vector<int> actual_roots(w.candidates.begin()+w.layer_offsets[w.layers-1],
                                    w.candidates.end());
  assert(actual_roots == expected_roots);

  // The caller uses flag=1; retain a separate flag=0 regression with parent-
  // level input and prepopulated visited/output sets, including root targets.
  const int parent = t.parents[target];
  initial_e = {&reference.elements[parent < 0 ? target : parent]};
  initial_v.clear(); visited_v.clear();
  for (auto* element : initial_e)
    for (auto* vertex : element->h_element->vertex) initial_v.insert(vertex);
  if (!initial_v.empty()) visited_v.insert(*initial_v.begin());
  compare_call(t,reference,initial_v,outer_v,visited_v,
               initial_e,outer_e,t.levels[target],neighbors,0);
  ++flag_zero_calls;
}

static void check_partition(const TetraTestMesh& mesh, int target,
                            const treecode_hnl::Workspace& w) {
  const auto& t = mesh.topology;
  assert(w.layers == static_cast<std::size_t>(t.levels[target]+2));
  assert(w.layers == w.layer_levels.size());
  assert(w.layer_offsets.size() == w.layers+1);
  assert(w.layer_offsets.front() == 0 && w.layer_offsets.back() == w.candidates.size());
  assert(!w.candidates.empty() && w.candidates.front() == target);
  assert(std::count(w.candidates.begin(),w.candidates.end(),target) == 1);
  std::vector<int> visits(mesh.leaves,0);
  std::set<int> all_nodes;
  long long volume = 0;
  for (std::size_t layer = 0; layer < w.layers; ++layer) {
    assert(w.layer_offsets[layer] <= w.layer_offsets[layer+1]);
    const int level = layer+1 == w.layers ? 0 : t.levels[target]-static_cast<int>(layer);
    assert(w.layer_levels[layer] == level);
    for (auto i = w.layer_offsets[layer]; i < w.layer_offsets[layer+1]; ++i) {
      const int node = w.candidates[i];
      assert(node >= 0 && static_cast<std::size_t>(node) < t.parents.size());
      assert(t.levels[node] == level && all_nodes.insert(node).second);
      volume += mesh.determinants[node];
      for (auto leaf = t.leaf_begin[node]; leaf < t.leaf_end[node]; ++leaf) {
        // Test-only intervals expose same-layer AND cross-layer overlaps.
        assert(visits[leaf] == 0);
        ++visits[leaf];
      }
    }
  }
  for (int count : visits) assert(count == 1);
  long long domain_volume = 0;
  for (int root : t.roots) domain_volume += mesh.determinants[root];
  assert(volume == domain_volume);
}

static void check_same_workspace_result(const treecode_hnl::Workspace& actual,
                                       const treecode_hnl::Workspace& expected) {
  assert(actual.candidates == expected.candidates);
  assert(actual.layer_offsets == expected.layer_offsets);
  assert(actual.layer_levels == expected.layer_levels);
  assert(actual.layers == expected.layers);
  assert(unique_ids(actual.initial_vertices) == unique_ids(expected.initial_vertices));
  assert(unique_ids(actual.initial_layer) == unique_ids(expected.initial_layer));
  assert(unique_ids(actual.outer_vertices_set) == unique_ids(expected.outer_vertices_set));
  assert(unique_ids(actual.outer_layer) == unique_ids(expected.outer_layer));
  assert(unique_ids(actual.vertices_set) == unique_ids(expected.vertices_set));
  assert(unique_ids(actual.neighbor_of_the_elements) ==
         unique_ids(expected.neighbor_of_the_elements));
  assert(actual.vertex_queries == expected.vertex_queries);
  assert(actual.star_entries == expected.star_entries);
  assert(actual.sibling_seed_visits == expected.sibling_seed_visits);
  assert(actual.sibling_groups == expected.sibling_groups);
  assert(actual.sibling_entries == expected.sibling_entries);
}

static void compare_vector_call(const treecode_hnl::Topology& t,
    OriginalMesh& reference, treecode_hnl::Workspace& w,
    VSet& initial_v, VSet& outer_v, VSet& visited_v,
    ESet& initial_e, ESet& outer_e, int level, ESet& neighbors, int flag) {
  const auto assign_ids = [](std::vector<int>& output, const auto& input) {
    const auto converted = ids(input);
    output.assign(converted.begin(),converted.end());
  };
  assign_ids(w.initial_vertices,initial_v);
  assign_ids(w.outer_vertices_set,outer_v);
  assign_ids(w.initial_layer,initial_e);
  assign_ids(w.outer_layer,outer_e);
  assign_ids(w.neighbor_of_the_elements,neighbors);
  // Restore a consistent cumulative history, without clearing the layer
  // scratch marks left by the preceding call: the constructor owns those.
  for (int vertex : w.vertices_set) w.vertex_visited[vertex] = 0;
  assign_ids(w.vertices_set,visited_v);
  for (int vertex : w.vertices_set) w.vertex_visited[vertex] = 1;
  treecode_hnl::construct_neighbor_layers(t,level,w,flag);
  compare_call(t,reference,initial_v,outer_v,visited_v,
               initial_e,outer_e,level,neighbors,flag);
  assert(unique_ids(w.initial_vertices) == ids(initial_v));
  assert(unique_ids(w.initial_layer) == ids(initial_e));
  assert(unique_ids(w.outer_vertices_set) == ids(outer_v));
  assert(unique_ids(w.outer_layer) == ids(outer_e));
  assert(unique_ids(w.vertices_set) == ids(visited_v));
  assert(unique_ids(w.neighbor_of_the_elements) == ids(neighbors));
  assert(std::is_sorted(w.outer_vertices_set.begin(),w.outer_vertices_set.end()));
  assert(std::is_sorted(w.outer_layer.begin(),w.outer_layer.end()));
  assert(std::is_sorted(w.neighbor_of_the_elements.begin(),w.neighbor_of_the_elements.end()));
  const auto visited = ids(visited_v);
  for (std::size_t vertex = 0; vertex < t.vertex_count; ++vertex) {
    assert(bool(w.vertex_visited[vertex]) == bool(visited.count(static_cast<int>(vertex))));
    assert(w.frontier_seen[vertex] == 0);
  }
  if (flag != 0 && flag != 1) {
    assert(w.touched_nodes.empty());
    assert(std::all_of(w.node_state.begin(),w.node_state.end(),
                       [](std::uint8_t state) { return state == 0; }));
  }
}

static std::size_t check_workspace_reuse() {
  std::size_t calls = 0, flag_one_calls = 0, flag_zero_calls = 0;
  for (int root_mode = 0; root_mode < 4; ++root_mode) {
    const TetraTestMesh mesh(2,0,root_mode);
    const auto& t = mesh.topology;
    auto solver_topology = t;
    solver_topology.leaf_begin.clear(); solver_topology.leaf_end.clear();
    OriginalMesh reference(t);
    treecode_hnl::Workspace reused(solver_topology);
    std::vector<int> targets = t.roots;
    for (int target = 0; target < static_cast<int>(t.parents.size()); ++target)
      if (t.children[target].empty()) targets.push_back(target);
    const auto check_target = [&](int target, treecode_hnl::Workspace& workspace) {
      treecode_hnl::Workspace fresh(solver_topology);
      treecode_hnl::build_candidates(solver_topology,target,fresh);
      std::vector<int> emitted;
      treecode_hnl::build_candidates(solver_topology,target,workspace,
          [&](int node) { emitted.push_back(node); });
      assert(workspace.candidates == emitted);
      check_same_workspace_result(workspace,fresh);
      check_partition(mesh,target,workspace);
      check_frontier_replay(t,reference,target,workspace,flag_one_calls,flag_zero_calls);
      ++calls;
    };
    // Repeated targets catch stale marks, while alternating low/high IDs and
    // reversing traversal exercises both directions across disconnected roots.
    for (int pass = 0; pass < 4; ++pass) {
      for (std::size_t i = 0; i < targets.size(); ++i) {
        const std::size_t index = i % 2 == 0 ? i/2 : targets.size()-1-i/2;
        check_target(targets[index],reused);
        if (i % 11 == 0) check_target(targets[index],reused);
      }
      std::reverse(targets.begin(),targets.end());
    }
  }
  return calls;
}

static std::size_t check_vector_flag_branches() {
  std::size_t calls = 0;
  for (int root_mode = 0; root_mode < 4; ++root_mode) {
    const TetraTestMesh mesh(2,0,root_mode);
    const auto& t = mesh.topology;
    OriginalMesh reference(t);
    treecode_hnl::Workspace w(t);
    for (int target = 0; target < static_cast<int>(t.parents.size()); ++target) {
      VSet initial_v, outer_v{&reference.vertices.front()},
          visited_v{&reference.vertices.back()};
      ESet initial_e, outer_e{&reference.elements.back()},
          neighbors{&reference.elements.front()};
      // Reuse the same byte-mark scratch storage across both valid branches,
      // invalid flags, empty outputs, and cumulative visited-vertex histories.
      for (int flag : {1,0,-1,2,1,0}) {
        const int parent = t.parents[target];
        initial_e = {&reference.elements[flag == 0 && parent >= 0 ? parent : target]};
        initial_v.clear();
        for (auto* element : initial_e)
          for (auto* vertex : element->h_element->vertex) initial_v.insert(vertex);
        compare_vector_call(t,reference,w,initial_v,outer_v,visited_v,
                            initial_e,outer_e,t.levels[target],neighbors,flag);
        ++calls;
      }
    }
  }
  return calls;
}

static void relabel_nodes(TetraTestMesh& mesh, const std::vector<int>& new_id) {
  const auto old = mesh.topology;
  const auto old_vertices = mesh.vertices;
  const auto old_determinants = mesh.determinants;
  auto& t = mesh.topology;
  assert(unique_ids(new_id).size() == old.parents.size());
  for (std::size_t node = 0; node < old.parents.size(); ++node) {
    const int mapped = new_id[node];
    t.parents[mapped] = old.parents[node] < 0 ? -1 : new_id[old.parents[node]];
    t.levels[mapped] = old.levels[node];
    t.children[mapped].clear();
    for (int child : old.children[node]) t.children[mapped].push_back(new_id[child]);
    t.vertex_stars[mapped] = old.vertex_stars[node];
    t.root_of_node[mapped] = new_id[old.root_of_node[node]];
    t.leaf_begin[mapped] = old.leaf_begin[node];
    t.leaf_end[mapped] = old.leaf_end[node];
    mesh.vertices[mapped] = old_vertices[node];
    mesh.determinants[mapped] = old_determinants[node];
  }
  for (std::size_t root = 0; root < old.roots.size(); ++root)
    t.roots[root] = new_id[old.roots[root]];
  for (std::size_t star = 0; star < old.stars.size(); ++star)
    for (std::size_t entry = 0; entry < old.stars[star].size(); ++entry)
      t.stars[star][entry] = new_id[old.stars[star][entry]];
  t.prepare_root_slots();
}

static std::size_t check_ordered_family_paths() {
  std::size_t calls = 0, flag_one_calls = 0, flag_zero_calls = 0;
  for (int root_mode = 0; root_mode < 4; ++root_mode) {
    const TetraTestMesh original_mesh(2,0,root_mode);
    const auto& ordered = original_mesh.topology;
    assert(ordered.ordered_child_families);
    auto forced_fallback = ordered;
    forced_fallback.ordered_child_families = false;
    treecode_hnl::Workspace fast(ordered), generic(forced_fallback);
    for (int target = 0; target < static_cast<int>(ordered.parents.size()); ++target) {
      if (!ordered.children[target].empty() && ordered.parents[target] >= 0) continue;
      treecode_hnl::build_candidates(ordered,target,fast);
      treecode_hnl::build_candidates(forced_fallback,target,generic);
      check_same_workspace_result(fast,generic);
      ++calls;
    }
    for (int permutation = 0; permutation < 3; ++permutation) {
      auto mesh = original_mesh;
      auto& t = mesh.topology;
      if (permutation < 2) {
        std::vector<int> new_id(t.parents.size());
        std::iota(new_id.begin(),new_id.end(),0);
        if (permutation == 0) std::reverse(new_id.begin(),new_id.end());
        else {
          std::mt19937 random(1789+root_mode);
          std::shuffle(new_id.begin(),new_id.end(),random);
        }
        relabel_nodes(mesh,new_id);
      } else {
        // IDs alone are insufficient: unsorted child arrays must also disable
        // direct family emission, even though the represented mesh is unchanged.
        for (auto& children : t.children) std::reverse(children.begin(),children.end());
        std::reverse(t.roots.begin(),t.roots.end());
        t.prepare_root_slots();
      }
      assert(!t.ordered_child_families);
      auto solver_topology = t;
      solver_topology.leaf_begin.clear(); solver_topology.leaf_end.clear();
      OriginalMesh reference(t);
      treecode_hnl::Workspace reused(solver_topology), direct(solver_topology);
      for (int target = 0; target < static_cast<int>(t.parents.size()); ++target) {
        if (!t.children[target].empty() && t.parents[target] >= 0) continue;
        treecode_hnl::build_candidates(solver_topology,target,reused);
        check_partition(mesh,target,reused);
        check_frontier_replay(t,reference,target,reused,flag_one_calls,flag_zero_calls);
        VSet initial_v, outer_v, visited_v{&reference.vertices.back()};
        ESet initial_e, outer_e, neighbors;
        for (int flag : {1,0,-1,1}) {
          const int parent = t.parents[target];
          initial_e = {&reference.elements[flag == 0 && parent >= 0 ? parent : target]};
          initial_v.clear();
          for (auto* element : initial_e)
            for (auto* vertex : element->h_element->vertex) initial_v.insert(vertex);
          compare_vector_call(t,reference,direct,initial_v,outer_v,visited_v,
                              initial_e,outer_e,t.levels[target],neighbors,flag);
        }
        ++calls;
      }
    }
  }
  return calls;
}

static std::size_t check_shared_target_groups() {
  std::size_t groups_checked = 0;
  for (int root_mode = 0; root_mode < 4; ++root_mode) {
    for (int test = 0; test < 32; ++test) {
      const TetraTestMesh mesh(1, static_cast<unsigned>(test), root_mode);
      const auto& t = mesh.topology;
      std::vector<int> leaves;
      for (int node = 0; node < static_cast<int>(t.parents.size()); ++node)
        if (t.children[node].empty()) leaves.push_back(node);
      const auto grouped = treecode_hnl::group_targets_by_ancestor(t, leaves, 1);
      assert(grouped.group_of_target.size() == leaves.size());
      std::vector<int> visits(leaves.size(), 0);
      treecode_hnl::Workspace workspace(t);
      for (std::size_t group = 0; group < grouped.anchors.size(); ++group) {
        const int anchor = grouped.anchors[group];
        treecode_hnl::build_candidates(t, anchor, workspace);
        check_partition(mesh, anchor, workspace);
        for (std::size_t position : grouped.target_positions[group]) {
          assert(position < leaves.size());
          assert(grouped.group_of_target[position] == static_cast<int>(group));
          const int leaf = leaves[position];
          assert(anchor == leaf || t.parents[leaf] == anchor);
          ++visits[position];
        }
        ++groups_checked;
      }
      assert(std::all_of(visits.begin(), visits.end(),
                         [](int count) { return count == 1; }));
    }
  }
  return groups_checked;
}

int main() {
  std::size_t partitions = 0, flag_one_calls = 0, flag_zero_calls = 0, root_targets = 0;
  const auto test_mesh = [&](int depth, unsigned refinement_mask, int root_mode) {
    const TetraTestMesh mesh(depth,refinement_mask,root_mode);
    const auto& t = mesh.topology;
    auto solver_topology = t;
    // No interval metadata is available to production construction. Coverage
    // is checked afterward; this test cannot repair emitted candidates.
    solver_topology.leaf_begin.clear(); solver_topology.leaf_end.clear();
    OriginalMesh reference(t);
    treecode_hnl::Workspace w(solver_topology);
    for (int target = 0; target < static_cast<int>(t.parents.size()); ++target) {
      if (!t.children[target].empty()) continue;
      int root = target;
      while (t.parents[root] >= 0) root = t.parents[root];
      assert(t.root_of_node[target] == root);
      std::vector<int> emitted;
      const auto& candidates = treecode_hnl::build_candidates(solver_topology,target,w,
          [&](int node) { emitted.push_back(node); });
      assert(candidates == emitted);
      check_partition(mesh,target,w);
      check_frontier_replay(t,reference,target,w,flag_one_calls,flag_zero_calls);
      if (t.levels[target] == 0) ++root_targets;
      ++partitions;
    }
  };
  for (int root_mode = 0; root_mode < 4; ++root_mode) {
    for (int test = 0; test < 267; ++test) {
      // Across the asymmetric face, levels may differ by at most one.
      if (root_mode == 3 && test == 2) continue;
      const int depth = test < 3 ? test+1 : 1;
      const unsigned mask = test < 3 ? 0u : test < 11 ? 1u << (test-3)
                                                                : test-11;
      test_mesh(depth,mask,root_mode);
    }
  }
  assert(partitions == 61136);
  for (int root_mode = 0; root_mode < 4; ++root_mode) test_mesh(0,0,root_mode);
  assert(partitions == 61150);
  assert(root_targets > 0);
  const std::size_t reuse_calls = check_workspace_reuse();
  const std::size_t vector_calls = check_vector_flag_branches();
  const std::size_t family_calls = check_ordered_family_paths();
  const std::size_t shared_groups = check_shared_target_groups();
  std::cout << "HNL tests passed: " << partitions
            << " genuine red-refined tetrahedral source covers partition every leaf "
            << "exactly once without interval metadata, including " << root_targets
            << " root targets.\n"
            << "Direct frontier handoff and one-level decrement match the verbatim "
            << "pointer constructor: " << flag_one_calls << " flag=1 calls; "
            << flag_zero_calls << " independent flag=0 calls also match.\n"
            << "Reusable workspace agrees with fresh construction, the original "
            << "constructor, and exact coverage across " << reuse_calls
            << " additional repeated/root/alternating-target calls.\n"
            << "Vector constructor matches both set-based oracles across "
            << vector_calls << " flag=0/1/invalid calls with reused byte marks.\n"
            << "Ordered-family fast/fallback paths and reversed/shuffled IDs or child order "
            << "match exact covers and original constructors across " << family_calls
            << " additional targets.\n"
            << "Sibling-target grouping assigns every leaf once and constructs "
            << shared_groups << " parent-anchored full-domain covers.\n";
}
