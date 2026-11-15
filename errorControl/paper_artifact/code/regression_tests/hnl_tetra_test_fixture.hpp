#ifndef TREECODE_HNL_TETRA_TEST_FIXTURE_HPP
#define TREECODE_HNL_TETRA_TEST_FIXTURE_HPP

#include "hnl_candidates.hpp"
#include <cassert>
#include <cstdlib>
#include <map>
#include <set>

// Genuine midpoint red refinement, not a repeated-vertex graph surrogate.
// Every tetrahedron has four distinct vertices, nonzero volume, and eight
// children each occupying exactly one eighth of its volume.
struct TetraTestMesh {
  using Point = std::array<int,3>;
  treecode_hnl::Topology topology;
  std::vector<Point> points;
  std::vector<std::array<int,4>> vertices;
  std::vector<long long> determinants; // Six times geometric volume.
  std::map<Point,int> vertex_ids;
  std::map<std::pair<int,int>,int> star_ids;
  int leaves = 0;

  int vertex(Point point) {
    const auto inserted = vertex_ids.emplace(point,static_cast<int>(points.size()));
    if (inserted.second) points.push_back(point);
    return inserted.first->second;
  }
  int midpoint(int a, int b) {
    Point point{};
    for (int k = 0; k < 3; ++k) {
      assert((points[a][k] + points[b][k]) % 2 == 0);
      point[k] = (points[a][k] + points[b][k]) / 2;
    }
    return vertex(point);
  }
  long long determinant(const std::array<int,4>& v) const {
    std::array<long long,3> a{}, b{}, c{};
    for (int k = 0; k < 3; ++k) {
      a[k] = points[v[1]][k] - points[v[0]][k];
      b[k] = points[v[2]][k] - points[v[0]][k];
      c[k] = points[v[3]][k] - points[v[0]][k];
    }
    return std::llabs(a[0]*(b[1]*c[2]-b[2]*c[1])
                    -a[1]*(b[0]*c[2]-b[2]*c[0])
                    +a[2]*(b[0]*c[1]-b[1]*c[0]));
  }
  int add(int parent, int level, std::array<int,4> v, int uniform_depth,
          unsigned refinement_mask, int child_index) {
    auto& t = topology;
    const int node = static_cast<int>(t.parents.size());
    t.parents.push_back(parent);
    t.root_of_node.push_back(parent < 0 ? node : t.root_of_node[parent]);
    t.levels.push_back(level);
    t.children.emplace_back(); t.vertex_stars.emplace_back();
    t.leaf_begin.push_back(leaves); t.leaf_end.push_back(leaves);
    vertices.push_back(v);
    assert(std::set<int>(v.begin(),v.end()).size() == 4);
    const long long volume = determinant(v);
    assert(volume > 0);
    determinants.push_back(volume);
    for (int k = 0; k < 4; ++k) {
      const int vertex_id = v[k];
      t.vertex_count = std::max(t.vertex_count,std::size_t(vertex_id+1));
      t.vertex_level_stars.resize(t.vertex_count);
      auto& by_level = t.vertex_level_stars[vertex_id];
      if (by_level.size() <= static_cast<std::size_t>(level))
        by_level.resize(level+1,-1);
      const auto inserted = star_ids.emplace(
          std::make_pair(level,vertex_id),static_cast<int>(t.stars.size()));
      const int star = inserted.first->second;
      if (inserted.second) {
        t.stars.emplace_back(); t.star_vertices.push_back(vertex_id);
      }
      t.vertex_stars[node][k] = star;
      t.stars[star].push_back(node); by_level[level] = star;
    }
    const bool selective = level == 1 && (refinement_mask & (1u << child_index));
    if (level < uniform_depth || selective) {
      const int a = v[0], b = v[1], c = v[2], d = v[3];
      const int ab = midpoint(a,b), ac = midpoint(a,c), ad = midpoint(a,d);
      const int bc = midpoint(b,c), bd = midpoint(b,d), cd = midpoint(c,d);
      // Four corners plus the central octahedron split along ab--cd.
      // The face refinement is identical on both sides of a shared face.
      const std::array<std::array<int,4>,8> children{{
        {{a,ab,ac,ad}}, {{b,ab,bc,bd}}, {{c,ac,bc,cd}}, {{d,ad,bd,cd}},
        {{ab,ac,ad,cd}}, {{ab,ac,bc,cd}}, {{ab,ad,bd,cd}}, {{ab,bc,bd,cd}}
      }};
      long long child_volume = 0;
      for (int child = 0; child < 8; ++child) {
        const int next = add(node,level+1,children[child],uniform_depth,
                             refinement_mask,child);
        t.children[node].push_back(next);
        child_volume += determinants[next];
        assert(determinants[next]*8 == volume);
      }
      assert(child_volume == volume);
    } else ++leaves;
    t.leaf_end[node] = leaves;
    return node;
  }

  // Root modes: single; face-touching; disconnected; face-touching with the
  // second root uniformly level one (asymmetric refinement across the face).
  TetraTestMesh(int uniform_depth, unsigned refinement_mask, int root_mode) {
    auto& t = topology;
    int a = vertex({{0,0,0}}), b = vertex({{128,0,0}});
    int c = vertex({{0,128,0}}), d = vertex({{0,0,128}});
    t.roots.push_back(add(-1,0,{{a,b,c,d}},uniform_depth,refinement_mask,-1));
    if (root_mode == 1 || root_mode == 3) {
      const int e = vertex({{0,0,-128}});
      t.roots.push_back(add(-1,0,{{a,b,c,e}},root_mode == 3 ? 1 : uniform_depth,
                           root_mode == 3 ? 0 : refinement_mask,-1));
    } else if (root_mode == 2) {
      a = vertex({{256,0,0}}); b = vertex({{384,0,0}});
      c = vertex({{256,128,0}}); d = vertex({{256,0,128}});
      t.roots.push_back(add(-1,0,{{a,b,c,d}},uniform_depth,refinement_mask,-1));
    }
    t.prepare_root_slots();
  }
};
#endif
