#ifndef TREECODE_HNL_VISUALIZATION_HPP
#define TREECODE_HNL_VISUALIZATION_HPP
#include <sstream>

struct HnlVisualizationComplete {};

inline void export_hnl_visualization(
    const treecode_hnl::Topology& topology,
    const std::vector<HElement<DIM>*>& elements,
    const std::vector<hElementCache>& cache,
    const HierarchySummary& hierarchy) {
  const char* filename = std::getenv("TREECODE_HNL_JSON");
  if (!filename || !*filename)
    throw std::runtime_error("Set TREECODE_HNL_JSON to the export filename");
  std::array<double, 3> requested{{0., 0., 0.}};
  if (const char* setting = std::getenv("TREECODE_HNL_TARGET")) {
    std::istringstream in(setting);
    std::string extra;
    if (!(in >> requested[0] >> requested[1] >> requested[2]) || (in >> extra))
      throw std::runtime_error("TREECODE_HNL_TARGET requires three coordinates");
    for (double value : requested)
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite target");
  }
  int target = -1;
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < elements.size(); ++i) {
    if (elements[i]->value != 0) continue;
    double distance = 0.;
    for (int d = 0; d < 3; ++d) {
      const double delta = cache[i].ref_point[d] - requested[d];
      distance += delta * delta;
    }
    if (distance < best) { best = distance; target = static_cast<int>(i); }
  }
  if (target < 0) throw std::runtime_error("Mesh has no target leaves");
  treecode_hnl::Workspace workspace(topology);
  const auto& cover = treecode_hnl::build_candidates(topology, target, workspace);
  // Dedicated visualization audit only; the production constructor does not
  // sort, filter, or repair the candidate layers.
  if (!validate_hnl_cover(cover, cache, hierarchy.leaf_count))
    throw std::runtime_error("Exported HNL cover has gaps or overlaps");
  if (workspace.layer_levels.empty() ||
      workspace.layer_offsets.size() != workspace.layer_levels.size() + 1 ||
      workspace.layer_offsets.front() != 0 ||
      workspace.layer_offsets.back() != cover.size() ||
      !std::is_sorted(workspace.layer_offsets.begin(), workspace.layer_offsets.end()))
    throw std::runtime_error("Invalid HNL layer metadata");
  if (workspace.layer_offsets[1] == 0 || cover.front() != target)
    throw std::runtime_error("First HNL layer must include the target leaf");
  std::ofstream out(filename);
  if (!out) throw std::runtime_error("Cannot open HNL JSON output");
  out << std::setprecision(17);
  const auto point = [&](const auto& p) {
    out << '[' << p[0] << ',' << p[1] << ',' << p[2] << ']';
  };
  const auto tetra = [&](int index) {
    out << '[';
    for (int v = 0; v < 4; ++v) {
      if (v) out << ',';
      point(*elements[index]->h_element->vertex[v]);
    }
    out << ']';
  };
  std::array<double, 3> lo, hi;
  lo.fill(std::numeric_limits<double>::infinity());
  hi.fill(-std::numeric_limits<double>::infinity());
  for (int root : topology.roots)
    for (const auto* vertex : elements[root]->h_element->vertex)
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], (*vertex)[d]);
        hi[d] = std::max(hi[d], (*vertex)[d]);
      }
  out << "{\"schema\":3,\"stage\":\"production disjoint HNL layers before MAC\",\"leaf_count\":"
      << hierarchy.leaf_count << ",\"domain_volume\":" << hierarchy.domain_volume
      << ",\"coverage_audit\":true,\"target_hierarchy_id\":" << target
      << ",\"target_leaf_id\":" << elements[target]->index
      << ",\"requested_point\":";
  point(requested);
  out << ",\"target\":"; point(cache[target].ref_point);
  out << ",\"target_vertices\":"; tetra(target);
  out << ",\"bounds\":["; point(lo); out << ','; point(hi); out << ']';
  out << ",\"layers\":[";
  for (std::size_t layer = 0; layer < workspace.layer_levels.size(); ++layer) {
    if (layer) out << ',';
    out << "{\"index\":" << layer << ",\"lookup_level\":" << workspace.layer_levels[layer]
        << ",\"coarse_completion\":"
        << (layer + 1 == workspace.layer_levels.size() ? "true" : "false")
        << ",\"cells\":[";
    for (std::size_t j = workspace.layer_offsets[layer];
         j < workspace.layer_offsets[layer + 1]; ++j) {
      if (j != workspace.layer_offsets[layer]) out << ',';
      const int node = cover[j];
      out << "{\"id\":" << node << ",\"level\":" << cache[node].level
          << ",\"volume\":" << cache[node].volume
          << ",\"leaf_interval\":[" << cache[node].subtree_leaf_begin
          << ',' << cache[node].subtree_leaf_end << "],\"vertices\":";
      tetra(node); out << '}';
    }
    out << "]}";
  }
  // Export only native leaves intersecting the illustrated section as context.
  out << "],\"background_section_tetrahedra\":[";
  bool first = true;
  const double z = cache[target].ref_point[2];
  for (std::size_t i = 0; i < elements.size(); ++i) {
    if (elements[i]->value != 0) continue;
    double zmin = std::numeric_limits<double>::infinity(), zmax = -zmin;
    for (const auto* vertex : elements[i]->h_element->vertex) {
      zmin = std::min(zmin, (*vertex)[2]); zmax = std::max(zmax, (*vertex)[2]);
    }
    if (z < zmin || z > zmax) continue;
    if (!first) out << ',';
    first = false;
    tetra(static_cast<int>(i));
  }
  out << "]}\n";
  out.close();
  if (!out) throw std::runtime_error("Failed writing HNL JSON");
  std::cout << "Exported " << workspace.layer_levels.size()
            << " disjoint production HNL layers (including target self) for leaf "
            << elements[target]->index << " to " << filename << '\n';
}
#endif
