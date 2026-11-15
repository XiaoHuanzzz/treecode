#ifndef TREECODE_ROOT_BASED_NEIGHBORS_HPP
#define TREECODE_ROOT_BASED_NEIGHBORS_HPP

// Benchmark-only alternative to geometric HNL preprocessing. Included by
// compare_exact.cpp after the shared MAC classifier and coverage validator.
// A single walk starts from ALL roots for each target. It never restarts a
// root search for each interaction, and uses no HNL topology or layer work.
inline void build_root_based_interactions(
    const std::vector<HElement<DIM>*>& elements,
    const std::vector<int>& roots,
    const treecode_mac::Geometry& mac_geometry,
    std::vector<hElementCache>& cache,
    std::vector<InteractionListStats>& statistics,
    std::size_t leaf_count,
    double theta)
{
#ifdef TREECODE_HNL_VERIFY
  constexpr bool audit = true;
#else
  constexpr bool audit = false;
#endif
  std::vector<int> leaf_hierarchy;
  if (audit) {
    leaf_hierarchy.resize(leaf_count, -1);
    for (std::size_t i = 0; i < elements.size(); ++i)
      if (elements[i]->value == 0)
        leaf_hierarchy[elements[i]->index] = static_cast<int>(i);
  }
  const auto& geometry = cache;
  const auto policy = treecode_list::Policy::environment();
  treecode_list::ScheduleScope schedule(policy);
  std::vector<int> targets;
  targets.reserve(leaf_count);
  for (std::size_t i = 0; i < elements.size(); ++i)
    if (elements[i]->value == 0) targets.push_back(static_cast<int>(i));
  int actual_threads = 1;
  std::size_t allocated_bytes = 0, used_bytes = 0;
  #pragma omp parallel default(none) num_threads(policy.team_size()) \
      shared(roots, cache, geometry, statistics, leaf_hierarchy, mac_geometry, targets, actual_threads) \
      firstprivate(theta, audit, leaf_count, policy) \
      reduction(+:allocated_bytes,used_bytes)
  {
  #pragma omp single
  actual_threads = omp_get_num_threads();
  InteractionWorkspace mac_workspace;
  #pragma omp for schedule(runtime) nowait
  for (std::size_t ordinal = 0; ordinal < targets.size(); ++ordinal) {
    const int i = targets[ordinal];
    // Same O(1) MAC test, contiguous integer lists, and child traversal as
    // HNL's classification stage. Only the initial source cover differs.
    classify_source_cover_by_mac(static_cast<int>(i), roots,
                                 cache[i], mac_geometry, theta, statistics[i], mac_workspace, policy.pooled);
    if (audit) {
      std::vector<int> cover = cache[i].far_hierarchy_indices.copy();
      for (int leaf : cache[i].neighbor_leaf_indices)
        cover.push_back(leaf_hierarchy[leaf]);
      statistics[i].unique_entries = validate_hnl_cover(cover, geometry, leaf_count);
    }
  }
  allocated_bytes += mac_workspace.list_storage.allocated_bytes();
  used_bytes += mac_workspace.list_storage.stored_bytes();
  }
  std::cout << "List construction OpenMP threads actually used: " << actual_threads
            << "; schedule=" << policy.schedule_name << ", chunk=" << policy.chunk
            << "; target queue=leaves in hierarchy order; storage="
            << (policy.pooled ? "POOLED" : "VECTORS") << '\n';
  if (policy.pooled)
    std::cout << "Interaction-list pool allocated/used: " << allocated_bytes
              << "/" << used_bytes << " bytes (integer blocks only)\n";
  std::cout << "Root-walk exhaustive interval audit="
            << (audit ? "ON" : "OFF") << '\n';
}
#endif
