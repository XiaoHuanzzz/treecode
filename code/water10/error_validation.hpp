#ifndef TREECODE_ERROR_VALIDATION_HPP
#define TREECODE_ERROR_VALIDATION_HPP

#include <cstdint>

namespace article_validation {

struct Target {
  AFEPack::Point<DIM> point;
  double weight = 0.0;
  HElement<DIM>* leaf = nullptr;
  int hierarchy_index = -1;
};

inline std::uint64_t splitmix(std::uint64_t& state) {
  std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

inline int setting(const char* name, int fallback, int low, int high) {
  const char* text = std::getenv(name);
  const int value = text ? parse_int(text, name) : fallback;
  if (value < low || value > high)
    throw std::runtime_error(std::string("Invalid validation setting: ") + name);
  return value;
}

inline std::vector<double> partition(int inner, int outer) {
  // Cover [-5,5], resolving the manufactured Gaussian in [-1,1].
  std::vector<double> edges;
  for (int i = 0; i < outer; ++i) edges.push_back(-5.0+4.0*i/outer);
  for (int i = 0; i < inner; ++i) edges.push_back(-1.0+2.0*i/inner);
  for (int i = 0; i <= outer; ++i) edges.push_back(1.0+4.0*i/outer);
  return edges;
}

inline double all_source_direct(
    const Target& target, const std::vector<ElementCache>& sources, int p) {
  // No MAC or multipole approximation. The quadrature samples and the
  // singular target-leaf rule are identical to those of the tree evaluator.
  double value = 0.0, correction = 0.0;
  for (const auto& source : sources) {
    const double piece = source.the_hElement == target.leaf
        ? singular_element_integral(target.point, source, p)
        : regular_quadrature_range(target.point, source, 0,
                                   source.q_point.size());
    const double adjusted = piece-correction;
    const double updated = value+adjusted;
    correction = (updated-value)-adjusted;
    value = updated;
  }
  return value;
}

} // namespace article_validation

int run_article_error_validation(
    const ProgramOptions& options,
    const std::vector<HElement<DIM>*>& roots,
    const std::vector<hElementCache>& hierarchy,
    const std::vector<ElementCache>& sources,
    const std::map<HElement<DIM>*,int>& indices) {
  using namespace article_validation;
  try {
    const int inner = setting("ERROR_VALIDATION_INNER", 12, 2, 128);
    const int outer = setting("ERROR_VALIDATION_OUTER", 2, 1, 16);
    const int seed = setting("ERROR_VALIDATION_SEED", 7351, 1, 1000000000);
    const bool direct = setting("ERROR_VALIDATION_DIRECT", 1, 0, 1) != 0;
    const auto edges = partition(inner,outer);
    const int side = static_cast<int>(edges.size())-1;
    const std::size_t count = static_cast<std::size_t>(side)*side*side;
    if (direct && count > 4096)
      throw std::runtime_error("Direct validation is limited to 4096 targets");
    double volume = 0;
    for (const auto& source : sources) volume += source.volume;
    if (std::abs(volume-1000.0) > 1e-6)
      throw std::runtime_error("This focused validation requires [-5,5]^3");
    std::vector<Target> targets(count);
    for (std::size_t k = 0; k < count; ++k) {
      Target& target = targets[k];
      const int cell[3] = {
          static_cast<int>(k%side),
          static_cast<int>((k/side)%side),
          static_cast<int>(k/(side*side))};
      std::uint64_t state = static_cast<std::uint64_t>(seed)
          + k*UINT64_C(0xd1342543de82ef95);
      target.weight = 1.0;
      for (int d = 0; d < 3; ++d) {
        const double offset = (static_cast<double>(splitmix(state)>>11)+0.5)
            /9007199254740992.0;
        const double width = edges[cell[d]+1]-edges[cell[d]];
        target.point[d] = edges[cell[d]]+width*offset;
        target.weight *= width;
      }
      target.leaf = findLeafContainingPoint(roots,target.point);
      if (!target.leaf) throw std::runtime_error("Validation target not located");
      target.hierarchy_index = indices.at(target.leaf);
    }

    std::vector<double> tree(count), exact(count), full;
    if (direct) full.resize(count);
    int failures = 0, actual_threads = 0;
    const double start = omp_get_wtime();
    #pragma omp parallel reduction(+:failures)
    {
      #pragma omp single
      actual_threads = omp_get_num_threads();
      std::array<double,treecode_multipole::maximum_coefficient_count> work{};
      #pragma omp for schedule(dynamic,16)
      for (std::size_t k = 0; k < count; ++k) {
        const auto& target = targets[k];
        tree[k] = multipole_approximation(
            target.point, target.leaf, hierarchy[target.hierarchy_index],
            hierarchy, sources, options.expansion_order,
            options.quadrature_accuracy, work.data());
        exact[k] = u(target.point);
        if (!std::isfinite(tree[k]) || !std::isfinite(exact[k])) ++failures;
      }
    }
    const double tree_seconds = omp_get_wtime()-start;
    const double direct_start = omp_get_wtime();
    if (direct) {
      #pragma omp parallel for schedule(dynamic,1) reduction(+:failures)
      for (std::size_t k = 0; k < count; ++k) {
        full[k] = all_source_direct(targets[k],sources,options.quadrature_accuracy);
        if (!std::isfinite(full[k])) ++failures;
      }
    }
    const double direct_seconds = direct ? omp_get_wtime()-direct_start : 0.0;
    if (failures) throw std::runtime_error("Nonfinite validation potentials");
    long double total2=0, fast2=0, integration2=0, cross=0, norm2=0, weight=0;
    double maximum_error=0;
    std::ofstream values("target_values.csv");
    values << std::setprecision(17) << "id,x,y,z,weight,tree,exact,direct\n";
    for (std::size_t k = 0; k < count; ++k) {
      const long double w=targets[k].weight;
      const long double error=tree[k]-exact[k];
      total2 += w*error*error;
      norm2 += w*exact[k]*exact[k];
      weight += w;
      maximum_error = std::max(maximum_error,std::abs(static_cast<double>(error)));
      if (direct) {
        const long double fast=tree[k]-full[k], integration=full[k]-exact[k];
        fast2 += w*fast*fast;
        integration2 += w*integration*integration;
        cross += w*fast*integration;
      }
      values << k;
      for (int d=0; d<3; ++d) values << ',' << targets[k].point[d];
      values << ',' << targets[k].weight << ',' << tree[k] << ',' << exact[k]
             << ',';
      if (direct) values << full[k];
      values << '\n';
    }
    values.close();
    if (!values || !(norm2>0) || std::abs(weight-1000.0L)>1e-9L)
      throw std::runtime_error("Invalid output, reference norm, or target weights");
    const long double residual = direct
        ? std::abs(total2-fast2-integration2-2*cross) : 0;
    if (direct && residual > 1e-12L*std::max(1e-20L,total2+fast2+integration2))
      throw std::runtime_error("Error-vector decomposition identity failed");
    std::ofstream summary("validation.csv");
    summary << "N,p,P,theta,inner,outer,seed,targets,threads,direct_enabled,"
        "E_total,E_relative,E_fast,E_integration,reference_norm,max_abs_error,"
        "error_cosine,identity_residual,tree_validation_s,direct_validation_s\n";
    summary << std::setprecision(17)
            << sources.size() << ',' << options.quadrature_accuracy << ','
            << options.expansion_order << ',' << options.mac_theta << ','
            << inner << ',' << outer << ',' << seed << ',' << count << ','
            << actual_threads << ',' << direct << ','
            << std::sqrt(total2) << ',' << std::sqrt(total2/norm2) << ',';
    if (direct) summary << std::sqrt(fast2);
    summary << ',';
    if (direct) summary << std::sqrt(integration2);
    summary << ',' << std::sqrt(norm2) << ',' << maximum_error << ',';
    if (direct && fast2>0 && integration2>0)
      summary << cross/std::sqrt(fast2*integration2);
    summary << ',' << residual << ',' << tree_seconds << ',' << direct_seconds << '\n';
    summary.close();
    if (!summary) throw std::runtime_error("Failed to write validation summary");
    std::cout << std::setprecision(15)
              << "Independent validation complete: " << count << " targets\n"
              << "Validation total absolute error: " << std::sqrt(total2) << '\n'
              << "Validation fast-summation error: " << std::sqrt(fast2) << '\n'
              << "Validation integration/reference error: "
              << std::sqrt(integration2) << '\n'
              << "Validation-only times (tree/direct): "
              << tree_seconds << " / " << direct_seconds << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error validation failed: " << error.what() << '\n';
    return 1;
  }
}
#endif
