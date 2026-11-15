#ifndef TREECODE_GENERAL_DOMAIN_MESH_CONTEXT_HPP
#define TREECODE_GENERAL_DOMAIN_MESH_CONTEXT_HPP

// Include after compare_exact.cpp with TREECODE_NO_MAIN defined.  This class
// owns every AFEPack object referenced by the regular mesh and FEM space.
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gd {

class MeshContext {
  struct TemplateFamily {
    TemplateGeometry<3> geometry;
    std::unique_ptr<TemplateDOF<3>> dof;
    CoordTransform<3> transform;
    std::unique_ptr<BasisFunctionAdmin<double, 3>> basis;
    UnitOutNormal<3> normals;

    explicit TemplateFamily(const std::string& prefix) {
      geometry.readData(prefix + ".tmp_geo");
      dof = std::make_unique<TemplateDOF<3>>(geometry);
      dof->readData(prefix + ".1.tmp_dof");
      transform.readData(prefix + ".crd_trs");
      basis = std::make_unique<BasisFunctionAdmin<double, 3>>(*dof);
      basis->readData(prefix + ".1.bas_fun");
      normals.readData(prefix + ".out_nrm");
    }
  };

  // Declaration order matters: fem/irregular are destroyed before the tree
  // and the template objects to which they refer.
  HGeometryTree<3> tree_;
  std::vector<std::unique_ptr<TemplateFamily>> families_;
  std::vector<TemplateElement<double, 3, 3>> template_elements_;
  std::size_t max_leaf_count_;

  void check_leaf_cap() {
    const std::size_t n = irregular->regularMesh().n_geometry(3);
    if (n > max_leaf_count_) {
      throw std::runtime_error(
          "General-domain mesh has " + std::to_string(n)
          + " active elements, exceeding the configured cap "
          + std::to_string(max_leaf_count_));
    }
  }

  void build_fem() {
    RegularMesh<3>& mesh = irregular->regularMesh();
    fem.reinit(mesh, template_elements_);
    const int n = mesh.n_geometry(3);
    fem.element().resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const int nv = mesh.geometry(3, i).n_vertex();
      const int family = nv == 4 ? 0 : nv == 5 ? 1 : nv == 7 ? 2 : -1;
      if (family < 0) {
        throw std::runtime_error("Unsupported AFEPack regular element type");
      }
      fem.element(i).reinit(fem, i, family);
    }
    fem.buildElement();
    fem.buildDof();
    fem.buildDofBoundaryMark();
  }

public:
  std::unique_ptr<IrregularMesh<3>> irregular;
  FEMSpace<double, 3> fem;

  explicit MeshContext(const std::string& mesh_path, int global_refinements,
                       std::size_t max_leaf_count = 300000)
      : max_leaf_count_(max_leaf_count),
        irregular(std::make_unique<IrregularMesh<3>>()) {
    if (global_refinements < 0 || max_leaf_count == 0) {
      throw std::invalid_argument("Invalid global refinement count or leaf cap");
    }
    tree_.readMesh(mesh_path);
    irregular->reinit(tree_);
    irregular->semiregularize();
    irregular->regularize(false);
    check_leaf_cap();
    for (int level = 0; level < global_refinements; ++level) {
      if (static_cast<std::size_t>(irregular->regularMesh().n_geometry(3))
          > max_leaf_count_ / 8) {
        throw std::runtime_error(
            "Next global refinement would exceed the configured leaf cap");
      }
      irregular->globalRefine(1);
      irregular->semiregularize();
      irregular->regularize(false);
      check_leaf_cap();
    }

    template_elements_.resize(3);
    const char* prefixes[] = {
        "tetrahedron", "twin_tetrahedron", "four_tetrahedron"};
    for (int i = 0; i < 3; ++i) {
      families_.push_back(std::make_unique<TemplateFamily>(prefixes[i]));
      TemplateFamily& family = *families_.back();
      template_elements_[i].reinit(
          family.geometry, *family.dof, family.transform,
          *family.basis, family.normals);
    }
    build_fem();
  }

  MeshContext(const MeshContext&) = delete;
  MeshContext& operator=(const MeshContext&) = delete;

  // Prescribed radial mesh monitor, not a residual/error estimator:
  // eta_K = integral_K [0.02 + (|x-(-2,-2,0)|^2 + 0.5^2)^(-1/2)] dx.
  // Recompute it on each new mesh with code-owned order-6 quadrature.
  void adapt(double tolerance, int passes, int refine_steps = 1) {
    adapt_with_monitor(tolerance, passes, refine_steps,
        [](double x, double y, double z) {
          x += 2.0;
          y += 2.0;
          return 0.02 + 1.0 / std::sqrt(
              x * x + y * y + z * z + 0.25);
        });
  }

  // The callable receives physical coordinates and must be safe to invoke
  // concurrently. The indicator is the order-6 integral of this monitor.
  template <class Monitor>
  void adapt_with_monitor(double tolerance, int passes, int refine_steps,
                          const Monitor& monitor) {
    if (!(tolerance > 0.0) || !std::isfinite(tolerance)
        || passes < 0 || refine_steps < 1) {
      throw std::invalid_argument("Invalid adaptation controls");
    }
    for (int pass = 0; pass < passes; ++pass) {
      check_leaf_cap();
      RegularMesh<3>& mesh = irregular->regularMesh();
      Indicator<3> indicator(mesh);
      const int n = fem.n_element();
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        const auto& geometry = fem.element(i).geometry();
        std::vector<AFEPack::Point<3>> vertices(
            static_cast<std::size_t>(geometry.n_vertex()));
        for (int j = 0; j < geometry.n_vertex(); ++j) {
          vertices[static_cast<std::size_t>(j)] =
              mesh.point()[geometry.vertex()[j]];
        }
        const auto rule = treecode_quadrature::build_physical_rule(vertices, 6);
        double value = 0.0;
        for (const auto& q : rule) {
          value += q.weight * monitor(q.point[0], q.point[1], q.point[2]);
        }
        indicator[geometry.index()] = value;
      }

      // Retain the old mesh and its regular view until fem has been rebound.
      auto old = std::move(irregular);
      irregular = std::make_unique<IrregularMesh<3>>();
      MeshAdaptor<3> adaptor(*old, *irregular);
      adaptor.setIndicator(indicator);
      adaptor.tolerence() = tolerance;
      adaptor.convergenceOrder() = 0.0;
      adaptor.refineStep() = refine_steps;
      adaptor.adapt();
      irregular->semiregularize();
      irregular->regularize(false);
      check_leaf_cap();
      build_fem();
      std::cout << "General-domain adaptation pass " << pass + 1
                << ": tolerance=" << tolerance
                << ", active elements=" << fem.n_element() << '\n';
    }
  }

  std::vector<HElement<3>*> roots() {
    std::vector<HElement<3>*> result;
    for (auto it = irregular->beginRootElement();
         it != irregular->endRootElement(); ++it) {
      result.push_back(&*it);
    }
    return result;
  }

  std::vector<std::array<AFEPack::Point<3>, 4>> tetrahedra() {
    const RegularMesh<3>& mesh = irregular->regularMesh();
    std::vector<std::array<AFEPack::Point<3>, 4>> result;
    result.reserve(static_cast<std::size_t>(fem.n_element()));
    for (int i = 0; i < fem.n_element(); ++i) {
      const auto& geometry = fem.element(i).geometry();
      const auto decomposition = treecode_quadrature::sub_tetrahedron_indices(
          static_cast<std::size_t>(geometry.n_vertex()));
      for (const auto& indices : decomposition) {
        std::array<AFEPack::Point<3>, 4> tetrahedron;
        for (int j = 0; j < 4; ++j) {
          tetrahedron[static_cast<std::size_t>(j)] = mesh.point()[
              geometry.vertex()[indices[static_cast<std::size_t>(j)]]];
        }
        result.push_back(tetrahedron);
      }
    }
    return result;
  }

  void export_dx(const std::string& path) {
    irregular->regularMesh().writeOpenDXData(path);
  }
};

} // namespace gd

#endif
