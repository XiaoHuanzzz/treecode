#include <iostream>
#include <AFEPack/HGeometry.h>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

/// for mesh
#include <AFEPack/HGeometry.h>
/// for template element
#include <AFEPack/TemplateElement.h>
/// for space
#include <AFEPack/DGFEMSpace.h>
/// for operators functionals
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
/// for AMG
#include <AFEPack/AMGSolver.h>


const int DIM = 3;
const double PI = 3.14159265358979323846;
const double algebraic_accuracy = 2;/// !!!!!!!!!!!!!!!!!!!!!!!!! for 3D, only tetrahedron has this number 6. Twin and four tetrahedron, no.
const double ZERO = 1.0e-12;

struct ElementCache : public ElementAdditionalData<double, DIM>
{
  HElement<DIM>* the_hElement;/// to build the connection between the element_cache and hElement_cache.
  
  double monopole;
  std::array<double, DIM> dipole;
  std::array<double, DIM * DIM> quadrupole;

  std::set<int> element_neighbors;/// to save Moore neighbors
  std::set<int> element_brothers;/// to save all possible brothers (may not be the Moore neighbor)
  std::set<int> element_patch;/// to save all elements in the near field, for multipole
};

void updateElementCache(std::vector<ElementCache > & element_cache,
      FEMSpace<double, DIM>& fem_space)
{
  element_cache.clear();
  
  const int& n_element = fem_space.n_element();
  element_cache.resize(n_element);

  const Mesh<DIM>&  r_mesh = fem_space.mesh();

  #pragma omp parallel for
  for(int i = 0;i < n_element;++ i){
    Element<double, DIM>& the_ele = fem_space.element(i);
    const int& ele_idx = the_ele.index();

    ElementCache& the_ec = element_cache[ele_idx];
    GeometryBM& the_geo = the_ele.geometry();

    const int& n_vtx = the_geo.n_vertex();
    std::vector<AFEPack::Point<DIM> > vertex(n_vtx);
    for(int j = 0;j < n_vtx;++ j){
      vertex[j] = r_mesh.point()[the_geo.vertex()[j]]; //
    }
    
    int& n_quadrature_point = the_ec.n_quadrature_point;
    double& volume = the_ec.volume;
    AFEPack::Point<DIM>& bc = the_ec.bc;
    std::vector<double>& Jxw = the_ec.Jxw;
    std::vector<AFEPack::Point<DIM> >& q_point = the_ec.q_point;

    std::vector<std::vector<double> >& basis_value = the_ec.basis_value;
    std::vector<std::vector<std::vector<double> > >& basis_gradient = the_ec.basis_gradient;


    barycenter(vertex, bc);
    
    double Tvolume = the_ele.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = the_ele.findQuadratureInfo(algebraic_accuracy); //
    std::vector<double> jacobian = the_ele.local_to_global_jacobian(quad_info.quadraturePoint());
    n_quadrature_point = quad_info.n_quadraturePoint();
    q_point = the_ele.local_to_global(quad_info.quadraturePoint());

    basis_value = the_ele.basis_function_value(q_point);
    basis_gradient = the_ele.basis_function_gradient(q_point);

    volume = 0.;
    Jxw.resize(n_quadrature_point);
    for (int l = 0;l < n_quadrature_point;l ++) {
      Jxw[l] = quad_info.weight(l)*jacobian[l]*Tvolume;
      volume += Jxw[l];      
    }
    
  }
}

int main(int argc, char* argv[])
{
  HGeometryTree<DIM> h_tree;
  h_tree.readMesh(argv[1]);

  std::unique_ptr<IrregularMesh<DIM>> ir_mesh = std::make_unique<IrregularMesh<DIM>>();

  ir_mesh->reinit(h_tree);
  ir_mesh->semiregularize();
    ir_mesh->regularize(false);
  ir_mesh->globalRefine(1);
  FEMSpace<double,DIM> fem_space;
    std::vector<TemplateElement<double,DIM,DIM> > template_elements;
    
    /// 对非正则网格做半正则化和正则化
    ir_mesh->semiregularize();
    ir_mesh->regularize(false);

    /// 这就是通过正则化得到的网格
    RegularMesh<DIM>& _r_mesh = ir_mesh->regularMesh();
    
    template_elements.resize(3);
   
   /// for tetrahedron
  TemplateGeometry<DIM> tet_geo;
  tet_geo.readData("tetrahedron.tmp_geo");
  TemplateDOF<DIM> tet_template_dof(tet_geo);
  tet_template_dof.readData("tetrahedron.1.tmp_dof");
  CoordTransform<DIM> tet_crd_trs;
  tet_crd_trs.readData("tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> tet_basFunctionAdmin(tet_template_dof);
  tet_basFunctionAdmin.readData("tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> tet_unit_outNormal;
  tet_unit_outNormal.readData("tetrahedron.out_nrm");

  template_elements[0].reinit(tet_geo,
			      tet_template_dof,
			      tet_crd_trs,
			      tet_basFunctionAdmin,
			      tet_unit_outNormal);

  /// for twin_tetrahedron
  TemplateGeometry<DIM> twin_tet_geo;
  twin_tet_geo.readData("twin_tetrahedron.tmp_geo");
  TemplateDOF<DIM> twin_tet_template_dof(twin_tet_geo);
  twin_tet_template_dof.readData("twin_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> twin_tet_crd_trs;
  twin_tet_crd_trs.readData("twin_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> twin_tet_basFunctionAdmin(twin_tet_template_dof);
  twin_tet_basFunctionAdmin.readData("twin_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> twin_tet_unit_outNormal;
  twin_tet_unit_outNormal.readData("twin_tetrahedron.out_nrm");

  template_elements[1].reinit(twin_tet_geo,
			      twin_tet_template_dof,
			      twin_tet_crd_trs,
			      twin_tet_basFunctionAdmin,
			      twin_tet_unit_outNormal);

  /// for four_tetrahedron
  TemplateGeometry<DIM> four_tet_geo;
  four_tet_geo.readData("four_tetrahedron.tmp_geo");
  TemplateDOF<DIM> four_tet_template_dof(four_tet_geo);
  four_tet_template_dof.readData("four_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> four_tet_crd_trs;
  four_tet_crd_trs.readData("four_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> four_tet_basFunctionAdmin(four_tet_template_dof);
  four_tet_basFunctionAdmin.readData("four_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> four_tet_unit_outNormal;
  four_tet_unit_outNormal.readData("four_tetrahedron.out_nrm");

  template_elements[2].reinit(four_tet_geo,
			      four_tet_template_dof,
			      four_tet_crd_trs,
			      four_tet_basFunctionAdmin,
			      four_tet_unit_outNormal);
 
   
    fem_space.reinit(_r_mesh, template_elements);
   int _n_element = _r_mesh.n_geometry(DIM);
   fem_space.element().resize(_n_element);
   for (int i = 0;i < _n_element;i ++) {
     if (_r_mesh.geometry(DIM,i).n_vertex() == 4) {
       fem_space.element(i).reinit(fem_space,i,0);
     } else if (_r_mesh.geometry(DIM,i).n_vertex() == 5){
       fem_space.element(i).reinit(fem_space,i,1);      
     } else if (_r_mesh.geometry(DIM,i).n_vertex() == 7){
       fem_space.element(i).reinit(fem_space,i,2); 
     } else{} //{std::cout << ""}
   }


   fem_space.buildElement();
   fem_space.buildDof();
   fem_space.buildDofBoundaryMark();

   std::vector<ElementCache> element_cache;
  
  updateElementCache(element_cache, fem_space);
  IrregularMesh<DIM>::ActiveIterator
    the_activeElement = ir_mesh->beginActiveElement(),
    end_activeElement = ir_mesh->endActiveElement();      
  for(;the_activeElement != end_activeElement;++ the_activeElement){
    const int& hElement_idx = the_activeElement->index;
    // std::cout << hElement_idx << std::endl;
    ElementCache& the_ec = element_cache[hElement_idx];
    the_ec.the_hElement = &(*the_activeElement);
    
  }

   const int& n_element = element_cache.size();
  
#pragma omp parallel for reduction(+:l2_error)
  for(int i = 0;i < n_element;++ i){

    Element<double, DIM>& the_ele = fem_space.element(i);
    ElementCache& the_ec = element_cache[i];
    HElement<DIM>* the_hElement = the_ec.the_hElement;
    if(the_hElement->value!= 0){
      continue;
    }
    
    double Tvolume = the_ele.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = the_ele.findQuadratureInfo(algebraic_accuracy);
    std::vector<double> jacobian = the_ele.local_to_global_jacobian(quad_info.quadraturePoint());
    const int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<AFEPack::Point<DIM> > q_point = the_ele.local_to_global(quad_info.quadraturePoint());
  }
  return 0;
}