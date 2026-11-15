// Separate general-domain experiment. The production single-center model is
// untouched; only this translation unit substitutes a positive source density.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <random>
#include <sstream>
#include <filesystem>
#include <numeric>

#ifndef TREECODE_GENERAL_DOMAIN_SOURCE_DEFINED
#define TREECODE_GENERAL_DOMAIN_SOURCE_DEFINED
#define TREECODE_GENERAL_DOMAIN_SOURCE_DESCRIPTION \
  "exp(-(x+2)^2-2*(y+2)^2-3*z^2), truncated to the mesh domain"
#define TREECODE_GAUSSIAN_TEST_PROBLEM_HPP
namespace treecode_test_problem {
constexpr double pi = 3.141592653589793238462643383279502884;
inline double source_value(double x, double y, double z) {
  x += 2.0; y += 2.0;
  return std::exp(-x*x - 2.0*y*y - 3.0*z*z);
}
inline double exact_value(double, double, double) {
  throw std::logic_error("No analytical potential is used in the general-domain experiment");
}
}
#endif
#define TREECODE_NO_MAIN
#include "../compare_exact.cpp"
#include "mesh_context.hpp"
#include "direct_reference.hpp"

namespace gd {
struct Target {
  AFEPack::Point<3> point;
  std::string group;
};

struct Options {
  std::string mode="solve", mesh, targets, output;
  std::string monitor="radial";
  int level=0, p=6, P=4, passes=2, refine_steps=1, near_depth=3;
  int volume_targets=128, core_targets=32;
  std::size_t leaf_cap=300000;
  double theta=0.5, tolerance=0.03;
  double monitor_width=1.2, monitor_floor=0.0001, monitor_gamma=4.0;
  bool adaptive=true, same_direct=false, lists_only=false;
};

double parse_monitor_real(const std::string& value, const std::string& name) {
  std::size_t consumed=0;
  const double result=std::stod(value,&consumed);
  if(consumed!=value.size() || !std::isfinite(result))
    throw std::invalid_argument("Expected a finite numeric value for "+name);
  return result;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i=1;i<argc;++i) {
    std::string a=argv[i];
    if (a=="--uniform") { o.adaptive=false; continue; }
    if (a=="--same-direct") { o.same_direct=true; continue; }
    if (a=="--lists-only") { o.lists_only=true; continue; }
    if (i+1==argc) throw std::invalid_argument("Missing value for "+a);
    std::string v=argv[++i];
    if(a=="--mode") o.mode=v;
    else if(a=="--mesh") o.mesh=v;
    else if(a=="--targets") o.targets=v;
    else if(a=="--output") o.output=v;
    else if(a=="--monitor") o.monitor=v;
    else if(a=="--monitor-width") o.monitor_width=parse_monitor_real(v,a);
    else if(a=="--monitor-floor") o.monitor_floor=parse_monitor_real(v,a);
    else if(a=="--monitor-gamma") o.monitor_gamma=parse_monitor_real(v,a);
    else if(a=="--level") o.level=std::stoi(v);
    else if(a=="--p") o.p=std::stoi(v);
    else if(a=="--P") o.P=std::stoi(v);
    else if(a=="--theta") o.theta=std::stod(v);
    else if(a=="--tol") o.tolerance=std::stod(v);
    else if(a=="--passes") o.passes=std::stoi(v);
    else if(a=="--refine-steps") o.refine_steps=std::stoi(v);
    else if(a=="--near-depth") o.near_depth=std::stoi(v);
    else if(a=="--volume-targets") o.volume_targets=std::stoi(v);
    else if(a=="--core-targets") o.core_targets=std::stoi(v);
    else if(a=="--leaf-cap") o.leaf_cap=std::stoull(v);
    else throw std::invalid_argument("Unknown argument "+a);
  }
  if(o.mesh.empty() || o.output.empty())
    throw std::invalid_argument("Required: --mesh FILE --output PREFIX; modes targets, solve, reference");
  if(o.mode!="targets" && o.mode!="solve" && o.mode!="reference")
    throw std::invalid_argument("Mode must be targets, solve, or reference");
  if(o.lists_only && o.mode!="solve")
    throw std::invalid_argument("--lists-only is valid only in solve mode");
  if(o.monitor!="radial" && o.monitor!="localized" && o.monitor!="radial_graded")
    throw std::invalid_argument("--monitor must be radial, localized, or radial_graded");
  if(!(o.monitor_width>0) || o.monitor_floor<0)
    throw std::invalid_argument("--monitor-width must be positive; --monitor-floor must be nonnegative");
  if(!(o.monitor_gamma>=1.0 && o.monitor_gamma<=20.0))
    throw std::invalid_argument("--monitor-gamma must be in [1,20]");
  if(o.p<2 || o.p>20 || o.P<1 || o.P>20 || o.level<0 || o.level>3 ||
     !(o.theta>0 && o.theta<1) || !(o.tolerance>0) || o.passes<0 ||
     o.refine_steps<1 || o.refine_steps>3 || o.near_depth<0 || o.near_depth>6 ||
     o.volume_targets<1 || o.core_targets<0 || o.leaf_cap<1 ||
     o.leaf_cap>10000000)
    throw std::invalid_argument("Invalid parameter range");
  std::filesystem::path parent=std::filesystem::path(o.output).parent_path();
  if(!parent.empty()) std::filesystem::create_directories(parent);
  return o;
}

void write_stats(const std::string& path,const Options& o,
                 const std::map<std::string,double>& stats) {
  std::ofstream out(path);
  const bool monitor_active=o.mode=="solve" && o.adaptive && o.passes>0;
  const char* indicator=!monitor_active ? "inactive" : o.monitor=="localized"
      ? "integral_K [floor + (1 + |x-(-2,-2,0)|^2/width^2)^(-3)] dx"
      : o.monitor=="radial_graded"
      ? "integral_K [1 + (gamma-1) r^2/(r^2+width^2)]^(-3) dx"
      : "integral_K (0.02 + 1/sqrt(|x-(-2,-2,0)|^2+0.25)) dx";
  out << std::setprecision(17)
      << "{\n  \"mode\": \"" << o.mode << "\",\n"
      << "  \"mesh_kind\": \"" << ((o.mode=="reference" || !o.adaptive)?"uniform":"adaptive")
      << "\",\n  \"source\": \"" TREECODE_GENERAL_DOMAIN_SOURCE_DESCRIPTION "\",\n"
      << "  \"monitor\": \"" << o.monitor << "\",\n"
      << "  \"monitor_active\": " << (monitor_active?"true":"false") << ",\n"
      << "  \"monitor_width\": " << o.monitor_width << ",\n"
      << "  \"monitor_floor\": " << o.monitor_floor << ",\n"
      << "  \"monitor_gamma\": " << o.monitor_gamma << ",\n"
      << "  \"indicator\": \"" << indicator << "\"";
  for(const auto& kv:stats) {
    if(!std::isfinite(kv.second)) throw std::runtime_error("Nonfinite statistic "+kv.first);
    out << ",\n  \"" << kv.first << "\": " << kv.second;
  }
  out << "\n}\n";
  if(!out) throw std::runtime_error("Cannot write "+path);
}

std::vector<Target> read_targets(const std::string& path) {
  std::ifstream in(path);
  if(!in) throw std::runtime_error("Cannot read targets "+path);
  std::string line; std::getline(in,line);
  std::vector<Target> t;
  while(std::getline(in,line)) {
    if(line.empty()) continue;
    std::replace(line.begin(),line.end(),',',' ');
    std::istringstream s(line); int id; Target v;
    if(!(s>>id>>v.group>>v.point[0]>>v.point[1]>>v.point[2]))
      throw std::runtime_error("Malformed target row");
    t.push_back(v);
  }
  if(t.empty()) throw std::runtime_error("Empty target set");
  return t;
}

void write_values(const std::string& path,const std::vector<Target>& t,
                  const std::vector<double>& values) {
  if(t.size()!=values.size()) throw std::runtime_error("Target/value size mismatch");
  std::ofstream out(path);
  out << "id,group,x,y,z,potential\n" << std::setprecision(17);
  for(std::size_t i=0;i<t.size();++i) {
    if(!std::isfinite(values[i])) throw std::runtime_error("Nonfinite potential at target "+std::to_string(i));
    out << i << ',' << t[i].group << ',' << t[i].point[0] << ',' << t[i].point[1]
        << ',' << t[i].point[2] << ',' << values[i] << '\n';
  }
  if(!out) throw std::runtime_error("Cannot write "+path);
}

void generate_targets(MeshContext& mesh,const Options& o) {
  auto tets=mesh.tetrahedra();
  std::vector<double> weights; double volume=0;
  for(const auto& t:tets) {
    double v=tetrahedronVolume(t[0],t[1],t[2],t[3]);
    weights.push_back(v); volume+=v;
  }
  std::mt19937_64 rng(20261108);
  std::discrete_distribution<std::size_t> cell(weights.begin(),weights.end());
  std::uniform_real_distribution<double> unif(0.0,1.0);
  std::ofstream out(o.output);
  out << "id,group,x,y,z\n" << std::setprecision(17);
  for(int i=0;i<o.volume_targets;++i) {
    const auto& tet=tets[cell(rng)];
    double l[4],sum=0;
    for(double& x:l) { x=-std::log(std::max(unif(rng),1e-15)); sum+=x; }
    AFEPack::Point<3> point;
    for(int d=0;d<3;++d) {
      point[d]=0;
      for(int k=0;k<4;++k) point[d]+=l[k]/sum*tet[k][d];
    }
    out << i << ",volume," << point[0] << ',' << point[1] << ',' << point[2] << '\n';
  }
  // Extra source-core targets are separate diagnostics, not unweighted
  // additions to a domain-volume error estimate.
  auto roots=mesh.roots();
  for(int i=0;i<o.core_targets;++i) {
    AFEPack::Point<3> point;
    for(;;) {
      double x=2*unif(rng)-1,y=2*unif(rng)-1,z=2*unif(rng)-1;
      if(x*x+y*y+z*z>=1) continue;
      point[0]=-2+0.9*x; point[1]=-2+0.9*y; point[2]=0.9*z;
      if(findLeafContainingPoint(roots,point)) break;
    }
    out << o.volume_targets+i << ",core," << point[0] << ',' << point[1] << ',' << point[2] << '\n';
  }
  if(!out) throw std::runtime_error("Cannot write target file");
  std::cout << "Generated " << o.volume_targets << " volume-uniform and "
            << o.core_targets << " source-core targets; mesh volume=" << volume << '\n';
}

int run(const Options& o) {
  const double start=omp_get_wtime();
  MeshContext mesh(o.mesh,o.mode=="targets"?0:o.level,o.leaf_cap);
  if(o.mode=="targets") { generate_targets(mesh,o); return 0; }
  if(o.mode=="solve" && o.adaptive) {
    if(o.monitor=="radial") {
      mesh.adapt(o.tolerance,o.passes,o.refine_steps);
    } else if(o.monitor=="localized") {
      const auto monitor=[&o](double x,double y,double z) {
        constexpr std::array<double,3> source_center{{-2.0,-2.0,0.0}};
        // Keep the arithmetic identical to generate_localized_mesh.cpp,
        // including division before squaring for extreme finite widths.
        const double dx=(x-source_center[0])/o.monitor_width;
        const double dy=(y-source_center[1])/o.monitor_width;
        const double dz=(z-source_center[2])/o.monitor_width;
        const double inverse=1.0/(1.0+dx*dx+dy*dy+dz*dz);
        return o.monitor_floor+inverse*inverse*inverse;
      };
      mesh.adapt_with_monitor(o.tolerance,o.passes,o.refine_steps,monitor);
    } else {
      const auto monitor=[&o](double x,double y,double z) {
        const double dx=(x+2.0)/o.monitor_width;
        const double dy=(y+2.0)/o.monitor_width;
        const double dz=z/o.monitor_width;
        const double scaled_radius_squared=dx*dx+dy*dy+dz*dz;
        const double transition=1.0-1.0/(1.0+scaled_radius_squared);
        const double relative_size=1.0+(o.monitor_gamma-1.0)*transition;
        return 1.0/(relative_size*relative_size*relative_size);
      };
      mesh.adapt_with_monitor(o.tolerance,o.passes,o.refine_steps,monitor);
    }
  }
  const double mesh_s=omp_get_wtime()-start;
  std::vector<Target> targets=read_targets(o.targets);
  std::vector<AFEPack::Point<3>> points;
  for(const auto& t:targets) points.push_back(t.point);
  auto roots=mesh.roots();
  for(const auto& point:points)
    if(!findLeafContainingPoint(roots,point)) throw std::runtime_error("Target outside the source mesh domain");
  int team=1;
  #pragma omp parallel
  {
    #pragma omp single
    team=omp_get_num_threads();
  }
  std::map<std::string,double> stats{
    {"N",mesh.fem.n_element()},{"root_count",roots.size()},
    {"level",o.level},{"p",o.p},{"P",o.P},{"theta",o.theta},
    {"tolerance",o.tolerance},{"passes",o.passes},{"threads",team},
    {"leaf_cap",static_cast<double>(o.leaf_cap)},
    {"mesh_s",mesh_s},{"target_count",targets.size()},{"near_depth",o.near_depth}};
  if(o.mode=="reference") {
    auto tets=mesh.tetrahedra();
    auto direct=direct_reference(tets,points,o.p,o.near_depth);
    write_values(o.output+".values.csv",targets,direct.values);
    stats["tetrahedra"]=tets.size(); stats["charge"]=direct.charge;
    stats["volume"]=direct.volume; stats["quadrature_points"]=direct.quadrature_points;
    stats["reference_preparation_s"]=direct.preparation_s;
    stats["reference_evaluation_s"]=direct.evaluation_s;
    stats["near_corrections"]=direct.near_corrections;
    stats["reference_total_s"]=mesh_s+direct.preparation_s+direct.evaluation_s;
    write_stats(o.output+".json",o,stats);
    mesh.export_dx(o.output+".dx");
    std::cout << "Reference complete: N=" << stats["N"] << ", Q=" << direct.charge
              << ", evaluation_s=" << direct.evaluation_s << '\n';
    return 0;
  }

  std::vector<hElementCache> hierarchy;
  std::map<HElement<DIM>*,int> mapping;
  std::map<std::string,double> interaction_diagnostics;
  double geometry_s=0,validation_s=0;
  double begin=omp_get_wtime();
  double list_kernel_s=update_hElementCache(*mesh.irregular,hierarchy,mapping,
      o.theta,true,&geometry_s,&validation_s,&team,&interaction_diagnostics);
  const double list_s=omp_get_wtime()-begin;
  stats["list_s"]=list_s;
  stats["list_kernel_s"]=list_kernel_s;
  stats["mapping_geometry_s"]=geometry_s;
  stats["hierarchy_validation_s"]=validation_s;
  stats["lists_only"]=o.lists_only ? 1.0 : 0.0;
  for(const auto& item : interaction_diagnostics) {
    stats[item.first]=item.second;
  }
  if(o.lists_only) {
    stats["total_solver_s"]=mesh_s+list_s;
    write_stats(o.output+".json",o,stats);
    mesh.export_dx(o.output+".dx");
    std::cout << "List diagnostics complete: N=" << stats["N"]
              << ", list_s=" << list_s << '\n';
    return 0;
  }
  std::vector<ElementCache> elements;
  begin=omp_get_wtime();
  updateElementCache(elements,mesh.fem,o.p);
  IrregularMesh<DIM>::ActiveIterator it=mesh.irregular->beginActiveElement();
  IrregularMesh<DIM>::ActiveIterator end=mesh.irregular->endActiveElement();
  for(;it!=end;++it) {
    auto& cache=elements.at(it->index);
    cache.the_hElement=&*it; cache.hierarchy_index=mapping.at(&*it);
  }
  double charge=0,volume=0; std::size_t qcount=0;
  for(const auto& cell:elements) {
    if(!cell.the_hElement) throw std::runtime_error("Leaf mapping missing");
    volume+=cell.volume; qcount+=cell.q_point.size();
    charge=std::accumulate(cell.source_charge.begin(),cell.source_charge.end(),charge);
  }
  double quadrature_s=omp_get_wtime()-begin;
  std::vector<HElement<DIM>*> nodes(hierarchy.size(),nullptr);
  for(const auto& kv:mapping) nodes[kv.second]=kv.first;
  begin=omp_get_wtime();
  #pragma omp parallel for schedule(dynamic,16)
  for(std::size_t i=0;i<nodes.size();++i)
    update_multipoleInfo_hElementCache(nodes[i],hierarchy,mapping,elements,o.P,false);
  double moments_s=omp_get_wtime()-begin;
  std::vector<double> leaf_values(elements.size());
  begin=omp_get_wtime();
  #pragma omp parallel for schedule(dynamic,16)
  for(std::size_t i=0;i<elements.size();++i) {
    const auto& cell=elements[i]; const auto& h=hierarchy[cell.hierarchy_index];
    std::array<double,treecode_multipole::maximum_coefficient_count> work;
    leaf_values[i]=multipole_approximation(h.ref_point,cell.the_hElement,h,
        hierarchy,elements,o.P,o.p,work.data());
  }
  const double evaluation_s=omp_get_wtime()-begin;
  double checksum=0;
  for(std::size_t i=0;i<elements.size();++i) {
    if(!std::isfinite(leaf_values[i])) throw std::runtime_error("Nonfinite leaf evaluation");
    checksum+=elements[i].volume*leaf_values[i];
  }
  std::vector<int> owners;
  for(const auto& point:points) owners.push_back(mapping.at(findLeafContainingPoint(roots,point)));
  std::vector<double> values(points.size());
  begin=omp_get_wtime();
  #pragma omp parallel for schedule(dynamic,1)
  for(std::size_t i=0;i<points.size();++i) {
    std::array<double,treecode_multipole::maximum_coefficient_count> work;
    int h=owners[i];
    values[i]=multipole_approximation(points[i],nodes[h],hierarchy[h],
        hierarchy,elements,o.P,o.p,work.data());
  }
  stats["validation_target_evaluation_s"]=omp_get_wtime()-begin;
  stats["charge"]=charge; stats["volume"]=volume; stats["quadrature_points"]=qcount;
  stats["list_s"]=list_s; stats["list_kernel_s"]=list_kernel_s;
  stats["quadrature_s"]=quadrature_s; stats["moments_s"]=moments_s;
  stats["preparation_s"]=quadrature_s+moments_s; stats["evaluation_s"]=evaluation_s;
  stats["total_solver_s"]=mesh_s+list_s+quadrature_s+moments_s+evaluation_s;
  stats["integrated_potential_checksum"]=checksum;
  std::size_t near=0,far=0;
  for(const auto& e:elements) {
    near+=hierarchy[e.hierarchy_index].neighbor_leaf_indices.size();
    far+=hierarchy[e.hierarchy_index].far_hierarchy_indices.size();
  }
  stats["near_entries"]=near;stats["far_entries"]=far;
  write_values(o.output+".values.csv",targets,values);
  if(o.same_direct) {
    auto direct=direct_reference(mesh.tetrahedra(),points,o.p,0);
    write_values(o.output+".direct.csv",targets,direct.values);
    stats["same_mesh_direct_s"]=direct.preparation_s+direct.evaluation_s;
    if(std::abs(direct.charge-charge)>1e-10*std::max(1.0,std::abs(charge)))
      throw std::runtime_error("Direct/production source charges disagree");
  }
  write_stats(o.output+".json",o,stats);
  mesh.export_dx(o.output+".dx");
  std::cout << "Solve complete: N=" << stats["N"] << ", Q=" << charge
            << ", list_s=" << list_s << ", eval_s=" << evaluation_s
            << ", total_s=" << stats["total_solver_s"] << '\n';
  return 0;
}
}

int main(int argc,char** argv) {
  try {
    std::cout << std::setprecision(15);
    return gd::run(gd::parse(argc,argv));
  } catch(const std::exception& e) {
    std::cerr << "GENERAL-DOMAIN ERROR: " << e.what() << '\n';
    return 1;
  }
}
