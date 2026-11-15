// Reuse the current production HNL/MAC/quadrature/multipole solver verbatim.
#define TREECODE_WATER10_MODEL 1
#define main water10_solver_main
#include "compare_exact.cpp"
#undef main
#include <filesystem>
#include <chrono>

int main(int argc,char** argv) {
  try {
    const auto options=parse_options(argc,argv);
    if(options.backend!=ComputeBackend::cpu)
      throw std::runtime_error("water10 currently supports CPU only; CUDA density kernels are not ported");
    if(options.indicator_mode!=IndicatorMode::radial)
      throw std::runtime_error("water10 currently requires RADIAL (30-center regularized radial monitor)");
    namespace fs=std::filesystem;
    const fs::path base=fs::current_path();
    const char* atom_env=std::getenv("WATER10_ATOMS");
    const fs::path atom_file=fs::absolute(atom_env?fs::path(atom_env):base/"water10_case/water10.centered.bohr.xyz");
    treecode_test_problem::load(atom_file.string());
    const char* pad_env=std::getenv("WATER10_PADDING");
    double padding=pad_env?parse_double(pad_env,"WATER10_PADDING"):4.0;
    if(!std::isfinite(padding)||padding<=1) throw std::runtime_error("WATER10_PADDING must exceed 1 bohr");
    std::ifstream mesh(fs::absolute(options.mesh_file));
    int n=0; if(!(mesh>>n)||n!=14) throw std::runtime_error("Use the original 14-vertex D.mesh template on [-5,5]^3");
    std::vector<std::array<double,3>> points(n);
    std::array<double,3> lo{1e30,1e30,1e30},hi{-1e30,-1e30,-1e30};
    for(auto& point:points) for(int j=0;j<3;++j) {
      if(!(mesh>>point[j])||!std::isfinite(point[j])) throw std::runtime_error("Invalid mesh vertex");
      lo[j]=std::min(lo[j],point[j]); hi[j]=std::max(hi[j],point[j]);
    }
    for(int j=0;j<3;++j) if(lo[j]!=-5||hi[j]!=5) throw std::runtime_error("Mesh template must span [-5,5]^3");
    const auto stamp=std::chrono::high_resolution_clock::now().time_since_epoch().count();
    fs::path run=base/"water10_case"/("run_"+std::to_string(stamp));
    if(!fs::create_directory(run)) throw std::runtime_error("Cannot create isolated run directory");
    std::ofstream out(run/"water10.mesh"); out<<std::setprecision(17)<<n<<'\n';
    for(const auto& point:points) {
      for(int j=0;j<3;++j) out<<point[j]*(treecode_test_problem::halfwidth[j]+padding)/5<<' ';
      out<<'\n';
    }
    out<<mesh.rdbuf(); out.close(); if(!out) throw std::runtime_error("Mesh write failed");
    fs::copy_file(atom_file,run/"atoms.bohr.xyz");
    std::ofstream meta(run/"parameters.txt");
    meta<<std::setprecision(17)<<"padding_bohr="<<padding<<"\nfull_space_electrons=100\nomitted_charge="
        <<treecode_test_problem::omitted_charge(padding)<<"\nargv=";
    for(int i=0;i<argc;++i) meta<<argv[i]<<' ';
    meta<<'\n';
    for(int j=0;j<3;++j) meta<<"box_halfwidth_"<<j<<'='<<treecode_test_problem::halfwidth[j]+padding<<'\n';
    meta.close();
    std::cout<<"Water decamer: positive Gaussian model, 30 atoms, 100 electrons\n"
             <<"Padding (bohr): "<<padding<<"; omitted charge: "<<treecode_test_problem::omitted_charge(padding)
             <<"\nOutput directory: "<<run<<"\n";
    fs::current_path(run);
    std::string filename="water10.mesh"; argv[1]=filename.data();
    const int status=water10_solver_main(argc,argv);
    fs::current_path(base); return status;
  } catch(const std::exception& e) { std::cerr<<"water10: "<<e.what()<<'\n'; return 1; }
}
