#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Positive manufactured electron density, not an SCF water density.
namespace treecode_test_problem {
constexpr double pi = 3.141592653589793238462643383279502884;
struct Atom { double q, alpha; std::array<double,3> r; };
inline std::vector<Atom> atoms;
inline std::array<double,3> halfwidth{};
inline void load(const std::string& path) {
  std::ifstream in(path);
  int count=0; std::string line;
  if (!(in >> count) || count != 30) throw std::runtime_error("Expected 30-atom XYZ: " + path);
  std::getline(in,line); std::getline(in,line);
  atoms.clear(); halfwidth={0,0,0}; int oxygen=0;
  for (int i=0;i<count;++i) {
    std::string symbol; Atom a;
    if (!(in>>symbol>>a.r[0]>>a.r[1]>>a.r[2]) || (symbol!="O" && symbol!="H"))
      throw std::runtime_error("Invalid atom in water10 XYZ");
    a.q=symbol=="O"?8:1; a.alpha=symbol=="O"?2:1; oxygen+=symbol=="O";
    for(int j=0;j<3;++j) {
      if(!std::isfinite(a.r[j])) throw std::runtime_error("Nonfinite atom coordinate");
      halfwidth[j]=std::max(halfwidth[j],std::abs(a.r[j]));
    }
    atoms.push_back(a);
  }
  if(oxygen!=10) throw std::runtime_error("Expected 10 oxygen and 20 hydrogen atoms");
}
inline double source_value(double x,double y,double z) {
  double rho=0;
  for(const auto& a:atoms) {
    double s=std::pow(x-a.r[0],2)+std::pow(y-a.r[1],2)+std::pow(z-a.r[2],2);
    rho+=a.q*std::pow(a.alpha/pi,1.5)*std::exp(-a.alpha*s);
  }
  return rho;
}
inline double exact_value(double x,double y,double z) {
  double v=0;
  for(const auto& a:atoms) {
    double r=std::sqrt(std::pow(x-a.r[0],2)+std::pow(y-a.r[1],2)+std::pow(z-a.r[2],2));
    v+=a.q*(r<1.e-8?2*std::sqrt(a.alpha/pi):std::erf(std::sqrt(a.alpha)*r)/r);
  }
  return v;
}
// Smooth, equally weighted radial monitors around ALL 30 atoms.
// Width 0.5 bohr regularizes the monitor; it does not change the density.
inline double radial_monitor(double x,double y,double z) {
  double m=0;
  for(const auto& a:atoms)
    m+=1/std::sqrt(.25+std::pow(x-a.r[0],2)+std::pow(y-a.r[1],2)+std::pow(z-a.r[2],2));
  return m/atoms.size();
}
inline double exact_hessian_frobenius_squared(double x,double y,double z) {
  double h[3][3]{};
  for(const auto& a:atoms) {
    double d[3]={x-a.r[0],y-a.r[1],z-a.r[2]};
    double s=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
    double c=2*std::sqrt(a.alpha/pi), t, b;
    if(a.alpha*s<1.e-5) {
      t=c*(-2*a.alpha/3+2*a.alpha*a.alpha*s/5-a.alpha*a.alpha*a.alpha*s*s/7);
      b=c*(4*a.alpha*a.alpha/5-4*a.alpha*a.alpha*a.alpha*s/7);
    } else {
      double r=std::sqrt(s), e=std::exp(-a.alpha*s), f=std::erf(std::sqrt(a.alpha)*r);
      t=c*e/s-f/(s*r);
      b=-2*c*a.alpha*e/s-3*c*e/(s*s)+3*f/(s*s*r);
    }
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
      h[i][j]+=a.q*((i==j?t:0)+b*d[i]*d[j]);
  }
  double result=0; for(auto& row:h) for(double v:row) result+=v*v; return result;
}
inline double omitted_charge(double padding) {
  double q=0;
  for(const auto& a:atoms) {
    double tail[3];
    for(int j=0;j<3;++j) {
      double l=halfwidth[j]+padding;
      tail[j]=.5*(std::erfc(std::sqrt(a.alpha)*(l-a.r[j]))+
                      std::erfc(std::sqrt(a.alpha)*(l+a.r[j])));
    }
    q+=a.q*(tail[0]+tail[1]+tail[2]-tail[0]*tail[1]-tail[0]*tail[2]-tail[1]*tail[2]+tail[0]*tail[1]*tail[2]);
  }
  return q;
}
}
