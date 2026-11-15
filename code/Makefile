MPICXX ?= mpicxx
AFEPACK_ROOT ?= /usr/local/src/AFEPack
DEALII_INCLUDE ?= /usr/local/include/deal.II

CXXFLAGS ?= -std=c++17 -O3 -g -Wall -Wextra -Wpedantic \
	-Wno-unused-parameter -I$(AFEPACK_ROOT)/library/include \
	-I$(DEALII_INCLUDE) -fopenmp
LDFLAGS ?= -L/local/lib
LDLIBS ?= -lAFEPack -ldeal_II -lgomp

AFEPACK_TEMPLATE_PATH := /usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/include/AFEPack/template/twin_tetrahedron/:/usr/local/include/AFEPack/template/four_tetrahedron/
export AFEPACK_TEMPLATE_PATH

.PHONY: all exact root-exact uniform reuse h2o water10 clean

all: exact

exact: tree_exact

tree_exact: compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

root-exact: tree_exact_root

tree_exact_root: compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_ROOT_BASED_LISTS compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

uniform: tree_uniform

tree_uniform: compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_UNIFORM_MESH compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

reuse: tree_reuse

tree_reuse: compare_reuse.cpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_reuse.cpp -o $@ $(LDFLAGS) $(LDLIBS)

h2o: tree_h2o

tree_h2o: h2o/compare_h2o.cpp h2o/h2o_three_center_test_problem.hpp \
		h2o/hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp \
		gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -I. -Ih2o h2o/compare_h2o.cpp -o $@ $(LDFLAGS) $(LDLIBS)

water10: tree_water10

tree_water10: water10/compare_water10.cpp water10/compare_exact.cpp \
		water10/water10_test_problem.hpp water10/local_quadrature.hpp \
		water10/cartesian_multipole.hpp water10/hnl_candidates.hpp \
		water10/hnl_family_mac.hpp water10/hnl_visualization.hpp
	$(MPICXX) $(CXXFLAGS) -Iwater10 water10/compare_water10.cpp -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f tree_exact tree_exact_root tree_uniform tree_reuse tree_h2o tree_water10
