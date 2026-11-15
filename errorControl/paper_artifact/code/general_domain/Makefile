CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra -Wno-unused-parameter -I/usr/local/include/deal.II
LDLIBS = -L/usr/local/lib -lAFEPack -ldeal_II
AFEPACK_TEMPLATE_PATH := /usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/include/AFEPack/template/twin_tetrahedron/:/usr/local/include/AFEPack/template/four_tetrahedron/
export AFEPACK_TEMPLATE_PATH

all: compare_general_domain
compare_general_domain: compare_general_domain.cpp mesh_context.hpp direct_reference.hpp ../compare_exact.cpp ../local_quadrature.hpp ../cartesian_multipole.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

compare_localized_domain: compare_general_domain.cpp mesh_context.hpp direct_reference.hpp ../compare_exact.cpp ../local_quadrature.hpp ../cartesian_multipole.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

compare_radial_test_domain: compare_radial_domain.cpp compare_general_domain.cpp mesh_context.hpp direct_reference.hpp ../compare_exact.cpp ../local_quadrature.hpp ../cartesian_multipole.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

generate_localized_mesh: generate_localized_mesh.cpp mesh_context.hpp ../compare_exact.cpp ../local_quadrature.hpp ../cartesian_multipole.hpp ../gaussian_test_problem.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

.PHONY: all pilot
pilot: compare_general_domain
	python3 run_study.py
