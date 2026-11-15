CXX ?= g++
MPICXX ?= mpicxx
CXXFLAGS ?= -std=c++17 -O2 -g -Wall -Wextra -Wpedantic
AFEPACK_INC ?= /usr/local/src/AFEPack/library/include
AFEPACK_LIBS ?= -L/local/lib -lAFEPack -ldeal_II -lgomp

TESTS = quadrature_test multipole_test hnl_test hnl_family_mac_test mac_geometry_test

.PHONY: all check clean

all: $(TESTS)

quadrature_test: quadrature_test.cpp local_quadrature.hpp
	$(MPICXX) $(CXXFLAGS) -I$(AFEPACK_INC) $< -o $@ $(AFEPACK_LIBS)

multipole_test: multipole_test.cpp cartesian_multipole.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

hnl_test: hnl_test.cpp hnl_candidates.hpp hnl_tetra_test_fixture.hpp hnl_original_constructor_reference.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

hnl_family_mac_test: hnl_family_mac_test.cpp hnl_family_mac.hpp hnl_candidates.hpp hnl_tetra_test_fixture.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

mac_geometry_test: mac_geometry_test.cpp mac_geometry.hpp
	$(MPICXX) $(CXXFLAGS) -I$(AFEPACK_INC) $< -o $@ $(AFEPACK_LIBS)

check: all
	./quadrature_test 20
	./multipole_test
	./hnl_test
	./hnl_family_mac_test
	./mac_geometry_test

clean:
	rm -f $(TESTS)
