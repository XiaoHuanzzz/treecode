MPICXX = mpicxx
.DEFAULT_GOAL := all
.PHONY: water10 run-water10
WATER10_RUN_ARGS ?= D.mesh 1 6 4 0.5 1 CPU 0.1 0.03 RADIAL 6 0
water10: tree_water10
tree_water10: compare_water10.cpp compare_exact.cpp water10_test_problem.hpp local_quadrature.hpp cartesian_multipole.hpp hnl_candidates.hpp hnl_family_mac.hpp hnl_visualization.hpp
	$(MPICXX) $(CXXFLAGS) compare_water10.cpp -o $@ $(LDFLAGS) $(LDLIBS)
run-water10: tree_water10
	./tree_water10 $(WATER10_RUN_ARGS)
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

CXXFLAGS = -std=c++17 -O3 -g -Wall -Wextra -Wpedantic -Wno-unused-parameter -I/usr/local/src/AFEPack/library/include -I/usr/local/include/deal.II -fopenmp
LDFLAGS = -L/local/lib
LDLIBS = -lAFEPack -ldeal_II -lgomp

SRC = compare_exact.cpp
OUT = tree
REFERENCE_SRC = uni2.cpp
REFERENCE_OUT = uni2
GPU_OUT = tree_gpu
GPU_HOST_OBJ = compare_exact.gpu.o
GPU_KERNEL_OBJ = compare_gpu.cuda.o
EXACT_OUT = tree_exact
EXACT_DIFFUSE_OUT = tree_exact_diffuse
EXACT_BASELINE_OUT = tree_exact_baseline
EXACT_SHARP_OUT = tree_exact_sharp
EXACT_GPU_OUT = tree_exact_gpu
UNIFORM_OUT = tree_uniform
TREE_REFERENCE_OUT = tree_reference
UNIFORM_REFERENCE_OUT = tree_uniform_reference
PLATEAU_OUT = tree_plateau
H2O_OUT = tree_h2o
REUSE_OUT = tree_reuse
MESH_MODES_OUT = tree_mesh_modes
EXACT_GPU_HOST_OBJ = $(GPU_HOST_OBJ)
QUADRATURE_TEST_OUT = local_quadrature_test
MULTIPOLE_TEST_OUT = multipole_test
HNL_TEST_OUT = hnl_test
ROOT_COMPARE_OUT = tree_root
HNL_VISUAL_OUT = tree_hnl_visualize
HNL_VERIFY_OUT = tree_hnl_verify
ROOT_VERIFY_OUT = tree_root_verify
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
CUDA_ARCH_FLAGS ?=
CUDA_LDFLAGS ?= -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart

COMPARE ?= 0
BACKEND ?= CPU
# Optional CPU execution policy: CACHED retains lists; STREAM evaluates one
# target at a time using worker-local lists (requires CPU and compare=1).
TREECODE_CPU_INTERACTIONS ?= CACHED
export TREECODE_CPU_INTERACTIONS
# Experimental admissible family coarsening; original geometric HNL unchanged.
TREECODE_HNL_MAC_FAMILIES ?= 0
export TREECODE_HNL_MAC_FAMILIES
# Experimental one-parent target grouping for CPU STREAM HNL. Each shared
# geometric cover is followed by the unchanged MAC for every target leaf.
TREECODE_HNL_SHARED_TARGETS ?= 0
export TREECODE_HNL_SHARED_TARGETS
# RADIAL remains the evidence-backed default.  GAUSSIAN_RADIAL uses the
# anisotropic radius sqrt(x^2 + 2 y^2 + 3 z^2), but matched-size tests give
# essentially the same error.  DENSITY_VARIANCE is experimental and needs
# separately calibrated tolerances (about 2.0e-4 and 2.0e-5 near 1.0e5 leaves).
ADAPT_TOLERANCE_STAGE1 ?= 1.0e-2
ADAPT_TOLERANCE_STAGE2 ?= 1.0e-3
INDICATOR_MODE ?= RADIAL
INDICATOR_ORDER ?= 8
REPORT_MESH_ESTIMATOR ?= 1
RUN_ARGS ?= D.mesh 2 6 4 0.5 $(COMPARE) CPU
REFERENCE_ARGS ?= D.mesh 5 6 results_u1.bin 4096
GPU_RUN_ARGS ?= D.mesh 2 2 4 0.5 $(COMPARE) $(BACKEND)
EXACT_RUN_ARGS ?= D.mesh 2 6 4 0.5 $(COMPARE) CPU \
	$(ADAPT_TOLERANCE_STAGE1) $(ADAPT_TOLERANCE_STAGE2) \
	$(INDICATOR_MODE) $(INDICATOR_ORDER) $(REPORT_MESH_ESTIMATOR)
ROOT_RUN_ARGS ?= D.mesh 2 6 4 0.5 1 CPU
EXACT_GPU_RUN_ARGS ?= D.mesh 2 6 4 0.5 $(COMPARE) $(BACKEND) \
	$(ADAPT_TOLERANCE_STAGE1) $(ADAPT_TOLERANCE_STAGE2) \
	$(INDICATOR_MODE) $(INDICATOR_ORDER) $(REPORT_MESH_ESTIMATOR)
MESH_COMPARE_P ?= 6
MESH_COMPARE_ORDER ?= 4
MESH_COMPARE_THETA ?= 0.5
MESH_COMPARE_THREADS ?= 128
MESH_COMPARE_REPEATS ?= 3
ADAPTIVE_BASE_LEVEL ?= 2
UNIFORM_LEVEL ?= 4
UNIFORM_RUN_ARGS ?= D.mesh $(UNIFORM_LEVEL) $(MESH_COMPARE_P) $(MESH_COMPARE_ORDER) $(MESH_COMPARE_THETA) 1 CPU
UNIFORM_COMPARISON_OUTPUT ?= uniform_adaptive_comparison.tsv
H2O_MESH_MODE ?= SINGLE
H2O_NEAR_FIELD_MODE ?= QUADRATURE
H2O_COMPARE ?= 1
H2O_ADAPT_TOLERANCE ?= 1.0e-3
H2O_INDICATOR_MODE ?= RADIAL
H2O_SINGULAR_MODE ?= CORE_ANALYTIC
H2O_CORE_RADIUS ?= 0.25
H2O_RUN_ARGS ?= D.mesh 2 6 4 0.5 $(H2O_COMPARE) CPU $(H2O_MESH_MODE) $(H2O_NEAR_FIELD_MODE) $(H2O_ADAPT_TOLERANCE) $(H2O_INDICATOR_MODE) $(H2O_SINGULAR_MODE) $(H2O_CORE_RADIUS)
REUSE_STEPS ?= 6
REUSE_DECAY ?= 0.5
REUSE_INITIAL_AMPLITUDE ?= 0.5
REUSE_OUTPUT ?= multirate_reuse_results.tsv
REUSE_RUN_ARGS ?= D.mesh 2 6 4 0.5 1 CPU $(REUSE_STEPS) $(REUSE_DECAY) $(REUSE_INITIAL_AMPLITUDE) $(REUSE_OUTPUT)
MESH_MODE ?= FROZEN
MESH_MODE_OUTPUT ?= mesh_mode_results.tsv
MESH_MODE_RUN_ARGS ?= D.mesh --mode $(MESH_MODE) --initial-refinements 2 --p 6 --P 4 --theta 0.5 --steps 6 --decay 0.5 --initial-amplitude 0.5 --initial-shift 0.8 --period 2 --max-reuse 4 --residual-trigger 2.0e-2 --adapt-tol 1.0e-6 --refine-steps 1 --move-relax 0.2 --move-iters 5 --output $(MESH_MODE_OUTPUT)

AFEPACK_TEMPLATE_PATH := /usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/include/AFEPack/template/twin_tetrahedron/:/usr/local/include/AFEPack/template/four_tetrahedron/
export AFEPACK_TEMPLATE_PATH

.PHONY: all run reference run-reference gpu run-gpu exact run-exact exact-gpu run-exact-gpu uniform run-uniform run-uniform-comparison common-validation plateau h2o run-h2o reuse run-reuse mesh-modes run-mesh-mode run-all-mesh-modes quadrature-test multipole-test clean rebuild

all: $(OUT)

# Exhaustive coverage diagnostics are confined to these TEST executables.
# Normal make run / run-exact / GPU binaries do not compile in the audits.
.PHONY: hnl-verify root-verify
hnl-verify: $(HNL_VERIFY_OUT)
root-verify: $(ROOT_VERIFY_OUT)

.PHONY: hnl-profile
hnl-profile: tree_hnl_profile

tree_hnl_profile: profile_hnl.cpp compare_exact.cpp hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) profile_hnl.cpp -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: hnl-layerwise-requested
hnl-layerwise-requested: tree_hnl_layerwise_requested

tree_hnl_layerwise_requested: compare_layerwise_requested.cpp compare_exact.cpp hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_layerwise_requested.cpp -o $@ $(LDFLAGS) $(LDLIBS)

$(HNL_VERIFY_OUT): compare_exact.cpp hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_HNL_VERIFY compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

$(ROOT_VERIFY_OUT): compare_root.cpp compare_exact.cpp root_based_neighbors.hpp hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_HNL_VERIFY compare_root.cpp -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: root-comparison run-root-comparison
.PHONY: hnl-visualize
hnl-visualize: $(HNL_VISUAL_OUT)

$(HNL_VISUAL_OUT): visualize_hnl.cpp hnl_visualization.hpp compare_exact.cpp hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) visualize_hnl.cpp -o $@ $(LDFLAGS) $(LDLIBS)

root-comparison: $(ROOT_COMPARE_OUT)

run-root-comparison: $(ROOT_COMPARE_OUT)
	./$(ROOT_COMPARE_OUT) $(ROOT_RUN_ARGS)

$(ROOT_COMPARE_OUT): compare_root.cpp compare_exact.cpp root_based_neighbors.hpp hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_root.cpp -o $@ $(LDFLAGS) $(LDLIBS)

# Every executable including compare_exact.cpp must rebuild when the HNL
# implementation changes (including the shared CUDA host object).
$(OUT) $(EXACT_OUT) $(UNIFORM_OUT) $(TREE_REFERENCE_OUT) $(UNIFORM_REFERENCE_OUT) $(PLATEAU_OUT) $(REUSE_OUT) $(MESH_MODES_OUT) $(GPU_HOST_OBJ): hnl_candidates.hpp
$(OUT) $(EXACT_OUT) $(UNIFORM_OUT) $(TREE_REFERENCE_OUT) $(UNIFORM_REFERENCE_OUT) $(PLATEAU_OUT) $(REUSE_OUT) $(MESH_MODES_OUT) $(GPU_HOST_OBJ) $(ROOT_COMPARE_OUT) $(HNL_VERIFY_OUT) $(ROOT_VERIFY_OUT) $(HNL_VISUAL_OUT) tree_hnl_profile tree_hnl_layerwise_requested: mac_geometry.hpp
$(OUT) $(EXACT_OUT) $(UNIFORM_OUT) $(TREE_REFERENCE_OUT) $(UNIFORM_REFERENCE_OUT) $(PLATEAU_OUT) $(REUSE_OUT) $(MESH_MODES_OUT) $(GPU_HOST_OBJ) $(ROOT_COMPARE_OUT) $(HNL_VERIFY_OUT) $(ROOT_VERIFY_OUT) $(HNL_VISUAL_OUT) tree_hnl_profile tree_hnl_layerwise_requested: hnl_family_mac.hpp

$(OUT): $(SRC) local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run: $(OUT)
	./$(OUT) $(RUN_ARGS)

reference: $(REFERENCE_OUT)

$(REFERENCE_OUT): $(REFERENCE_SRC) gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run-reference: $(REFERENCE_OUT)
	./$(REFERENCE_OUT) $(REFERENCE_ARGS)

gpu: $(GPU_OUT)

$(GPU_HOST_OBJ): compare_exact.cpp compare_gpu.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_USE_CUDA -c $< -o $@

$(GPU_KERNEL_OBJ): compare_gpu.cu compare_gpu.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) -c $< -o $@

$(GPU_OUT): $(GPU_HOST_OBJ) $(GPU_KERNEL_OBJ)
	$(MPICXX) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(CUDA_LDFLAGS)

run-gpu: $(GPU_OUT)
	./$(GPU_OUT) $(GPU_RUN_ARGS)

exact: $(EXACT_OUT)

exact-profiles: $(EXACT_DIFFUSE_OUT) $(EXACT_BASELINE_OUT) $(EXACT_SHARP_OUT)

$(EXACT_DIFFUSE_OUT): compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_POSITIVE_GAUSSIAN -DTREECODE_GAUSSIAN_ALPHA=1.0 compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

$(EXACT_BASELINE_OUT): compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_POSITIVE_GAUSSIAN -DTREECODE_GAUSSIAN_ALPHA=4.0 compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

$(EXACT_SHARP_OUT): compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_POSITIVE_GAUSSIAN -DTREECODE_GAUSSIAN_ALPHA=16.0 compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

$(EXACT_OUT): compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

run-exact: $(EXACT_OUT)
	./$(EXACT_OUT) $(EXACT_RUN_ARGS)

exact-gpu: $(EXACT_GPU_OUT)

$(EXACT_GPU_OUT): $(EXACT_GPU_HOST_OBJ) $(GPU_KERNEL_OBJ)
	$(MPICXX) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(CUDA_LDFLAGS)

run-exact-gpu: $(EXACT_GPU_OUT)
	./$(EXACT_GPU_OUT) $(EXACT_GPU_RUN_ARGS)

uniform: $(UNIFORM_OUT)

$(UNIFORM_OUT): compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_UNIFORM_MESH compare_exact.cpp -o $@ $(LDFLAGS) $(LDLIBS)

run-uniform: $(UNIFORM_OUT)
	./$(UNIFORM_OUT) $(UNIFORM_RUN_ARGS)

run-uniform-comparison: $(EXACT_OUT) $(UNIFORM_OUT)
	OMP_NUM_THREADS=$(MESH_COMPARE_THREADS) REPEATS=$(MESH_COMPARE_REPEATS) ADAPTIVE_LEVEL=$(ADAPTIVE_BASE_LEVEL) UNIFORM_LEVEL=$(UNIFORM_LEVEL) P_QUAD=$(MESH_COMPARE_P) P_EXP=$(MESH_COMPARE_ORDER) THETA=$(MESH_COMPARE_THETA) RESULT_FILE=$(UNIFORM_COMPARISON_OUTPUT) ./run_uniform_comparison.sh

$(TREE_REFERENCE_OUT): compare_parallel.cpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_parallel.cpp -o $@ $(LDFLAGS) $(LDLIBS)

$(UNIFORM_REFERENCE_OUT): compare_parallel.cpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_UNIFORM_MESH compare_parallel.cpp -o $@ $(LDFLAGS) $(LDLIBS)

common-validation: $(TREE_REFERENCE_OUT) $(UNIFORM_REFERENCE_OUT)

$(PLATEAU_OUT): compare_plateau.cpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_plateau.cpp -o $@ $(LDFLAGS) $(LDLIBS)

plateau: $(PLATEAU_OUT)

# Independent error-component and target-sampling checks; never appends
# validation timings to the production results.txt.
.PHONY: error-validation
error-validation: tree_error_validation

tree_error_validation: compare_error_validation.cpp error_validation.hpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp hnl_candidates.hpp hnl_family_mac.hpp mac_geometry.hpp hnl_visualization.hpp
	$(MPICXX) $(CXXFLAGS) compare_error_validation.cpp -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: uniform-error-validation
uniform-error-validation: tree_uniform_error_validation

tree_uniform_error_validation: compare_uniform_error_validation.cpp compare_error_validation.cpp error_validation.hpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp hnl_candidates.hpp hnl_family_mac.hpp mac_geometry.hpp hnl_visualization.hpp
	$(MPICXX) $(CXXFLAGS) compare_uniform_error_validation.cpp -o $@ $(LDFLAGS) $(LDLIBS)


h2o: $(H2O_OUT)

$(H2O_OUT): compare_h2o.cpp h2o_three_center_test_problem.hpp gaussian_test_problem.hpp local_quadrature.hpp cartesian_multipole.hpp hnl_candidates.hpp
	$(MPICXX) $(CXXFLAGS) compare_h2o.cpp -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: h2o-verify
h2o-verify: tree_h2o_hnl_verify
tree_h2o_hnl_verify: compare_h2o.cpp h2o_three_center_test_problem.hpp gaussian_test_problem.hpp local_quadrature.hpp cartesian_multipole.hpp hnl_candidates.hpp
	$(MPICXX) $(CXXFLAGS) -DTREECODE_HNL_VERIFY compare_h2o.cpp -o $@ $(LDFLAGS) $(LDLIBS)

run-h2o: $(H2O_OUT)
	./$(H2O_OUT) $(H2O_RUN_ARGS)

.PHONY: h2o-fine-validation
h2o-fine-validation: tree_h2o_fine_validation

tree_h2o_fine_validation: compare_h2o_fine_validation.cpp compare_h2o.cpp hnl_candidates.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp h2o_three_center_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

reuse: $(REUSE_OUT)

$(REUSE_OUT): compare_reuse.cpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_reuse.cpp -o $@ $(LDFLAGS) $(LDLIBS)

run-reuse: $(REUSE_OUT)
	./$(REUSE_OUT) $(REUSE_RUN_ARGS)

mesh-modes: $(MESH_MODES_OUT)

$(MESH_MODES_OUT): compare_mesh_modes.cpp compare_exact.cpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) compare_mesh_modes.cpp -o $@ $(LDFLAGS) $(LDLIBS)

run-mesh-mode: $(MESH_MODES_OUT)
	./$(MESH_MODES_OUT) $(MESH_MODE_RUN_ARGS)

run-all-mesh-modes: $(MESH_MODES_OUT)
	@for mode in FROZEN EVERY_H PERIODIC_H RESIDUAL_H MOVING_R HYBRID_HR; do \
	  echo "Running $$mode"; \
	  ./$(MESH_MODES_OUT) D.mesh --mode $$mode --initial-refinements 2 --p 6 --P 4 --theta 0.5 --steps 6 --decay 0.5 --initial-amplitude 0.5 --initial-shift 0.8 --period 2 --max-reuse 4 --residual-trigger 2.0e-2 --adapt-tol 1.0e-6 --refine-steps 1 --move-relax 0.2 --move-iters 5 --output mesh_mode_$$mode.tsv || exit $$?; \
	done

$(QUADRATURE_TEST_OUT): quadrature_test.cpp local_quadrature.hpp
	$(MPICXX) $(CXXFLAGS) quadrature_test.cpp -o $@ $(LDFLAGS) $(LDLIBS)

quadrature-test: $(QUADRATURE_TEST_OUT)
	./$(QUADRATURE_TEST_OUT) 20

$(MULTIPOLE_TEST_OUT): multipole_test.cpp cartesian_multipole.hpp
	$(MPICXX) $(CXXFLAGS) multipole_test.cpp -o $@

multipole-test: $(MULTIPOLE_TEST_OUT)
	./$(MULTIPOLE_TEST_OUT)

$(HNL_TEST_OUT): hnl_test.cpp hnl_candidates.hpp hnl_original_constructor_reference.hpp hnl_tetra_test_fixture.hpp
	$(MPICXX) $(CXXFLAGS) hnl_test.cpp -o $@

.PHONY: hnl-test
hnl-test: $(HNL_TEST_OUT)
	./$(HNL_TEST_OUT)

.PHONY: mac-geometry-test
mac-geometry-test: mac_geometry_test
	./mac_geometry_test

mac_geometry_test: mac_geometry_test.cpp mac_geometry.hpp
	$(MPICXX) $(CXXFLAGS) mac_geometry_test.cpp -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: hnl-family-test streaming-test
hnl-family-test: hnl_family_mac_test
	./hnl_family_mac_test

hnl_family_mac_test: hnl_family_mac_test.cpp hnl_family_mac.hpp hnl_candidates.hpp hnl_tetra_test_fixture.hpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

streaming-test: streaming_interaction_test
	OMP_NUM_THREADS=1 ./streaming_interaction_test

streaming_interaction_test: streaming_interaction_test.cpp compare_exact.cpp hnl_candidates.hpp hnl_family_mac.hpp mac_geometry.hpp hnl_tetra_test_fixture.hpp local_quadrature.hpp cartesian_multipole.hpp gaussian_test_problem.hpp
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

rebuild:
	$(MAKE) --always-make $(OUT)

clean:
	rm -f $(OUT) $(REFERENCE_OUT) $(GPU_OUT) $(EXACT_OUT) $(EXACT_GPU_OUT) $(UNIFORM_OUT) $(TREE_REFERENCE_OUT) $(UNIFORM_REFERENCE_OUT) $(PLATEAU_OUT) $(H2O_OUT) $(REUSE_OUT) $(MESH_MODES_OUT) $(QUADRATURE_TEST_OUT) $(MULTIPOLE_TEST_OUT) $(HNL_TEST_OUT) $(GPU_HOST_OBJ) $(EXACT_GPU_HOST_OBJ) $(GPU_KERNEL_OBJ)
