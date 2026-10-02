#!/usr/bin/env bash
set -u

threads="${OMP_NUM_THREADS:-8}"
results_dir="experiment_results"
requested_study="${STUDY:-all}"
case "${requested_study}" in
  all|tol|n|p|P) ;;
  *)
    echo "STUDY must be one of: all, tol, n, p, P" >&2
    exit 2
    ;;
esac
if [[ "${requested_study}" == "all" ]]; then
  csv_file="${results_dir}/three_parameter_sweeps.csv"
else
  csv_file="${results_dir}/${requested_study}_sweep.csv"
fi
export AFEPACK_TEMPLATE_PATH="/usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/include/AFEPack/template/twin_tetrahedron/:/usr/local/include/AFEPack/template/four_tetrahedron/"

mkdir -p "${results_dir}"
printf '%s\n' \
  'study,case,p,P,n,theta,tol1,order1,steps1,tol2,order2,steps2,direct_reference,indicator_p,mesh_only,gaussian_ax,gaussian_ay,gaussian_az,elements,quadrature_points,relative_exact_error,relative_tree_direct_error,relative_direct_exact_error,evaluation_seconds,status' \
  > "${csv_file}"

extract_last_field()
{
  local pattern="$1"
  local file="$2"
  awk -F': ' -v pattern="${pattern}" \
    'index($0, pattern) == 1 {value=$NF} END {print value}' "${file}"
}

run_case()
{
  local study="$1"
  local case_name="$2"
  local p="$3"
  local P="$4"
  local n="$5"
  local theta="$6"
  local tol1="$7"
  local order1="$8"
  local steps1="$9"
  local tol2="${10}"
  local order2="${11}"
  local steps2="${12}"
  local direct_reference="${13}"
  local indicator_p="${14}"
  local mesh_only="${15}"
  local gaussian_ax="${16}"
  local gaussian_ay="${17}"
  local gaussian_az="${18}"
  local log_file="${results_dir}/${study}_${case_name}.log"

  echo "Running ${study}/${case_name}: p=${p}, P=${P}, n=${n}, theta=${theta}"
  OMP_NUM_THREADS="${threads}" ./tree \
    D.mesh "${p}" "${P}" "${n}" "${theta}" \
    "${tol1}" "${order1}" "${steps1}" \
    "${tol2}" "${order2}" "${steps2}" \
    "${direct_reference}" "${indicator_p}" "${mesh_only}" \
    "${gaussian_ax}" "${gaussian_ay}" "${gaussian_az}" \
    > "${log_file}" 2>&1
  local status=$?

  local elements
  local quadrature_points
  local relative_exact_error
  local relative_tree_direct_error
  local relative_direct_exact_error
  local evaluation_seconds
  elements="$(awk \
    '/^Final leaf elements:/ {value=$4} \
     /^Validated [0-9]+ exact leaf partitions/ {value=$2} \
     END {print value}' \
    "${log_file}" | tail -n 1)"
  quadrature_points="$(extract_last_field \
    'Exact-comparison quadrature points' "${log_file}")"
  relative_exact_error="$(extract_last_field \
    'Relative L2 error against exact solution' "${log_file}")"
  relative_tree_direct_error="$(extract_last_field \
    'Relative L2 treecode error against direct solution' "${log_file}")"
  relative_direct_exact_error="$(extract_last_field \
    'Relative L2 direct-solution error against exact solution' "${log_file}")"
  evaluation_seconds="$(extract_last_field \
    'Exact-comparison evaluation time' "${log_file}" | awk '{print $1}')"

  printf '%s\n' \
    "${study},${case_name},${p},${P},${n},${theta},${tol1},${order1},${steps1},${tol2},${order2},${steps2},${direct_reference},${indicator_p},${mesh_only},${gaussian_ax},${gaussian_ay},${gaussian_az},${elements},${quadrature_points},${relative_exact_error},${relative_tree_direct_error},${relative_direct_exact_error},${evaluation_seconds},${status}" \
    >> "${csv_file}"

  if (( status != 0 )); then
    echo "FAILED ${study}/${case_name}; see ${log_file}"
  elif (( mesh_only == 1 )); then
    echo "Finished ${study}/${case_name}: leaf elements=${elements}"
  else
    echo "Finished ${study}/${case_name}: relative exact error=${relative_exact_error}"
  fi
}

# Preliminary study: hold the first tolerance at 1e-3 and vary only the
# second adaptive-stage tolerance.
if [[ "${requested_study}" == "all" || "${requested_study}" == "tol" ]]; then
  for tol2 in 0.01 0.003 0.001 0.0008 0.0005; do
    case_name="tol2_${tol2/./p}"
    run_case tol "${case_name}" 1 1 0 0.8 \
      0.001 0 5 "${tol2}" 0 2 0 4 0 4 8 12
  done
fi

# Study 1: h-convergence through uniform global refinement.  Setting both
# adaptation step counts to zero keeps the sequence genuinely uniform:
# 24*8^n = 192, 1536, 12288, and 98304 tetrahedra for n=1,2,3,4.
if [[ "${requested_study}" == "all" || "${requested_study}" == "n" ]]; then
  for n in 1 2 3 4; do
    run_case n "n${n}" 2 8 "${n}" 0.4 \
      0.001 0 0 0.001 0 0 0 4 0 4 8 12
  done
fi

# Study 2: quadrature algebraic accuracy on the same approximately
# 100,000-leaf adaptive mesh.  p=1..4 is the practical CPU list; higher
# values should be run with the GPU target because quadrature cost grows
# rapidly.
if [[ "${requested_study}" == "all" || "${requested_study}" == "p" ]]; then
  for p in 1 2 3 4; do
    run_case p "p${p}" "${p}" 12 0 0.3 \
      0.001 0 5 0.001 0 2 0 4 0 4 8 12
  done
fi

# Study 3: multipole-order convergence on the same adaptive mesh.  p, theta,
# tolerances, and adaptation depths remain fixed.
if [[ "${requested_study}" == "all" || "${requested_study}" == "P" ]]; then
  for P in 1 2 3 4 6 8 10 12; do
    run_case P "P${P}" 2 "${P}" 0 0.6 \
      0.001 0 5 0.001 0 2 0 4 0 4 8 12
  done
fi

echo "Results written to ${csv_file}"
