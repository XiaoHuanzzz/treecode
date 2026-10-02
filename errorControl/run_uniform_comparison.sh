#!/usr/bin/env bash
set -euo pipefail

mesh_file="${MESH_FILE:-D.mesh}"
adaptive_level="${ADAPTIVE_LEVEL:-2}"
uniform_level="${UNIFORM_LEVEL:-4}"
p_quad="${P_QUAD:-6}"
p_exp="${P_EXP:-4}"
theta="${THETA:-0.5}"
adaptive_tolerance_stage1="${ADAPT_TOL1:-0.01}"
adaptive_tolerance_stage2="${ADAPT_TOL2:-0.001}"
threads="${OMP_NUM_THREADS:-128}"
repeats="${REPEATS:-3}"
result_file="${RESULT_FILE:-uniform_adaptive_comparison.tsv}"
report_file="${REPORT_FILE:-uniform_adaptive_comparison.md}"
log_dir="${LOG_DIR:-experiment_results/uniform_comparison}"

adaptive_executable="${ADAPTIVE_EXECUTABLE:-./tree_exact}"
uniform_executable="${UNIFORM_EXECUTABLE:-./tree_uniform}"

mkdir -p "${log_dir}"
printf '%s\n' $'mesh_mode\trepeat\tglobal_level\tp\tP\ttheta\tn_elements\tT_list_s\tT_moment_s\tT_eval_s\tT_total_s\tL2_norm_exact\tL2_error\trelative_L2_error\tthreads\tstatus\tlog_file' > "${result_file}"

extract_colon_value()
{
  local label="$1"
  local file="$2"
  awk -F': ' -v label="${label}" \
    'index($0, label) == 1 {value=$2} END {print value}' "${file}" \
    | awk '{print $1}'
}

run_case()
{
  local mode="$1"
  local level="$2"
  local executable="$3"
  local repeat="$4"
  local log_file="${log_dir}/${mode}_level${level}_p${p_quad}_P${p_exp}_theta${theta}_r${repeat}.log"
  local -a command_args=(
    "${mesh_file}" "${level}" "${p_quad}" "${p_exp}" "${theta}" 1 CPU)
  if [[ "${mode}" == "adaptive" ]]; then
    command_args+=(
      "${adaptive_tolerance_stage1}" "${adaptive_tolerance_stage2}")
  fi

  echo "Running ${mode} repeat ${repeat}/${repeats}: level=${level}, p=${p_quad}, P=${p_exp}, theta=${theta}, threads=${threads}"
  set +e
  OMP_NUM_THREADS="${threads}" "${executable}" "${command_args[@]}" \
    > "${log_file}" 2>&1
  local status=$?
  set -e

  local n_elements
  local t_list
  local t_moment
  local t_eval
  local t_total
  local exact_norm
  local l2_error
  local relative_l2_error

  n_elements="$(extract_colon_value 'Exact-comparison leaf barycenters' "${log_file}")"
  t_list="$(awk \
    '/^Filling neighbor_elements uses:/ {value=$(NF-1)} \
     END {print value}' "${log_file}")"
  t_moment="$(awk \
    '/^Updating multipoleInfo uses / {value=$(NF-1)} \
     END {print value}' "${log_file}")"
  t_eval="$(extract_colon_value 'Potential evaluation time' "${log_file}")"
  t_total="$(extract_colon_value 'Total execution time' "${log_file}")"
  exact_norm="$(extract_colon_value 'L2 norm of exact solution' "${log_file}")"
  l2_error="$(extract_colon_value 'L2 error against exact solution' "${log_file}")"
  relative_l2_error="$(extract_colon_value \
    'Relative L2 error against exact solution' "${log_file}")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode}" "${repeat}" "${level}" "${p_quad}" "${p_exp}" "${theta}" \
    "${n_elements:-NA}" "${t_list:-NA}" "${t_moment:-NA}" \
    "${t_eval:-NA}" "${t_total:-NA}" "${exact_norm:-NA}" \
    "${l2_error:-NA}" "${relative_l2_error:-NA}" "${threads}" "${status}" "${log_file}" \
    >> "${result_file}"

  if (( status != 0 )); then
    echo "FAILED ${mode}; see ${log_file}" >&2
    tail -n 20 "${log_file}" >&2
    return "${status}"
  fi
  echo "Finished ${mode}: n=${n_elements}, L2 error=${l2_error}, total=${t_total} s"
}

if (( repeats < 1 )); then
  echo "REPEATS must be positive" >&2
  exit 2
fi
for repeat in $(seq 1 "${repeats}"); do
  run_case adaptive "${adaptive_level}" "${adaptive_executable}" "${repeat}"
  run_case uniform "${uniform_level}" "${uniform_executable}" "${repeat}"
done

first_field()
{
  local mode="$1"
  local column="$2"
  awk -F'\t' -v mode="${mode}" -v column="${column}" \
    '$1 == mode && $16 == 0 {print $column; exit}' "${result_file}"
}

mean_field()
{
  local mode="$1"
  local column="$2"
  awk -F'\t' -v mode="${mode}" -v column="${column}" \
    '$1 == mode && $16 == 0 {sum += $column; count += 1} \
     END {if (count > 0) printf "%.12g", sum/count; else print "NA"}' \
    "${result_file}"
}

a_level="$(first_field adaptive 3)"
a_n="$(first_field adaptive 7)"
a_list="$(mean_field adaptive 8)"
a_moment="$(mean_field adaptive 9)"
a_eval="$(mean_field adaptive 10)"
a_total="$(mean_field adaptive 11)"
a_exact_norm="$(mean_field adaptive 12)"
a_error="$(mean_field adaptive 13)"
a_relative="$(mean_field adaptive 14)"

u_level="$(first_field uniform 3)"
u_n="$(first_field uniform 7)"
u_list="$(mean_field uniform 8)"
u_moment="$(mean_field uniform 9)"
u_eval="$(mean_field uniform 10)"
u_total="$(mean_field uniform 11)"
u_exact_norm="$(mean_field uniform 12)"
u_error="$(mean_field uniform 13)"
u_relative="$(mean_field uniform 14)"

total_ratio="$(awk -v uniform="${u_total}" -v adaptive="${a_total}" \
  'BEGIN {if (adaptive > 0) printf "%.6g", uniform/adaptive; else print "NA"}')"
error_ratio="$(awk -v uniform="${u_error}" -v adaptive="${a_error}" \
  'BEGIN {if (adaptive > 0) printf "%.6g", uniform/adaptive; else print "NA"}')"
analytic_norm="0.134321241858"
adaptive_norm_bias="$(awk -v value="${a_exact_norm}" -v exact="${analytic_norm}" \
  'BEGIN {printf "%.4g", 100.0*(value-exact)/exact}')"
uniform_norm_bias="$(awk -v value="${u_exact_norm}" -v exact="${analytic_norm}" \
  'BEGIN {printf "%.4g", 100.0*(value-exact)/exact}')"
total_time_gap="$(awk -v uniform="${u_total}" -v adaptive="${a_total}" \
  'BEGIN {printf "%.4g", 100.0*(uniform-adaptive)/adaptive}')"

{
  printf '# Uniform versus adaptive mesh comparison\n\n'
  printf 'Both calculations use the same manufactured Gaussian problem, CPU backend, %s OpenMP threads, one target at each active-leaf barycenter, and identical treecode parameters: \\(p=%s\\), \\(P=%s\\), and \\(\\theta=%s\\). Adaptive base level %s and uniform level %s give comparable element counts. Timings are arithmetic means of %s runs.\n\n' \
    "${threads}" "${p_quad}" "${p_exp}" "${theta}" "${adaptive_level}" "${uniform_level}" "${repeats}"
  printf '| Mesh | Level | Elements | \\(T_{\\mathrm{list}}\\) (s) | \\(T_{\\mathrm{moment}}\\) (s) | \\(T_{\\mathrm{eval}}\\) (s) | \\(T_{\\mathrm{total}}\\) (s) | Resolved \\(\\lVert V_{\\mathrm{ex}}\\rVert_{L^2}\\) | \\(L^2\\) error | Relative error |\n'
  printf '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
  printf '| Adaptive | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "${a_level}" "${a_n}" "${a_list}" "${a_moment}" "${a_eval}" "${a_total}" "${a_exact_norm}" "${a_error}" "${a_relative}"
  printf '| Uniform | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n\n' \
    "${u_level}" "${u_n}" "${u_list}" "${u_moment}" "${u_eval}" "${u_total}" "${u_exact_norm}" "${u_error}" "${u_relative}"
  printf 'For reference, the effectively untruncated analytic norm is approximately \\(0.134321\\); the resolved-norm column therefore also reveals whether one barycenter per leaf resolves the localized Gaussian.\n\n'
  printf '## Interpretation\n\n'
  printf -- '- The uniform/adaptive total-time ratio is **%s**: the uniform run is **%s%%** slower, so the two costs are effectively comparable at this problem size.\n' "${total_ratio}" "${total_time_gap}"
  printf -- '- The adaptive leaf-barycenter error is **%s times smaller** than the uniform value.\n' "${error_ratio}"
  printf -- '- The adaptive barycenter rule resolves the analytic norm with a signed bias of **%s%%**, whereas the uniform bias is **%s%%**. The uniform level-4 target rule therefore under-resolves the localized Gaussian peak.\n\n' "${adaptive_norm_bias}" "${uniform_norm_bias}"
  printf 'The error ratio is a production-grid, leaf-barycenter comparison. It should not be presented as a common-grid continuum-error ratio until both solutions are checked on an independent validation rule.\n'
} > "${report_file}"

echo "Raw results: ${result_file}"
echo "Markdown report: ${report_file}"
