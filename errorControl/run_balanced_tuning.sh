#!/usr/bin/env bash
set -euo pipefail

mesh_file="${MESH_FILE:-D.mesh}"
executable="${EXECUTABLE:-./tree_exact}"
threads="${OMP_NUM_THREADS:-128}"
result_file="${RESULT_FILE:-balanced_parameter_screen.tsv}"
selection_file="${SELECTION_FILE:-balanced_parameter_selection.tsv}"
log_dir="${LOG_DIR:-experiment_results/balanced_parameter_screen}"
error_target="${ERROR_TARGET:-1.0e-4}"
resume="${RESUME:-0}"

export AFEPACK_TEMPLATE_PATH="${AFEPACK_TEMPLATE_PATH:-/usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/include/AFEPack/template/twin_tetrahedron/:/usr/local/include/AFEPack/template/four_tetrahedron/}"

mkdir -p "${log_dir}"
if [[ "${resume}" != 1 || ! -s "${result_file}" ]]; then
printf '%s\n' $'phase\tscale\tp\tP\ttheta\ttolerance_stage1\ttolerance_stage2\tn_elements\tT_list_s\tT_moment_s\tT_eval_s\tT_total_s\tL2_error\trelative_L2_error\tthreads\tstatus\tlog_file' > "${result_file}"
fi

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
  local phase="$1"
  local scale="$2"
  local tolerance_stage1="$3"
  local tolerance_stage2="$4"
  local p_quad="$5"
  local p_exp="$6"
  local theta="$7"
  local theta_tag="${theta//./p}"
  local log_file="${log_dir}/${phase}_${scale}_p${p_quad}_P${p_exp}_theta${theta_tag}.log"

  if awk -F'\t' -v phase="${phase}" -v scale="${scale}" \
      -v p="${p_quad}" -v P="${p_exp}" -v theta="${theta}" \
      '$1 == phase && $2 == scale && $3 == p && $4 == P && $5 == theta && $16 == 0 {found=1} END {exit !found}' \
      "${result_file}"; then
    echo "Skipping completed ${phase} ${scale}: p=${p_quad}, P=${p_exp}, theta=${theta}"
    return
  fi

  echo "Running ${phase} ${scale}: p=${p_quad}, P=${p_exp}, theta=${theta}, tolerances=${tolerance_stage1}/${tolerance_stage2}"
  set +e
  OMP_NUM_THREADS="${threads}" "${executable}" \
    "${mesh_file}" 0 "${p_quad}" "${p_exp}" "${theta}" 1 CPU \
    "${tolerance_stage1}" "${tolerance_stage2}" \
    > "${log_file}" 2>&1
  local status=$?
  set -e

  local n_elements="NA"
  local t_list="NA"
  local t_moment="NA"
  local t_eval="NA"
  local t_total="NA"
  local l2_error="NA"
  local relative_l2_error="NA"
  if (( status == 0 )); then
    n_elements="$(extract_colon_value 'Exact-comparison leaf barycenters' "${log_file}")"
    t_list="$(awk '/^Filling neighbor_elements uses:/ {value=$(NF-1)} END {print value}' "${log_file}")"
    t_moment="$(awk '/^Updating multipoleInfo uses / {value=$(NF-1)} END {print value}' "${log_file}")"
    t_eval="$(extract_colon_value 'Potential evaluation time' "${log_file}")"
    t_total="$(extract_colon_value 'Total execution time' "${log_file}")"
    l2_error="$(extract_colon_value 'L2 error against exact solution' "${log_file}")"
    relative_l2_error="$(extract_colon_value 'Relative L2 error against exact solution' "${log_file}")"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${phase}" "${scale}" "${p_quad}" "${p_exp}" "${theta}" \
    "${tolerance_stage1}" "${tolerance_stage2}" "${n_elements}" \
    "${t_list}" "${t_moment}" "${t_eval}" "${t_total}" \
    "${l2_error}" "${relative_l2_error}" "${threads}" "${status}" \
    "${log_file}" >> "${result_file}"

  if (( status != 0 )); then
    echo "FAILED; see ${log_file}" >&2
    tail -n 30 "${log_file}" >&2
    return "${status}"
  fi
  echo "Finished ${scale}: n=${n_elements}, error=${l2_error}, total=${t_total} s"
}

field_for_case()
{
  local phase="$1"
  local scale="$2"
  local p_quad="$3"
  local p_exp="$4"
  local theta="$5"
  local column="$6"
  awk -F'\t' -v phase="${phase}" -v scale="${scale}" \
      -v p="${p_quad}" -v P="${p_exp}" -v theta="${theta}" \
      -v column="${column}" \
    '$1 == phase && $2 == scale && $3 == p && $4 == P && $5 == theta && $16 == 0 {value=$column} END {print value}' \
    "${result_file}"
}

improvement_exceeds_ten_percent()
{
  local former="$1"
  local latter="$2"
  awk -v former="${former}" -v latter="${latter}" \
    'BEGIN {exit !(former > 0.0 && latter < 0.9 * former)}'
}

error_meets_target()
{
  local error="$1"
  awk -v error="${error}" -v target="${error_target}" \
    'BEGIN {exit !(error != "" && error+0.0 <= target+0.0)}'
}

choose_p()
{
  local scale="$1"
  awk -F'\t' -v scale="${scale}" -v target="${error_target}" '
    $1 == "p_screen" && $2 == scale && $16 == 0 {
      error[$3]=$13
      if (!have_min || $13 < minimum) {minimum=$13; have_min=1}
    }
    END {
      chosen=1000000
      for (p in error) {
        if (error[p] <= target && p+0 < chosen) chosen=p+0
      }
      if (chosen == 1000000) {
        threshold=1.2*minimum
        for (p in error) {
          if (error[p] <= threshold && p+0 < chosen) chosen=p+0
        }
      }
      if (chosen < 1000000) print chosen
    }' "${result_file}"
}

declare -a scales=(L2 L3 L4 L5)
declare -a tolerances_stage1=(1.0 0.1 0.01 0.001)
declare -a tolerances_stage2=(0.1 0.01 0.001 0.0001)

# Phase 1: select the lowest quadrature accuracy within 20% of the best
# screened error. P=8 and theta=0.5 suppress most far-field contamination.
declare -A selected_p
for index in "${!scales[@]}"; do
  scale="${scales[index]}"
  tolerance_stage1="${tolerances_stage1[index]}"
  tolerance_stage2="${tolerances_stage2[index]}"
  for p_quad in 2 4 6 8; do
    run_case p_screen "${scale}" "${tolerance_stage1}" \
      "${tolerance_stage2}" "${p_quad}" 8 0.5
  done

  error_p6="$(field_for_case p_screen "${scale}" 6 8 0.5 13)"
  error_p8="$(field_for_case p_screen "${scale}" 8 8 0.5 13)"
  if ! error_meets_target "${error_p8}" && \
      improvement_exceeds_ten_percent "${error_p6}" "${error_p8}"; then
    run_case p_screen "${scale}" "${tolerance_stage1}" \
      "${tolerance_stage2}" 12 8 0.5
    error_p12="$(field_for_case p_screen "${scale}" 12 8 0.5 13)"
    if ! error_meets_target "${error_p12}" && \
        improvement_exceeds_ten_percent "${error_p8}" "${error_p12}"; then
      run_case p_screen "${scale}" "${tolerance_stage1}" \
        "${tolerance_stage2}" 16 8 0.5
    fi
  fi
  selected_p["${scale}"]="$(choose_p "${scale}")"
  echo "Selected p=${selected_p[${scale}]} for ${scale} screening"
done

# Phase 2: a coupled P/theta screen. Extreme theta=0.1 was eliminated by the
# existing medium-mesh experiment because it was direct-interaction dominated.
declare -a coupled_candidates=(
  "2 0.9" "2 0.7" "2 0.5"
  "4 0.7" "4 0.5" "4 0.3"
  "6 0.7" "6 0.5" "6 0.3"
  "8 0.7" "8 0.5")

for index in "${!scales[@]}"; do
  scale="${scales[index]}"
  tolerance_stage1="${tolerances_stage1[index]}"
  tolerance_stage2="${tolerances_stage2[index]}"
  p_quad="${selected_p[${scale}]}"
  for candidate in "${coupled_candidates[@]}"; do
    read -r p_exp theta <<< "${candidate}"
    run_case coupled_screen "${scale}" "${tolerance_stage1}" \
      "${tolerance_stage2}" "${p_quad}" "${p_exp}" "${theta}"
  done

  error_P6="$(field_for_case coupled_screen "${scale}" "${p_quad}" 6 0.5 13)"
  error_P8="$(field_for_case coupled_screen "${scale}" "${p_quad}" 8 0.5 13)"
  if ! error_meets_target "${error_P8}" && \
      improvement_exceeds_ten_percent "${error_P6}" "${error_P8}"; then
    run_case coupled_screen "${scale}" "${tolerance_stage1}" \
      "${tolerance_stage2}" "${p_quad}" 12 0.5
    error_P12="$(field_for_case coupled_screen "${scale}" "${p_quad}" 12 0.5 13)"
    if ! error_meets_target "${error_P12}" && \
        improvement_exceeds_ten_percent "${error_P8}" "${error_P12}"; then
      run_case coupled_screen "${scale}" "${tolerance_stage1}" \
        "${tolerance_stage2}" "${p_quad}" 16 0.5
    fi
  fi
done

# Choose the minimum-total-time coupled case whose native-grid error is no
# larger than the requested target, with a 20%-of-best fallback.
printf '%s\n' $'scale\tp\tP\ttheta\tn_elements\tT_list_s\tT_moment_s\tT_eval_s\tT_total_s\tL2_error\trelative_L2_error\terror_threshold' > "${selection_file}"
for scale in "${scales[@]}"; do
  awk -F'\t' -v OFS='\t' -v scale="${scale}" -v target="${error_target}" '
    $1 == "coupled_screen" && $2 == scale && $16 == 0 {
      rows[++count]=$0
      if (!have_min || $13 < minimum) {minimum=$13; have_min=1}
    }
    END {
      have_target=0
      for (i=1; i<=count; ++i) {
        split(rows[i], field, "\t")
        if (field[13] <= target) have_target=1
      }
      threshold=have_target ? target : 1.2*minimum
      best_time=1.0e300
      for (i=1; i<=count; ++i) {
        split(rows[i], field, "\t")
        if (field[13] <= threshold && field[12] < best_time) {
          best_time=field[12]
          best=rows[i]
        }
      }
      split(best, field, "\t")
      print field[2],field[3],field[4],field[5],field[8],field[9],field[10],field[11],field[12],field[13],field[14],threshold
    }' "${result_file}" >> "${selection_file}"
done

echo "Raw staged results: ${result_file}"
echo "Selected balanced configurations: ${selection_file}"
