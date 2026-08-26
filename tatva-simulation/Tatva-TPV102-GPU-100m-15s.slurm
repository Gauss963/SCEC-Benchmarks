#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=TPV102-100m-gpu1
#SBATCH --output=/work/gauss112/SCEC-Code-Verification/tpv102-gpu-100m-%j.out
#SBATCH --error=/work/gauss112/SCEC-Code-Verification/tpv102-gpu-100m-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT="${TATVA_SIM_ROOT:-/work/gauss112/SCEC-Code-Verification/SCEC-Code-Validation/tatva-simulation}"
# Resolved from the installed package below unless the caller sets it.
TATVA_ROOT="${TATVA_ROOT:-}"
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
RUN_NAME="${TPV102_RUN_NAME:-tpv102_100m_15s_gpu1_h200}"
PLOT_NAME="${TPV102_PLOT_NAME:-tpv102_validation_uguca_specfem3d_100m_15s_gpu1_h200}"
OPERATOR_BATCH_SIZE="${OPERATOR_BATCH_SIZE:-65536}"
OUTPUT="$ROOT/output/$RUN_NAME"
CHECKPOINT="$ROOT/output/${RUN_NAME}_checkpoint.npz"
UGUCA_REFERENCE="$ROOT/reference/tpv102/ke_100m"
SPECFEM3D_REFERENCE="$ROOT/reference/tpv102/kaneko_100m"
PLOT="$ROOT/Plot/$PLOT_NAME"
STATUS="$ROOT/output/${RUN_NAME}_status.json"
GPU_METRICS="$ROOT/output/${RUN_NAME}_nvidia_smi.csv"
GPU_SUMMARY="$ROOT/output/${RUN_NAME}_gpu_summary.json"
BASELINE_OUTPUT="${BASELINE_OUTPUT:-}"
BASELINE_LABEL="${BASELINE_LABEL:-Tatva original domain 100 m}"
CURRENT_LABEL="${CURRENT_LABEL:-Tatva expanded/current 100 m H200}"

domain_args=()
if [[ -n "${X_MIN:-}" || -n "${X_MAX:-}" || -n "${Y_MIN:-}" || -n "${Y_MAX:-}" ]]; then
  [[ -n "${X_MIN:-}" && -n "${X_MAX:-}" && -n "${Y_MIN:-}" && -n "${Y_MAX:-}" ]] || {
    echo "X_MIN, X_MAX, Y_MIN, and Y_MAX must be set together." >&2
    exit 1
  }
  domain_args=(--x-min "$X_MIN" --x-max "$X_MAX" --y-min "$Y_MIN" --y-max "$Y_MAX")
fi
if [[ -n "${Z_EXTENT:-}" ]]; then
  domain_args+=(--z-extent "$Z_EXTENT")
fi

mesh_args=()
if [[ "${GRADED_MESH:-0}" == "1" ]]; then
  for variable in FINE_X_MIN FINE_X_MAX FINE_Y_MIN FINE_Y_MAX FINE_Z_EXTENT MAX_MESH_SIZE MESH_GROWTH_RATIO; do
    [[ -n "${!variable:-}" ]] || { echo "$variable is required for GRADED_MESH=1." >&2; exit 1; }
  done
  mesh_args=(
    --graded-mesh
    --fine-x-min "$FINE_X_MIN" --fine-x-max "$FINE_X_MAX"
    --fine-y-min "$FINE_Y_MIN" --fine-y-max "$FINE_Y_MAX"
    --fine-z-extent "$FINE_Z_EXTENT"
    --max-mesh-size "$MAX_MESH_SIZE"
    --mesh-growth-ratio "$MESH_GROWTH_RATIO"
  )
fi

dump_args=()
if [[ -n "${UGUCA_DUMP_INTERVAL:-}" ]]; then
  dump_args=(--uguca-dump-interval "$UGUCA_DUMP_INTERVAL")
  [[ -n "${UGUCA_DUMP_DIR:-}" ]] && dump_args+=(--uguca-dump-dir "$UGUCA_DUMP_DIR")
  [[ -n "${UGUCA_DUMP_NAME:-}" ]] && dump_args+=(--uguca-dump-name "$UGUCA_DUMP_NAME")
  [[ -n "${UGUCA_DUMP_X_MIN:-}" ]] && dump_args+=(--uguca-dump-x-min "$UGUCA_DUMP_X_MIN")
  [[ -n "${UGUCA_DUMP_X_MAX:-}" ]] && dump_args+=(--uguca-dump-x-max "$UGUCA_DUMP_X_MAX")
  [[ -n "${UGUCA_DUMP_Y_MIN:-}" ]] && dump_args+=(--uguca-dump-y-min "$UGUCA_DUMP_Y_MIN")
  [[ -n "${UGUCA_DUMP_Y_MAX:-}" ]] && dump_args+=(--uguca-dump-y-max "$UGUCA_DUMP_Y_MAX")
fi

module purge
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
if [[ -z "$TATVA_ROOT" ]]; then
  TATVA_ROOT=$("$PYTHON" -c 'import pathlib, tatva; print(pathlib.Path(tatva.__file__).resolve().parent.parent)')
fi
[[ -f "$TATVA_ROOT/tests/test_friction.py" ]] || {
  echo "Missing Tatva tests: $TATVA_ROOT/tests/test_friction.py" >&2
  exit 1
}
echo "Tatva checkout: $TATVA_ROOT"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEMORY_FRACTION:-0.94}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset LD_LIBRARY_PATH || true

mkdir -p "$ROOT/output" "$ROOT/Plot"
cd "$ROOT"

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_epoch=$(date +%s)
simulation_exit=null
validation_exit=null
comparison_exit=null
plot_exit=null
tail_analysis_exit=null
monitor_pid=

write_status() {
  local exit_code=$?
  local finished_at
  local elapsed_seconds
  finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  elapsed_seconds=$(( $(date +%s) - start_epoch ))
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
  cat > "$STATUS" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-unknown}",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "elapsed_seconds": $elapsed_seconds,
  "script_exit": $exit_code,
  "simulation_exit": $simulation_exit,
  "validation_exit": $validation_exit,
  "comparison_exit": $comparison_exit,
  "plot_exit": $plot_exit,
  "tail_analysis_exit": $tail_analysis_exit,
  "output": "$OUTPUT",
  "checkpoint": "$CHECKPOINT",
  "gpu_metrics": "$GPU_METRICS"
}
EOF
}
trap write_status EXIT

echo "TPV102 H200 validation started at $started_at"
echo "Job=${SLURM_JOB_ID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python=$PYTHON ($($PYTHON --version 2>&1))"
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,power.limit,driver_version \
  --format=csv,noheader
"$PYTHON" - <<'PY'
import jax
import tatva

devices = jax.devices()
print(f"JAX {jax.__version__}; devices={devices}; Tatva={tatva.__file__}")
if len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit(f"Expected exactly one GPU device, found {devices}")
PY

"$PYTHON" -m pytest -q test_tpv101.py test_tpv102.py "$TATVA_ROOT/tests/test_friction.py"
"$PYTHON" tpv102.py --preset hpc-100m --duration 15 \
  --operator-batch-size "$OPERATOR_BATCH_SIZE" "${domain_args[@]}" "${mesh_args[@]}" \
  "${dump_args[@]}" --dry-run

gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
nvidia-smi -i "$gpu_id" \
  --query-gpu=timestamp,name,uuid,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm \
  --format=csv,noheader,nounits -l 2 > "$GPU_METRICS" &
monitor_pid=$!

resume_args=()
if [[ -f "$CHECKPOINT" ]]; then
  echo "Resuming from $CHECKPOINT"
  resume_args=(--resume)
else
  echo "Starting a new TPV102 100 m / 15 s single-GPU run"
fi

set +e
TIMEFORMAT='simulation wall=%3R s user=%3U s system=%3S s'
time "$PYTHON" tpv102.py \
  --preset hpc-100m \
  --duration 15 \
  --operator-batch-size "$OPERATOR_BATCH_SIZE" \
  "${domain_args[@]}" \
  "${mesh_args[@]}" \
  "${dump_args[@]}" \
  --checkpoint-path "$CHECKPOINT" \
  --checkpoint-interval 0.5 \
  --output-dir "$OUTPUT" \
  "${resume_args[@]}"
simulation_exit=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
"$PYTHON" summarize_gpu_metrics.py "$GPU_METRICS" --output "$GPU_SUMMARY"
if [[ $simulation_exit -ne 0 ]]; then
  exit "$simulation_exit"
fi

set +e
"$PYTHON" validate_scec_dump.py "$OUTPUT"
validation_exit=$?
"$PYTHON" compare_scec_reference.py "$OUTPUT" "$UGUCA_REFERENCE" \
  --problem TPV102 --output "$OUTPUT/uguca_reference_comparison.json"
uguca_exit=$?
"$PYTHON" compare_scec_reference.py "$OUTPUT" "$SPECFEM3D_REFERENCE" \
  --problem TPV102 --output "$OUTPUT/specfem3d_reference_comparison.json"
specfem_exit=$?
comparison_exit=$(( uguca_exit != 0 || specfem_exit != 0 ))
candidate_args=(--candidate "$CURRENT_LABEL=$OUTPUT")
if [[ -n "$BASELINE_OUTPUT" && -d "$BASELINE_OUTPUT" && "$BASELINE_OUTPUT" != "$OUTPUT" ]]; then
  candidate_args=(
    --candidate "$BASELINE_LABEL=$BASELINE_OUTPUT"
    --candidate "$CURRENT_LABEL=$OUTPUT"
  )
fi
"$PYTHON" plot_tpv101_validation.py \
  --problem TPV102 \
  --reference-root "$ROOT/reference/tpv102" \
  --reference-user ke \
  --reference-user kaneko \
  --output-dir "$PLOT" \
  "${candidate_args[@]}"
plot_exit=$?
if [[ -n "$BASELINE_OUTPUT" && -d "$BASELINE_OUTPUT" && "$BASELINE_OUTPUT" != "$OUTPUT" ]]; then
  "$PYTHON" analyze_horizontal_shear_tail.py \
    --candidate "$BASELINE_LABEL=$BASELINE_OUTPUT" \
    --candidate "$CURRENT_LABEL=$OUTPUT" \
    --reference "UGUCA=$UGUCA_REFERENCE" \
    --reference "SPECFEM3D=$SPECFEM3D_REFERENCE" \
    --output-dir "$PLOT/tail_analysis"
  tail_analysis_exit=$?
else
  tail_analysis_exit=0
fi
set -e

if [[ $validation_exit -ne 0 || $comparison_exit -ne 0 || $plot_exit -ne 0 || $tail_analysis_exit -ne 0 ]]; then
  echo "Simulation completed, but one or more validation stages failed." >&2
  exit 2
fi

echo "TPV102 H200 validation completed successfully at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
