#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=dev
#SBATCH --job-name=TPV101-100m-gpu-bench
#SBATCH --output=/work/gauss112/SCEC-Code-Verification/gpu-benchmark-%j.out
#SBATCH --error=/work/gauss112/SCEC-Code-Verification/gpu-benchmark-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=/work/gauss112/SCEC-Code-Verification/SCEC-Code-Validation/tatva-simulation
TATVA_ROOT=/work/gauss112/tatva-v0.11.4-gpu
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
RESULT_ROOT="$ROOT/output/gpu_100m_benchmark_${SLURM_JOB_ID}"
OUTPUT="$RESULT_ROOT/simulation"
GPU_METRICS="$RESULT_ROOT/nvidia_smi.csv"
OPERATOR_BATCH_SIZE=${OPERATOR_BATCH_SIZE:-131072}
BENCHMARK_DURATION=${BENCHMARK_DURATION:-0.05}

module purge
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset LD_LIBRARY_PATH || true

mkdir -p "$RESULT_ROOT"
cd "$ROOT"

echo "TPV101 H200 benchmark started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job=${SLURM_JOB_ID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python=$PYTHON ($($PYTHON --version 2>&1))"
echo "operator_batch_size=$OPERATOR_BATCH_SIZE duration=$BENCHMARK_DURATION"
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

gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
nvidia-smi -i "$gpu_id" \
  --query-gpu=timestamp,name,uuid,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm \
  --format=csv,noheader,nounits -l 1 > "$GPU_METRICS" &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

TIMEFORMAT='benchmark wall=%3R s user=%3U s system=%3S s'
time "$PYTHON" tpv101.py \
  --preset hpc-100m \
  --duration "$BENCHMARK_DURATION" \
  --operator-batch-size "$OPERATOR_BATCH_SIZE" \
  --output-dir "$OUTPUT"

cleanup_monitor
trap - EXIT

"$PYTHON" validate_scec_dump.py "$OUTPUT"
"$PYTHON" summarize_gpu_metrics.py "$GPU_METRICS"

echo "TPV101 H200 benchmark completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
