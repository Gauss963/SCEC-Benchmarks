#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=TPV101-graded-large-refine
#SBATCH --output=/work/gauss112/SCEC-Code-Verification/tpv101-graded-large-refine-%j.out
#SBATCH --error=/work/gauss112/SCEC-Code-Verification/tpv101-graded-large-refine-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

# Runs from its own checkout so the validated tree is never modified.
ROOT="${TATVA_SIM_ROOT:-/work/gauss112/SCEC-Code-Verification/SCEC-Code-Validation/tatva-simulation-refine}"
export TATVA_SIM_ROOT="$ROOT"
export TPV101_RUN_NAME=tpv101_100m_15s_graded_large_refine_gpu1_h200
export TPV101_PLOT_NAME=tpv101_validation_graded_large_refine_vs_graded_large_100m_15s
export BASELINE_OUTPUT="$ROOT/output/tpv101_100m_15s_graded_large_gpu1_h200"
export BASELINE_LABEL="Tatva graded large domain (750 m outer)"
export CURRENT_LABEL="Tatva graded large domain (500 m outer)"
# Reduced against the 750 m run because the refined mesh grows the element count.
export OPERATOR_BATCH_SIZE=16384
# The JAX pool has to leave device memory free for the compiled executable's
# module image, which is loaded outside the pool: at 0.94 the CUBIN load failed
# even though the pool itself was only 74% used (jobs 302462 and 302463).
export XLA_MEMORY_FRACTION=0.85

# Unchanged domain: the boundary distances that keep reflections outside the
# 15 s window are the same as in the 750 m outer-mesh run.
export X_MIN=-50000
export X_MAX=50000
export Y_MIN=-35000
export Y_MAX=50000
export Z_EXTENT=30000

# Unchanged 100 m core, finer and more gradual outer mesh:
# max cell 750 -> 500 m, growth ratio 1.05 -> 1.04.
# 300 m / 1.03 ran the H200 out of memory (jobs 302407 and 302409);
# see docs/graded-large-domain-refine.md.
export GRADED_MESH=1
export FINE_X_MIN=-18000
export FINE_X_MAX=18000
export FINE_Y_MIN=-3000
export FINE_Y_MAX=18000
export FINE_Z_EXTENT=8000
export MAX_MESH_SIZE=500
export MESH_GROWTH_RATIO=1.04

# UGUCA-format interface dump over the uniform 100 m fault window, so the same
# analysis scripts can animate Tatva and UGUCA results.
export UGUCA_DUMP_INTERVAL=0.1
export UGUCA_DUMP_X_MIN=-18000
export UGUCA_DUMP_X_MAX=18000
export UGUCA_DUMP_Y_MIN=-3000
export UGUCA_DUMP_Y_MAX=18000

exec bash "$ROOT/Tatva-TPV101-GPU-100m-15s.slurm"
