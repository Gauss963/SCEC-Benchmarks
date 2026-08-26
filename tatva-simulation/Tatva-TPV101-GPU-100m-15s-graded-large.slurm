#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=TPV101-graded-large
#SBATCH --output=/work/gauss112/SCEC-Code-Verification/tpv101-graded-large-%j.out
#SBATCH --error=/work/gauss112/SCEC-Code-Verification/tpv101-graded-large-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/SCEC-Code-Verification/SCEC-Code-Validation/tatva-simulation
export TPV101_RUN_NAME=tpv101_100m_15s_graded_large_gpu1_h200
export TPV101_PLOT_NAME=tpv101_validation_graded_large_vs_uniform_expanded_100m_15s
export BASELINE_OUTPUT="$ROOT/output/tpv101_100m_15s_xy_expanded_gpu1_h200"
export BASELINE_LABEL="Tatva uniform expanded domain"
export CURRENT_LABEL="Tatva graded large domain"
export OPERATOR_BATCH_SIZE=65536
export XLA_MEMORY_FRACTION=0.94

export X_MIN=-50000
export X_MAX=50000
export Y_MIN=-35000
export Y_MAX=50000
export Z_EXTENT=30000

export GRADED_MESH=1
export FINE_X_MIN=-18000
export FINE_X_MAX=18000
export FINE_Y_MIN=-3000
export FINE_Y_MAX=18000
export FINE_Z_EXTENT=8000
export MAX_MESH_SIZE=750
export MESH_GROWTH_RATIO=1.05

exec bash "$ROOT/Tatva-TPV101-GPU-100m-15s.slurm"
