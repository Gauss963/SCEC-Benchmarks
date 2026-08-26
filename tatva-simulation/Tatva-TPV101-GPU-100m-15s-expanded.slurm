#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=TPV101-xy-expanded
#SBATCH --output=/work/gauss112/SCEC-Code-Verification/tpv101-xy-expanded-%j.out
#SBATCH --error=/work/gauss112/SCEC-Code-Verification/tpv101-xy-expanded-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/SCEC-Code-Verification/SCEC-Code-Validation/tatva-simulation
export TPV101_RUN_NAME=tpv101_100m_15s_xy_expanded_gpu1_h200
export TPV101_PLOT_NAME=tpv101_validation_xy_domain_comparison_100m_15s_gpu1_h200
export BASELINE_OUTPUT="$ROOT/output/tpv101_100m_15s_gpu1_h200"
export OPERATOR_BATCH_SIZE=65536
export XLA_MEMORY_FRACTION=0.94
export X_MIN=-30000
export X_MAX=30000
export Y_MIN=-16000
export Y_MAX=30000

exec bash "$ROOT/Tatva-TPV101-GPU-100m-15s.slurm"
