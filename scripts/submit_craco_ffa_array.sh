#!/bin/bash
#SBATCH --job-name=craco_ffa
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=slurm_ffa_%A_%a.out
#SBATCH --error=slurm_ffa_%A_%a.err
# NOTE: Set the array range below using:
#   python search_craco_ozstar.py --list-only
# Example for 50 DM trials running up to 10 at a time:
#SBATCH --array=0-49%10

echo "================================================================="
echo "Slurm Job ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname) at $(date)"
echo "================================================================="

source /etc/profile
module load python-scientific/3.11.5-foss-2023b

PYTHON=/fred/oz451/azic/scripts/crystalball_nt/bin/python
SCRIPT_DIR=/fred/oz451/azic/scripts/soft/src/riptide/scripts

$PYTHON ${SCRIPT_DIR}/search_craco_ozstar.py \
  --data-dir /fred/oz451/azic/data/SB77974 \
  --dm-min 0.0 --dm-max 500.0 \
  --pmin 1.0 --pmax 5.0 \
  --cand-x 274 --cand-y 10 \
  --out-prefix craco_ffa_search \
  --array-id ${SLURM_ARRAY_TASK_ID}

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
