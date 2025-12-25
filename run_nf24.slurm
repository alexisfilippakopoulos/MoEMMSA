#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod

# --- Single GPU / single task ---
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# --- Short runtime ---
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.err

# --- Environment ---
module purge
module load profile/deeplrn
module load cuda/12.1
module load gcc/11.3.0

# --- Conda environment ---
source ~/miniconda3/bin/activate
conda activate deepmlf-cu12
cd /leonardo_work/EUHPC_A04_051/alexfil/MoEMMSA

# --- Run Python experiment ---
python experiments/regression/mult_base.py \
  -m msalm \
  -d sims \
  -g 0 \
  --exp-name lr_0001__k_2__nf_24-v2 \
  -c MMSA/config/regression/deepmlf/sims/ours_24tokens.json \
  --res-save-dir MMSA/results/sims/lrs_0001/k_2/nf_24-v2 \
  -n 2 \
  -s 1990 -s 1991 -s 1992 -s 1993 -s 1994
