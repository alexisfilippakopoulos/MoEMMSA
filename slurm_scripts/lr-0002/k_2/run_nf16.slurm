#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod

# --- Single GPU / single task ---
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

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
  --exp-name lr_0002__k_2__nf_16-v2 \
  -c MMSA/config/regression/deepmlf/sims/lr-0002/k_2/ours_16tokens.json \
  --res-save-dir MMSA/results/sims/lrs_0002/sparse-mm-token-lvl-router/k_2/nf_16-v2-test \
  -n 2 \
  -s 1990 -s 1991 -s 1992 -s 1993 -s 1994
