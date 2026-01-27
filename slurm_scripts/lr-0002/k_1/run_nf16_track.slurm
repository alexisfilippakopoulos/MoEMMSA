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
cd /leonardo_scratch/large/userexternal/afilippa/MoEMMSA-all/MoEMMSA/

# --- Run Python experiment ---
python experiments/regression/mult_base.py \
  -m msalm \
  -d sims \
  -g 0 \
  --exp-name lr_0002__k_1__nf_16-lb-0_01-track  \
  -c MMSA/config/regression/deepmlf/sims/lr-0002/k_1/ours_16tokens.json \
  --res-save-dir MMSA/results/sims/lrs_0002/sparse-mm-token-lvl-router/k_1/nf_16-lb-0_01-track  \
  -n 2 \
  -s 1994
