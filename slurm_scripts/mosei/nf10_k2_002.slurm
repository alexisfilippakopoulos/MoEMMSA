#!/bin/bash
#SBATCH --job-name=thesis-nf10-mosei-k2-0002
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
cd /leonardo_scratch/large/userexternal/afilippa/MoEMMSA

export TIKTOKEN_CACHE_DIR=/leonardo_work/EUHPC_A04_051/alexfil/tiktoken_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Run Python experiment ---
python experiments/regression/mult_base.py \
  -m msalm \
  -d mosei \
  -g 0 \
  --exp-name thesis-mosei-nf10-k2-02 \
  -c MMSA/config/regression/deepmlf/mosei/large_nf10_k2_0002.json  \
  --res-save-dir MMSA/results/mosei/nf10_k2_0002 \
  -n 2 \
  -s 1990 -s 1991 -s 1992 -s 1993 -s 1994
