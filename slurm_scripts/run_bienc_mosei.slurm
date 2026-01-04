#!/bin/bash
#SBATCH --job-name=gpu_check_1gpu
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod

# --- Single GPU / single task ---
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# --- Short runtime ---
#SBATCH --time=01:00:00
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

export TIKTOKEN_CACHE_DIR=/leonardo_work/EUHPC_A04_051/alexfil/tiktoken_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Run Python experiment ---
python experiments/regression/mult_base.py \
  -m bienc \
  -d mosei \
  -g 0 \
  --exp-name bienc-mosei-1990 \
  -c /leonardo_work/EUHPC_A04_051/alexfil/DeepMLF-Reprod/MMSA/config/regression/deepmlf/mosei/my_bienc.json \
  --res-save-dir MMSA/results/bienc/mosei \
  -n 2 \
  -s 1990
