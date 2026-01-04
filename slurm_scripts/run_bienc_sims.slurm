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

# --- Run Python experiment ---
python experiments/regression/mult_base.py \
  -m bienc \
  -d sims \
  -g 0 \
  --exp-name bienc-sims-1990 \
  -c /leonardo_work/EUHPC_A04_051/alexfil/DeepMLF-Reprod/MMSA/config/regression/deepmlf/sims/my_bienc.json \
  --res-save-dir MMSA/results/bienc/sims \
  -n 2 \
  -s 1990

