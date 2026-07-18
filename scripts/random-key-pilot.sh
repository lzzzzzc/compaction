#!/bin/bash
#SBATCH --job-name=q-q-rand
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --requeue

export HOME=/home/$USER
source ~/.bashrc
cd ~/compaction
conda activate compaction

seeds=(17 67 117)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"

python -u -m evaluation.run_qa_evaluation \
  --name "t0.1_repeat_random_subset_s${seed}" \
  --model-name Qwen/Qwen3-4B \
  --dataset-name quality \
  --n-articles 10 \
  --start-article 0 \
  --methods random_subset_keys_nnls2_-3_3_lsq \
  --target-size 0.1 \
  --query-config repeat \
  --algorithm-config key-selection \
  --compute-stats 1 \
  --verbose-logging 1 \
  --seed "$seed" \
  --log-dir logs/qa_evaluation/qwen-quality/random-key-pilot
