#!/usr/bin/env bash
#SBATCH -t 0-4:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu

# nvidia-smi
# export HF_HOME=/scratch/daweili5/hf_cache
# llamafactory-cli train config/qwen3-8b.yaml
llamafactory-cli train config/gpt-oss-20b.yaml