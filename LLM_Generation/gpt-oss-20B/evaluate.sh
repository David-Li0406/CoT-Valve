#!/usr/bin/env bash
#SBATCH -t 0-6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu

python gpt_oss_20b_csqa.py
python gpt_oss_20b_gpqa.py
python gpt_oss_20b_gsm8k.py
python gpt_oss_20b_math.py