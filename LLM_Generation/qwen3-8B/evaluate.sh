#!/usr/bin/env bash
#SBATCH -t 0-6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu

# python qwen3_csqa.py
# python qwen3_gpqa.py
# python qwen3_gsm8k.py
# python qwen3_math.py


python deepconf_qwen3_csqa_online.py \
  --model_name "/scratch/daweili5/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
  --mode deepconf-low \
  --K 1 --N_init 16 --tau 0.95 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 2560 --temperature 0.7 --top_p 0.95 \
  --save_pred deepconf_low_csqa.jsonl \
  --max_questions 100