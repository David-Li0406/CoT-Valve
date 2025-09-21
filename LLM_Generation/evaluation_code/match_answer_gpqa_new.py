import json
import re
from tqdm import tqdm
import os
# os.environ["HF_HOME"]="/scratch/daweili5/hf_cache"

def extract_pred_answer(text):
    """
    优先提取最后一个 "Final Answer" 后的第一个 \boxed{...}，否则回退到全文最后一个
    """
    # 查找最后一个 "Final Answer" 的位置
    last_final_answer_pos = text.rfind("Final Answer")
    
    # 如果找到了 "Final Answer"
    if last_final_answer_pos != -1:
        # 截取从 "Final Answer" 开始到文本末尾的部分
        substring_after_final_answer = text[last_final_answer_pos:]
        # 在这个子字符串中查找第一个 \boxed{...}
        match = re.search(r"\\boxed\s*\{\s*\{?([A-D])\}?\s*\}", substring_after_final_answer, re.DOTALL)
        if match:
            # 如果找到，就返回匹配到的内容
            return match.group(1)
            
    # 回退逻辑：如果未找到 "Final Answer"，或者其后没有 \boxed{...}，
    # 则在全文中查找最后一个 \boxed{...}
    matches = re.findall(r"\\boxed\s*\{\s*\{?([A-D])\}?\s*\}", text, re.DOTALL)
    return matches[-1] if matches else None

def is_correct(pred, gold):
    if pred is None:
        return False
    return pred.strip().upper() == gold.strip().upper()

def load_model_outputs(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def load_gold_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_exoplanet(json_path_gold, jsonl_path_model):
    gold_data = load_gold_data(json_path_gold)
    model_outputs = load_model_outputs(jsonl_path_model)

    assert len(gold_data) == len(model_outputs), f"样本数量不一致：gold={len(gold_data)}, pred={len(model_outputs)}"

    correct = 0
    total = 0
    results = []

    for gold_item, model_text in tqdm(zip(gold_data, model_outputs), total=len(gold_data)):
        gold_answer = gold_item["final_answer"].strip().upper()
        pred_answer = extract_pred_answer(model_text)
        if pred_answer == None:
            print(model_text[-100:])
            print("\n")

        is_corr = is_correct(pred_answer, gold_answer)
        results.append({
            "question": gold_item["prompt"][0]["value"],
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "correct": is_corr,
            "model_output": model_text
        })

        if is_corr:
            correct += 1
        total += 1

    acc = correct / total
    print(f"Exoplanet Accuracy: {acc:.4f} ({correct}/{total})")

# 执行
if __name__ == "__main__":
    evaluate_exoplanet("../LLM_Generation/gpt_oss_20b/gpqa_diamond.json", "/scratch/daweili5/cot-valve/LLM_Generation/gpt_oss_20b/gpt_oss_20b_gpqa_diamond_outputs.jsonl")
