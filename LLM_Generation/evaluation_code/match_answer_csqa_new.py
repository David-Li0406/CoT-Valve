import re
import json
from datasets import load_dataset
from tqdm import tqdm
import os
# os.environ["HF_HOME"]="/scratch/daweili5/hf_cache"

# 1. 提取预测答案（优先提取最后一个 "Final Answer" 后的第一个 \boxed{...}，否则回退到全文最后一个）
def extract_pred_answer(text):
    # 查找最后一个 "Final Answer" 的位置
    last_final_answer_pos = text.rfind("Final Answer")
    
    # 如果找到了 "Final Answer"
    if last_final_answer_pos != -1:
        # 截取从 "Final Answer" 开始到文本末尾的部分
        substring_after_final_answer = text[last_final_answer_pos:]
        # 在这个子字符串中查找第一个 \boxed{...}
        match = re.search(r"\\boxed\s*\{\s*\{?([A-E])\}?\s*\}", substring_after_final_answer, re.DOTALL)
        if match:
            # 如果找到，就返回匹配到的内容
            return match.group(1)
            
    # 回退逻辑：如果未找到 "Final Answer"，或者其后没有 \boxed{...}，
    # 则在全文中查找最后一个 \boxed{...}
    matches = re.findall(r"\\boxed\s*\{\s*\{?([A-E])\}?\s*\}", text, re.DOTALL)
    return matches[-1] if matches else None

# 2. 判断预测是否正确
def is_correct(pred, gold):
    if pred is None:
        return False
    return pred.strip().upper() == gold.strip().upper()

# 3. 加载生成的 jsonl 文件（每行是模型生成的完整响应）
def load_model_outputs(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

# 4. 主评价函数
def evaluate_predictions(jsonl_path):
    # 加载CSQA验证集
    csqa_dataset = load_dataset("tau/commonsense_qa", split="validation")

    # 加载模型输出（每行是完整文本）
    model_outputs = load_model_outputs(jsonl_path)

    # 安全检查：数量是否对齐
    assert len(csqa_dataset) == len(model_outputs), f"长度不一致：CSQA={len(csqa_dataset)}, 生成={len(model_outputs)}"

    correct = 0
    total = 0
    for item, gen_text in tqdm(zip(csqa_dataset, model_outputs), total=len(csqa_dataset)):
        gt_answer = item["answerKey"].strip().upper()
        pred_answer = extract_pred_answer(gen_text)
        if (pred_answer == None):
            print(gen_text)
        if is_correct(pred_answer, gt_answer):
            correct += 1
        total += 1

    print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")

# 调用
if __name__ == "__main__":
    evaluate_predictions("../LLM_Generation/gpt_oss_20b/gpt_oss_20b_csqa_outputs.jsonl")
