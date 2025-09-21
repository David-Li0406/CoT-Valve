import re
from datasets import load_dataset
from tqdm import tqdm
import os
# os.environ["HF_HOME"]="/scratch/daweili5/hf_cache"

# 加载 MATH-500 测试集
math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# 加载模型生成的答案（每行一个样本）
with open("../LLM_Generation/gpt_oss_20b/gpt_oss_20b_math500_outputs.jsonl", "r", encoding="utf-8") as f:
    model_outputs = [line.strip() for line in f]

assert len(model_outputs) == len(math_dataset), "生成结果数量与数据集不一致"

def _extract_content(text: str, start_pos: int = 0) -> str | None:
    """一个辅助函数，用于从指定位置开始提取第一个 \boxed{...} 的内容。"""
    # 定位 \boxed
    box_start = text.find(r'\boxed', start_pos)
    if box_start == -1:
        return None

    brace_start = text.find('{', box_start)
    if brace_start == -1:
        return None

    i = brace_start + 1
    brace_count = 1
    content = []

    while i < len(text):
        c = text[i]
        if c == '{':
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                break
        content.append(c)
        i += 1

    if brace_count != 0:
        return None

    return ''.join(content).strip()

def extract_answer_with_fallback(text: str) -> str | None:
    """
    提取答案的修正逻辑：
    1. 优先提取最后一个 "Final Answer" 后的第一个 \boxed{...} 的内容。
    2. 若未找到，则回退到全文最后一个 \boxed{...} 的内容。
    """
    # 1. 优先策略：寻找 "Final Answer" 后的 \boxed{...}
    matches = list(re.finditer(r'Final Answer[:：]?', text, re.IGNORECASE))
    if matches:
        # 从最后一个 "Final Answer" 之后开始搜索
        search_area_start = matches[-1].end()
        candidate = _extract_content(text, search_area_start)
        
        # 如果在 "Final Answer" 之后找到了任何 \boxed 内容，就直接返回它
        if candidate is not None:
            return candidate

    # 2. 回退策略：如果优先策略没找到，则寻找全文最后一个 \boxed{...}
    last_box_pos = text.rfind(r'\boxed')
    if last_box_pos != -1:
        return _extract_content(text, last_box_pos)

    # 3. 如果所有方法都失败
    return None

def clean_latex(expr):
    """移除空格、换行、美元符号，标准化"""
    return re.sub(r'\s+|\$+', '', expr)

# 评估
correct = 0
results = []

for i in tqdm(range(len(math_dataset))):
    pred = extract_answer_with_fallback(model_outputs[i])
    gt = math_dataset[i]["answer"]

    pred_clean = clean_latex(pred) if pred else ""
    pred_clean = pred_clean.replace("\d", "")
    pred_clean = pred_clean.replace("\\\\", "\\")
    gt_clean = clean_latex(gt)
    
    # print(pred_clean)
    # print(gt_clean)

    is_correct = (pred_clean == gt_clean) or (pred_clean != "" and pred_clean in gt_clean) or (gt_clean in pred_clean)
    if not is_correct:
        print("")
        print("id: " + str(i))
        print(pred_clean)
        # print(pred)
        print(gt_clean)
        # print(gt)
        print("")
    if is_correct:
        correct += 1

    results.append({
        "index": i,
        "question": math_dataset[i]["problem"],
        "ground_truth": gt,
        "prediction": pred,
        "correct": is_correct
    })

# 打印准确率
accuracy = correct / len(math_dataset)
print(f"Accuracy: {accuracy:.2%} ({correct}/{len(math_dataset)})")
