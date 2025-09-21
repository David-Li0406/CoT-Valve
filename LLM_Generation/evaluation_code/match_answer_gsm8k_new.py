import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from fractions import Fraction
import os
# os.environ["HF_HOME"]="/scratch/daweili5/hf_cache"

# 加载 GSM8K 测试集标准答案
gsm8k = load_dataset("openai/gsm8k", name="main", split="test")
gt_answers = []

for item in gsm8k:
    # 提取答案中 #### 后的整数部分
    match = re.search(r"####\s*(-?\d+)", item["answer"])
    if match:
        gt_answers.append(int(match.group(1)))
    else:
        gt_answers.append(None)  # 若无法匹配，保留 None 占位

# 加载生成结果，每行为一个完整答案
def load_txt_predictions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
    
def extract_numbers_from_boxed_after_final_answer(text: str):
    """
    优先从“Final Answer”之后的第一个 \boxed{...} 中抽取数字；
    若未找到或没有“Final Answer”，则回退到全文最后一个 \boxed{...}。
    返回 [int/float, ...]，若未找到则 []。
    """
    if not text:
        return []

    def _find_boxed_contents(s: str):
        """返回 s 中按出现顺序提取到的所有 \boxed{...} 内容（已去掉外层花括号）。"""
        results = []
        pos = 0
        while True:
            m = re.search(r'\\boxed\s*\{', s[pos:])
            if not m:
                break
            start_brace = pos + m.end() - 1  # 指向匹配到的 '{'
            i = start_brace + 1
            depth = 1
            while i < len(s) and depth > 0:
                c = s[i]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                # start_brace+1 .. i-2 是内容（i 已经指到 '}' 后一位）
                content = s[start_brace+1:i-1]
                results.append(content)
                pos = i
            else:
                # 花括号不配对，停止
                break
        return results

    def _normalize_latex_to_plain(content: str) -> str:
        """把常见 latex 形式规整为便于抽取数字的普通文本。"""
        # \frac{a}{b}, \dfrac{a}{b} -> a/b （仅处理数字的分数）
        content = re.sub(
            r'\\(?:d?frac)\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}',
            r'\1/\2',
            content
        )
        # 去掉宏命令（如 \cdot \text 等），但保留可能的负号/数字/斜杠/点号
        content = re.sub(r'\\[a-zA-Z]+', '', content)
        # 去掉美元符、千位逗号与多余空白
        content = content.replace('$', '').replace(',', ' ')
        # 去掉剩余的花括号（避免干扰）
        content = content.replace('{', ' ').replace('}', ' ')
        # 规范空白
        content = re.sub(r'\s+', ' ', content).strip()
        return content

    # 1) 先从 Final Answer 之后取第一个 boxed
    final_matches = list(re.finditer(r'Final Answer[:：]?\s*', text, re.IGNORECASE))
    boxed_content = None
    if final_matches:
        tail = text[final_matches[-1].end():]
        tail_boxed = _find_boxed_contents(tail)
        if tail_boxed:
            boxed_content = tail_boxed[0]

    # 2) 若失败，则回退到全文最后一个 boxed
    if boxed_content is None:
        all_boxed = _find_boxed_contents(text)
        if all_boxed:
            boxed_content = all_boxed[-1]  # 最后一个
        else:
            return []

    # 3) 规范化并抽取数字
    content = _normalize_latex_to_plain(boxed_content)

    # 支持整数/小数/真分数（如 3/4）；允许负号
    num_pat = r'-?\d+(?:\.\d+)?(?:/\d+)?'
    raw_nums = re.findall(num_pat, content)

    numbers = []
    for s in raw_nums:
        try:
            if '/' in s:
                f = Fraction(s)
                numbers.append(int(f) if f.denominator == 1 else float(f))
            elif '.' in s:
                numbers.append(float(s))
            else:
                numbers.append(int(s))
        except Exception:
            continue

    return numbers

# 评估函数（判断答案是否包含 ground truth）
def evaluate(predictions, gt_answers):
    results = []
    correct = 0
    total = 0

    for idx, (pred_text, gt) in enumerate(zip(predictions, gt_answers)):
        if gt is None:
            continue  # 忽略无法识别的 ground truth

        extracted = extract_numbers_from_boxed_after_final_answer(pred_text)
        if extracted:
            is_correct = (gt == extracted[0])
        else:
            is_correct = False

        if not is_correct:
            # print(extracted)
            # print(gt)
            if not extracted:
                print(pred_text)

        results.append({
            "index": idx,
            "ground_truth": gt,
            "extracted_answers": extracted,
            "prediction_text": pred_text,
            "correct": is_correct
        })

        # if idx <= 10:
            # print(results[idx])

        total += 1
        if is_correct:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return results, acc

# 主入口
if __name__ == "__main__":
    predictions = load_txt_predictions("..e/LLM_Generation/gpt_oss_20b/gpt_oss_20b_gsm8k_outputs.jsonl")  # 替换为你的预测路径
    results, acc = evaluate(predictions, gt_answers)
    print(f"GSM8K Accuracy: {acc:.2%}")
