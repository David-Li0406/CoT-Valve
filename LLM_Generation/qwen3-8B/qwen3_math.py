from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import re
from tqdm import tqdm
import json
import os
os.environ["HF_HOME"]="/scratch/daweili5/hf_cache"

# 模型路径
model_name = "/scratch/daweili5/cot-valve/saves/Qwen3-8B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# ✅ 加载 HuggingFaceH4 提供的 MATH-500 数据集
math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# 构造消息格式
def build_messages(question):
    return [
        {"role": "user", "content": f"{question}\nPlease answer step by step. End your response with: Final Answer: \\boxed{{your final answer here}}. Make sure to wrap your final answer in \\boxed{{}}."}
    ]

# 从模型生成文本中提取 Final Answer 后内容（支持非数字）
def extract_pred_answer(text):
    match = re.search(r"Final Answer:\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None

# 判断预测是否正确（宽松匹配：只要包含标准答案即可）
def is_correct(pred, gt):
    if not pred or not gt:
        return False
    return gt.replace(" ", "") in pred.replace(" ", "").replace(",", "")

all_predictions = []

# 推理与评估函数
def evaluate(model, tokenizer, dataset):
    correct = 0
    total = 0

    for item in tqdm(dataset):
        question = item["problem"]
        gt_answer = item["answer"].strip()
        messages = build_messages(question)

        # 构造输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Qwen3 的思维模式
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=16384,
                do_sample=True
            )

        # 去掉 prompt 部分，保留新增生成内容
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred_answer = extract_pred_answer(response)

        # print(response)

        all_predictions.append(response.strip())

        if is_correct(pred_answer, gt_answer):
            correct += 1
        total += 1

    print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")

    # 保存所有完整输出
    with open("qwen3_8b_math500_outputs.jsonl", "w", encoding="utf-8") as f:
        for line in all_predictions:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + "\n")

# 执行评估
evaluate(model, tokenizer, math_dataset)

