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

# 加载 GSM8K 测试集
gsm8k = load_dataset("gsm8k", "main", split="test")

# 构造消息格式
def build_messages(question):
    return [
        {"role": "user", "content": f"{question}\nPlease answer step by step. End your response with: Final Answer: \\boxed{{your final answer here}}. Make sure to wrap your final answer in \\boxed{{}}."}
    ]

# 提取 Final Answer 后的数字
def extract_answer(text):
    match = re.search(r"Final Answer:\s*([-\d.,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    return None

all_predictions = []

# 推理与评估函数
def evaluate(model, tokenizer, dataset, max_samples=100):
    correct = 0
    total = 0

    for item in tqdm(dataset.select(range(max_samples))):
        question = item["question"]
        gt_answer = extract_answer(item["answer"])
        messages = build_messages(question)

        # 构造带 enable_thinking 的输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # ✅ 保留思考模式
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=16384,
                do_sample=True
            )

        # 移除 prompt 部分，保留新增生成
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred_answer = extract_answer(response)

        all_predictions.append(response.strip())  # ✅ 保留完整输出，包括 <think>

        if pred_answer == gt_answer:
            correct += 1
        total += 1

    print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")

    # 保存所有生成结果
    with open("qwen3_8b_gsm8k_outputs.jsonl", "w", encoding="utf-8") as f:
        for line in all_predictions:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + "\n")

# 执行评估
evaluate(model, tokenizer, gsm8k, max_samples=1319)

