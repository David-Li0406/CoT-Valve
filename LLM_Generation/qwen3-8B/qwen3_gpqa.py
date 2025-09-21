from transformers import AutoTokenizer, AutoModelForCausalLM
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

# ✅ 加载本地 GPQA Diamond JSON 文件
with open("gpqa_diamond.json", "r", encoding="utf-8") as f:
    gpqa_dataset = json.load(f)

# 构造消息格式（仿照 MATH 的chat格式）
def build_messages(question_text):
    return [
        {
            "role": "user",
            "content": (
                f"Question: {question_text}\n\n"
                "Please answer step by step. End your response with: Final Answer: \\boxed{{Choice X}}. Make sure to wrap your final answer in \\boxed{{}}."
            )
        }
    ] 

# ✅ 提取 "The correct answer is (X)" 中的字母
def extract_pred_answer(text):
    match = re.search(r"Final Answer: \\boxed{{\s*\(([A-D])\)}}", text)
    if match:
        return match.group(1)
    return None

# ✅ 判断是否正确（精确匹配 A/B/C/D）
def is_correct(pred, gt):
    return pred == gt

# ✅ 推理与评估函数
def evaluate(model, tokenizer, dataset):
    correct = 0
    total = 0
    all_predictions = []

    for item in tqdm(dataset):
        question_text = item["prompt"][0]["value"]
        gt_answer = item["final_answer"].strip().upper()

        messages = build_messages(question_text)

        # 构造 tokenizer 输入（chat 格式）
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Qwen3 专属参数
        )
        # print(text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=16384,
                do_sample=True,
                temperature=0.7
            )

        # 截取新生成的token部分
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
    with open("qwen3_8b_gpqa_diamond_outputs.jsonl", "w", encoding="utf-8") as f:
        for line in all_predictions:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + "\n")

# ✅ 执行评估
evaluate(model, tokenizer, gpqa_dataset)
