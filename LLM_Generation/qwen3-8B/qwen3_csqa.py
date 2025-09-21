from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import re
import json
from tqdm import tqdm
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


# ✅ 加载 CommonsenseQA 测试集
csqa_dataset = load_dataset("tau/commonsense_qa", split="validation")

# ✅ 构造消息格式（仿照你 MATH 风格）
def build_messages(question, choices):
    choice_labels = choices["label"]
    choice_texts = choices["text"]
    # 构造 A. xxx\nB. xxx 格式
    choice_lines = [f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)]
    choice_block = "\n".join(choice_lines)
    return [
        {
            "role": "user",
            "content": (
                f"{question}\n\nChoices:\n{choice_block}\n\n"
                "Please answer step by step. End your response with: Final Answer: \\boxed{{A/B/C/D/E}}. Make sure to wrap your final answer in \\boxed{{}}."
            )
        }
    ]

# 提取预测答案字母
def extract_pred_answer(text):
    m = re.search(r"Final Answer: \\boxed\{([A-E])\}", text)
    return m.group(1) if m else None

# 判断是否准确
def is_correct(pred, gt):
    return pred == gt

# 推理与评估函数
def evaluate(model, tokenizer, dataset):
    correct = 0
    total = 0
    all_predictions = []

    for item in tqdm(dataset):
        question = item["question"]
        choices = item["choices"]  # dict: {"label": [...], "text": [...]}
        gt_answer = item["answerKey"].strip().upper()  # e.g., "C"

        messages = build_messages(question, choices)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        # print(text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_outputs = model.generate(
                **model_inputs,
                max_new_tokens=16384,
                do_sample=True,
                temperature=0.7
            )

        gen_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, gen_outputs)
        ]
        response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        pred_answer = extract_pred_answer(response)

        # print(response)

        all_predictions.append(response.strip())

        if is_correct(pred_answer, gt_answer):
            correct += 1
        total += 1

    print(f"CommonsenseQA Accuracy: {correct/total:.4f} ({correct}/{total})")

    with open("qwen3_8b_csqa_outputs.jsonl", "w", encoding="utf-8") as f:
        for ln in all_predictions:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")

# 执行评估
evaluate(model, tokenizer, csqa_dataset)
