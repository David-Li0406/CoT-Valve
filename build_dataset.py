from datasets import load_dataset
import json
import os
os.environ["HF_HOME"]="/scratch/daweili5/hf_cache"

mixchain = load_dataset("horseee/MixChain-Z-PRM12K", split="train")
dataset = []
for item in mixchain:
    idx = None
    shortest = max([item["solution_1_token"], item["solution_2_token"], item["solution_3_token"], item["solution_4_token"]])
    # print(shortest)
    # print([item["solution_1_token"], item["solution_2_token"], item["solution_3_token"], item["solution_4_token"]])
    # exit()
    for i in ["1", "2", "3", "4"]:
        if item[f"solution_{i}_token"] <= shortest and item[f"correct_{i}"]:
            idx = i
            shortest = item[f"solution_{i}_token"]
    if idx is None:
        # print(111)
        # print(shortest)
        # print([item["solution_1_token"], item["solution_2_token"], item["solution_3_token"], item["solution_4_token"]])
        for i in ["1", "2", "3", "4"]:
            if item[f"solution_{i}_token"] <= shortest:
                idx = i
                shortest = item[f"solution_{i}_token"]
                # print(item[f"solution_{i}_token"], shortest)
    response = item[f"solution_{idx}"]
    response_list = response.split("\n\n")
    response = "<think>" + "\n\n".join(response_list[:-1]) + "</think>" + "\n\n"+ response_list[-1]
    dataset.append({
        "instruction": item["question"],
        "output": response
    })

with open("data/MixChain-Z-PRM12K.json", "w") as f:
    json.dump(dataset, f, indent=2)