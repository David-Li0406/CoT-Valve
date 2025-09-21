#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online DeepConf (Lowest Group Confidence) on CSQA with Qwen3-8B

Implements:
  - Warm-up (N_init) to set stopping threshold s from lowest-group-confidence
    using keep ratio η (10%: DeepConf-low | 90%: DeepConf-high)
  - Online token-by-token generation with early stop if current group confidence < s
  - Confidence-weighted majority voting over extracted final answers
  - Adaptive consensus stopping (stop when confidence-weighted consensus ≥ τ)
  - Budget K (max traces) cap

References:
  - Deep Think with Confidence (DeepConf): online/offline algo, lowest group confidence,
    filtering + confidence-weighted voting, adaptive consensus.  (arXiv:2508.15260)  [CITE]
  - Qwen3-8B CSQA baseline loader/template/extractor: your given script                [CITE]
"""

import os
import re
import json
import math
import argparse
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------------------------------------------------------
# I/O & env
# ---------------------------------------------------------------------
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")

def parse_args():
    ap = argparse.ArgumentParser()
    # model & data
    ap.add_argument("--model_name", type=str, default="/scratch/daweili5/cot-valve/saves/Qwen3-8B")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seed", type=int, default=42)

    # sampling
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=2560)

    # DeepConf (online) core
    ap.add_argument("--K", type=int, default=256, help="max trace budget per question")
    ap.add_argument("--N_init", type=int, default=16, help="warm-up traces per question")
    ap.add_argument("--eta", type=float, default=10.0, choices=[10.0, 90.0],
                    help="keep top-η% confidence (10 → DeepConf-low, 90 → DeepConf-high)")
    ap.add_argument("--tau", type=float, default=0.95, help="consensus threshold β")
    ap.add_argument("--group_window", type=int, default=256,
                    help="sliding window size for group confidence (CSQA较短，默认256；数学题可用2048)")
    ap.add_argument("--topk_conf", type=int, default=5,
                    help="token confidence: negative avg log-prob of top-k tokens")

    # limits and housekeeping
    ap.add_argument("--max_questions", type=int, default=-1, help="-1 = all")
    ap.add_argument("--save_pred", type=str, default="deepconf_qwen3_csqa_preds.jsonl")
    ap.add_argument("--mode", type=str, default="deepconf-low",
                    choices=["deepconf-low", "deepconf-high"],
                    help="quick alias: sets eta=10 or 90 if provided")
    return ap.parse_args()


# ---------------------------------------------------------------------
# Prompting & extraction (aligned with your baseline)  [CITE]
# ---------------------------------------------------------------------
def build_messages(question, choices):
    choice_labels = choices["label"]
    choice_texts = choices["text"]
    lines = [f"{lab}. {txt}" for lab, txt in zip(choice_labels, choice_texts)]
    return [
        {
            "role": "user",
            "content": (
                f"{question}\n\nChoices:\n" + "\n".join(lines) +
                "\n\nPlease answer step by step. End your response with: "
                "Final Answer: \\boxed{A/B/C/D/E}. Make sure to wrap your final answer in \\boxed{}."
            )
        }
    ]


EXTRACT_RE = re.compile(r"Final Answer:\s*\\boxed\{([A-E])\}", re.IGNORECASE)

def extract_final_choice(text):
    m = EXTRACT_RE.search(text)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------
# Sampling helpers (token-by-token nucleus sampling)
# ---------------------------------------------------------------------
def top_p_sample(probs, top_p=0.95):
    """Nucleus sampling on a 1D probability tensor (size = vocab)."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumsum > top_p).nonzero(as_tuple=True)[0]
    if len(cutoff) > 0:
        last = cutoff[0]
        keep = sorted_probs[:last + 1]
        keep_idx = sorted_idx[:last + 1]
        keep = keep / keep.sum()
        pick = torch.multinomial(keep, num_samples=1)
        return keep_idx[pick]
    else:
        # already sums to < top_p, sample from all
        return torch.multinomial(sorted_probs, num_samples=1).index_select(0, torch.tensor([0], device=probs.device))


def token_confidence_from_logits(logits, k=5, temperature=1.0):
    """
    Ci = - (1/k) * sum_j log P_i(j), where j iterates over top-k tokens at position i.
    """
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
    topk = min(k, probs.size(-1))
    pvals, _ = torch.topk(probs, k=topk, dim=-1)  # (topk,)
    # avoid log(0)
    pvals = torch.clamp(pvals, min=1e-12)
    Ci = - torch.log(pvals).mean()
    return Ci.item(), probs


def current_group_confidence(conf_hist, group_window):
    """
    CG_k = average of last 'group_window' token confidences (overlapping sliding window).
    """
    if len(conf_hist) == 0:
        return float("inf")
    w = min(group_window, len(conf_hist))
    return sum(conf_hist[-w:]) / w


def lowest_group_confidence(conf_hist, group_window):
    """
    Over the entire trace, compute the minimum sliding-window average (lowest group confidence).
    """
    if not conf_hist:
        return float("inf")
    w = min(group_window, len(conf_hist))
    # precompute prefix sums
    ps = [0.0]
    for c in conf_hist:
        ps.append(ps[-1] + c)
    lows = float("inf")
    # slide
    for end in range(w, len(ps)):
        cg = (ps[end] - ps[end - w]) / w
        if cg < lows:
            lows = cg
    return lows


# ---------------------------------------------------------------------
# One-trace online generation with optional early stop w.r.t threshold s
# ---------------------------------------------------------------------
def generate_one_trace_online(model, tokenizer, prompt_ids, device, args, stop_threshold=None):
    """
    Returns:
      text: decoded completion
      final_choice: 'A'..'E' or None
      conf_hist: [Ci ...]
      lowest_gc: float (lowest group confidence across the trace)
      tokens_used: int
    """
    model.eval()
    use_early_stop = (stop_threshold is not None)

    with torch.no_grad():
        input_ids = prompt_ids.clone()  # (1, L)
        past_key_values = None
        conf_hist = []
        gen_token_ids = []

        for step in range(args.max_new_tokens):
            outputs = model(input_ids=input_ids if past_key_values is None else None,
                            past_key_values=past_key_values,
                            use_cache=True)
            logits = outputs.logits[:, -1, :].squeeze(0)  # (vocab,)
            past_key_values = outputs.past_key_values

            Ci, probs = token_confidence_from_logits(logits, k=args.topk_conf, temperature=args.temperature)
            conf_hist.append(Ci)

            # online early stop: check group confidence vs threshold
            if use_early_stop:
                cg_now = current_group_confidence(conf_hist, args.group_window)
                if cg_now < stop_threshold:
                    break

            # sample next token (top-p nucleus)
            next_token_id = top_p_sample(probs, top_p=args.top_p)
            next_token_id = next_token_id.view(1, 1).to(device)
            gen_token_ids.append(next_token_id.item())

            # append for next step
            input_ids = next_token_id

            # quick stop on newline bursts if already produced final answer
            # (we still check explicit extractor after decode)
            if len(gen_token_ids) >= 6:
                # small heuristic: if we see many line breaks after "Final Answer", bail out
                pass

        # decode only the generated part
        gen_ids = torch.tensor(gen_token_ids, device=device).view(1, -1) if gen_token_ids else torch.empty((1,0), dtype=torch.long, device=device)
        text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0] if gen_ids.numel() > 0 else ""
        final_choice = extract_final_choice(text)
        lowest_gc = lowest_group_confidence(conf_hist, args.group_window)
        return text, final_choice, conf_hist, lowest_gc, len(gen_token_ids)


# ---------------------------------------------------------------------
# Confidence-weighted majority voting + consensus
# ---------------------------------------------------------------------
def weighted_vote_and_beta(answers, weights):
    """
    answers: list like ['A','C',None,...]
    weights: list of floats (same length) -- per-trace confidence used for vote
    Returns:
      winner (str or None), beta (float in [0,1])
    """
    wsum = defaultdict(float)
    total = 0.0
    for a, w in zip(answers, weights):
        if a is None:
            continue
        wsum[a] += max(w, 0.0)
        total += max(w, 0.0)
    if total <= 0:
        return None, 0.0
    winner, win_w = max(wsum.items(), key=lambda kv: kv[1])
    beta = win_w / total
    return winner, beta


# ---------------------------------------------------------------------
# Threshold from warm-up via percentile over lowest-group-confidence
# s = Percentile_{100-η} (C_t over warm-up)   (η=10 → 90th percentile; η=90 → 10th)
# ---------------------------------------------------------------------
def percentile(values, q):
    if not values:
        return -float("inf")
    vals = sorted(values)
    rank = (len(vals) - 1) * (q / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (rank - lo)


# ---------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    if args.mode == "deepconf-low":
        args.eta = 10.0
    elif args.mode == "deepconf-high":
        args.eta = 90.0

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )
    device = model.device

    # dataset
    ds = load_dataset("tau/commonsense_qa", split=args.split)

    total = 0
    correct = 0
    all_logs = []

    # Iterate samples
    for idx, item in enumerate(tqdm(ds, desc="CSQA")):
        if args.max_questions > 0 and idx >= args.max_questions:
            break

        question = item["question"]
        choices = item["choices"]
        gt = str(item["answerKey"]).strip().upper()

        # build prompt using Qwen chat template
        messages = build_messages(question, choices)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        prompt_ids = tokenizer([prompt_text], return_tensors="pt").to(device).input_ids

        # ------------------ Warm-up (no early stop) ------------------
        warm_lowests = []
        warm_answers = []
        warm_weights = []

        for _ in range(args.N_init):
            text, ans, conf_hist, lowest_gc, _ = generate_one_trace_online(
                model, tokenizer, prompt_ids, device, args, stop_threshold=None
            )
            warm_lowests.append(lowest_gc)
            # For weighting, we can reuse the same “lowest group confidence” to be consistent
            warm_answers.append(ans)
            warm_weights.append(lowest_gc)

        # s = Percentile_{100-η}(warm_lowests)
        s = percentile(warm_lowests, 100.0 - args.eta)

        # First, aggregate warm-up traces (confidence-weighted vote)
        win, beta = weighted_vote_and_beta(warm_answers, warm_weights)
        traces = args.N_init
        answers = list(warm_answers)
        weights = list(warm_weights)
        tokens_used = 0  # optional accounting (we could track per-trace if needed)

        # ------------------ Online generation with early stop ------------------
        while traces < args.K:
            # consensus stop
            if win is not None and beta >= args.tau:
                break

            text, ans, conf_hist, lowest_gc, _ = generate_one_trace_online(
                model, tokenizer, prompt_ids, device, args, stop_threshold=s
            )
            answers.append(ans)
            weights.append(lowest_gc)
            traces += 1

            win, beta = weighted_vote_and_beta(answers, weights)

        # final decision for this question
        pred = win if win is not None else "A"  # default fallback
        is_ok = (pred == gt)
        total += 1
        correct += 1 if is_ok else 0

        all_logs.append({
            "index": idx,
            "ground_truth": gt,
            "pred": pred,
            "consensus_beta": beta,
            "eta": args.eta,
            "tau": args.tau,
            "K": args.K,
            "N_init": args.N_init,
            "group_window": args.group_window,
            "notes": f"stop_threshold={s:.6f}, traces_used={traces}"
        })

    acc = correct / max(1, total)
    print(f"[DeepConf ({'low' if args.eta==10.0 else 'high'})] CSQA Accuracy: {acc:.4f}  ({correct}/{total})")

    with open(args.save_pred, "w", encoding="utf-8") as f:
        for row in all_logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
