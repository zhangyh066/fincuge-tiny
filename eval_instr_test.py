# -*- coding: utf-8 -*-
"""小实验：只在指令里加「中性」的判据，不改模型，看准确率变不变。

背景：现在的指令只有一句「这个文本的情感倾向是积极、消极还是中性的。」
——从没告诉模型"什么算中性"。而中性类恰恰是错得最多的（召回 0.47）。

做法：同一批 120 条（每类 40，和之前 70.8% 那批完全一致），
      同一个模型 adapter-emotion-16k，只换提示词，跑两遍对比。

成本：显卡上约 2 分钟。
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import json
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_util import build_user_text

BASE = "D:/ai/models/Qwen3-0.6B"
ADAPTER = "D:/ai/out/adapter-emotion-16k"
EVAL_JSON = "D:/ai/data/fincuge/emotion_eval.json"

LABELS = ["积极", "消极", "中性"]
TOK = {"积极": 99666, "消极": 114217, "中性": 15946}   # 三个标签的首 token

# 新指令：在原句后面加三条判据
CRITERIA = (
    "\n判断标准："
    "积极=明确看好、看涨、表达满意或期待；"
    "消极=明确看空、看跌、表达不满或担忧；"
    "中性=只陈述事实、提问、罗列信息，或情绪不明确。"
)


def gold_label(txt):
    hit = [w for w in LABELS if w in txt]
    return hit[0] if len(hit) == 1 else None


def build_prompt(row, with_criteria):
    ins = str(row["instruction"]).strip()
    inp = str(row["input"]).strip()
    if with_criteria:
        ins = ins + CRITERIA
    return f"{ins}\n{inp}" if inp else ins


def main():
    raw = json.load(open(EVAL_JSON, encoding="utf-8"))
    samples = []
    for r in raw:
        g = gold_label(r["output"])
        if g:
            samples.append({**r, "gold": g})

    # 分层取 120 条：每类前 40 条，和之前完全同一批（保证可比）
    by = {}
    for s in samples:
        by.setdefault(s["gold"], []).append(s)
    picked = []
    for lab in LABELS:
        picked.extend(by[lab][:40])
    print(f"评估 {len(picked)} 条  {Counter(s['gold'] for s in picked).most_common()}\n")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    has_think = "enable_thinking" in (tok.chat_template or "")

    results = {}
    for tag, with_criteria in [("原指令", False), ("加判据", True)]:
        t0 = time.time()
        preds, confs = [], []
        with torch.inference_mode():
            for s in picked:
                kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
                if has_think:
                    kw["enable_thinking"] = False
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": build_prompt(s, with_criteria)}], **kw)
                ids = enc["input_ids"] if hasattr(enc, "keys") else enc
                ids = ids.cuda()
                logits = model(input_ids=ids).logits[0, -1, :].float()
                p = torch.softmax(logits, dim=-1)
                raw3 = {lab: p[TOK[lab]].item() for lab in LABELS}
                total = sum(raw3.values())
                probs = {lab: v / total for lab, v in raw3.items()}
                best = max(probs, key=probs.get)
                preds.append(best)
                confs.append(probs[best])
        acc = sum(1 for s, p in zip(picked, preds) if s["gold"] == p) / len(picked)
        results[tag] = (preds, confs, acc, time.time() - t0)
        print(f"[{tag}] 准确率 {acc:.1%}  用时 {time.time()-t0:.0f}s  "
              f"预测分布 {Counter(preds).most_common()}")

    print()
    for tag in results:
        preds, confs, acc, _ = results[tag]
        print(f"  {tag} 各类 F1：", end="")
        out = []
        for lab in LABELS:
            tp = sum(1 for s, p in zip(picked, preds) if s["gold"] == lab and p == lab)
            fp = sum(1 for s, p in zip(picked, preds) if s["gold"] != lab and p == lab)
            fn = sum(1 for s, p in zip(picked, preds) if s["gold"] == lab and p != lab)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            out.append(f"{lab} {f1:.2f}")
        print("  ".join(out))

    a0 = results["原指令"][2]
    a1 = results["加判据"][2]
    print(f"\n  准确率 {a0:.1%} → {a1:.1%}（{a1-a0:+.1%}）")
    print(f"  参考：同一模型全量 2,020 条是 75.2%")


if __name__ == "__main__":
    main()
