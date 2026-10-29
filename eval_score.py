# -*- coding: utf-8 -*-
"""按「整句写法」打分，而不是只看第一个词是不是标签。

背景：
    训练数据的答案有 5 种前缀写法 × 3 个标签：
        ''                + 积极。  →  「积极。」
        '所属情感是'        + 积极。  →  「所属情感是积极。」
        '以上文本情感极性属于' + 积极。  →  「以上文本情感极性属于积极。」
        '该文本情感是'      + 积极。  →  「该文本情感是积极的。」(注意这类结尾是"的")
        '情感倾向是'        + 积极。  →  「情感倾向是积极。」
    模型在第一个位置只把 31.5% 的概率给三个标签词，其余给了这些前缀。

    之前我们只取「第一个位置、标签词的概率」，等于强行假设答案必须从标签词开头，
    而 80% 的训练样本不是这样。评分方式和数据对不上。

本脚本的做法（受 5 种写法这个事实启发）：
    对每种前缀 p，算  logP(p | 问题) + logP(标签 | 问题 + p)
    然后对同一类别的 5 种写法做 logsumexp（等价于「把所有合法写法加起来」）。
    哪个类别总分高就判哪个。

    直觉：模型如果觉得「以上文本情感极性属于」+「消极」整体最顺，就判消极，
    而不是因为第一个词不是"消极"就把它判丢。

开销：每条样本 5 次前向（每种前缀一次），比只看首词贵 5 倍，但比重训便宜得多。

用法：
    python eval_score.py --limit 120 --cpu
    python eval_score.py --limit 600 --adapter D:/ai/out/adapter-emotion-vb
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_util import build_user_text

BASE = "D:/ai/models/Qwen3-0.6B"
EVAL_JSON = "D:/ai/data/fincuge/emotion_eval.json"
OUT_JSON = "D:/ai/logs/eval_score_emotion.json"

LABELS = ["积极", "消极", "中性"]
# 训练数据里实际出现的 5 种前缀（含结尾差异），顺序按出现频次
PREFIXES = ["", "所属情感是", "以上文本情感极性属于", "该文本情感是", "情感倾向是"]
TAILS = {"该文本情感是": "的。"}          # 这一类结尾多一个"的"，其余都是"。"


def gold_label(txt):
    hit = [w for w in LABELS if w in txt]
    return hit[0] if len(hit) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--adapter", default="D:/ai/out/adapter-emotion")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = "cpu" if args.cpu else "cuda:0"
    dt = torch.float32 if args.cpu else torch.float16

    raw = json.load(open(EVAL_JSON, encoding="utf-8"))
    samples = []
    for r in raw:
        g = gold_label(r["output"])
        if g:
            samples.append({**r, "gold": g})

    # 分层抽样，和之前的口径一致，方便对比
    by = {}
    for s in samples:
        by.setdefault(s["gold"], []).append(s)
    per = max(1, args.limit // len(by))
    picked = []
    for lab, arr in by.items():
        picked.extend(arr[:per])
    samples = picked[:args.limit]
    print(f"评估 {len(samples)} 条  {Counter(s['gold'] for s in samples).most_common()}")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dt, device_map={"": dev}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    base = model.get_base_model()
    lm = base.lm_head
    has_think = "enable_thinking" in (tok.chat_template or "")

    # 每种前缀的 token（含结尾符号，保证和训练时一致）
    prefix_ids = {}
    for p in PREFIXES:
        prefix_ids[p] = tok(p, add_special_tokens=False)["input_ids"]

    label_first = {lab: tok(lab, add_special_tokens=False)["input_ids"][0] for lab in LABELS}
    tail_ids = {lab: tok(lab + "。", add_special_tokens=False)["input_ids"][1:] for lab in LABELS}

    results = []
    t0 = time.time()

    with torch.inference_mode():
        for i, s in enumerate(samples, 1):
            kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
            if has_think:
                kw["enable_thinking"] = False
            enc = tok.apply_chat_template([{"role": "user", "content": build_user_text(s)}], **kw)
            p_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            if isinstance(p_ids, list):
                p_ids = torch.tensor([p_ids])
            p_ids = p_ids.to(dev)

            class_scores = {lab: [] for lab in LABELS}   # 原始对数概率，稍后 logsumexp

            for p in PREFIXES:
                pid = prefix_ids[p]
                ids = torch.cat([p_ids, torch.tensor([pid], device=dev)], dim=1) if pid else p_ids
                out = base.model(input_ids=ids, use_cache=False)
                h = out.last_hidden_state

                # 1) logP(前缀 | 问题)：前缀每个 token 的对数概率之和
                lp_prefix = 0.0
                if pid:
                    n_p = len(pid)
                    start = ids.shape[1] - n_p
                    for j in range(n_p):
                        logits = lm(h[0, start - 1 + j]).float()
                        lp = torch.log_softmax(logits, dim=-1)[pid[j]].item()
                        lp_prefix += lp

                # 2) logP(标签 | 问题+前缀)：只看最后位置的分布
                logits_last = torch.log_softmax(lm(h[0, -1]).float(), dim=-1)
                for lab in LABELS:
                    # 只算标签首 token（'中性'的第二个字 P(性|中)=1，前面已实测确认，可忽略）
                    lp = lp_prefix + logits_last[label_first[lab]].item()
                    class_scores[lab].append(lp)

            # 3) 对同一类别的 5 种写法做 logsumexp
            final = {lab: float(torch.logsumexp(torch.tensor(v), dim=0)) for lab, v in class_scores.items()}
            pred = max(final, key=final.get)
            results.append({"gold": s["gold"], "pred": pred, "scores": final, "input": s["input"]})

            if i % 20 == 0 or i == len(samples):
                acc = sum(1 for r in results if r["pred"] == r["gold"]) / len(results)
                print(f"  {i}/{len(samples)}  当前准确率 {acc:.1%}")

    n = len(results)
    acc = sum(1 for r in results if r["pred"] == r["gold"]) / n
    print(f"\n{'='*60}\n  按写法加总打分  准确率 {acc:.1%}（{n} 条，{time.time()-t0:.0f} 秒）\n{'='*60}")

    print("\n  各类 F1：")
    for lab in LABELS:
        tp = sum(1 for r in results if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in results if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in results if r["gold"] == lab and r["pred"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        print(f"    {lab}: 精确 {p:.2f}  召回 {rc:.2f}  F1 {f1:.2f}  （{sum(1 for r in results if r['gold']==lab)} 条）")

    print(f"\n  预测分布: {Counter(r['pred'] for r in results).most_common()}")
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  明细 {OUT_JSON}")
    print(f"\n  对照：只看第一个词的做法，同一 120 条是 65.0%")


if __name__ == "__main__":
    main()
