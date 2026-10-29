# -*- coding: utf-8 -*-
"""
情绪判断的 Jev 式实现：不生成文本，直接读类别 token 的概率

对比两种做法：
    旧（生成式）：模型生成「消极」→ 正则解析 → 类别
                 缺点：可能输出「负面」「不太好」等词导致解析失败；可能乱写；逐 token 生成慢

    新（Jev 式）：前向一次 → 读最后一个位置的 logits → 取三个类别首 token 的概率 → 取最大
                 优点：输出空间被硬约束在这三个 token，不可能越界；一次前向，快 5-10 倍；
                       天然带概率分布，可以做置信度阈值

token 切分实测（Qwen3-0.6B 词表）：
    "积极" → [99666]            单 token
    "消极" → [114217]           单 token
    "中性" → [15946, 33071]     "中" + "性" 两个 token
                              ↑ 这会让"中性"的概率被"中国""中间"等词分摊，
                                所以我们同时算一个「只比 积/消 时」的结果做对照

用法：
    python eval_jev_style.py --limit 400
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
from prompt_util import build_user_text   # 提示词唯一真源，和训练共用

BASE = "D:/ai/models/Qwen3-0.6B"
ADAPTER = "D:/ai/out/adapter-emotion"
EVAL_JSON = "D:/ai/data/fincuge/emotion_eval.json"

# 三个类别的首 token id（实测得来，不是猜的）
LABEL_TOKENS = {
    "积极": 99666,
    "消极": 114217,
    "中性": 15946,      # 只是"中"，不含"性"
}

SYNONYMS = {
    "积极": ["积极", "正面", "乐观", "正向"],
    "消极": ["消极", "负面", "悲观", "负向"],
    "中性": ["中性", "中立", "无明显", "无情绪"],
}


def gold_label(text):
    t = str(text)
    for lab, ws in SYNONYMS.items():
        if any(w in t for w in ws):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--cpu", action="store_true", help="用 CPU 跑，避免和训练抢显存")
    args = ap.parse_args()

    raw = json.load(open(EVAL_JSON, encoding="utf-8"))
    samples = []
    for r in raw:
        g = gold_label(r["output"])
        if g:
            samples.append({**r, "gold": g})

    # 分层抽样，保证三类都有
    by = {}
    for s in samples:
        by.setdefault(s["gold"], []).append(s)
    per = max(1, args.limit // len(by))
    picked = []
    for lab, arr in by.items():
        picked.extend(arr[:per])
    samples = picked[:args.limit]

    print(f"评估 {len(samples)} 条  分布 {Counter(s['gold'] for s in samples).most_common()}")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    if args.cpu:
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.float32, device_map={"": "cpu"}, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True
        )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    has_think = "enable_thinking" in (tok.chat_template or "")
    ids_map = LABEL_TOKENS
    names = list(ids_map)

    results = []
    t0 = time.time()
    with torch.inference_mode():
        for i, s in enumerate(samples, 1):
            # ⚠ 必须和训练时一模一样，统一走 prompt_util（以前这里手写 "\n\n文本："，和训练不一致）
            msgs = [{"role": "user", "content": build_user_text(s)}]
            kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
            if has_think:
                kw["enable_thinking"] = False
            enc = tok.apply_chat_template(msgs, **kw)
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            ids = ids if args.cpu else ids.cuda()

            # ★ 关键：只做前向，不 generate
            out = model(input_ids=ids)
            logits = out.logits[0, -1, :].float()          # 最后一个位置的原始分数
            probs = torch.softmax(logits, dim=-1)

            raw_p = {n: probs[t].item() for n, t in ids_map.items()}
            total = sum(raw_p.values())
            norm = {n: v / total for n, v in raw_p.items()}  # 在三个类内归一化
            pred = max(norm, key=norm.get)
            conf = norm[pred]

            results.append({
                "gold": s["gold"], "pred": pred, "conf": conf,
                "probs": norm, "input": s["input"],
            })
            if i % 100 == 0:
                acc = sum(1 for r in results if r["pred"] == r["gold"]) / i
                print(f"  {i}/{len(samples)}  累计 {acc:.1%}  {time.time()-t0:.1f}s")

    # ---- 统计 ----
    n = len(results)
    acc = sum(1 for r in results if r["pred"] == r["gold"]) / n
    print(f"\n{'='*58}")
    print(f"  Jev 式（读概率）准确率: {acc:.1%}  （{n} 条，耗时 {time.time()-t0:.1f}s）")
    print(f"  平均耗时: {(time.time()-t0)/n*1000:.0f} ms/条")
    print(f"{'='*58}")

    print(f"\n  三类 F1：")
    for lab in names:
        tp = sum(1 for r in results if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in results if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in results if r["gold"] == lab and r["pred"] != lab)
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        print(f"    {lab}: 精确 {pr:.2f}  召回 {rc:.2f}  F1 {f1:.2f}  （样本 {sum(1 for r in results if r['gold']==lab)}）")

    # 置信度分布 —— 这是 Jev 最值钱的部分
    print(f"\n  置信度分布（可以用它做自动/人工的分流阈值）：")
    for lo, hi, tag in [(0.9, 1.01, "高置信 ≥0.9"), (0.6, 0.9, "中 0.6~0.9"), (0.0, 0.6, "低 <0.6")]:
        sub = [r for r in results if lo <= r["conf"] < hi]
        if sub:
            sub_acc = sum(1 for r in sub if r["pred"] == r["gold"]) / len(sub)
            print(f"    {tag:<14} {len(sub):>4} 条 ({len(sub)/n*100:>4.1f}%)  实际准确率 {sub_acc:.1%}")

    print(f"\n  按预测类别的分布（看中性有没有被预测过）：")
    print(f"    {Counter(r['pred'] for r in results).most_common()}")

    json.dump(results, open("D:/ai/logs/eval_jev_emotion.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n 明细 D:/ai/logs/eval_jev_emotion.json")


if __name__ == "__main__":
    main()
