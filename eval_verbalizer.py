# -*- coding: utf-8 -*-
"""
情绪判断 · Verbalizer 标准实现（对照实验 B 版）

背景（有论文依据，不是拍脑袋）：
    EgoProactive 2026 论文 "The verbalizer reformulation" 一节给出对比：
        生成式做二分类：Macro-F1 0.4517（精确 1.000 但召回只有 0.138，掉进"安全角落"）
        Verbalizer 读概率：Macro-F1 0.7008
    论文原文："The decision is read from the renormalized probability over these two logits alone.
              The operating point τ becomes tunable after training without retraining."

为什么需要 completion scoring：
    Qwen3-0.6B 词表实测：
        "积极" → [99666]              完整词，单 token
        "消极" → [114217]             完整词，单 token
        "中性" → [15946, 33071]       "中" + "性" 两个 token
    如果只读首 token，中性的概率会被"中国/中间/中盘"等词分摊，不公平。
    → 对"中性"补算 P(性|中)，让三类都是「完整词的概率」

三类最终概率：
    P(积极) = P(99666)                        ← 已经是一整词
    P(消极) = P(114217)                       ← 已经是一整词
    P(中性) = P(15946) × P(性 | 中)            ← 需要补第二个 token
    然后在三类内部归一化

用法：
    python eval_verbalizer.py --limit 200 --cpu
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

BASE = os.environ.get("QWEN_MODEL") or ("D:/ai/models/Qwen3-0.6B" if os.path.isdir("D:/ai/models/Qwen3-0.6B") else "Qwen/Qwen3-0.6B")
ADAPTER = "D:/ai/out/adapter-emotion"
EVAL_JSON = "D:/ai/data/fincuge/emotion_eval.json"

# 实测得到的 token id
TOK_POS = 99666        # "积极" 整词
TOK_NEG = 114217       # "消极" 整词
TOK_ZHONG = 15946      # "中" 首字
TOK_XING = 33071       # "性" 第二字

SYN_GOLD = {
    "积极": ["积极", "正面", "乐观", "正向"],
    "消极": ["消极", "负面", "悲观", "负向"],
    "中性": ["中性", "中立", "无明显", "无情绪"],
}


def gold_label(text):
    t = str(text)
    for lab, ws in SYN_GOLD.items():
        if any(w in t for w in ws):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--cpu", action="store_true")
    # 批大小：默认 1 = 完全按老逻辑一条一条跑（数值不变）。
    # 之所以加这个：单条 0.42s，全量 2020 条要 14 分钟，而 agent 的前台命令通道
    # 到 9-10 分钟会被系统 SIGTERM 掐断，导致全量评估永远跑不完。
    # 左 padding + 取最后一个位置，数学上与原逻辑等价（B=1 时无任何 padding）。
    ap.add_argument("--batch", type=int, default=1)
    # 断点续跑：结果逐批写 logs/eval_vb_cache_<adapter>.jsonl，重跑同一条命令会自动跳过已完成的
    ap.add_argument("--resume", type=int, default=1)
    args = ap.parse_args()

    dev = "cpu" if args.cpu else "cuda:0"
    dt = torch.float32 if args.cpu else torch.float16

    raw = json.load(open(EVAL_JSON, encoding="utf-8"))
    samples = []
    for r in raw:
        g = gold_label(r["output"])
        if g:
            samples.append({**r, "gold": g})

    by = {}
    for s in samples:
        by.setdefault(s["gold"], []).append(s)
    if args.limit and args.limit > 0:
        # 分层抽样：每类取 limit/类数 条，用于和历次同口径对比
        per = max(1, args.limit // len(by))
        picked = []
        for lab, arr in by.items():
            picked.extend(arr[:per])
        samples = picked[:args.limit]
    # limit <= 0 时不动，用全部样本、保持原始分布（和论文口径一致的那一种）
    print(f"评估 {len(samples)} 条  {Counter(s['gold'] for s in samples).most_common()}")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # 左 padding：最后一位永远是真实末 token，直接取 logits[:, -1]
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dt, device_map={"": dev}, trust_remote_code=True
    )
    if args.adapter:                      # 空字符串 = 评基座（零样本对照）
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    has_think = "enable_thinking" in (tok.chat_template or "")
    bs = max(1, args.batch)

    # ── 断点续跑：每批结果立刻落到 jsonl，被系统 SIGTERM 掐掉也不丢进度 ──
    cache_path = os.path.join("D:/ai/logs", f"eval_vb_cache_{os.path.basename(os.path.normpath(args.adapter)) if args.adapter else 'base'}.jsonl")
    cached = {}
    if args.resume and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue          # 上次被杀可能写残了半行，跳过
                if 0 <= rec["i"] < len(samples):
                    cached[rec["i"]] = rec
        print(f"  [续跑] 已有 {len(cached)} 条结果，跳过它们")
    cache_file = open(cache_path, "a", encoding="utf-8")

    results = [None] * len(samples)
    for i, rec in cached.items():
        results[i] = rec
    t0 = time.time()
    done = len(cached)

    with torch.inference_mode():
        for b0 in range(0, len(samples), bs):
            idxs = [i for i in range(b0, min(b0 + bs, len(samples))) if i not in cached]
            if not idxs:
                continue
            chunk = [samples[i] for i in idxs]
            # ⚠ 必须和训练时一模一样。以前这里写死了 "\n\n文本："，训练却是 "\n"，
            #    模型没见过这种格式，评估结果被系统性压低。现在统一走公共函数。
            convs = [[{"role": "user", "content": build_user_text(s)}] for s in chunk]
            kw = {"tokenize": True, "add_generation_prompt": True, "return_dict": True,
                  "return_tensors": "pt", "padding": True}
            if has_think:
                kw["enable_thinking"] = False
            enc = tok.apply_chat_template(convs, **kw)
            ids = enc["input_ids"].to(dev)
            mask = enc["attention_mask"].to(dev)

            # ── 一次前向拿全部：末尾直接接一个"中"字 ──
            # 左 padding ⇒ 每行长度相同，位置 -1 是接上去的"中"，位置 -2 是真实的末 token。
            # 所以 out[:, -2] 给出 P(积极)/P(消极)/P(中)，out[:, -1] 给出 P(性|中)，
            # 和原来"跑两次前向"在数学上完全等价，但时间省一半。
            extra = torch.full((ids.size(0), 1), TOK_ZHONG, device=dev, dtype=ids.dtype)
            ids1 = torch.cat([ids, extra], dim=1)
            mask1 = torch.cat([mask, torch.ones((mask.size(0), 1), device=dev, dtype=mask.dtype)], dim=1)
            out = model(input_ids=ids1, attention_mask=mask1)

            p1 = torch.softmax(out.logits[:, -2, :].float(), dim=-1)
            p2 = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
            p_pos = p1[:, TOK_POS]
            p_neg = p1[:, TOK_NEG]
            p_zhong = p1[:, TOK_ZHONG]
            p_neu = p_zhong * p2[:, TOK_XING]

            # ── 三类内归一化 ──
            total = p_pos + p_neg + p_neu
            for j, (i, s) in enumerate(zip(idxs, chunk)):
                a, b, c, t = p_pos[j].item(), p_neg[j].item(), p_neu[j].item(), total[j].item()
                if t <= 0:
                    probs = {"积极": 1/3, "消极": 1/3, "中性": 1/3}
                else:
                    probs = {"积极": a/t, "消极": b/t, "中性": c/t}
                pred = max(probs, key=probs.get)
                rec = {"i": i, "gold": s["gold"], "pred": pred, "conf": probs[pred],
                       "probs": probs, "raw3": (a, b, c), "input": s["input"]}
                results[i] = rec
                cache_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_file.flush()
            os.fsync(cache_file.fileno())

            done += len(chunk)
            if done // 50 != (done - len(chunk)) // 50:
                acc = sum(1 for r in results[:done] if r and r["pred"] == r["gold"]) / done
                print(f"  {done}/{len(samples)}  {acc:.1%}  {time.time()-t0:.0f}s")
    cache_file.close()

    results = [r for r in results if r is not None]
    if len(results) < len(samples):
        print(f"\n  ⚠ 只完成了 {len(results)}/{len(samples)} 条（被中断）。"
              f"重跑同一条命令会从断点继续（--resume 已默认开启）。")
        return

    # ── 统计 ──
    n = len(results)
    acc = sum(1 for r in results if r["pred"] == r["gold"]) / n
    print(f"\n{'='*58}")
    print(f"  Verbalizer 标准版（completion scoring）")
    print(f"  准确率 {acc:.1%}  （{n} 条，{time.time()-t0:.0f}s，{(time.time()-t0)/n*1000:.0f} ms/条）")
    print(f"{'='*58}")

    print("\n  三类 F1：")
    for lab in ["积极", "消极", "中性"]:
        tp = sum(1 for r in results if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in results if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in results if r["gold"] == lab and r["pred"] != lab)
        pr = tp/(tp+fp) if tp+fp else 0
        rc = tp/(tp+fn) if tp+fn else 0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
        cnt = sum(1 for r in results if r["gold"] == lab)
        print(f"    {lab}: 精确 {pr:.2f}  召回 {rc:.2f}  F1 {f1:.2f}  （{cnt} 条）")

    print(f"\n  预测分布: {Counter(r['pred'] for r in results).most_common()}")

    print(f"\n  置信度分层（selective prediction 框架）:")
    for lo, hi, tag in [(0.9, 1.01, "≥0.9"), (0.6, 0.9, "0.6~0.9"), (0.0, 0.6, "<0.6")]:
        sub = [r for r in results if lo <= r["conf"] < hi]
        if sub:
            a = sum(1 for r in sub if r["pred"] == r["gold"]) / len(sub)
            print(f"    {tag:<10} {len(sub):>4} 条 ({len(sub)/n*100:>4.1f}%)  准确率 {a:.1%}")

    json.dump(results, open("D:/ai/logs/eval_verbalizer_emotion.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  明细 D:/ai/logs/eval_verbalizer_emotion.json")


if __name__ == "__main__":
    main()
