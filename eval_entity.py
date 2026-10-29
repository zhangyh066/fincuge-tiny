# -*- coding: utf-8 -*-
"""FinESE 事件主体抽取 · 全量评估（对齐原方案的指标口径）

原方案怎么评（2026-10-29 查证）：
    DISC-FinLLM（arXiv 2310.15205）用 **F1**；Golden Touchstone（arXiv 2411.06272）
    因为原数据缺 BIO 标注，改用 **ORMACC（整体正则匹配准确率）**。
    两者本质都是"生成答案 → 取出主体 → 跟金标比"，本脚本同时给这两个口径。

输出四项：
    1. 严格匹配准确率  —— 去标点后完全相同（≈ ORMACC 严格版，主指标）
    2. 宽松匹配准确率  —— 互相包含也算对（"雅虎公司" vs "雅虎"）
    3. 实体级 P / R / F1 —— 论文口径
    4. 幻觉率          —— 预测的主体根本不在原文里（生成式任务最危险的错法）

工程说明：断点续跑。1,464 条生成约 15 分钟，而 agent 的命令通道会被掐断，
    所以每 50 条落盘一次 logs/eval_entity_cache_<adapter>.jsonl，重跑自动续。

用法：
    python eval_entity.py --adapter D:/ai/out/adapter-entity
    python eval_entity.py --adapter ""            # 评基座（零样本对照）
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from prompt_util import build_user_text
from train_sft import extract_entity, entity_match

BASE = "D:/ai/models/Qwen3-0.6B"
EVAL_JSON = "D:/ai/data/fincuge/entity_eval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="D:/ai/out/adapter-entity")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量 1,464 条")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    dev = "cpu" if args.cpu else "cuda:0"
    dt = torch.float32 if args.cpu else torch.float16

    samples = json.load(open(EVAL_JSON, encoding="utf-8"))
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    print(f"评估 {len(samples)} 条（FinESE 事件主体抽取）")

    tag = os.path.basename(os.path.normpath(args.adapter)) if args.adapter else "base"
    cache_path = f"D:/ai/logs/eval_entity_cache_{tag}.jsonl"
    done = {}
    if args.resume and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                done[rec["i"]] = rec
        print(f"  [续跑] 已有 {len(done)} 条")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dt, device_map={"": dev},
                                                trust_remote_code=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    has_think = "enable_thinking" in (tok.chat_template or "")

    results = [None] * len(samples)
    for i, r in done.items():
        results[i] = r

    f = open(cache_path, "a", encoding="utf-8")
    t0 = time.time()
    n_done = len(done)

    with torch.inference_mode():
        for i, s in enumerate(samples):
            if results[i] is not None:
                continue
            msgs = [{"role": "user", "content": build_user_text(s)}]
            kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
            if has_think:
                kw["enable_thinking"] = False
            enc = tok.apply_chat_template(msgs, **kw)
            ids = enc["input_ids"] if hasattr(enc, "keys") else torch.tensor([enc])
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(dev)
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

            gold_text = s.get("output", "")
            p, g = extract_entity(gen), extract_entity(gold_text)
            ok_s, ok_l = entity_match(p, g)
            rec = {"i": i, "gen": gen, "pred": p, "gold": g,
                   "strict": bool(ok_s), "loose": bool(ok_l),
                   "in_text": (p in s.get("input", "")) if p else False}
            results[i] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                f.flush()
                acc = sum(1 for r in results if r and r["strict"]) / n_done
                print(f"  {n_done}/{len(samples)}  严格 {acc:.1%}  {time.time()-t0:.0f}s")
    f.close()

    results = [r for r in results if r]
    n = len(results)
    if n < len(samples):
        print(f"\n  ⚠ 只完成 {n}/{len(samples)} 条，重跑同一命令会从断点继续")
        return

    strict = sum(r["strict"] for r in results) / n
    loose = sum(r["loose"] for r in results) / n
    tp = sum(1 for r in results if r["strict"])
    fp = n - tp                        # 预测错 = 假阳（每样本只有一个实体）
    fn = n - tp                        # 该对没对 = 漏
    prec = tp / (tp + fp) if tp + fp else 0
    rec_ = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0
    hallu = sum(1 for r in results if r["pred"] and not r["in_text"]) / n
    empty = sum(1 for r in results if not r["pred"]) / n

    print(f"\n{'='*60}")
    print(f"  FinESE 事件主体抽取 · {tag} · {n} 条")
    print(f"{'='*60}")
    print(f"  严格匹配准确率（≈ORMACC）: {strict:.1%}")
    print(f"  宽松匹配准确率            : {loose:.1%}")
    print(f"  实体级 P / R / F1（论文口径）: {prec:.3f} / {rec_:.3f} / {f1:.3f}")
    print(f"  幻觉率（预测不在原文）    : {hallu:.1%}")
    print(f"  空预测率                  : {empty:.1%}")
    print(f"  耗时 {time.time()-t0:.0f}s")
    print(f"\n  明细 {cache_path}")

    print("\n  错例（前 8 条，看是抄错了还是抄歪了）：")
    bad = [r for r in results if not r["strict"]][:8]
    for r in bad:
        print(f"    gold={r['gold']!r}  pred={r['pred']!r}  原文里有={r['in_text']}")
        print(f"      生成={r['gen'][:70]!r}")


if __name__ == "__main__":
    main()
