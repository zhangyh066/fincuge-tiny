# -*- coding: utf-8 -*-
"""
adapter 评估：在完整 eval 集上对比「基座」与「基座+adapter」

为什么要跑完整 eval 而不是抽样：
    抽 300 条，准确率的 95% 置信区间大约 ±5.7 个点。
    如果微调只涨 3 个点，抽样根本分不出来是提升还是噪声。
    2020 条的置信区间约 ±2.2 个点，能看出真实差异。

输出什么：
    1. 总体准确率（基座 vs adapter）
    2. 三类的精确率/召回率/F1 对比
    3. 混淆矩阵（看中性有没有被修复）
    4. 错误样例（看模型错在哪）

用法：
    D:\\ai\\env.bat
    python eval_adapter.py --adapter D:/ai/out/adapter-emotion
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
EVAL_JSON = "D:/ai/data/fincuge/emotion_eval.json"
OUT_JSON = "D:/ai/logs/eval_emotion_compare.json"

LABELS = ["积极", "消极", "中性"]

# 同义词映射 —— 第一版评估就是漏了这个，把"负面"判成无法解析，准确率被低估 20 个点
SYNONYMS = {
    "积极": ["积极", "正面", "乐观", "正向", "看多"],
    "消极": ["消极", "负面", "悲观", "负向", "看空"],
    "中性": ["中性", "中立", "无明显", "无情绪", "持平"],
}


def parse_label(text):
    t = str(text)
    hits = []
    for lab, words in SYNONYMS.items():
        for w in words:
            i = t.find(w)
            if i >= 0:
                hits.append((i, lab))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


DEV = "cuda:0"          # main 里会根据 --cpu 改成 "cpu"
DTYPE = torch.float16


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=DTYPE, device_map={"": DEV}, trust_remote_code=True
    )
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m, tok


def evaluate(model, tok, samples, tag):
    """跑完所有样本，返回预测列表"""
    has_think = "enable_thinking" in (tok.chat_template or "")
    preds = []
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
            ids = ids.to(DEV)
            out = model.generate(
                ids, max_new_tokens=16, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
            raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            preds.append({"pred": parse_label(raw), "raw": raw})

            if i % 400 == 0:
                acc = sum(1 for j, p in enumerate(preds) if p["pred"] == samples[j]["gold"]) / i
                print(f"  [{tag}] {i}/{len(samples)}  累计 {acc:.1%}  {time.time()-t0:.0f}s")
    return preds


def report(name, samples, preds):
    n = len(samples)
    correct = sum(1 for s, p in zip(samples, preds) if p["pred"] == s["gold"])
    unparsed = sum(1 for p in preds if p["pred"] is None)
    acc = correct / n
    print(f"\n【{name}】准确率 {correct}/{n} = {acc:.1%}（无法解析 {unparsed} 条）")

    print(f"  {'类别':<6}{'精确率':>7}{'召回率':>7}{'F1':>7}{'样本数':>7}")
    f1s = {}
    for lab in LABELS:
        tp = sum(1 for s, p in zip(samples, preds) if s["gold"] == lab and p["pred"] == lab)
        fp = sum(1 for s, p in zip(samples, preds) if s["gold"] != lab and p["pred"] == lab)
        fn = sum(1 for s, p in zip(samples, preds) if s["gold"] == lab and p["pred"] != lab)
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        f1s[lab] = f1
        print(f"  {lab:<6}{pr:>7.2f}{rc:>7.2f}{f1:>7.2f}{sum(1 for s in samples if s['gold']==lab):>7}")

    print(f"  混淆矩阵（行=标准，列=预测）:")
    print(f"  {'':>6}" + "".join(f"{c:>8}" for c in LABELS) + f"{'未解析':>8}")
    for g in LABELS:
        row = [sum(1 for s, p in zip(samples, preds) if s["gold"] == g and p["pred"] == c) for c in LABELS]
        unp = sum(1 for s, p in zip(samples, preds) if s["gold"] == g and p["pred"] is None)
        print(f"  {g:>6}" + "".join(f"{v:>8}" for v in row) + f"{unp:>8}")
    return acc, f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="D:/ai/out/adapter-emotion")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="用 CPU 跑，不占显卡（慢很多，但显卡忙的时候能用）")
    ap.add_argument("--skip_base", action="store_true", help="只测微调后，不跑基座（省一半时间）")
    ap.add_argument("--tag", default="", help="结果文件名后缀，防止覆盖历史结果，例：--tag promptfix")
    args = ap.parse_args()

    # 输出文件名加后缀，避免把旧结果（比如 72.2% 那份）覆盖掉
    out_json = OUT_JSON
    if args.tag:
        out_json = OUT_JSON.replace(".json", f"_{args.tag}.json")

    global DEV, DTYPE
    if args.cpu:
        DEV, DTYPE = "cpu", torch.float32

    raw = json.load(open(EVAL_JSON, encoding="utf-8"))
    samples = []
    for r in raw:
        lab = parse_label(r["output"])
        if lab:
            samples.append({**r, "gold": lab})
    if args.limit:
        samples = samples[:args.limit]
    print(f"评估集 {len(samples)} 条  分布 {Counter(s['gold'] for s in samples).most_common()}")
    print(f"  （全猜'积极'的基线 = {sum(1 for s in samples if s['gold']=='积极')/len(samples):.1%}）\n")

    # ---- 基座 ----
    if args.skip_base:
        print("（已跳过基座评估，只测微调后）\n")
        preds_base = None
        acc_b = f1_b = None
    else:
        print("加载基座…")
        m, tok = load()
        preds_base = evaluate(m, tok, samples, "基座")
        acc_b, f1_b = report("基座（未微调）", samples, preds_base)
        del m
        torch.cuda.empty_cache()

    if args.base_only:
        return

    # ---- 基座 + adapter ----
    print("\n加载基座 + adapter…")
    m, tok = load(args.adapter)
    preds_ft = evaluate(m, tok, samples, "微调后")
    acc_f, f1_f = report("微调后", samples, preds_ft)

    # ---- 对比 ----
    if args.skip_base:
        print("\n" + "=" * 62)
        print("  微调后结果（没跑基座，无对比）")
        print("=" * 62)
        print(f"  总体准确率:  {acc_f:.1%}")
        for lab in LABELS:
            print(f"  {lab} F1:     {f1_f[lab]:.2f}")
    else:
        print("\n" + "=" * 62)
        print("  对比")
        print("=" * 62)
        print(f"  总体准确率:  {acc_b:.1%}  →  {acc_f:.1%}   ({acc_f-acc_b:+.1%})")
        for lab in LABELS:
            print(f"  {lab} F1:     {f1_b[lab]:.2f}  →  {f1_f[lab]:.2f}   ({f1_f[lab]-f1_b[lab]:+.2f})")

    ok = acc_f >= 0.75 and f1_f["中性"] >= 0.55
    print(f"\n  达标线（总体≥75% 且 中性F1≥0.55）:  {'✅ 通过' if ok else '❌ 未通过'}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n": len(samples),
            "base": {"acc": acc_b, "f1": f1_b},
            "finetuned": {"acc": acc_f, "f1": f1_f},
            "samples": [{"input": s["input"], "gold": s["gold"],
                         "base": pb["pred"] if pb else None,
                         "ft": pf["pred"], "ft_raw": pf["raw"]}
                        for s, pb, pf in zip(samples, preds_base or [None] * len(samples), preds_ft)],
        }, f, ensure_ascii=False, indent=1)
    print(f"  明细已存 {out_json}")


if __name__ == "__main__":
    main()
