# -*- coding: utf-8 -*-
"""
基座评估：Qwen3-0.6B 在金融舆情情绪分类（FINFE）上的真实水平

为什么必须先做这一步：
    这次微调的目标是"提升情绪分类准确率"。
    但如果基座本来就能做到 80%，微调既没必要、还可能训坏。
    先量出基线的真实数字，才有资格谈"要不要训"。

评估设计（防止自己骗自己）：
    1. 分层抽样 300 条（积极/消极/中性 各 100），避免类别不均衡导致的虚高
    2. 报三个数：总体准确率、各类 F1、混淆矩阵
    3. 对比"全猜积极"的基线（44.2%），低于这个数说明模型纯属瞎猜

用法：
    D:\\ai\\env.bat
    D:\\ai\\venv\\Scripts\\python.exe D:\\ai\\eval_baseline.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import re
import time
from collections import Counter, defaultdict

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_util import build_user_text   # 提示词唯一真源，和训练共用

MODEL = "D:/ai/models/Qwen3-0.6B"
EVAL_PARQUET = "D:/ai/data/candidates/fincuge_eval.parquet"
TASK = "FINFE"
PER_CLASS = 100          # 每类抽多少条
MAX_NEW = 24             # 输出很短，24 个 token 够
OUT_JSON = "D:/ai/logs/eval_baseline_finfe.json"

LABELS = ["积极", "消极", "中性"]

# ⚠ 这里踩过一个坑：第一版只匹配"积极/消极/中性"三个词，
# 结果模型输出的是"负面""正面"，44% 被判成"无法解析"，准确率被严重低估。
# 教训：写解析规则前，先跑 20 条看看模型到底输出什么格式。
SYNONYMS = {
    "积极": ["积极", "正面", "乐观", "正向", "看多"],
    "消极": ["消极", "负面", "悲观", "负向", "看空"],
    "中性": ["中性", "中立", "无明显", "无情绪", "持平"],
}


def parse_label(text: str):
    """从模型输出里抠标签，支持同义词。抠不到返回 None。"""
    t = str(text)
    # 按位置找最早出现的那个词，避免"不是消极"这类否定句被误判
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


def main():
    df = pd.read_parquet(EVAL_PARQUET)
    df = df[df["task"] == TASK].copy()
    df["label"] = df["output"].apply(parse_label)

    # ---- 分层抽样 ----
    samples = []
    for lab in LABELS:
        sub = df[df["label"] == lab]
        n = min(PER_CLASS, len(sub))
        samples.extend(sub.sample(n=n, random_state=42).to_dict("records"))
    print(f"[1/3] 抽样 {len(samples)} 条（每类 {PER_CLASS}）")
    print("     ", Counter(s["label"] for s in samples).most_common())

    # ---- 加载模型 ----
    print(f"[2/3] 加载 {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True
    ).eval()

    SYS = "你是一个金融文本情绪分析助手。只判断情绪极性，不做其他分析。"
    # ⚠ 训练时没有 system 消息，所以基线这里默认也不加。
    #    否则「微调前 vs 微调后」喂的不是同一个输入，比出来的是提示词差异，不是模型差异。
    #    真要单独试探 system 提示的作用，把下面改成 True。
    WITH_SYSTEM = False
    HAS_THINK = "enable_thinking" in (tok.chat_template or "")

    @torch.inference_mode()
    def infer(instruction, text):
        msgs = []
        if WITH_SYSTEM:
            msgs.append({"role": "system", "content": SYS})
        # 提示词统一走 prompt_util（以前手写 "\n\n文本："，和训练不一致）
        msgs.append({"role": "user",
                     "content": build_user_text({"instruction": instruction, "input": text})})
        kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        if HAS_THINK:
            kw["enable_thinking"] = False
        enc = tok.apply_chat_template(msgs, **kw)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        ids = ids.cuda()
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,     # 贪心解码，保证可复现
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    # ---- 逐条评估 ----
    print(f"[3/3] 评估中（{len(samples)} 条，贪心解码）\n")
    t0 = time.time()
    records = []
    for i, s in enumerate(samples, 1):
        raw = infer(str(s["instruction"]), str(s["input"]))
        pred = parse_label(raw)
        records.append({
            "input": str(s["input"]),
            "gold": s["label"],
            "raw_output": raw,
            "pred": pred,
        })
        if i % 50 == 0:
            acc = sum(1 for r in records if r["pred"] == r["gold"]) / len(records)
            print(f"  {i}/{len(samples)}  累计准确率 {acc:.1%}  已用 {time.time()-t0:.0f}s")

    # ---- 统计 ----
    n = len(records)
    correct = sum(1 for r in records if r["pred"] == r["gold"])
    unparsed = sum(1 for r in records if r["pred"] is None)

    print("\n" + "=" * 62)
    print(f"  总体准确率: {correct}/{n} = {correct/n:.1%}")
    print(f"  无法解析输出: {unparsed} 条 ({unparsed/n:.1%})")
    print(f"  参考基线（全猜'积极'）: {sum(1 for r in records if r['gold']=='积极')/n:.1%}")
    print("=" * 62)

    # 混淆矩阵
    print("\n混淆矩阵（行=标准答案，列=模型判断）:")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in LABELS) + f"{'未解析':>8}")
    for g in LABELS:
        row = [sum(1 for r in records if r["gold"] == g and r["pred"] == p) for p in LABELS]
        unp = sum(1 for r in records if r["gold"] == g and r["pred"] is None)
        print(f"{g:>8}" + "".join(f"{v:>8}" for v in row) + f"{unp:>8}")

    # 各类 F1
    print("\n各类指标:")
    for lab in LABELS:
        tp = sum(1 for r in records if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in records if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in records if r["gold"] == lab and r["pred"] != lab)
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        print(f"  {lab}: 精确率 {p:.2f}  召回率 {rc:.2f}  F1 {f1:.2f}")

    # 保存明细
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL, "task": TASK, "n": n,
            "accuracy": correct / n,
            "majority_baseline": sum(1 for r in records if r['gold'] == '积极') / n,
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n明细已存: {OUT_JSON}")

    # 错误样例
    print("\n错的最典型的 5 条:")
    wrong = [r for r in records if r["pred"] != r["gold"]][:5]
    for r in wrong:
        print(f"  输入: {r['input'][:40]}")
        print(f"    标准 {r['gold']} / 模型 {r['pred']} / 原输出: {r['raw_output'][:50]}")


if __name__ == "__main__":
    main()
