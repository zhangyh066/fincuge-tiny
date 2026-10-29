# -*- coding: utf-8 -*-
"""
通用多分类评估：基座 vs 基座+adapter

支持两个任务：
    emotion   情绪三分类（积极/消极/中性）
    newscat7  新闻七分类（公司/新冠/经济/行业/大盘/政策/其它）

用法：
    python eval_multi.py --task newscat7                       # 只测基座
    python eval_multi.py --task newscat7 --adapter 路径         # 基座 + adapter 对比
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

TASKS = {
    "emotion": {
        "eval": "D:/ai/data/fincuge/emotion_eval.json",
        "labels": ["积极", "消极", "中性"],
        "alias": {
            "积极": ["积极", "正面", "乐观", "正向", "看多"],
            "消极": ["消极", "负面", "悲观", "负向", "看空"],
            "中性": ["中性", "中立", "无明显", "无情绪"],
        },
        "gold_from": "output",
    },
    "newscat7": {
        "eval": "D:/ai/data/fincuge/newscat7_eval.json",
        "labels": ["公司", "新冠", "经济", "行业", "大盘", "政策", "其它"],
        "alias": {
            "公司": ["公司", "个股", "组织"],
            "新冠": ["新冠", "疫情"],
            "经济": ["经济"],
            "行业": ["行业", "板块"],
            "大盘": ["大盘", "指数"],
            "政策": ["政策"],
            "其它": ["其它", "其他"],
        },
        "gold_from": "output",
    },
}


def parse_label(text, alias):
    """按别名匹配，优先匹配更长的词（避免'中国'吃掉'国际'这类误判）"""
    t = str(text)
    hits = []
    for lab, words in alias.items():
        for w in words:
            i = t.find(w)
            if i >= 0:
                hits.append((-len(w), i, lab))   # 长词优先
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def load_model(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True
    )
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m, tok


def run_eval(model, tok, samples, alias, tag):
    has_think = "enable_thinking" in (tok.chat_template or "")
    preds = []
    t0 = time.time()
    with torch.inference_mode():
        for i, s in enumerate(samples, 1):
            # ⚠ 必须和训练时一模一样，统一走 prompt_util
            #   （这里原来手写 "\n\n新闻："，训练却是 "\n"，又是一处不一致）
            msgs = [{"role": "user", "content": build_user_text(s)}]
            kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
            if has_think:
                kw["enable_thinking"] = False
            enc = tok.apply_chat_template(msgs, **kw)
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            ids = ids.cuda()
            out = model.generate(ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            preds.append({"pred": parse_label(raw, alias), "raw": raw})
            if i % 200 == 0:
                acc = sum(1 for j, p in enumerate(preds) if p["pred"] == samples[j]["gold"]) / i
                print(f"  [{tag}] {i}/{len(samples)}  {acc:.1%}  {time.time()-t0:.0f}s")
    return preds


def report(name, samples, preds, labels):
    n = len(samples)
    correct = sum(1 for s, p in zip(samples, preds) if p["pred"] == s["gold"])
    unp = sum(1 for p in preds if p["pred"] is None)
    acc = correct / n
    print(f"\n【{name}】{correct}/{n} = {acc:.1%}（未解析 {unp}）")
    print(f"  {'类别':<6}{'精确':>7}{'召回':>7}{'F1':>7}{'样本':>7}")
    f1s = {}
    for lab in labels:
        tp = sum(1 for s, p in zip(samples, preds) if s["gold"] == lab and p["pred"] == lab)
        fp = sum(1 for s, p in zip(samples, preds) if s["gold"] != lab and p["pred"] == lab)
        fn = sum(1 for s, p in zip(samples, preds) if s["gold"] == lab and p["pred"] != lab)
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        f1s[lab] = f1
        cnt = sum(1 for s in samples if s["gold"] == lab)
        print(f"  {lab:<6}{pr:>7.2f}{rc:>7.2f}{f1:>7.2f}{cnt:>7}")

    print(f"  混淆矩阵（行=标准）：")
    print(f"  {'':>6}" + "".join(f"{c:>7}" for c in labels) + f"{'未解析':>7}")
    for g in labels:
        row = [sum(1 for s, p in zip(samples, preds) if s["gold"] == g and p["pred"] == c) for c in labels]
        unp_g = sum(1 for s, p in zip(samples, preds) if s["gold"] == g and p["pred"] is None)
        print(f"  {g:>6}" + "".join(f"{v:>7}" for v in row) + f"{unp_g:>7}")
    return acc, f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="newscat7", choices=list(TASKS))
    ap.add_argument("--adapter", default=None)
    # 断电/赶时间时用 --limit 抽样评估，默认 0 = 全量
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = TASKS[args.task]
    alias = cfg["alias"]
    labels = cfg["labels"]
    raw = json.load(open(cfg["eval"], encoding="utf-8"))

    samples = []
    for r in raw:
        g = parse_label(r[cfg["gold_from"]], alias)
        if g:
            samples.append({**r, "gold": g})
    # 分层截断：保证每个类别都保留样本，而不是只取前 N 条（会导致类别偏斜）
    if args.limit and args.limit < len(samples):
        by = {}
        for s in samples: by.setdefault(s['gold'], []).append(s)
        per = max(1, args.limit // len(by))
        picked = []
        for lab, arr in by.items(): picked.extend(arr[:per])
        samples = picked[:args.limit]
    print(f"任务 {args.task}  评估 {len(samples)} 条")
    dist = Counter(s["gold"] for s in samples)
    print(f"  分布 {dist.most_common()}")
    maj, cnt = dist.most_common(1)[0]
    print(f"  全猜「{maj}」的基线 = {cnt/len(samples):.1%}\n")

    print("加载基座…")
    m, tok = load_model()
    pb = run_eval(m, tok, samples, alias, "基座")
    acc_b, f1_b = report("基座", samples, pb, labels)
    del m
    torch.cuda.empty_cache()

    if not args.adapter:
        return

    print(f"\n加载 adapter {args.adapter} …")
    m, tok = load_model(args.adapter)
    pf = run_eval(m, tok, samples, alias, "微调后")
    acc_f, f1_f = report("微调后", samples, pf, labels)

    print("\n" + "=" * 60)
    print(f"  准确率 {acc_b:.1%} → {acc_f:.1%}  ({acc_f-acc_b:+.1%})")
    for lab in labels:
        print(f"  {lab} F1  {f1_b[lab]:.2f} → {f1_f[lab]:.2f}  ({f1_f[lab]-f1_b[lab]:+.2f})")
    print("=" * 60)

    out = f"D:/ai/logs/eval_{args.task}_compare.json"
    json.dump({
        "n": len(samples), "base": {"acc": acc_b, "f1": f1_b}, "ft": {"acc": acc_f, "f1": f1_f},
        "samples": [{"input": s["input"], "gold": s["gold"], "base": a["pred"], "ft": b["pred"]}
                    for s, a, b in zip(samples, pb, pf)],
    }, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"明细 {out}")


if __name__ == "__main__":
    main()
