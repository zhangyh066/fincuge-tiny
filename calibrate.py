# -*- coding: utf-8 -*-
"""阈值标定：把三条「置信度-对错」记录变成可用的三档阈值

数据来源（都是逐条落盘的标定缓存）：
    情绪   logs/eval_vb_cache_adapter-emotion-vb.jsonl
    新闻   logs/harness_calib_newscat.jsonl
    实体   logs/harness_calib_entity.jsonl

选法（先定规则再看数，避免事后挑参数）：
    θ_hi：自动通过档的准确率 ≥95%，且覆盖率不低于 30%，取满足条件的最高阈值
    θ_lo：拒答档（conf < θ_lo）的准确率 ≤60%，取满足条件的最低阈值
"""
import json
import os

CACHES = {
    "emotion": "D:/ai/logs/eval_vb_cache_adapter-emotion-vb.jsonl",
    "newscat": "D:/ai/logs/harness_calib_newscat.jsonl",
    "entity": "D:/ai/logs/harness_calib_entity.jsonl",
}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "ok" not in r:
                r["ok"] = (r["pred"] == r["gold"])
            if r.get("gold") is None:
                continue
            rows.append(r)
    return rows


def main():
    print(f"{'任务':<8} {'样本':>6} {'总准确率':>8}")
    for task, path in CACHES.items():
        rows = load(path)
        n = len(rows)
        if n == 0:
            print(f"{task}: 无数据")
            continue
        acc = sum(r["ok"] for r in rows) / n
        print(f"\n=== {task}  n={n}  总体 {acc:.1%} ===")
        for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            sub = [r for r in rows if r["conf"] >= t]
            if not sub:
                print(f"  θ≥{t:.2f}  覆盖   0.0%")
                continue
            a = sum(r["ok"] for r in sub) / len(sub)
            print(f"  θ≥{t:.2f}  覆盖 {len(sub)/n:5.1%}  该档准确率 {a:5.1%}")

        # θ_hi：最高阈值，使该档准确率≥95% 且覆盖率≥30%
        hi = None
        for i in range(99, 49, -1):
            t = i / 100
            sub = [r for r in rows if r["conf"] >= t]
            if len(sub) / n < 0.30:
                continue
            if sum(r["ok"] for r in sub) / len(sub) >= 0.95:
                hi = (t, len(sub) / n, sum(r["ok"] for r in sub) / len(sub))
                break
        # θ_lo：最低阈值，使 conf<θ 的样本准确率≤60%
        lo = None
        for i in range(50, 100):
            t = i / 100
            sub = [r for r in rows if r["conf"] < t]
            if len(sub) < 20:
                continue
            if sum(r["ok"] for r in sub) / len(sub) <= 0.60:
                lo = (t, len(sub) / n, sum(r["ok"] for r in sub) / len(sub))
                break

        if hi:
            print(f"  → θ_hi = {hi[0]:.2f}   覆盖 {hi[1]:.1%}，该档准确率 {hi[2]:.1%}")
        else:
            print("  → 找不到满足条件的 θ_hi（没有哪一档能到 95%），建议保持 hint 为主")
        if lo:
            print(f"  → θ_lo = {lo[0]:.2f}   拒答 {lo[1]:.1%}，该档准确率 {lo[2]:.1%}")
        else:
            print("  → 找不到准确率 ≤60% 的低置信档，不建议设拒答线")


if __name__ == "__main__":
    main()
