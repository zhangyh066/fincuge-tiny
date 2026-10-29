# -*- coding: utf-8 -*-
"""质量指标：schema 合法率 / 每成功一次的成本 / 延迟分位

为什么需要这三个：
    只看准确率会漏掉两类失败——"格式烂到无法解析"和"为了准把延迟拖崩"。
    成熟方案（SLM agent 综述）建议同时看 schema 合法率、cost per successful task、p95 延迟。

数据来源（都是已有缓存，不跑模型）：
    情绪   logs/eval_vb_cache_adapter-emotion-vb.jsonl
    新闻   logs/harness_calib_newscat.jsonl
    实体   logs/harness_calib_entity.jsonl
    延迟   logs/harness_trace.jsonl（线上真实调用记录，样本少时仅供参考）
"""
import json
import os
import statistics as st

CACHES = {
    "情绪": "D:/ai/logs/eval_vb_cache_adapter-emotion-vb.jsonl",
    "新闻": "D:/ai/logs/harness_calib_newscat.jsonl",
    "实体": "D:/ai/logs/harness_calib_entity.jsonl",
}
TRACE = "D:/ai/logs/harness_trace.jsonl"


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def main():
    print(f"{'任务':<6} {'样本':>6} {'准确率':>8} {'schema合法率':>12} {'空预测':>8}")
    lat_all = []
    for name, path in CACHES.items():
        rows = load(path)
        if not rows:
            print(f"{name}: 无缓存")
            continue
        n = len(rows)
        ok = sum(1 for r in rows if r.get("ok", r.get("pred") == r.get("gold")))
        empty = sum(1 for r in rows if not str(r.get("pred", "")).strip())
        # schema 合法：预测非空；实体任务额外要求"在原文里"（缓存里有 in_text 就用）
        if "in_text" in rows[0]:
            valid = sum(1 for r in rows if str(r.get("pred", "")).strip() and r.get("in_text"))
        else:
            valid = n - empty
        print(f"{name:<6} {n:>6} {ok/n:>8.1%} {valid/n:>12.1%} {empty/n:>8.1%}")

    trace = load(TRACE)
    if trace:
        lat = sorted(t.get("latency_ms", 0) for t in trace if t.get("latency_ms") is not None)
        if lat:
            p50 = lat[len(lat) // 2]
            p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
            succ = sum(1 for t in trace if t.get("decision") != "abstain") / len(trace)
            print(f"\n线上调用 {len(trace)} 次：p50 {p50} ms  p95 {p95} ms  未拒答率 {succ:.1%}")
            if succ > 0:
                print(f"每成功一次的平均耗时（≈成本代理指标）: {st.mean(lat)/succ:.0f} ms")
        else:
            print("\n线上 trace 里没有延迟记录")
    else:
        print(f"\n还没有线上调用记录（{TRACE}），延迟和成本指标要等实际用起来才有")


if __name__ == "__main__":
    main()
