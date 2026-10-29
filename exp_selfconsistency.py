# -*- coding: utf-8 -*-
"""实验：0.6B 上自一致性投票到底有没有用

背景：想用"采样 3 次投票"救低置信样本，但如果 0.6B 的错误是系统性的
（每次都挑同一个错公司名），投票就是白花钱。所以先测，再决定要不要上。

对照四组（同一批 200 条实体评估样本）：
    A greedy 单次（就是现在线上的做法）
    B 采样单次（temperature=0.7, top_p=0.9）
    C 采样 3 次投票（多数票；平票取第一个）
    D 只在 greedy 置信度 < 0.89 时投票，其余沿用 greedy

带断点续跑，被系统掐断后重跑同一条命令会接着来。
"""
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from harness import Harness, SKILLS, gold_of
from train_sft import extract_entity, entity_match

EVAL = "D:/ai/data/fincuge/entity_eval.json"
CACHE = "D:/ai/logs/exp_selfconsistency.jsonl"
N = 200
K = 3
TEMP = 0.7
LOW_CONF = 0.89


class _Tee:
    """同时写控制台和日志文件，避免依赖 PowerShell 的 Tee-Object（编码不可控）"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                if "\n" in s:
                    st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def main():
    os.makedirs("D:/ai/logs", exist_ok=True)
    try:
        _log = open("D:/ai/logs/exp_sc_run.log", "a", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, _log)
    except Exception as e:
        print(f"[提示] 日志文件写不了（{e}），只输出到控制台")
    rows = json.load(open(EVAL, encoding="utf-8"))
    idx = sorted(random.Random(0).sample(range(len(rows)), N))

    done = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["i"]] = r
        print(f"[续跑] 已有 {len(done)}/{N} 条")

    h = Harness(verbose=True)
    h._use("entity")
    f = open(CACHE, "a", encoding="utf-8")
    t0 = time.time()

    for k, i in enumerate(idx):
        if i in done:
            continue
        row = rows[i]
        gold = extract_entity(row.get("output", ""))
        ids = h._encode(row)

        # A greedy
        with torch.inference_mode():
            out = h.model.generate(ids, max_new_tokens=24, do_sample=False,
                                   pad_token_id=h.tok.pad_token_id,
                                   eos_token_id=h.tok.eos_token_id,
                                   return_dict_in_generate=True, output_scores=True)
            seq = out.sequences[0][ids.shape[1]:]
            greedy = h.tok.decode(seq, skip_special_tokens=True).strip()
            lps = []
            for t, sc in enumerate(out.scores):
                if t >= len(seq):
                    break
                lp = torch.log_softmax(sc[0].float(), dim=-1)
                lps.append(lp[seq[t]].item())
            conf = float(torch.tensor(lps).mean().exp()) if lps else 0.0

            # B/C 采样 K 次
            samples = []
            for _ in range(K):
                o = h.model.generate(ids, max_new_tokens=24, do_sample=True,
                                     temperature=TEMP, top_p=0.9,
                                     pad_token_id=h.tok.pad_token_id,
                                     eos_token_id=h.tok.eos_token_id)
                samples.append(h.tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip())

        preds = [extract_entity(s) for s in samples]
        vote = Counter(preds).most_common(1)[0][0] if preds else ""

        rec = {"i": i, "gold": gold, "greedy": extract_entity(greedy),
               "conf": round(conf, 4), "samples": preds, "vote": vote}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        n = len(done) + k + 1
        if n % 20 == 0:
            print(f"  {n}/{N}  {time.time()-t0:.0f}s")
    f.close()

    recs = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["i"] in set(idx)]
    n = len(recs)
    a = sum(entity_match(r["greedy"], r["gold"])[0] for r in recs) / n
    b = sum(sum(entity_match(p, r["gold"])[0] for p in r["samples"]) / K for r in recs) / n
    c = sum(entity_match(r["vote"], r["gold"])[0] for r in recs) / n
    d = sum(entity_match(r["greedy"] if r["conf"] >= LOW_CONF else r["vote"], r["gold"])[0]
            for r in recs) / n

    print(f"\n{'='*56}")
    print(f"  实体任务 · {n} 条（随机种子 0）· K={K}  temperature={TEMP}")
    print(f"{'='*56}")
    print(f"  A greedy 单次（现状）        {a:.1%}")
    print(f"  B 采样单次（3 次的平均）      {b:.1%}")
    print(f"  C 采样 {K} 次投票             {c:.1%}")
    print(f"  D 只在 conf<{LOW_CONF} 时投票  {d:.1%}")
    low = [r for r in recs if r["conf"] < LOW_CONF]
    if low:
        la = sum(entity_match(r["greedy"], r["gold"])[0] for r in low) / len(low)
        lc = sum(entity_match(r["vote"], r["gold"])[0] for r in low) / len(low)
        print(f"\n  只看低置信片段（{len(low)} 条，占 {len(low)/n:.0%}）：greedy {la:.1%} → 投票 {lc:.1%}")
    agree = sum(1 for r in recs if len(set(r["samples"])) == 1) / n
    print(f"  3 次采样完全一致的比例        {agree:.1%}（越高说明错误越系统性）")


if __name__ == "__main__":
    main()
