# -*- coding: utf-8 -*-
"""收尾脚本：接着 exp_sc_v2.jsonl 的 131 条跑完 200 条

为什么要单独一个文件：沙箱进入 fail-closed 后，**已有文件无法写入**（新建可以），
所以不去改 exp_selfconsistency.py，也不往旧缓存里追加。

输入 logs/exp_sc_v2.jsonl（已有结果）→ 输出 logs/exp_sc_v3.jsonl（新文件）
"""
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from harness import Harness
from train_sft import extract_entity, entity_match

EVAL = "D:/ai/data/fincuge/entity_eval.json"
SRC = "D:/ai/logs/exp_sc_v2.jsonl"
DST = "D:/ai/logs/exp_sc_v3.jsonl"
N, K, TEMP, LOW = 200, 3, 0.7, 0.89


def load(p):
    rows = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    rows = json.load(open(EVAL, encoding="utf-8"))
    idx = sorted(random.Random(0).sample(range(len(rows)), N))

    done = {r["i"]: r for r in load(SRC)}
    done.update({r["i"]: r for r in load(DST)})
    print(f"[续跑] 已有 {len(done)}/{N} 条")

    need = [i for i in idx if i not in done]
    if need:
        h = Harness(verbose=True)
        h._use("entity")
        with open(DST, "a", encoding="utf-8") as f:
            t0 = time.time()
            for n, i in enumerate(need, 1):
                row = rows[i]
                gold = extract_entity(row.get("output", ""))
                ids = h._encode(row)
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
                    samples = []
                    for _ in range(K):
                        o = h.model.generate(ids, max_new_tokens=24, do_sample=True,
                                             temperature=TEMP, top_p=0.9,
                                             pad_token_id=h.tok.pad_token_id,
                                             eos_token_id=h.tok.eos_token_id)
                        samples.append(h.tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip())
                preds = [extract_entity(s) for s in samples]
                rec = {"i": i, "gold": gold, "greedy": extract_entity(greedy),
                       "conf": round(conf, 4), "samples": preds,
                       "vote": Counter(preds).most_common(1)[0][0] if preds else ""}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n % 10 == 0:
                    print(f"  {len(done)+n}/{N}  {time.time()-t0:.0f}s", flush=True)

    recs = [r for r in done.values()] + [r for r in load(DST)]
    uniq = {r["i"]: r for r in recs}
    recs = [uniq[i] for i in idx if i in uniq]
    n = len(recs)

    a = sum(entity_match(r["greedy"], r["gold"])[0] for r in recs) / n
    b = sum(sum(entity_match(p, r["gold"])[0] for p in r["samples"]) / K for r in recs) / n
    c = sum(entity_match(r["vote"], r["gold"])[0] for r in recs) / n
    d = sum(entity_match(r["greedy"] if r["conf"] >= LOW else r["vote"], r["gold"])[0] for r in recs) / n
    low = [r for r in recs if r["conf"] < LOW]
    agree = sum(1 for r in recs if len(set(r["samples"])) == 1) / n
    resc = sum(1 for r in recs if not entity_match(r["greedy"], r["gold"])[0] and entity_match(r["vote"], r["gold"])[0])
    broke = sum(1 for r in recs if entity_match(r["greedy"], r["gold"])[0] and not entity_match(r["vote"], r["gold"])[0])

    print(f"\n{'='*54}")
    print(f"  实体任务 · {n} 条 · K={K} · temperature={TEMP}")
    print(f"{'='*54}")
    print(f"  A greedy 单次（现状）      {a:.1%}")
    print(f"  B 采样单次（3 次平均）      {b:.1%}")
    print(f"  C 采样 {K} 次投票           {c:.1%}")
    print(f"  D 只在 conf<{LOW} 时投票    {d:.1%}")
    print(f"  投票救回 {resc} 条，坑掉 {broke} 条")
    print(f"  3 次采样完全一致的比例      {agree:.1%}")
    if low:
        la = sum(entity_match(r["greedy"], r["gold"])[0] for r in low) / len(low)
        lc = sum(entity_match(r["vote"], r["gold"])[0] for r in low) / len(low)
        print(f"  低置信片段 {len(low)} 条（占 {len(low)/n:.0%}）：greedy {la:.1%} → 投票 {lc:.1%}")


if __name__ == "__main__":
    main()
