# -*- coding: utf-8 -*-
"""FINESE 实体抽取 · 训练前数据体检（只读，不改任何数据）

查四件事，全是"训练 2 小时之前必须先知道"的：
  1. 答案能不能被 extract_entity 正确抽出实体（抽不出 = 判分器坏，训了也白训）
  2. 抽出的实体是否真的出现在原文 input 里（不在原文 = 模型没法"抄"，只能靠记，几乎不可能学会）
  3. 答案模板有多少种写法（决定模型要学会"套模板"还是只学实体）
  4. 答案要多少 token（决定 max_new_tokens=32 会不会截断）
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sft import extract_entity, entity_match   # 判分器唯一真源

TRAIN = "D:/ai/data/fincuge/entity_train.json"
EVAL = "D:/ai/data/fincuge/entity_eval.json"
BASE = "D:/ai/models/Qwen3-0.6B"


def audit(path, tok=None):
    rows = json.load(open(path, encoding="utf-8"))
    n = len(rows)
    fail, not_in_text, empty = [], [], 0
    tpl = Counter()
    tok_lens = []
    ents = []

    for r in rows:
        out, inp = str(r["output"]), str(r["input"])
        ent = extract_entity(out)
        if not ent:
            fail.append(r)
            continue
        ents.append(ent)
        if ent not in inp:
            not_in_text.append((ent, inp[:40]))
        # 模板 = 答案把实体挖掉后剩下的骨架
        tpl[out.replace(ent, "<E>")] += 1
        if tok is not None:
            tok_lens.append(len(tok.encode(out, add_special_tokens=False)))

    print(f"\n{'='*66}")
    print(f"  {os.path.basename(path)}  共 {n} 条")
    print(f"{'='*66}")
    print(f"  1) extract_entity 抽不出实体      : {len(fail):>5} 条 ({len(fail)/n:.1%})")
    print(f"  2) 实体不在原文 input 里          : {len(not_in_text):>5} 条 ({len(not_in_text)/n:.1%})")
    print(f"  3) 答案模板种类（实体挖掉后骨架）  : {len(tpl):>5} 种")
    for t, c in tpl.most_common(5):
        print(f"       {c:>5}×  {t}")
    if tok_lens:
        tok_lens.sort()
        p = lambda q: tok_lens[min(len(tok_lens) - 1, int(len(tok_lens) * q))]
        print(f"  4) 答案 token 数  P50={p(.5)}  P95={p(.95)}  P99={p(.99)}  最长={tok_lens[-1]}")
        print(f"       → max_new_tokens=32 会截断 {sum(1 for x in tok_lens if x > 32)/len(tok_lens):.1%} 的答案")
    if not_in_text:
        print("     不在原文的样例（前 5）：")
        for e, s in not_in_text[:5]:
            print(f"       实体={e!r}  原文={s!r}")
    if fail:
        print("     抽不出的样例（前 3）：")
        for r in fail[:3]:
            print(f"       output={r['output'][:60]!r}")
    return {"n": n, "fail": len(fail), "not_in": len(not_in_text), "tpl": len(tpl)}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    a = audit(TRAIN, tok)
    b = audit(EVAL, tok)

    print(f"\n{'='*66}")
    print("  结论")
    print(f"{'='*66}")
    if a["fail"] or b["fail"]:
        print("  ✗ 判分器有抽不出的样本 → 先修 extract_entity，别急着训")
    else:
        print("  ✓ 判分器 100% 能抽出实体")
    if a["not_in"]:
        print(f"  ⚠ 训练集 {a['not_in']} 条实体不在原文 → 这部分模型只能死记，是准确率天花板的一部分")
    else:
        print("  ✓ 训练集实体全部在原文里 → 模型可以照抄，任务可学")
    print(f"  答案模板 {a['tpl']} 种（训练集）/{b['tpl']} 种（评估集）"
          f" → {'模板多，模型要学会套模板' if a['tpl'] > 50 else '模板不多，模型学得动'}")


if __name__ == "__main__":
    main()
