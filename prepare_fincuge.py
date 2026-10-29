# -*- coding: utf-8 -*-
"""
把 FinCUGE 的三个任务转成训练/评估格式（instruction / input / output）

只转我们真正要训的三个：
    FINFE  论坛情绪分析      → adapter「情绪」
    FINNL  新闻分类          → adapter「新闻分类」
    FINESE 事件主体抽取      → adapter「主体抽取」

其他任务不转：
    FINNA 摘要是生成任务，0.6B 弱项
    FINRE 数据质量差（大量"没有提及"）
    FINQA / FINCQA 正文挤在 instruction 里，格式不规整
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import os
from collections import Counter

import pandas as pd

SRC_TRAIN = "D:/ai/data/candidates/fincuge_train.parquet"
SRC_EVAL = "D:/ai/data/candidates/fincuge_eval.parquet"
OUT_DIR = "D:/ai/data/fincuge"

TASKS = {
    "FINFE": "emotion",
    "FINNL": "newscat",
    "FINESE": "entity",
}


def convert(df, task):
    sub = df[df["task"] == task]
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "instruction": str(r["instruction"]).strip(),
            "input": str(r["input"]).strip(),
            "output": str(r["output"]).strip(),
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tr = pd.read_parquet(SRC_TRAIN)
    ev = pd.read_parquet(SRC_EVAL)

    for task, name in TASKS.items():
        tr_rows = convert(tr, task)
        ev_rows = convert(ev, task)

        p_tr = f"{OUT_DIR}/{name}_train.json"
        p_ev = f"{OUT_DIR}/{name}_eval.json"
        with open(p_tr, "w", encoding="utf-8") as f:
            json.dump(tr_rows, f, ensure_ascii=False, indent=1)
        with open(p_ev, "w", encoding="utf-8") as f:
            json.dump(ev_rows, f, ensure_ascii=False, indent=1)

        # 统计输出标签分布，确认类别情况
        print(f"{task} → {name}")
        print(f"  训练 {len(tr_rows):,} 条  →  {p_tr}")
        print(f"  评估 {len(ev_rows):,} 条  →  {p_ev}")
        # 抽取标签（输出很长时只看前 20 字）
        labs = Counter(r["output"][:20] for r in tr_rows)
        print(f"  输出样式 Top3: {[k for k, _ in labs.most_common(3)]}")
        print()


if __name__ == "__main__":
    main()
