# -*- coding: utf-8 -*-
"""
FINNL 新闻分类数据清洗：15 类 → 7 类

为什么要简化，这是数据本身决定的：
    原始 15 类里，"中国"只有 2 条、"外汇"35 条、"虚拟货币"60 条。
    模型不可能从 2 条样本里学会一个类别。留着只会拖低准确率，
    而且评估时这些类的 F1 恒为 0，看着刺眼但没意义。

简化方案：
    Top 6 保留：公司(31.7%) 新冠(15.5%) 经济(10.8%) 行业(9.0%) 大盘(8.1%) 政策(7.5%)
    合计 82.6%，剩下全部归入"其它"

同时修掉标注不规范的问题：
    "外国" → 国际 → 其它（国际类样本极少）
    "组织" → 公司
    "更新"/"正常" → 其它
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import collections

SRC = {
    "train": "D:/ai/data/fincuge/newscat_train.json",
    "eval": "D:/ai/data/fincuge/newscat_eval.json",
}
OUT = {
    "train": "D:/ai/data/fincuge/newscat7_train.json",
    "eval": "D:/ai/data/fincuge/newscat7_eval.json",
}

KEEP = ["公司", "新冠", "经济", "行业", "大盘", "政策"]
OTHER = "其它"

# 原始输出里可能出现的各种说法 → 标准类别
ALIAS = {
    "公司": ["公司", "个股", "组织"],
    "行业": ["行业", "板块"],
    "大盘": ["大盘"],
    "经济": ["经济"],
    "政策": ["政策"],
    "新冠": ["新冠", "疫情"],
    # 以下都归"其它"
    "中国": ["中国"],
    "国际": ["国际", "外国"],
    "期货": ["期货"],
    "债券": ["债券"],
    "房地产": ["房地产", "楼市"],
    "外汇": ["外汇"],
    "虚拟货币": ["虚拟货币", "加密货币"],
    "能源": ["能源"],
    "政治": ["政治"],
    "其它": ["其它", "其他", "更新", "正常"],
}


def normalize(output_text: str) -> str:
    """把各种写法的输出归一到标准类别"""
    s = str(output_text)
    # 按别名长度倒序匹配，避免"中国"吃掉"国际"之类的误判
    hits = []
    for std, words in ALIAS.items():
        for w in words:
            if w in s:
                hits.append((len(w), std))
    if not hits:
        return OTHER
    hits.sort(reverse=True)
    std = hits[0][1]
    return std if std in KEEP else OTHER


def convert(rows):
    out = []
    for r in rows:
        cat = normalize(r["output"])
        # instruction 里原本写"15 个类别"，改成实际的 7 类，否则指令和数据对不上
        instr = ("针对以下新闻进行分类，类别包括：" + "、".join(KEEP + [OTHER]) +
                 "。请直接回答类别名称。")
        out.append({
            "instruction": instr,
            "input": r["input"],
            "output": f"以上新闻类别是{cat}。",
            "_cat": cat,
        })
    return out


def main():
    for split in ["train", "eval"]:
        rows = json.load(open(SRC[split], encoding="utf-8"))
        conv = convert(rows)
        with open(OUT[split], "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in r.items() if k != "_cat"} for r in conv],
                      f, ensure_ascii=False, indent=1)

        c = collections.Counter(r["_cat"] for r in conv)
        print(f"{split}: {len(conv)} 条 → {OUT[split]}")
        for k, v in c.most_common():
            print(f"    {k:<5} {v:>5}  {v/len(conv)*100:>5.1f}%")
        print()


if __name__ == "__main__":
    main()
