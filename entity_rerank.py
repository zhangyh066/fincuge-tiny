# -*- coding: utf-8 -*-
"""实体候选重排 + 别名归一化：离线 A/B，不用 GPU

背景：实体任务 1,464 条里错了 203 条，其中 78% 是"原文有多个公司名，挑错了"，
22% 是"多一个字少一个字"。这两类都不需要重训，靠后处理就能救一部分。

做法：
  1. 从原文里抽出所有疑似机构名（按后缀 + 引号 + 股票代码三种线索）
  2. 给每个候选打分：离事件关键词越近越高、出现次数越多越高、位置越靠前越高
  3. 试几种策略，和模型的原始答案比
  4. 另外单独测"别名归一化"（去掉 有限公司/集团/股份 等尾巴再比）能救回多少严格匹配

数据全部来自已有缓存，跑一次几秒钟。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sft import extract_entity, entity_match  # 判分唯一真源

EVAL = "D:/ai/data/fincuge/entity_eval.json"
CACHE = "D:/ai/logs/eval_entity_cache_adapter-entity.jsonl"

# 机构名后缀（长的放前面，正则里先匹配更长的）
SUFFIX = ("股份有限公司", "有限责任公司", "有限公司", "公司", "集团", "股份",
          "银行", "证券", "基金", "保险", "信托", "资管", "控股", "实业",
          "科技", "平台", "理财", "投资", "资本", "担保", "小贷", "网贷")
ORG_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9（）()·]{2,18}(?:" + "|".join(SUFFIX) + r")")
CODE_RE = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})[（(]\d{6}[)）]")
QUOTE_RE = re.compile(r"[“\"「『]([\u4e00-\u9fa5A-Za-z0-9·]{2,20})[”\"」』]")

EVENT_WORDS = ("主体", "违规", "立案", "集资", "减持", "欺诈", "传销", "调查",
               "处罚", "事件", "风险", "爆雷", "违约", "变更", "离职")


def candidates(text):
    """从原文里抽疑似机构名，返回 [(名字, 首次出现位置)]"""
    seen = {}
    for m in ORG_RE.finditer(text):
        name = m.group(0)
        if len(name) >= 4 and name not in seen:
            seen[name] = m.start()
    for m in CODE_RE.finditer(text):
        name = m.group(1)
        if len(name) >= 2 and name not in seen:
            seen[name] = m.start()
    for m in QUOTE_RE.finditer(text):
        name = m.group(1)
        if len(name) >= 2 and name not in seen:
            seen[name] = m.start()
    return list(seen.items())


def score(name, pos, text):
    """离事件词越近、出现越多、越靠前 → 分越高"""
    d = min((abs(text.find(w) - pos) for w in EVENT_WORDS if text.find(w) >= 0), default=len(text))
    freq = text.count(name)
    return 1.0 / (1.0 + d / 20.0) + 0.3 * min(freq, 3) + 0.5 * (1 - pos / max(1, len(text)))


def norm_alias(s):
    """别名归一化：去掉常见的公司后缀再比"""
    t = str(s or "").strip().strip("“”\"'。.")
    for suf in ("股份有限公司", "有限责任公司", "有限公司", "公司", "集团",
                "股份", "控股", "实业"):
        if t.endswith(suf) and len(t) > len(suf) + 1:
            t = t[: -len(suf)]
            break
    return t


def main():
    rows = json.load(open(EVAL, encoding="utf-8"))
    cache = {}
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                cache[r["i"]] = r

    n = 0
    base = base_norm = 0
    v_fill = v_fill_norm = 0     # 模型答案不在候选里 → 用规则 Top1
    v_always = v_always_norm = 0  # 一律用规则 Top1
    v_guard = v_guard_norm = 0    # 模型答案在候选里就信它，否则 Top1（同 v_fill，但只在有候选时）
    hit_in_cand = 0
    rescued = broken = 0

    for i, row in enumerate(rows):
        rec = cache.get(i)
        if not rec:
            continue
        text = str(row.get("input", ""))
        gold = extract_entity(row.get("output", ""))
        pred = rec["pred"]
        cands = candidates(text)
        top1 = max(cands, key=lambda kv: score(kv[0], kv[1], text))[0] if cands else ""

        n += 1
        ok_base, _ = entity_match(pred, gold)
        base += int(ok_base)
        base_norm += int(norm_alias(pred) == norm_alias(gold) and bool(gold))

        preds = [pred, top1][0] if pred else top1
        v_fill += int(entity_match(pred if (pred in dict(cands)) else top1, gold)[0])
        v_fill_norm += int(norm_alias(pred if (pred in dict(cands)) else top1) == norm_alias(gold))
        v_always += int(entity_match(top1, gold)[0])
        v_always_norm += int(norm_alias(top1) == norm_alias(gold))
        v_guard += int(entity_match(pred if pred in dict(cands) else top1, gold)[0])
        v_guard_norm += int(norm_alias(pred if pred in dict(cands) else top1) == norm_alias(gold))

        if pred in dict(cands):
            hit_in_cand += 1
        # 规则版本救了谁、又坑了谁
        rule_ok = entity_match(top1, gold)[0]
        if not ok_base and rule_ok:
            rescued += 1
        elif ok_base and not rule_ok:
            broken += 1

    print(f"样本 {n} 条")
    print(f"  模型原始（严格全等）        {base/n:6.2%}")
    print(f"  模型原始 + 别名归一化        {base_norm/n:6.2%}   （归一化救回 {base_norm-base:+d} 条）")
    print()
    print(f"  模型答案在候选集合里的比例    {hit_in_cand/n:6.2%}")
    print(f"  策略1 不在候选里才用规则 Top1  {v_fill/n:6.2%}   （+别名归一化 {v_fill_norm/n:6.2%}）")
    print(f"  策略2 一律用规则 Top1        {v_always/n:6.2%}   （+别名归一化 {v_always_norm/n:6.2%}）")
    print()
    print(f"  规则 Top1 相对模型：救回 {rescued} 条，坑掉 {broken} 条")


if __name__ == "__main__":
    main()
