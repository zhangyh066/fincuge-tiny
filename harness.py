# -*- coding: utf-8 -*-
"""微小模型 agent · harness 第一版（技能层 + 路由 + 校验 + 三档判定 + 单行日志）

对应设计文档 D:/ai/AGENT_HARNESS.md 的 §2~§10：
    [1] 前置清洗 → [2] 路由（先只做 L1 规则）→ [3] 技能执行 → [4] 校验
    → [5] 三档判定 → [6] 输出 + 一行结构化日志

三条铁律（文档 §0/§12）：
    1. 技能之间不互相调用：一次请求只走一个技能，路由决定走哪个
    2. 提示词模板逐字节固定：不许动态拼（实测换措辞掉 10pp）
    3. 每次输出都能从日志复原走哪一步

用法：
    python harness.py "这家公司业绩暴雷，股价要崩"          # 自动路由
    python harness.py "某某公司被查" --task entity          # 强制任务
    python harness.py "..." --instruction "分析以下新闻，抽取信批违规事件相关的主体信息。"
    python harness.py --selftest --n 50                     # 和已有评估缓存逐条比对
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from prompt_util import build_user_text      # 提示词唯一真源
from train_sft import extract_entity, entity_match   # 判定唯一真源

BASE = os.environ.get("QWEN_MODEL") or ("D:/ai/models/Qwen3-0.6B" if os.path.isdir("D:/ai/models/Qwen3-0.6B") else "Qwen/Qwen3-0.6B")
TRACE = "D:/ai/logs/harness_trace.jsonl"

HERE = os.path.dirname(os.path.abspath(__file__))

# adapter 优先用仓库内的 weights/，clone 下来就能跑；找不到再退回本地训练产物
ADAPTER_FALLBACK = {
    "emotion": ["out/adapter-emotion-vb", "out/adapter-emotion-16k"],
    "newscat": ["out/adapter-newscat"],
    "entity": ["out/adapter-entity"],
}


def resolve_adapter(task, primary):
    for c in [primary] + ADAPTER_FALLBACK.get(task, []):
        p = c if os.path.isabs(c) else os.path.join(HERE, c)
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            return p
    return primary

EVAL_FILES = {
    "emotion": "D:/ai/data/fincuge/emotion_eval.json",
    "newscat": "D:/ai/data/fincuge/newscat7_eval.json",
    "entity": "D:/ai/data/fincuge/entity_eval.json",
}

# ── 情绪：受限 softmax 要读的 token id（与 eval_verbalizer.py 完全一致）──
TOK_POS, TOK_NEG, TOK_ZHONG, TOK_XING = 99666, 114217, 15946, 33071
EMOTION_LABELS = ["积极", "消极", "中性"]

# ── 新闻：七分类的别名表（与 eval_multi.py 完全一致）──
NEWS_LABELS = ["公司", "新冠", "经济", "行业", "大盘", "政策", "其它"]
NEWS_ALIAS = {
    "公司": ["公司", "个股", "组织"],
    "新冠": ["新冠", "疫情"],
    "经济": ["经济"],
    "行业": ["行业", "板块"],
    "大盘": ["大盘", "指数"],
    "政策": ["政策"],
    "其它": ["其它", "其他"],
}

# ── 每个技能的配置。instruction 从训练数据里取最常见的那一条，逐字节固定 ──
#   情绪训练集有 9 种指令（每种约 1800 条），取最常见的；
#   新闻只有 1 种指令，直接用；
#   实体有 121 种、且都带「事件类型」——真实使用时必须由用户提供，否则是分布外输入，
#   这里放一个通用兜底并在结果里打 flag=instruction_synthesized。
SKILLS = {
    "emotion": {
        "adapter": "weights/emotion",
        "mode": "restricted",
        "instruction": "判断以下句子的情感属性。",
        # 实测（2,020 条）：≥0.90 覆盖 37.0%、该档 94.3%；≥0.95 覆盖 21.7%、95.7%。
        # 没有哪一档能同时满足「95% 准确 + 30% 覆盖」，取 0.90 作为折中。
        "hi": 0.90, "lo": 0.50, "calibrated": True,
    },
    "newscat": {
        "adapter": "weights/newscat",
        "mode": "generate",
        "max_new": 16,
        "instruction": "针对以下新闻进行分类，类别包括：公司、新冠、经济、行业、大盘、政策、其它。请直接回答类别名称。",
        # 实测（884 条）：≥0.95 覆盖 79.3%、该档 95.4%；≥0.99 覆盖 47.4%、98.6%。
        # 置信度整体偏高，没有低置信带（<0.6 的样本为 0），所以不设拒答线。
        "hi": 0.95, "lo": 0.50, "calibrated": True,
    },
    "entity": {
        "adapter": "weights/entity",
        "mode": "generate",
        "max_new": 24,
        "instruction": "分析以下新闻，抽取相关事件的主体信息。",
        # 实测（1,464 条）：≥0.89 覆盖 38.7%、该档 95.1%；<0.79 只剩 1.4% 样本、准确率 5.0%。
        "hi": 0.89, "lo": 0.79, "calibrated": True,
    },
}

# ── L1 路由规则：只在高精度时生效，宁可漏判交给反问 ──
ROUTE_RULES = {
    "emotion": ["情绪", "情感", "态度", "正面", "负面", "积极", "消极", "中性",
                "看多", "看空", "乐观", "悲观", "利好", "利空", "偏正面", "偏负面"],
    "newscat": ["分类", "哪一类", "类别", "属于什么", "新闻类型", "归类", "哪类"],
    "entity": ["主体", "哪家公司", "哪家", "谁", "公司名", "机构", "主体信息"],
}


def clean_input(text):
    """[1] 前置清洗：去首尾空白、压掉多余空行、截断超长"""
    t = str(text or "").strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def route(text):
    """[2] 路由（L1 规则）。返回 (task, conf, reason)

    规则：取文本里**最后出现**的任务线索（中文语序里后置的往往是真实意图）。
    无命中返回 (None, 0.0, ...)，由上层去反问用户。
    """
    best = None
    for task, kws in ROUTE_RULES.items():
        for kw in kws:
            pos = text.rfind(kw)
            if pos >= 0 and (best is None or pos > best[0]):
                best = (pos, task, kw)
    if best is None:
        return None, 0.0, "无任务线索"
    pos, task, kw = best
    return task, 0.9, f'命中"{kw}"@{pos}'


def parse_news(text):
    """新闻答案解析：按别名匹配，长词优先（与 eval_multi.py 一致）"""
    t = str(text)
    hits = []
    for lab, words in NEWS_ALIAS.items():
        for w in words:
            i = t.find(w)
            if i >= 0:
                hits.append((-len(w), i, lab))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


class Harness:
    def __init__(self, device="cuda:0", verbose=True):
        self.dev = device
        self.tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.float16, device_map={"": device}, trust_remote_code=True
        )
        # 三个 adapter 全部挂上（LoRA 权重每个约 10MB，4GB 卡装得下），按请求切换
        names = list(SKILLS)
        self.paths = {n: resolve_adapter(n, SKILLS[n]["adapter"]) for n in names}
        self.model = PeftModel.from_pretrained(base, self.paths[names[0]],
                                              adapter_name=names[0])
        for n in names[1:]:
            self.model.load_adapter(self.paths[n], adapter_name=n)
        self.model.eval()
        self.has_think = "enable_thinking" in (self.tok.chat_template or "")
        self._cur = None
        if verbose:
            print(f"[harness] 已挂载 adapter: {names}  设备 {device}")

    # ---------- 内部工具 ----------
    def _use(self, task):
        if self._cur != task:
            self.model.set_adapter(task)
            self._cur = task

    def _encode(self, row):
        msgs = [{"role": "user", "content": build_user_text(row)}]
        kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        if self.has_think:
            kw["enable_thinking"] = False
        enc = self.tok.apply_chat_template(msgs, **kw)
        return enc["input_ids"].to(self.dev)

    def _emotion(self, row):
        """受限 softmax：一次前向读出 P(积极)/P(消极)/P(中)，末尾接"中"读 P(性|中)"""
        ids = self._encode(row)
        extra = torch.full((1, 1), TOK_ZHONG, device=self.dev, dtype=ids.dtype)
        ids1 = torch.cat([ids, extra], dim=1)
        out = self.model(input_ids=ids1)
        p1 = torch.softmax(out.logits[0, -2, :].float(), dim=-1)
        p2 = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_pos, p_neg = p1[TOK_POS].item(), p1[TOK_NEG].item()
        p_neu = p1[TOK_ZHONG].item() * p2[TOK_XING].item()
        total = p_pos + p_neg + p_neu
        if total <= 0:
            return "中性", 1 / 3, {}
        probs = {"积极": p_pos / total, "消极": p_neg / total, "中性": p_neu / total}
        pred = max(probs, key=probs.get)
        return pred, probs[pred], probs

    def _generate(self, row, max_new):
        """生成式：生成答案，同时用各步的 token 概率算置信度（几何平均）"""
        ids = self._encode(row)
        with torch.inference_mode():
            out = self.model.generate(
                ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
        seq = out.sequences[0][ids.shape[1]:]
        text = self.tok.decode(seq, skip_special_tokens=True).strip()
        lps = []
        for t, sc in enumerate(out.scores):
            if t >= len(seq):
                break
            lp = torch.log_softmax(sc[0].float(), dim=-1)
            lps.append(lp[seq[t]].item())
        conf = float(torch.tensor(lps).mean().exp()) if lps else 0.0
        return text, conf

    # ---------- 校验层 ----------
    # 拦截项（必须拒答）：答案根本不成立
    # 提醒项（不拒答，但把 auto 降级成 hint）：可能有瑕疵，值得人看一眼
    BLOCKING = {"label_illegal", "entity_empty", "entity_not_in_text"}
    # 只有"内容层面"的提醒才降级；threshold_uncalibrated / instruction_synthesized
    # 属于元信息，不该让每一次回答都退化成 hint（否则这两个任务永远进不了 auto）
    DOWNGRADE = {"format_multiline", "entity_length", "entity_punct", "entity_multiple"}

    def _validate(self, task, answer, row, raw):
        """[4] 硬规则校验。返回 (ok, flags)"""
        flags = []
        if "\n" in raw.strip() or raw.strip().startswith(("-", "*", "1.")):
            flags.append("format_multiline")
        if task == "emotion":
            if answer not in EMOTION_LABELS:
                flags.append("label_illegal")
        elif task == "newscat":
            if answer not in NEWS_LABELS:
                flags.append("label_illegal")
        elif task == "entity":
            if not answer:
                flags.append("entity_empty")
            else:
                if not (2 <= len(answer) <= 40):
                    flags.append("entity_length")
                # 注意：括号/空格在合法公司名里很常见（"××（中国）有限公司"），
                # 所以这里只作提醒，不能当拦截理由
                if re.search(r"[，。；、\n\"'“”*]", answer):
                    flags.append("entity_punct")
                if answer not in str(row.get("input", "")):
                    flags.append("entity_not_in_text")
                if raw.count("**") >= 4 or raw.count("\n- ") >= 2:
                    flags.append("entity_multiple")
        ok = not any(f in self.BLOCKING for f in flags)
        return ok, flags

    # ---------- 主入口 ----------
    def answer(self, text=None, task_hint=None, row=None, instruction=None):
        t0 = time.time()
        if row is None:
            body = clean_input(text)
            if not body:
                return self._reject("empty_input", t0, None, 0.0, "输入为空")
            task, rconf, reason = (task_hint, 1.0, "上游指定任务") if task_hint else route(body)
            if task is None:
                return self._reject("no_route", t0, None, 0.0,
                                    "看不出你想做哪件事（情绪/新闻分类/抽主体），请说明")
            cfg = SKILLS[task]
            ins = instruction or cfg["instruction"]
            row = {"instruction": ins, "input": body}
            synthesized = instruction is None and task == "entity"
        else:
            body = str(row.get("input", ""))
            task, rconf, reason = (task_hint, 1.0, "评估模式指定任务") if task_hint else route(body)
            task = task or "emotion"
            synthesized = False

        cfg = SKILLS[task]
        self._use(task)

        with torch.inference_mode():
            if cfg["mode"] == "restricted":
                pred, conf, detail = self._emotion(row)
                raw = pred
            else:
                raw, conf = self._generate(row, cfg["max_new"])
                if task == "newscat":
                    pred = parse_news(raw)
                    if pred is None:
                        pred = raw[:10]
                else:
                    pred = extract_entity(raw)
                detail = {}

        ok, flags = self._validate(task, pred, row, raw)
        if synthesized:
            flags.append("instruction_synthesized")
        if not cfg["calibrated"]:
            flags.append("threshold_uncalibrated")

        # [5] 三档判定：拦截项 → 拒答；有提醒项 → 最高只到"提示存疑"
        if not ok:
            decision = "abstain"
        elif conf >= cfg["hi"]:
            decision = "hint" if any(f in self.DOWNGRADE for f in flags) else "auto"
        elif conf >= cfg["lo"]:
            decision = "hint"
        else:
            decision = "abstain"

        res = {
            "task": task,
            "answer": pred if ok else "我不确定",
            "raw": raw[:120],
            "confidence": round(float(conf), 4),
            "decision": decision,
            "flags": flags,
            "route": {"conf": rconf, "reason": reason},
            "probs": {k: round(v, 4) for k, v in detail.items()} if detail else None,
            "attempts": 1,
            "latency_ms": int((time.time() - t0) * 1000),
            "trace_id": hashlib.md5((body or "")[:200].encode("utf-8")).hexdigest()[:8],
        }
        self._log(res, body)
        return res

    def _reject(self, why, t0, task, conf, msg):
        res = {"task": task, "answer": msg, "raw": "", "confidence": conf,
               "decision": "abstain", "flags": [why], "route": {"conf": 0.0, "reason": why},
               "probs": None, "attempts": 0,
               "latency_ms": int((time.time() - t0) * 1000), "trace_id": "-"}
        self._log(res, "")
        return res

    def _log(self, res, body):
        """[6] 单行结构化日志，任何一次输出都能复原走哪一步"""
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "input_len": len(body or ""),
               **res}
        try:
            with open(TRACE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


# ============================================================
# 自检：harness 的输出必须和已有评估脚本逐条一致
# ============================================================
def selftest(n=50):
    def load(p):
        rows = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return {r["i"]: r for r in rows}

    h = Harness()
    print(f"\n{'='*62}\n  自检：每个任务前 {n} 条，与已有评估缓存逐条比对\n{'='*62}")

    # 情绪：对 eval_verbalizer 的缓存
    ev = json.load(open("D:/ai/data/fincuge/emotion_eval.json", encoding="utf-8"))
    cache = load("D:/ai/logs/eval_vb_cache_adapter-emotion-vb.jsonl")
    same = tot = 0
    for i, r in enumerate(ev[:n]):
        res = h.answer(row=r, task_hint="emotion")
        ref = cache.get(str(i)) or cache.get(i)
        tot += 1
        if ref and res["answer"] == ref["pred"]:
            same += 1
    print(f"  情绪   : {same}/{tot} 与缓存一致")

    # 新闻：没有逐条缓存，只报准确率
    nw = json.load(open("D:/ai/data/fincuge/newscat7_eval.json", encoding="utf-8"))
    right = 0
    for r in nw[:n]:
        res = h.answer(row=r, task_hint="newscat")
        gold = parse_news(r.get("output", ""))
        right += int(res["answer"] == gold)
    print(f"  新闻   : {right}/{n} 正确（{right/n:.1%}）")

    # 实体：对 eval_entity 的缓存
    en = json.load(open("D:/ai/data/fincuge/entity_eval.json", encoding="utf-8"))
    cache2 = load("D:/ai/logs/eval_entity_cache_adapter-entity.jsonl")
    same = tot = 0
    for i, r in enumerate(en[:n]):
        res = h.answer(row=r, task_hint="entity")
        ref = cache2.get(str(i)) or cache2.get(i)
        tot += 1
        if ref and res["answer"] == ref["pred"]:
            same += 1
    print(f"  实体   : {same}/{tot} 与缓存一致")


EMOTION_ALIAS = {
    "积极": ["积极", "正面", "乐观", "正向"],
    "消极": ["消极", "负面", "悲观", "负向"],
    "中性": ["中性", "中立", "无明显", "无情绪"],
}


def gold_of(task, row):
    """从金标 output 里取出标准答案（与各评估脚本同一套规则）"""
    out = str(row.get("output", ""))
    if task == "emotion":
        for lab, ws in EMOTION_ALIAS.items():
            if any(w in out for w in ws):
                return lab
        return None
    if task == "newscat":
        return parse_news(out)
    return extract_entity(out)


def calibrate(task, limit=0):
    """[第 2 步] 阈值标定：把每条样本的「置信度 + 对错」落盘

    为什么需要：现在新闻/实体的 θ_hi=0.9、θ_lo=0.6 是拍脑袋的，没有数据支撑。
    做法：跑一遍整个评估集，记 (conf, 对错)，然后画覆盖率-准确率曲线定阈值。

    带断点续跑（agent 命令通道约 9 分钟被掐，实体 1,464 条要分几段）。
    """
    path = f"D:/ai/logs/harness_calib_{task}.jsonl"
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                done[rec["i"]] = rec

    rows = json.load(open(EVAL_FILES[task], encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    h = Harness(verbose=True)
    h._use(task)
    print(f"  [标定 {task}] 共 {len(rows)} 条，已有 {len(done)} 条，继续…")

    f = open(path, "a", encoding="utf-8")
    n = len(done)
    right = sum(1 for r in done.values() if r["ok"])
    t0 = time.time()
    for i, row in enumerate(rows):
        if i in done:
            continue
        with torch.inference_mode():
            if task == "emotion":
                pred, conf, _ = h._emotion(row)
            elif task == "newscat":
                raw, conf = h._generate(row, SKILLS["newscat"]["max_new"])
                pred = parse_news(raw) or raw[:10]
            else:
                raw, conf = h._generate(row, SKILLS["entity"]["max_new"])
                pred = extract_entity(raw)
        gold = gold_of(task, row)
        if task == "entity":
            ok, _ = entity_match(pred, gold)
        else:
            ok = (pred == gold)
        f.write(json.dumps({"i": i, "gold": gold, "pred": pred,
                            "conf": float(conf), "ok": bool(ok)}, ensure_ascii=False) + "\n")
        n += 1
        right += int(bool(ok))
        if n % 50 == 0:
            f.flush()
            print(f"    {n}/{len(rows)}  准确率 {right/n:.1%}  {time.time()-t0:.0f}s")
    f.close()
    print(f"  [标定 {task}] 完成 {n} 条，整体准确率 {right/max(1,n):.1%} → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--task", default=None, choices=list(SKILLS))
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", default=None, choices=list(SKILLS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.calibrate, args.limit)
        return
    if args.selftest:
        selftest(args.n)
        return
    if not args.text:
        print("用法: python harness.py \"你的文本\" [--task emotion|newscat|entity]")
        return

    h = Harness()
    res = h.answer(args.text, task_hint=args.task, instruction=args.instruction)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
