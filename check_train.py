# -*- coding: utf-8 -*-
"""训练流程正确性审计。

分两部分：
  python check_train.py              只查数据/掩码（纯 CPU，几秒）
  python check_train.py --gpu_smoke  再用显卡真训 20 步，看 loss 是否下降（约 1 分钟）

为什么要这个脚本：微调里最容易错、又最不容易发现的就是「label 掩码」——
掩错了照样能训练、loss 照样降，但模型学到的是「背问题」而不是「答问题」，
最后效果差还找不到原因。所以这里把每一步都断言死。
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import random

import torch

import train_sft as T
from prompt_util import build_user_text

DATA = "D:/ai/data/fincuge/emotion_train.json"
TOK_PATH = "D:/ai/models/Qwen3-0.6B"


def get_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def part_data(tok, path, seq_len, val_words):
    print("=" * 66)
    print("  一、数据与掩码检查")
    print("=" * 66)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"数据 {os.path.basename(path)}：{len(rows):,} 条")

    # ---- 1. 标签覆盖：每条 output 必须恰好命中一个标签词 ----
    bad = []
    for i, r in enumerate(rows):
        hit = [w for w in val_words if w in str(r.get("output", ""))]
        if len(hit) != 1:
            bad.append((i, r.get("output")))
    print(f"[1] 标签唯一性：{len(rows) - len(bad)}/{len(rows)} 条恰好命中 1 个标签"
          + (f"  ❌ 异常样例 {bad[:3]}" if bad else "  ✅"))
    assert not bad, "有样本命中 0 个或多个标签词"

    # ---- 2. 掩码检查：prompt 全 -100，答案部分完全等于这条自己的 answer ----
    ds = T.SFTDataset(tok, 300, seq_len, path, val_ratio=0.1, val_words=val_words)
    problems = []
    for idx in random.Random(0).sample(range(len(ds.items)), min(60, len(ds.items))):
        item = ds.items[idx]
        row = ds.item_rows[idx]                 # 用它自己的原始行比对，不能拿 rows[0] 硬套
        ids, labels = item["input_ids"], item["labels"]
        sup = [t for t, l in zip(ids, labels) if l != -100]
        n_prompt = len(ids) - len(sup)
        if labels[:n_prompt] != [-100] * n_prompt:
            problems.append((idx, "prompt 部分没被掩住"))
            continue
        got = tok.decode(sup, skip_special_tokens=True).strip()
        want = str(row["output"]).strip()
        if got != want:
            problems.append((idx, f"解码 {got!r} ≠ 原文 {want!r}"))
    print(f"[2] label 掩码：抽查 60 条，prompt 全为 -100、答案逐字还原原文"
          + ("  ✅" if not problems else f"  ❌ {problems[:3]}"))
    assert not problems, f"掩码有问题：{problems[:3]}"

    # ---- 3. 抽一条人眼核对 ----
    item = ds.items[0]
    sup = [t for t, l in zip(item["input_ids"], item["labels"]) if l != -100]
    print(f"[3] 人工核对第 1 条：")
    print(f"    被监督的 token 数 = {len(sup)}，解码 = {tok.decode(sup, skip_special_tokens=True)!r}")
    print(f"    原始 output      = {ds.item_rows[0]['output']!r}")

    # ---- 4. collate 补齐检查：padding 位置必须 label=-100、mask=0 ----
    batch = [ds.items[i] for i in range(3)]
    ids, labels, mask = T.collate(batch, tok.pad_token_id)
    maxlen = ids.shape[1]
    pad_ok = True
    for row in range(3):
        real = len(batch[row]["input_ids"])
        if mask[row, real:].sum().item() != 0:
            pad_ok = False
        if any(l != -100 for l in labels[row, real:].tolist()):
            pad_ok = False
        if real < maxlen and tok.pad_token_id != ids[row, real].item():
            pad_ok = False
    print(f"[4] padding：3 条变长样本补齐到 {maxlen}，padding 位置 mask=0 且 label=-100"
          + ("  ✅" if pad_ok else "  ❌"))
    assert pad_ok

    # ---- 5. 训练/验证互斥 ----
    key = lambda r: (str(r.get("instruction")), str(r.get("input")))
    train_keys = {key(r) for r in ds.train_rows}
    val_keys = {key(r) for r in ds.val_rows}
    overlap = train_keys & val_keys
    ratio = len(ds.val_rows) / max(1, len(ds.val_rows) + len(ds.train_rows))
    print(f"[5] 训练/验证切分：训练 {len(ds.train_rows)} 条 / 验证 {len(ds.val_rows)} 条"
          f"（验证占 {ratio:.1%}，目标 10%）"
          + ("  ✅" if not overlap else f"  ❌ 重复 {len(overlap)} 条"))
    assert not overlap, f"验证集有 {len(overlap)} 条出现在训练集中"

    # ---- 5b. 同一句话会不会被切成多条（股吧语料常有重复，重复会让验证虚高）----
    from collections import Counter
    cnt = Counter(key(r) for r in ds.train_rows)
    dup = sum(v - 1 for v in cnt.values() if v > 1)
    print(f"[5b] 训练集里重复的正文：{dup} 条（{dup/max(1,len(ds.train_rows)):.1%}）"
          + ("  ⚠ 有重复，验证分数可能虚高" if dup / max(1, len(ds.train_rows)) > 0.02 else "  ✅"))

    # ---- 6. 长度分布：有多少条会被 seq_len 截掉 ----
    over = sum(1 for it in ds.items if len(it["input_ids"]) > seq_len)
    lens = sorted(len(it["input_ids"]) for it in ds.items)
    print(f"[6] 序列长度：中位数 {lens[len(lens)//2]}，最长 {lens[-1]}，"
          f"seq_len={seq_len}，超出被丢弃 {over} 条"
          + ("  ✅" if over == 0 else "  ⚠ 有样本被丢"))
    print()


def part_gpu(args):
    print("=" * 66)
    print("  二、真训几步，看 loss 会不会降")
    print("=" * 66)
    tok = get_tok()
    model, tok = T.load_model(T.__dict__.get("MODEL", "D:/ai/models/Qwen3-0.6B"), "all", 8)

    ds = T.SFTDataset(tok, 64, 256, args.data_path, val_ratio=0.0, val_words=[])
    # 故意只用 16 条反复训：样本固定，loss 必须一路降。降不下去说明反向传播有问题。
    items = ds.items[:16]
    print(f"用 {len(items)} 条样本反复训练 {args.steps} 步（batch={args.bs}），loss 应该明显下降\n")

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    losses = []
    t0 = __import__("time").time()

    for step in range(1, args.steps + 1):
        idx = [(step * args.bs + j) % len(items) for j in range(args.bs)]
        ids, labels, mask = T.collate([items[i] for i in idx], tok.pad_token_id)
        ids, labels, mask = ids.cuda(), labels.cuda(), mask.cuda()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
        if step <= 3 or step % 5 == 0:
            print(f"  第 {step:>2} 步  loss {out.loss.item():.4f}")

    dt = __import__("time").time() - t0
    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    print()
    print(f"前 3 步平均 loss {first:.4f} → 后 3 步平均 {last:.4f}  "
          f"（{'下降 ✅' if last < first * 0.8 else '没明显下降 ❌'}）")
    print(f"吞吐：{args.steps * args.bs / dt:.1f} 条/秒"
          f"（{dt/args.steps*1000:.0f} 毫秒/步，batch={args.bs}）")
    print(f"峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print()
    print("  参考：batch=1 时实测 0.84 秒/条（情绪任务）。上面这个数如果明显更快，就说明提速有效。")
    assert last < first * 0.8, "loss 没降下来，训练流程有问题"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_smoke", action="store_true", help="额外跑真训练的冒烟测试")
    ap.add_argument("--data_path", default=DATA)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--val_words", default="积极,消极,中性")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    tok = get_tok()
    part_data(tok, args.data_path, args.seq_len,
              [w.strip() for w in args.val_words.split(",") if w.strip()])
    if args.gpu_smoke:
        part_gpu(args)
    print("全部检查通过。")


if __name__ == "__main__":
    main()
