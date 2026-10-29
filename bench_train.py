# -*- coding: utf-8 -*-
"""训练速度实测：不同 batch / 检查点 / 序列长度下，每秒能训多少条。

为什么需要这个：GTX 1650 只有 4GB，能不能跑是硬约束，但跑多快决定了实验能排几轮。
先量出来再定方案，别凭感觉估。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import torch

import train_sft as T

DATA = "D:/ai/data/fincuge/emotion_train.json"


def bench(bs, seq_len, use_ckpt, steps=6, lr=2e-4):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tag = f"batch={bs} seq={seq_len} 检查点={'开' if use_ckpt else '关'}"
    try:
        model, tok = T.load_model("D:/ai/models/Qwen3-0.6B", "all", 8)
        if not use_ckpt:
            model.gradient_checkpointing_disable()
            # 关掉检查点后，输入不再需要强制 requires_grad，可减少一份显存
            try:
                model.disable_input_require_grads()
            except Exception:
                pass

        ds = T.SFTDataset(tok, 64, seq_len, DATA, val_ratio=0.0, val_words=[])
        items = ds.items[:16]
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=lr)

        # 先热 1 步，把 cuBLAS 句柄、显存池这些都建好，再开始计时
        ids, labels, mask = T.collate(items[:bs], tok.pad_token_id)
        out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        t0 = time.time()
        losses = []
        for s in range(steps):
            idx = [(s * bs + j) % len(items) for j in range(bs)]
            ids, labels, mask = T.collate([items[i] for i in idx], tok.pad_token_id)
            out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
        dt = time.time() - t0

        rate = steps * bs / dt
        print(f"{tag:34s} → {rate:5.1f} 条/秒 | {dt/steps*1000:5.0f} ms/步 "
              f"| 峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f}G "
              f"| loss {losses[0]:.2f}→{losses[-1]:.2f}")
        del model, opt, trainable
        torch.cuda.empty_cache()
        return rate
    except RuntimeError as e:
        msg = "显存不足" if "out of memory" in str(e).lower() or "ALLOC_FAILED" in str(e) else str(e)[:60]
        print(f"{tag:34s} → 失败（{msg}）")
        torch.cuda.empty_cache()
        return 0.0


if __name__ == "__main__":
    print("=" * 96)
    print("  训练速度实测（同样 6 步，只改配置）")
    print("=" * 96)
    results = []
    for cfg in [(2, 128, True), (4, 128, True), (4, 128, False), (2, 128, False),
                (8, 96, False), (8, 96, True), (16, 96, True)]:
        results.append((cfg, bench(*cfg)))
    print()
    best = max(results, key=lambda x: x[1])
    print(f"最快配置：batch={best[0][0]} seq={best[0][1]} 检查点={'开' if best[0][2] else '关'}"
          f"，{best[1]:.1f} 条/秒")
    if best[1] > 0:
        print(f"16,157 条跑一轮约 {16157/best[1]/60:.0f} 分钟；两轮约 {16157*2/best[1]/60:.0f} 分钟")
