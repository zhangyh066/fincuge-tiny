# -*- coding: utf-8 -*-
"""验证「只算答案位置」的 loss 和标准写法等价。

这是改了训练核心逻辑，不验证就上长跑 = 拿 2 小时赌一个可能算错的 loss。
在 CPU 上用 fp32 跑，逐条对比两种算法给的数值（fp32 下应该几乎完全一致）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer

import train_sft as T

BASE = "D:/ai/models/Qwen3-0.6B"


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    ds = T.SFTDataset(tok, 12, 128, "D:/ai/data/fincuge/emotion_train.json",
                      val_ratio=0.0, val_words=[])
    print(f"\n用 {len(ds.items)} 条样本做对比，序列长度各不相同（含 padding）\n")

    # CPU + fp32，避免 fp16 的舍入掩盖真实差异
    model = T.load_model(BASE, "all", 8, device="cpu", dtype=torch.float32)[0]
    model.eval()

    for tag, bs in [("单条", 1), ("两条（含 padding）", 2), ("四条", 4)]:
        items = ds.items[:bs]
        ids, labels, mask = T.collate(items, tok.pad_token_id)
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=mask, labels=labels).loss.item()
            new = T.answer_only_loss(model, ids, labels, mask).item()
        diff = abs(ref - new)
        rel = diff / max(abs(ref), 1e-9)
        ok = "✅" if rel < 1e-4 else "❌"
        print(f"{tag:16s} 标准 {ref:.6f} | 只算答案 {new:.6f} | 差 {diff:.2e}（相对 {rel:.1e}） {ok}")
        assert rel < 1e-4, f"{tag} 两种算法不等价，差 {diff}"

    print("\n结论：两种写法在数值上等价，可以放心用省显存的版本。")


if __name__ == "__main__":
    main()
