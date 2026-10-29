# -*- coding: utf-8 -*-
"""
Qwen3-0.6B 显存可行性实测

目的：不靠估算，直接量出「这个模型在这张 4GB 卡上，用多大的 seq_len 能训」。

测四件事：
  1. 光加载模型占多少显存
  2. 挂上 LoRA（7 层全挂）又多占多少
  3. 不同 seq_len 下前向+反向的峰值
  4. 哪一档会 OOM

注意：这里量的是 torch 统计值。nvidia-smi 显示的数字会比它多 1.2-1.4 GB
（CUDA 上下文 + 显存碎片 + cuBLAS 工作区），判断能不能跑要看 nvidia-smi。
所以脚本最后会把两个数都打出来。

用法：
    D:\\ai\\env.bat
    D:\\ai\\venv\\Scripts\\python.exe D:\\ai\\test_qwen3.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "D:/ai/models/Qwen3-0.6B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def gb(x):
    return x / 1024 ** 3


def nvidia_used():
    """读 nvidia-smi 的真实占用（MB）——比 torch 的统计更接近物理现实"""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return float(out)
    except Exception:
        return -1


def main():
    print("=" * 64)
    print("  Qwen3-0.6B 显存实测")
    print("=" * 64)
    print(f"  测试前 nvidia-smi 占用: {nvidia_used():.0f} MiB\n")

    # ---------- 1. 加载模型 ----------
    print("[1] 加载模型（fp16）")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True
    )
    torch.cuda.synchronize()
    after_load = torch.cuda.memory_allocated()
    print(f"    模型权重占用: {gb(after_load):.2f} GB  (torch 统计)")
    print(f"    nvidia-smi : {nvidia_used():.0f} MiB\n")

    # ---------- 2. 挂 LoRA ----------
    print("[2] 挂 LoRA（r=8，7 个层全挂）")
    cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=TARGETS,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    torch.cuda.synchronize()
    print(f"    挂完后占用: {gb(torch.cuda.memory_allocated()):.2f} GB\n")

    # ---------- 3. 逐档测 seq_len ----------
    # Qwen3 有思考模式，微调时必须显式关掉，否则学到的是"带思考链的回答"
    print("[3] 逐档试 seq_len（前向 + 反向 + 优化器步进）")
    print("    注意：enable_thinking=False 是 Qwen3 特有，不写会带上  thinking 段\n")

    for seq_len in (2048, 1536, 1024, 768, 512):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 造一条该长度的假数据（真实训练里的 token 分布不影响显存，长度才影响）
            prompt_ids = tok.apply_chat_template(
                [{"role": "user", "content": "测试"}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            if hasattr(prompt_ids, "keys"):
                prompt_ids = prompt_ids["input_ids"]
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]

            pad = seq_len - len(prompt_ids)
            if pad <= 0:
                print(f"    seq_len {seq_len:>5}: 跳过（比 prompt 还短）")
                continue
            ids = torch.tensor([list(prompt_ids) + [tok.pad_token_id] * pad], device="cuda")
            labels = ids.clone()
            labels[:, :len(prompt_ids)] = -100        # 和真实训练一样的 mask
            attn = torch.ones_like(ids)

            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / 8).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            real = nvidia_used()
            # 预留 1.3GB 给 CUDA 上下文和碎片，反推"物理上能不能跑"
            verdict = "可以" if real + 300 < 4096 else "危险"
            print(f"    seq_len {seq_len:>5}: torch 峰值 {gb(peak):5.2f} GB | "
                  f"nvidia-smi {real:>5.0f} MiB | {verdict}")

        except torch.cuda.OutOfMemoryError:
            print(f"    seq_len {seq_len:>5}: ❌ OOM（显存不够）")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    seq_len {seq_len:>5}: 跳过（{type(e).__name__}: {str(e)[:50]}）")
            gc.collect()
            torch.cuda.empty_cache()

    # ---------- 4. 结论 ----------
    print("\n" + "=" * 64)
    print("  判读方法：")
    print("    · torch 峰值 + 1.3 GB ≈ nvidia-smi 实际占用")
    print("    · nvidia-smi 到 3900 MiB 以上就可能 OOM（要留渲染和碎片余量）")
    print("    · 上面标「可以」的最大 seq_len，就是你能用的上限")
    print("=" * 64)


if __name__ == "__main__":
    main()
