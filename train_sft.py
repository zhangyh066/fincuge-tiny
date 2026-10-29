# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B-Instruct 指令微调 · 手写训练循环

这个脚本故意不用 Trainer。整个训练过程就是这几步，明明白白：

    取一个 batch → 前向 → 算 loss → 反向 → 累积梯度 → 每 N 步更新一次参数

看懂这五步，你就知道训练是怎么回事了。剩下的都是围绕它的工程细节。

用法：
    D:\\ai\\env.bat
    D:\\ai\\venv\\Scripts\\python.exe D:\\ai\\train_sft.py

显存预期：约 2.0–2.5 GB / 4 GB（batch=1, seq=512, 开梯度检查点）
"""

import os
import sys
import time
import json
import argparse

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，中文会炸

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

try:
    from prompt_util import build_user_text   # 提示词唯一真源
except ImportError:                            # 从别的目录被调用时兜底
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prompt_util import build_user_text


# ============================================================
# 0. 参数
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="D:/ai/models/Qwen3-0.6B")
    p.add_argument("--data_path", default="D:/ai/data/alpaca_zh_51k.json")
    p.add_argument("--num_samples", type=int, default=3000)

    # ── 下面这几个默认值，对齐的是两家官方模板，不是我拍的 ──
    # 来源：
    #   LlamaFactory  examples/train_lora/qwen3_lora_sft.yaml
    #   unsloth       python_scripts/Qwen2.5_(7B)-Alpaca.py
    #
    # epochs   : LF 用 3.0
    # seq_len  : 两家都是 2048；这里给 1024（4GB 显存留余量，alpaca 数据也没那么长）
    # lr       : LF 1e-4，unsloth 2e-4，取 2e-4（小模型 + 小数据，偏大一点收敛快）
    # warmup   : LF 显式写 0.1（原来是 0.05，改过来）
    # lora_r   : LF 用 8，unsloth 用 16，取 8（4GB 显存优先）
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    # target_modules 的选择，这是本次对照最关键的一处差异：
    #   attn = 只挂注意力的 q/k/v/o       → 只能改"看哪里"
    #   all  = 再加 MLP 的 gate/up/down   → 才能改"怎么加工"，事实知识在这
    # 两家官方都是 all（LF 写 lora_target: all，unsloth 显式列了 7 层）。
    # 0.5B 挂全 7 层后参数量 1.08M → 4.40M（占 0.89%），显存只多 38MB。
    p.add_argument("--lora_target", default="all", choices=["attn", "all"])
    p.add_argument("--lora_r", type=int, default=8)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--out", default="D:/ai/out/lora-qwen05")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)  # 每 N 步存一次 checkpoint（防断电）
    # ── 验证集：训练时判断过拟合的唯一手段 ──
    # val_ratio 从训练数据里切多少做验证（默认 10%，验证我不能省）；
    # val_words 是这个任务的标签词（逗号分隔），按标签分层切保证每类都有。
    # 不传 val_words 就不切分（保持老行为）。
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--val_words", default="")
    p.add_argument("--val_limit", type=int, default=300)  # 每次验证最多跑多少条
    # 验证指标：label=分类任务按标签词判对错；entity=自由文本抽取，按抽出的主体判对错
    p.add_argument("--val_metric", default="label", choices=["label", "entity"])
    # ── 续训 ──
    # 只能恢复 LoRA 权重，恢复不了优化器状态（checkpoint 里没存 Adam 的动量），
    # 所以是「拿训到一半的权重接着训」，不是严格的断点续训。这点必须说清楚，
    # 免得以为和没断过完全一样。
    # 只算答案位置的 logits，显存占用大幅下降（长文本任务必须开，否则 lm_head 会爆）
    p.add_argument("--fast_loss", type=lambda x: str(x).lower() not in ("0", "false", "no"),
                   default=True, help="只对答案位置算 logits，省显存；默认开")
    p.add_argument("--resume", default="", help="从已有 adapter 目录恢复权重继续训")
    p.add_argument("--resume_step", type=int, default=0, help="从第几次参数更新接着算（决定学习率曲线位置）")
    return p.parse_args()


# target_modules 的两套候选。名字都是从模型的 named_modules 里实际打印出来的，
# Qwen2.5 全系（0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B）层名完全一致，
# 所以官方针对 4B/7B 写的配置，换到 0.5B 上照样能用。
LORA_TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all":  ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
}


# ============================================================
# 1. 模型与分词器
# ============================================================
def load_model(model_name, lora_target="all", lora_r=8, device="cuda:0", dtype=torch.float16):
    print(f"[1/5] 加载模型 {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ===== 坑 1：Qwen 默认没有 pad_token =====
    # 不设的话，padding 时要么报错，要么拿 unk 去填，把 loss 搞乱。
    # 惯例是拿 eos 当 pad。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # 训练用右边 padding，推理才用左边

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,              # ===== 坑 2：必须 fp16，不能 bf16 =====
                                        # GTX 1650 是 Turing 架构，bf16 支持很差，
                                        # 用了会慢到离谱甚至数值错误
        device_map={"": device},
        trust_remote_code=True,
    )

    # ===== 关键：只训练 LoRA 参数，基座全部冻结 =====
    # 0.5B 全参微调要 8GB+ 显存，LoRA 只要 2GB。差别就在这。
    #
    # target_modules 用哪套，直接决定"能学到什么"：
    #   attn 只动注意力的 q/k/v/o  -> 学说话风格够，学领域知识不够
    #   all  再加上 MLP 的三层     -> 事实性知识主要存在 MLP 里，学领域必须挂
    # 两家官方模板（LlamaFactory 的 lora_target: all、unsloth 显式列 7 层）都是后者。
    lora_cfg = LoraConfig(
        r=lora_r,                            # 秩。LG 官方用 8，unsloth 用 16
        lora_alpha=2 * lora_r,               # 缩放系数，惯例取 2×r
        target_modules=LORA_TARGETS[lora_target],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()   # 会打印"只训练了 0.x% 的参数"

    # ===== 关键：梯度检查点 =====
    # 原理：前向时不保存中间激活，反向时重新算一遍。
    # 代价：多花约 30% 时间。收益：激活显存从 1.5GB 降到 300MB。
    # 4G 显存能不能跑，全看这一行。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()   # 配合检查点，让梯度能回流到 embedding
    return model, tokenizer


# ============================================================
# 2. 数据
# ============================================================
class SFTDataset(Dataset):
    """把 alpaca 格式转成 chat 格式，并且只对「助手回答」算 loss。

    val_ratio > 0 时，从数据里切出一部分做验证集（分层，保证每个类别都有），
    训练集和验证集互不重叠。这是训练时判断过拟合的唯一手段。
    """

    def __init__(self, tokenizer, num_samples, seq_len, data_path,
                 val_ratio=0.0, val_words=None):
        print(f"[2/5] 加载数据 {data_path}，取 {num_samples} 条")
        with open(data_path, encoding="utf-8") as f:
            rows = json.load(f)[:num_samples]

        # ── 切分训练/验证（分层 + 固定种子，保证可复现）──
        self.val_rows = []
        self.train_rows = []
        if val_ratio > 0 and not val_words:
            # 没有类别标签的任务（如主体抽取）：无法分层，就按固定种子随机切
            import random
            rng = random.Random(42)
            rows = list(rows)
            rng.shuffle(rows)
            n_val = max(1, int(len(rows) * val_ratio))
            self.val_rows = rows[:n_val]
            self.train_rows = rows[n_val:]
            rows = self.train_rows
            print(f"      切分（随机）：训练 {len(rows)} 条 / 验证 {len(self.val_rows)} 条")
        if val_ratio > 0 and val_words:
            import random
            rng = random.Random(42)
            groups = {}
            for r in rows:
                out = str(r.get("output", ""))
                lab = next((w for w in val_words if w in out), "_other")
                groups.setdefault(lab, []).append(r)
            train_rows = []
            for lab, g in sorted(groups.items()):
                rng.shuffle(g)
                n_val = max(1, int(len(g) * val_ratio)) if len(g) > 4 else 0
                train_rows.extend(g[n_val:])
                self.val_rows.extend(g[:n_val])
            rows = train_rows
            self.train_rows = train_rows
            print(f"      切分：训练 {len(rows)} 条 / 验证 {len(self.val_rows)} 条"
                  f"（验证集按类别分层，各类分布 "
                  f"{ {k: sum(1 for r in self.val_rows if next((w for w in val_words if w in str(r.get('output',''))), None)==k) for k in [x for x in groups if x!='_other']} }）")

        # 记下每条 item 对应哪一行原始数据 —— 只为了能核对「这条样本被监督的到底是哪段文字」，
        # 不参与训练。没有这个映射，事后想验掩码对不对只能靠猜。
        self.item_rows = []
        self.items = []
        skipped = 0
        for row in rows:
            prompt_msgs = [{"role": "user", "content": self._user_text(row)}]

            # 用模型自带的模板，保证和它训练时见过的格式一致。
            #
            # ⚠ Qwen3 特有：它有"思考模式"，默认可能在回答前先生成一段  thinking...<｜end▁of▁thinking｜>。
            #   微调时必须显式关掉（enable_thinking=False），否则你训练的是"带思考链的回答"，
            #   推理时也得带着思考链，和预期不符。
            #   对应 LlamaFactory 里的 `template: qwen3_nothink`。
            #
            # 但 Qwen2.5 的模板不支持这个参数，传了会报错，所以要判断一下。
            _kw = {"tokenize": True, "add_generation_prompt": True}
            if "enable_thinking" in (tokenizer.chat_template or ""):
                _kw["enable_thinking"] = False

            enc = tokenizer.apply_chat_template(prompt_msgs, **_kw)
            # transformers 5.x 返回 BatchEncoding，4.x 返回 list，这里两种都兼容
            prompt_ids = enc["input_ids"] if hasattr(enc, "keys") else list(enc)
            if len(prompt_ids) and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            answer_ids = tokenizer(
                str(row.get("output", "")) + tokenizer.eos_token, add_special_tokens=False
            )["input_ids"]

            input_ids = prompt_ids + answer_ids
            if len(input_ids) > seq_len:
                skipped += 1
                continue
            # ===== 核心：label 掩码 =====
            # prompt 部分填 -100，PyTorch 的 cross_entropy 会忽略 -100。
            # 这样模型只学「怎么回答」，不学「怎么提问」。
            # 不加掩码的话，模型会连你的问题一起背下来，效果明显变差。
            labels = [-100] * len(prompt_ids) + answer_ids
            self.items.append({"input_ids": input_ids, "labels": labels})
            self.item_rows.append(row)

        print(f"      可用 {len(self.items)} 条，超长丢弃 {skipped} 条")

    @staticmethod
    def _user_text(row):
        # ⚠ 必须走公共函数，和离线评估保持一致。
        #   曾经这里和 eval_verbalizer.py 拼得不一样（多了"\n\n文本："），评估结果全被拖低。
        return build_user_text(row)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    """把一个 batch 的变长序列补齐到同长。"""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        n = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * n)
        labels.append(x["labels"] + [-100] * n)     # padding 不参与 loss
        attn.append([1] * len(x["input_ids"]) + [0] * n)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


# ============================================================
# 3. 学习率调度
# ============================================================
def lr_at(step, total, base_lr, warmup):
    """预热 + 余弦退火。手写的，看得见每一步的 lr 是多少。"""
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    import math
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================
# 3.5 验证集评估（判断过拟合的唯一手段）
# ============================================================
@torch.inference_mode()
def evaluate_val(model, tokenizer, val_rows, val_words, seq_len, limit=300, verbose=False):
    """在验证集上算准确率。

    为什么必须有：训练 loss 会一路降，但那不代表模型变好，可能只是把训练数据背下来了。
    唯一能看出来的办法，就是每轮结束在不参与训练的样本上测一次。
    判据：训练 loss 降、验证准确率掉 = 过拟合，立刻停。

    做法：用生成方式（和真实部署一致），然后在生成文本里找标签词。
    标签词没命中算错（保守估计，宁可低估准确率）。
    """
    if not val_rows or not val_words:
        return None, 0

    # ⚠ 不能直接取前 N 条：验证集是按类别顺序拼起来的（中性→消极→积极），
    #   取前 300 条会全落在第一个类上。今天真踩过这个坑：报出 53.7%，
    #   其实是「300 条全中性」的准确率，不是整体准确率。
    rows_used = list(val_rows)
    if len(rows_used) > limit:
        import random
        rng = random.Random(0)
        by = {}
        for r in rows_used:
            lab = next((w for w in val_words if w in str(r.get("output", ""))), None)
            by.setdefault(lab, []).append(r)
        per = max(1, limit // max(1, len(by)))
        picked = []
        for lab in sorted(by, key=lambda x: str(x)):
            arr = list(by[lab])
            rng.shuffle(arr)
            picked.extend(arr[:per])
        rows_used = picked[:limit]

    was_training = model.training
    model.eval()
    correct = total = 0
    has_think = "enable_thinking" in (tokenizer.chat_template or "")
    device = next(model.parameters()).device

    for r in rows_used:
        msgs = [{"role": "user", "content": build_user_text(r)}]
        kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        if has_think:
            kw["enable_thinking"] = False
        enc = tokenizer.apply_chat_template(msgs, **kw)
        ids = enc["input_ids"] if hasattr(enc, "keys") else torch.tensor([enc])
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)

        out = model.generate(
            ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        gold_txt = str(r.get("output", ""))
        gold = next((w for w in val_words if w in gold_txt), None)
        pred = next((w for w in val_words if w in gen), None)
        if gold is not None:
            total += 1
            correct += int(pred == gold)
        if verbose and total <= 3:
            print(f"        样例 生成={gen.strip()[:30]!r}  gold={gold}  pred={pred}")

    if was_training:
        model.train()
    return (correct / total if total else None), total


# ============================================================
# 3.5c 主体抽取任务的验证：按"抽出来的主体"判对错
# ============================================================
def extract_entity(text):
    """从模型的回答里抽出主体名。

    训练数据的答案长这样（模板千变万化）：
        「所属资金账户风险事件的金融主体是雅虎。」
        「高管负面事件的主体是：钱妈妈。」
    统一规则：取最后一个"是/："之后的内容，去掉句号和引号。
    模型如果不按模板回答、直接吐了个公司名，也能正常处理。
    """
    t = str(text).strip().rstrip("。.").strip()
    for sep in ("是：", "是:", "是", "：", ":"):
        if sep in t:
            t = t.rsplit(sep, 1)[1]
            break
    return t.strip().strip('"“”\'').strip()


def entity_match(pred, gold):
    """两段主体文本算不算匹配。

    返回 (严格, 宽松)：严格=去掉空格标点后完全相同；
    宽松=一方包含另一方（"雅虎" vs "雅虎公司"这种应该算对）。
    """
    # 用集合过滤而不是正则，避免引号转义把代码写坏
    drop = set(" \t\n，,。.、\"'“”‘’")
    norm = lambda x: "".join(ch for ch in str(x) if ch not in drop)
    a, b = norm(pred), norm(gold)
    if not a or not b:
        return False, False
    return a == b, (a in b or b in a)


@torch.inference_mode()
def evaluate_val_entity(model, tokenizer, val_rows, limit=200):
    """主体抽取的验证：生成回答 → 抽主体 → 和标准答案比。

    为什么不能沿用分类那套：这里的答案没有"标签词"，1464 条评估里
    就有 9,894 种不同写法，只能比对抽出来的实体本身。
    """
    if not val_rows:
        return None, 0

    import random
    rows_used = list(val_rows)
    if len(rows_used) > limit:
        random.Random(0).shuffle(rows_used)
        rows_used = rows_used[:limit]

    was_training = model.training
    model.eval()
    has_think = "enable_thinking" in (tokenizer.chat_template or "")
    device = next(model.parameters()).device
    strict = loose = total = 0

    for r in rows_used:
        msgs = [{"role": "user", "content": build_user_text(r)}]
        kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        if has_think:
            kw["enable_thinking"] = False
        enc = tokenizer.apply_chat_template(msgs, **kw)
        ids = enc["input_ids"] if hasattr(enc, "keys") else torch.tensor([enc])
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        out = model.generate(ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        p, g = extract_entity(gen), extract_entity(r.get("output", ""))
        ok_s, ok_l = entity_match(p, g)
        total += 1
        strict += int(ok_s)
        loose += int(ok_l)

    if was_training:
        model.train()
    if total == 0:
        return None, 0
    # 严格匹配率作为主指标（和准确率同量纲），宽松值另外打印
    return strict / total, total


# ============================================================
# 3.5b 只算答案位置的 loss（省显存的关键）
# ============================================================
def answer_only_loss(model, ids, labels, mask):
    """只在「答案那几个 token」上算 logits 和 loss。

    为什么必须这么写：
        lm_head 的输出形状是 (batch, 序列长度, 15万词表)。
        序列 512、batch 2 时这一层就要 2×512×151936×4 字节 ≈ 600MB，4GB 显卡直接爆。
        但真正被监督的只有答案那 1~3 个 token（word 标签）或十几个 token（整句），
        前面几百个位置的 logits 全是白算。

    做法：先只跑 transformer 拿隐状态（不带 lm_head），再把答案位置那几行
        切片送进 lm_head。数学上和 huggingface 的 labels 版完全等价，
        区别只是不把没用的位置投影到词表。

    对应关系：预测第 t 个 token，用的是第 t-1 个位置的隐状态。
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = base.model(input_ids=ids, attention_mask=mask, use_cache=False)
    h = out.last_hidden_state               # (B, L, H)
    lm = base.lm_head

    total = None
    n = 0
    for i in range(ids.shape[0]):
        lab = labels[i]
        idx = (lab != -100).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        s = int(idx[0])
        e = int(idx[-1]) + 1                # 答案区间 [s, e)
        if s == 0:                          # 理论上不会发生（前面必有 prompt）
            continue
        logits = lm(h[i, s - 1:e - 1, :]).float()
        loss_i = torch.nn.functional.cross_entropy(logits, lab[s:e], reduction="sum")
        total = loss_i if total is None else total + loss_i
        n += e - s
    if total is None:
        return torch.zeros((), device=ids.device, requires_grad=True)
    return total / max(n, 1)


# ============================================================
# 3.6 续训：把已有 adapter 的权重装回来
# ============================================================
def load_resume_weights(adapter_dir, model):
    """把某个已训好的 LoRA adapter 权重装进当前模型。

    ⚠ 只能恢复权重，恢复不了 Adam 的动量状态（checkpoint 里没存），
      所以严格说这是「热启动」而不是「断点续训」。实践上影响很小，
      因为 LoRA 的优化器状态本来就只有 5M 参数那么大，重建很快能追上。
    """
    import glob
    from peft import set_peft_model_state_dict

    cand = (glob.glob(os.path.join(adapter_dir, "adapter_model.safetensors")) +
            glob.glob(os.path.join(adapter_dir, "adapter_model.bin")))
    if not cand:
        raise FileNotFoundError(f"{adapter_dir} 里没有 adapter_model.safetensors/.bin")

    path = cand[0]
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu")

    res = set_peft_model_state_dict(model, sd)
    missing = len(getattr(res, "unexpected_keys", []) or [])
    print(f"      [续训] 已从 {os.path.basename(adapter_dir)} 恢复 "
          f"{len(sd)} 个权重张量" + (f"，另有 {missing} 个未匹配" if missing else ""))


# ============================================================
# 4. 训练
# ============================================================
def main():
    args = parse_args()
    print("=" * 62)
    print(f"  device = {torch.cuda.get_device_name(0)}")
    print(f"  显存   = {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 62)

    model, tokenizer = load_model(args.model, args.lora_target, args.lora_r)
    if args.resume:
        load_resume_weights(args.resume, model)
    _vw = [w.strip() for w in args.val_words.split(",") if w.strip()]
    ds = SFTDataset(tokenizer, args.num_samples, args.seq_len, args.data_path,
                    val_ratio=args.val_ratio, val_words=_vw)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    # 只把 LoRA 参数交给优化器 —— 这就是 LoRA 省显存的根
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    val_history = []
    best_val = -1.0
    best_epoch = 0
    steps_per_epoch = len(loader)
    new_updates = steps_per_epoch * args.epochs // args.grad_accum
    total_steps = args.resume_step + new_updates
    warmup = min(int(total_steps * args.warmup_ratio), args.resume_step) if args.resume_step else int(total_steps * args.warmup_ratio)
    print(f"[3/5] 每个 epoch {steps_per_epoch} 个 batch，"
          f"共 {total_steps} 次参数更新，预热 {warmup} 步"
          + (f"（从第 {args.resume_step} 步接续）" if args.resume_step else ""))

    model.train()
    global_step, micro_step = args.resume_step, 0
    got_grad_checked = False
    t0 = time.time()
    loss_window = []
    history = []

    print("[4/5] 开始训练\n")
    for epoch in range(args.epochs):
        for i, (ids, labels, mask) in enumerate(loader):
            ids = ids.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            # ---- 前向 ----
            if args.fast_loss:
                # 只对答案位置算 logits：省显存，也让前向更快（见 3.5b 的说明）
                step_loss = answer_only_loss(model, ids, labels, mask)
            else:
                step_loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            # loss 已经按 token 平均过了。梯度累积时除以 accum，
            # 保证累积 N 步的梯度 ≈ 一个大 batch 的梯度，而不是 N 倍。
            loss = step_loss / args.grad_accum
            loss.backward()

            loss_window.append(step_loss.item())
            micro_step += 1

            # ---- 累积够 N 步再更新 ----
            if micro_step % args.grad_accum == 0:
                cur_lr = lr_at(global_step, total_steps, args.lr, warmup)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr

                # 第一次更新前自检一次：确认 LoRA 参数真的拿到了梯度。
                # 为什么必须查：如果梯度没接上（比如新写的 loss 忘了让隐状态参与反传），
                # 训练会「安静地空转」—— loss 照样打印，参数一动不动，跑 3 小时白费。
                if not got_grad_checked:
                    ngrad = sum(1 for p_ in trainable if p_.grad is not None)
                    gsum = sum(float(p_.grad.norm()) for p_ in trainable if p_.grad is not None)
                    print(f"      [自检] 拿到梯度的参数 {ngrad}/{len(trainable)}，"
                          f"梯度范数和 {gsum:.3f}")
                    if ngrad == 0 or gsum == 0:
                        raise RuntimeError("没有任何参数拿到梯度，训练等于空转，已中止")
                    got_grad_checked = True

                # 梯度裁剪：防止个别样本把参数带飞。这不是可选项，是保命的
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)   # set_to_none 比填 0 省显存
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = sum(loss_window) / len(loss_window)
                    loss_window.clear()
                    used = torch.cuda.max_memory_allocated() / 1024**3
                    elapsed = time.time() - t0
                    # ⚠ ETA 必须按「本次实际跑过的步数」算，不能除以 global_step。
                    #   续训时 global_step 从 200 起，elapsed 只统计了新跑的这段，
                    #   除下去会把每步耗时算小一半，报出来的 ETA 偏乐观（今天就把 54 分钟说成 15 分钟）。
                    done_now = max(1, global_step - args.resume_step)
                    eta = elapsed / done_now * (total_steps - global_step)
                    print(f"  epoch {epoch+1}/{args.epochs} | step {global_step:>4}/{total_steps} "
                          f"| loss {avg:.4f} | lr {cur_lr:.2e} | 峰值显存 {used:.2f}G "
                          f"| 已用 {elapsed/60:.1f}min | 剩 {eta/60:.1f}min")
                    history.append({"step": global_step, "loss": avg, "lr": cur_lr})

                # 定期落盘，防断电白跑。
                # 只存 LoRA 权重（20MB），不存优化器状态 —— 断电后不能精确续训，
                # 但至少手上有一个"训到一半"的可用 adapter，能先看效果。
                if args.save_every and global_step % args.save_every == 0:
                    try:
                        model.save_pretrained(args.out + "_ckpt")
                        with open(args.out + "_ckpt/history.json", "w", encoding="utf-8") as f:
                            json.dump(history, f, ensure_ascii=False)
                        print(f"      [checkpoint] step {global_step} 已存到 {args.out}_ckpt")
                    except Exception as e:
                        print(f"      [checkpoint] 保存失败: {e}")

        # 一轮结束时如果攒的梯度没凑够 grad_accum，也要更新一次。
        # 不这么做的话：这批样本白算，梯度还会串到下一轮头几个 batch，
        # 等于把两轮的数据混在一次更新里 —— 这是原脚本的真实缺陷，已修。
        if micro_step % args.grad_accum != 0:
            cur_lr = lr_at(global_step, total_steps, args.lr, warmup)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            micro_step = 0

        print(f"  —— epoch {epoch+1} 结束 ——（本轮耗时 {(time.time()-t0)/60:.1f} 分钟）")
        # 每轮结束跑验证：训练 loss 降但验证准确率掉 = 过拟合，该停
        if args.val_ratio > 0:
            # 分类任务按标签词判；主体抽取按"抽出来的实体"判，两套指标不能混用
            if args.val_metric == "entity":
                v_acc, v_n = evaluate_val_entity(model, tokenizer, ds.val_rows, args.val_limit)
                v_name = "实体完全匹配率"
            else:
                v_acc, v_n = evaluate_val(model, tokenizer, ds.val_rows, _vw, args.seq_len, args.val_limit)
                v_name = "准确率"
            if v_acc is not None:
                print(f"      [验证] epoch {epoch+1} {v_name} {v_acc:.1%}（{v_n} 条）")
                val_history.append({"epoch": epoch+1, "val_acc": v_acc, "n": v_n})
                if v_acc > best_val:
                    best_val = v_acc
                    best_epoch = epoch + 1
                    try:
                        model.save_pretrained(args.out + "_best")
                        print(f"      [验证] 新最佳，已存 {args.out}_best")
                    except Exception:
                        pass
        print()

    # ============================================================
    # 5. 保存
    # ============================================================
    os.makedirs(args.out, exist_ok=True)
    # 只存 LoRA adapter，不存整个 0.5B 基座。约 5MB，而不是 1GB。
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 验证集记录一并存下来：事后能看出「第几轮最好」，而不是只看最后一次
    if val_history:
        with open(os.path.join(args.out, "val_history.json"), "w", encoding="utf-8") as f:
            json.dump(val_history, f, ensure_ascii=False, indent=2)
        curve = "  ".join(f"第{h['epoch']}轮 {h['val_acc']:.1%}" for h in val_history)
        print(f"      验证准确率逐轮：{curve}")
        print(f"      最佳轮次：第 {best_epoch} 轮（{best_val:.1%}），adapter 在 {args.out}_best")
        if best_epoch < len(val_history):
            print(f"      ⚠ 最后一轮不是最好的 → 后面几轮过拟合了，部署请用 {args.out}_best")

    print(f"[5/5] 完成。adapter 存到 {args.out}")
    print(f"      总耗时 {(time.time()-t0)/60:.1f} 分钟，峰值显存 "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"      下一步：跑 D:\\ai\\infer_compare.py 看微调前后的差别")


if __name__ == "__main__":
    main()
