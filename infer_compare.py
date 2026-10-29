# -*- coding: utf-8 -*-
"""
微调前后对比 —— 验证 adapter 到底学没学到东西

这个脚本干的事很简单：同一个问题，分别问「原始模型」和「贴了 adapter 的模型」，
把两边回答并排打出来。

为什么要单独做这一步：
    训练的 loss 降了，只能说明模型在训练集上拟合得更好。
    它到底是"学会了回答风格"，还是"背下了训练集"，只有看输出才知道。

怎么看结果：
    两边完全一样      -> 微调没生效（adapter 路径错 / loss 没降 / LoRA 没挂上）
    微调后格式更统一  -> 学到了，正常
    微调后变傻、重复  -> 过拟合，减轮数或减学习率
    微调后答非所问    -> chat 模板对不上，训练和推理的格式必须完全一致

用法：
    D:\\ai\\env.bat
    D:\\ai\\venv\\Scripts\\python.exe D:\\ai\\infer_compare.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，中文会炸

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 基座模型和微调产物的位置。换成你自己的路径时，只要改这两行。
BASE = "D:/ai/models/Qwen3-0.6B"
ADAPTER = "D:/ai/out/lora-qwen3"

# 测试问题。选开放式、需要组织语言的题 —— 这类题最能看出"说话风格"的变化。
# 不要用"1+1等于几"这种有唯一答案的题，微调前后都会答对，看不出差别。
QUESTIONS = [
    "用三句话向小学生解释什么是光合作用。",
    "我想学做饭，但完全没基础，第一步该做什么？",
    "把这句话改得更委婉：你这个方案不行，重做。",
]


def load_base_model():
    """加载一份干净的基座模型。

    注意：这个函数会被调用两次，一次给"微调前"，一次给"微调后"。
    绝对不能让两个对比对象共用一个 model 对象 ——
    因为 PeftModel.from_pretrained 是【就地修改】传入的 model，
    共用的话，"微调前"那个也会被改掉，两边输出会一模一样。
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    # Qwen 系列默认没有 pad_token，不设的话：
    #   训练时 padding 会报错或拿 unk 去填；
    #   推理时 generate 会警告并可能生成错误结果。
    # 惯例是拿 eos 当 pad。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,       # 必须 fp16：GTX 1650 是 Turing，bf16 支持极差
        device_map={"": "cuda:0"},       # 4GB 显存，0.5B 模型 fp16 只占 1GB，整卡放得下
        trust_remote_code=True,
    )
    model.eval()                         # 关掉 dropout，否则同样输入每次输出都不同
    return model, tokenizer


@torch.inference_mode()                  # 推理不需要梯度，省显存也更快
def ask(model, tokenizer, question, max_new_tokens=180):
    """问一个问题，返回模型的回答文本。"""
    messages = [{"role": "user", "content": question}]

    # 必须用和训练时一模一样的 chat 模板。
    # 训练用的是 apply_chat_template，推理也必须用它，不能手拼字符串。
    _kw = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
    # Qwen3 有思考模式，推理时也要关掉，否则会先吐一段  thinking 再回答。
    # 和训练脚本、以及 LlamaFactory 的 qwen3_nothink 模板保持一致。
    if "enable_thinking" in (tokenizer.chat_template or ""):
        _kw["enable_thinking"] = False

    enc = tokenizer.apply_chat_template(messages, **_kw)
    # transformers 5.x 返回 BatchEncoding（有 .keys()），4.x 直接返回 tensor。
    # 不判断版本的话，5.x 上直接 .cuda() 会报 AttributeError。
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    ids = ids.cuda()

    output = model.generate(
        ids,
        max_new_tokens=max_new_tokens,   # 最多生成多少个新 token
        do_sample=True,                  # 采样而不是贪心，输出更自然
        temperature=0.7,                 # 温度越高越随机。对比实验时这个值要固定
        top_p=0.9,
        repetition_penalty=1.1,          # 惩罚重复，防止小模型复读机
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # output[0] 的前 ids.shape[1] 个 token 是我们喂进去的问题，要切掉，
    # 只解码模型新生成的部分。
    new_tokens = output[0][ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    # ⚠ 关键：两个模型不能同时加载。
    # Qwen3-0.6B 一个就占 1.1GB（fp16），两个 2.2GB，加上 CUDA 上下文 1.2GB
    # 就到 3.4GB，生成时的 cuBLAS 工作区一开就 CUBLAS_STATUS_ALLOC_FAILED。
    # 所以必须「跑完一段、卸掉、再跑下一段」，把答案先存下来最后一起打印。
    answers = {}

    print("第一轮：原始模型")
    m, tok = load_base_model()
    answers["before"] = [ask(m, tok, q) for q in QUESTIONS]
    del m
    gc.collect()
    torch.cuda.empty_cache()

    print("第二轮：加载 adapter")
    m, _ = load_base_model()
    m = PeftModel.from_pretrained(m, ADAPTER)
    m.eval()
    answers["after"] = [ask(m, tok, q) for q in QUESTIONS]
    del m
    gc.collect()
    torch.cuda.empty_cache()

    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 66)
        print(f"问题 {i}：{q}")
        print("-" * 66)
        print("【微调前】")
        print(answers["before"][i - 1])
        print("-" * 66)
        print("【微调后】")
        print(answers["after"][i - 1])

    print("=" * 66)
    print("判读方法：")
    print("  · 两边完全一样 -> adapter 没生效，检查路径和训练日志里的 loss")
    print("  · 微调后更啰嗦/格式更统一 -> 学到了训练集的风格，LoRA 起作用了")
    print("  · 微调后变傻/重复 -> 过拟合，减 epochs 或把 lr 调小")
    print("  · 微调后答非所问 -> 训练和推理的 chat 模板不一致")


if __name__ == "__main__":
    main()
