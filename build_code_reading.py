# -*- coding: utf-8 -*-
"""把源码渲染成「逐行精读」HTML：行号 | 代码 | 这一行在干什么 | 代码后面配讲解

用法：
    python build_code_reading.py
输出：
    docs/源码精读.html
"""
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "docs", "源码精读.html")

# ────────────────────────────────────────────────────────────
# 逐行注释：按「行区间 → 这句话管什么」写，渲染时该区间每一行都会带上说明
# 讲解：在某一行之后插入一段 prose，讲清这段代码为什么这么写
# ────────────────────────────────────────────────────────────
FILES = [
    {
        "path": "prompt_util.py",
        "title": "prompt_util.py · 提示词拼装的唯一真源",
        "role": "24 行，全项目最该先读的文件。它只有一个函数，但决定了训练和评估喂给模型的东西完全一致。",
        "blocks": [
            (1, 1, "编码声明：告诉 Python 这个文件是 UTF-8"),
            (2, 12, "模块文档：记录了一次真实事故——训练用 `指令\\n正文`、评估用 `指令\\n\\n文本：正文`，模型从没见过后者，评估分数被系统性拖低"),
            (13, 14, "空行：分隔文档与代码"),
            (15, 15, "定义唯一的拼装函数，训练脚本和所有评估脚本都 import 它"),
            (16, 20, "函数文档：说明入参是 alpaca 格式的一条数据，返回一段文本"),
            (21, 21, "取出 instruction 并去掉首尾空白"),
            (22, 22, "取出 input 并去掉首尾空白"),
            (23, 23, "注释：有正文才换行接上，没正文就只用指令"),
            (24, 24, "真正拼装：正文存在时用「指令 + 换行 + 正文」，否则只返回指令。全项目只有这一处拼字符串"),
        ],
        "walkthrough": [
            (24, "为什么只有 24 行却最重要",
             "模型看到的输入只有两种来源：训练时和推理时。只要这两处有一处拼法不同，模型就是在「考试时见到没复习过的格式」，分数会系统性偏低。"
             "我们的做法是让两边都调同一个函数，谁都不许自己写 f-string。代价是改提示词要重训模型，收益是任何一次评估结果都可复现。"),
        ],
    },
    {
        "path": "train_sft.py",
        "title": "train_sft.py · 手写训练循环（707 行）",
        "role": "整个项目的心脏。故意不用 Trainer，把「取 batch → 前向 → 算 loss → 反向 → 累积 → 更新」六步摊开写，每一步都能单独验证。",
        "blocks": [
            (1, 1, "编码声明：UTF-8"),
            (2, 16, "模块文档：说明为什么不用 Trainer、训练的五步流程、怎么运行、显存预期"),
            (18, 22, "导入标准库：os/sys/time/json/argparse"),
            (24, 24, "把标准输出改成 UTF-8。Windows 控制台默认 GBK，打印中文会报 UnicodeEncodeError"),
            (26, 29, "导入 torch 与数据集工具、transformers 的模型/分词器、peft 的 LoRA 接口"),
            (31, 35, "导入提示词函数；如果从别的目录调用时找不到，就把当前文件所在目录加进 sys.path 再导一次"),
            (38, 40, "分节注释：0. 参数"),
            (41, 42, "定义参数解析函数，创建一个解析器"),
            (43, 43, "--model 三级回退：环境变量 QWEN_MODEL → 本地 models 目录 → HuggingFace 仓库名"),
            (44, 45, "数据路径与最多用多少条样本的默认值"),
            (47, 56, "注释块：这几个默认值不是拍的，逐项标了来源（LLaMA-Factory 与 unsloth 的官方模板），并说明为什么这么取"),
            (57, 60, "四个核心超参：epochs=3、seq_len=512、lr=2e-4、warmup=0.1"),
            (62, 66, "注释块：target_modules 两套方案的差别——只挂注意力只能改「看哪里」，挂上 MLP 才能改「怎么加工」"),
            (67, 68, "lora_target（attn/all）与秩 r 的默认值 8"),
            (70, 75, "batch、梯度累积、梯度裁剪阈值、输出目录、日志间隔、存档间隔"),
            (76, 84, "验证集相关参数：切分比例、标签词、每次验证条数、验证指标类型。注意这里 val_words 为空就不切分"),
            (85, 93, "续训与 fast_loss 参数。注释明确写了：只能恢复权重，恢复不了 Adam 动量，所以是「热启动」不是严格续训"),
            (94, 94, "解析并返回参数对象"),
            (97, 99, "注释：模块名是从模型的 named_modules 里打印出来的，Qwen2.5 全系层名一致，所以官方给 7B 的配置 0.5B 也能用"),
            (100, 104, "两套候选：attn 只 4 个注意力投影；all 再加 gate/up/down 三个 MLP 投影"),
            (107, 109, "分节注释：1. 模型与分词器"),
            (110, 112, "load_model 签名（target、r、设备、精度）+ 打印 + 加载分词器"),
            (114, 119, "坑 1：Qwen 没有 pad_token，直接 padding 会报错或用 unk 填空。惯例拿 eos 当 pad，并把 padding 放右边"),
            (121, 128, "加载模型本体。精度必须 fp16——坑 2：GTX 1650 是 Turing，bf16 支持很差，会慢到离谱甚至数值错"),
            (130, 136, "注释：只训 LoRA、基座全冻；target 选哪套决定能学到什么（事实知识主要存在 MLP 里）"),
            (137, 144, "LoRA 配置：秩 r、alpha=2r、目标模块、dropout 0.05、不训 bias、任务是因果语言模型"),
            (145, 146, "把 LoRA 挂到模型上，并打印「可训练参数 / 全部参数 / 占比」"),
            (148, 153, "梯度检查点：前向不存中间激活、反向重算。多花约 30% 时间，换激活显存从 1.5GB 降到 300MB"),
            (154, 154, "返回模型与分词器"),
            (157, 159, "分节注释：2. 数据"),
            (160, 165, "数据集类文档：把 alpaca 转 chat 格式，只对助手回答算 loss；val_ratio>0 时切出验证集"),
            (167, 171, "构造函数：读 json、只取前 num_samples 条"),
            (173, 186, "没有标签词的任务（如主体抽取）无法分层：固定种子随机切，保证可复现"),
            (187, 205, "有标签词的任务：按标签分组后每组切 10% 做验证，保证验证集里每类都有样本"),
            (207, 211, "记下每条样本对应原始数据的哪一行，只为事后核对掩码，不参与训练"),
            (212, 213, "遍历每一行，先构造 user 消息"),
            (215, 225, "关键注释：Qwen3 有思考模式，微调必须显式关掉（否则学到的是带 thinking 的回答）；Qwen2.5 模板没有这个参数，传了会报错，所以要判断"),
            (227, 231, "套用模型的 chat 模板；transformers 5.x 返回对象、4.x 返回 list，这里两种都兼容"),
            (232, 234, "把答案 token 化，并在结尾补 eos，让模型学会「说完要停」"),
            (236, 239, "拼成完整序列；超过 seq_len 的直接丢弃并计数（不是截断，截断会截掉答案）"),
            (240, 246, "核心：label 掩码。prompt 部分全部填 -100，交叉熵会忽略它，模型只学「怎么答」不学「怎么问」"),
            (248, 248, "打印可用条数与丢弃条数，这个数异常说明序列长度设小了"),
            (250, 254, "_user_text 强制走公共函数。注释记着历史 bug：这里曾和评估脚本拼得不一样"),
            (256, 260, "数据集协议：长度与按下标取样本"),
            (263, 276, "collate：把一个 batch 补齐到等长；label 补 -100 不参与 loss；同时产出 attention mask"),
            (279, 288, "学习率调度：先线性预热到 base_lr，再余弦退火到接近 0。手写的，每一步 lr 都可见"),
            (291, 304, "验证函数文档：训练 loss 会一直降，只有验证集能看出过拟合"),
            (305, 306, "没有验证数据就直接返回，不报错"),
            (308, 325, "按类别分层抽 limit 条。注释记着真踩过的坑：直接取前 300 条会全落在同一个类，报出 53.7% 的假分数"),
            (327, 332, "切到 eval 模式（关 dropout），准备统计变量"),
            (333, 351, "逐条生成：greedy 解码、最多 16 个新 token，和线上推理方式保持一致"),
            (352, 360, "在生成文本里找标签词，命中才算对；没命中按错算（保守估计）"),
            (362, 364, "恢复原来的训练/推理模式，返回准确率与样本数"),
            (367, 369, "分节注释：3.5c 主体抽取的验证"),
            (370, 384, "从回答里抽主体：取最后一个「是」或「：」后面的内容，去掉句号引号。这样模型不按模板、直接吐公司名也能处理"),
            (387, 399, "判匹配：严格=去掉空格标点后完全相同；宽松=一方包含另一方（「雅虎」vs「雅虎公司」）"),
            (402, 408, "实体验证函数文档：答案没有标签词，只能比抽出来的实体"),
            (409, 421, "抽样、切 eval 模式、判断模板是否支持 enable_thinking"),
            (424, 437, "逐条生成，最多 32 个新 token（实体的答案是一整句，比分类长）"),
            (439, 443, "两边都抽实体后比较，同时累计严格与宽松命中数"),
            (445, 450, "返回严格匹配率作为主指标，宽松值另行打印"),
            (453, 470, "answer_only_loss 的文档：把 lm_head 的显存账算给你看——序列 512、batch 2 时光这一层就约 600MB"),
            (471, 474, "取出基座模型，只跑 transformer 拿隐状态（不经过 lm_head），再拿到输出投影层"),
            (476, 490, "逐条找答案区间：定位非 -100 的位置，切出 [s-1, e-1) 这一段隐状态送进 lm_head，算交叉熵"),
            (491, 493, "没有答案 token 时返回 0（保持可反传），否则返回按 token 数平均的 loss"),
            (496, 505, "续训函数文档：只能恢复权重，恢复不了优化器动量"),
            (506, 524, "找 adapter 文件（safetensors 优先）、加载、灌进模型，并打印恢复了多少个张量"),
            (527, 535, "main 开头：打印显卡型号与总显存，先确认环境对不对"),
            (537, 547, "加载模型 → 可选续训 → 切数据 → 建 DataLoader（batch、打乱、collate、单进程加载）"),
            (549, 551, "只把 requires_grad 的参数交给 AdamW。这一行就是 LoRA 省显存的根：优化器只看 505 万参数"),
            (553, 562, "算清总步数、预热步数，打印训练计划"),
            (564, 569, "切训练模式、初始化步数与计时器"),
            (571, 576, "进入双重循环（epoch × batch），先把数据搬到 GPU"),
            (578, 587, "前向：fast_loss 走只算答案的写法，否则走 huggingface 原生。loss 除以累积步数，保证梯度量级等于大 batch"),
            (589, 590, "记录 loss、累加 micro_step"),
            (592, 596, "累积够 grad_accum 步：按当前步数算 lr，写进优化器"),
            (598, 608, "第一次更新前做梯度自检：统计拿到梯度的参数个数与梯度范数和。如果为 0 直接抛错，避免「打印着 loss 但参数一动不动」地空转 3 小时"),
            (610, 615, "梯度裁剪防炸、更新参数、清空梯度（set_to_none 比填 0 省显存）、步数 +1"),
            (617, 630, "每 log_every 步打印：平均 loss、当前 lr、峰值显存、已用时间、预计剩余。ETA 用「本次实际跑的步数」算，不能用 global_step（续训时会算小一半）"),
            (632, 642, "每 save_every 步落盘：只存 adapter（约 20MB）和历史曲线，断电至少有东西可用"),
            (644, 655, "epoch 末尾如果梯度没攒满，补一次更新。否则这批数据白算，梯度还会串到下一轮"),
            (657, 658, "打印本轮耗时"),
            (659, 677, "每轮结束跑验证：分类按标签词、实体按抽出的实体；验证创新高就另存一份 _best"),
            (680, 688, "保存 adapter、分词器、loss 曲线。只存 adapter 不存基座，所以是 20MB 而不是 1GB"),
            (690, 698, "保存验证曲线，并提示「最后一轮不是最好」——这时部署应该用 _best 而不是最后一轮"),
            (700, 703, "收尾打印总耗时与峰值显存"),
            (706, 707, "脚本入口：直接运行时才执行 main"),
        ],
        "walkthrough": [
            (146, "为什么可训练参数是 0.8395%",
             "LoRA 只给指定的线性层加两个小矩阵 A、B。我们把 q/k/v/o 与 gate/up/down 共 7 层都挂上，r=8，"
             "单个模块新增参数 = r×(输入维度+输出维度)。手算下来 5,046,272，占 601,096,192 的 0.84%，"
             "和这一行打印出来的数字完全一致。"),
            (246, "掩码为什么必须是 -100",
             "如果 prompt 也参与 loss，模型会连「用户怎么提问」一起背下来，等于把输入当成了要预测的内容。"
             "PyTorch 的 cross_entropy 有一个约定：label 为 -100 的位置直接跳过。所以这一行不是技巧，是标准做法。"),
            (325, "分层抽样这个坑值得单独说",
             "验证集是按类别顺序拼起来的（中性→消极→积极）。如果直接取前 300 条，拿到的全是中性类，"
             "报出来的 53.7% 其实只是「中性类准确率」。这不是模型问题，是评估代码的问题，所以我们改成先分组再抽。"),
            (493, "只对答案算 loss 的三步推导",
             "① lm_head 的输出是 (batch, 序列长度, 词表)：2×512×151,936 ≈ 1.5 亿个数，fp16 就是 311MB；"
             "② 反向传播还要给梯度留一份，再加 cross_entropy 内部升 fp32，这一层就要 0.3~0.9GB；"
             "③ 但被监督的只有答案那几个 token（分类 2~3 个、实体约 18 个）。"
             "于是这里先跑 transformer 拿隐状态，只把答案位置切片送进 lm_head，数学等价但显存省掉约 600MB。"
             "注意切片是 [s-1, e-1)：预测第 t 个 token 用的是第 t-1 个位置的隐状态。"),
            (608, "这十几行是整个脚本最值钱的自检",
             "写完新的 loss 函数很容易漏掉一处：隐状态没有参与反传。表现是 loss 照常打印、参数一动不动，"
             "跑满 3 小时你才发现模型没变。这段代码统计「有多少参数拿到了梯度」，为 0 就直接抛异常，把 3 小时的损失变成 3 秒钟的报错。"),
            (630, "ETA 算错一次的真实教训",
             "续训时 global_step 从 200 开始，但计时器只统计新跑的这段。如果直接 elapsed/global_step，"
             "每步耗时会算小一半，报出来的剩余时间偏乐观——我们当时把 54 分钟说成了 15 分钟。"),
        ],
    },
]

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root{{--bg:#f7f8fa;--card:#fff;--line:#e4e7eb;--text:#1f2328;--muted:#6b7280;
    --brand:#c2410c;--brand-soft:#fff4ec;--code:#0f172a;--code-bg:#f6f8fa;}}
  *{{box-sizing:border-box}}
  html{{scroll-behavior:smooth}}
  body{{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
    font-size:15px;line-height:1.8}}
  .layout{{display:grid;grid-template-columns:250px minmax(0,1fr);gap:36px;
    max-width:1240px;margin:0 auto;padding:0 24px 100px}}
  aside{{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;padding:30px 0}}
  aside .brand{{font-weight:650;margin-bottom:2px}}
  aside .sub{{color:var(--muted);font-size:12px;margin-bottom:16px}}
  aside a{{display:block;padding:5px 9px;border-radius:6px;color:var(--muted);
    text-decoration:none;font-size:13px}}
  aside a:hover{{background:#eef0f3;color:var(--text)}}
  aside .grp{{font-size:11px;color:#9aa3ad;margin:14px 0 4px;padding-left:9px;letter-spacing:.5px}}
  main{{padding-top:30px;min-width:0}}

  .hero{{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:30px 34px;margin-bottom:22px}}
  .hero h1{{margin:0 0 8px;font-size:23px;font-weight:650;letter-spacing:-.2px}}
  .hero p{{margin:6px 0;color:var(--muted);font-size:14px}}

  .file{{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:26px 30px 30px;margin-bottom:26px}}
  .file h2{{margin:0 0 6px;font-size:19px;font-weight:650}}
  .file .role{{color:var(--muted);font-size:13.5px;margin:0 0 18px}}
  .file h3{{font-size:14px;margin:26px 0 10px;font-weight:650;color:var(--brand)}}

  table.code{{width:100%;border-collapse:collapse;font-size:13px;margin:0 0 6px}}
  table.code td{{padding:0 8px;vertical-align:top;border-bottom:1px solid #f4f6f8}}
  .codecell{{overflow-x:auto;max-width:100%}}
  table.code tr:hover td{{background:#fbfcfd}}
  td.no{{width:44px;text-align:right;color:#b0b7c0;font-family:Consolas,monospace;
    font-size:11.5px;user-select:none;white-space:nowrap}}
  td.src{{font-family:"JetBrains Mono",Consolas,monospace;font-size:12.2px;
    line-height:1.5;color:#1f2328;width:54%}}
  td.src .codecell{{white-space:pre}}
  td.cm{{color:#5b6470;font-size:12px;line-height:1.5;padding-left:14px}}
  td.cm .cont{{color:#c9ced6}}
  tr.c td.src{{color:#3f7d4f}}
  tr.c td.cm{{color:#7b8794}}
  tr.blank td.cm{{color:#a8b0ba}}

  .note{{margin:22px 0 6px;border-left:3px solid var(--brand);background:var(--brand-soft);
    border-radius:0 10px 10px 0;padding:14px 18px}}
  .note b{{display:block;font-size:13.5px;margin-bottom:6px;color:#9a3412}}
  .note p{{margin:0;font-size:13.5px;color:#7c2d12;line-height:1.85}}

  code{{font-family:"JetBrains Mono",Consolas,monospace;font-size:12.6px;
    background:var(--code-bg);border:1px solid var(--line);border-radius:5px;padding:1px 5px}}

  @media (max-width:980px){{
    .layout{{grid-template-columns:1fr;padding:0 14px 60px}}
    aside{{position:static;max-height:none;border-bottom:1px solid var(--line)}}
    td.src{{width:auto}}
    td.cm{{display:none}}
  }}
</style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="brand">源码逐行精读</div><div style="margin:0 0 14px;font-size:12.5px;line-height:1.9"><a href="教程.html">教程</a> · <a href="面试问答.html">面试问答</a> · <a href="源码精读.html">源码精读</a></div>
    <div class="sub">FinCUGE-Tiny · 模型相关 Python 代码</div>
    {nav}
  </aside>
  <main>
    <div class="hero">
      <h1>源码逐行精读</h1>
      <p>左边是行号，中间是这一行的代码，右边是这一行在干什么。代码块之后配一段讲解，说清为什么这么写、当时踩了什么坑。</p>
      <p>读的顺序：先 <b>prompt_util.py</b>（24 行，决定训练与推理的输入一致），再 <b>train_sft.py</b>（训练全流程）。后续文件按 harness → eval → webui 补齐。</p>
    </div>
    {body}
  </main>
</div>
</body>
</html>
"""


def esc(s):
    return html.escape(s, quote=False)


def render_file(spec):
    path = os.path.join(HERE, spec["path"])
    lines = open(path, encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    # 行号 → 注释
    cmap = {}
    for s, e, txt in spec["blocks"]:
        for i in range(s, min(e, len(lines)) + 1):
            cmap[i] = txt
    notes = {ln: (t, p) for ln, t, p in spec.get("walkthrough", [])}

    out = [f'<section class="file" id="{spec["path"].replace(".", "-")}">',
           f'  <h2>{esc(spec["title"])}</h2>',
           f'  <p class="role">{esc(spec["role"])}</p>',
           '  <table class="code">']
    # 长区间的注释只在首行写全，后续行放一个竖线，避免同一句话刷屏
    starts = {s for s, e, _ in spec["blocks"]}
    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()
        cls = ""
        if not stripped:
            cls = "blank"
        elif stripped.startswith("#"):
            cls = "c"
        note = cmap.get(i, "")
        if note and i not in starts:
            note = '<span class="cont">│ 同上</span>'
        out.append(f'    <tr class="{cls}"><td class="no">{i}</td>'
                   f'<td class="src"><div class="codecell">{esc(raw) if stripped else "&nbsp;"}</div></td>'
                   f'<td class="cm">{note}</td></tr>')
        if i in notes:
            t, p = notes[i]
            out.append('  </table>')
            out.append(f'  <div class="note"><b>{esc(t)}</b><p>{esc(p)}</p></div>')
            out.append('  <table class="code">')
    out.append('  </table>')
    out.append('</section>')
    return "\n".join(out)


def main():
    nav = ['<div class="grp">模型相关代码</div>']
    for f in FILES:
        nav.append(f'<a href="#{f["path"].replace(".", "-")}">{f["path"]}</a>')
    nav.append('<div class="grp">待补</div>')
    for x in ("harness.py（641 行）", "eval_verbalizer.py（240 行）",
              "eval_multi.py（204 行）", "eval_entity.py（160 行）",
              "calibrate.py（89 行）", "webui.py（311 行）"):
        nav.append(f'<span style="display:block;padding:5px 9px;color:#b0b7c0;font-size:12.5px">{x}</span>')

    body = "\n".join(render_file(f) for f in FILES)
    html_out = TEMPLATE.format(title="源码逐行精读 · FinCUGE-Tiny",
                               nav="\n    ".join(nav), body=body)
    with open(DST, "w", encoding="utf-8") as fh:
        fh.write(html_out)
    print("写入", DST, os.path.getsize(DST), "字节")


if __name__ == "__main__":
    main()
