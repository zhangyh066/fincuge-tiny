# -*- coding: utf-8 -*-
"""把「教程 + 源码逐行精读 + 面试问答」合成一个 HTML 手册

用法：python build_handbook.py
输出：docs/项目手册.html
"""
import html
import os
import re

import markdown

from build_code_reading import FILES as BASE_FILES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "项目手册.html")

HARNESS = {
    "path": "harness.py",
    "title": "harness.py · 外壳：路由 / 校验 / 三档判定（641 行）",
    "role": "模型之外自建的那一层。它决定一次请求走哪个技能、答案合不合法、能不能信，以及把每次调用记成一行日志。",
    "blocks": [
        (1, 1, "编码声明：UTF-8"),
        (2, 18, "模块文档：六步流程（清洗→路由→执行→校验→判定→日志）、三条铁律（技能不互调、提示词逐字节固定、日志可复原）、四种用法"),
        (19, 26, "导入标准库：argparse/hashlib/json/os/re/sys/time/Counter"),
        (28, 29, "输出改 UTF-8；把当前目录加进 sys.path，保证被别的目录调用时也能 import 到同目录模块"),
        (31, 33, "torch、transformers、peft"),
        (35, 36, "从 prompt_util 导入提示词拼装、从 train_sft 导入实体抽取与匹配——判定和训练必须同一套规则，否则分数对不上"),
        (38, 38, "基座路径三级回退：环境变量 → 本地目录 → HuggingFace 名字"),
        (39, 39, "日志文件：每次调用追加一行 JSONL"),
        (41, 43, "输入上限 448 token。训练数据 P99 只有 201，真实新闻可能长得多；超长必须显式处理，不能静默截断"),
        (45, 52, "自一致性投票的四个常量。注释里直接写了实测结论：0.88 触发 33%、89.0%→91.0%、成本 1.66 倍"),
        (53, 53, "当前文件所在目录，用于把相对路径的 adapter 解析成绝对路径"),
        (55, 60, "三个任务各自的 adapter 候选目录，按顺序找"),
        (63, 68, "解析 adapter 路径：绝对路径直接用，相对路径按 HERE 拼；判断标准是目录里有 adapter_config.json，不是目录存在"),
        (70, 74, "三个评估集文件路径，标定和自检都用它"),
        (76, 78, "情绪三个标签的 token id。受限 softmax 只读这几个位置，所以必须写死并和评估脚本一致"),
        (80, 90, "新闻七类的标签与别名表：模型可能输出「板块」「指数」这类同义词，解析时按别名归一"),
        (92, 96, "SKILLS 的总注释：指令从训练数据里取最常见的一条、逐字节固定；实体任务有 121 种指令且都带事件类型"),
        (97, 105, "情绪技能配置：受限 softmax、指令固定、阈值 0.90/0.50。注释给的是 2,020 条实测的覆盖率与准确率"),
        (106, 114, "新闻技能配置：生成式、最多 16 token、阈值 0.95。注释说明它没有低置信带，所以 lo=0.5 实际不会触发"),
        (115, 123, "实体技能配置：生成式、24 token、阈值 0.89/0.79，注释写明低于 0.79 的样本只剩 1.4% 且准确率 5%"),
        (125, 133, "路由规则表。注释强调「只在高精度时生效，宁可漏判交给反问」——这条决定了没命中就返回 None 而不是瞎猜"),
        (136, 140, "清洗：去首尾空白 + 把 3 个以上连续换行压成 2 个"),
        (143, 158, "路由：对每个关键词找「最后一次出现」的位置，取位置最靠后的那个任务。中文语序里后置的往往是真实意图；一个都没命中返回 None"),
        (161, 173, "新闻答案解析：按别名找命中位置，长词优先（避免「大盘」被「大」之类误吃），找不到返回 None"),
        (176, 184, "初始化：分词器（pad 用 eos、padding 靠左）、fp16 基座。推理用 left padding，和训练时的 right 不同"),
        (185, 191, "把三个 adapter 全挂上同一个基座（每个约 10~20MB），之后按名字切换，不用重载模型"),
        (192, 196, "切 eval 模式；判断模板是否支持 enable_thinking；verbose 时打印挂载结果"),
        (199, 202, "_use：只有任务变了才 set_adapter，避免每次请求都切一遍"),
        (204, 210, "_encode：走模型自带 chat 模板、关掉思考模式、返回 GPU 上的 input_ids"),
        (212, 227, "_emotion 的受限 softmax：末尾拼接一个「中」字，用它和金标位置上的「性」联合算出中性概率；其余两个标签直接读对应 token 的概率，最后在三者上重新归一化"),
        (229, 247, "_generate：贪婪生成，同时收集每一步的 logprob，取几何平均作为置信度（也就是平均每 token 概率）"),
        (249, 268, "_generate_vote：采样 k 次（温度 0.7、top_p 0.9），每次抽出实体，取众数；置信度用「一致度」= 众数票数/k"),
        (270, 276, "两级校验的定义：BLOCKING 是必须拒答的（标签非法、实体为空、实体不在原文）；DOWNGRADE 只把 auto 降成 hint（多行、长度、标点、多个实体）"),
        (278, 304, "_validate 的具体规则：先看是否多行；分类任务看标签是否合法；实体任务依次查空、长度 2~40、标点、是否出现在原文、是否像在列多个答案"),
        (306, 331, "answer 前半段：清洗 → 超长截断 → 空输入拒答 → 路由（或上游指定任务）→ 组装 row"),
        (333, 356, "执行：受限 softmax 或生成式；新闻做别名解析、解析失败取前 10 字；实体先抽实体，置信度低于 0.88 再投票并覆盖结果"),
        (358, 366, "汇总 flags：voted、input_truncated、instruction_synthesized，以及未标定的技能"),
        (368, 385, "三档判定。注意：投票后的置信度是一致度，尺度不同，所以单独一套规则——1.0 才可能 auto，0.67 只给 hint，0.33 直接拒答"),
        (387, 401, "组装返回：schema_valid、answer（不合法就统一回「我不确定」）、置信度、决策、flags、路由依据、三类概率、耗时、用输入哈希做 trace_id"),
        (404, 410, "_reject：拒答的统一出口，保证拒答和正常回答的字段结构完全一致"),
        (412, 420, "_log：一行 JSONL 落盘，写失败也不影响主流程（用 try 包住）"),
        (426, 470, "selftest：拿 harness 的输出和三个评估脚本的缓存逐条比对。这是「外壳没有偷偷改判分口径」的证据"),
        (473, 490, "情绪别名表 + gold_of：从金标里取标准答案，三种任务三种取法"),
        (493, 551, "calibrate：跑整个评估集，把每条样本的 (置信度, 对错) 落盘，之后画覆盖率-准确率曲线定阈值。带断点续跑，每 50 条 flush"),
        (554, 607, "chat：交互模式。实体任务会追问事件类型，因为训练指令里带事件类型，不给就是分布外输入"),
        (610, 641, "main：参数分派（chat / calibrate / selftest / 单条推理）"),
    ],
    "walkthrough": [
        (227, "受限 softmax 为什么要拼一个「中」字",
         "情绪三个标签里，「积极」「消极」各是一个 token，直接读概率就行。但「中性」在词表里不是一个 token，"
         "模型输出的是「中」+「性」两个字。所以这里把「中」拼到输入末尾，读最后一个位置的 P(中)，"
         "再读「中」之后位置的 P(性|中)，两者相乘就是中性的概率。三个数再归一化，得到三类概率。"),
        (304, "校验为什么必须分两级",
         "早期版本把「公司名带括号」判成标点违规，直接拒答。但「××（中国）有限公司」是合法公司名，"
         "结果自检 49/50。改成分级后：只有「标签非法、实体为空、实体不在原文」才拒答，"
         "括号、长度这类只降级成「建议核对」，自检回到 50/50。这条经验对所有做校验层的人都适用。"),
        (385, "为什么投票后要另写一套判定",
         "贪婪解的置信度是 token 概率的几何平均，一般落在 0.5~1.0；投票后是「一致度」，只有 1.0 / 0.67 / 0.33 三个档。"
         "如果沿用 0.90/0.50 这两个阈值，一个 2/3 票一致、实际答对的样本会被判成拒答。"
         "我第一次接投票时就踩了这个坑，所以这里单独写了一套。"),
    ],
}

FILES = BASE_FILES + [HARNESS]

PART1 = ("第一部分 · 从零上手", "docs/教程-从零微调.md",
         "14 个步骤，每步写清「这一节解决什么、你要做什么、应该看到什么」。")
PART3 = ("第三部分 · 面试问答", "docs/面试问答.md",
         "25 道题，含三道白板推导。依据是四份公开面经，答案里的数字都来自本项目实测。")

TPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinCUGE-Tiny 项目手册</title>
<style>
  :root{--bg:#f7f8fa;--card:#fff;--line:#e4e7eb;--text:#1f2328;--muted:#6b7280;
    --brand:#c2410c;--brand-soft:#fff4ec;--code-bg:#f6f8fa;}
  *{box-sizing:border-box} html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
    font-size:15px;line-height:1.85}
  #bar{position:fixed;top:0;left:0;height:3px;background:var(--brand);width:0;z-index:99}
  .layout{display:grid;grid-template-columns:262px minmax(0,1fr);gap:36px;
    max-width:1240px;margin:0 auto;padding:0 24px 110px}
  aside{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;padding:30px 0;font-size:13px}
  aside .brand{font-weight:650;font-size:14.5px}
  aside .sub{color:var(--muted);font-size:11.5px;margin:2px 0 14px}
  aside .grp{font-size:11px;color:#9aa3ad;margin:16px 0 5px;padding-left:8px;letter-spacing:.4px}
  aside ul{list-style:none;margin:0;padding-left:0}
  aside ul ul{padding-left:12px;border-left:1px solid var(--line);margin:2px 0 2px 3px}
  aside a{display:block;padding:2.5px 8px;border-radius:6px;color:var(--muted);text-decoration:none;line-height:1.5}
  aside a:hover{background:#eef0f3;color:var(--text)}
  aside a.active{background:var(--brand-soft);color:var(--brand);font-weight:600}
  main{padding-top:28px;min-width:0}
  .hero{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px 34px;margin-bottom:20px}
  .hero h1{margin:0 0 8px;font-size:24px;font-weight:650;letter-spacing:-.2px}
  .hero p{margin:6px 0;color:var(--muted);font-size:14px}
  .hero .toc3{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
  .hero .toc3 a{flex:1;min-width:200px;border:1px solid var(--line);border-radius:10px;padding:10px 14px;
    text-decoration:none;color:var(--text);font-size:13px;background:#fbfcfd}
  .hero .toc3 a:hover{border-color:#f0c9b4;background:var(--brand-soft)}
  .hero .toc3 b{display:block;font-size:13.5px;margin-bottom:2px}
  .hero .toc3 span{color:var(--muted);font-size:12px}
  article{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:40px 48px 54px}

  .part{font-size:22px;font-weight:650;margin:8px 0 6px;padding-top:6px}
  .part-lead{color:var(--muted);font-size:13.5px;margin:0 0 22px}

  article h1{font-size:21px;margin:44px 0 14px;font-weight:650;padding-top:26px;border-top:1px solid var(--line)}
  article h1:first-of-type{border-top:none;padding-top:0}
  article h2{font-size:18px;margin:30px 0 10px;font-weight:650}
  article h3{font-size:15.5px;margin:22px 0 8px;font-weight:650;color:var(--brand)}
  article p{margin:10px 0} article strong{font-weight:650}
  article ul,article ol{margin:10px 0;padding-left:22px} article li{margin:4px 0}
  article hr{border:none;border-top:1px solid var(--line);margin:30px 0}
  blockquote{margin:14px 0;padding:10px 16px;background:var(--brand-soft);
    border-left:3px solid var(--brand);border-radius:0 8px 8px 0;color:#7c2d12}
  blockquote p{margin:4px 0} blockquote code{background:#fff;border-color:#f3d8c8}
  pre{background:var(--code-bg);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
    overflow-x:auto;margin:14px 0}
  pre code{font-family:"JetBrains Mono",Consolas,monospace;font-size:12.6px;line-height:1.6;color:#24292f;
    background:none;border:none;padding:0}
  code{font-family:"JetBrains Mono",Consolas,monospace;font-size:12.7px;background:var(--code-bg);
    border:1px solid var(--line);border-radius:5px;padding:1px 5px}
  table{border-collapse:collapse;margin:15px 0;font-size:13.5px;border:1px solid var(--line);
    border-radius:10px;overflow:hidden;width:100%}
  th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
  th{background:#f2f4f7;font-weight:650;white-space:nowrap}
  tr:last-child td{border-bottom:none} tbody tr:nth-child(even){background:#fbfcfd}
  article table.code{font-size:13px;margin:0 0 4px;border:none;border-radius:0;overflow:visible}
  article table.code td{padding:0 8px;border-bottom:1px solid #f4f6f8}
  article table.code tr:hover td{background:#fbfcfd}
  td.no{width:42px;text-align:right;color:#b0b7c0;font-family:Consolas,monospace;font-size:11.5px;user-select:none;white-space:nowrap}
  td.src{font-family:"JetBrains Mono",Consolas,monospace;font-size:12.1px;line-height:1.5;color:#1f2328;width:54%}
  td.src .codecell{white-space:pre;overflow-x:auto;max-width:100%}
  td.cm{color:#5b6470;font-size:11.9px;line-height:1.5;padding-left:14px}
  td.cm .cont{color:#c9ced6}
  tr.c td.src{color:#3f7d4f}
  .note{margin:20px 0 4px;border-left:3px solid var(--brand);background:var(--brand-soft);
    border-radius:0 10px 10px 0;padding:13px 18px}
  .note b{display:block;font-size:13.5px;margin-bottom:5px;color:#9a3412}
  .note p{margin:0;font-size:13.4px;color:#7c2d12}
  .filecard{border:1px solid var(--line);border-radius:12px;padding:20px 24px 24px;margin:22px 0}
  .filecard h3{margin:0 0 4px;font-size:16.5px;color:var(--text)}
  .filecard .role{color:var(--muted);font-size:13px;margin:0 0 14px}
  .pending{border:1px dashed #d7dbe0;border-radius:10px;padding:14px 18px;margin:18px 0;
    color:var(--muted);font-size:13px;background:#fcfcfd}
  .pending b{color:var(--text)}
  #top{position:fixed;right:22px;bottom:22px;width:36px;height:36px;border-radius:50%;background:#fff;
    border:1px solid var(--line);display:grid;place-items:center;color:var(--muted);text-decoration:none;
    box-shadow:0 2px 8px rgba(16,24,40,.08);opacity:0;pointer-events:none;transition:.2s;font-size:14px}
  #top.show{opacity:1;pointer-events:auto}
  @media (max-width:1000px){.layout{grid-template-columns:1fr;padding:0 14px 70px}
    aside{position:static;max-height:none;border-bottom:1px solid var(--line);padding:20px 0 8px}
    aside ul ul{display:none} article{padding:22px 18px 34px} td.cm{display:none}}
</style>
</head>
<body>
<div id="bar"></div>
<div class="layout">
  <aside>
    <div class="brand">FinCUGE-Tiny 项目手册</div>
    <div class="sub">教程 · 源码 · 面试</div>
    __NAV__
  </aside>
  <main>
    <div class="hero">
      <h1>FinCUGE-Tiny 项目手册</h1>
      <p>三部分合成一本：怎么从零跑起来、每一行代码在干什么、面试怎么答。</p>
      <div class="toc3">
        <a href="#part1"><b>第一部分 · 从零上手</b><span>14 步，装环境到评估</span></a>
        <a href="#part2"><b>第二部分 · 源码精读</b><span>逐行注释，含讲解</span></a>
        <a href="#part3"><b>第三部分 · 面试问答</b><span>25 题，含白板推导</span></a>
      </div>
    </div>
    <article>
      __BODY__
    </article>
  </main>
</div>
<a id="top" href="#">↑</a>
<script>
const bar=document.getElementById('bar'),top=document.getElementById('top');
function s(){const h=document.documentElement,p=h.scrollTop/(h.scrollHeight-h.clientHeight||1);
  bar.style.width=(p*100).toFixed(2)+'%';top.classList.toggle('show',h.scrollTop>600)}
document.addEventListener('scroll',s,{passive:true});s();
const links=[...document.querySelectorAll('aside a')];
const tg=links.map(a=>document.querySelector(decodeURIComponent(a.getAttribute('href')))).filter(Boolean);
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;
  links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+id));}}),
  {rootMargin:'-8% 0px -84% 0px'});
tg.forEach(t=>io.observe(t));
</script>
</body></html>
"""


def md_to_html(path, depth="1-2"):
    raw = open(os.path.join(HERE, path), encoding="utf-8").read()
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": depth}})
    body = md.convert(raw)
    return body, md.toc


def esc(s):
    return html.escape(s, quote=False)


def render_file(spec):
    lines = open(os.path.join(HERE, spec["path"]), encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    cmap = {}
    for s, e, t in spec["blocks"]:
        for i in range(s, min(e, len(lines)) + 1):
            cmap[i] = t
    starts = {s for s, e, _ in spec["blocks"]}
    notes = {ln: (t, p) for ln, t, p in spec.get("walkthrough", [])}

    out = [f'<div class="filecard" id="{spec["path"].replace(".", "-")}">',
           f'  <h3>{esc(spec["title"])}</h3>',
           f'  <p class="role">{esc(spec["role"])}</p>',
           '  <table class="code">']
    for i, raw in enumerate(lines, 1):
        st = raw.strip()
        cls = "blank" if not st else ("c" if st.startswith("#") else "")
        note = cmap.get(i, "")
        if not st:
            note = ""
        elif note and i not in starts:
            note = '<span class="cont">│ 同上</span>'
        out.append(f'    <tr class="{cls}"><td class="no">{i}</td>'
                   f'<td class="src"><div class="codecell">{esc(raw) if st else "&nbsp;"}</div></td>'
                   f'<td class="cm">{note}</td></tr>')
        if i in notes:
            t, p = notes[i]
            out.append('  </table>')
            out.append(f'  <div class="note"><b>{esc(t)}</b><p>{esc(p)}</p></div>')
            out.append('  <table class="code">')
    out.append('  </table>')
    out.append('</div>')
    return "\n".join(out)


def main():
    tut_body, tut_toc = md_to_html(PART1[1], depth="1")
    qa_body, qa_toc = md_to_html(PART3[1])
    # 去掉各自的一级标题（由 part 标题承担）
    tut_body = re.sub(r"<h1[^>]*>.*?</h1>", "", tut_body, count=1)
    qa_body = re.sub(r"<h1[^>]*>.*?</h1>", "", qa_body, count=1)

    parts = [
        f'<h1 class="part" id="part1">{PART1[0]}</h1>',
        f'<p class="part-lead">{PART1[2]}</p>',
        tut_body,
        f'<h1 class="part" id="part2">第二部分 · 源码逐行精读</h1>',
        '<p class="part-lead">左边行号、中间代码、右边这一行在干什么。区间注释只在第一行写全，其余标「同上」；'
        '代码块之后的方框讲「为什么这么写」。</p>',
    ]
    parts += [render_file(f) for f in FILES]
    parts.append('<div class="pending"><b>下一批待补：</b>'
                 'eval_verbalizer.py（240 行）、eval_multi.py（204 行）、eval_entity.py（160 行）、'
                 'calibrate.py（89 行）、webui.py（311 行）。'
                 '这几个是评估与前端，处理方式同上：先读代码，再逐行写注释。</div>')
    parts += [
        f'<h1 class="part" id="part3">{PART3[0]}</h1>',
        f'<p class="part-lead">{PART3[2]}</p>',
        qa_body,
    ]

    nav = ['<div class="grp">三部分</div>',
           '<a href="#part1">第一部分 · 从零上手</a>',
           '<a href="#part2">第二部分 · 源码精读</a>',
           '<a href="#part3">第三部分 · 面试问答</a>',
           '<div class="grp">源码文件</div>']
    for f in FILES:
        nav.append(f'<a href="#{f["path"].replace(".", "-")}">{f["path"]}</a>')
    nav.append('<div class="grp">教程步骤 / 面试题</div>')
    nav.append('<div style="padding:0 8px">' + tut_toc + qa_toc + '</div>')

    out = TPL.replace("__NAV__", "\n    ".join(nav)).replace("__BODY__", "\n".join(parts))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print("写入", OUT, os.path.getsize(OUT), "字节")


if __name__ == "__main__":
    main()
