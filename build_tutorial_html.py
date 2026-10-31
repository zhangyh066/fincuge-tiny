# -*- coding: utf-8 -*-
"""把 docs/教程-从零微调.md 编译成单页 HTML（带侧边目录、代码复制、阅读进度）

用法：
    python build_tutorial_html.py
输出：
    docs/教程.html
"""
import os
import re
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "docs", "教程-从零微调.md")
DST = os.path.join(HERE, "docs", "教程.html")

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root{{
    --bg:#f7f8fa; --card:#ffffff; --line:#e4e7eb; --text:#1f2328; --muted:#6b7280;
    --brand:#c2410c; --brand-soft:#fff4ec; --code-bg:#f6f8fa;
  }}
  *{{box-sizing:border-box}}
  html{{scroll-behavior:smooth}}
  body{{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
    font-size:15px;line-height:1.85;-webkit-font-smoothing:antialiased}}

  #progress{{position:fixed;top:0;left:0;height:3px;background:var(--brand);
    width:0;z-index:100;transition:width .1s linear}}

  .layout{{display:grid;grid-template-columns:264px minmax(0,1fr);gap:40px;
    max-width:1180px;margin:0 auto;padding:0 28px 120px}}

  /* 侧边目录 */
  aside{{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;
    padding:34px 0 40px;font-size:13px}}
  aside .brand{{font-weight:650;font-size:15px;margin-bottom:4px}}
  aside .sub{{color:var(--muted);font-size:12px;margin-bottom:18px}}
  aside .toc ul{{list-style:none;margin:0;padding-left:0}}
  aside .toc ul ul{{padding-left:14px;border-left:1px solid var(--line);margin:2px 0 2px 3px}}
  aside .toc li{{margin:1px 0}}
  aside .toc a{{display:block;padding:3px 8px;border-radius:6px;color:var(--muted);
    text-decoration:none;line-height:1.5}}
  aside .toc a:hover{{background:#eef0f3;color:var(--text)}}
  aside .toc a.active{{background:var(--brand-soft);color:var(--brand);font-weight:600}}

  /* 正文 */
  main{{padding-top:34px;min-width:0}}
  article{{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:44px 52px 60px;box-shadow:0 1px 2px rgba(16,24,40,.04)}}

  article h1{{font-size:26px;line-height:1.35;margin:0 0 18px;font-weight:650;letter-spacing:-.2px}}
  article h1:not(:first-child){{margin-top:56px;padding-top:34px;border-top:1px solid var(--line)}}
  article h2{{font-size:20px;margin:38px 0 12px;font-weight:650}}
  article h3{{font-size:16.5px;margin:28px 0 8px;font-weight:650;color:var(--brand)}}
  article h1+blockquote{{margin-top:0}}

  article p{{margin:12px 0}}
  article strong{{font-weight:650}}
  article a{{color:var(--brand);text-decoration:none;border-bottom:1px solid rgba(194,65,12,.3)}}
  article a:hover{{border-bottom-color:var(--brand)}}

  article ul,article ol{{margin:12px 0;padding-left:24px}}
  article li{{margin:5px 0}}
  article hr{{border:none;border-top:1px solid var(--line);margin:38px 0}}

  /* 引用块：本教程用来放"应该看到""如果不对怎么办" */
  blockquote{{margin:16px 0;padding:12px 18px;background:var(--brand-soft);
    border-left:3px solid var(--brand);border-radius:0 8px 8px 0;color:#7c2d12}}
  blockquote p{{margin:6px 0}}
  blockquote code{{background:#fff;border-color:#f3d8c8}}

  /* 代码 */
  pre{{position:relative;background:var(--code-bg);border:1px solid var(--line);
    border-radius:10px;padding:14px 16px;overflow-x:auto;margin:16px 0}}
  pre code{{font-family:"JetBrains Mono",Consolas,"Courier New",monospace;font-size:12.8px;
    line-height:1.65;color:#24292f;background:none;padding:0;border:none;white-space:pre}}
  code{{font-family:"JetBrains Mono",Consolas,"Courier New",monospace;font-size:12.9px;
    background:var(--code-bg);border:1px solid var(--line);border-radius:5px;padding:1px 5px}}
  .copy{{position:absolute;top:8px;right:8px;font-size:11.5px;color:var(--muted);
    background:#fff;border:1px solid var(--line);border-radius:6px;padding:3px 8px;
    cursor:pointer;opacity:0;transition:opacity .15s}}
  pre:hover .copy{{opacity:1}}
  .copy:hover{{color:var(--brand);border-color:#f0c9b4}}

  /* 表格 */
  table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:13.8px;
    border:1px solid var(--line);border-radius:10px;overflow:hidden}}
  th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}
  th{{background:#f2f4f7;font-weight:650;white-space:nowrap}}
  tr:last-child td{{border-bottom:none}}
  tbody tr:nth-child(even){{background:#fbfcfd}}
  article td:first-child{{white-space:nowrap}}

  .meta{{color:var(--muted);font-size:12.5px;margin:0 0 26px}}

  #top{{position:fixed;right:26px;bottom:26px;width:40px;height:40px;border-radius:50%;
    background:#fff;border:1px solid var(--line);box-shadow:0 2px 8px rgba(16,24,40,.08);
    display:grid;place-items:center;cursor:pointer;color:var(--muted);text-decoration:none;
    font-size:15px;opacity:0;pointer-events:none;transition:opacity .2s}}
  #top.show{{opacity:1;pointer-events:auto}}

  @media (max-width:960px){{
    .layout{{grid-template-columns:1fr;gap:0;padding:0 16px 80px}}
    aside{{position:static;max-height:none;padding:24px 0 8px;border-bottom:1px solid var(--line);margin-bottom:16px}}
    aside .toc ul ul{{display:none}}
    article{{padding:26px 22px 40px;border-radius:12px}}
    article h1{{font-size:22px}}
  }}
  @media print{{
    aside,#progress,#top,.copy{{display:none!important}}
    .layout{{display:block;max-width:none;padding:0}}
    article{{border:none;box-shadow:none;padding:0}}
    pre,table{{break-inside:avoid}}
  }}
</style>
</head>
<body>
<div id="progress"></div>
<div class="layout">
  <aside>
    <div class="brand">FinCUGE-Tiny 教程</div>
    <div class="sub">从零开始 · 4GB 显卡微调</div>
    <nav class="toc">{toc}</nav>
  </aside>
  <main>
    <article>
      {body}
    </article>
  </main>
</div>
<a id="top" href="#" title="回到顶部">↑</a>
<script>
const bar = document.getElementById('progress');
const topBtn = document.getElementById('top');
function onScroll(){{
  const h = document.documentElement;
  const p = h.scrollTop / (h.scrollHeight - h.clientHeight || 1);
  bar.style.width = (p * 100).toFixed(2) + '%';
  topBtn.classList.toggle('show', h.scrollTop > 600);
}}
document.addEventListener('scroll', onScroll, {{passive:true}}); onScroll();

// 目录高亮
const links = [...document.querySelectorAll('aside .toc a')];
const targets = links.map(a => document.querySelector(decodeURIComponent(a.getAttribute('href')))).filter(Boolean);
const io = new IntersectionObserver(es => {{
  es.forEach(e => {{
    if (!e.isIntersecting) return;
    const id = e.target.id;
    links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + id));
  }});
}}, {{rootMargin: '-10% 0px -80% 0px'}});
targets.forEach(t => io.observe(t));

// 代码复制
document.querySelectorAll('pre').forEach(pre => {{
  const btn = document.createElement('button');
  btn.className = 'copy'; btn.textContent = '复制';
  btn.onclick = async () => {{
    const code = pre.querySelector('code');
    try {{
      await navigator.clipboard.writeText(code ? code.innerText : pre.innerText);
      btn.textContent = '已复制'; setTimeout(() => btn.textContent = '复制', 1200);
    }} catch(e) {{ btn.textContent = '复制失败'; }}
  }};
  pre.appendChild(btn);
}});
</script>
</body>
</html>
"""


def main():
    src = open(SRC, encoding="utf-8").read()
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
                           extension_configs={"toc": {"toc_depth": "1", "permalink": False}})
    body = md.convert(src)
    toc = md.toc

    # 给每个 h1（第 N 步）后面补一个"回目录"链接，长文档里很实用
    body = body.replace("</h1>", '</h1>')

    title = "从零开始：在 4GB 显卡上微调中文金融小模型"
    html = TEMPLATE.format(title=title, toc=toc, body=body)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(html)
    print("生成", DST, os.path.getsize(DST), "字节")
    print("小节目录条目数：", toc.count("<a href="))


if __name__ == "__main__":
    main()
