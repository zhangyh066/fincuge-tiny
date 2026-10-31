# -*- coding: utf-8 -*-
"""通用 markdown → 单页 HTML（和教程同一套样式）

用法：
    python build_md_html.py                       # 默认：面试问答.md → 面试问答.html
    python build_md_html.py 输入.md 输出.html "标题"
"""
import os
import sys
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))

TPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root{{--bg:#f7f8fa;--card:#fff;--line:#e4e7eb;--text:#1f2328;--muted:#6b7280;
    --brand:#c2410c;--brand-soft:#fff4ec;--code-bg:#f6f8fa;}}
  *{{box-sizing:border-box}} html{{scroll-behavior:smooth}}
  body{{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
    font-size:15px;line-height:1.85}}
  #bar{{position:fixed;top:0;left:0;height:3px;background:var(--brand);width:0;z-index:99}}
  .layout{{display:grid;grid-template-columns:256px minmax(0,1fr);gap:38px;
    max-width:1180px;margin:0 auto;padding:0 26px 110px}}
  aside{{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;padding:32px 0;font-size:13px}}
  aside .brand{{font-weight:650;font-size:14.5px;margin-bottom:2px}}
  aside .sub{{color:var(--muted);font-size:12px;margin-bottom:16px}}
  aside ul{{list-style:none;margin:0;padding-left:0}}
  aside ul ul{{padding-left:13px;border-left:1px solid var(--line);margin:2px 0 2px 3px}}
  aside a{{display:block;padding:3px 8px;border-radius:6px;color:var(--muted);text-decoration:none;line-height:1.5}}
  aside a:hover{{background:#eef0f3;color:var(--text)}}
  aside a.active{{background:var(--brand-soft);color:var(--brand);font-weight:600}}
  main{{padding-top:32px;min-width:0}}
  article{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:42px 50px 56px;
    box-shadow:0 1px 2px rgba(16,24,40,.04)}}
  article h1{{font-size:25px;margin:0 0 16px;font-weight:650;letter-spacing:-.2px}}
  article h1:not(:first-child){{margin-top:52px;padding-top:30px;border-top:1px solid var(--line)}}
  article h2{{font-size:19px;margin:34px 0 10px;font-weight:650}}
  article h3{{font-size:16px;margin:26px 0 8px;font-weight:650;color:var(--brand)}}
  article p{{margin:11px 0}} article strong{{font-weight:650}}
  article a{{color:var(--brand);text-decoration:none;border-bottom:1px solid rgba(194,65,12,.3)}}
  article ul,article ol{{margin:11px 0;padding-left:23px}} article li{{margin:5px 0}}
  article hr{{border:none;border-top:1px solid var(--line);margin:34px 0}}
  blockquote{{margin:15px 0;padding:11px 17px;background:var(--brand-soft);
    border-left:3px solid var(--brand);border-radius:0 8px 8px 0;color:#7c2d12}}
  blockquote p{{margin:5px 0}} blockquote code{{background:#fff;border-color:#f3d8c8}}
  pre{{position:relative;background:var(--code-bg);border:1px solid var(--line);border-radius:10px;
    padding:13px 15px;overflow-x:auto;margin:15px 0}}
  pre code{{font-family:"JetBrains Mono",Consolas,monospace;font-size:12.7px;line-height:1.65;color:#24292f;background:none;border:none;padding:0}}
  code{{font-family:"JetBrains Mono",Consolas,monospace;font-size:12.8px;background:var(--code-bg);
    border:1px solid var(--line);border-radius:5px;padding:1px 5px}}
  table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:13.6px;border:1px solid var(--line);
    border-radius:10px;overflow:hidden}}
  th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}
  th{{background:#f2f4f7;font-weight:650;white-space:nowrap}}
  tr:last-child td{{border-bottom:none}} tbody tr:nth-child(even){{background:#fbfcfd}}
  #top{{position:fixed;right:24px;bottom:24px;width:38px;height:38px;border-radius:50%;background:#fff;
    border:1px solid var(--line);display:grid;place-items:center;color:var(--muted);text-decoration:none;
    box-shadow:0 2px 8px rgba(16,24,40,.08);opacity:0;pointer-events:none;transition:.2s}}
  #top.show{{opacity:1;pointer-events:auto}}
  @media (max-width:960px){{.layout{{grid-template-columns:1fr;padding:0 14px 60px}}
    aside{{position:static;max-height:none;border-bottom:1px solid var(--line);padding:22px 0 10px}}
    aside ul ul{{display:none}} article{{padding:24px 20px 36px}}}}
</style>
</head>
<body>
<div id="bar"></div>
<div class="layout">
  <aside><div class="brand">{title}</div><div style="margin:0 0 14px;font-size:12.5px;line-height:1.9"><a href="教程.html">教程</a> · <a href="面试问答.html">面试问答</a> · <a href="源码精读.html">源码精读</a></div><div class="sub">{sub}</div><nav>{toc}</nav></aside>
  <main><article>{body}</article></main>
</div>
<a id="top" href="#">↑</a>
<script>
const bar=document.getElementById('bar'),top=document.getElementById('top');
function s(){{const h=document.documentElement,p=h.scrollTop/(h.scrollHeight-h.clientHeight||1);
  bar.style.width=(p*100).toFixed(2)+'%';top.classList.toggle('show',h.scrollTop>600)}}
document.addEventListener('scroll',s,{{passive:true}});s();
const links=[...document.querySelectorAll('aside a')];
const tg=links.map(a=>document.querySelector(decodeURIComponent(a.getAttribute('href')))).filter(Boolean);
const io=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting){{const id=e.target.id;
  links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+id));}}}}),
  {{rootMargin:'-8% 0px -84% 0px'}});
tg.forEach(t=>io.observe(t));
</script>
</body></html>
"""


def build(src, dst, title, sub):
    text = open(os.path.join(HERE, src), encoding="utf-8").read()
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "1-2"}})
    body = md.convert(text)
    out = TPL.format(title=title, sub=sub, toc=md.toc, body=body)
    with open(os.path.join(HERE, dst), "w", encoding="utf-8") as f:
        f.write(out)
    print("写入", dst, os.path.getsize(os.path.join(HERE, dst)), "字节")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        build("docs/面试问答.md", "docs/面试问答.html", "面试问答准备", "FinCUGE-Tiny · 依据公开面经整理")
