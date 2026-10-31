# -*- coding: utf-8 -*-
"""生成架构草图 assets/arch.svg（五层横向流程）

配色与 results.svg / compare-*.svg 一致：灰底框、橙色强调。
"""
import os

W, H = 940, 272
BOX_W, BOX_H = 168, 132
GAP = 22
TOP = 74
X0 = 22

BRAND = "#c2410c"
BRAND_SOFT = "#fdece3"
LINE = "#e4e7eb"
TEXT = "#1f2328"
MUTED = "#57606a"
FAINT = "#8c959f"

LAYERS = [
    ("数据层", ["FinCUGE 三任务", "新闻 15 类 → 7 类", "序列按 P99 定"], "1.6 万条"),
    ("训练层", ["LoRA 0.84% 参数", "只对答案算 loss", "梯度累积 + 检查点"], "3 小时 / 任务"),
    ("评测层", ["受限 softmax", "实体四口径", "全量 + 置信分层"], "4 套指标"),
    ("外壳层", ["路由 → 校验", "三档判定 + 投票", "单行 trace"], "3 档判定"),
    ("前端", ["三标签页网页", "命令行交互", "一键启动脚本"], "1 秒级"),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build():
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">',
         f'  <rect width="{W}" height="{H}" fill="#ffffff"/>']

    for i, (name, items, foot) in enumerate(LAYERS):
        x = X0 + i * (BOX_W + GAP)
        highlight = name in ("外壳层",)
        fill = BRAND_SOFT if highlight else "#fbfcfd"
        stroke = BRAND if highlight else LINE
        p.append(f'  <rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="12" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{1.6 if highlight else 1}"/>')
        # 层名
        p.append(f'  <text x="{x + 16}" y="{TOP + 30}" font-size="14.5" font-weight="650" '
                 f'fill="{BRAND if highlight else TEXT}">{esc(name)}</text>')
        # 要点
        for j, it in enumerate(items):
            p.append(f'  <text x="{x + 16}" y="{TOP + 56 + j * 22}" font-size="11.5" fill="{MUTED}">'
                     f'· {esc(it)}</text>')
        # 底部指标
        p.append(f'  <text x="{x + 16}" y="{TOP + BOX_H - 13}" font-size="11.5" font-weight="600" '
                 f'fill="{BRAND if highlight else FAINT}">{esc(foot)}</text>')

        # 箭头
        if i < len(LAYERS) - 1:
            ax = x + BOX_W
            p.append(f'  <line x1="{ax + 4}" y1="{TOP + BOX_H / 2}" x2="{ax + GAP - 4}" '
                     f'y2="{TOP + BOX_H / 2}" stroke="#c9ced6" stroke-width="1.4"/>')
            p.append(f'  <path d="M{ax + GAP - 4},{TOP + BOX_H / 2} l-5,-3.4 v6.8 z" fill="#c9ced6"/>')

    # 顶部说明 + 底部说明
    p.append(f'  <text x="{X0}" y="36" font-size="13" font-weight="600" fill="{TEXT}">'
             f'五层结构：数据进去，结论出来，中间每一步都可单独验证</text>')
    p.append(f'  <text x="{X0}" y="58" font-size="11.5" fill="{FAINT}">'
             f'全部跑在一张 GTX 1650（4GB）上；橙色框是模型之外自建的调度与质检层</text>')
    p.append(f'  <text x="{X0}" y="{H - 20}" font-size="11" fill="{FAINT}">'
             f'训练与推理均离线，不依赖任何外部 API；模型权重 20MB/adapter，三个任务共 58MB。</text>')
    p.append('</svg>')
    return "\n".join(p)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "arch.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(build())
    print("写入", out, os.path.getsize(out), "字节")
