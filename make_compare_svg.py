# -*- coding: utf-8 -*-
"""生成三张公开对照图（每个任务一张，横向条形）

样式与 README 开头的 assets/results.svg 保持一致：
    灰 = 公开模型，橙 = 本项目，浅灰网格，同一刻度 0~1
"""
import os

W, H = 760, 336
X0, X1 = 236, 636          # 条形区域 400px 对应 0~1.0
ROW_H, ROW_GAP = 16, 32
TOP = 68
GRAY, ORANGE = "#b6bcc4", "#e0672a"
TEXT, MUTED, GRID = "#1f2328", "#57606a", "#eaecef"

TASKS = {
    "emotion": {
        "title": "情绪三分类 · FinFE-CN · Weighted-F1",
        "rows": [
            ("Touchstone-GPT", 0.7888, "7B · 金融预训练 + 指令微调", GRAY),
            ("本项目 Qwen3-0.6B", 0.7542, "0.6B · LoRA / 4GB 显卡", ORANGE),
            ("GPT-4o", 0.6593, "零样本", GRAY),
            ("Qwen2-7B-Instruct", 0.6274, "零样本", GRAY),
            ("DISC-FinLLM", 0.4177, "13B · 金融指令微调", GRAY),
            ("Llama-3-8B-Instruct", 0.3633, "零样本", GRAY),
            ("CFGPT1-7B", 0.2528, "7B · 金融继续预训练", GRAY),
        ],
        "note": "评估切片：公开用 test，本项目用 valid。公开模型均为零样本，本项目为微调。",
    },
    "newscat": {
        "title": "新闻分类 · FinNL-CN · ORMACC（类别数不同，仅供参考）",
        "rows": [
            ("Touchstone-GPT", 0.8360, "15 类 · 金融预训练 + 微调", GRAY),
            ("本项目 Qwen3-0.6B", 0.8971, "7 类 · LoRA / 4GB 显卡", ORANGE),
            ("GPT-4o", 0.3303, "15 类 · 零样本", GRAY),
            ("CFGPT1-7B", 0.0894, "15 类 · 金融继续预训练", GRAY),
            ("Llama-3-8B-Instruct", 0.0747, "15 类 · 零样本", GRAY),
            ("Qwen2-7B-Instruct", 0.0622, "15 类 · 零样本", GRAY),
            ("DISC-FinLLM", 0.0011, "15 类 · 金融指令微调", GRAY),
        ],
        "note": "本项目为 7 类（长尾合并为其它），公开为 15 类，条形长度不可直接比较。",
    },
    "entity": {
        "title": "事件主体抽取 · FinESE-CN · ORMACC",
        "rows": [
            ("Touchstone-GPT", 0.9074, "7B · 金融预训练 + 指令微调", GRAY),
            ("本项目 Qwen3-0.6B", 0.8610, "0.6B · LoRA / 4GB 显卡", ORANGE),
            ("GPT-4o", 0.6867, "零样本", GRAY),
            ("DISC-FinLLM", 0.4346, "13B · 金融指令微调", GRAY),
            ("CFGPT1-7B", 0.3863, "7B · 金融继续预训练", GRAY),
            ("Qwen2-7B-Instruct", 0.3678, "零样本", GRAY),
            ("Llama-3-8B-Instruct", 0.3088, "零样本", GRAY),
        ],
        "note": "评估切片：公开用 test（1,782 条），本项目用 1,464 条。公开模型均为零样本，本项目为微调。",
    },
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(key, cfg):
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">',
         f'  <rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'  <text x="24" y="28" font-size="15" font-weight="600" fill="{TEXT}">{esc(cfg["title"])}</text>']

    # 图例
    p.append(f'  <rect x="556" y="18" width="11" height="11" fill="{GRAY}" rx="2"/>')
    p.append(f'  <text x="573" y="27" font-size="11" fill="{MUTED}">公开模型</text>')
    p.append(f'  <rect x="626" y="18" width="11" height="11" fill="{ORANGE}" rx="2"/>')
    p.append(f'  <text x="643" y="27" font-size="11" fill="{MUTED}">本项目</text>')

    # 网格与刻度
    for i in range(5):
        v = i / 4
        x = X0 + (X1 - X0) * v
        p.append(f'  <line x1="{x:.0f}" y1="{TOP - 8}" x2="{x:.0f}" y2="{TOP + 7*ROW_GAP - 8}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        p.append(f'  <text x="{x:.0f}" y="{TOP + 7*ROW_GAP + 8}" font-size="10" fill="#8c959f" '
                 f'text-anchor="middle">{v:.2f}</text>')

    # 每行
    for i, (name, val, desc, color) in enumerate(cfg["rows"]):
        y = TOP + i * ROW_GAP
        bw = max(1.5, (X1 - X0) * val)
        p.append(f'  <text x="{X0 - 10}" y="{y + 12}" font-size="12" fill="{TEXT}" text-anchor="end">{esc(name)}</text>')
        p.append(f'  <rect x="{X0}" y="{y}" width="{bw:.1f}" height="{ROW_H}" fill="{color}" rx="3"/>')
        weight = "600" if color == ORANGE else "400"
        fill = "#b8501c" if color == ORANGE else MUTED
        p.append(f'  <text x="{X0 + bw + 7:.1f}" y="{y + 12}" font-size="11.5" font-weight="{weight}" '
                 f'fill="{fill}">{val:.3f}</text>')
        p.append(f'  <text x="{X0 + bw + 48:.1f}" y="{y + 12}" font-size="10.5" fill="#8c959f">{esc(desc)}</text>')

    p.append(f'  <text x="24" y="{H - 10}" font-size="10.5" fill="#8c959f">{esc(cfg["note"])}</text>')
    p.append('</svg>')
    return "\n".join(p)


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(out, exist_ok=True)
    for key, cfg in TASKS.items():
        path = os.path.join(out, f"compare-{key}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(build(key, cfg))
        print("写入", path, os.path.getsize(path), "字节")


if __name__ == "__main__":
    main()
