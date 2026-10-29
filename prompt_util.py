# -*- coding: utf-8 -*-
"""全项目唯一的「用户提示词」拼装函数。

⚠ 历史 bug（2026-10-27 发现）：
    训练脚本用的是   f"{instruction}\n{input}"
    评估脚本用的是   f"{instruction}\n\n文本：{input}"
两者不一致 —— 模型从来没见过 "\n\n文本：" 这种写法，却被用这种格式考试，
导致所有评估指标系统性偏低（65.8% / 72.2% 都被这个问题拖累）。

修复方式：训练和评估都 import 这一个函数，谁都不许自己拼字符串。
如果以后要改提示词格式，改这一处，然后把模型重训一遍即可。
"""


def build_user_text(row):
    """把 alpaca 格式的一条数据，拼成喂给模型的 user 内容。

    row: dict，至少含 instruction / input 两个键
    返回: str
    """
    ins = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", "")).strip()
    # 有正文就把正文换行接在指令后面；没有正文就只用指令
    return f"{ins}\n{inp}" if inp else ins
