# -*- coding: utf-8 -*-
"""路由回归测试：不加载模型，纯规则，秒级跑完

用法：
    python tests/test_routing.py

为什么单独有这个测试：关键词路由是最容易被随手改动的地方，
改一个词可能救回一条、也可能把另一条搞错。把它固定成 24 条样本，
每次改 ROUTE_RULES 之后跑一遍，通过率掉到 90% 以下就退出码 1。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import route  # noqa: E402

PASS_LINE = 0.90


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing_cases.json")
    cases = json.load(open(path, encoding="utf-8"))

    ok = 0
    bad = []
    for c in cases:
        got, conf, reason = route(c["q"])
        if got == c["expect"]:
            ok += 1
        else:
            bad.append((c["q"], c["expect"], got, reason))

    rate = ok / len(cases)
    print(f"路由回归：{ok}/{len(cases)} = {rate:.1%}")
    for q, exp, got, reason in bad:
        print(f"  期望={exp} 实际={got} 依据={reason}  ← {q}")

    if rate < PASS_LINE:
        print(f"低于 {PASS_LINE:.0%}，视为回归失败")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
