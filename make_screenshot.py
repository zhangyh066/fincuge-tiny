# -*- coding: utf-8 -*-
"""给 README 生成界面截图（Edge 无头模式）

做法：临时起一个服务 → 打开 ?demo=xxx（页面会自动填样例并真实调一次接口）→ 截图 → 关服务
不需要 playwright，用系统自带的 Edge 就行。
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import webui
from harness import Harness

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = os.path.join(HERE, "assets")


def main():
    os.makedirs(OUT, exist_ok=True)
    webui.H = Harness(verbose=False)

    srv = ThreadingHTTPServer(("127.0.0.1", 7860), webui.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(1)

    profile = tempfile.mkdtemp(prefix="edge-shot-")
    shots = [("emotion", "1100,700"), ("newscat", "1100,700"), ("entity", "1100,840")]
    for task, size in shots:
        out = os.path.join(OUT, f"webui-{task}.png")
        cmd = [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
               "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={profile}",
               f"--window-size={size}", "--virtual-time-budget=25000",
               f"--screenshot={out}",
               f"http://127.0.0.1:7860/?demo={task}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
            ok = os.path.exists(out) and os.path.getsize(out) > 5000
            print(f"  {task:8} 退出码 {r.returncode}  图片 {'OK' if ok else '失败'}  "
                  f"{os.path.getsize(out) if os.path.exists(out) else 0} 字节")
        except Exception as e:
            print(f"  {task:8} 截图异常: {e}")

    srv.shutdown()
    print("完成，图片在", OUT)


if __name__ == "__main__":
    main()
