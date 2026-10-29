# -*- coding: utf-8 -*-
"""本地网页版 agent：三个标签页分别对应情绪 / 新闻分类 / 主体抽取

为什么用标准库：不引入 gradio/streamlit 这类大依赖，装起来快、坏起来少。
模型只加载一次，页面通过 fetch 调本地接口。

启动：
    python webui.py            # 默认 127.0.0.1:7860
    python webui.py --port 8000

接口（POST JSON）：
    /api/emotion  {"text": "..."}
    /api/newscat  {"text": "..."}
    /api/entity   {"text": "...", "event": "涉嫌非法集资"}
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness import Harness

H = None
LOCK = threading.Lock()

# 实体任务的事件类型：实体的训练指令里带事件类型，不给就是分布外输入。
# 这些是从训练数据里挑的高频类型，页面下拉选，不用手打。
EVENT_TYPES = [
    "信批违规", "资金账户风险", "高管负面", "涉嫌非法集资", "涉嫌欺诈",
    "实控人股东变更", "业绩下滑", "交易违规", "涉嫌传销", "非法吸收公众存款",
    "不能履职", "数据泄露",
]

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinCUGE-Tiny · 本地小模型</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --text:#1f2328; --muted:#6b7280;
    --brand:#c2410c; --brand-soft:#fdece3;
    --ok:#0f766e; --ok-bg:#e6f6f4; --warn:#92400e; --warn-bg:#fdf1dc; --bad:#b42318; --bad-bg:#fdecea;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;}
  header{padding:22px 28px 0}
  h1{margin:0 0 6px;font-size:19px;font-weight:650}
  .sub{color:var(--muted);font-size:13px;margin-bottom:18px}
  .wrap{max-width:980px;margin:0 auto;padding:0 20px 60px}
  .tabs{display:flex;gap:6px;border-bottom:1px solid var(--line);margin-bottom:18px}
  .tab{padding:10px 16px;border:none;background:none;cursor:pointer;font-size:14px;
       color:var(--muted);border-bottom:2px solid transparent;font-family:inherit}
  .tab.active{color:var(--brand);border-bottom-color:var(--brand);font-weight:600}
  .panel{display:none}
  .panel.active{display:block}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
  textarea{width:100%;min-height:110px;resize:vertical;padding:11px 12px;border:1px solid var(--line);
           border-radius:8px;font-size:14px;font-family:inherit;line-height:1.6;outline:none}
  textarea:focus{border-color:#c9ced6;box-shadow:0 0 0 3px #eef1f5}
  .row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
  select,input[type=text]{padding:8px 10px;border:1px solid var(--line);border-radius:8px;
       font-size:13px;font-family:inherit;background:#fff;outline:none}
  select:focus,input[type=text]:focus{border-color:#c9ced6}
  button.go{padding:9px 20px;border:none;border-radius:8px;background:var(--brand);color:#fff;
       font-size:14px;cursor:pointer;font-family:inherit}
  button.go:disabled{opacity:.55;cursor:default}
  .hint{color:var(--muted);font-size:12px}
  .out{margin-top:16px;display:none}
  .out.show{display:block}
  .badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
  .b-auto{color:var(--ok);background:var(--ok-bg)}
  .b-hint{color:var(--warn);background:var(--warn-bg)}
  .b-abstain{color:var(--bad);background:var(--bad-bg)}
  .ans{font-size:26px;font-weight:650;margin:10px 0 4px;letter-spacing:.3px}
  .meta{color:var(--muted);font-size:12px;margin-bottom:12px}
  .bars{display:flex;flex-direction:column;gap:7px;margin-top:10px}
  .bar{display:flex;align-items:center;gap:9px;font-size:12px}
  .bar .name{width:34px;color:var(--muted)}
  .bar .track{flex:1;display:block;height:8px;background:#eef0f3;border-radius:99px;overflow:hidden}
  .bar .fill{display:block;height:100%;background:var(--brand);border-radius:99px}
  .bar .val{width:48px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
  .flags{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}
  .flag{font-size:11px;color:var(--muted);background:#f1f3f5;border-radius:6px;padding:3px 7px}
  .err{margin-top:14px;color:var(--bad);font-size:13px}
  footer{color:var(--muted);font-size:12px;margin-top:26px;text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>FinCUGE-Tiny · 本地小模型</h1>
    <div class="sub">Qwen3-0.6B + LoRA，跑在你这台 4GB 显卡上。三个标签页各管一件事，不用手输类型。</div>
  </header>

  <div class="tabs">
    <button class="tab active" data-p="emotion">情绪分类</button>
    <button class="tab" data-p="newscat">新闻七分类</button>
    <button class="tab" data-p="entity">事件主体抽取</button>
  </div>

  <section class="panel active" id="p-emotion">
    <div class="card">
      <textarea id="t-emotion" placeholder="粘贴一段话，比如：公司发布业绩预告，预计净利润同比增长120%，远超市场预期。"></textarea>
      <div class="row">
        <button class="go" data-run="emotion">判断情绪</button>
        <span class="hint">输出：积极 / 消极 / 中性，附三类概率</span>
      </div>
      <div class="out" id="o-emotion"></div>
    </div>
  </section>

  <section class="panel" id="p-newscat">
    <div class="card">
      <textarea id="t-newscat" placeholder="粘贴一条新闻，比如：证监会发布新规，进一步规范上市公司信息披露行为。"></textarea>
      <div class="row">
        <button class="go" data-run="newscat">判断类别</button>
        <span class="hint">输出：公司 / 新冠 / 经济 / 行业 / 大盘 / 政策 / 其它</span>
      </div>
      <div class="out" id="o-newscat"></div>
    </div>
  </section>

  <section class="panel" id="p-entity">
    <div class="card">
      <textarea id="t-entity" placeholder="粘贴一条新闻，比如：某某平台因涉嫌非法集资被警方立案侦查，涉案金额超10亿元。"></textarea>
      <div class="row">
        <label class="hint">事件类型</label>
        <select id="ev"></select>
        <input type="text" id="ev-custom" placeholder="手填事件类型" style="display:none;width:190px">
        <button class="go" data-run="entity">抽取主体</button>
      </div>
      <div class="row"><span class="hint">事件类型是模型的必需输入（训练数据里每条指令都带），选一个即可，不用打全称。</span></div>
      <div class="out" id="o-entity"></div>
    </div>
  </section>

  <footer>置信度低会标 <b>存疑</b>，不合规会直接 <b>拒答</b> —— 这是设计好的，不是出错。</footer>
</div>

<script>
const EVENTS = __EVENTS__;
const sel = document.getElementById('ev');
function fillEvents(){
  EVENTS.forEach(e => { const o=document.createElement('option'); o.value=e; o.textContent=e; sel.appendChild(o); });
  const o=document.createElement('option'); o.value='__custom__'; o.textContent='其它（手填）'; sel.appendChild(o);
}
fillEvents();
sel.addEventListener('change', () => {
  document.getElementById('ev-custom').style.display = sel.value==='__custom__' ? 'inline-block' : 'none';
});

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('p-' + btn.dataset.p).classList.add('active');
  });
});

function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function render(host, d){
  const cls = d.decision === 'auto' ? 'b-auto' : (d.decision === 'hint' ? 'b-hint' : 'b-abstain');
  const label = {auto:'有把握', hint:'建议核对', abstain:'拒答'}[d.decision] || d.decision;
  let html = `<span class="badge ${cls}">${label}</span>`;
  html += `<div class="ans">${esc(d.answer)}</div>`;
  html += `<div class="meta">置信 ${(d.confidence*100).toFixed(0)}% · ${d.latency_ms} ms · 任务 ${d.task}</div>`;
  if (d.probs){
    html += '<div class="bars">';
    for (const k of ['积极','消极','中性']){
      const v = d.probs[k] || 0;
      html += `<div class="bar"><span class="name">${k}</span>
        <span class="track"><span class="fill" style="width:${(v*100).toFixed(1)}%"></span></span>
        <span class="val">${(v*100).toFixed(1)}%</span></div>`;
    }
    html += '</div>';
  }
  if (d.flags && d.flags.length){
    html += '<div class="flags">' + d.flags.map(f => `<span class="flag">${esc(f)}</span>`).join('') + '</div>';
  }
  host.innerHTML = html; host.classList.add('show');
}

async function runTask(task, btn) {
    const ta = document.getElementById('t-' + task);
    const out = document.getElementById('o-' + task);
    const text = ta.value.trim();
    if (!text){ out.innerHTML = '<div class="err">先写点内容</div>'; out.classList.add('show'); return; }
    let payload = {text};
    if (task === 'entity'){
      let ev = sel.value;
      if (ev === '__custom__') ev = document.getElementById('ev-custom').value.trim();
      payload.event = ev;
    }
    btn.disabled = true; btn.textContent = '推理中…';
    try{
      const r = await fetch('/api/' + task, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || '服务返回错误');
      render(out, d);
    }catch(e){
      out.innerHTML = '<div class="err">出错：' + esc(e.message) + '</div>'; out.classList.add('show');
    }finally{
      btn.disabled = false; btn.textContent = task === 'emotion' ? '判断情绪' : (task === 'newscat' ? '判断类别' : '抽取主体');
    }
}

document.querySelectorAll('button[data-run]').forEach(btn => {
  btn.addEventListener('click', () => runTask(btn.dataset.run, btn));
});

// 截图/演示模式：?demo=emotion|newscat|entity，自动填样例并真实调一次接口
const DEMO_TEXT = {
  emotion: '公司发布业绩预告，预计净利润同比增长120%，远超市场预期。',
  newscat: '证监会发布新规，进一步规范上市公司信息披露行为。',
  entity: '某某平台因涉嫌非法集资被警方立案侦查，涉案金额超10亿元。'
};
const demo = new URLSearchParams(location.search).get('demo');
if (demo && DEMO_TEXT[demo]) {
  document.querySelector('.tab[data-p="' + demo + '"]').click();
  document.getElementById('t-' + demo).value = DEMO_TEXT[demo];
  if (demo === 'entity') sel.value = '涉嫌非法集资';
  runTask(demo, document.querySelector('button[data-run="' + demo + '"]'));
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 不刷屏

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]      # 去掉 ?demo=xxx 这类查询串
        if path in ("/", "/index.html"):
            page = PAGE.replace("__EVENTS__", json.dumps(EVENT_TYPES, ensure_ascii=False))
            self._send(200, page, "text/html; charset=utf-8")
        elif path == "/api/health":
            self._send(200, json.dumps({"ok": H is not None}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        task = path.split("/")[-1]
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            text = (payload.get("text") or "").strip()
            if not text:
                self._send(400, json.dumps({"error": "文本为空"}, ensure_ascii=False))
                return
            if task not in ("emotion", "newscat", "entity"):
                self._send(400, json.dumps({"error": "未知任务"}, ensure_ascii=False))
                return

            instr = None
            if task == "entity":
                ev = (payload.get("event") or "").strip()
                if ev:
                    instr = f"分析以下新闻，抽取{ev}事件相关的主体信息。"

            t0 = time.time()
            with LOCK:
                res = H.answer(text, task_hint=task, instruction=instr)
            res["wall_ms"] = int((time.time() - t0) * 1000)
            self._send(200, json.dumps(res, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))


def main():
    global H
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print("[1/2] 加载模型（约 15 秒）…")
    H = Harness()
    print(f"[2/2] 服务已启动：http://{args.host}:{args.port}")
    print("      按 Ctrl+C 停止")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        srv.shutdown()


if __name__ == "__main__":
    main()
