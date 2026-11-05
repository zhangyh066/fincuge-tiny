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
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from batch_runner import BatchManager
from harness import Harness

H = None
BATCH = None
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
  input[type=file]{display:none}
  .file-name{font-size:13px;color:var(--muted)}
  .progress{height:10px;background:#eef0f3;border-radius:99px;overflow:hidden;margin:12px 0 8px}
  .progress>span{display:block;height:100%;background:var(--brand);transition:width .2s}
  .job{border-top:1px solid var(--line);padding:16px 0}
  .job:first-child{border-top:0}
  .job-title{font-size:14px;font-weight:650;word-break:break-all}
  .job-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  .job-actions button,.job-actions a{border:1px solid var(--line);border-radius:7px;background:#fff;
       color:var(--text);padding:6px 10px;font-size:12px;cursor:pointer;text-decoration:none}
  .output-path{font-family:Consolas,monospace;font-size:12px;word-break:break-all;color:var(--muted);margin-top:8px}
  .counts{font-size:12px;color:var(--muted);line-height:1.8}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>FinCUGE-Tiny · 本地小模型</h1>
    <div class="sub">Qwen3-0.6B + LoRA，跑在你这台 4GB 显卡上。支持单条分析与 CSV/XLSX 批量任务。</div>
  </header>

  <div class="tabs">
    <button class="tab active" data-p="emotion">情绪分类</button>
    <button class="tab" data-p="newscat">新闻七分类</button>
    <button class="tab" data-p="entity">事件主体抽取</button>
    <button class="tab" data-p="batch">批量任务</button>
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
      <div class="row"><span class="hint">事件类型是模型的必需输入；没有适用类型时可选“无”，此时不执行抽取。</span></div>
      <div class="out" id="o-entity"></div>
    </div>
  </section>

  <section class="panel" id="p-batch">
    <div class="card">
      <div class="row">
        <input type="file" id="batch-file" accept=".csv,.xlsx">
        <button class="go" id="batch-select">选择 CSV/XLSX 文件</button>
        <span class="file-name" id="batch-file-name">尚未选择文件</span>
      </div>
      <div class="row">
        <label class="hint">批量任务</label>
        <select id="batch-task">
          <option value="emotion">情绪分类</option>
          <option value="newscat">新闻七分类</option>
          <option value="entity">事件主体抽取</option>
        </select>
        <label class="hint"><input type="checkbox" id="batch-header"> 首行是表头</label>
      </div>
      <div class="row" id="batch-event-row" style="display:none">
        <label class="hint">事件类型</label>
        <select id="batch-event"></select>
        <input type="text" id="batch-event-custom" placeholder="手填事件类型" style="display:none">
      </div>
      <div class="row"><span class="hint">只读取第一列；输出为“数据内容 / 结果 / 判定”。结果固定保存到项目目录下的 batch_results，完成后显示完整路径。</span></div>
      <div class="row"><button class="go" id="batch-upload">上传并开始</button></div>
      <div class="err" id="batch-error"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <b>任务列表</b>
      <div class="hint" style="margin-top:5px">关闭页面不会停止后台任务；取消会在当前行推理结束后生效。</div>
      <div id="batch-jobs"></div>
    </div>
  </section>

  <footer>置信度低会标 <b>存疑</b>，不合规会直接 <b>拒答</b> —— 这是设计好的，不是出错。</footer>
</div>

<script>
const EVENTS = __EVENTS__;
const sel = document.getElementById('ev');
const batchSel = document.getElementById('batch-event');
function fillEvents(select){
  EVENTS.forEach(e => { const o=document.createElement('option'); o.value=e; o.textContent=e; select.appendChild(o); });
  const noneOption=document.createElement('option'); noneOption.value='无'; noneOption.textContent='无（不适用/不确定）'; select.appendChild(noneOption);
  const o=document.createElement('option'); o.value='__custom__'; o.textContent='其它（手填）'; select.appendChild(o);
}
fillEvents(sel);
fillEvents(batchSel);
sel.addEventListener('change', () => {
  document.getElementById('ev-custom').style.display = sel.value==='__custom__' ? 'inline-block' : 'none';
});
batchSel.addEventListener('change', () => {
  document.getElementById('batch-event-custom').style.display = batchSel.value==='__custom__' ? 'inline-block' : 'none';
});
document.getElementById('batch-task').addEventListener('change', e => {
  document.getElementById('batch-event-row').style.display = e.target.value==='entity' ? 'flex' : 'none';
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
  const needsInput = d.status === 'need_input' || d.status === 'need_task';
  const noEvent = d.status === 'no_event';
  const cls = needsInput || noEvent || d.decision === 'hint' ? 'b-hint' : (d.decision === 'auto' ? 'b-auto' : 'b-abstain');
  const label = noEvent ? '无适用事件' : (needsInput ? '需要补充' : ({auto:'有把握', hint:'建议核对', abstain:'拒答'}[d.decision] || d.decision));
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

let batchFile = null;
document.getElementById('batch-select').addEventListener('click', () => document.getElementById('batch-file').click());
document.getElementById('batch-file').addEventListener('change', e => {
  batchFile = e.target.files[0] || null;
  document.getElementById('batch-file-name').textContent = batchFile ? `${batchFile.name} · ${(batchFile.size/1024/1024).toFixed(2)} MB` : '尚未选择文件';
});

document.getElementById('batch-upload').addEventListener('click', async () => {
  const err = document.getElementById('batch-error');
  err.textContent = '';
  if (!batchFile){ err.textContent = '请先选择 CSV 或 XLSX 文件'; return; }
  const task = document.getElementById('batch-task').value;
  let eventType = '';
  if (task === 'entity'){
    eventType = batchSel.value;
    if (eventType === '__custom__') eventType = document.getElementById('batch-event-custom').value.trim();
    if (!eventType){ err.textContent = '实体任务必须选择事件类型；不适用时请选择“无”'; return; }
  }
  const query = new URLSearchParams({
    task, event_type: eventType, filename: batchFile.name,
    first_row_header: document.getElementById('batch-header').checked ? '1' : '0'
  });
  const btn = document.getElementById('batch-upload');
  btn.disabled = true;
  try {
    const r = await fetch('/api/batch/jobs?' + query.toString(), {
      method:'POST', body:batchFile, headers:{'Content-Type':'application/octet-stream'}
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || '上传失败');
    await loadJobs();
  }catch(e){ err.textContent = e.message; }
  finally{ btn.disabled = false; }
});

function formatEta(seconds){
  if (seconds === null || seconds === undefined) return '估算中';
  if (seconds >= 3600) return `${Math.floor(seconds/3600)}小时${Math.round(seconds%3600/60)}分钟`;
  if (seconds >= 60) return `${Math.ceil(seconds/60)}分钟`;
  return `${seconds}秒`;
}

async function loadJobs(){
  const host = document.getElementById('batch-jobs');
  try {
    const r = await fetch('/api/batch/jobs');
    const jobs = await r.json();
    if (!r.ok) throw new Error(jobs.error || '任务读取失败');
    const names = {emotion:'情绪分类', newscat:'新闻七分类', entity:'主体抽取'};
    host.innerHTML = jobs.length ? jobs.map(j => {
      const total = j.total || 0, done = j.processed || 0;
      const percent = total ? Math.min(100, done/total*100) : 0;
      const counts = j.counts || {};
      let actions = '';
      if (['running','scanning','queued'].includes(j.status)) actions += `<button data-action="cancel" data-id="${j.id}">取消</button>`;
      if (['cancelled','interrupted','failed'].includes(j.status)) actions += `<button data-action="resume" data-id="${j.id}">续跑</button>`;
      if (j.result_available){
        actions += `<a href="/api/batch/jobs/${j.id}/download">下载结果</a>`;
        actions += `<button data-action="copy" data-id="${j.id}">复制保存路径</button>`;
        actions += `<button data-action="reveal" data-id="${j.id}">打开所在文件夹</button>`;
      }
      return `<div class="job">
        <div class="job-title">${esc(j.original_filename)} · ${esc(names[j.task] || j.task)}</div>
        <div class="progress"><span style="width:${percent.toFixed(2)}%"></span></div>
        <div class="counts">状态 ${esc(j.status)} · ${done}/${total} 条（${percent.toFixed(1)}%） · 速度 ${j.rows_per_second || 0} 条/秒 · 预计剩余 ${formatEta(j.eta_seconds)}<br>
        有把握 ${counts['有把握'] || 0} · 建议核对 ${counts['建议核对'] || 0} · 拒答 ${counts['拒答'] || 0} · 处理失败 ${counts['处理失败'] || 0}</div>
        ${j.error ? `<div class="err">${esc(j.error)}</div>` : ''}
        <div class="output-path">保存位置：${esc(j.output_path || '尚未生成')}</div>
        <div class="job-actions">${actions}</div>
      </div>`;
    }).join('') : '<div class="hint" style="margin-top:12px">还没有批量任务</div>';
  }catch(e){ host.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

async function batchAction(id, action){
  if (action === 'copy'){
    const r = await fetch('/api/batch/jobs/' + id);
    const j = await r.json();
    await navigator.clipboard.writeText(j.output_path);
    return;
  }
  const r = await fetch(`/api/batch/jobs/${id}/${action}`, {method:'POST'});
  const d = await r.json();
  if (!r.ok) alert(d.error || '操作失败');
  await loadJobs();
}

document.getElementById('batch-jobs').addEventListener('click', e => {
  const btn = e.target.closest('button[data-action]');
  if (btn) batchAction(btn.dataset.id, btn.dataset.action).catch(e => alert(e.message));
});
loadJobs();
setInterval(loadJobs, 2000);

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

    def _send_file(self, path):
        path = Path(path)
        ctype = "text/csv; charset=utf-8" if path.suffix == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name)}")
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = PAGE.replace("__EVENTS__", json.dumps(EVENT_TYPES, ensure_ascii=False))
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self._send(200, json.dumps({"ok": H is not None, "batch": BATCH is not None}))
            return
        if path == "/api/batch/jobs":
            self._send(200, json.dumps(BATCH.list_jobs(), ensure_ascii=False))
            return
        if path == "/api/batch/config":
            self._send(200, json.dumps({"results_root": str(BATCH.results_root),
                                        "max_upload_bytes": BATCH.max_upload_bytes}, ensure_ascii=False))
            return
        match = re.fullmatch(r"/api/batch/jobs/([0-9a-f]{32})", path)
        if match:
            self._send(200, json.dumps(BATCH.get_job(match.group(1)), ensure_ascii=False))
            return
        match = re.fullmatch(r"/api/batch/jobs/([0-9a-f]{32})/download", path)
        if match:
            try:
                self._send_file(BATCH.result_path(match.group(1)))
            except (KeyError, FileNotFoundError, ValueError) as e:
                self._send(404, json.dumps({"error": str(e)}, ensure_ascii=False))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            if path == "/api/batch/jobs":
                params = parse_qs(urlparse(self.path).query)
                size = int(self.headers.get("Content-Length", 0))
                if size <= 0:
                    raise ValueError("上传内容为空")
                if size > BATCH.max_upload_bytes:
                    self.close_connection = True
                    raise ValueError("文件超过上传大小上限")
                job = BATCH.create_job(
                    self.rfile,
                    params.get("filename", [""])[0],
                    params.get("task", [""])[0],
                    params.get("event_type", [None])[0],
                    params.get("first_row_header", ["0"])[0] == "1",
                    content_length=size,
                )
                self._send(201, json.dumps(job, ensure_ascii=False))
                return
            match = re.fullmatch(r"/api/batch/jobs/([0-9a-f]{32})/(cancel|resume|reveal)", path)
            if match:
                job_id, action = match.groups()
                if action == "cancel":
                    result = BATCH.cancel_job(job_id)
                elif action == "resume":
                    result = BATCH.resume_job(job_id)
                else:
                    result = {"output_path": BATCH.reveal_result(job_id)}
                self._send(200, json.dumps(result, ensure_ascii=False))
                return

            task = path.split("/")[-1]
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            text = (payload.get("text") or "").strip()
            if not text:
                self._send(400, json.dumps({"error": "文本为空"}, ensure_ascii=False))
                return
            if task not in ("emotion", "newscat", "entity"):
                self._send(400, json.dumps({"error": "未知任务"}, ensure_ascii=False))
                return
            event_type = (payload.get("event") or "").strip() if task == "entity" else None
            t0 = time.time()
            with LOCK:
                res = H.answer(text, task_hint=task, event_type=event_type)
            res["wall_ms"] = int((time.time() - t0) * 1000)
            self._send(200, json.dumps(res, ensure_ascii=False))
        except (ValueError, KeyError, FileNotFoundError) as e:
            self._send(400, json.dumps({"error": str(e)}, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))


def main():
    global H, BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print("[1/2] 加载模型（约 15 秒）…")
    H = Harness()
    BATCH = BatchManager(H, LOCK)
    print(f"[2/2] 服务已启动：http://{args.host}:{args.port}")
    print(f"      批量结果保存：{BATCH.results_root}")
    print("      按 Ctrl+C 停止")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        BATCH.shutdown()
        srv.shutdown()


if __name__ == "__main__":
    main()
