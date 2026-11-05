# -*- coding: utf-8 -*-
"""单 GPU 批量任务：流式处理、逐行落盘、取消、续跑与结果保存。"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from batch_io import (build_output_path, count_source_rows, export_results,
                      file_format, iter_first_column)

TASKS = {"emotion", "newscat", "entity"}
ACTIVE_STATES = {"queued", "scanning", "running", "exporting"}


def now_text():
    return datetime.now().isoformat(timespec="seconds")


def map_result(res):
    status = res.get("status", "completed")
    answer = str(res.get("answer", ""))
    flags = set(res.get("flags") or [])
    if status == "no_event":
        return "无", "无适用事件"
    if status in {"need_input", "need_task"}:
        return answer, "需要补充"
    if status == "completed":
        if res.get("decision") == "auto" and res.get("schema_valid") and "input_truncated" not in flags:
            return answer, "有把握"
        if res.get("decision") == "hint" or "input_truncated" in flags:
            return answer, "建议核对"
        return answer or "我不确定", "拒答"
    return answer or "无法判断", "拒答"


class JobCancelled(Exception):
    pass


class BatchManager:
    def __init__(self, harness, gpu_lock, jobs_root=None, results_root=None,
                 max_upload_bytes=200 * 1024 * 1024, autostart=True):
        here = Path(__file__).resolve().parent
        self.harness = harness
        self.gpu_lock = gpu_lock
        self.jobs_root = Path(jobs_root) if jobs_root else here / "data" / "batch_jobs"
        self.results_root = Path(results_root) if results_root else here / "batch_results"
        self.max_upload_bytes = max_upload_bytes
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition()
        self._jobs = {}
        self._cancel_events = {}
        self._shutdown = threading.Event()
        self._worker = None
        self._load_jobs()
        if autostart:
            self._worker = threading.Thread(target=self._worker_loop, name="batch-worker", daemon=True)
            self._worker.start()

    def _load_jobs(self):
        for job_path in self.jobs_root.glob("*/job.json"):
            try:
                job = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") in ACTIVE_STATES:
                job["status"] = "interrupted"
                job["error"] = "服务重启，任务已暂停，请手动续跑"
                self._save_job(job)
            self._jobs[job["id"]] = job
            self._cancel_events[job["id"]] = threading.Event()

    @staticmethod
    def _save_job(job):
        path = Path(job["job_dir"]) / "job.json"
        temp = path.with_name("job.json.tmp")
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def _commit(self, job):
        with self._condition:
            self._jobs[job["id"]] = job
            self._save_job(job)

    def _wake(self):
        with self._condition:
            self._condition.notify_all()

    def create_job(self, upload_stream, original_filename, task, event_type=None,
                   first_row_header=False, content_length=None):
        source_format = file_format(original_filename)
        if task not in TASKS:
            raise ValueError("未知任务")
        event_type = str(event_type or "").strip()
        if task == "entity" and not event_type:
            raise ValueError("实体任务必须选择事件类型；不适用时请选择“无”")
        if task != "entity":
            event_type = None

        job_id = uuid.uuid4().hex
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True)
        self._cancel_events[job_id] = threading.Event()
        source_path = job_dir / f"source.{source_format}"
        digest = hashlib.sha256()
        size = 0
        remaining = content_length
        try:
            with open(source_path, "wb") as f:
                while remaining is None or remaining > 0:
                    read_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
                    chunk = upload_stream.read(read_size)
                    if not chunk:
                        break
                    if remaining is not None:
                        remaining -= len(chunk)
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise ValueError("文件超过上传大小上限")
                    digest.update(chunk)
                    f.write(chunk)
            if size == 0:
                raise ValueError("文件为空")
            if content_length is not None and size != content_length:
                raise ValueError("上传内容不完整")
            output_path = build_output_path(
                self.results_root, original_filename, task, job_id, source_format)
            job = {
                "id": job_id,
                "job_dir": str(job_dir),
                "source_path": str(source_path),
                "source_sha256": digest.hexdigest(),
                "original_filename": str(original_filename).replace("\\", "/").split("/")[-1],
                "format": source_format,
                "task": task,
                "event_type": event_type,
                "first_row_header": bool(first_row_header),
                "status": "queued",
                "cancel_requested": False,
                "created_at": now_text(),
                "started_at": None,
                "finished_at": None,
                "total": 0,
                "processed": 0,
                "counts": {},
                "rows_per_second": 0.0,
                "eta_seconds": None,
                "output_path": output_path,
                "error": None,
            }
            self._commit(job)
            self._wake()
            return self.get_job(job_id)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def _with_availability(self, job):
        job = copy.deepcopy(job)
        job["result_available"] = bool(job.get("output_path") and os.path.exists(job["output_path"]))
        return job

    def get_job(self, job_id):
        with self._condition:
            if job_id not in self._jobs:
                raise KeyError("任务不存在")
            return self._with_availability(self._jobs[job_id])

    def list_jobs(self):
        with self._condition:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: (j.get("created_at", ""), j["id"]), reverse=True)
        return [self._with_availability(j) for j in jobs]

    def cancel_job(self, job_id):
        job = self.get_job(job_id)
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = now_text()
            self._cancel_events[job_id].set()
            self._commit(job)
        elif job["status"] in {"running", "scanning"}:
            job["cancel_requested"] = True
            self._commit(job)
            self._cancel_events[job_id].set()
        return self.get_job(job_id)

    def resume_job(self, job_id):
        job = self.get_job(job_id)
        if job["status"] not in {"cancelled", "interrupted", "failed"}:
            raise ValueError("只有已取消、中断或失败的任务可以续跑")
        self._cancel_events[job_id].clear()
        job["status"] = "queued"
        job["cancel_requested"] = False
        job["finished_at"] = None
        job["error"] = None
        self._commit(job)
        self._wake()
        return self.get_job(job_id)

    def result_path(self, job_id):
        job = self.get_job(job_id)
        if not job["result_available"]:
            raise FileNotFoundError("结果文件尚未生成")
        path = Path(job["output_path"]).resolve()
        if path.parent != self.results_root.resolve():
            raise ValueError("非法结果路径")
        return path

    def reveal_result(self, job_id):
        path = self.result_path(job_id)
        if os.name != "nt":
            raise ValueError("只有本机 Windows 支持打开所在文件夹")
        subprocess.Popen(["explorer.exe", f"/select,{str(path)}"], close_fds=True)
        return str(path)

    def _worker_loop(self):
        while not self._shutdown.is_set():
            with self._condition:
                queued = [j for j in self._jobs.values() if j.get("status") == "queued"]
                queued.sort(key=lambda j: (j.get("created_at", ""), j["id"]))
                if not queued:
                    self._condition.wait(timeout=1)
                    continue
                job_id = queued[0]["id"]
            try:
                self.run_job(job_id)
            except Exception:
                continue

    def shutdown(self):
        self._shutdown.set()
        self._wake()

    @staticmethod
    def _append_jsonl(path, record):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def _load_records(self, job):
        path = Path(job["job_dir"]) / "results.jsonl"
        records = {}
        if not path.exists():
            return records

        def consume(line, allow_truncated, line_offset):
            if not line.strip():
                return
            try:
                rec = json.loads(line)
                records[int(rec["row_index"])] = rec
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as e:
                if not allow_truncated:
                    raise ValueError("结果断点中间存在损坏记录") from e
                with open(path, "r+b") as checkpoint:
                    checkpoint.seek(line_offset)
                    checkpoint.truncate()
                self._append_jsonl(Path(job["job_dir"]) / "errors.jsonl", {
                    "row_index": None, "error": "truncated_checkpoint_tail", "detail": str(e),
                    "ts": now_text(),
                })

        with open(path, "rb") as f:
            previous = None
            previous_offset = 0
            offset = 0
            for line in f:
                current_offset = offset
                offset += len(line)
                if previous is not None:
                    consume(previous, False, previous_offset)
                previous = line
                previous_offset = current_offset
            if previous is not None:
                consume(previous, True, previous_offset)
        return records

    def _set_progress(self, job, records, session_started, session_start_processed):
        job["processed"] = len(records)
        job["counts"] = dict(Counter(r["judgement"] for r in records.values()))
        elapsed = max(0.001, time.time() - session_started)
        done_now = len(records) - session_start_processed
        job["rows_per_second"] = round(done_now / elapsed, 4) if done_now > 0 else 0.0
        remaining = max(0, job["total"] - job["processed"])
        job["eta_seconds"] = int(remaining / job["rows_per_second"]) if job["rows_per_second"] > 0 else None
        self._commit(job)

    def _export(self, job, records):
        if not records:
            return
        job["status"] = "exporting"
        self._commit(job)
        export_results(job["source_path"], job["format"], job["first_row_header"],
                       records, job["output_path"])

    def run_job(self, job_id):
        job = self.get_job(job_id)
        if job["status"] not in {"queued", "interrupted", "cancelled", "failed"}:
            raise ValueError("任务当前不可执行")
        records = {}
        session_started = time.time()
        session_start_processed = 0
        try:
            if self._cancel_events[job_id].is_set():
                raise JobCancelled()
            job["status"] = "scanning"
            job["error"] = None
            self._commit(job)
            with open(job["source_path"], "rb") as source:
                source_hash = hashlib.file_digest(source, "sha256").hexdigest()
            if source_hash != job["source_sha256"]:
                raise ValueError("上传文件已变化，不能续跑")
            job["total"] = count_source_rows(
                job["source_path"], job["format"], job["first_row_header"])
            records = self._load_records(job)
            session_start_processed = len(records)
            self._set_progress(job, records, session_started, session_start_processed)
            job["status"] = "running"
            job["started_at"] = job.get("started_at") or now_text()
            self._commit(job)
            results_path = Path(job["job_dir"]) / "results.jsonl"
            consecutive_errors = 0

            for row_index, text in iter_first_column(
                    job["source_path"], job["format"], job["first_row_header"]):
                if row_index in records:
                    continue
                if self._cancel_events[job_id].is_set():
                    raise JobCancelled()
                try:
                    with self.gpu_lock:
                        res = self.harness.answer(
                            text, task_hint=job["task"], event_type=job["event_type"])
                    result, judgement = map_result(res)
                    consecutive_errors = 0
                    rec = {
                        "row_index": row_index,
                        "result": result,
                        "judgement": judgement,
                        "status": res.get("status"),
                        "decision": res.get("decision"),
                        "confidence": res.get("confidence"),
                        "flags": res.get("flags") or [],
                        "latency_ms": res.get("latency_ms"),
                        "generation_count": res.get("generation_count"),
                    }
                except Exception as e:
                    consecutive_errors += 1
                    rec = {
                        "row_index": row_index,
                        "result": "处理失败",
                        "judgement": "处理失败",
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                    }
                    self._append_jsonl(Path(job["job_dir"]) / "errors.jsonl", {
                        "row_index": row_index, "error": rec["error"], "ts": now_text(),
                    })
                records[row_index] = rec
                self._append_jsonl(results_path, rec)
                if len(records) % 10 == 0:
                    self._set_progress(job, records, session_started, session_start_processed)
                if consecutive_errors >= 20:
                    raise RuntimeError("连续 20 行处理失败，任务已停止")
                time.sleep(0)

            self._export(job, records)
            job["status"] = "completed"
            job["finished_at"] = now_text()
            self._set_progress(job, records, session_started, session_start_processed)
        except JobCancelled:
            self._export(job, records)
            job["status"] = "cancelled"
            job["finished_at"] = now_text()
            self._set_progress(job, records, session_started, session_start_processed)
        except Exception as e:
            try:
                self._export(job, records)
            except Exception as export_error:
                job["output_path"] = None
                job["error"] = f"导出失败：{type(export_error).__name__}: {export_error}"
            job["status"] = "failed"
            if job.get("error") is None:
                job["error"] = f"{type(e).__name__}: {e}"
            job["finished_at"] = now_text()
            self._set_progress(job, records, session_started, session_start_processed)
        return self.get_job(job_id)
