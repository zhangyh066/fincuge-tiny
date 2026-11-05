# -*- coding: utf-8 -*-
import csv
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from openpyxl import Workbook, load_workbook

from batch_io import (count_source_rows, export_results, file_format,
                      iter_first_column, safe_cell)
from batch_runner import BatchManager, map_result


class ResultMappingTest(unittest.TestCase):
    def test_status_matches_single_row_ui(self):
        cases = [
            ({"status": "completed", "answer": "积极", "decision": "auto", "schema_valid": True}, ("积极", "有把握")),
            ({"status": "completed", "answer": "积极", "decision": "hint", "schema_valid": True}, ("积极", "建议核对")),
            ({"status": "completed", "answer": "我不确定", "decision": "abstain", "schema_valid": False}, ("我不确定", "拒答")),
            ({"status": "need_input", "answer": "请选择事件类型"}, ("请选择事件类型", "需要补充")),
            ({"status": "no_event", "answer": "无"}, ("无", "无适用事件")),
        ]
        for res, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(map_result(res), expected)

    def test_truncated_auto_is_downgraded(self):
        res = {"status": "completed", "answer": "政策", "decision": "auto",
               "schema_valid": True, "flags": ["input_truncated"]}
        self.assertEqual(map_result(res), ("政策", "建议核对"))


class FileIOTest(unittest.TestCase):
    def test_csv_streaming_blank_rows_and_partial_export(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "input.csv"
            output = Path(d) / "output.csv"
            source.write_bytes("第一条\n\n第二条\n".encode("utf-8"))
            rows = list(iter_first_column(source, "csv"))
            self.assertEqual(rows, [(0, "第一条"), (1, ""), (2, "第二条")])
            records = {
                0: {"result": "积极", "judgement": "有把握"},
                2: {"result": "消极", "judgement": "建议核对"},
            }
            export_results(source, "csv", False, records, output)
            with open(output, encoding="utf-8-sig", newline="") as f:
                result = list(csv.reader(f))
            self.assertEqual(result[0], ["数据内容", "结果", "判定"])
            self.assertEqual(result[2], ["", "未处理", "未处理"])
            self.assertEqual(result[3], ["第二条", "消极", "建议核对"])

    def test_gb18030_and_header(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "input.csv"
            source.write_bytes("内容\n利润下滑\n".encode("gb18030"))
            self.assertEqual(count_source_rows(source, "csv", True), 1)
            self.assertEqual(list(iter_first_column(source, "csv", True)), [(0, "利润下滑")])

    def test_xlsx_roundtrip_and_formula_protection(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "input.xlsx"
            output = Path(d) / "output.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["内容"])
            sheet.append(["=1+1"])
            sheet.append(["正常文本"])
            workbook.save(source)
            records = {
                0: {"result": "中性", "judgement": "有把握"},
                1: {"result": "积极", "judgement": "有把握"},
            }
            export_results(source, "xlsx", True, records, output)
            result = load_workbook(output, data_only=False).active
            self.assertEqual(result.max_row, 3)
            self.assertEqual(result.cell(2, 1).value, "'=1+1")
            self.assertEqual(result.cell(2, 1).data_type, "s")
            self.assertEqual(result.cell(3, 2).value, "积极")

    def test_extension_and_cell_safety(self):
        self.assertEqual(file_format("a/b.XLSX"), "xlsx")
        with self.assertRaises(ValueError):
            file_format("bad.exe")
        self.assertEqual(safe_cell("  @cmd"), "'  @cmd")
        self.assertEqual(safe_cell("正常"), "正常")


class FakeHarness:
    def __init__(self):
        self.calls = []
        self.manager = None
        self.cancel_on_first = False

    def answer(self, text, task_hint, event_type):
        self.calls.append(text)
        if self.cancel_on_first and len(self.calls) == 1:
            self.manager.cancel_job(self.manager.list_jobs()[0]["id"])
        if text == "错误":
            raise RuntimeError("模拟单行异常")
        if text == "核对":
            decision = "hint"
        else:
            decision = "auto"
        return {"status": "completed", "answer": "积极", "decision": decision,
                "schema_valid": True, "flags": [], "confidence": 0.96,
                "generation_count": 1, "latency_ms": 1}


class BatchManagerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.harness = FakeHarness()
        self.manager = BatchManager(
            self.harness, threading.Lock(), jobs_root=self.root / "jobs",
            results_root=self.root / "results", autostart=False)
        self.harness.manager = self.manager

    def tearDown(self):
        self.manager.shutdown()
        self.temp.cleanup()

    def create_csv(self, text="业绩增长\n核对\n", name="input.csv"):
        return self.manager.create_job(io.BytesIO(text.encode("utf-8")), name, "emotion")

    def test_completed_job_saves_and_exports(self):
        job = self.create_csv()
        done = self.manager.run_job(job["id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["processed"], 2)
        self.assertEqual(done["counts"], {"有把握": 1, "建议核对": 1})
        self.assertTrue(done["result_available"])
        path = self.manager.result_path(job["id"])
        self.assertEqual(path.parent, (self.root / "results").resolve())
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual([r[2] for r in rows[1:]], ["有把握", "建议核对"])

    def test_cancel_resume_and_truncated_checkpoint(self):
        self.harness.cancel_on_first = True
        job = self.create_csv("第一条\n第二条\n第三条\n")
        cancelled = self.manager.run_job(job["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["processed"], 1)
        self.assertEqual(self.harness.calls, ["第一条"])
        with open(Path(job["job_dir"]) / "results.jsonl", "a", encoding="utf-8") as f:
            f.write('{"row_index":')
        self.harness.cancel_on_first = False
        self.manager.resume_job(job["id"])
        done = self.manager.run_job(job["id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["processed"], 3)
        self.assertEqual(self.harness.calls, ["第一条", "第二条", "第三条"])
        with open(done["output_path"], encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 4)
        with open(Path(job["job_dir"]) / "results.jsonl", "rb") as f:
            for line in f:
                json.loads(line)

    def test_queued_cancellation_never_calls_model(self):
        job = self.create_csv("第一条\n第二条\n")
        self.manager.cancel_job(job["id"])
        cancelled = self.manager.run_job(job["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["processed"], 0)
        self.assertEqual(self.harness.calls, [])

    def test_row_error_does_not_stop_job(self):
        job = self.create_csv("第一条\n错误\n第三条\n")
        done = self.manager.run_job(job["id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["counts"]["处理失败"], 1)
        self.assertEqual(done["processed"], 3)

    def test_upload_limit_and_path_safety(self):
        small = BatchManager(FakeHarness(), threading.Lock(), jobs_root=self.root / "small-jobs",
                             results_root=self.root / "small-results", max_upload_bytes=4, autostart=False)
        with self.assertRaises(ValueError):
            small.create_job(io.BytesIO(b"too large"), "bad.csv", "emotion")
        job = self.manager.create_job(io.BytesIO("文本\n".encode()), "../../安全.csv", "emotion")
        self.assertEqual(Path(job["output_path"]).parent, self.root / "results")
        self.assertEqual(job["original_filename"], "安全.csv")

    def test_ten_thousand_rows_streaming(self):
        text = "".join(f"第{i}条\n" for i in range(10000))
        job = self.manager.create_job(io.BytesIO(text.encode("utf-8")), "large.csv", "emotion")
        done = self.manager.run_job(job["id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["processed"], 10000)
        with open(done["output_path"], encoding="utf-8-sig", newline="") as f:
            self.assertEqual(sum(1 for _ in csv.reader(f)), 10001)


if __name__ == "__main__":
    unittest.main()
