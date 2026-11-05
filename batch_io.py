# -*- coding: utf-8 -*-
"""批量任务的流式 CSV/XLSX 输入与三列结果导出。"""
import csv
import os
import re
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook, Workbook
from openpyxl.cell import WriteOnlyCell

OUTPUT_HEADERS = ("数据内容", "结果", "判定")
SUPPORTED_FORMATS = {"csv", "xlsx"}


def file_format(filename):
    suffix = Path(str(filename).replace("\\", "/")).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError("只支持 .csv 和 .xlsx 文件")
    return suffix


def safe_cell(value):
    text = "" if value is None else str(value)
    if text.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _csv_encoding(path):
    import codecs
    with open(path, "rb") as f:
        sample = f.read(64 * 1024)
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    try:
        decoder.decode(sample, final=False)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "gb18030"


def iter_first_column(path, source_format, first_row_header=False):
    """逐行返回 (从 0 开始的数据行号, 第一列文本)，不保留整张表。"""
    if source_format == "csv":
        with open(path, "r", encoding=_csv_encoding(path), newline="") as f:
            reader = csv.reader(f)
            if first_row_header:
                next(reader, None)
            for row_index, row in enumerate(reader):
                yield row_index, "" if not row or row[0] is None else str(row[0])
        return
    if source_format != "xlsx":
        raise ValueError(f"不支持的格式：{source_format}")

    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(min_col=1, max_col=1, values_only=True)
        if first_row_header:
            next(rows, None)
        for row_index, row in enumerate(rows):
            yield row_index, "" if not row or row[0] is None else str(row[0])
    finally:
        workbook.close()


def count_source_rows(path, source_format, first_row_header=False):
    return sum(1 for _ in iter_first_column(path, source_format, first_row_header))


def build_output_path(results_root, original_filename, task, job_id, source_format, now=None):
    original = str(original_filename).replace("\\", "/").split("/")[-1]
    stem = Path(original).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .") or "批量结果"
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    filename = f"{stem[:60]}_{task}_{stamp}_{job_id[:8]}.{source_format}"
    return str(Path(results_root) / filename)


def export_results(source_path, source_format, first_row_header, records, output_path):
    """按原行序导出；未处理行明确标记，部分结果也可安全生成。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        if source_format == "csv":
            with open(temp_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(OUTPUT_HEADERS)
                for row_index, text in iter_first_column(source_path, source_format, first_row_header):
                    rec = records.get(row_index)
                    result = rec["result"] if rec else "未处理"
                    judgement = rec["judgement"] if rec else "未处理"
                    writer.writerow([safe_cell(text), safe_cell(result), safe_cell(judgement)])
        elif source_format == "xlsx":
            workbook = Workbook(write_only=True)
            sheet = workbook.create_sheet("批量结果")
            sheet.append(OUTPUT_HEADERS)
            for row_index, text in iter_first_column(source_path, source_format, first_row_header):
                rec = records.get(row_index)
                values = [text, rec["result"] if rec else "未处理", rec["judgement"] if rec else "未处理"]
                cells = []
                for value in values:
                    cell = WriteOnlyCell(sheet, value=safe_cell(value))
                    cell.data_type = "s"
                    cells.append(cell)
                sheet.append(cells)
            workbook.save(temp_path)
        else:
            raise ValueError(f"不支持的格式：{source_format}")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
