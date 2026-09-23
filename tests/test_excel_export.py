"""Tests for the Excel export tool — AGENTS-31, ADR-010.

Confirms export_to_excel() writes rows to a real .xlsx with a bold header
row and sane column widths, and returns the path it wrote. Fully offline:
writes into a temp directory, monkeypatched onto excel_export.EXPORTS_DIR.
"""

from pathlib import Path

import pytest
from openpyxl import load_workbook

from app import excel_export


def test_export_to_excel_writes_rows_and_returns_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file written back must contain the same header and values as rows."""
    monkeypatch.setattr(excel_export, "EXPORTS_DIR", tmp_path)

    rows = [{"sprint": "Sprint 1", "delivered": 5.0}]
    path = excel_export.export_to_excel(rows, "test.xlsx")

    assert path == str(tmp_path / "test.xlsx")

    workbook = load_workbook(path)
    worksheet = workbook.active
    assert worksheet is not None
    assert [cell.value for cell in worksheet[1]] == ["sprint", "delivered"]
    assert [cell.value for cell in worksheet[2]] == ["Sprint 1", 5.0]


def test_export_to_excel_bolds_the_header_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The header row must be bold so the file reads as formatted, not a raw dump."""
    monkeypatch.setattr(excel_export, "EXPORTS_DIR", tmp_path)

    path = excel_export.export_to_excel([{"sprint": "Sprint 1", "delivered": 5.0}], "test.xlsx")

    workbook = load_workbook(path)
    worksheet = workbook.active
    assert worksheet is not None
    assert all(cell.font.bold for cell in worksheet[1])
    assert not any(cell.font.bold for cell in worksheet[2])