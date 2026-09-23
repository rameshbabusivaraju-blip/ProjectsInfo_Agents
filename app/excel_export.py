"""Excel export tool — ADR-010.

The one Tool-type question in the catalogue (I4) needs the agent to hand back
a spreadsheet, not just prose. export_to_excel() is that tool: it takes a
result set already produced by a reviewed SQL query and writes it to a
formatted .xlsx. It never computes or reshapes a number, only lays rows out —
success is unambiguous, the file exists with the right rows or it does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Font

EXPORTS_DIR = Path("exports")


def export_to_excel(rows: list[dict[str, Any]], filename: str) -> str:
    """Write rows to a formatted .xlsx under exports/ and return the file's path.

    Bold header row, columns sized to their content — enough to count as
    "formatted" per ADR-010 without building a styling system nobody asked for.
    """
    EXPORTS_DIR.mkdir(exist_ok=True)
    path = EXPORTS_DIR / filename

    frame = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Sheet1")
        worksheet = writer.sheets["Sheet1"]

        for cell in worksheet[1]:
            cell.font = Font(bold=True)

        for column_cells in worksheet.columns:
            width = max(len(str(cell.value)) for cell in column_cells) + 2
            worksheet.column_dimensions[column_cells[0].column_letter].width = width

    return str(path)