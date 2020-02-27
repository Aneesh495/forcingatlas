"""Text tables that keep missing markers as NaN.

`NA`, `NaN`, `nan`, and an empty field are missing. The token `0` is a real
zero. Nothing in this module substitutes one for the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.errors import ContractError


_MISSING = {"", "na", "nan", "n/a", "none"}


@dataclass(frozen=True, slots=True)
class NumericTable:
    columns: tuple[str, ...]
    data: dict[str, np.ndarray]
    comments: tuple[str, ...] = ()

    def column(self, name: str) -> np.ndarray:
        try:
            return self.data[name]
        except KeyError as exc:
            raise ContractError(f"table has no column {name!r}; columns are {self.columns}") from exc


def _split_csv_line(line: str, delimiter: str) -> list[str]:
    # The pinned climate tables do not quote fields that contain the delimiter.
    # A quoted field is still accepted so a future file does not split inside quotes.
    if '"' not in line:
        return [part.strip() for part in line.split(delimiter)]
    fields: list[str] = []
    current: list[str] = []
    quoted = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == '"':
            if quoted and index + 1 < len(line) and line[index + 1] == '"':
                current.append('"')
                index += 2
                continue
            quoted = not quoted
            index += 1
            continue
        if char == delimiter and not quoted:
            fields.append("".join(current).strip())
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    fields.append("".join(current).strip())
    return fields


def parse_numeric_token(token: str) -> float:
    text = token.strip()
    if text.lower() in _MISSING:
        return float("nan")
    try:
        return float(text)
    except ValueError as exc:
        raise ContractError(f"value {token!r} is neither numeric nor a missing marker") from exc


def parse_delimited(
    text: str,
    *,
    delimiter: str = ",",
    comment_prefixes: tuple[str, ...] = ("#", "%"),
    header: bool = True,
) -> NumericTable:
    """Parse a delimited numeric table.

    Comment lines are kept. A header row is required when `header` is true.
    Each data row must have the same width as the header. Short rows are
    rejected rather than padded with zeros.
    """

    comments: list[str] = []
    rows: list[list[str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in comment_prefixes):
            comments.append(stripped)
            continue
        rows.append(_split_csv_line(stripped, delimiter))
    if not rows:
        raise ContractError("table has no data rows")
    if header:
        columns = tuple(rows[0])
        body = rows[1:]
    else:
        width = len(rows[0])
        columns = tuple(f"c{index}" for index in range(width))
        body = rows
    if not body:
        raise ContractError("table has a header and no data rows")
    width = len(columns)
    if any(not name for name in columns):
        # An empty leading header is how the IGCC component ERF file labels the year.
        columns = tuple(name if name else "time" for name in columns)
    if len(set(columns)) != len(columns):
        raise ContractError(f"duplicate column names in {columns}")
    data = {name: np.empty(len(body), dtype=np.float64) for name in columns}
    for row_index, row in enumerate(body):
        if len(row) != width:
            raise ContractError(
                f"row {row_index} has {len(row)} fields, header has {width}. Refusing to pad."
            )
        for name, token in zip(columns, row, strict=True):
            data[name][row_index] = parse_numeric_token(token)
    return NumericTable(columns, data, tuple(comments))


def parse_berkeley_sections(text: str) -> tuple[tuple[str, ...], list[tuple[str, NumericTable]]]:
    """Split the Berkeley land-ocean text file into its two sea-ice treatments.

    The file is not one table. The first section infers air temperature over
    sea ice. The second infers water temperature. Returning a single stacked
    array would double-count every month.
    """

    lines = text.splitlines()
    comments = [line.strip() for line in lines if line.strip().startswith("%")]
    sections: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_rows: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%"):
            if "Global Average Temperature Anomaly" in stripped:
                if current_name is not None and current_rows:
                    sections.append((current_name, current_rows))
                current_name = stripped.lstrip("%").strip()
                current_rows = []
            continue
        if not stripped or set(stripped) <= {"%"}:
            continue
        if "Year" in stripped and "Month" in stripped and "Anomaly" in stripped:
            continue
        if current_name is None:
            continue
        current_rows.append(stripped)
    if current_name is not None and current_rows:
        sections.append((current_name, current_rows))
    if len(sections) != 2:
        raise ContractError(f"expected two Berkeley sea-ice sections, found {len(sections)}")
    tables: list[tuple[str, NumericTable]] = []
    columns = (
        "year",
        "month",
        "monthly_anomaly",
        "monthly_uncertainty_95",
        "annual_anomaly",
        "annual_uncertainty_95",
        "five_year_anomaly",
        "five_year_uncertainty_95",
        "ten_year_anomaly",
        "ten_year_uncertainty_95",
        "twenty_year_anomaly",
        "twenty_year_uncertainty_95",
    )
    for name, rows in sections:
        data = {column: np.empty(len(rows), dtype=np.float64) for column in columns}
        for index, row in enumerate(rows):
            parts = row.split()
            if len(parts) != 12:
                raise ContractError(f"Berkeley row has {len(parts)} fields, expected 12: {row[:80]}")
            for column, token in zip(columns, parts, strict=True):
                data[column][index] = parse_numeric_token(token)
        tables.append((name, NumericTable(columns, data)))
    return tuple(comments), tables


def find_comment_value(comments: tuple[str, ...] | list[str], label: str) -> str | None:
    """Return the remainder of the first comment that contains `label`."""

    for comment in comments:
        if label in comment:
            return comment.split(label, 1)[1].strip(" :%")
    return None
