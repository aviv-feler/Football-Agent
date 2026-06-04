"""
Inspect all project data files and write data/data_source_inventory.csv.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


DATA_DIR = Path("data")
OUT = DATA_DIR / "data_source_inventory.csv"


def classify(columns: list[str], path: Path) -> str:
    s = {c.lower() for c in columns}
    name = path.name.lower()
    if "players" in name or {"player_id", "player_name"} & s:
        if "appearances" in name:
            return "player match appearances"
        if "valuations" in name:
            return "player valuations"
        if "lineups" in name:
            return "match lineups"
        return "players"
    if "clubs" in name or "club_id" in s:
        return "clubs/teams"
    if "games" in name or {"home_club_id", "away_club_id"} & s:
        return "matches/results"
    if "events" in name:
        return "match events"
    if "competitions" in name:
        return "leagues/competitions"
    if "countries" in name or "national_teams" in name:
        return "countries/national teams"
    if "schedule" in name or "fixture" in name:
        return "fixtures"
    if "feature_meta" in name:
        return "model metadata"
    return "unknown"


def read_xlsx_header(path: Path) -> tuple[int | None, int | None, list[str], list[dict]]:
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared.append("".join(texts))
        sheet_name = next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        root = ET.fromstring(z.read(sheet_name))
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = []
        max_col = 0
        for row in root.findall(".//a:row", ns)[:4]:
            vals = []
            for c in row.findall("a:c", ns):
                ref = c.attrib.get("r", "")
                letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = 0
                for ch in letters:
                    col_idx = col_idx * 26 + ord(ch.upper()) - 64
                max_col = max(max_col, col_idx)
                v = c.find("a:v", ns)
                val = ""
                if v is not None:
                    val = v.text or ""
                    if c.attrib.get("t") == "s":
                        val = shared[int(val)]
                vals.append(val)
            rows.append(vals)
        header = rows[0] if rows else []
        sample = [dict(zip(header, row)) for row in rows[1:3] if header]
        row_count = len(root.findall(".//a:row", ns)) - 1
        return row_count, max_col, header, sample


def main():
    records = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls", ".json"}:
            continue
        record = {
            "file_name": str(path).replace("\\", "/"),
            "rows": None,
            "columns_count": None,
            "columns": "",
            "sample_rows": "",
            "data_type": "unknown",
        }
        try:
            if path.suffix.lower() == ".csv":
                sample = pd.read_csv(path, nrows=3, low_memory=False)
                record["rows"] = max(sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1, 0)
                record["columns_count"] = len(sample.columns)
                record["columns"] = json.dumps(list(sample.columns), ensure_ascii=False)
                record["sample_rows"] = sample.head(2).to_json(orient="records", force_ascii=False)
                record["data_type"] = classify(list(sample.columns), path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                rows, cols, header, sample = read_xlsx_header(path)
                record["rows"] = rows
                record["columns_count"] = cols
                record["columns"] = json.dumps(header, ensure_ascii=False)
                record["sample_rows"] = json.dumps(sample, ensure_ascii=False)
                record["data_type"] = classify(header, path)
            else:
                data = json.load(open(path, encoding="utf-8"))
                keys = list(data.keys()) if isinstance(data, dict) else []
                record["rows"] = len(data) if isinstance(data, list) else 1
                record["columns_count"] = len(keys)
                record["columns"] = json.dumps(keys, ensure_ascii=False)
                record["sample_rows"] = json.dumps(data if not isinstance(data, dict) else {k: data[k] for k in keys[:5]}, ensure_ascii=False)[:1000]
                record["data_type"] = classify(keys, path)
        except Exception as exc:
            record["sample_rows"] = f"ERROR: {exc}"
        records.append(record)

    df = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
