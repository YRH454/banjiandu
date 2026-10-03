"""Read-only lookup of public scientific definitions; no training/data/GPU access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/experiment_catalog_20261004.json"

def find(crop=None, method=None, budget=None, stage=None, run_id=None):
    rows = json.loads(CATALOG.read_text(encoding="utf-8"))["runs"]
    if budget is not None:
        value = float(str(budget).rstrip("%"))
        if value not in (1, 5, 10, 20, 30, 100):
            raise ValueError("Budget must be 1, 5, 10, 20, 30, or 100 percent")
        budget = f"{int(value):03d}"
    return [r for r in rows
            if (crop is None or r["dataset"] == crop)
            and (method is None or r["method"].lower() == method.lower())
            and (budget is None or r["budget"] == budget)
            and (stage is None or (r["stage"] or "").upper() == stage.upper())
            and (run_id is None or r["run_id"] == run_id)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crop", choices=("apple", "cassava", "rice", "banana"))
    p.add_argument("--method", help="meanteacher, fixmatch, freematch, or exact archived method name")
    p.add_argument("--stage", choices=("S1", "S2", "S3", "S4"))
    p.add_argument("--budget", help="Percent, e.g. 10 or 010 or 10%%")
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        rows = find(a.crop, a.method, a.budget, a.stage, a.run_id)
    except ValueError as e:
        p.error(str(e))
    if not rows:
        p.error("No registered scientific definition matches; no new config was generated")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("Public views only: not completion proofs or historical launch bindings.")

if __name__ == "__main__":
    main()
