"""CPU-only planning and three-seed aggregation; never launches training/SSH."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fair_benchmark.report import summarize_results
from fair_benchmark.references import source_fingerprint
from fair_benchmark.spec import load_protocol, make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Emit a new public config plan to stdout; not GPU-admitted")
    plan.add_argument("--policy", choices=("fixed", "adaptive"), default="fixed")
    plan.add_argument("--dataset", choices=("apple", "cassava", "rice", "banana"))
    plan.add_argument("--budget", type=int, choices=(1, 5, 10, 20, 30, 100))
    plan.add_argument("--method")
    plan.add_argument("--counts-only", action="store_true")
    commands.add_parser("source-fingerprint", help="Emit public code SHA256 mapping; not real execution admission")
    summary = commands.add_parser("summarize", help="Validate private result files, emit aggregate statistics only")
    summary.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            output = make_plan(args.policy)
            rows = [r for r in output["runs"] if (args.dataset is None or r["dataset"] == args.dataset)
                    and (args.budget is None or r["budget_percent"] == args.budget)
                    and (args.method is None or r["method"] == args.method)]
            if not rows:
                parser.error("No registered fair-v2 configuration matches")
            output["runs"] = rows
            output["selected_configurations"] = len(rows)
            if args.counts_only:
                del output["runs"]
        elif args.command == "source-fingerprint":
            output = source_fingerprint()
        else:
            output = summarize_results([json.loads(p.read_text(encoding="utf-8")) for p in args.results])
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
