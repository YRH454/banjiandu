"""CPU-only planning and three-seed aggregation; never launches training/SSH."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v3", "v4"), default="v4", help="Default v4: BCE plateau then adaptive modules; v3 is legacy")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Emit a new public config plan to stdout; not GPU-admitted")
    plan.add_argument("--policy", choices=("fixed", "adaptive"), help="Legacy v3 only; v4 uses stage-specific stopping")
    plan.add_argument("--stage", choices=("bce", "adaptation", "matched_bce"))
    plan.add_argument("--match-method")
    plan.add_argument("--dataset", choices=("apple", "cassava", "rice", "banana"))
    plan.add_argument("--budget", type=int, choices=(1, 5, 10, 20, 30, 100))
    plan.add_argument("--method")
    plan.add_argument("--table", choices=("main", "ablation", "warmup_control", "bce_reference", "matched_bce"))
    plan.add_argument("--counts-only", action="store_true")
    commands.add_parser("source-fingerprint", help="Emit public code SHA256 mapping; not real execution admission")
    summary = commands.add_parser("summarize", help="Validate private result files, emit aggregate statistics only")
    summary.add_argument("results", nargs="+", type=Path)
    summary.add_argument("--view", choices=("test", "validation_diagnostic"), default="test")
    summary.add_argument("--table", choices=("main", "ablation", "warmup_control", "bce_reference", "matched_bce"))
    args = parser.parse_args()
    try:
        if args.version == "v4":
            from plateau_benchmark.report import summarize_results
            from plateau_benchmark.spec import make_plan, source_fingerprint
        else:
            from fair_benchmark.report import summarize_results
            from fair_benchmark.references import source_fingerprint
            from fair_benchmark.spec import make_plan
        if args.command == "plan":
            if args.version == "v4" and (args.policy is not None or args.table == "warmup_control"):
                parser.error("v4 has adaptive phase budgets; use --version v3 for legacy policies/warmup_control")
            if args.version == "v3" and (args.stage is not None or args.match_method is not None or args.table in ("bce_reference", "matched_bce")):
                parser.error("v3 does not have plateau stages/matching targets")
            output = make_plan() if args.version == "v4" else make_plan(args.policy or "fixed")
            rows = [r for r in output["runs"] if (args.dataset is None or r["dataset"] == args.dataset)
                    and (args.budget is None or r["budget_percent"] == args.budget)
                    and (args.method is None or r["method"] == args.method)
                    and (args.stage is None or r.get("stage") == args.stage)
                    and (args.match_method is None or r.get("match_method") == args.match_method)
                    and (args.table is None or args.table in r["roles"])]
            if not rows:
                parser.error("No registered configuration matches")
            output["runs"] = rows
            output["selected_configurations"] = len(rows)
            if args.counts_only:
                del output["runs"]
        elif args.command == "source-fingerprint":
            output = source_fingerprint()
        else:
            output = summarize_results([json.loads(p.read_text(encoding="utf-8")) for p in args.results], view=args.view, table=args.table)
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
