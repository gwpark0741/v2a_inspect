#!/usr/bin/env python3
"""Backfill cost_usd and usage into existing experiment result JSONs from Langfuse traces.

Usage:
    python scripts/backfill_cost.py --results-dir experiment_results/run_2026-04-01/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v2a_inspect.observability import fetch_trace_cost, flush_langfuse  # noqa: E402


def backfill(results_dir: Path) -> None:
    vtt_dir = results_dir / "results" / "vtt"
    if not vtt_dir.exists():
        print(f"No results found in {vtt_dir}")
        return

    flush_langfuse()

    updated = 0
    skipped = 0
    failed = 0

    for json_file in sorted(vtt_dir.rglob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        trace_id = data.get("trace_id")

        if not trace_id:
            skipped += 1
            print(f"  SKIP (no trace_id): {json_file.name}")
            continue

        if data.get("cost_usd") is not None:
            skipped += 1
            print(f"  SKIP (already has cost): {json_file.name}")
            continue

        cost_data = fetch_trace_cost(trace_id)
        if cost_data is None:
            failed += 1
            print(f"  FAIL (fetch error): {json_file.name} trace={trace_id[:12]}")
            continue

        data["cost_usd"] = cost_data["total_cost"]
        data["usage"] = cost_data["usage_details"]
        json_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        cost_str = (
            f"${cost_data['total_cost']:.6f}" if cost_data["total_cost"] else "$0"
        )
        print(f"  OK: {json_file.name} → {cost_str}")
        updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill cost data from Langfuse traces"
    )
    parser.add_argument("--results-dir", required=True, type=str)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    backfill(results_dir)


if __name__ == "__main__":
    main()
