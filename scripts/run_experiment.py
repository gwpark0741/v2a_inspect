#!/usr/bin/env python3
"""Experiment runner for LLM Video-to-Text comparison.

Runs experiments through the LangGraph pipeline with full Langfuse tracing.

Usage:
    # Compare models (prompt fixed to event-v1)
    python scripts/run_experiment.py --phase model-compare --videos data/videos/*.mp4

    # Compare prompts (model fixed to best from model-compare)
    python scripts/run_experiment.py --phase prompt-compare --best-model gemini:gemini-3.1-pro-preview --videos data/videos/*.mp4

    # Dry run (show what would execute without calling APIs)
    python scripts/run_experiment.py --phase model-compare --videos data/videos/*.mp4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Ensure project is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v2a_inspect.experiment.frames import save_frames_to_dir  # noqa: E402
from v2a_inspect.observability import (  # noqa: E402
    build_cli_trace_context,
    fetch_trace_cost,
    flush_langfuse,
)
from v2a_inspect.runner import run_inspect  # noqa: E402
from v2a_inspect.workflows import InspectOptions  # noqa: E402


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentRun:
    experiment_id: str
    video_path: str
    provider: Literal["gemini", "openai"]
    model: str
    prompt_label: str
    fps: float = 3.0
    temperature: float = 0.1
    repeat_index: int = 0


@dataclass
class ExperimentResult:
    run: dict
    scene_analysis: dict | None
    grouped_analysis: dict | None
    elapsed_seconds: float
    trace_id: str | None
    error: str | None
    timestamp: str
    cost_usd: float | None = None
    usage: dict | None = None


# --------------------------------------------------------------------------- #
# Experiment execution via LangGraph pipeline
# --------------------------------------------------------------------------- #


PROMPT_LABEL_TO_MODE = {
    "event-v1": "default",
    "event-v2": "v2",
    "event-v3": "v3",
    "object-based": "object_based",
    "event-constrained": "constrained",
}


def execute_run(run: ExperimentRun) -> ExperimentResult:
    """Execute a single experiment run through the LangGraph pipeline."""

    ts = datetime.now(timezone.utc).isoformat()

    scene_analysis_mode = PROMPT_LABEL_TO_MODE.get(run.prompt_label, "default")

    options = InspectOptions(
        provider=run.provider,
        model_name=run.model,
        fps=run.fps,
        temperature=run.temperature,
        scene_analysis_mode=scene_analysis_mode,
        enable_vlm_verify=True,
        enable_model_select=False,
    )

    trace_context = build_cli_trace_context(
        "analyze",
        tags=["experiment", run.experiment_id, run.prompt_label],
        metadata={
            "experiment_id": run.experiment_id,
            "prompt_label": run.prompt_label,
            "repeat_index": run.repeat_index,
        },
    )

    start = time.monotonic()
    try:
        final_state = run_inspect(
            run.video_path,
            options=options,
            trace_context=trace_context,
        )
        elapsed = time.monotonic() - start

        scene_analysis = final_state.get("scene_analysis")
        scene_analysis_dict = scene_analysis.model_dump() if scene_analysis else None
        grouped = final_state.get("grouped_analysis")
        grouped_dict = grouped.model_dump() if grouped else None
        trace_id = final_state.get("trace_id")

        # Fetch cost/usage from Langfuse
        cost_usd = None
        usage = None
        if trace_id:
            flush_langfuse()
            time.sleep(2)
            cost_data = fetch_trace_cost(trace_id)
            if cost_data:
                cost_usd = cost_data["total_cost"]
                usage = cost_data["usage_details"]

        return ExperimentResult(
            run=asdict(run),
            scene_analysis=scene_analysis_dict,
            grouped_analysis=grouped_dict,
            elapsed_seconds=round(elapsed, 2),
            trace_id=trace_id,
            error=None,
            timestamp=ts,
            cost_usd=cost_usd,
            usage=usage,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return ExperimentResult(
            run=asdict(run),
            scene_analysis=None,
            grouped_analysis=None,
            elapsed_seconds=round(elapsed, 2),
            trace_id=None,
            error=str(exc),
            timestamp=ts,
        )


def save_result(result: ExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n"
    output_path.write_text(payload, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Experiment matrix builders
# --------------------------------------------------------------------------- #

TARGET_MODELS = [
    ("gemini", "gemini-3.1-pro-preview"),
    ("openai", "gpt-5.4"),
]


def build_model_compare(
    video_paths: list[str], repeats: int = 3
) -> list[ExperimentRun]:
    """model-compare: Compare models with prompt fixed to event-v1."""
    runs = []
    for vp in video_paths:
        for provider, model in TARGET_MODELS:
            for r in range(repeats):
                runs.append(
                    ExperimentRun(
                        experiment_id="model-compare",
                        video_path=vp,
                        provider=provider,
                        model=model,
                        prompt_label="event-v1",
                        repeat_index=r,
                    )
                )
    return runs


TARGET_PROMPTS = ["event-v1", "object-based"]


def build_full_compare(video_paths: list[str], repeats: int = 3) -> list[ExperimentRun]:
    """full-compare: All combinations of models × prompts × videos × repeats."""
    runs = []
    for vp in video_paths:
        for provider, model in TARGET_MODELS:
            for label in TARGET_PROMPTS:
                for r in range(repeats):
                    runs.append(
                        ExperimentRun(
                            experiment_id="full-compare",
                            video_path=vp,
                            provider=provider,
                            model=model,
                            prompt_label=label,
                            repeat_index=r,
                        )
                    )
    return runs


def build_prompt_compare(
    video_paths: list[str],
    best_model: tuple[str, str],
    repeats: int = 3,
) -> list[ExperimentRun]:
    """prompt-compare: Compare prompts with model fixed."""
    provider, model = best_model
    runs = []
    for vp in video_paths:
        for label in TARGET_PROMPTS:
            for r in range(repeats):
                runs.append(
                    ExperimentRun(
                        experiment_id="prompt-compare",
                        video_path=vp,
                        provider=provider,
                        model=model,
                        prompt_label=label,
                        repeat_index=r,
                    )
                )
    return runs


# --------------------------------------------------------------------------- #
# Output path helpers
# --------------------------------------------------------------------------- #


def result_path(output_dir: Path, run: ExperimentRun) -> Path:
    video_name = Path(run.video_path).stem
    filename = (
        f"{run.experiment_id}_{run.model}_{run.prompt_label}_r{run.repeat_index}.json"
    )
    return output_dir / "results" / "vtt" / video_name / filename


def run_experiment_matrix(
    runs: list[ExperimentRun],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> list[ExperimentResult]:
    """Run all experiments, skipping already-completed ones."""
    results = []
    total = len(runs)

    for i, run in enumerate(runs, 1):
        out_path = result_path(output_dir, run)
        video_name = Path(run.video_path).stem

        if out_path.exists():
            print(f"  [{i}/{total}] SKIP (exists): {out_path.name}")
            continue

        print(
            f"  [{i}/{total}] {run.experiment_id} | {run.provider}:{run.model} | {run.prompt_label} "
            f"| r{run.repeat_index} | {video_name}"
        )

        if dry_run:
            print(f"           → (dry run) would write to {out_path}")
            continue

        result = execute_run(run)
        save_result(result, out_path)

        status = "OK" if result.error is None else f"ERROR: {result.error[:80]}"
        trace_info = f" | trace={result.trace_id[:8]}" if result.trace_id else ""
        print(f"           → {result.elapsed_seconds}s | {status}{trace_info}")

        results.append(result)

    return results


# --------------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------------- #


def extract_all_frames(
    video_paths: list[str],
    output_dir: Path,
    fps: float = 3.0,
) -> None:
    """Extract and save frames for all videos (for HTML reports)."""
    for vp in video_paths:
        video_name = Path(vp).stem
        frame_dir = output_dir / "frames" / video_name
        if frame_dir.exists() and any(frame_dir.iterdir()):
            print(f"  Frames already extracted: {video_name}")
            continue
        print(f"  Extracting frames: {video_name} (fps={fps})")
        save_frames_to_dir(vp, str(frame_dir), fps=fps)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "fps": 3.0,
        "temperature": 0.0,
        "videos": args.videos,
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
        meta["git_hash"] = git_hash
    except Exception:
        pass

    meta_path = output_dir / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_best_model(value: str) -> tuple[str, str]:
    """Parse 'provider:model' string like 'gemini:gemini-3.1-pro-preview'."""
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected 'provider:model' format (e.g., 'gemini:gemini-3.1-pro-preview'), got: {value}"
        )
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="V2A Inspect Experiment Runner")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["model-compare", "prompt-compare", "full-compare"],
        help="model-compare: compare LLMs | prompt-compare: compare prompts | full-compare: all combinations",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Paths to video files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiment_results/run_YYYY-MM-DD)",
    )
    parser.add_argument(
        "--best-model",
        type=str,
        default=None,
        help="For prompt-compare: best model in 'provider:model' format (e.g., gemini:gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeat runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing",
    )
    args = parser.parse_args()

    # Validate video files
    for vp in args.videos:
        if not Path(vp).exists():
            parser.error(f"Video file not found: {vp}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_dir = PROJECT_ROOT / "experiment_results" / f"run_{date_str}"

    print(f"\n{'=' * 60}")
    print(f"V2A Inspect Experiment — {args.phase}")
    print(f"{'=' * 60}")
    print(f"Output: {output_dir}")
    print(f"Videos: {len(args.videos)}")
    print(f"Repeats: {args.repeats}")
    print(f"Dry run: {args.dry_run}")

    # Build experiment matrix
    if args.phase == "model-compare":
        runs = build_model_compare(args.videos, repeats=args.repeats)
        print(
            f"Total runs: {len(runs)} ({len(args.videos)} videos × 3 models × {args.repeats} repeats)"
        )

    elif args.phase == "prompt-compare":
        if not args.best_model:
            parser.error(
                "prompt-compare requires --best-model (e.g., gemini:gemini-3.1-pro-preview)"
            )
        best = parse_best_model(args.best_model)
        runs = build_prompt_compare(args.videos, best_model=best, repeats=args.repeats)
        print(f"Best model: {best[0]}:{best[1]}")
        print(
            f"Total runs: {len(runs)} ({len(args.videos)} videos × {len(TARGET_PROMPTS)} prompts × {args.repeats} repeats)"
        )

    elif args.phase == "full-compare":
        runs = build_full_compare(args.videos, repeats=args.repeats)
        print(
            f"Total runs: {len(runs)} ({len(args.videos)} videos × {len(TARGET_MODELS)} models × {len(TARGET_PROMPTS)} prompts × {args.repeats} repeats)"
        )

    else:
        parser.error(f"Unknown phase: {args.phase}")
        return

    print(f"{'=' * 60}\n")

    # Save metadata
    if not args.dry_run:
        save_metadata(output_dir, args)

    # Extract frames for HTML reports
    print("Frame extraction")
    if not args.dry_run:
        extract_all_frames(args.videos, output_dir, fps=3.0)
    else:
        print("  (dry run) skipping frame extraction")

    # Run experiments
    print(f"\nRunning {args.phase}")
    results = run_experiment_matrix(runs, output_dir, dry_run=args.dry_run)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    completed = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    if completed:
        avg_time = sum(r.elapsed_seconds for r in completed) / len(completed)
        print(f"Avg latency: {avg_time:.1f}s")
        trace_ids = [r.trace_id for r in completed if r.trace_id]
        if trace_ids:
            print(f"Langfuse traces: {len(trace_ids)} recorded")
    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r.run['model']} / {r.run['prompt_label']}: {r.error[:100]}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
