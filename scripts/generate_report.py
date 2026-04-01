#!/usr/bin/env python3
"""Generate HTML comparison reports from experiment results.

Usage:
    python scripts/generate_report.py --results-dir experiment_results/run_2026-03-31/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from jinja2 import Environment, FileSystemLoader

# Ensure project is importable for extract_clip
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"


# --------------------------------------------------------------------------- #
# Data structures for report
# --------------------------------------------------------------------------- #


@dataclass
class MetricValue:
    value: float
    display: str
    is_best: bool = False
    is_worst: bool = False


@dataclass
class ModelStats:
    avg_scenes: float = 0.0
    avg_events_per_scene: float = 0.0
    avg_desc_length: float = 0.0
    avg_latency: float = 0.0
    avg_cost: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    avg_groups: float = 0.0
    stability: float = 0.0
    error_rate: float = 0.0
    is_best_scenes: bool = False
    is_best_latency: bool = False
    is_best_cost: bool = False


@dataclass
class PromptStats:
    avg_scenes: float = 0.0
    avg_events_per_scene: float = 0.0
    avg_desc_length: float = 0.0
    avg_latency: float = 0.0
    change_description: str = ""


# --------------------------------------------------------------------------- #
# Result loading
# --------------------------------------------------------------------------- #


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all result JSONs grouped by video name."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    vtt_dir = results_dir / "results" / "vtt"
    if not vtt_dir.exists():
        print(f"No VTT results found in {vtt_dir}")
        return grouped

    for video_dir in sorted(vtt_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        for json_file in sorted(video_dir.glob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            grouped[video_dir.name].append(data)

    return grouped


def load_metadata(results_dir: Path) -> dict:
    meta_path = results_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"timestamp": "unknown", "fps": 3.0, "temperature": 0.0}


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #


def compute_scene_count(sa: dict) -> int:
    return len(sa.get("scenes", []))


def compute_total_events(sa: dict) -> int:
    return sum(len(s.get("audio_events", [])) for s in sa.get("scenes", []))


def compute_events_per_scene(sa: dict) -> float:
    scenes = sa.get("scenes", [])
    if not scenes:
        return 0.0
    return compute_total_events(sa) / len(scenes)


def compute_avg_desc_length(sa: dict) -> float:
    lengths = []
    for scene in sa.get("scenes", []):
        lengths.append(len(scene.get("background_ambience", "")))
        for ev in scene.get("audio_events", []):
            lengths.append(len(ev.get("description", "")))
    return mean(lengths) if lengths else 0.0


def compute_model_summary(
    results_by_video: dict[str, list[dict]],
) -> dict[str, ModelStats]:
    """Compute per-model summary stats across all videos."""

    model_results: dict[str, list[dict]] = defaultdict(list)
    for video_results in results_by_video.values():
        for r in video_results:
            if r["run"]["experiment_id"] == "model-compare" and r.get("scene_analysis"):
                model_results[r["run"]["model"]].append(r)

    summary: dict[str, ModelStats] = {}
    for model_name, results in model_results.items():
        valid = [r for r in results if r.get("scene_analysis")]
        if not valid:
            continue

        scene_counts = [compute_scene_count(r["scene_analysis"]) for r in valid]
        eps_values = [compute_events_per_scene(r["scene_analysis"]) for r in valid]
        desc_lengths = [compute_avg_desc_length(r["scene_analysis"]) for r in valid]
        latencies = [r["elapsed_seconds"] for r in valid]
        costs = [r.get("cost_usd", 0.0) or 0.0 for r in valid]

        # Token usage
        # "input" only counts non-cached new tokens; add "input_cache_read" for the full picture.
        # "output" excludes reasoning tokens; add "output_reasoning" for the full picture.
        input_tokens = []
        output_tokens = []
        total_tokens = []
        for r in valid:
            usage = r.get("usage")
            if usage:
                full_input = usage.get("input", 0) + usage.get("input_cache_read", 0)
                full_output = usage.get("output", 0) + usage.get("output_reasoning", 0)
                input_tokens.append(full_input)
                output_tokens.append(full_output)
                total_tokens.append(usage.get("total", 0))

        # Group counts
        group_counts = []
        for r in valid:
            ga = r.get("grouped_analysis")
            if ga:
                group_counts.append(len(ga.get("groups", [])))

        error_count = sum(1 for r in results if r.get("error"))

        # Stability: check scene count agreement across repeats
        stability_scores = []
        by_video_repeat: dict[str, list[int]] = defaultdict(list)
        for r in valid:
            video_name = Path(r["run"]["video_path"]).stem
            by_video_repeat[video_name].append(compute_scene_count(r["scene_analysis"]))
        for counts in by_video_repeat.values():
            if len(counts) >= 2:
                mode_count = max(set(counts), key=counts.count)
                stability_scores.append(counts.count(mode_count) / len(counts))

        summary[model_name] = ModelStats(
            avg_scenes=mean(scene_counts),
            avg_events_per_scene=mean(eps_values),
            avg_desc_length=mean(desc_lengths),
            avg_latency=mean(latencies),
            avg_cost=mean(costs) if costs else 0.0,
            avg_input_tokens=mean(input_tokens) if input_tokens else 0.0,
            avg_output_tokens=mean(output_tokens) if output_tokens else 0.0,
            avg_total_tokens=mean(total_tokens) if total_tokens else 0.0,
            avg_groups=mean(group_counts) if group_counts else 0.0,
            stability=mean(stability_scores) if stability_scores else 0.0,
            error_rate=error_count / len(results) if results else 0.0,
        )

    # Mark best values
    if summary:
        best_latency = min(s.avg_latency for s in summary.values())
        best_cost = min(s.avg_cost for s in summary.values())
        for stats in summary.values():
            if stats.avg_latency == best_latency:
                stats.is_best_latency = True
            if stats.avg_cost == best_cost:
                stats.is_best_cost = True

    return summary


def compute_prompt_summary(
    results_by_video: dict[str, list[dict]],
) -> dict[str, PromptStats]:
    """Compute per-prompt summary for Phase 3B results."""

    prompt_results: dict[str, list[dict]] = defaultdict(list)
    for video_results in results_by_video.values():
        for r in video_results:
            if r["run"]["experiment_id"] == "prompt-compare" and r.get(
                "scene_analysis"
            ):
                prompt_results[r["run"]["prompt_label"]].append(r)

    # Also include event-v1 from 3A for the best model
    for video_results in results_by_video.values():
        for r in video_results:
            if (
                r["run"]["experiment_id"] == "model-compare"
                and r["run"]["prompt_label"] == "event-v1"
                and r.get("scene_analysis")
            ):
                prompt_results["event-v1"].append(r)

    change_descriptions = {
        "event-v1": "베이스라인 (변경 없음)",
        "event-v2": "Anti-hallucination 지시 추가",
        "event-v3": "Few-shot 예시 추가",
    }

    summary: dict[str, PromptStats] = {}
    for label, results in sorted(prompt_results.items()):
        valid = [r for r in results if r.get("scene_analysis")]
        if not valid:
            continue

        summary[label] = PromptStats(
            avg_scenes=mean([compute_scene_count(r["scene_analysis"]) for r in valid]),
            avg_events_per_scene=mean(
                [compute_events_per_scene(r["scene_analysis"]) for r in valid]
            ),
            avg_desc_length=mean(
                [compute_avg_desc_length(r["scene_analysis"]) for r in valid]
            ),
            avg_latency=mean([r["elapsed_seconds"] for r in valid]),
            change_description=change_descriptions.get(label, ""),
        )

    return summary


# --------------------------------------------------------------------------- #
# Scene-level comparison data
# --------------------------------------------------------------------------- #


def _format_scene_analysis(
    sa: dict,
    grouped: dict | None = None,
) -> list[dict]:
    """Convert a scene_analysis dict into a list of scene_info dicts."""
    # Build track_to_group mapping from grouped_analysis
    track_to_group: dict[str, str] = {}
    groups_by_id: dict[str, dict] = {}
    if grouped:
        track_to_group = grouped.get("track_to_group", {})
        for g in grouped.get("groups", []):
            groups_by_id[g["group_id"]] = g

    scenes = sa.get("scenes", [])
    result = []
    for i, s in enumerate(scenes):
        scene_info: dict = {
            "index": i,
            "start": s.get("time_range", {}).get("start", "?"),
            "end": s.get("time_range", {}).get("end", "?"),
        }
        # Background with group info
        bg_track_id = f"s{i}_bg"
        bg_group_id = track_to_group.get(bg_track_id)
        scene_info["background"] = s.get("background_ambience", "—")
        scene_info["bg_group_id"] = bg_group_id

        events_formatted = []
        for ei, ev in enumerate(s.get("audio_events", [])):
            tr = ev.get("time_range", {})
            ts_list = ev.get("event_timestamps", [])
            ts_str = ", ".join(f"{t['time']}s" for t in ts_list[:5])
            if len(ts_list) > 5:
                ts_str += f" (+{len(ts_list) - 5} more)"
            ev_track_id = f"s{i}_ev{ei}"
            ev_group_id = track_to_group.get(ev_track_id)
            ev_group = groups_by_id.get(ev_group_id, {}) if ev_group_id else {}
            events_formatted.append(
                {
                    "description": ev.get("description", ""),
                    "time_range": f"{tr.get('start', '?')}s - {tr.get('end', '?')}s",
                    "source_visible": ev.get("source_visible", "?"),
                    "timestamps": ts_str,
                    "group_id": ev_group_id,
                    "vlm_verified": ev_group.get("vlm_verified", False),
                }
            )
        scene_info["events"] = events_formatted
        result.append(scene_info)
    return result


def build_scene_comparison(
    video_results: list[dict],
    experiment_id: str,
) -> tuple[list[str], list[dict], int]:
    """Build scene-by-scene comparison data for a video.

    Returns (column_names, repeats_data, repeat_count).
    repeats_data is a list of per-repeat dicts, each containing scenes keyed by column.
    """
    # Group by (column, repeat_index) — store both scene_analysis and grouped_analysis
    column_repeat_results: dict[str, dict[int, dict]] = defaultdict(dict)
    column_repeat_grouped: dict[str, dict[int, dict | None]] = defaultdict(dict)
    for r in video_results:
        if r["run"]["experiment_id"] != experiment_id:
            continue
        if not r.get("scene_analysis"):
            continue
        if experiment_id == "model-compare":
            col = r["run"]["model"]
        else:
            col = r["run"]["prompt_label"]
        ri = r["run"]["repeat_index"]
        column_repeat_results[col][ri] = r["scene_analysis"]
        column_repeat_grouped[col][ri] = r.get("grouped_analysis")

    if not column_repeat_results:
        return [], [], 0

    columns = sorted(column_repeat_results.keys())
    all_repeats = sorted(
        {ri for col_data in column_repeat_results.values() for ri in col_data}
    )

    repeats_data = []
    for ri in all_repeats:
        # Build merged scene list for this repeat
        col_scenes: dict[str, list[dict]] = {}
        col_groups: dict[str, list[dict]] = {}
        max_scenes = 0
        for col in columns:
            sa = column_repeat_results[col].get(ri)
            ga = column_repeat_grouped[col].get(ri)
            if sa:
                formatted = _format_scene_analysis(sa, grouped=ga)
                col_scenes[col] = formatted
                max_scenes = max(max_scenes, len(formatted))
            else:
                col_scenes[col] = []
            # Extract group list for this column
            col_groups[col] = ga.get("groups", []) if ga else []

        scenes_data = []
        for i in range(max_scenes):
            scene_info: dict = {
                "index": i,
                "start": "?",
                "end": "?",
                "backgrounds": {},
                "bg_group_ids": {},
                "events": {},
            }
            for col in columns:
                if i < len(col_scenes[col]):
                    s = col_scenes[col][i]
                    scene_info["start"] = s["start"]
                    scene_info["end"] = s["end"]
                    scene_info["backgrounds"][col] = s["background"]
                    scene_info["bg_group_ids"][col] = s.get("bg_group_id")
                    scene_info["events"][col] = s["events"]
                else:
                    scene_info["backgrounds"][col] = "(scene not detected)"
                    scene_info["bg_group_ids"][col] = None
                    scene_info["events"][col] = []
            scenes_data.append(scene_info)

        repeats_data.append(
            {
                "repeat_index": ri,
                "scenes": scenes_data,
                "groups": col_groups,
            }
        )

    return columns, repeats_data, len(all_repeats)


# --------------------------------------------------------------------------- #
# Video clip extraction for reports
# --------------------------------------------------------------------------- #


def extract_scene_clips(
    video_path: str,
    scenes_data: list[dict],
    clip_dir: Path,
    *,
    path_prefix: str = "../clips",
) -> None:
    """Extract video clips for each scene and set clip_path on scene dicts."""
    from v2a_inspect.ui.video import extract_clip

    clip_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    for scene in scenes_data:
        start = scene.get("start")
        end = scene.get("end")
        if start == "?" or end == "?":
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (ValueError, TypeError):
            continue

        clip_path = extract_clip(video_path, start_f, end_f, str(clip_dir))
        if clip_path:
            scene["clip_path"] = f"{path_prefix}/{video_name}/{Path(clip_path).name}"


# --------------------------------------------------------------------------- #
# Frame data for templates
# --------------------------------------------------------------------------- #


def load_frame_data(results_dir: Path, video_name: str) -> list[dict]:
    """Load frame file paths for a video."""
    frame_dir = results_dir / "frames" / video_name
    if not frame_dir.exists():
        return []

    frames = []
    for img_path in sorted(frame_dir.glob("frame_*.jpg")):
        # Extract timestamp from filename like "frame_001.5s.jpg"
        stem = img_path.stem  # "frame_001.5s"
        time_str = stem.replace("frame_", "").replace("s", "")
        try:
            t = float(time_str)
        except ValueError:
            t = 0.0
        rel_path = f"../frames/{video_name}/{img_path.name}"
        frames.append({"time": f"{t:.1f}", "path": rel_path})

    return frames


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #


def generate_reports(results_dir: Path) -> None:
    """Generate all HTML reports from experiment results."""

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )

    metadata = load_metadata(results_dir)
    results_by_video = load_results(results_dir)

    if not results_by_video:
        print("No results found. Nothing to generate.")
        return

    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "per_video").mkdir(exist_ok=True)

    # Compute summaries
    model_summary = compute_model_summary(results_by_video)
    prompt_summary = compute_prompt_summary(results_by_video)

    # Determine best model from 3A
    best_model = ""
    if model_summary:
        # Simple heuristic: highest stability, then lowest cost
        best_model = max(
            model_summary.keys(),
            key=lambda m: (model_summary[m].stability, -model_summary[m].avg_cost),
        )

    # Video info
    videos_info = []
    for video_name, results in sorted(results_by_video.items()):
        duration = "?"
        for r in results:
            if r.get("scene_analysis") and r["scene_analysis"].get("total_duration"):
                duration = r["scene_analysis"]["total_duration"]
                break
        videos_info.append(
            {
                "name": video_name,
                "duration": duration,
                "result_count": len(results),
            }
        )

    # --- index.html ---
    print("Generating index.html")
    tmpl = env.get_template("report_index.html.j2")
    html = tmpl.render(
        metadata=metadata,
        videos=videos_info,
        model_summary=model_summary,
        prompt_summary=prompt_summary if prompt_summary else None,
        best_model=best_model,
    )
    (reports_dir / "index.html").write_text(html, encoding="utf-8")

    # --- Per-video & per-video-model pages ---
    (reports_dir / "per_video").mkdir(exist_ok=True)
    all_model_names = sorted(model_summary.keys()) if model_summary else []
    per_video_tmpl = env.get_template("report_per_video.html.j2")
    per_video_model_tmpl = env.get_template("report_per_video_model.html.j2")

    for video_name, results in sorted(results_by_video.items()):
        print(f"Generating per_video/{video_name}.html")

        # Get video metadata
        duration = "?"
        video_path = None
        for r in results:
            if r.get("scene_analysis") and r["scene_analysis"].get("total_duration"):
                duration = r["scene_analysis"]["total_duration"]
            if video_path is None:
                video_path = r["run"].get("video_path")

        frames = load_frame_data(results_dir, video_name)

        # Build per-model summary for video overview page
        models_summary = []
        for model_name in all_model_names:
            model_results = [
                r
                for r in results
                if r["run"].get("model") == model_name
                and r["run"].get("experiment_id") == "model-compare"
                and r.get("scene_analysis")
            ]
            if not model_results:
                continue
            safe_model = model_name.replace("/", "_")
            avg_scenes = mean(
                [compute_scene_count(r["scene_analysis"]) for r in model_results]
            )
            avg_events = mean(
                [compute_events_per_scene(r["scene_analysis"]) for r in model_results]
            )
            avg_latency = mean([r["elapsed_seconds"] for r in model_results])
            models_summary.append(
                {
                    "name": model_name,
                    "safe_name": safe_model,
                    "avg_scenes": avg_scenes,
                    "avg_events_per_scene": avg_events,
                    "avg_latency": avg_latency,
                    "run_count": len(model_results),
                }
            )

        # Video overview page
        html = per_video_tmpl.render(
            metadata=metadata,
            video_name=video_name,
            duration=duration,
            frames=frames,
            models=models_summary,
            all_videos=sorted(results_by_video.keys()),
        )
        (reports_dir / "per_video" / f"{video_name}.html").write_text(
            html, encoding="utf-8"
        )

        # Per-video-model detail pages (with prompt tabs)
        for model_name in all_model_names:
            safe_model = model_name.replace("/", "_")
            model_results = [
                r
                for r in results
                if r["run"].get("model") == model_name and r.get("scene_analysis")
            ]
            if not model_results:
                continue

            # Group by prompt_label
            prompt_labels = sorted({r["run"]["prompt_label"] for r in model_results})
            prompts_data = []
            for label in prompt_labels:
                prompt_results = [
                    r for r in model_results if r["run"]["prompt_label"] == label
                ]
                repeats_data = []
                all_repeats = sorted({r["run"]["repeat_index"] for r in prompt_results})
                for ri in all_repeats:
                    r = next(
                        (r for r in prompt_results if r["run"]["repeat_index"] == ri),
                        None,
                    )
                    if not r or not r.get("scene_analysis"):
                        continue
                    ga = r.get("grouped_analysis")
                    scenes = _format_scene_analysis(r["scene_analysis"], grouped=ga)
                    groups = ga.get("groups", []) if ga else []

                    if video_path and Path(video_path).exists():
                        clip_dir = reports_dir / "clips" / video_name
                        extract_scene_clips(
                            video_path,
                            scenes,
                            clip_dir,
                            path_prefix="../clips",
                        )

                    repeats_data.append(
                        {
                            "repeat_index": ri,
                            "scenes": scenes,
                            "groups": groups,
                        }
                    )

                prompts_data.append(
                    {
                        "label": label,
                        "repeats": repeats_data,
                        "repeat_count": len(repeats_data),
                    }
                )

            page_name = f"{video_name}_{safe_model}.html"
            print(f"  Generating per_video/{page_name}")
            html = per_video_model_tmpl.render(
                metadata=metadata,
                video_name=video_name,
                duration=duration,
                model_name=model_name,
                frames=frames,
                prompts=prompts_data,
                all_models=[m["name"] for m in models_summary],
                all_videos=sorted(results_by_video.keys()),
            )
            (reports_dir / "per_video" / page_name).write_text(html, encoding="utf-8")

    # --- vtt_prompt_comparison.html (Phase 3B) ---
    has_3b = any(
        r["run"]["experiment_id"] == "prompt-compare"
        for results in results_by_video.values()
        for r in results
    )
    if has_3b:
        print("Generating vtt_prompt_comparison.html")
        video_data_3b = []
        all_columns_3b = []
        total_repeats_3b = 0
        for video_name, results in sorted(results_by_video.items()):
            columns, repeats_data, repeat_count = build_scene_comparison(
                results, "prompt-compare"
            )
            if not columns:
                continue
            all_columns_3b = columns
            total_repeats_3b = max(total_repeats_3b, repeat_count)
            frames = load_frame_data(results_dir, video_name)
            duration = "?"
            for r in results:
                if r.get("scene_analysis") and r["scene_analysis"].get(
                    "total_duration"
                ):
                    duration = r["scene_analysis"]["total_duration"]
                    break
            video_data_3b.append(
                {
                    "name": video_name,
                    "duration": duration,
                    "frames": frames,
                    "repeats": repeats_data,
                    "metrics": {},
                    "stability": {},
                }
            )

        html = tmpl.render(
            metadata=metadata,
            page_title=f"실험 3B: 프롬프트별 VTT 성능 비교 (model={best_model})",
            experiment_id="prompt-compare",
            control_description=f"통제: model={best_model}, fps=3, temp=0.0 | 독립변인: 프롬프트",
            independent_var="프롬프트 (event-v1, event-v2, event-v3)",
            controlled_vars=f"model={best_model}, fps=3, temperature=0.0, schema=events",
            repeats=metadata.get("repeats", 3),
            repeat_count=total_repeats_3b,
            columns=all_columns_3b,
            videos=video_data_3b,
        )
        (reports_dir / "vtt_prompt_comparison.html").write_text(html, encoding="utf-8")

    print(f"\nReports generated in: {reports_dir}")
    print(f"Open {reports_dir / 'index.html'} in a browser to view.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML experiment reports")
    parser.add_argument(
        "--results-dir",
        required=True,
        type=str,
        help="Path to experiment results directory",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    generate_reports(results_dir)


if __name__ == "__main__":
    main()
