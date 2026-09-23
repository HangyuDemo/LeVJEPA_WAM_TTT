#!/usr/bin/env python3
"""Generate one combined LIBERO-Plus result table for all model directories.

The input directory is expected to contain one directory for each suite:
``libero_spatial``, ``libero_object``, ``libero_goal`` and ``libero_10``.
Each suite directory must contain a completed ``summary-*.json`` file written
by ``experiments/robot/libero/run_libero_eval.py``.

The script recomputes the rates from integer success/episode counts and checks
the counts against ``episode_results`` before writing the table.  It writes:

* ``libero_plus_results.md``  - Markdown table
* ``libero_plus_results.csv`` - machine-readable table
* ``libero_plus_results.json`` - counts, rates and source summary paths
* ``libero_plus_results.png`` - a paper-style table image (unless ``--no-png``)

Example:
    python generate_libero_plus_table.py \
        /home/ha865618/project/LeVJEPA_WAM_TTT/rollout_ttt_eval/1st_round \
        --params 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SUITES: Tuple[Tuple[str, str], ...] = (
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "Long/10"),
)

CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("Camera Viewpoints", "Camera"),
    ("Robot Initial States", "Robot"),
    ("Language Instructions", "Language"),
    ("Light Conditions", "Light"),
    ("Background Textures", "Back."),
    ("Sensor Noise", "Noise"),
    ("Objects Layout", "Layout"),
)

MODEL_ORDER: Tuple[str, ...] = (
    "jepa-wam-ttt-inline-action-kv-vjepa21-context8-gb1-pb1",
    "jepa-wam-ttt-inline-jepa-memory-vjepa21-context8-gb1-pb1",
    "jepa-wam-ttt-wrapper-action-kv-vjepa21-context8-gb1-pb1",
    "jepa-wam-ttt-wrapper-jepa-memory-vjepa21-context8-gb1-pb1",
)

_SUMMARY_RE = re.compile(r"^summary-.*\.json$")
_EPS = 1e-9
DISPLAY_DIGITS = 1


class DataError(RuntimeError):
    """Raised when a result directory is incomplete or internally inconsistent."""


def _as_nonnegative_int(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataError(f"{source}: {field} must be a non-negative integer, got {value!r}")
    return value


def _rate_percent(successes: int, episodes: int, field: str) -> float:
    if episodes <= 0:
        raise DataError(f"{field}: episode count must be positive, got {episodes}")
    return successes * 100.0 / episodes


def _check_close(actual: float, expected: float, field: str, source: Path) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7):
        raise DataError(
            f"{source}: {field} disagrees with integer counts: "
            f"recorded={actual!r}, recomputed={expected!r}"
        )


def _latest_summary(suite_dir: Path) -> Path:
    summaries = sorted(
        path for path in suite_dir.iterdir() if path.is_file() and _SUMMARY_RE.match(path.name)
    )
    if not summaries:
        raise DataError(f"{suite_dir}: no summary-*.json file found")
    # The evaluator names files with YYYY_MM_DD-HH_MM_SS, so lexical order is
    # chronological and does not depend on filesystem mtimes.
    return summaries[-1]


def _validate_episode_results(
    payload: Mapping[str, Any],
    successes: int,
    episodes: int,
    category_results: Mapping[str, Any],
    source: Path,
) -> None:
    episode_results = payload.get("episode_results")
    if not isinstance(episode_results, list):
        raise DataError(f"{source}: episode_results is missing or is not a list")
    if len(episode_results) != episodes:
        raise DataError(
            f"{source}: episode_results has {len(episode_results)} entries, "
            f"but total_episodes is {episodes}"
        )

    counted_successes = sum(1 for episode in episode_results if episode.get("success") is True)
    if counted_successes != successes:
        raise DataError(
            f"{source}: episode_results contains {counted_successes} successes, "
            f"but successes is {successes}"
        )

    counted_categories: Dict[str, Dict[str, int]] = {}
    for episode in episode_results:
        category = episode.get("category")
        if category not in category_results:
            raise DataError(f"{source}: unknown episode category {category!r}")
        stats = counted_categories.setdefault(category, {"successes": 0, "episodes": 0})
        stats["episodes"] += 1
        if episode.get("success") is True:
            stats["successes"] += 1

    for category, _label in CATEGORIES:
        recorded = category_results.get(category)
        if not isinstance(recorded, Mapping):
            raise DataError(f"{source}: category_results lacks {category!r}")
        recorded_successes = _as_nonnegative_int(recorded.get("successes"), f"{category}.successes", source)
        recorded_episodes = _as_nonnegative_int(recorded.get("episodes"), f"{category}.episodes", source)
        counted = counted_categories.get(category, {"successes": 0, "episodes": 0})
        if (recorded_successes, recorded_episodes) != (
            counted["successes"],
            counted["episodes"],
        ):
            raise DataError(
                f"{source}: {category} counts disagree with episode_results: "
                f"recorded={recorded_successes}/{recorded_episodes}, "
                f"recomputed={counted['successes']}/{counted['episodes']}"
            )


def load_suite(model_dir: Path, suite_name: str, suite_label: str) -> Dict[str, Any]:
    suite_dir = model_dir / suite_name
    if not suite_dir.is_dir():
        raise DataError(f"{model_dir}: missing suite directory {suite_name}")
    summary_path = _latest_summary(suite_dir)
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{summary_path}: invalid JSON: {exc}") from exc

    if payload.get("status") != "completed":
        raise DataError(
            f"{summary_path}: status is {payload.get('status')!r}; "
            "refusing to include an incomplete evaluation"
        )

    successes = _as_nonnegative_int(payload.get("successes"), "successes", summary_path)
    episodes = _as_nonnegative_int(payload.get("total_episodes"), "total_episodes", summary_path)
    if successes > episodes:
        raise DataError(f"{summary_path}: successes cannot exceed total_episodes")
    rate = _rate_percent(successes, episodes, str(summary_path))
    _check_close(float(payload.get("success_rate")), rate / 100.0, "success_rate", summary_path)

    category_results = payload.get("category_results")
    if not isinstance(category_results, Mapping):
        raise DataError(f"{summary_path}: category_results is missing or is not an object")
    _validate_episode_results(payload, successes, episodes, category_results, summary_path)

    categories: Dict[str, Dict[str, Any]] = {}
    category_episode_sum = 0
    category_success_sum = 0
    for category, label in CATEGORIES:
        recorded = category_results[category]
        category_successes = _as_nonnegative_int(recorded["successes"], f"{category}.successes", summary_path)
        category_episodes = _as_nonnegative_int(recorded["episodes"], f"{category}.episodes", summary_path)
        category_rate = _rate_percent(category_successes, category_episodes, f"{summary_path}:{category}")
        _check_close(
            float(recorded.get("success_rate")),
            category_rate / 100.0,
            f"{category}.success_rate",
            summary_path,
        )
        categories[category] = {
            "label": label,
            "successes": category_successes,
            "episodes": category_episodes,
            "success_rate_percent": category_rate,
        }
        category_episode_sum += category_episodes
        category_success_sum += category_successes

    if (category_success_sum, category_episode_sum) != (successes, episodes):
        raise DataError(
            f"{summary_path}: category totals {category_success_sum}/{category_episode_sum} "
            f"do not match overall totals {successes}/{episodes}"
        )

    return {
        "label": suite_label,
        "successes": successes,
        "episodes": episodes,
        "success_rate_percent": rate,
        "status": payload["status"],
        "summary_path": str(summary_path.resolve()),
        "categories": categories,
    }


def infer_method(model_dir: Path) -> str:
    name = model_dir.name
    patterns = (
        ("inline-action-kv", "Inline + Action-KV"),
        ("inline-jepa-memory", "Inline + JEPA Memory"),
        ("wrapper-action-kv", "Wrapper + Action-KV"),
        ("wrapper-jepa-memory", "Wrapper + JEPA Memory"),
    )
    for token, label in patterns:
        if token in name:
            return label
    return name


def build_result(model_dir: Path, method: str, params: str) -> Dict[str, Any]:
    suites: Dict[str, Dict[str, Any]] = {}
    for suite_name, suite_label in SUITES:
        suites[suite_name] = load_suite(model_dir, suite_name, suite_label)

    suite_rates = [suites[name]["success_rate_percent"] for name, _label in SUITES]
    category_totals: Dict[str, Dict[str, Any]] = {
        category: {"label": label, "successes": 0, "episodes": 0}
        for category, label in CATEGORIES
    }
    for suite in suites.values():
        for category, _label in CATEGORIES:
            category_totals[category]["successes"] += suite["categories"][category]["successes"]
            category_totals[category]["episodes"] += suite["categories"][category]["episodes"]

    for category, _label in CATEGORIES:
        values = category_totals[category]
        values["success_rate_percent"] = _rate_percent(
            values["successes"], values["episodes"], f"pooled {category}"
        )

    overall_successes = sum(suite["successes"] for suite in suites.values())
    overall_episodes = sum(suite["episodes"] for suite in suites.values())
    return {
        "method": method,
        "params": params,
        "model_dir": str(model_dir.resolve()),
        "complete": True,
        "suites": suites,
        "suite_macro_average_percent": sum(suite_rates) / len(suite_rates),
        "categories": category_totals,
        "category_macro_average_percent": sum(
            values["success_rate_percent"] for values in category_totals.values()
        )
        / len(category_totals),
        "overall_successes": overall_successes,
        "overall_episodes": overall_episodes,
        "overall_micro_average_percent": _rate_percent(
            overall_successes, overall_episodes, "overall pooled results"
        ),
    }


def discover_model_dirs(root_dir: Path) -> List[Path]:
    """Find the four expected model directories in a round directory."""
    if not root_dir.is_dir():
        raise DataError(f"model root does not exist: {root_dir}")

    candidates = {
        path.name: path
        for path in root_dir.iterdir()
        if path.is_dir() and path.name.startswith("jepa-wam-ttt-")
    }
    ordered = [candidates[name] for name in MODEL_ORDER if name in candidates]
    unknown = sorted(path for name, path in candidates.items() if name not in MODEL_ORDER)
    ordered.extend(unknown)

    if len(ordered) != len(MODEL_ORDER):
        found = ", ".join(path.name for path in ordered) or "none"
        raise DataError(
            f"{root_dir}: expected {len(MODEL_ORDER)} model directories, found {len(ordered)}: {found}"
        )
    return ordered


def _table_columns() -> List[Tuple[str, str]]:
    # The category order intentionally matches the reference table:
    # Camera, Robot, Language, Light, Back., Noise, Layout, Avg.
    columns = [("Method", "method"), ("Params.", "params")]
    columns.extend((label, f"suite:{name}") for name, label in SUITES)
    columns.append(("Suite Avg.", "suite_avg"))
    columns.extend((label, f"category:{name}") for name, label in CATEGORIES)
    columns.append(("Avg.", "category_avg"))
    return columns


def _row(result: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": result["method"],
        "params": result["params"],
        "suite_avg": result["suite_macro_average_percent"],
        "category_avg": result["category_macro_average_percent"],
    }
    for suite_name, _label in SUITES:
        row[f"suite:{suite_name}"] = result["suites"][suite_name]["success_rate_percent"]
    for category, _label in CATEGORIES:
        row[f"category:{category}"] = result["categories"][category]["success_rate_percent"]
    return row


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.{DISPLAY_DIGITS}f}"
    return str(value)


def _round_floats(value: Any) -> Any:
    """Round presentation values while retaining integer success/episode counts."""
    if isinstance(value, float):
        return round(value, DISPLAY_DIGITS)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def write_markdown(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = _table_columns()
    rows = [_row(result) for result in results]
    header = "| " + " | ".join(label for label, _key in columns) + " |"
    separator = "|" + "|".join(
        ":--" if key in ("method", "params") else "--:"
        for _label, key in columns
    ) + "|"
    values = [
        "| " + " | ".join(_format_cell(row[key]) for _label, key in columns) + " |"
        for row in rows
    ]
    text = "\n".join(
        [
            header,
            separator,
            *values,
            "",
            "Suite Avg. is the macro average of the four task-suite rates "
            "(Spatial, Object, Goal and Long/10). Avg. is the macro average "
            "of the seven category rates after pooling episodes across the "
            "four suites.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_csv(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = _table_columns()
    rows = []
    for result in results:
        row = _row(result)
        rows.append(
            {
                key: round(value, DISPLAY_DIGITS) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for _label, key in columns])
        writer.writeheader()
        writer.writerows(rows)


def write_json(root_dir: Path, results: Sequence[Mapping[str, Any]], path: Path) -> None:
    payload = {
        "model_root": str(root_dir.resolve()),
        "complete": True,
        "models": list(results),
    }
    path.write_text(
        json.dumps(_round_floats(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_png(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DataError(
            "PNG output requires matplotlib; use --no-png or install matplotlib"
        ) from exc

    columns = _table_columns()
    rows = [_row(result) for result in results]
    labels = [label for label, _key in columns]
    values = [
        [_format_cell(row[key]) for _label, key in columns]
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(22, max(2.3, 1.35 + 0.45 * len(rows))))
    axis.axis("off")
    table = axis.table(
        cellText=values,
        colLabels=labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.12, 0.98, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        if row_index == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#e9e8ff")
        if column_index == 0:
            cell.set_width(0.18)
        elif column_index == 1:
            cell.set_width(0.055)
        else:
            cell.set_width(0.06)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_root",
        type=Path,
        help="round directory containing the four model rollout directories",
    )
    parser.add_argument("--params", default="-", help="Params. cell for all rows (default: -)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: MODEL_ROOT/eval_table)",
    )
    parser.add_argument("--no-png", action="store_true", help="do not generate the PNG table")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_root = args.model_root.expanduser().resolve()
    if not model_root.is_dir():
        print(f"error: model root does not exist: {model_root}", file=sys.stderr)
        return 2

    try:
        model_dirs = discover_model_dirs(model_root)
        results = [
            build_result(model_dir, method=infer_method(model_dir), params=args.params)
            for model_dir in model_dirs
        ]
        output_dir = (args.output_dir or model_root / "eval_table").expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_markdown(results, output_dir / "libero_plus_results.md")
        write_csv(results, output_dir / "libero_plus_results.csv")
        write_json(model_root, results, output_dir / "libero_plus_results.json")
        if not args.no_png:
            write_png(results, output_dir / "libero_plus_results.png")
    except (DataError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print((output_dir / "libero_plus_results.md").resolve())
    print((output_dir / "libero_plus_results.csv").resolve())
    print((output_dir / "libero_plus_results.json").resolve())
    if not args.no_png:
        print((output_dir / "libero_plus_results.png").resolve())
    print()
    print((output_dir / "libero_plus_results.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
