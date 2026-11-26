"""Plot the frozen streaming-tag versus JSON command-format experiment."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TAG_COLOR = "#E76F51"
JSON_COLOR = "#457B9D"
CATEGORY_ORDER = ("single", "parameterized", "speech", "tachie", "multi")
CATEGORY_LABELS = {
    "single": "Single", "parameterized": "Parameterized", "speech": "Speech",
    "tachie": "Character", "multi": "Multi-command",
}


def build_chart_data(metrics: dict, records: list[dict]) -> dict:
    angle = metrics["formats"]["angle"]
    json_values = metrics["formats"]["json"]
    paired = metrics["paired_correct"]
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        tokens = record.get("completion_tokens")
        if record.get("format") in ("angle", "json") and isinstance(tokens, (int, float)):
            grouped[record["category"]][record["format"]].append(float(tokens))
    present = [
        category for category in CATEGORY_ORDER
        if grouped[category]["angle"] and grouped[category]["json"]
    ]
    return {
        "token_labels": [
            "Mean output", "Median output", "Paired-correct mean",
        ],
        "token_groups": {
            "angle": [
                angle["mean_completion_tokens"], angle["median_completion_tokens"],
                paired["mean_angle_tokens"],
            ],
            "json": [
                json_values["mean_completion_tokens"], json_values["median_completion_tokens"],
                paired["mean_json_tokens"],
            ],
        },
        "category_keys": present,
        "category_labels": [CATEGORY_LABELS[category] for category in present],
        "category_tokens": {
            format_name: [
                sum(grouped[category][format_name]) / len(grouped[category][format_name])
                for category in present
            ]
            for format_name in ("angle", "json")
        },
        "reliability_labels": ["Strict parse", "Exact semantics"],
        "reliability": {
            "angle": [angle["strict_parse_rate"], angle["semantic_exact_rate"]],
            "json": [json_values["strict_parse_rate"], json_values["semantic_exact_rate"]],
        },
        "savings_rate": paired["token_savings_rate"],
        "difference_ci95": paired["json_minus_angle_ci95"],
        "pair_count": paired.get("pair_count"),
    }


def _grouped_bars(
    ax, labels, left, right, *, ylabel: str, percent: bool = False,
    tick_size: int = 20, value_size: int = 19,
) -> None:
    import numpy as np
    positions = np.arange(len(labels), dtype=float)
    width = 0.30
    separation = 0.40
    left_bars = ax.bar(positions - separation / 2, left, width, color=TAG_COLOR, label="Streaming tags")
    right_bars = ax.bar(positions + separation / 2, right, width, color=JSON_COLOR, label="JSON")
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.grid(axis="y", color="#D8DEE4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for label in ax.get_xticklabels():
        label.set_rotation(18)
        label.set_horizontalalignment("right")
        label.set_fontsize(tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.tick_params(axis="x", pad=6)
    if percent:
        ax.set_ylim(0, 1.12)
        ax.set_yticks([index / 10 for index in range(0, 11, 2)])
        ax.set_yticklabels([f"{index}%" for index in range(0, 101, 20)])
        formatter = lambda value: f"{value * 100:.1f}%"
        offset = 0.025
    else:
        maximum = max([*left, *right])
        ax.set_ylim(0, maximum * 1.22)
        formatter = lambda value: f"{value:.2f}"
        offset = maximum * 0.025
    for series_index, bars in enumerate((left_bars, right_bars)):
        for bar in bars:
            value = bar.get_height()
            series_offset = (0.05 if percent else maximum * 0.04) * series_index
            horizontal_offset = -0.025 if series_index == 0 else 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2 + horizontal_offset,
                value + offset + series_offset,
                formatter(value),
                ha="center", va="bottom", fontsize=value_size, fontweight="bold",
            )


def _style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 20,
        "axes.edgecolor": "#6B7280", "axes.labelcolor": "#24292F",
        "text.color": "#24292F", "xtick.color": "#3D444D", "ytick.color": "#3D444D",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def render_charts(data: dict, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    legend = [
        Patch(facecolor=TAG_COLOR, label="Streaming tags (<...>)"),
        Patch(facecolor=JSON_COLOR, label="JSON"),
    ]
    outputs = []

    def save(figure, stem: str) -> None:
        for extension in ("png", "pdf"):
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.8, 6.4))
    _grouped_bars(
        axis, data["token_labels"], data["token_groups"]["angle"],
        data["token_groups"]["json"], ylabel="Completion tokens",
    )
    figure.suptitle(
        "Streaming Tags Use Fewer Output Tokens",
        fontsize=25, fontweight="bold", y=0.98,
    )
    low, high = data["difference_ci95"]
    figure.text(
        0.5, 0.905,
        f"Paired-correct savings: {data['savings_rate'] * 100:.1f}%  |  JSON − tags 95% CI: {low:.2f}–{high:.2f} tokens",
        ha="center", va="center", fontsize=16,
    )
    figure.legend(
        handles=legend, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.865), ncol=2, fontsize=18,
    )
    figure.subplots_adjust(top=0.73, bottom=0.25, left=0.16, right=0.98)
    save(figure, "stream-format-output-tokens")

    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    _grouped_bars(
        axis, data["category_labels"], data["category_tokens"]["angle"],
        data["category_tokens"]["json"], ylabel="Mean completion tokens",
    )
    figure.suptitle("Output Tokens by Command Category", fontsize=25, fontweight="bold", y=0.98)
    figure.legend(
        handles=legend, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.90), ncol=2, fontsize=18,
    )
    figure.subplots_adjust(top=0.79, bottom=0.24, left=0.16, right=0.98)
    save(figure, "stream-format-category-tokens")

    figure, axis = plt.subplots(figsize=(7.8, 6.2))
    _grouped_bars(
        axis, data["reliability_labels"], data["reliability"]["angle"],
        data["reliability"]["json"], ylabel="Success rate", percent=True,
    )
    figure.suptitle("Automatic Format Reliability", fontsize=25, fontweight="bold", y=0.98)
    figure.legend(
        handles=legend, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.90), ncol=2, fontsize=18,
    )
    figure.subplots_adjust(top=0.79, bottom=0.22, left=0.17, right=0.98)
    save(figure, "stream-format-reliability")

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    _grouped_bars(
        axes[0], data["token_labels"], data["token_groups"]["angle"],
        data["token_groups"]["json"], ylabel="Completion tokens",
        tick_size=17, value_size=19,
    )
    axes[0].set_title("Output Token Cost", fontsize=23, fontweight="bold", pad=12)
    _grouped_bars(
        axes[1], data["category_labels"], data["category_tokens"]["angle"],
        data["category_tokens"]["json"], ylabel="Mean completion tokens",
        tick_size=16, value_size=18,
    )
    axes[1].set_title("Token Cost by Category", fontsize=23, fontweight="bold", pad=12)
    _grouped_bars(
        axes[2], data["reliability_labels"], data["reliability"]["angle"],
        data["reliability"]["json"], ylabel="Success rate", percent=True,
        tick_size=17, value_size=19,
    )
    axes[2].set_title("Reliability", fontsize=23, fontweight="bold", pad=12)
    figure.legend(handles=legend, frameon=False, loc="upper center", ncol=2, fontsize=20)
    figure.subplots_adjust(top=0.80, bottom=0.27, left=0.06, right=0.995, wspace=0.27)
    save(figure, "stream-format-summary")
    return outputs


def write_chart_data(data: dict, output: Path) -> Path:
    rows = []
    for index, label in enumerate(data["token_labels"]):
        for format_name in ("angle", "json"):
            rows.append(("overall_tokens", label, format_name, data["token_groups"][format_name][index]))
    for index, label in enumerate(data["category_labels"]):
        for format_name in ("angle", "json"):
            rows.append(("category_tokens", label, format_name, data["category_tokens"][format_name][index]))
    for index, label in enumerate(data["reliability_labels"]):
        for format_name in ("angle", "json"):
            rows.append(("reliability", label, format_name, data["reliability"][format_name][index]))
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("section", "measure", "format", "value"))
        writer.writerows(rows)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path(r"D:\AI_cache\hum_project_evaluation\runs\20261126-command-format-formal"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/figures/2026-11-26-command-format"),
    )
    args = parser.parse_args(argv)
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in (args.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    data = build_chart_data(metrics, records)
    outputs = render_charts(data, args.output_dir)
    outputs.append(write_chart_data(data, args.output_dir / "chart-data.csv"))
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
