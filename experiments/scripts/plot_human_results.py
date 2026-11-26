"""Create publication-ready charts from the frozen formal evaluation metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEEPSEEK_COLOR = "#E76F51"
SPECIALIST_COLOR = "#457B9D"


def build_chart_data(penn: dict, ravdess: dict) -> list[dict]:
    penn_models = penn["models"]
    emotion_models = ravdess["models"]
    return [
        {
            "slug": "activity-accuracy",
            "title": "Activity Classification — Penn Action (n=60)",
            "short_title": "Activity\nPenn Action (n=60)",
            "task": "Activity",
            "metric": "Accuracy",
            "labels": ["DeepSeek", "X-CLIP Large", "X-CLIP Base"],
            "model_ids": ["deepseek-flash", "xclip-large-patch14", "xclip-base-patch32"],
            "values": [
                penn_models["deepseek-flash"]["activity"]["accuracy"],
                penn_models["xclip-large-patch14"]["activity"]["accuracy"],
                penn_models["xclip-base-patch32"]["activity"]["accuracy"],
            ],
        },
        {
            "slug": "pose-pck-at-020",
            "title": "Pose Estimation — Penn Action (n=60)",
            "short_title": "Pose PCK@0.20\nPenn Action (n=60)",
            "task": "Pose",
            "metric": "PCK@0.20",
            "labels": ["DeepSeek", "YOLO11x", "YOLO11l", "YOLO11m", "YOLO11s", "YOLO11n"],
            "model_ids": ["deepseek-flash", *[f"yolo11{size}-pose" for size in "xlmsn"]],
            "values": [
                penn_models["deepseek-flash"]["pose"]["pck_at_0_20"],
                *[
                    penn_models[f"yolo11{size}-pose"]["pose"]["pck_at_0_20"]
                    for size in "xlmsn"
                ],
            ],
        },
        {
            "slug": "emotion-accuracy",
            "title": "Emotion Classification — RAVDESS (n=56)",
            "short_title": "Emotion\nRAVDESS (n=56)",
            "task": "Emotion",
            "metric": "Accuracy",
            "labels": ["DeepSeek", "HSEmotion B2", "HSEmotion B0"],
            "model_ids": ["deepseek-flash", "hsemotion-enet-b2-8", "hsemotion-enet-b0-8-afew"],
            "values": [
                emotion_models["deepseek-flash"]["accuracy"],
                emotion_models["hsemotion-enet-b2-8"]["accuracy"],
                emotion_models["hsemotion-enet-b0-8-afew"]["accuracy"],
            ],
        },
    ]


def _style_axis(
    ax, chart: dict, *, title: str | None = None, title_size: int = 26,
    tick_size: int = 21, value_size: int = 21, stagger_values: bool = False,
) -> None:
    colors = [DEEPSEEK_COLOR] + [SPECIALIST_COLOR] * (len(chart["values"]) - 1)
    bars = ax.bar(chart["labels"], chart["values"], color=colors, width=0.42)
    resolved_title = chart["title"] if title is None else title
    ax.set_title(resolved_title, fontsize=title_size, fontweight="bold", pad=14)
    ax.set_ylabel(chart["metric"], fontsize=22)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([index / 10 for index in range(0, 11, 2)])
    ax.grid(axis="y", color="#D8DEE4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=15, labelsize=tick_size, pad=6)
    ax.tick_params(axis="y", labelsize=tick_size)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    for index, (bar, value) in enumerate(zip(bars, chart["values"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025 + (0.035 if stagger_values and index % 2 else 0),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=value_size,
            fontweight="bold",
        )


def render_charts(charts: list[dict], output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.edgecolor": "#6B7280",
        "axes.labelcolor": "#24292F",
        "text.color": "#24292F",
        "xtick.color": "#3D444D",
        "ytick.color": "#3D444D",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    legend = [
        Patch(facecolor=DEEPSEEK_COLOR, label="DeepSeek"),
        Patch(facecolor=SPECIALIST_COLOR, label="Specialist baseline"),
    ]
    outputs = []
    for chart in charts:
        figure, axis = plt.subplots(figsize=(8.4, 6.2))
        _style_axis(
            axis, chart, title="",
            stagger_values=chart["task"] == "Pose",
        )
        figure.suptitle(chart["title"], fontsize=25, fontweight="bold", y=0.98)
        figure.legend(
            handles=legend, frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, 0.90), ncol=2, fontsize=18,
        )
        figure.subplots_adjust(top=0.77, bottom=0.22, left=0.14, right=0.98)
        for extension in ("png", "pdf"):
            path = output_dir / f"{chart['slug']}.{extension}"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
    for axis, chart in zip(axes, charts):
        _style_axis(
            axis, chart, title=chart["short_title"], title_size=22,
            tick_size=17, value_size=17,
            stagger_values=chart["task"] == "Pose",
        )
    figure.legend(handles=legend, frameon=False, loc="upper center", ncol=2, fontsize=20)
    figure.subplots_adjust(top=0.72, bottom=0.25, left=0.06, right=0.995, wspace=0.23)
    for extension in ("png", "pdf"):
        path = output_dir / f"human-understanding-summary.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def write_chart_data(charts: list[dict], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task", "metric", "model", "display_name", "value"))
        writer.writeheader()
        for chart in charts:
            for model, label, value in zip(chart["model_ids"], chart["labels"], chart["values"]):
                writer.writerow({
                    "task": chart["task"], "metric": chart["metric"],
                    "model": model, "display_name": label, "value": f"{value:.12g}",
                })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path(r"D:\AI_cache\hum_project_evaluation"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/figures/2026-11-26-human-understanding"),
    )
    args = parser.parse_args(argv)
    runs = args.cache_root / "runs"
    penn = json.loads((runs / "20261126-penn-formal" / "metrics.json").read_text(encoding="utf-8"))
    ravdess = json.loads((runs / "20261126-ravdess-formal" / "metrics.json").read_text(encoding="utf-8"))
    charts = build_chart_data(penn, ravdess)
    outputs = render_charts(charts, args.output_dir)
    outputs.append(write_chart_data(charts, args.output_dir / "chart-data.csv"))
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
