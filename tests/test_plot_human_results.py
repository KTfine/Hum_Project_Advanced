from experiments.scripts.plot_human_results import build_chart_data


def test_chart_data_uses_accuracy_and_only_pck_at_020_for_pose():
    penn = {"models": {
        "deepseek-flash": {
            "activity": {"accuracy": 0.9},
            "pose": {"pck_at_0_05": 0.1, "pck_at_0_10": 0.2, "pck_at_0_20": 0.6},
        },
        "xclip-large-patch14": {"activity": {"accuracy": 0.7}},
        "xclip-base-patch32": {"activity": {"accuracy": 0.6}},
        **{
            f"yolo11{size}-pose": {"pose": {"pck_at_0_20": 0.7}}
            for size in "xlmsn"
        },
    }}
    ravdess = {"models": {
        "deepseek-flash": {"accuracy": 0.6},
        "hsemotion-enet-b2-8": {"accuracy": 0.5},
        "hsemotion-enet-b0-8-afew": {"accuracy": 0.4},
    }}

    charts = build_chart_data(penn, ravdess)

    assert [chart["metric"] for chart in charts] == ["Accuracy", "PCK@0.20", "Accuracy"]
    assert charts[1]["values"] == [0.6, 0.7, 0.7, 0.7, 0.7, 0.7]
