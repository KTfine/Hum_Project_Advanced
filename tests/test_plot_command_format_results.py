from experiments.scripts.plot_command_format_results import build_chart_data


def test_chart_data_uses_frozen_token_and_reliability_metrics():
    metrics = {
        "formats": {
            "angle": {
                "mean_completion_tokens": 10.0, "median_completion_tokens": 8.0,
                "strict_parse_rate": 0.9, "semantic_exact_rate": 0.8,
                "tokens_per_correct_command": 7.0,
            },
            "json": {
                "mean_completion_tokens": 20.0, "median_completion_tokens": 18.0,
                "strict_parse_rate": 1.0, "semantic_exact_rate": 0.8,
                "tokens_per_correct_command": 14.0,
            },
        },
        "paired_correct": {
            "mean_angle_tokens": 9.0, "mean_json_tokens": 19.0,
            "token_savings_rate": 0.4, "json_minus_angle_ci95": [8.0, 12.0],
        },
    }
    records = [
        {"category": "single", "format": "angle", "completion_tokens": 4},
        {"category": "single", "format": "json", "completion_tokens": 10},
        {"category": "speech", "format": "angle", "completion_tokens": 8},
        {"category": "speech", "format": "json", "completion_tokens": 16},
    ]

    data = build_chart_data(metrics, records)

    assert data["token_groups"]["angle"] == [10.0, 8.0, 9.0]
    assert data["token_groups"]["json"] == [20.0, 18.0, 19.0]
    assert data["category_labels"] == ["Single", "Speech"]
    assert data["category_tokens"]["angle"] == [4.0, 8.0]
    assert data["reliability"]["angle"] == [0.9, 0.8]
    assert data["savings_rate"] == 0.4
