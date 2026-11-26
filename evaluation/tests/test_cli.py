from pathlib import Path

from evaluation.cli import build_parser


def test_cli_exposes_only_current_formal_evaluation_commands():
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers.choices) == {
        "doctor", "prepare-penn", "prepare-ravdess", "run-penn", "run-ravdess",
    }


def test_prepare_commands_keep_frozen_sample_counts_and_seed():
    penn = build_parser().parse_args([
        "prepare-penn", "--count-per-class", "4", "--seed", "3800",
    ])
    ravdess = build_parser().parse_args([
        "prepare-ravdess", "--count-per-class", "8", "--seed", "3800",
    ])
    assert (penn.count_per_class, penn.seed) == (4, 3800)
    assert (ravdess.count_per_class, ravdess.seed) == (8, 3800)


def test_formal_run_commands_accept_frozen_manifests():
    penn = build_parser().parse_args([
        "run-penn", "--manifest", "penn.json", "--run-id", "formal-penn",
    ])
    ravdess = build_parser().parse_args([
        "run-ravdess", "--manifest", "ravdess.json", "--run-id", "formal-ravdess",
    ])
    assert penn.manifest == Path("penn.json")
    assert ravdess.manifest == Path("ravdess.json")
