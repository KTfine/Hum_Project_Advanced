from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from evaluation.config import DEFAULT_CACHE_ROOT, EvalPaths, configure_external_caches


def _paths(value: str) -> EvalPaths:
    return EvalPaths(Path(value)).ensure()


def doctor(paths: EvalPaths) -> dict:
    result = {
        "python": sys.version,
        "cache_root": str(paths.cache_root),
        "cache_drive": paths.cache_root.drive,
        "deepseek_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "packages": {
            name: bool(importlib.util.find_spec(name))
            for name in ("torch", "transformers", "ultralytics", "emotiefflib", "onnxruntime")
        },
    }
    try:
        from photo_assistant.api.client import configured_api_key
        result["deepseek_key_present"] = bool(configured_api_key())
    except Exception:
        pass
    if importlib.util.find_spec("torch"):
        import torch
        result.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Penn Action and RAVDESS automatic evaluation",
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")

    prepare_penn = subparsers.add_parser("prepare-penn")
    prepare_penn.add_argument("--count-per-class", type=int, default=4, choices=range(1, 7))
    prepare_penn.add_argument("--seed", type=int, default=3800)

    prepare_ravdess = subparsers.add_parser("prepare-ravdess")
    prepare_ravdess.add_argument("--count-per-class", type=int, default=8, choices=range(1, 9))
    prepare_ravdess.add_argument("--seed", type=int, default=3800)

    run_penn = subparsers.add_parser("run-penn")
    run_penn.add_argument("--manifest", type=Path, required=True)
    run_penn.add_argument("--run-id", required=True)
    run_penn.add_argument("--no-deepseek", action="store_true")

    run_ravdess = subparsers.add_parser("run-ravdess")
    run_ravdess.add_argument("--manifest", type=Path, required=True)
    run_ravdess.add_argument("--run-id", required=True)
    run_ravdess.add_argument("--no-deepseek", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _paths(args.cache_root)
    configure_external_caches(paths)
    if args.command == "doctor":
        print(json.dumps(doctor(paths), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-penn":
        from evaluation.data.penn_action import prepare_penn_action
        print(prepare_penn_action(
            paths.datasets / "penn-action",
            count_per_class=args.count_per_class,
            seed=args.seed,
        ))
        return 0
    if args.command == "prepare-ravdess":
        from evaluation.data.ravdess import prepare_ravdess
        print(prepare_ravdess(
            paths.datasets / "ravdess",
            count_per_class=args.count_per_class,
            seed=args.seed,
        ))
        return 0
    if args.command == "run-penn":
        from evaluation.formal import run_penn_formal
        print(run_penn_formal(
            paths, args.manifest, run_id=args.run_id,
            use_deepseek=not args.no_deepseek,
        ))
        return 0
    if args.command == "run-ravdess":
        from evaluation.formal import run_ravdess_formal
        print(run_ravdess_formal(
            paths, args.manifest, run_id=args.run_id,
            use_deepseek=not args.no_deepseek,
        ))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
