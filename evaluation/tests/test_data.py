from pathlib import Path

import pytest

from evaluation.config import EvalPaths, configure_external_caches
from evaluation.io import append_jsonl, read_jsonl


def test_eval_paths_rejects_c_drive():
    with pytest.raises(ValueError, match="D: drive"):
        EvalPaths(cache_root=Path("C:/AI_cache"))


def test_eval_paths_builds_external_layout():
    paths = EvalPaths(cache_root=Path("D:/AI_cache/test_eval"))
    assert paths.datasets == Path("D:/AI_cache/test_eval/datasets")
    assert paths.models == Path("D:/AI_cache/test_eval/models")
    assert paths.runs == Path("D:/AI_cache/test_eval/runs")


def test_external_cache_configuration_includes_model_caches():
    paths = EvalPaths(cache_root=Path("D:/AI_cache/test_eval"))
    values = configure_external_caches(paths)
    assert values["HF_HOME"] == "D:\\AI_cache\\test_eval\\huggingface"
    assert values["YOLO_CONFIG_DIR"] == "D:\\AI_cache\\test_eval\\ultralytics"


def test_append_jsonl_persists_each_completed_sample(tmp_path):
    output = tmp_path / "predictions.jsonl"
    append_jsonl(output, {"sample_id": "a", "model": "m", "status": "ok"})
    append_jsonl(output, {"sample_id": "b", "model": "m", "status": "error"})
    assert [record["sample_id"] for record in read_jsonl(output)] == ["a", "b"]
