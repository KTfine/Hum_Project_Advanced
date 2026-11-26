"""Paired DeepSeek experiment for the native angle-tag and JSON command formats."""
from __future__ import annotations

from collections import defaultdict
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import time
from typing import Any

import numpy as np

from photo_assistant.protocol import StreamingCommandParser
from photo_assistant.api.client import ApiError, DeepSeekClient


ANGLE_SYSTEM_PROMPT = """You are a command generator. Return only native angle-bracket commands, with no prose or Markdown. Output exactly the requested commands; never add <blank> unless the user explicitly requests no response and no action.
Available forms:
<capture> <follow> <peek> <stand> <blank> <sit> <brace>
<speak:内容> where the speech follows the colon directly; never write text= or quotation marks
<teleport:x=number,y=number> with optional scale
<sit:distance=near|far> or <sit:x=number,y=number>
<brace:x=number>
<rotate:to|by,value=number,ms=integer>
<layer:to,index=integer>
<tachie:face=value,hand=value,arm=value,torso=value,legs=value>
tachie values must be copied exactly without translation:
face=微笑|龇牙笑|高冷表情|害羞|惊讶|思考|难过|鼓脸小生气;
hand=比耶|无动作; arm=自然下垂|伸直举高过头|背在身后|左手扶墙|右手扶墙;
torso=正着站|半身歪斜向右; legs=站着|坐着|蹲下.
When several commands are requested, concatenate them in the requested order without spaces. Preserve requested speech exactly."""

JSON_SYSTEM_PROMPT = """You are a command generator. Return only one minified JSON object, with no prose or Markdown. Output exactly the requested commands; never add a blank command unless the user explicitly requests no response and no action.
The top level is {"commands":[...]}. Each command has a "type" field and its direct parameters:
capture, follow, peek, stand, blank, sit, brace take no other fields when unparameterized;
speak uses "text":"内容";
teleport uses x, y and optional scale;
sit uses distance, or x and y;
brace uses x;
rotate uses mode="to" or "by", value and ms;
layer uses index;
tachie uses face, hand, arm, torso and legs.
tachie values must be copied exactly without translation:
face=微笑|龇牙笑|高冷表情|害羞|惊讶|思考|难过|鼓脸小生气;
hand=比耶|无动作; arm=自然下垂|伸直举高过头|背在身后|左手扶墙|右手扶墙;
torso=正着站|半身歪斜向右; legs=站着|坐着|蹲下.
When several commands are requested, keep them in the requested array order. Preserve requested speech exactly. Do not add explanatory fields."""


def _command(kind: str, **params: Any) -> dict:
    return {"kind": kind, "params": params}


def scenario_manifest() -> list[dict]:
    pose = lambda face, hand, arm, torso, legs: _command(
        "tachie", face=face, hand=hand, arm=arm, torso=torso, legs=legs,
    )
    return [
        {"scenario_id": "single-01", "category": "single", "user_prompt": "只执行拍照，不要说话。", "expected_commands": [_command("capture")]},
        {"scenario_id": "single-02", "category": "single", "user_prompt": "只跟到我身边，不要说话。", "expected_commands": [_command("follow")]},
        {"scenario_id": "single-03", "category": "single", "user_prompt": "只从我的肩后探出脸，不要说话。", "expected_commands": [_command("peek")]},
        {"scenario_id": "single-04", "category": "single", "user_prompt": "只站直，不要说话。", "expected_commands": [_command("stand")]},
        {"scenario_id": "single-05", "category": "single", "user_prompt": "只就近坐下，不要说话。", "expected_commands": [_command("sit")]},
        {"scenario_id": "single-06", "category": "single", "user_prompt": "无需回应，也不要执行动作。", "expected_commands": [_command("blank")]},
        {"scenario_id": "param-01", "category": "parameterized", "user_prompt": "只移动到归一化坐标 x=0.35、y=0.85，不要说话。", "expected_commands": [_command("teleport", x=0.35, y=0.85)]},
        {"scenario_id": "param-02", "category": "parameterized", "user_prompt": "只移动到 x=0.72、y=0.90，并把大小倍率设为 1.2，不要说话。", "expected_commands": [_command("teleport", x=0.72, y=0.9, scale=1.2)]},
        {"scenario_id": "param-03", "category": "parameterized", "user_prompt": "只选择离镜头最远的可用座位，不要说话。", "expected_commands": [_command("sit", distance="far")]},
        {"scenario_id": "param-04", "category": "parameterized", "user_prompt": "只坐到归一化坐标 x=0.4、y=0.6 指定的坐面，不要说话。", "expected_commands": [_command("sit", x=0.4, y=0.6)]},
        {"scenario_id": "param-05", "category": "parameterized", "user_prompt": "只在画面横坐标 x=0.25 的位置扶墙，不要说话。", "expected_commands": [_command("brace", x=0.25)]},
        {"scenario_id": "param-06", "category": "parameterized", "user_prompt": "只用 500 毫秒旋转到 45 度，不要说话。", "expected_commands": [_command("rotate", mode="to", ms=500, value=45.0)]},
        {"scenario_id": "speech-01", "category": "speech", "user_prompt": "只说“看镜头。”，不要执行其他动作。", "expected_commands": [_command("speak", text="看镜头。")]},
        {"scenario_id": "speech-02", "category": "speech", "user_prompt": "只说“放松肩膀。”，不要执行其他动作。", "expected_commands": [_command("speak", text="放松肩膀。")]},
        {"scenario_id": "speech-03", "category": "speech", "user_prompt": "只说“准备好了吗？”，不要执行其他动作。", "expected_commands": [_command("speak", text="准备好了吗？")]},
        {"scenario_id": "speech-04", "category": "speech", "user_prompt": "只说“向左一点，再抬头。”，不要执行其他动作。", "expected_commands": [_command("speak", text="向左一点，再抬头。")]},
        {"scenario_id": "speech-05", "category": "speech", "user_prompt": "只说“Three, two, one.”，不要执行其他动作。", "expected_commands": [_command("speak", text="Three, two, one.")]},
        {"scenario_id": "speech-06", "category": "speech", "user_prompt": "只说“好，我们拍吧！”，不要执行其他动作。", "expected_commands": [_command("speak", text="好，我们拍吧！")]},
        {"scenario_id": "tachie-01", "category": "tachie", "user_prompt": "只切换为：微笑、比耶、自然下垂、正着站、站着。", "expected_commands": [pose("微笑", "比耶", "自然下垂", "正着站", "站着")]},
        {"scenario_id": "tachie-02", "category": "tachie", "user_prompt": "只切换为：害羞、无动作、背在身后、正着站、站着。", "expected_commands": [pose("害羞", "无动作", "背在身后", "正着站", "站着")]},
        {"scenario_id": "tachie-03", "category": "tachie", "user_prompt": "只切换为：惊讶、无动作、伸直举高过头、正着站、站着。", "expected_commands": [pose("惊讶", "无动作", "伸直举高过头", "正着站", "站着")]},
        {"scenario_id": "tachie-04", "category": "tachie", "user_prompt": "只切换为：思考、无动作、自然下垂、半身歪斜向右、站着。", "expected_commands": [pose("思考", "无动作", "自然下垂", "半身歪斜向右", "站着")]},
        {"scenario_id": "tachie-05", "category": "tachie", "user_prompt": "只切换为：难过、无动作、自然下垂、正着站、坐着。", "expected_commands": [pose("难过", "无动作", "自然下垂", "正着站", "坐着")]},
        {"scenario_id": "tachie-06", "category": "tachie", "user_prompt": "只切换为：鼓脸小生气、无动作、右手扶墙、正着站、站着。", "expected_commands": [pose("鼓脸小生气", "无动作", "右手扶墙", "正着站", "站着")]},
        {"scenario_id": "multi-01", "category": "multi", "user_prompt": "依次站直、跟到我身边、拍照，不要说话。", "expected_commands": [_command("stand"), _command("follow"), _command("capture")]},
        {"scenario_id": "multi-02", "category": "multi", "user_prompt": "先移动到 x=0.7、y=0.9，再说“到啦。”，严格按此顺序。", "expected_commands": [_command("teleport", x=0.7, y=0.9), _command("speak", text="到啦。")]},
        {"scenario_id": "multi-03", "category": "multi", "user_prompt": "先说“请看这里。”，再拍照，严格按此顺序。", "expected_commands": [_command("speak", text="请看这里。"), _command("capture")]},
        {"scenario_id": "multi-04", "category": "multi", "user_prompt": "依次从肩后探头、说“这样可以吗？”，不要做其他事。", "expected_commands": [_command("peek"), _command("speak", text="这样可以吗？")]},
        {"scenario_id": "multi-05", "category": "multi", "user_prompt": "依次移动到 x=0.2、y=0.8，把图层设为 index=1，然后拍照。", "expected_commands": [_command("teleport", x=0.2, y=0.8), _command("layer", index=1), _command("capture")]},
        {"scenario_id": "multi-06", "category": "multi", "user_prompt": "依次站直、移动到 x=0.5、y=0.88、说“完成。”，不要做其他事。", "expected_commands": [_command("stand"), _command("teleport", x=0.5, y=0.88), _command("speak", text="完成。")]},
    ]


def _normalize(commands: list) -> list[dict]:
    return [{"kind": item.kind, "params": item.params} for item in commands]


def parse_angle_output(text: str) -> list[dict]:
    value = text.strip()
    tags = re.findall(r"<[^<>]*>", value)
    if not value or "".join(tags) != value:
        raise ValueError("angle output must contain native tags only")
    parser = StreamingCommandParser()
    commands = parser.feed("comparison", value)
    parser.flush("comparison")
    if len(commands) != len(tags):
        raise ValueError("one or more angle tags are invalid")
    return _normalize(commands)


def _json_command_to_angle(value: dict) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ValueError("each JSON command must be an object with string type")
    kind = value["type"]
    params = {key: item for key, item in value.items() if key != "type"}
    no_params = {"capture", "follow", "peek", "stand", "blank"}
    if kind in no_params:
        if params:
            raise ValueError(f"{kind} accepts no JSON fields")
        return f"<{kind}>"
    if kind in {"sit", "brace"} and not params:
        return f"<{kind}>"
    if kind == "speak":
        if set(params) != {"text"} or not isinstance(params["text"], str):
            raise ValueError("speak requires only text")
        return f"<speak:{params['text']}>"
    allowed = {
        "teleport": {"x", "y", "scale"},
        "sit": {"distance", "x", "y"},
        "brace": {"x"},
        "rotate": {"mode", "value", "ms"},
        "layer": {"index"},
        "tachie": {"face", "hand", "arm", "torso", "legs"},
    }
    if kind not in allowed or not params or not set(params) <= allowed[kind]:
        raise ValueError(f"unsupported or extra JSON fields for {kind}")
    if kind == "rotate":
        mode = params.pop("mode", None)
        if mode not in {"to", "by"}:
            raise ValueError("rotate requires mode")
        prefix = f"<rotate:{mode}"
    elif kind == "layer":
        prefix = "<layer:to"
    else:
        prefix = f"<{kind}:"
    body = ",".join(f"{key}={params[key]}" for key in params)
    separator = "," if kind in {"rotate", "layer"} else ""
    return f"{prefix}{separator}{body}>"


def parse_json_output(text: str) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("output is not one JSON document") from exc
    if not isinstance(value, dict) or set(value) != {"commands"}:
        raise ValueError("JSON top level must contain only commands")
    commands = value["commands"]
    if not isinstance(commands, list) or not commands:
        raise ValueError("commands must be a non-empty list")
    tags = "".join(_json_command_to_angle(dict(item)) for item in commands)
    return parse_angle_output(tags)


def completion_tokens(metadata: dict) -> int:
    value = metadata.get("usage", {}).get("completion_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("provider completion_tokens missing")
    return value


def format_order(index: int) -> tuple[str, str]:
    return ("angle", "json") if index % 2 == 0 else ("json", "angle")


def build_messages(format_name: str, user_prompt: str) -> list[dict]:
    prompts = {"angle": ANGLE_SYSTEM_PROMPT, "json": JSON_SYSTEM_PROMPT}
    if format_name not in prompts:
        raise ValueError(f"unknown command format: {format_name}")
    return [
        {"role": "system", "content": prompts[format_name]},
        {"role": "user", "content": user_prompt},
    ]


def record_matches(record: dict, scenario: dict, format_name: str, model: str) -> bool:
    return (
        record.get("scenario_id") == scenario.get("scenario_id")
        and record.get("format") == format_name
        and record.get("model") == model
        and record.get("expected_commands") == scenario.get("expected_commands")
    )


def semantic_equal(actual: list[dict], expected: list[dict]) -> bool:
    def canonical(commands: list[dict]) -> list[dict]:
        normalized = []
        for command in commands:
            item = {"kind": command["kind"], "params": dict(command["params"])}
            if item == {"kind": "sit", "params": {}}:
                item["params"] = {"distance": "near"}
            normalized.append(item)
        return normalized
    return canonical(actual) == canonical(expected)


def _bootstrap(values: list[float], iterations: int, seed: int) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    sampled = generator.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def score_records(records: list[dict], *, bootstrap_iterations: int = 10_000, seed: int = 3800) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        grouped[record["format"]].append(record)
        by_pair[record["scenario_id"]][record["format"]] = record
    formats = {}
    for name in ("angle", "json"):
        rows = grouped[name]
        observed = [row["completion_tokens"] for row in rows if row.get("completion_tokens") is not None]
        exact = [row for row in rows if row.get("semantic_exact")]
        correct_commands = sum(len(row["expected_commands"]) for row in exact)
        total_tokens = sum(observed)
        formats[name] = {
            "sample_count": len(rows),
            "token_observed_count": len(observed),
            "mean_completion_tokens": mean(observed) if observed else None,
            "median_completion_tokens": median(observed) if observed else None,
            "strict_parse_rate": sum(bool(row.get("parse_ok")) for row in rows) / len(rows) if rows else 0.0,
            "semantic_exact_rate": len(exact) / len(rows) if rows else 0.0,
            "correct_command_count": correct_commands,
            "tokens_per_correct_command": total_tokens / correct_commands if correct_commands else None,
            "correct_commands_per_100_tokens": 100 * correct_commands / total_tokens if total_tokens else 0.0,
        }
    paired = []
    for values in by_pair.values():
        if set(values) == {"angle", "json"} and all(values[name].get("semantic_exact") for name in values):
            angle = values["angle"].get("completion_tokens")
            json_tokens = values["json"].get("completion_tokens")
            if angle is not None and json_tokens is not None:
                paired.append((angle, json_tokens))
    angle_tokens = [item[0] for item in paired]
    json_tokens = [item[1] for item in paired]
    differences = [right - left for left, right in paired]
    mean_angle = mean(angle_tokens) if paired else None
    mean_json = mean(json_tokens) if paired else None
    return {
        "schema_version": 1,
        "formats": formats,
        "paired_correct": {
            "pair_count": len(paired),
            "mean_angle_tokens": mean_angle,
            "mean_json_tokens": mean_json,
            "mean_json_minus_angle_tokens": mean(differences) if paired else None,
            "json_minus_angle_ci95": _bootstrap(differences, bootstrap_iterations, seed),
            "token_savings_rate": 1 - mean_angle / mean_json if paired and mean_json else None,
            "bootstrap_iterations": bootstrap_iterations,
            "seed": seed,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _request_record(
    client: DeepSeekClient, scenario: dict, format_name: str, *, model: str,
    retries: int = 2,
) -> dict:
    record = {
        "scenario_id": scenario["scenario_id"], "category": scenario["category"],
        "format": format_name, "model": model,
        "expected_commands": scenario["expected_commands"],
    }
    attempts = []
    started = time.perf_counter()
    result = None
    for attempt in range(retries + 1):
        try:
            result = await client.complete(build_messages(format_name, scenario["user_prompt"]))
            break
        except ApiError as exc:
            attempts.append(exc.result.metadata)
            if attempt == retries:
                record.update(
                    status="error", reason=str(exc), attempts=attempts,
                    parse_ok=False, semantic_exact=False,
                    completion_tokens=None, elapsed_s=time.perf_counter() - started,
                )
                return record
            await asyncio.sleep(0.5 * (attempt + 1))
    assert result is not None
    record.update(
        status="ok", raw_text=result.text, metadata=result.metadata,
        attempts=attempts, elapsed_s=time.perf_counter() - started,
    )
    try:
        record["completion_tokens"] = completion_tokens(result.metadata)
    except ValueError as exc:
        record["completion_tokens"] = None
        record["token_count_error"] = str(exc)
    parser = parse_angle_output if format_name == "angle" else parse_json_output
    try:
        parsed = parser(result.text)
        record.update(
            parse_ok=True, parsed_commands=parsed,
            semantic_exact=semantic_equal(parsed, scenario["expected_commands"]),
        )
    except ValueError as exc:
        record.update(parse_ok=False, semantic_exact=False, parse_error=str(exc))
    return record


def _load_records(path: Path, scenarios: list[dict], model: str) -> tuple[list[dict], set[tuple[str, str]]]:
    if not path.is_file():
        return [], set()
    scenario_by_id = {item["scenario_id"]: item for item in scenarios}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = set()
    for record in records:
        key = (record.get("scenario_id"), record.get("format"))
        scenario = scenario_by_id.get(key[0])
        if key in keys or scenario is None or not record_matches(record, scenario, key[1], model):
            raise ValueError(f"existing prediction identity mismatch: {key}")
        keys.add(key)
    return records, keys


def _report(metrics: dict) -> str:
    angle, json_metrics = metrics["formats"]["angle"], metrics["formats"]["json"]
    paired = metrics["paired_correct"]
    return f"""# Command format comparison

All semantic checks are automatic and exact. Failures remain in the reliability denominator.

| Format | Mean completion tokens | Strict parse rate | Semantic exact rate | Tokens/correct command |
|---|---:|---:|---:|---:|
| Angle tags | {angle['mean_completion_tokens']:.3f} | {angle['strict_parse_rate']:.3f} | {angle['semantic_exact_rate']:.3f} | {angle['tokens_per_correct_command']:.3f} |
| JSON | {json_metrics['mean_completion_tokens']:.3f} | {json_metrics['strict_parse_rate']:.3f} | {json_metrics['semantic_exact_rate']:.3f} | {json_metrics['tokens_per_correct_command']:.3f} |

Semantically correct pairs: {paired['pair_count']}. On these pairs, angle tags used {paired['mean_angle_tokens']:.3f} tokens and JSON used {paired['mean_json_tokens']:.3f}; savings rate: {paired['token_savings_rate']:.3%}.
"""


async def run_experiment(run_dir: Path, *, model: str = "deepseek-flash") -> Path:
    # Python 3.10 compatibility used by the evaluation environment.
    if not hasattr(asyncio, "timeout"):
        import async_timeout
        asyncio.timeout = async_timeout.timeout  # type: ignore[attr-defined]
    run_dir.mkdir(parents=True, exist_ok=True)
    scenarios = scenario_manifest()
    manifest_path = run_dir / "manifest.json"
    frozen_manifest = {"schema_version": 1, "model": model, "scenarios": scenarios}
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != frozen_manifest:
            raise ValueError("existing manifest does not match frozen scenarios")
    else:
        _atomic_json(manifest_path, frozen_manifest)
    predictions_path = run_dir / "predictions.jsonl"
    records, completed = _load_records(predictions_path, scenarios, model)
    async with DeepSeekClient(model=model, thinking="off", timeout=180, max_tokens=512) as client:
        for index, scenario in enumerate(scenarios):
            for format_name in format_order(index):
                key = (scenario["scenario_id"], format_name)
                if key in completed:
                    continue
                record = await _request_record(client, scenario, format_name, model=model)
                with predictions_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                records.append(record)
                completed.add(key)
    metrics = score_records(records)
    metrics.update(
        dataset="30 frozen paired command requests",
        scoring_policy="strict automatic parse and exact ordered command equality; all failures retained",
        manifest_sha256=_sha256(manifest_path),
        predictions_sha256=_sha256(predictions_path),
    )
    _atomic_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics), encoding="utf-8")
    _atomic_json(run_dir / "run_config.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "thinking": "off", "max_tokens": 512,
        "angle_system_sha256": hashlib.sha256(ANGLE_SYSTEM_PROMPT.encode()).hexdigest(),
        "json_system_sha256": hashlib.sha256(JSON_SYSTEM_PROMPT.encode()).hexdigest(),
        "manifest_sha256": metrics["manifest_sha256"],
        "predictions_sha256": metrics["predictions_sha256"],
    })
    return run_dir / "metrics.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path(r"D:\AI_cache\hum_project_evaluation\runs\20261126-command-format-formal"),
    )
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args(argv)
    print(asyncio.run(run_experiment(args.run_dir, model=args.model)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
