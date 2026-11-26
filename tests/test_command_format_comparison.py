import json

import pytest

from experiments.scripts.command_format_comparison import (
    build_messages,
    completion_tokens,
    format_order,
    parse_angle_output,
    parse_json_output,
    record_matches,
    scenario_manifest,
    semantic_equal,
    score_records,
)


def test_manifest_has_thirty_unique_balanced_scenarios():
    scenarios = scenario_manifest()
    assert len(scenarios) == 30
    assert len({item["scenario_id"] for item in scenarios}) == 30
    assert {item["category"] for item in scenarios} == {
        "single", "parameterized", "speech", "tachie", "multi",
    }
    assert all(item["expected_commands"] for item in scenarios)


def test_angle_parser_requires_only_valid_native_tags_and_preserves_order():
    value = "<teleport:x=0.35,y=0.85><speak:来吧。><capture>"
    assert parse_angle_output(value) == [
        {"kind": "teleport", "params": {"x": 0.35, "y": 0.85}},
        {"kind": "speak", "params": {"text": "来吧。"}},
        {"kind": "capture", "params": {}},
    ]
    for invalid in (
        "好的。<capture>",
        "```<capture>```",
        "<teleport:x=0.35>",
        "<capture><unknown>",
    ):
        with pytest.raises(ValueError):
            parse_angle_output(invalid)


def test_json_parser_uses_equivalent_compact_schema_and_rejects_extra_fields():
    value = json.dumps({"commands": [
        {"type": "teleport", "x": 0.35, "y": 0.85},
        {"type": "speak", "text": "来吧。"},
        {"type": "capture"},
    ]}, ensure_ascii=False)
    assert parse_json_output(value) == [
        {"kind": "teleport", "params": {"x": 0.35, "y": 0.85}},
        {"kind": "speak", "params": {"text": "来吧。"}},
        {"kind": "capture", "params": {}},
    ]
    with pytest.raises(ValueError):
        parse_json_output('{"commands":[{"type":"capture","comment":"now"}]}')
    with pytest.raises(ValueError):
        parse_json_output('result: {"commands":[{"type":"capture"}]}')


def test_completion_tokens_reads_provider_usage_without_estimating():
    assert completion_tokens({"usage": {"completion_tokens": 17}}) == 17
    with pytest.raises(ValueError):
        completion_tokens({})


def test_call_order_alternates_by_scenario_index():
    assert format_order(0) == ("angle", "json")
    assert format_order(1) == ("json", "angle")


def test_format_prompts_request_the_same_command_vocabulary_without_fences():
    angle = build_messages("angle", "只拍照。")
    json_messages = build_messages("json", "只拍照。")
    assert angle[-1] == json_messages[-1] == {"role": "user", "content": "只拍照。"}
    assert "capture" in angle[0]["content"] and "capture" in json_messages[0]["content"]
    assert "Markdown" in angle[0]["content"] and "Markdown" in json_messages[0]["content"]
    assert "<speak:内容>" in angle[0]["content"]
    assert '"text":"内容"' in json_messages[0]["content"]
    for value in ("微笑", "无动作", "自然下垂", "正着站", "站着"):
        assert value in angle[0]["content"] and value in json_messages[0]["content"]


def test_semantic_comparison_treats_documented_near_sit_alias_as_equal():
    assert semantic_equal(
        [{"kind": "sit", "params": {"distance": "near"}}],
        [{"kind": "sit", "params": {}}],
    )
    assert not semantic_equal(
        [{"kind": "sit", "params": {"distance": "far"}}],
        [{"kind": "sit", "params": {}}],
    )


def test_resumable_record_requires_matching_frozen_identity():
    scenario = scenario_manifest()[0]
    record = {
        "scenario_id": scenario["scenario_id"], "format": "angle",
        "model": "deepseek-flash", "expected_commands": scenario["expected_commands"],
    }
    assert record_matches(record, scenario, "angle", "deepseek-flash")
    assert not record_matches(record, scenario, "json", "deepseek-flash")
    changed = dict(scenario, expected_commands=[{"kind": "blank", "params": {}}])
    assert not record_matches(record, changed, "angle", "deepseek-flash")


def test_scoring_keeps_failures_and_only_compares_semantically_correct_pairs():
    expected = [{"kind": "capture", "params": {}}]
    records = [
        {"scenario_id": "s1", "format": "angle", "status": "ok", "parse_ok": True,
         "semantic_exact": True, "completion_tokens": 2, "expected_commands": expected},
        {"scenario_id": "s1", "format": "json", "status": "ok", "parse_ok": True,
         "semantic_exact": True, "completion_tokens": 8, "expected_commands": expected},
        {"scenario_id": "s2", "format": "angle", "status": "ok", "parse_ok": True,
         "semantic_exact": False, "completion_tokens": 1, "expected_commands": expected},
        {"scenario_id": "s2", "format": "json", "status": "error", "parse_ok": False,
         "semantic_exact": False, "completion_tokens": 0, "expected_commands": expected},
    ]
    metrics = score_records(records, bootstrap_iterations=100, seed=3800)
    assert metrics["formats"]["angle"]["sample_count"] == 2
    assert metrics["formats"]["angle"]["strict_parse_rate"] == 1.0
    assert metrics["formats"]["angle"]["semantic_exact_rate"] == 0.5
    assert metrics["formats"]["json"]["strict_parse_rate"] == 0.5
    assert metrics["paired_correct"]["pair_count"] == 1
    assert metrics["paired_correct"]["mean_angle_tokens"] == 2
    assert metrics["paired_correct"]["mean_json_tokens"] == 8
    assert metrics["paired_correct"]["token_savings_rate"] == 0.75
