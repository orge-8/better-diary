"""v1.10.0 场景证据回归：证据回填、场景合并、事实校验与跨天防重。

样本全部为匿名合成对话，不复用真机聊天原文。
"""

import importlib
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
for candidate in (
    os.environ.get("MAIBOT_DEVKIT_DIR", ""),
    str(PLUGIN_DIR.parent / "tools" / "maibot-devkit"),
):
    if candidate and (Path(candidate) / "fakehost.py").is_file():
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

from fakehost import load_plugin_module  # noqa: E402

saved_path = list(sys.path)
PLUGIN_MOD = load_plugin_module(str(PLUGIN_DIR), module_name="better_diary_scene_test")
sys.path[:] = saved_path
SCENE = importlib.import_module(f"{PLUGIN_MOD.__name__}.bd_scene")
BD = importlib.import_module(f"{PLUGIN_MOD.__name__}.bd_prompts")

BASE = 1789507200.0


def _msg(offset, uid, nickname, text, group="g1"):
    return {
        "timestamp": str(BASE + offset),
        "processed_plain_text": text,
        "is_picture": False,
        "message_info": {
            "user_info": {"user_id": uid, "user_nickname": nickname},
            "group_info": {"group_id": group},
        },
    }


def _conversation(include_third=False):
    messages = [
        _msg(0, "u1", "阿岚", "做噩梦被吓醒了！"),
        _msg(30, "bot", "鸣澜", "醒了就好，我还在呢"),
        _msg(60, "u1", "阿岚", "你的手好热"),
        _msg(90, "bot", "鸣澜", "暖手宝自愿上岗"),
        _msg(120, "u1", "阿岚", "喜欢……"),
    ]
    if include_third:
        messages.insert(2, _msg(45, "u2", "旁人", "我也醒了"))
    return messages


def test_attach_evidence_keeps_quote_and_real_bot_reply():
    events = [{"who": "阿岚", "what": "做噩梦被吓醒找我哭诉", "quote": "做噩梦被吓醒了！", "score": 5}]
    enriched = SCENE.attach_evidence(
        events, _conversation(), bot_qq="bot", bot_labels={"我", "鸣澜"}
    )
    assert enriched[0]["evidence_ok"] is True
    assert enriched[0]["quote"] == "做噩梦被吓醒了！"
    assert any(turn["who"] == "我" and "我还在" in turn["text"] for turn in enriched[0]["evidence_turns"])
    rendered = BD.events_to_text(enriched, self_labels={"我", "鸣澜"})
    assert "我: 醒了就好，我还在呢" in rendered


def test_attach_evidence_marks_event_without_source_message():
    events = [{"who": "不存在的人", "what": "讲了一件完全无关的事", "quote": "没有这句", "score": 5}]
    enriched = SCENE.attach_evidence(events, _conversation(), bot_qq="bot")
    assert enriched[0]["evidence_ok"] is False
    assert enriched[0]["evidence_risks"]


def test_derived_time_claim_is_replaced_by_source_text():
    event = {
        "who": "小乙",
        "what": "00:00关门前13秒刷脸进门",
        "quote": "",
        "score": 4,
    }
    messages = [_msg(0, "u2", "小乙", "23:58刷脸进门，00:13才回宿舍", group="g2")]
    enriched = SCENE.attach_evidence([event], messages, bot_qq="bot")
    assert "13秒" not in enriched[0]["what"]
    assert "关门前13秒" not in enriched[0]["what"]
    assert enriched[0]["summary_rewritten"] is True


def test_same_person_continuous_fragments_merge_into_one_episode():
    messages = _conversation()
    events = [
        {"who": "阿岚", "what": "做噩梦被吓醒找我哭诉", "quote": "做噩梦被吓醒了！", "score": 5},
        {"who": "阿岚", "what": "她说喜欢我的手热", "quote": "喜欢……", "score": 4},
    ]
    enriched = SCENE.attach_evidence(events, messages, bot_qq="bot")
    merged = SCENE.merge_scene_fragments(enriched, messages, bot_qq="bot")
    assert len(merged) == 1
    assert merged[0]["fragment_count"] == 2
    selected = BD.select_events_for_write(merged, max_events=4, max_per_who=1)
    assert len(selected) == 1


def test_third_party_between_fragments_prevents_merge():
    messages = _conversation(include_third=True)
    events = [
        {"who": "阿岚", "what": "做噩梦被吓醒找我哭诉", "quote": "做噩梦被吓醒了！", "score": 5},
        {"who": "阿岚", "what": "她说喜欢我的手热", "quote": "喜欢……", "score": 4},
    ]
    enriched = SCENE.attach_evidence(events, messages, bot_qq="bot")
    merged = SCENE.merge_scene_fragments(enriched, messages, bot_qq="bot")
    assert len(merged) == 2


def test_selection_is_variable_and_does_not_fill_fourth_slot():
    events = [
        {"who": "甲", "what": "高分一", "quote": "", "score": 5},
        {"who": "乙", "what": "高分二", "quote": "", "score": 4},
        {"who": "丙", "what": "群趣闻", "quote": "", "score": 3},
        {"who": "丁", "what": "另一件", "quote": "", "score": 3},
    ]
    selected = BD.select_events_for_write(events, max_events=4, max_per_who=1)
    assert [e["who"] for e in selected] == ["甲", "乙"]
    quiet = BD.select_events_for_write(events[2:], max_events=4)
    assert len(quiet) == 2  # 没有高分主场景时，允许两件 3 分组成一个安静日
    assert len(BD.select_events_for_write(events[2:3], max_events=4)) == 1


def test_recent_same_person_same_topic_without_progress_is_skipped():
    recent = [{"who": "阿岚", "what": "喜欢我", "quote": "", "topic_bucket": "affection"}]
    repeated = {"who": "阿岚", "what": "又说喜欢我", "quote": "", "score": 5, "topic_bucket": "affection"}
    new_progress = {
        "who": "阿岚", "what": "终于把喜欢说清楚了", "quote": "",
        "score": 5, "topic_bucket": "affection",
    }
    assert BD.select_events_for_write(
        [repeated], max_events=4, max_per_who=1, recent_events=recent
    ) == []
    kept = BD.select_events_for_write(
        [new_progress], max_events=4, max_per_who=1, recent_events=recent
    )
    assert len(kept) == 1


def test_fact_and_style_guards_catch_real_failure_modes():
    facts = SCENE.diary_fact_risks(
        "2026年11月12日 星期四。\n关门前13秒刷脸进门，外面下着雨。",
        "小乙: 23:58刷脸，00:13回宿舍",
    )
    assert any("13秒" in item for item in facts)
    assert any("天气" in item for item in facts)
    recent = ["睡前脑子还热着，群里还没消停。", "眼皮已经打架了，脑子里还转着几件事。"]
    assert SCENE.style_repeat_reasons("眼皮早沉了，脑子还转着今天的事，先记一笔。", recent)
    assert not SCENE.style_repeat_reasons("食堂的锅盔排队排到了门口。", recent)


def test_malformed_json_is_retryable_but_explicit_empty_is_not():
    assert BD.extraction_note_retryable(
        "JSON 解析失败（截取 785 字）",
        '[{"who": "甲", "what": "给"我"充电"}]',
    )
    assert BD.extraction_note_retryable("模型返回为空", "")
    assert not BD.extraction_note_retryable("模型明确返回空数组（确实无事）", "[]")


def test_weather_is_present_every_day_with_evidence_or_fallback():
    assert BD.weather_label("当天明显有雨（素材里出现了「下雨」）。", "多云") == "雨"
    assert BD.weather_label("素材里提到了天气（「好冷」，对应「多云」）。", "晴") == "多云"
    assert BD.weather_label("", "阴") == "阴"
    prompt = BD.build_write_prompt(
        date_str="2026年11月14日 星期六",
        events_text="- 甲：聊了一件小事",
        name="鸣澜",
        persona="测试",
        weather_fallback="阴",
    )
    assert "第一行固定为 2026年11月14日 星期六，阴。" in prompt
    forced = BD.ensure_date_line(
        "2026年11月14日 星期六，晴。\n今天聊得挺晚。",
        "2026年11月14日 星期六",
        weather_override="阴",
    )
    assert forced.startswith("2026年11月14日 星期六，阴。")
    assert "晴。" not in forced.split("\n", 1)[0]


def test_invalid_weather_fallback_is_normalized_to_cloudy():
    cfg = PLUGIN_MOD.BetterDiaryConfig.model_validate({
        "plugin": {"enabled": True, "config_version": "1.0.0"},
        "diary": {"weather_fallback": "暴雨"},
    })
    assert cfg.diary.weather_fallback == "多云"


def test_opening_hint_avoids_recent_two_choices():
    date = "2026-11-13"
    first = BD.opening_hint(date)
    second = BD.opening_hint(date, avoid=[first])
    third = BD.opening_hint(date, avoid=[first, second])
    assert len({first, second, third}) == 3
