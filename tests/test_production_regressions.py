"""v1.8.0 真机数据回归：天气误判 / 旧线索尾巴 / 隐私 / 引用与结构定式。

所有素材均为**匿名化合成样本**，不复用真机聊天原文。反向验证对应
2026-10-25~31 七天 trace 中已确认的形态：歌名里的「雪」被当成天气、
「多云 + 第二行雪」、旧项目连续四天进正文、三件事逐段引用。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
_DEVKIT_CANDIDATES = [
    os.environ.get("MAIBOT_DEVKIT_DIR", ""),
    str(PLUGIN_DIR.parent / "tools" / "maibot-devkit"),
    str(PLUGIN_DIR / "maibot-devkit"),
]
for _cand in _DEVKIT_CANDIDATES:
    if _cand and (Path(_cand) / "fakehost.py").is_file():
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

from fakehost import load_plugin_module  # noqa: E402

sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
BD = importlib.import_module(
    f"{load_plugin_module(str(PLUGIN_DIR), module_name='better_diary_prod_test').__name__}.bd_prompts"
)


# ------------------------------------------------------------------ 天气

def test_weather_ignores_snow_inside_song_title_and_input_method():
    """真机 10-27 / 10-31：歌名与输入法里的「雪」不得触发天气钉死。"""
    assert BD.weather_hint("甲：棠梨蘸雪也是磨出来的") == ""
    assert BD.weather_hint("乙：我就是从雪花换过来的") == ""


def test_explicit_snow_expression_is_still_evidence():
    hint = BD.weather_hint("甲：外面下雪了，地上积雪挺厚")
    assert "雪" in hint and "下雪" in hint


def test_snow_date_line_is_kept_as_one_line():
    """日期行认「雪」之后，不能再拆成「多云 + 第二行雪」。"""
    out = BD.ensure_date_line(
        "2026年10月27日 星期二，雪。\n正文只有一句。",
        "2026年10月27日 星期二",
    )
    assert out.splitlines()[0] == "2026年10月27日 星期二，雪。"
    assert "雪。" not in out.splitlines()[1:]
    assert "多云" not in out


# ------------------------------------------------------------------ 连续性

def test_continuity_only_keeps_items_with_new_evidence():
    """没有当天新证据的旧尾巴不得注入。"""
    state = {
        "previous_summary": "新歌命名还没定；旧梗还在排队",
        "ongoing_projects": ["新歌命名", "电台周边"],
        "unresolved_items": ["旧梗排队"],
    }
    kept = BD.filter_continuity_by_evidence(state, "甲：新歌命名终于定了，就叫晚风")
    assert kept.get("ongoing_projects") == ["新歌命名"]
    assert "unresolved_items" not in kept
    assert "旧梗还在排队" not in kept.get("previous_summary", "")


def test_continuity_is_empty_without_related_material():
    state = {"ongoing_projects": ["新歌命名"], "unresolved_items": ["旧梗排队"]}
    assert BD.filter_continuity_by_evidence(state, "甲：今天吃了面，挺香") == {}


def test_filtered_continuity_render_has_no_stale_tail():
    state = {
        "previous_summary": "新歌命名还没定；旧梗还在排队",
        "ongoing_projects": ["新歌命名", "电台周边"],
    }
    kept = BD.filter_continuity_by_evidence(state, "甲：新歌命名终于定了")
    rendered = BD.build_continuity_line(kept)
    assert "新歌命名" in rendered
    assert "旧梗" not in rendered and "电台周边" not in rendered


def test_empty_continuity_does_not_crash():
    assert BD.filter_continuity_by_evidence({}, "任意素材") == {}
    assert BD.filter_continuity_by_evidence({"ongoing_projects": ["甲"]}, "") == {}


# ------------------------------------------------------------------ 隐私

def test_sensitive_event_is_anonymized_and_quote_removed():
    events = [
        {
            "who": "小明",
            "what": "小明说家里把孩子打掉了，之后一直很难受",
            "quote": "我妈又忘了我的事",
            "score": 5,
        },
        {"who": "小红", "what": "分享了一首歌", "quote": "好听", "score": 5},
    ]
    cleaned, count = BD.redact_sensitive_events(events)
    assert count == 1
    assert cleaned[0]["who"] == "一位群友"
    assert cleaned[0]["quote"] == ""
    assert "小明" not in cleaned[0]["what"]
    assert cleaned[1] == events[1]


def test_sensitive_event_also_redacts_unselected_third_party_names():
    """敏感摘要里提到的第三人即使没入选，也必须被抹掉。"""
    events = [
        {
            "who": "小明",
            "what": "小明说小红家里出了事，自己听了很难受",
            "quote": "",
            "score": 5,
        },
    ]
    cleaned, count = BD.redact_sensitive_events(
        events, known_names=["小明", "小红（同学）"],
    )
    assert count == 1
    assert "小明" not in cleaned[0]["what"]
    # 群名片带后缀，摘要只写了前半段：别名也必须替换
    assert "小红" not in cleaned[0]["what"]


def test_normal_event_is_not_anonymized():
    events = [{"who": "小红", "what": "分享了一首歌", "quote": "好听", "score": 5}]
    cleaned, count = BD.redact_sensitive_events(events)
    assert count == 0 and cleaned == events


def test_sensitive_text_never_reaches_write_prompt():
    """匿名化后的素材链（redact → events_to_text → prompt）不得残留身份与原话。"""
    events = [{
        "who": "小明",
        "what": "小明说家里把孩子打掉了，之后一直很难受",
        "quote": "我妈又忘了我的事",
        "score": 5,
    }]
    cleaned, _ = BD.redact_sensitive_events(events)
    material = BD.events_to_text(cleaned)
    prompt = BD.build_write_prompt(
        date_str="2026年11月1日 星期日",
        events_text=material,
        name="测试",
        persona="是测试角色",
    )
    assert "小明" not in material and "小明" not in prompt
    assert "我妈又忘了我的事" not in material and "我妈又忘了我的事" not in prompt
    assert "一位群友" in prompt


# ------------------------------------------------------------------ 引用与选材

def test_only_one_external_quote_reaches_prompt():
    events = [
        {"who": "甲", "what": "说了一件事", "quote": "第一句", "score": 5},
        {"who": "乙", "what": "说了另一件", "quote": "第二句", "score": 5},
        {"who": "丙", "what": "说了第三件", "quote": "第三句", "score": 5},
    ]
    limited, removed = BD.limit_event_quotes(events, limit=1)
    assert removed == 2
    assert [e["quote"] for e in limited] == ["第一句", "", ""]


def test_selection_does_not_force_three_items():
    """⚠️ v1.9.0 语义已变：本测试原本断言「宁可只写一件」，现在断言「绝不吃低分垃圾」。

    v1.8.0 的原意是反对 ``events[:max_events]`` 无条件写满；但它连带把
    「还有 3 分候选却空着席位」也禁掉了。v1.9.0 依据六天真机数据把两件事拆开：
    - **广度**靠 backfill_score=3 回填（3 分事件在真机里正是「买牙膏被推销自行车」
      「查好感度嫌分低嚷着拉满」这类好玩的群事，恰恰是该进日记的题材）；
    - **反凑字**仍然成立：1~2 分的水群/刷屏永不入选，且篇幅由
      ``choose_word_budget`` 按素材字数独立决定，回填一件薄素材不会把档位推高。
    """
    events = [
        {"who": "甲", "what": "高分事", "quote": "", "score": 5},
        {"who": "乙", "what": "低分事", "quote": "", "score": 3},
        {"who": "丙", "what": "更低分", "quote": "", "score": 1},
    ]
    selected = BD.select_events_for_write(events, max_events=3, min_score=4)
    # 3 分回填进来了，1 分绝不进来
    assert [e["what"] for e in selected] == ["高分事", "低分事"]
    assert "更低分" not in [e["what"] for e in selected]


def test_selection_never_backfills_below_backfill_score():
    """回填下限是硬约束：全场只有 1~2 分时，只保留最高分那一件，绝不凑数。"""
    events = [
        {"who": "甲", "what": "还行", "quote": "", "score": 2},
        {"who": "乙", "what": "水群", "quote": "", "score": 1},
        {"who": "丙", "what": "刷屏", "quote": "", "score": 1},
    ]
    selected = BD.select_events_for_write(events, max_events=4, min_score=4)
    assert len(selected) == 1
    assert selected[0]["what"] == "还行"


def test_selection_caps_same_person():
    """真机 2026-11-06 入选 3 件全是同一人；同人上限 2 后第三席必须让位给别人。"""
    events = [
        {"who": "心理委员", "what": "歌词接龙", "quote": "", "score": 5},
        {"who": "心理委员", "what": "嘴硬却贴过来", "quote": "", "score": 4},
        {"who": "心理委员", "what": "问我为什么老看她动态", "quote": "", "score": 4},
        {"who": "魈猫咪", "what": "花858买声库", "quote": "", "score": 4},
    ]
    selected = BD.select_events_for_write(events, max_events=4, min_score=4, max_per_who=2)
    whos = [e["who"] for e in selected]
    assert whos.count("心理委员") == 2
    assert "魈猫咪" in whos
    assert len(selected) == 3


def test_selection_stops_instead_of_backfilling_a_fourth_slot():
    """v1.10.0：有 4 分候选时不再用 3 分凑第 4 席；1~4 件自然变化。"""
    events = [
        {"who": "A", "what": "a1", "quote": "", "score": 5},
        {"who": "A", "what": "a2", "quote": "", "score": 5},
        {"who": "A", "what": "a3", "quote": "", "score": 5},
        {"who": "B", "what": "b1", "quote": "", "score": 4},
        {"who": "C", "what": "c1", "quote": "", "score": 3},
        {"who": "D", "what": "d1", "quote": "", "score": 3},
    ]
    selected = BD.select_events_for_write(events, max_events=4, max_per_who=2)
    whos = [e["who"] for e in selected]
    assert whos.count("A") == 2
    assert "B" in whos
    assert "C" not in whos and "D" not in whos
    assert len(selected) == 3


def test_selection_keeps_one_event_when_none_meets_threshold():
    events = [{"who": "甲", "what": "唯一候选", "quote": "", "score": 3}]
    assert len(BD.select_events_for_write(events, max_events=3, min_score=4)) == 1


def test_material_chars_counts_quote():
    """v1.9.0：quote 本来就进成文 prompt，档位素材量必须把它算进去。

    真机六天回放：what-only 是 55~96 字（4/6 天误判短档，成品 325→124 字连降），
    what+quote 是 90~148 字（5 中 1 长，目标回到 245~350）。
    """
    events = [{"who": "甲", "what": "五字摘要", "quote": "一句挺长的原话内容", "score": 5}]
    got = BD.event_material_chars(events)
    assert got == len("五字摘要") + len("一句挺长的原话内容")
    assert got > len("五字摘要")


def test_real_trace_replay_tier_recovers():
    """用 2026-11-02~11-07 六天真机素材回放：确认档位不再全落短档。"""
    # (what字数, quote字数) 取自 selection.json 的入选三件（旧策略口径）
    real_days = {
        "11-02": (78, 126),
        "11-03": (96, 148),
        "11-04": (55, 93),
        "11-05": (59, 90),
        "11-06": (61, 112),
        "11-07": (64, 104),
    }
    old_tiers = [
        BD.choose_word_budget(350, 150, event_count=3, material_chars=w)["tier"]
        for w, _ in real_days.values()
    ]
    new_tiers = [
        BD.choose_word_budget(350, 150, event_count=3, material_chars=wq)["tier"]
        for _, wq in real_days.values()
    ]
    assert old_tiers.count("短") == 4, f"旧口径回放对不上真机：{old_tiers}"
    assert new_tiers.count("短") == 0, f"新口径仍落短档：{new_tiers}"
    assert new_tiers.count("长") == 1
    assert new_tiers.count("中") == 5


def test_word_budget_uses_material_chars_not_event_count():
    """三件短素材可以比一件长素材更短，证明篇幅不再由事件数直接决定。"""
    many_short = BD.choose_word_budget(350, 150, event_count=3, material_chars=60)
    one_long = BD.choose_word_budget(350, 150, event_count=1, material_chars=200)
    assert many_short["tier"] == "短"
    assert one_long["tier"] == "长"
    assert many_short["target"] < one_long["target"]


def test_write_prompt_has_single_quote_and_no_padding_instruction():
    prompt = BD.build_write_prompt(
        date_str="2026年11月1日 星期日",
        events_text="- 甲：说了一件事（原话：「一句」）",
        name="测试",
        persona="是测试角色",
    )
    assert "全篇最多引一处" in prompt
    assert "多补一句感受" not in prompt
    assert "不必全写、也不必一件一段" in prompt
    assert "涉及他人隐私" in prompt
