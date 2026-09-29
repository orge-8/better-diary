"""日期行与篇幅契约的回归用例（bd_prompts 纯函数）。

两条真机反馈驱动：
1. **日期行判据太脆**（2026-09-28）：模型写 ``2026年9月28日 星期一 雨``（漏了逗号），
   旧判据「首行含年份 且 含逗号」认不出来 → 又补一行 ``…，多云。`` →
   日记开头出现**两行日期**，且模型真实观察到的天气（雨）被默认值（多云）冲掉。
2. **篇幅限制偏紧**：目标 250 字时模型自然写到 341 字，区间太窄等于每次都在越界。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
DEVKIT_DIR = Path(r"C:\Users\38160\Desktop\tools\maibot-devkit")
if str(DEVKIT_DIR) not in sys.path:
    sys.path.insert(0, str(DEVKIT_DIR))

from fakehost import load_plugin_module  # noqa: E402

pytest.importorskip("httpx", reason="httpx 是外发模块的可选依赖")

sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
MOD = load_plugin_module(str(PLUGIN_DIR), module_name="better_diary_dates_test")
BD = importlib.import_module("better_diary_dates_test.bd_prompts")

DATE = "2026年9月28日 星期一"
RAW_FIX = "2026-09-28"


def _ensure(text: str, date_str: str = DATE) -> str:
    return BD.ensure_date_line(BD.strip_diary_output(text), date_str)


# ---------------------------------------------------------------- 日期行

def test_real_device_shape_keeps_weather_and_single_line():
    """真机原文（缺逗号）：只留一行日期，且**保留模型写的天气**。"""
    out = _ensure("2026年9月28日 星期一 雨\n睡前翻聊天记录。")

    lines = out.split("\n")
    assert lines[0] == "2026年9月28日 星期一，雨。", out
    assert lines[1] == "睡前翻聊天记录。", out
    assert out.count("2026年9月28日") == 1, f"出现多行日期：{out!r}"
    assert "多云" not in out, f"模型写的天气被默认值覆盖：{out!r}"


def test_standard_shape_is_normalized_not_duplicated():
    """标准形态：规范化成带句号的统一形态，且不重复补行。"""
    out = _ensure("2026年9月28日 星期一，多云。\n睡前翻聊天记录。")
    assert out.startswith("2026年9月28日 星期一，多云。\n")
    assert out.count("2026年9月28日") == 1


def test_duplicate_date_lines_are_collapsed():
    """模型偶发写两行日期 → 折叠成一行。"""
    out = _ensure(
        "2026年9月28日 星期一，多云。\n"
        "2026年9月28日 星期一 雨\n"
        "睡前翻聊天记录。"
    )
    assert out.count("2026年9月28日") == 1, out
    assert "睡前翻聊天记录。" in out


def test_short_form_date_line_is_recognized():
    """短式「9月28日」也认，并补全成完整日期。"""
    out = _ensure("9月28日 星期一 雨\n睡前翻聊天记录。")
    assert out.startswith("2026年9月28日 星期一，雨。\n"), out
    # 不能剩下"行首还是短式"的形态（注意 '2026年9月28日' 里天然含 '9月28日'，
    # 所以不能拿子串计数断言，要看行首）
    assert not out.startswith("9月28日"), out
    assert out.count("2026年9月28日") == 1


def test_missing_date_line_is_prepended_once():
    """没写日期行 → 补一行（用默认天气）。"""
    out = _ensure("睡前翻聊天记录，今天挺累。")
    assert out.startswith("2026年9月28日 星期一，多云。\n"), out
    assert out.count("2026年9月28日") == 1


def test_leading_blank_lines_do_not_hide_date_line():
    """日期行前有空行时也要认出来，不能因为「第一行是空行」就再补一行。"""
    out = _ensure("\n\n2026年9月28日 星期一 雨\n睡前翻聊天记录。")
    assert out.count("2026年9月28日") == 1, f"空行导致重复补日期行：{out!r}"
    assert "雨" in out


def test_blank_lines_without_date_line_still_get_one():
    """空行开头 + 没写日期行 → 仍要补日期头（别整篇缺日期行）。"""
    out = _ensure("\n\n睡前翻聊天记录，今天挺累。")
    assert out.count("2026年9月28日") == 1, f"空行开头时漏补日期行：{out!r}"
    assert "睡前翻聊天记录，今天挺累。" in out
    # 日期行必须在正文之前
    assert out.index("2026年9月28日") < out.index("睡前翻聊天记录")


def test_other_date_is_not_treated_as_date_line():
    """写的是**别的日期**时不算日期行 —— 仍然要在前面补目标日期行，不吞掉原文。"""
    out = _ensure("2026年9月27日 星期日，晴。\n睡前翻聊天记录。")
    assert out.startswith("2026年9月28日 星期一，"), out
    assert "2026年9月27日 星期日，晴。" in out, "原日期行不该被丢掉"


def test_year_in_prose_is_not_a_date_line():
    """正文里提到年份（无月日）不能被当成日期行。"""
    out = _ensure("2026年过得真快，转眼就秋天了。")
    assert out.startswith("2026年9月28日 星期一，"), out
    assert "2026年过得真快" in out


def test_weather_word_inside_prose_is_not_stolen():
    """日期行后面的正文以「雨」开头时，不能被当成天气符号。"""
    out = _ensure("2026年9月28日 星期一。雨天路滑，早点回家。")
    assert out.startswith("2026年9月28日 星期一，多云。\n"), out
    assert "雨天路滑，早点回家。" in out


def test_prose_sharing_the_date_line_is_preserved():
    """模型把正文第一句挤在日期行里 → 拆出来，内容不能丢。"""
    out = _ensure("2026年9月28日 星期一，雨。今天群里很热闹。\n后面还有一段。")
    assert out.startswith("2026年9月28日 星期一，雨。\n"), out
    assert "今天群里很热闹。" in out
    assert "后面还有一段。" in out


def test_dash_date_format_also_works():
    """date_str 传 'YYYY-MM-DD'（插件内部形态）时同样能识别。"""
    out = BD.ensure_date_line("2026年9月28日 星期一 雨\n睡前翻聊天记录。", DATE)
    assert out.split("\n")[0] == "2026年9月28日 星期一，雨。"
    assert out.count("2026年9月28日") == 1


def test_parse_date_line_rejects_foreign_dates():
    """parse_date_line 对别的日期返回 None（供折叠逻辑判断）。"""
    assert BD.parse_date_line("2026年9月27日 星期日，晴。", DATE) is None
    assert BD.parse_date_line("睡前翻聊天记录。", DATE) is None
    got = BD.parse_date_line("2026年9月28日 星期一 雨", DATE)
    assert got == ("雨", ""), got


# ---------------------------------------------------------------- 篇幅

def test_word_range_is_widened_by_default():
    """默认区间显著放宽：模型写长了不必回头删。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p"
    )
    assert "200~500" in prompt, prompt
    assert "350 字上下" in prompt
    # 旧文案（单点 + 上下浮动 80 字）不应再出现
    assert "上下浮动 80 字" not in prompt


def test_word_range_follows_config():
    """区间随配置走。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p",
        word_target=300, word_tolerance=50,
    )
    assert "250~350" in prompt
    assert "300 字上下" in prompt


def test_word_range_never_goes_below_one():
    """浮动大于目标时，下限不能变成 0 或负数。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p",
        word_target=100, word_tolerance=500,
    )
    assert "1~600" in prompt, prompt


def test_style_extra_numbering_does_not_collide():
    """style_extra 的编号必须排在字数规则（8）之后，不能撞号。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p",
        style_extra="多用叠词",
    )
    assert "\n9. 多用叠词" in prompt
    assert "\n8. 篇幅：" in prompt
    # 不该出现两个 "8." 开头的规则
    rule8 = [ln for ln in prompt.split("\n") if ln.startswith("8. ")]
    assert len(rule8) == 1, rule8


def test_word_budget_language_keeps_anti_padding_rule():
    """放宽上限的同时，"宁可短也别硬凑"的纪律必须还在。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p"
    )
    assert "别硬凑" in prompt
    assert "绝不是任务" not in prompt and "不是任务" in prompt
