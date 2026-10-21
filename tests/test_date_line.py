"""日期行与篇幅契约的回归用例（bd_prompts 纯函数）。

两条真机反馈驱动：
1. **日期行判据太脆**（2026-09-28）：模型写 ``2026年9月28日 星期一 雨``（漏了逗号），
   旧判据「首行含年份 且 含逗号」认不出来 → 又补一行 ``…，多云。`` →
   日记开头出现**两行日期**，且模型真实观察到的天气（雨）被默认值（多云）冲掉。
2. **篇幅限制偏紧**：目标 250 字时模型自然写到 341 字，区间太窄等于每次都在越界。
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
# devkit（fakehost）只在开发机跑测试时需要；优先环境变量，其次探测同级目录。
_DEVKIT_CANDIDATES = [
    os.environ.get("MAIBOT_DEVKIT_DIR", ""),
    str(PLUGIN_DIR.parent / "tools" / "maibot-devkit"),
    str(PLUGIN_DIR / "maibot-devkit"),
    r"C:\Users\38160\Desktop\tools\maibot-devkit",
]
for _cand in _DEVKIT_CANDIDATES:
    if _cand and (Path(_cand) / "fakehost.py").is_file():
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

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


def test_other_date_is_rewritten_not_kept_as_second_line():
    """写的是**别的日期**时（v1.4.3 折中方案）：改写成目标日期行，不吞掉该行里的正文。

    旧契约是"另补一行、原日期行原样留着" —— 真机实测那会在正文里留下**两行日期**，
    读者看到的第一件事是日记写错了日子。
    """
    out = _ensure("2026年9月27日 星期日，晴。\n睡前翻聊天记录。")
    assert out.startswith("2026年9月28日 星期一，"), out
    assert "2026年9月27日" not in out, f"别的日期行不该留在正文里：{out!r}"
    assert "睡前翻聊天记录。" in out, f"正文被吞掉了：{out!r}"
    assert out.count("2026年9月28日") == 1, out
    # 别的日子的天气（晴）不能被当成今天的天气
    assert "晴" not in out.split("\n")[0], out


def test_overridden_date_line_is_recoverable_for_archiving():
    """被换掉的那一行必须能取回来（落进存档 model_date_line + 打日志）。

    折中方案的关键：成品干净（只有一行正确日期），但**模型写错日期**这件事不静默 ——
    它是提示「补写基准/prompt 有问题」的信号，不能只靠人翻日志。
    """
    raw = "2026年9月27日 星期日，晴。今天睡到中午才起。\n睡前翻聊天记录。"
    out = BD.ensure_date_line(BD.strip_diary_output(raw), DATE)
    got = BD.date_line_overridden(BD.strip_diary_output(raw), DATE)
    assert got == "2026年9月27日 星期日，晴。今天睡到中午才起。", got
    assert "2026年9月27日" not in out


def test_second_line_other_date_is_collapsed_and_reported():
    """真机 2026-09-28 的确切形态：**首行对、第二行写错**。

    这是 v1.4.3 修的那个 bug —— 折叠循环只认目标日期就 `break`，于是第二行日期
    原样留在正文里，成品出现两行日期。**这个形态此前完全没有用例**，所以把折叠逻辑
    整个退回旧行为时测试全绿（反向验证时发现的假绿），这里补上：

    1. 成品只留一行日期；
    2. 被折叠掉的那一行要能被取回来报出去（否则「折叠了但没人知道」）。
    """
    raw = "2026年9月28日 星期一，多云。\n2026年9月27日 星期日，晴。\n睡前翻聊天记录。"
    out = BD.ensure_date_line(raw, DATE)
    assert out.count("2026年9月28日") == 1, out
    assert "2026年9月27日" not in out, f"第二行日期没被折叠：{out!r}"
    assert "睡前翻聊天记录。" in out
    assert BD.date_line_overridden(raw, DATE) == "2026年9月27日 星期日，晴。"


def test_second_line_same_date_keeps_body_and_reports_nothing():
    """第二行是**同一天**（模型重复写）→ 折叠掉，且不算「写错日期」。"""
    raw = "2026年9月28日 星期一，多云。\n2026年9月28日 星期一 雨\n睡前翻聊天记录。"
    out = BD.ensure_date_line(raw, DATE)
    assert out.count("2026年9月28日") == 1, out
    assert "睡前翻聊天记录。" in out
    assert BD.date_line_overridden(raw, DATE) == ""


def test_format_only_difference_is_not_reported_as_override():
    """只是格式差异（漏逗号 / 短式）不算「模型写错日期」——记下来只会是噪音。"""
    for raw in (
        "2026年9月28日 星期一 雨\n睡前翻聊天记录。",
        "9月28日 星期一 雨\n睡前翻聊天记录。",
        "2026年9月28日 星期一，多云。\n睡前翻聊天记录。",
    ):
        out = BD.ensure_date_line(BD.strip_diary_output(raw), DATE)
        assert BD.date_line_overridden(BD.strip_diary_output(raw), DATE) == "", raw
        assert out.count("2026年9月28日") == 1, out


def test_no_date_line_at_all_is_not_reported_as_override():
    """模型压根没写日期行时，补行也不算「改写了它的日期」。"""
    raw = "睡前翻聊天记录，今天挺累。"
    out = BD.ensure_date_line(BD.strip_diary_output(raw), DATE)
    assert BD.date_line_overridden(BD.strip_diary_output(raw), DATE) == ""
    # 正文提到年份（不是日期行）同理
    raw2 = "2026年过得真快，转眼就秋天了。"
    out2 = BD.ensure_date_line(BD.strip_diary_output(raw2), DATE)
    assert BD.date_line_overridden(BD.strip_diary_output(raw2), DATE) == ""
    assert out2.count("2026年9月28日") == 1


def test_override_detection_survives_meta_stripping():
    """真实链路形态：raw 里带 META 段、日期行是错的 —— 仍要能取回被换掉的那一行。

    ``_generate`` 传进来的 raw 是 split_meta 之后的正文 ``body``，这里按同样顺序走一遍。
    """
    raw = (
        "2026年9月27日 星期日，晴。\n"
        "睡前翻聊天记录。\n\n"
        + BD.META_MARKER
        + '\n{"topics": ["新歌"]}'
    )
    body, meta = BD.split_meta(BD.strip_diary_output(raw))
    assert meta == {"topics": ["新歌"]}, meta
    out = BD.ensure_date_line(body, DATE)
    assert BD.date_line_overridden(body, DATE) == "2026年9月27日 星期日，晴。"
    # META 永不进正文
    assert BD.META_MARKER not in out


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


def test_trailing_duplicate_date_line_is_removed():
    """真机 2026-10-08：模型在结尾又写了一遍日期行（天气还和首行不一致）。"""
    body = (
        "2026年10月8日 星期四，阴。\n"
        "稿子改完了，眼睛还是有点酸，但整个人清醒过来了。\n"
        "小北 昨天发的图还没细看，等明天精神再好一点补上。\n"
        "2026年10月8日 星期四，晴"
    )
    out = BD.ensure_date_line(body, "2026年10月8日 星期四")
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert lines[0] == "2026年10月8日 星期四，阴。", out
    assert out.count("2026年10月8日") == 1, f"结尾的日期行没删掉：{out!r}"
    assert "晴" not in out, f"结尾那行的天气不该被留下来：{out!r}"
    assert "等明天精神再好一点补上。" in out, "正文不能被误删"


def test_date_line_only_at_the_end_is_moved_to_the_top():
    """模型把日期行只写在结尾时：搬上去，并**沿用它的天气**（别浪费）。"""
    out = BD.ensure_date_line(
        "稿子改完了，今天终于有力气干点别的。\n2026年10月8日 星期四，晴",
        "2026年10月8日 星期四",
    )
    assert out.split("\n")[0] == "2026年10月8日 星期四，晴。", out
    assert "晴" in out and "多云" not in out, out
    assert "稿子改完了，今天终于有力气干点别的。" in out


def test_trailing_weather_not_used_when_head_has_one():
    """首行已有日期行时，结尾那行的天气不采用（首行才是权威位置）。"""
    out = BD.ensure_date_line(
        "2026年10月8日 星期四，雨。\n正文一句。\n2026年10月8日 星期四，晴",
        "2026年10月8日 星期四",
    )
    assert out.split("\n")[0] == "2026年10月8日 星期四，雨。", out
    assert out.count("2026年10月8日") == 1, out


def test_trailing_line_that_only_looks_like_prose_is_kept():
    """结尾是正文（提到年份但不是日期行）→ 不能被删。"""
    body = "2026年10月8日 星期四，阴。\n今天聊到 2026年过得真快。"
    out = BD.ensure_date_line(body, "2026年10月8日 星期四")
    assert "2026年过得真快" in out, out


def test_trailing_date_line_helper_returns_weather_and_stripped_text():
    got = BD.trailing_date_line("正文。\n2026年10月8日 星期四，晴", "2026年10月8日 星期四")
    assert got is not None and got[0] == "晴" and got[1] == "正文。", got
    assert BD.trailing_date_line("只有正文，没有日期行。", "2026年10月8日 星期四") is None


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
    """style_extra 必须排在**所有规则之后**，且规则区编号不撞号。

    v1.5.0 起编号由 ``enumerate`` 动态生成 —— 历史上有两次因为改规则条数把
    style_extra 的编号算错（撞号会让模型把两条规则读成一条），交给程序就再也不会错。
    """
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p",
        style_extra="多用叠词",
    )
    assert "\n8. 多用叠词" in prompt
    assert "\n7. 篇幅：" in prompt
    # 每条规则编号在**规则区**内只能出现一次（撞号会让模型把两条读成一条）
    rules_region, _, _ = prompt.partition("日记正文写完后")
    nums = [ln.split(".", 1)[0] for ln in rules_region.split("\n") if re.match(r"^\d{1,2}\. ", ln)]
    assert nums, rules_region
    assert len(nums) == len(set(nums)), f"规则编号撞号：{nums}"
    # 编号必须是连续递增的 1..N（编号乱跳同样会让模型读串）
    assert nums == [str(i) for i in range(1, len(nums) + 1)], f"规则编号不连续：{nums}"


def test_opening_hint_is_rotated_and_deterministic():
    """开场骨架：同一天永远同一种（可复现），不同天轮换得开（不构成定式）。

    「每天同一副骨架」是最大的 AI tell —— 哪怕每句都像人写的，连看三篇就认出来了。
    """
    hint = BD.opening_hint(DATE)
    assert hint == BD.opening_hint(DATE)
    seen = {BD.opening_hint(f"2026-01-{d:02d}") for d in range(1, 29)}
    assert len(seen) >= 3, seen  # 28 天里至少要轮换到 3 种以上
    # hint 里不许出现「」：引号在本 prompt 里有专属含义（只能装别人的原话），
    # 出现在开场提示里会给规则 3 制造歧义
    assert "「" not in hint and "」" not in hint, hint


def test_write_prompt_uses_human_viewpoint_and_opening():
    """视角转换：「整理聊天记录」→「脑子里还留着这些片段」+ 开场 hint 进 prompt。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p"
    )
    assert "这一天过完了，睡前随手记一下" in prompt
    # 旧的书记员视角不能留
    assert "睡前翻了翻今天的聊天记录" not in prompt
    assert f"开场：{BD.opening_hint(DATE)}" in prompt


def test_quote_duty_is_downgraded_but_bounds_kept():
    """引用降级：不再下达"至少一处引用"的指标，但**边界约束一条不能少**。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p"
    )
    assert "没给就不引" in prompt
    assert "至少一处用「」引用" not in prompt
    # 边界硬约束
    for must in ("只能放", "一字不改", "外面", "一对「」只装一个人的一句话", "引用后回复"):
        assert must in prompt, must


def test_self_check_rules_are_present_with_counterexample():
    """三种病（空转动作 / 复述自己 / 每段感受收尾）的约束在 v1.5.0 里**压缩但没删**。

    这三条治的是真机成品里的三种现象：
    - a 空转动作（"我问她想好名字了没，她没答"）—— 动作后没有具体内容
    - b 自己复述自己（"我说不知道…其实我也好奇过"）—— 同一意思写两遍
    - c 每段都以"我的感受"收尾 —— 结构性单调

    ⚠️ v1.5.0 把它们压成规则 4 的**一句话**（10 条编号规则会让模型把算力花在合规上，
    产出反而谨慎、平、安全），但约束必须还在 —— 判定方法是**这三种现象还有对应禁令**。
    """
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p"
    )
    # a：只有动作没内容的句子
    assert "只有动作没内容" in prompt, "空转约束丢了"
    # b：换个说法又说一遍
    assert "换个说法又说一遍" in prompt, "复述自己的约束丢了"
    # c：每段都拿感受收尾
    assert "每段都拿我的感受收尾" in prompt, "感受收尾约束丢了"
    # 事实纪律仍然交叉引用 a（有真素材才能写"我问"）—— 否则模型会拿"文风自由"开脱
    assert "只有动作没内容" in prompt and "素材里没写我问过" in prompt


def test_self_check_rules_do_not_touch_existing_contracts():
    """新增生成侧约束不得改变任何**判定/规范化**契约（只是让模型少犯那些毛病）。"""
    # 引用纪律、事实纪律、篇幅纪律都还在
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p"
    )
    for must in ("只能放", "一字不改", "事实纪律", "别硬凑", "不要标题"):
        assert must in prompt, must
    # 真机成品（本次优化依据的那篇）本身必须仍是"可发布"的 —— 新规则只影响生成，
    # 不改发布判据，所以成品不会被事后判死
    real = (
        "2026年9月30日 星期三，晴。\n"
        "凌晨群里还在刷屏，有人夸我最近说话越来越像人了，被这么直接地讲出来，"
        "有点不好意思，嘴上随便应了句。\n"
        "她还没给新歌取名，不知道会是什么风格。"
    )
    assert BD.diary_output_problem(real) == ""


def test_word_budget_language_keeps_anti_padding_rule():
    """放宽上限的同时，"宁可短也别硬凑"的纪律必须还在。"""
    prompt = BD.build_write_prompt(
        date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p"
    )
    assert "别硬凑" in prompt
    assert "绝不是任务" not in prompt and "不是任务" in prompt
