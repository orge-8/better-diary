"""v1.5.0「去 AI 味」三件套：开场轮换 / 篇幅三档 / AI 腔检测。

这三个都是**纯函数**，可以脱离 MaiBot 单测 —— 它们承担的是"让日记不像 AI 写的"
这件事里可验证的那部分（确定性、边界、命中率），其余交给真机观察。

真机回读口径（本方案的回归指标）：
- ``篇幅档位: 中（目标 245，区间 140~350，依据：入选 2 件）``
- ``AI 腔检测命中 2 项: 总结式收尾、展望收尾``
- 存档字段 ``ai_tone_hits``（数组，空 = 没命中）
"""

from __future__ import annotations

import datetime
import importlib
import os
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

# 与其余用例同一套做法：**按真机的包式加载**取模块（并保证插件目录不在 sys.path 上）。
# 直接 flat import bd_prompts 会随 pytest 的收集顺序漂移 —— 别的
# 用例会把 PLUGIN_DIR 从 sys.path 摘掉，本文件就可能 ImportError。
sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
BD = importlib.import_module(
    f"{load_plugin_module(str(PLUGIN_DIR), module_name='better_diary_tone_test').__name__}.bd_prompts"
)

ai_tone_hits = BD.ai_tone_hits
choose_word_budget = BD.choose_word_budget
opening_hint = BD.opening_hint


# ------------------------------------------------------------------ 开场轮换

@pytest.mark.parametrize(
    "date_str",
    ["2026年9月28日 星期一", "2026-09-29", "2026-10-01", "2027-01-01"],
)
def test_opening_hint_is_deterministic(date_str):
    """同一天永远同一副骨架 —— 出问题能精确定位到是哪一种开场。"""
    assert opening_hint(date_str) == opening_hint(date_str)


def test_opening_hint_rotates_enough():
    """连着写半个月，不能每天都用同一种开场（那样等于没有轮换）。"""
    days = [
        (datetime.date(2026, 10, 1) + datetime.timedelta(days=i)).isoformat()
        for i in range(20)
    ]
    seen = {opening_hint(d) for d in days}
    assert len(seen) >= 3, f"20 天里只轮换到 {len(seen)} 种开场：{seen}"


def test_opening_hint_survives_bad_input():
    """开场提示不值当让成文失败：空串 / None 都要能兜住。"""
    assert opening_hint("")
    assert opening_hint(None) == opening_hint("")


def test_opening_hint_never_uses_diary_quote_brackets():
    """hint 里不许出现「」。

    引号在本插件的 prompt 里有**专属含义**（只能装别人的原话、一字不改），
    开场提示里带「」会让模型以为那也是引用 —— 等于给规则 3 制造歧义。
    """
    days = [(datetime.date(2026, 1, 1) + datetime.timedelta(days=i)).isoformat() for i in range(40)]
    for hint in {opening_hint(d) for d in days}:
        assert "「" not in hint and "」" not in hint, hint


# ------------------------------------------------------------------ 篇幅三档

def test_budget_tiers_by_event_count():
    """档位判据：≤1 短 / 2 中 / ≥3 长；默认 350±150 → 140±75 / 245±105 / 350±150。"""
    short = choose_word_budget(350, 150, event_count=1)
    assert (short["tier"], short["target"], short["tolerance"]) == ("短", 140, 75)
    assert (short["lo"], short["hi"]) == (65, 215)

    mid = choose_word_budget(350, 150, event_count=2)
    assert (mid["tier"], mid["target"], mid["tolerance"]) == ("中", 245, 105)
    assert (mid["lo"], mid["hi"]) == (140, 350)

    long_ = choose_word_budget(350, 150, event_count=3)
    assert (long_["tier"], long_["target"], long_["tolerance"]) == ("长", 350, 150)
    assert (long_["lo"], long_["hi"]) == (200, 500)

    # ≥3 件都是长档（只有依据不同）
    more = choose_word_budget(350, 150, event_count=9)
    assert more["tier"] == long_["tier"] == "长"
    assert (more["target"], more["lo"], more["hi"]) == (long_["target"], long_["lo"], long_["hi"])
    assert more["reason"] == "入选 9 件"


def test_budget_zero_events_is_short():
    """入选 0 件（含降级那天）一律短写 —— 没素材还写长篇就是在凑字。"""
    got = choose_word_budget(350, 150, event_count=0)
    assert got["tier"] == "短"
    assert got["reason"] == "入选 0 件"


def test_budget_timeline_tail_is_short():
    """选材降级 → 短档，且 reason 点名原因（和"确实只选出 1 件"区分开）。"""
    got = choose_word_budget(350, 150, event_count=5, material_mode="timeline_tail")
    assert got["tier"] == "短"
    assert got["reason"] == "选材降级"


def test_budget_reason_is_loggable_without_chat_content():
    """reason 只能含档位依据，绝不能捎带聊天内容（本插件日志硬约定）。"""
    for count in (0, 1, 2, 3, 7):
        reason = choose_word_budget(350, 150, event_count=count)["reason"]
        assert reason.startswith("入选") or reason == "选材降级", reason
        assert len(reason) <= 12, reason


def test_budget_never_dips_below_the_semantic_gate():
    """短档再短也不能逼近 ``MIN_DIARY_CHARS``（20 字）那条语义闸。

    把目标压到 20 字附近，等于天天把模型往「输出过短，疑似未成文」上推 ——
    生成侧引导绝不能和发布闸打架。
    """
    from bd_prompts import MIN_DIARY_CHARS

    for target in (80, 100, 200, 350, 800):
        short = choose_word_budget(target, 150, event_count=1)
        assert short["target"] > MIN_DIARY_CHARS * 2, (target, short)
        # 极端配置下（容差远大于目标）下限也不许掉进语义闸的射程里
        wild = choose_word_budget(target, 500, event_count=1)
        assert wild["lo"] > MIN_DIARY_CHARS, (target, wild)


def test_budget_lo_is_always_positive():
    """容差大于目标时下限不能变成 0 或负数。"""
    got = choose_word_budget(60, 500, event_count=3)
    assert got["lo"] >= 1
    assert got["hi"] >= got["lo"]


def test_budget_is_deterministic():
    got = choose_word_budget(300, 100, event_count=2)
    assert got == choose_word_budget(300, 100, event_count=2)


@pytest.mark.parametrize("target,tol", [(350, 150), (300, 50), (120, 40)])
def test_budget_tiers_are_ordered(target, tol):
    """三档必须严格递增 —— 否则"素材多反而写短"这种反效果排查不出来。"""
    short = choose_word_budget(target, tol, event_count=1)
    mid = choose_word_budget(target, tol, event_count=2)
    long_ = choose_word_budget(target, tol, event_count=3)
    assert short["target"] < mid["target"] < long_["target"]
    assert short["hi"] <= mid["hi"] <= long_["hi"]


# ------------------------------------------------------------------ AI 腔检测

CLEAN = (
    "2026年9月28日 星期一，阴。\n"
    "晚上老张说「稿子改完了 眼睛有点酸 先躺一会儿」，声音听着就不对劲。\n"
    "我跟着熬到十二点，最后也没等到她那句早安。"
)


def test_clean_diary_has_no_hits():
    """真人写的（真机成品级）不能有误报 —— 误报会把注意力引到没问题的句子上。"""
    assert ai_tone_hits(CLEAN) == []


def test_empty_content_has_no_hits():
    assert ai_tone_hits("") == []
    assert ai_tone_hits("   \n  ") == []


@pytest.mark.parametrize(
    "body,label",
    [
        ("今天真是收获满满的一天。", "总结式收尾"),
        ("我说不上为什么，明天也要继续加油才行。", "展望收尾"),
        ("这一整天过得很充实。", "治愈系词汇"),
        ("那一刻我意识到，我其实一直在拖着。", "顿悟句式"),
        ("竟然真的抽到了！！", "连续感叹号"),
        ("一边看书，一边听歌，一边等我妈叫我吃饭。", "排比句式"),
    ],
)
def test_each_ai_tell_is_caught(body, label):
    """六项特征各有一条命中用例 —— 漏掉任何一项，回归指标就少一个维度。"""
    hits = ai_tone_hits(body)
    assert label in hits, f"{label} 未命中：{hits}"


def test_hits_are_deduped_and_stable():
    """同一特征命中多次也只报一项 —— 这是分布统计不是逐句批改。"""
    body = "真是充实的一天！我意识到要慢下来，也很治愈，明天也要加油！！"
    hits = ai_tone_hits(body)
    assert hits == sorted(hits, key=hits.index)
    assert len(hits) == len(set(hits))
    assert {"总结式收尾", "展望收尾", "治愈系词汇", "顿悟句式", "连续感叹号"} <= set(hits)


def test_hits_never_leak_chat_content():
    """返回值只是标签：标量之外一个字都不许带（日志不得出现聊天逐字原话）。"""
    body = "她说「我明天也要去看医生哦」，我说明天也要记得吃药。"
    hits = ai_tone_hits(body)
    assert hits == ["展望收尾"]
    for hit in hits:
        assert "医生" not in hit and "她说" not in hit


def test_natural_repetition_is_not_flagged_as_parallelism():
    """正常口语里的重复（同一个字开头但结构不同）不该被当成排比。

    检测器只认「连续三个短分句开头两字完全相同」这种真排比，
    「一杯茶，一本书」这种只能算巧合 —— 宁可漏报也不误报。
    """
    assert ai_tone_hits("一杯茶，一本书，一个下午就这么过去了。") == []


def test_quoted_speech_does_not_trigger_day_summary():
    """别人的话里出现"真是…的一天"不算作者的 AI 腔 —— 只把误报率放在正文 header 之外。

    注意：检测器是纯正则，拿不到"这句是谁说的"，所以这一条留作**已知局限**，
    命中只留痕、绝不据此判失败。
    """
    body = "她发了句「真是难熬的一天」，后面就没动静了。"
    assert "总结式收尾" in ai_tone_hits(body)
