"""引用回复拆分回归用例（v1.5.1，真机 2026-10-07 归属病的根治）。

背景（2026-10-09 源码级核实）：适配器把引用目标回查成 ``target_message_content``
写进 reply 段，而 Host ``process_reply_component`` 分支①把它**不加前缀**直接拼进
``processed_plain_text`` —— 「被引原文 + 我的回复」无缝黏在一条消息里，说话人却记成
回复者本人。真机那次（回查成功）日记写成
``老张…说「稿子改完了 眼睛有点酸 先躺一会儿 我也困得慌」``：一对「」里焊了两个人的话。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
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

sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
MOD = load_plugin_module(str(PLUGIN_DIR), module_name="better_diary_reply_split_test")
BD = importlib.import_module("better_diary_reply_split_test.bd_prompts")

DAY_TS = 1789507200.0  # 2026-09-26 16:00 UTC+8 附近，具体时刻不影响断言
QUOTED = "稿子改完了 眼睛有点酸 先躺一会儿"
MY_REPLY = "我也困得慌😭"


def _reply_msg(
    *,
    sender_id: str = "10000",
    nickname: str = "鸣澜",
    body: str | None = None,
    quoted: str = QUOTED,
    quoted_sender: dict | None = None,
    with_segments: bool = True,
    is_picture: bool = False,
) -> dict:
    """构造一条（bot 发出的）引用回复消息 dict，形态对齐 message_utils 的输出。"""
    if body is None:
        body = f"{quoted}{MY_REPLY}"  # Host 分支①：被引原文 + 回复，无缝拼接
    msg = {
        "timestamp": str(DAY_TS),
        "processed_plain_text": body,
        "is_picture": is_picture,
        "message_info": {
            "user_info": {"user_id": sender_id, "user_nickname": nickname},
            "group_info": {"group_id": "123456"},
        },
    }
    if with_segments:
        msg["raw_message"] = [
            {
                "type": "reply",
                "data": {
                    "target_message_id": "90001",
                    "target_message_content": quoted,
                    "target_message_sender_id": (quoted_sender or {}).get("id", "111"),
                    "target_message_sender_nickname": (quoted_sender or {}).get("nick", "阿岚"),
                    "target_message_sender_cardname": (quoted_sender or {}).get("card"),
                },
            },
            {"type": "text", "data": MY_REPLY},
        ]
    return msg


def _normal_msg(uid: str, nick: str, text: str, ts: float = DAY_TS - 60) -> dict:
    return {
        "timestamp": str(ts),
        "processed_plain_text": text,
        "is_picture": False,
        "message_info": {
            "user_info": {"user_id": uid, "user_nickname": nick},
            "group_info": {"group_id": "123456"},
        },
    }


def test_quoted_reply_is_split_into_two_lines():
    """回查成功的引用回复：被引原文独立成行归真说话人，我的回复归「我」。"""
    timeline, stats = BD.build_timeline([_reply_msg()], bot_qq="10000")
    assert f"阿岚: {QUOTED}" in timeline, timeline
    assert f"我: {MY_REPLY}" in timeline, timeline
    # 黏连形态不许再出现
    assert f"{QUOTED}{MY_REPLY}" not in timeline.replace("\n", ""), timeline
    assert stats["reply_split"] == 1
    assert stats["total"] == 1 and stats["bot"] == 1  # 被引行是虚拟行，不占消息计数


def test_quoted_line_not_duplicated_when_original_already_in_timeline():
    """被引消息本身就在当天记录里：只剥离、不重复插行。"""
    messages = [
        _normal_msg("111", "阿岚", QUOTED),
        _reply_msg(),
    ]
    timeline, stats = BD.build_timeline(messages, bot_qq="10000")
    assert timeline.count(QUOTED) == 1, timeline
    assert f"我: {MY_REPLY}" in timeline
    assert stats["reply_split"] == 1


def test_group_cardname_wins_over_nickname():
    """群里优先用群名片（cardname）—— 昵称和名片不一致时以名片为准。"""
    msg = _reply_msg(quoted_sender={"id": "111", "nick": "阿岚", "card": "老张"})
    timeline, _ = BD.build_timeline([msg], bot_qq="10000")
    assert f"老张: {QUOTED}" in timeline, timeline


def test_quoted_from_self_is_attributed_to_me():
    """被引的若是 bot 自己说过的话：归「我」（它本来就属于我，且不进「」）。"""
    msg = _reply_msg(
        quoted="今晚吃锅盔",
        quoted_sender={"id": "10000", "nick": "鸣澜"},
        body=f"今晚吃锅盔{MY_REPLY}",
    )
    timeline, _ = BD.build_timeline([msg], bot_qq="10000")
    assert "我: 今晚吃锅盔" in timeline, timeline
    assert f"我: {MY_REPLY}" in timeline, timeline


def test_missing_reply_segment_keeps_old_behavior():
    """没有 raw_message（旧存档 / 回查失败被丢弃）：完全维持旧行为。"""
    msg = _reply_msg(with_segments=False)
    timeline, stats = BD.build_timeline([msg], bot_qq="10000")
    assert f"我: {QUOTED}{MY_REPLY}" in timeline, timeline
    assert stats["reply_split"] == 0


def test_unlocatable_reply_segment_is_ignored():
    """回查失败的 reply 段（只有 id 没有 content）：忽略，走旧行为。"""
    msg = _reply_msg(with_segments=False)
    msg["raw_message"] = [{"type": "reply", "data": {"target_message_id": "90001"}}]
    timeline, stats = BD.build_timeline([msg], bot_qq="10000")
    assert stats["reply_split"] == 0
    assert f"我: {QUOTED}{MY_REPLY}" in timeline, timeline


def test_strip_failure_falls_back_conservatively():
    """正文形态与预期不符（[图片] 前缀导致 startswith 失败）：不拆也不丢。"""
    msg = _reply_msg(is_picture=True, body=f"{QUOTED}{MY_REPLY}")
    timeline, stats = BD.build_timeline([msg], bot_qq="10000")
    assert stats["reply_split"] == 0
    assert QUOTED in timeline and MY_REPLY in timeline, timeline


def test_time_ordering_of_split_lines():
    """被引行插在回复行之前 —— 被引必然早于回复，时序成立。"""
    timeline, _ = BD.build_timeline([_reply_msg()], bot_qq="10000")
    assert timeline.index(f"阿岚: {QUOTED}") < timeline.index(f"我: {MY_REPLY}")


def test_reply_only_message_has_no_empty_line():
    """纯引用无正文（QQ 允许）：只出被引行，不出空「我: 」行。"""
    msg = _reply_msg(body=QUOTED)
    timeline, stats = BD.build_timeline([msg], bot_qq="10000")
    assert f"阿岚: {QUOTED}" in timeline
    assert "我: \n" not in timeline and not timeline.rstrip().endswith("我:"), timeline
    assert stats["reply_split"] == 1


def test_self_quote_check_benefits_from_split():
    """拆分后 ``self_quote_in_brackets`` 不再把别人的原话当成我的话。

    旧形态下被引原文黏在「我:」行里 → 体检把 QUOTED 认成我说的 →
    模型若照素材写「QUOTED 我也困得慌」进引号，体检反而失明。
    """
    timeline, _ = BD.build_timeline([_reply_msg()], bot_qq="10000")
    self_lines = [t for who, t in BD.timeline_speaker_lines(timeline) if who == "我"]
    assert self_lines == [MY_REPLY], self_lines
    # 模型把被引原文错装进引号的形态，现在能被精确抓住
    risk = BD.self_quote_in_brackets(f"老张说「{QUOTED}」，我说我也困得慌。", self_lines)
    assert risk == "", risk
