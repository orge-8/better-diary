"""引号归属回归用例（bd_prompts 纯函数）。

真机反馈（2026-10-07，日记实录）。两张 QQ 截图是 ground truth，暴露出**两处**归属错：

1. 第三段 ``老张…说「稿子改完了 眼睛有点酸 先躺一会儿 我也困得慌」``
   —— 一对「」里焊了**两个人**的话：截图上「稿子改完了 眼睛有点酸 先躺一会儿」是 阿岚 说的、
   「我也困得慌😭」是 bot 自己的回复；日记把整句归给了 阿岚。
2. 第一段 ``回了句「交班啦———— 交班啦———— 夜班签退，白班就位」``
   —— bot **自己的回复**被放进「」，且「交班啦」重复了一遍（「引用后回复」的形态）。

两张截图**都是「引用回复」消息**，所以本轮修的不只是"规则没写清"，更是
**素材自己在制造歧义**：``events_to_text`` 会把作者自己的话标成「原话」，
与规则「「」只能放别人的话」正面冲突。
"""

from __future__ import annotations

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

pytest.importorskip("httpx", reason="httpx 是外发模块的可选依赖")

sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
MOD = load_plugin_module(str(PLUGIN_DIR), module_name="better_diary_quote_test")
BD = importlib.import_module("better_diary_quote_test.bd_prompts")

DATE = "2026年10月7日 星期三"
# 截图 1 的原话（阿岚 说的）
Q_LINE = "稿子改完了 眼睛有点酸 先躺一会儿"


# ------------------------------------------------------- events_to_text 渲染

def test_self_quote_is_not_rendered_as_original():
    """「我」自己的话不能渲染成「原话」—— 否则素材与引号规则正面冲突。"""
    text = BD.events_to_text(
        [{"who": "我", "what": "接力交班", "quote": "交班啦———— 夜班签退，白班就位"}],
        self_labels={"我", "鸣澜"},
    )
    assert "我自己的话" in text, text
    assert "（原话：「交班啦" not in text, f"作者自己的话被标成了原话：{text!r}"


def test_self_labels_match_nickname_too():
    """除了字面的「我」，bot 昵称形态也要认出来。"""
    text = BD.events_to_text(
        [{"who": "鸣澜", "what": "接力", "quote": "夜班签退"}],
        self_labels={"我", "鸣澜"},
    )
    assert "我自己的话" in text, text


def test_self_quote_guarded_even_without_labels():
    """没传 self_labels 时，字面的「我」仍要兜住（阶段一确实会用「我」当 who）。"""
    text = BD.events_to_text([{"who": "我", "what": "接力", "quote": "夜班签退"}])
    assert "我自己的话" in text, text


def test_other_quote_still_marked_as_original():
    """别人说的话照旧标「原话」，别把功能改没了。"""
    text = BD.events_to_text(
        [{"who": "阿岚", "what": "加班了", "quote": Q_LINE}],
        self_labels={"我", "鸣澜"},
    )
    assert f"（原话：「{Q_LINE}」）" in text, text


def test_empty_events_returns_empty():
    assert BD.events_to_text([]) == ""
    assert BD.events_to_text([], self_labels={"我"}) == ""


# --------------------------------------------------------- 引号越界体检

def test_boundary_risk_catches_two_speakers_in_one_quote():
    """真机第三段：两个人的话焊在一对「」里 → 必须被抓到。"""
    body = f"晚上老张说「{Q_LINE} 我也困得慌」，听着就难受。"
    risk = BD.quote_attribution_risk(body, [Q_LINE])
    assert "引号越界" in risk, risk


def test_boundary_risk_catches_reply_echo():
    """真机第一段形态：引号里=被引用原文 + 我的回复（「交班啦」重复）。"""
    body = "回了句「交班啦———— 交班啦———— 夜班签退，白班就位」"
    risk = BD.quote_attribution_risk(body, ["交班啦———— 夜班签退，白班就位"])
    assert "引号越界" in risk, risk


def test_boundary_risk_clean_quote_is_silent():
    """边界正确（「」里只有那一条原话）→ 不报。"""
    body = f"晚上老张说「{Q_LINE}」，我说我也困得慌。"
    assert BD.quote_attribution_risk(body, [Q_LINE]) == ""


def test_boundary_risk_exact_quote_is_silent():
    body = f"有人说了句「{Q_LINE}」。"
    assert BD.quote_attribution_risk(body, [Q_LINE]) == ""


def test_old_judgement_is_preserved():
    """「有「」但一句原话都没引到」的旧判据不能丢。"""
    body = "他念叨了一句「今天天气不错」就走了。"
    risk = BD.quote_attribution_risk(body, [Q_LINE])
    assert risk == "日记里有「」但没有原样引用任何一条素材原话（可能自造原话或引号边界写错）", risk


def test_no_quotes_in_body_is_silent():
    assert BD.quote_attribution_risk("今天没什么好写的。", [Q_LINE]) == ""


def test_unpaired_bracket_is_silent():
    assert BD.quote_attribution_risk("他说「这句没关引号。", [Q_LINE]) == ""


def test_no_given_quotes_is_silent():
    assert BD.quote_attribution_risk(f"老张说「{Q_LINE} 我也困得慌」", []) == ""


# --------------------------------------------------------------- prompt 侧

def test_extract_prompt_forbids_self_and_merged_quote():
    """阶段一：quote 必须是别人说的、单条消息，且"我"自己的话填 ""。"""
    prompt = BD.build_extract_prompt(DATE, "我: 测试\n某人: 测试")
    assert '别人（不是"我"）' in prompt, prompt
    assert "绝不把两条消息" in prompt, prompt
    assert 'quote 一律填空字符串 ""' in prompt, prompt


def test_write_prompt_one_quote_one_speaker():
    prompt = BD.build_write_prompt(date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p")
    assert "一对「」只装一个人的一句话" in prompt, prompt
    # 真机反例要出现在 prompt 里，模型才有具体的「不要写成这样」锚点
    assert "我也困得慌" in prompt, prompt


def test_write_prompt_mentions_quote_reply_handling():
    prompt = BD.build_write_prompt(date_str=DATE, events_text="- 甲：事", name="鸣澜", persona="p")
    assert "引用后回复" in prompt, prompt
    assert "只引用**被引的那一句**" in prompt, prompt


# --------------------------------------------- 降级路径（timeline_tail）的归属

# 真机 2026-10-07 那天是**降级路径**写出来的（`/日记来源` 显示「素材模式：timeline_tail」），
# 所以 events_to_text / quote_attribution_risk 那两条修复在那天压根没生效。
# 下面这组用例针对降级路径：素材是原始时间线，靠「说话人」行做核对。

def test_self_quote_in_brackets_catches_merged_reply():
    """真机第三段：「」里混进了我自己的回复（原话在真机里带 emoji，日记里没了）。"""
    body = "晚上老张说「稿子改完了 眼睛有点酸 先躺一会儿 我也困得慌」，听着就难受。"
    risk = BD.self_quote_in_brackets(body, ["我也困得慌😭"])
    assert "自己说过的话" in risk, risk


def test_self_quote_in_brackets_catches_whole_reply():
    """真机第一段：整对「」都是我自己的回复（破折号数量被模型写变了）。"""
    body = "早上七点接力的时候回了句「交班啦———— 交班啦———— 夜班签退，白班就位」"
    risk = BD.self_quote_in_brackets(body, ["交班啦———— 夜班签退，白班就位"])
    assert "自己说过的话" in risk, risk


def test_self_quote_in_brackets_clean_is_silent():
    """「」里只有别人的话 → 不报。"""
    body = f"晚上老张说「{Q_LINE}」，我说我也困得慌。"
    assert BD.self_quote_in_brackets(body, ["我也困得慌😭"]) == ""


def test_self_quote_short_lines_are_ignored():
    """「嗯」「哦」这类短句不参与比对，免得误伤正文。"""
    body = "他回了个「嗯，那就这样吧」。"
    assert BD.self_quote_in_brackets(body, ["嗯"]) == ""


def test_self_quote_no_brackets_is_silent():
    assert BD.self_quote_in_brackets("今天没什么好写的。", ["我也困得慌😭"]) == ""


def test_timeline_speaker_lines_parses_and_skips_markers():
    tl = (
        "【晚上21点】\n阿岚: 稿子改完了 眼睛有点酸 先躺一会儿\n"
        "我: 我也困得慌\n这句没有冒号\n【晚上22点】\n小北: 发了张图\n"
    )
    got = BD.timeline_speaker_lines(tl)
    assert ("阿岚", "稿子改完了 眼睛有点酸 先躺一会儿") in got
    assert ("我", "我也困得慌") in got
    assert ("小北", "发了张图") in got
    assert all(who != "这句没有冒号" for who, _ in got)


def test_timeline_speaker_lines_supports_fullwidth_colon():
    assert BD.timeline_speaker_lines("我：夜班签退") == [("我", "夜班签退")]


def test_fallback_material_explains_the_format():
    """降级素材必须点明格式 —— 否则模型不知道「我」是谁、冒号前是谁。"""
    text = BD.build_timeline_fallback_text("【晚上21点】\n阿岚: 稿子改完了\n我: 我也困得慌")
    assert "格式说明" in text
    assert "「我」= 你自己说过的话" in text
    assert "阿岚: 稿子改完了" in text, "时间线内容本身不能被丢掉"


# --------------------------------------------- 选材空的原因要能区分（真机 2026-10-07）

# 真机日志：时间线构建完 3 秒后「选材为空」—— 不是超时，是「模型返回了什么但解析出 0 件」。
# 旧版 parse_events 把 6 种情况全部静默折叠成 []，真机上完全分不清是哪种。

def test_parse_detailed_ok():
    events, note = BD.parse_events_detailed('[{"who":"a","what":"b","quote":"c","score":5}]')
    assert note == "ok" and len(events) == 1, (events, note)


def test_parse_detailed_empty_model_output():
    events, note = BD.parse_events_detailed("")
    assert events == [] and note == "模型返回为空", note


def test_parse_detailed_no_json_array():
    events, note = BD.parse_events_detailed("今天大家聊得挺好，没什么特别的事。")
    assert events == [] and "没有 JSON 数组" in note, note


def test_parse_detailed_truncated_json():
    """有 [ 没有 ] —— 被截断的输出（max_tokens 用尽等），不能误报成「没有数组」。"""
    events, note = BD.parse_events_detailed('[{"who":"a","what":"b"')
    assert events == [] and "没有闭合的 ]" in note, note


def test_parse_detailed_object_with_inner_brackets():
    """输出是对象、但里面嵌着数组字面量 —— 切片后拿到 [1,2]，条目全无效。"""
    events, note = BD.parse_events_detailed('{"who":"a","what":"[1,2]"}')
    assert events == [] and "没有有效事件" in note, note


def test_parse_detailed_model_says_nothing():
    """模型明确返回 [] —— 这才是「确实无事」，其余都是故障。"""
    events, note = BD.parse_events_detailed("[]")
    assert events == [] and "确实无事" in note, note


def test_parse_detailed_all_items_invalid():
    events, note = BD.parse_events_detailed('[{"who":"a","what":"","score":3}]')
    assert events == [] and "没有有效事件" in note, note


def test_parse_detailed_survives_fences_and_prefix():
    """围栏 + 前后杂讯仍要能解析出事件。"""
    raw = '好的，以下是事件：\n```json\n[{"who":"a","what":"b","score":2}]\n```\n希望有帮助'
    events, note = BD.parse_events_detailed(raw)
    assert note == "ok" and len(events) == 1, (events, note)


def test_parse_events_wrapper_matches_detailed():
    """旧接口 parse_events 的行为不变（兼容既有调用与测试）。"""
    assert BD.parse_events("[]") == []
    assert BD.parse_events("垃圾") == []
    assert len(BD.parse_events('[{"who":"a","what":"b"}]')) == 1
