"""本地零 LLM 检索（bd_search）的回归用例：/问日记 与 /那年今日。"""

from __future__ import annotations

import asyncio
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

from fakehost import FakeHost, build_context, load_plugin_module  # noqa: E402

pytest.importorskip("httpx", reason="httpx 是外发模块的可选依赖")

sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
MOD = load_plugin_module(str(PLUGIN_DIR), module_name="better_diary_search_test")
SEARCH = importlib.import_module("better_diary_search_test.bd_search")

ARCHIVE = {
    "2026-09-20": {
        "content": "聊到萤火虫，糊糊的，像做梦。",
        "events": [{"who": "Acer", "what": "拍了萤火虫照片", "quote": "像做梦一样",
                    "score": 3, "event_id": "ev_x"}],
        "meta": {"topics": ["萤火虫"], "people": ["Acer"], "projects": [], "unresolved": []},
    },
    "2026-09-26": {
        "content": "聊到一首歌，前奏一响就跪了。",
        "events": [{"who": "Тоша", "what": "丢来一首洛天依", "quote": "死在春天里",
                    "score": 5, "event_id": "ev_y"}],
        "meta": {"topics": ["歌"], "people": ["Тоша"], "projects": [], "unresolved": ["歌名"]},
    },
    "2025-09-26": {
        "content": "去年今天的事。",
        "events": [{"who": "旧岁逢春", "what": "说了晚上好", "quote": "", "score": 2,
                    "event_id": "ev_z"}],
        "meta": {"topics": [], "people": ["旧岁逢春"], "projects": [], "unresolved": []},
    },
}


def test_search_single_term_hits_event():
    hits = SEARCH.search_diaries(ARCHIVE, "萤火虫")
    assert [h["date"] for h in hits] == ["2026-09-20"]
    # 摘要优先取「命中关键词的那件事」，而不是正文开头
    assert "萤火虫" in hits[0]["summary"]


def test_search_multi_term_is_and():
    assert [h["date"] for h in SEARCH.search_diaries(ARCHIVE, "歌 洛天依")] == ["2026-09-26"]
    # 只命中其一不算
    assert SEARCH.search_diaries(ARCHIVE, "歌 萤火虫") == []


def test_search_no_match_and_empty_query():
    assert SEARCH.search_diaries(ARCHIVE, "不存在的词") == []
    assert SEARCH.search_diaries(ARCHIVE, "   ") == []
    assert SEARCH.search_diaries(ARCHIVE, "") == []


def test_search_ignores_contentless_entries():
    broken = {**ARCHIVE, "2026-01-01": {"content": "", "events": [], "meta": {}}}
    assert [h["date"] for h in SEARCH.search_diaries(broken, "萤火虫")] == ["2026-09-20"]


def test_search_matches_meta_tags_too():
    # 「歌名」只出现在 META 的 unresolved 里，正文与事件里都没有
    hits = SEARCH.search_diaries(ARCHIVE, "歌名")
    assert [h["date"] for h in hits] == ["2026-09-26"]


def test_search_sorted_desc_with_limit():
    hits = SEARCH.search_diaries(ARCHIVE, "Acer", limit=1)
    assert [h["date"] for h in hits] == ["2026-09-20"]


def test_on_this_day_only_earlier_years():
    assert [h["date"] for h in SEARCH.on_this_day(ARCHIVE, datetime.date(2026, 9, 26))] == ["2025-09-26"]
    # 同年的今天不算「那年」；未来年份也不算
    assert SEARCH.on_this_day(ARCHIVE, datetime.date(2025, 9, 26)) == []
    # 未来年份的同月同日都算「那年」，按日期倒序
    future = SEARCH.on_this_day(ARCHIVE, datetime.date(2027, 9, 26))
    assert [h["date"] for h in future] == ["2026-09-26", "2025-09-26"]


def test_on_this_day_ignores_malformed_dates():
    broken = {**ARCHIVE, "not-a-date": {"content": "x", "events": [], "meta": {}}}
    assert [h["date"] for h in SEARCH.on_this_day(broken, datetime.date(2026, 9, 26))] == ["2025-09-26"]


def test_format_hits_empty_and_non_empty():
    assert SEARCH.format_hits("t", [], "没有") == "没有"
    text = SEARCH.format_hits("问「萤火虫」", SEARCH.search_diaries(ARCHIVE, "萤火虫"), "没有")
    assert text.startswith("问「萤火虫」（1 天）：")
    assert "- 2026-09-20：" in text


def test_ask_command_hits_and_hints(tmp_path):
    plugin = MOD.create_plugin()
    host = FakeHost()
    ctx = build_context("org.orge-8.better-diary", rpc_call=host.rpc_call)
    plugin._set_context(ctx)
    plugin.set_plugin_config({"plugin": {"enabled": True, "config_version": "1.0.0"},
                              "schedule": {"enabled": False},
                              "security": {"admin_ids": ["123456789"]}})
    plugin._data_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "diaries.json").write_text(
        __import__("json").dumps(ARCHIVE, ensure_ascii=False), encoding="utf-8")

    # v1.3.4 起只读命令也受权限闸保护，测试里用管理员身份触发
    admin = {"user_id": "123456789"}
    asyncio.run(plugin.cmd_diary_ask(matched_groups={"query": "萤火虫"}, stream_id="s1", **admin))
    joined = "\n".join(host.sent_texts)
    assert "2026-09-20" in joined and "萤火虫" in joined

    before = len(host.sent_texts)
    asyncio.run(plugin.cmd_diary_ask(matched_groups={}, stream_id="s1", **admin))
    assert "问什么" in host.sent_texts[before]
