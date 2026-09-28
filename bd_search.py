"""better-diary 本地检索：/问日记 与 /那年今日 的纯逻辑（零 LLM、零依赖）。

借鉴 astrbot_plugin_diary_writer 的 ``diary/retrieval.py``：所有检索都是
**本地字符串匹配 + 日期比对**，不调用模型 —— 数据来源是 ``diaries.json``
（v1.2.8 起含证据链 events 与 META 标签），所以结果天然可追溯。

要点：
- 检索范围 = 日期 + 正文 + 选材事件（who/what/quote）+ META 标签
- 多个关键词用空格隔开，**全部命中**才算匹配（AND，减少常见词噪音）
- ``/那年今日`` 只看**年份更早**的同月同日（今年的今天不算「那年」）
- 本模块不接触 self.ctx，可脱机单测
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

SEARCH_LIMIT_DEFAULT = 5

# 摘要截断长度（终端友好，也避免把整篇日记刷进聊天）
_SNIPPET_LEN = 60

# META 里参与检索的键
_META_SEARCH_KEYS = ("topics", "people", "projects", "unresolved")


def _meta_list(entry: Dict[str, Any], key: str) -> List[str]:
    meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
    values = meta.get(key) if isinstance(meta.get(key), list) else []
    return [str(v).strip() for v in values if str(v).strip()]


def _event_texts(entry: Dict[str, Any]) -> List[str]:
    """摊平选材事件的 who / what / quote，供命中定位与摘要用。"""
    events = entry.get("events") if isinstance(entry.get("events"), list) else []
    texts: List[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for key in ("who", "what", "quote"):
            value = str(event.get(key) or "").strip()
            if value:
                texts.append(value)
    return texts


def _haystack(date_str: str, entry: Dict[str, Any]) -> str:
    """该日期日记的检索范围（统一 casefold）。"""
    parts: List[str] = [date_str, str(entry.get("content") or "")]
    parts.extend(_event_texts(entry))
    for key in _META_SEARCH_KEYS:
        parts.extend(_meta_list(entry, key))
    return " ".join(parts).casefold()


def _one_line(text: str, limit: int = _SNIPPET_LEN) -> str:
    """压成单行并截断（换行会打乱列表结构）。"""
    collapsed = " ".join(str(text or "").split())
    return collapsed[:limit] + ("…" if len(collapsed) > limit else "")


def _summary(entry: Dict[str, Any], terms: List[str]) -> str:
    """挑一条最有信息量的摘要：命中关键词的事件 > 第一条事件 > META 话题 > 正文开头。"""
    lowered = [t.casefold() for t in terms if t.strip()]
    if lowered:
        for text in _event_texts(entry):
            if any(t in text.casefold() for t in lowered):
                return _one_line(text)
    events = entry.get("events") if isinstance(entry.get("events"), list) else []
    for event in events:
        if isinstance(event, dict) and str(event.get("what") or "").strip():
            return _one_line(str(event["what"]))
    topics = _meta_list(entry, "topics")
    if topics:
        return _one_line("、".join(topics))
    return _one_line(str(entry.get("content") or ""))


def search_diaries(
    archive: Dict[str, Any], query: str, limit: int = SEARCH_LIMIT_DEFAULT
) -> List[Dict[str, str]]:
    """按关键词搜索历史日记。多关键词 AND，按日期倒序取前 ``limit`` 条。

    Returns:
        [{"date": "2026-09-20", "summary": "…"}]，无命中返回空列表。
    """
    terms = [t for t in str(query or "").replace("\u3000", " ").split() if t.strip()]
    if not terms:
        return []
    lowered = [t.casefold() for t in terms]
    hits: List[Dict[str, str]] = []
    for date_str in sorted(archive, reverse=True):
        entry = archive.get(date_str)
        if not isinstance(entry, dict) or not str(entry.get("content") or "").strip():
            continue
        haystack = _haystack(date_str, entry)
        if not all(term in haystack for term in lowered):
            continue
        hits.append({"date": date_str, "summary": _summary(entry, terms)})
        if len(hits) >= max(1, int(limit)):
            break
    return hits


def on_this_day(
    archive: Dict[str, Any],
    today: datetime.date | None = None,
    limit: int = SEARCH_LIMIT_DEFAULT,
) -> List[Dict[str, str]]:
    """往年同月同日的日记（**年份必须更早**），按日期倒序。"""
    today = today or datetime.date.today()
    results: List[Dict[str, str]] = []
    for date_str in sorted(archive, reverse=True):
        try:
            value = datetime.date.fromisoformat(str(date_str))
        except ValueError:
            continue
        if value.year >= today.year:
            continue
        if (value.month, value.day) != (today.month, today.day):
            continue
        entry = archive.get(date_str)
        if not isinstance(entry, dict) or not str(entry.get("content") or "").strip():
            continue
        results.append({"date": date_str, "summary": _summary(entry, [])})
        if len(results) >= max(1, int(limit)):
            break
    return results


def format_hits(title: str, hits: List[Dict[str, str]], empty_hint: str) -> str:
    """把检索结果渲染成聊天文本。空结果时直接给提示语。"""
    if not hits:
        return empty_hint
    lines = [f"{title}（{len(hits)} 天）："]
    for item in hits:
        lines.append(f"- {item.get('date', '?')}：{item.get('summary', '')}")
    return "\n".join(lines)
