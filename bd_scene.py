"""better-diary 场景证据模块：把「事件摘要」还原成可核验的对话现场。

设计目标（v1.10.0）：

1. 阶段一只负责挑值得写的事，但它的输出**不能当作事实本身**。这里按原消息
   回查锚点，补上时间、会话、现场互动与我的真实回应；找不到证据的候选会被标记，
   不进入成文。
2. 同一段连续对话可能被阶段一拆成两条摘要（真机 2026-11-12 的守夜片段就是
   两条）。这里在选材前把同一人、同一会话、中间没有第三人插话的连续片段合并成
   episode，避免一个场景占掉两个名额。
3. 只做确定性处理，不调用模型。所有原始消息引用都以短哈希保存，避免把 QQ 消息
   ID 原样写进存档。
"""

from __future__ import annotations

import datetime
import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence

# 现场最多保留几轮、每轮正文多长。上限用于给成文 prompt 设硬边界，
# 不追求把整段聊天塞进去 —— 只保留足以支撑叙事的「前一拍 / 我的回应 / 后续」。
_MAX_TURNS = 10
_MAX_TURN_CHARS = 100
_MAX_EVIDENCE_CHARS = 1200
# 合并同一人连续片段时允许的最大间隔；再长就应视为新场景。
_MERGE_GAP_SECONDS = 90 * 60
# 现场扩展的时间窗：同一段连续聊天里，相邻消息间隔不应超过这个值。
_SCENE_GAP_SECONDS = 20 * 60

_CLAUSE_SPLIT_RE = re.compile(r"[，,、；;。！？!?]")
_SPACE_RE = re.compile(r"\s+")
_NORMALIZE_RE = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ffa-zA-Z0-9]")
_NUM_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:秒|分钟|小时|天|岁|块|元|次|个|人|把|首|楼|度|℃|%)"
)
_STANDALONE_NUM_RE = re.compile(r"(?<![\d.])(\d{3,})(?![\d.])")
_DERIVED_TIME_RE = re.compile(r"(?:关门|开门|截止|熄灯|打卡|上课)[前后]")

_DISTRESS_RE = re.compile(r"噩梦|没安全感|难过|崩溃|哭|疼|不舒服|焦虑|害怕|失眠")
_AFFECTION_RE = re.compile(r"喜欢|爱你|想你|表白|贴贴|抱抱|大姐姐|妹妹|亲亲|520|1314")
_ACHIEVEMENT_RE = re.compile(r"完成|做完|拿到|通过|抽到|发布|杀青|成了|终于|涨到|考过|录完")
_CONFLICT_RE = re.compile(r"撤回|管理员|警告|吵架|怼|生气|举报|禁言|挂人|骂")
_PROGRESS_RE = re.compile(
    r"终于|第一次|结果|后来|改成|答应|拒绝|确定|完成|涨到|实现|开始|结束|"
    r"撤回|道歉|解决|定了|没了|拿到了|做完了|通过了"
)


def _msg_ts(msg: Mapping[str, Any]) -> float:
    try:
        return float(msg.get("timestamp", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _msg_text(msg: Mapping[str, Any]) -> str:
    text = str(msg.get("processed_plain_text") or "").strip()
    if msg.get("is_picture") and not text.startswith("[图片]"):
        text = f"[图片]{text}" if text else "[图片]"
    return text


def _msg_who(msg: Mapping[str, Any]) -> tuple[str, str]:
    info = msg.get("message_info") or {}
    user = info.get("user_info") or {} if isinstance(info, dict) else {}
    return (
        str(user.get("user_id", "") or ""),
        str(user.get("user_nickname", "") or "某人").strip() or "某人",
    )


def _msg_session(msg: Mapping[str, Any]) -> str:
    info = msg.get("message_info") or {}
    group = info.get("group_info") or {} if isinstance(info, dict) else {}
    gid = str(group.get("group_id", "") or "").strip()
    if gid:
        return f"group:{gid}"
    _, who = _msg_who(msg)
    return f"private:{who}"


def _stable_ref(parts: Sequence[Any]) -> str:
    joined = "\x1f".join(str(p or "") for p in parts)
    return "m_" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _norm(text: str) -> str:
    return _NORMALIZE_RE.sub("", str(text or "")).lower()


def _grams(text: str) -> set[str]:
    value = _norm(text)
    if len(value) < 2:
        return {value} if value else set()
    return {value[i : i + 2] for i in range(len(value) - 1)}


def _overlap(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / max(1, min(len(ga), len(gb)))


def _longest_common(a: str, b: str) -> int:
    matcher = SequenceMatcher(None, _norm(a), _norm(b), autojunk=False)
    match = matcher.find_longest_match(0, len(_norm(a)), 0, len(_norm(b)))
    return match.size


def _bucket(text: str) -> str:
    value = str(text or "")
    if _DISTRESS_RE.search(value):
        return "mood"
    if _AFFECTION_RE.search(value):
        return "affection"
    if _ACHIEVEMENT_RE.search(value):
        return "achievement"
    if _CONFLICT_RE.search(value):
        return "conflict"
    return "daily"


def topic_bucket(event: Mapping[str, Any]) -> str:
    """粗粒度题材桶，只用于多样性与跨天重复判定，不对外展示。"""
    return str(event.get("topic_bucket") or "").strip() or _bucket(
        f"{event.get('what') or ''}\n{event.get('quote') or ''}"
    )


def has_progress(text: str) -> bool:
    """是否出现「事情有推进」的显式信号。"""
    return bool(_PROGRESS_RE.search(str(text or "")))


def repeat_reason(event: Mapping[str, Any], recent_events: Iterable[Mapping[str, Any]]) -> str:
    """同一人 + 同一题材 + 没有新进展 → 返回重复原因（空串表示不算重复）。

    这不是硬轮换人物：只要当天有「终于 / 结果 / 改成 / 涨到」这类新进展，
    或者摘要内容明显不同，就允许再次出现。
    """
    who = str(event.get("who") or "").strip()
    if not who or who == "我":
        return ""
    bucket = topic_bucket(event)
    text = f"{event.get('what') or ''}\n{event.get('quote') or ''}"
    progress = has_progress(text)
    for old in recent_events or []:
        if not isinstance(old, Mapping):
            continue
        if str(old.get("who") or "").strip() != who:
            continue
        if topic_bucket(old) != bucket:
            continue
        if progress:
            continue
        old_text = f"{old.get('what') or ''}\n{old.get('quote') or ''}"
        similarity = _overlap(text, old_text)
        if similarity >= 0.32 or bucket in {"affection", "mood"}:
            return f"同人同题材（{bucket}）近期已写过且当天没有新进展"
    return ""


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]], bot_qq: str = ""
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, msg in enumerate(sorted(messages, key=_msg_ts)):
        if not isinstance(msg, Mapping):
            continue
        uid, nickname = _msg_who(msg)
        is_bot = bool(bot_qq and uid == bot_qq)
        who = "我" if is_bot else nickname
        text = _msg_text(msg)
        if not text:
            continue
        session = _msg_session(msg)
        raw_id = str(
            msg.get("message_id")
            or msg.get("msg_id")
            or msg.get("id")
            or ""
        ).strip()
        ref = _stable_ref((session, _msg_ts(msg), uid, text, raw_id))
        out.append(
            {
                "index": index,
                "ref": ref,
                "who": who,
                "nickname": nickname,
                "uid": uid,
                "text": text,
                "ts": _msg_ts(msg),
                "session": _stable_ref((session,)),
                "is_bot": is_bot,
            }
        )
    return out


def _anchor_score(event: Mapping[str, Any], msg: Mapping[str, Any], same_who: bool) -> float:
    what = str(event.get("what") or "")
    score = _overlap(what, str(msg.get("text") or ""))
    if _longest_common(what, str(msg.get("text") or "")) >= 4:
        score += 0.25
    if same_who:
        score += 0.35
    return score


def _find_anchor(
    event: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    self_labels: set[str],
) -> dict[str, Any] | None:
    who = str(event.get("who") or "").strip()
    quote = str(event.get("quote") or "").strip()
    same_who = [
        m for m in messages
        if (m["who"] == "我" and who in self_labels)
        or (m["who"] != "我" and m["who"] == who)
    ]
    pool = same_who or list(messages)
    if quote:
        exact = [m for m in pool if quote in str(m["text"]) or _norm(quote) in _norm(str(m["text"]))]
        if exact:
            return max(exact, key=lambda m: _anchor_score(event, m, m in same_who))
    if not pool:
        return None
    best = max(pool, key=lambda m: _anchor_score(event, m, m in same_who))
    if _anchor_score(event, best, best in same_who) < 0.25:
        return None
    return best


def _scene_turns(
    anchor: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    primary_who: str = "",
) -> list[dict[str, Any]]:
    """从锚点向前后取同一会话的连续现场，保留我的真实回应。

    只保留「主要当事人 + 我」的连续轮次：阶段一的一条事件通常只围绕一个人，
    把后面无关第三人的话一起塞进现场既会污染事实，也会扩大隐私暴露面。
    群像事件由多条独立 episode 表达，不靠一条摘要吞下所有参与者。
    """
    session = anchor["session"]
    anchor_ts = float(anchor["ts"])
    nearby = [
        m for m in messages
        if m["session"] == session and abs(float(m["ts"]) - anchor_ts) <= _SCENE_GAP_SECONDS * 2
    ]
    if primary_who and primary_who != "我":
        nearby = [m for m in nearby if m.get("is_bot") or m.get("who") == primary_who]
    if not nearby:
        return [dict(anchor)]
    pos = min(range(len(nearby)), key=lambda i: abs(float(nearby[i]["ts"]) - anchor_ts))
    start = max(0, pos - 2)
    turns: list[dict[str, Any]] = []
    last_ts = 0.0
    for msg in nearby[start : pos + 8]:
        if turns and float(msg["ts"]) - last_ts > _SCENE_GAP_SECONDS:
            break
        turns.append(dict(msg))
        last_ts = float(msg["ts"])
        if len(turns) >= _MAX_TURNS:
            break
    if anchor not in turns:
        turns.insert(0, dict(anchor))
    return turns


def _time_phrase(ts: float) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(ts)
    except (OSError, OverflowError, ValueError):
        return ""
    return f"{dt.hour:02d}:{dt.minute:02d}"


def _fact_risks_for_event(event: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> list[str]:
    evidence = "\n".join(str(t.get("text") or "") for t in turns)
    compact = _SPACE_RE.sub("", evidence)
    risks: list[str] = []
    what = str(event.get("what") or "")
    for token in _NUM_UNIT_RE.findall(what) + _STANDALONE_NUM_RE.findall(what):
        if _SPACE_RE.sub("", token) not in compact:
            risks.append("摘要里的数量/时间在证据中没有原样依据")
            break
    for phrase in _DERIVED_TIME_RE.findall(what):
        if phrase not in evidence:
            risks.append("摘要包含推算出来的时间关系")
            break
    if not turns:
        risks.append("没有可回查的消息现场")
    return risks


def _trim_turns(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    used = 0
    for turn in turns:
        text = str(turn.get("text") or "")[:_MAX_TURN_CHARS]
        if not text:
            continue
        if out and used + len(text) > _MAX_EVIDENCE_CHARS:
            break
        out.append(
            {
                "who": "我" if turn.get("is_bot") else str(turn.get("who") or "某人"),
                "text": text,
                "ts": float(turn.get("ts") or 0),
                "ref": str(turn.get("ref") or ""),
                "_index": int(turn.get("index") or 0),
                "_session": str(turn.get("session") or ""),
                "_is_bot": bool(turn.get("is_bot")),
            }
        )
        used += len(text)
        if len(out) >= _MAX_TURNS:
            break
    return out


def attach_evidence(
    events: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    *,
    bot_qq: str = "",
    bot_labels: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """给阶段一事件补上现场证据、题材桶与事实风险标记（纯函数）。"""
    normalized = _normalize_messages(messages, bot_qq)
    labels = {str(x).strip() for x in bot_labels if str(x).strip()} | {"我"}
    out: list[dict[str, Any]] = []
    for raw in events or []:
        if not isinstance(raw, Mapping):
            continue
        event = dict(raw)
        who = str(event.get("who") or "").strip()
        if who in labels:
            event["who"] = "我"
        anchor = _find_anchor(event, normalized, labels)
        if anchor is None:
            event["evidence_ok"] = False
            event["evidence_risks"] = ["找不到对应原消息"]
            event["topic_bucket"] = topic_bucket(event)
            out.append(event)
            continue
        turns = _trim_turns(_scene_turns(anchor, normalized, str(event.get("who") or "")))
        bot_actions = [t["text"] for t in turns if t.get("_is_bot")]
        summary_risks = _fact_risks_for_event(event, turns)
        risks = list(summary_risks)
        quote = str(event.get("quote") or "").strip()
        quote_valid = bool(quote) and _norm(quote) in _norm("\n".join(t["text"] for t in turns))
        if quote and not quote_valid:
            event["quote"] = ""
            risks.append("原话无法在证据中逐字回查，已清空引用")
        event.update(
            {
                "evidence_ok": True,
                "evidence_refs": [t["ref"] for t in turns if t.get("ref")],
                "evidence_turns": [
                    {k: v for k, v in t.items() if not k.startswith("_")} for t in turns
                ],
                "anchor_ref": str(anchor.get("ref") or ""),
                "anchor_index": int(anchor.get("index") or 0),
                "anchor_ts": float(anchor.get("ts") or 0),
                "start_ts": float(turns[0].get("ts") or 0) if turns else 0.0,
                "end_ts": float(turns[-1].get("ts") or 0) if turns else 0.0,
                "time_label": _time_phrase(float(anchor.get("ts") or 0)),
                "session_key": str(anchor.get("session") or ""),
                "my_actions": bot_actions[:4],
                "engagement": "direct" if bot_actions else "public",
                "topic_bucket": topic_bucket(event),
                "evidence_risks": risks,
                "quote_valid": quote_valid,
            }
        )
        if summary_risks:
            # 摘要里的推算时间不能进成文；优先用可回查的原话，其次用锚点消息本身，
            # 现场仍完整保留，避免为了「事实正确」把整段经历删掉。
            anchor_text = str(anchor.get("text") or "")
            if quote_valid and quote:
                event["what"] = quote[:80]
            elif anchor_text:
                event["what"] = anchor_text[:80]
            event["summary_rewritten"] = True
        event["scene_completeness"] = scene_completeness(event)
        out.append(event)
    return out


def _between_only_known(
    prev: Mapping[str, Any],
    nxt: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    start = int(prev.get("anchor_index") or 0)
    end = int(nxt.get("anchor_index") or 0)
    if end <= start:
        return False
    allowed = {str(prev.get("who") or ""), "我"}
    for msg in messages:
        if not (start < int(msg["index"]) < end):
            continue
        if str(msg.get("who") or "") not in allowed:
            return False
    return True


def merge_scene_fragments(
    events: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    *,
    bot_qq: str = "",
) -> list[dict[str, Any]]:
    """把同一人、同一会话、中间无第三人插话的连续摘要合并成一个 episode。"""
    normalized = _normalize_messages(messages, bot_qq)
    items = [dict(e) for e in events if isinstance(e, Mapping)]
    if len(items) < 2:
        return items
    ordered = sorted(
        items,
        key=lambda e: (
            float(e.get("anchor_ts") or 0),
            -int(e.get("score") or 0),
        ),
    )
    merged: list[dict[str, Any]] = []
    for event in ordered:
        if not merged:
            merged.append(event)
            continue
        prev = merged[-1]
        same_scene = (
            bool(prev.get("session_key"))
            and prev.get("session_key") == event.get("session_key")
            and str(prev.get("who") or "") == str(event.get("who") or "")
            and str(prev.get("who") or "") != "我"
            and abs(float(event.get("anchor_ts") or 0) - float(prev.get("anchor_ts") or 0))
            <= _MERGE_GAP_SECONDS
            and _between_only_known(prev, event, normalized)
            and (
                topic_bucket(prev) == topic_bucket(event)
                or prev.get("engagement") == event.get("engagement") == "direct"
            )
        )
        if not same_scene:
            merged.append(event)
            continue

        merged_turns: list[dict[str, Any]] = []
        seen: set[str] = set()
        for turn in list(prev.get("evidence_turns") or []) + list(event.get("evidence_turns") or []):
            key = str(turn.get("ref") or f"{turn.get('who')}@{turn.get('ts')}:{turn.get('text')}")
            if key in seen:
                continue
            seen.add(key)
            merged_turns.append(dict(turn))
        merged_turns.sort(key=lambda t: float(t.get("ts") or 0))
        quotes = [
            str(x).strip()
            for x in (prev.get("quote"), event.get("quote"))
            if str(x or "").strip()
        ]
        parts = [
            str(x).strip(" ；;")
            for x in (prev.get("what"), event.get("what"))
            if str(x or "").strip()
        ]
        prev.update(
            {
                "what": "；".join(dict.fromkeys(parts))[:180],
                "quote": quotes[0] if quotes else "",
                "score": max(int(prev.get("score") or 0), int(event.get("score") or 0)),
                "evidence_turns": merged_turns,
                "evidence_refs": [str(t.get("ref") or "") for t in merged_turns if t.get("ref")],
                "start_ts": min(float(prev.get("start_ts") or 0), float(event.get("start_ts") or 0)),
                "end_ts": max(float(prev.get("end_ts") or 0), float(event.get("end_ts") or 0)),
                "my_actions": list(
                    dict.fromkeys(list(prev.get("my_actions") or []) + list(event.get("my_actions") or []))
                )[:6],
                "engagement": "direct" if "direct" in {
                    prev.get("engagement"), event.get("engagement")
                } else "public",
                "fragment_count": int(prev.get("fragment_count") or 1) + int(event.get("fragment_count") or 1),
                "evidence_risks": list(
                    dict.fromkeys(list(prev.get("evidence_risks") or []) + list(event.get("evidence_risks") or []))
                ),
                "evidence_ok": bool(prev.get("evidence_ok", True) and event.get("evidence_ok", True)),
            }
        )
        prev["scene_completeness"] = scene_completeness(prev)
    return merged


def scene_completeness(event: Mapping[str, Any]) -> int:
    """确定性的完整度分数：有原话、有我的回应、现场轮次多、有结果词。"""
    score = 0
    if event.get("quote"):
        score += 2
    actions = event.get("my_actions") or []
    if actions:
        score += 2
    score += min(3, len(event.get("evidence_turns") or []) // 2)
    if has_progress(f"{event.get('what') or ''}\n{event.get('quote') or ''}"):
        score += 1
    if event.get("fragment_count"):
        score += 1
    return score


def _first_sentence(text: str) -> str:
    body = str(text or "").strip()
    if "\n" in body:
        body = body.split("\n", 1)[1].strip()
    return _CLAUSE_SPLIT_RE.split(body, maxsplit=1)[0].strip()


def _last_sentence(text: str) -> str:
    body = str(text or "").strip()
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?])", body) if p.strip()]
    return parts[-1] if parts else body[-60:]


def opening_similarity(text: str, recent_openings: Sequence[str]) -> float:
    """首句与近期开头的最大相似度（只用于风格防重复，不做发布判定）。"""
    first = _first_sentence(text)
    if not first:
        return 0.0
    best = 0.0
    for old in recent_openings or []:
        old_first = _first_sentence(str(old or ""))
        if not old_first:
            continue
        best = max(best, SequenceMatcher(None, _norm(first), _norm(old_first), autojunk=False).ratio())
    return best


def ending_similarity(text: str, recent_openings: Sequence[str]) -> float:
    """末句与近期结尾的最大相似度，防止每天用同一种「评价式收尾」。"""
    last = _last_sentence(text)
    if not last:
        return 0.0
    best = 0.0
    for old in recent_openings or []:
        old_last = _last_sentence(str(old or ""))
        if not old_last:
            continue
        best = max(best, SequenceMatcher(None, _norm(last), _norm(old_last), autojunk=False).ratio())
    return best


_OPENING_TICS = ("眼皮", "脑子", "先记", "记两笔", "睡前", "已经")


def opening_repeats(text: str, recent_openings: Sequence[str], *, threshold: float = 0.56) -> bool:
    """首句相似度过高，或命中两个以上近期固定起手词。"""
    if opening_similarity(text, recent_openings) >= threshold:
        return True
    head = str(text or "").split("\n", 1)[-1][:40]
    tics = [word for word in _OPENING_TICS if word in head]
    if len(tics) < 2:
        return False
    for old in recent_openings or []:
        old_head = str(old or "").split("\n", 1)[-1][:40]
        if len([word for word in tics if word in old_head]) >= 2:
            return True
    return False


def style_repeat_reasons(text: str, recent_openings: Sequence[str]) -> list[str]:
    """返回需要返工的风格重复原因；事实问题不在这里判。"""
    reasons: list[str] = []
    if opening_repeats(text, recent_openings):
        reasons.append("开头与近期日记过于相似")
    if ending_similarity(text, recent_openings) >= 0.68:
        reasons.append("结尾与近期日记过于相似")
    return reasons


def diary_fact_risks(body: str, evidence: str) -> list[str]:
    """成文后的轻量事实核对：数字/时间单位必须能在当天素材里逐字找到。"""
    text = str(body or "")
    lines = text.split("\n")
    if lines and re.match(r"^\s*\d{4}\s*年", lines[0]):
        text = "\n".join(lines[1:])
    compact = _SPACE_RE.sub("", str(evidence or ""))
    risks: list[str] = []
    tokens = list(dict.fromkeys(_NUM_UNIT_RE.findall(text) + _STANDALONE_NUM_RE.findall(text)))
    for token in tokens:
        plain = _SPACE_RE.sub("", token)
        if plain.isdigit() and len(plain) == 4 and 1900 <= int(plain) <= 2100:
            continue
        if plain not in compact:
            risks.append(f"数字/时间「{plain}」在当天素材中没有依据")
    # 天气只在原文有明确说法时才能写成现实天气。歌词、冷、雪花输入法都不算证据。
    source_weather = re.search(r"下雨|下雪|大雨|暴雨|积雪|雨好大|淋湿|被淹", str(evidence or ""))
    body_weather = re.search(r"(?:窗外|外面|屋外)[^。！？\n]{0,8}(?:雨|雪)|雨一直下|下着雨|下雪了", text)
    if body_weather and not source_weather:
        risks.append("正文写了窗外天气，但当天素材没有天气依据")
    return risks


def build_recent_context(diaries: Mapping[str, Any], date_str: str, days: int = 3) -> dict[str, Any]:
    """从本地存档取出近期事件、首句与开场提示，供选材去重和成文防重使用。"""
    try:
        target = datetime.date.fromisoformat(str(date_str))
    except ValueError:
        return {"events": [], "openings": [], "opening_hints": []}
    keys: list[str] = []
    for offset in range(1, max(0, days) + 1):
        keys.append((target - datetime.timedelta(days=offset)).isoformat())
    events: list[dict[str, Any]] = []
    openings: list[str] = []
    hints: list[str] = []
    for key in keys:
        entry = diaries.get(key)
        if not isinstance(entry, Mapping):
            continue
        for event in entry.get("events") or []:
            if isinstance(event, Mapping):
                events.append(dict(event))
        content = str(entry.get("content") or "").strip()
        if content:
            openings.append(content.split("\n", 1)[-1].strip())
        hint = str(entry.get("opening_hint") or "").strip()
        if hint:
            hints.append(hint)
    return {"events": events, "openings": openings[:3], "opening_hints": hints[:3]}
