"""better-diary 纯逻辑模块：时间线构建 + 两阶段 prompt。

本模块不接触 self.ctx，全部为纯函数，可脱机单测。
消息 dict 字段参考 Host `src/plugin_runtime/host/message_utils.py:_session_message_to_dict`：
timestamp(str unix) / processed_plain_text / is_picture /
message_info.user_info.{user_id, user_nickname} / message_info.group_info.group_id
"""

import datetime
import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Tuple

# 单条消息在时间线里的最大文本长度（过长消息截断，保 token 预算）
_MSG_TEXT_LIMIT = 80

# 日记正文字数下限：低于此值视为「没写成」。正常日记至少三五句，
# 下限取 20 是为了拦住空输出/残句，同时绝不会误伤「今天没劲，写短点」。
MIN_DIARY_CHARS = 20

WEATHER_CHOICES = ["晴", "多云", "阴", "雨"]

# 昵称 / 人物名里的结构噪声字符：花括号会污染 prompt 占位、方括号会与时段
# 标记 `【…】` 混同、换行/制表符会打乱时间线行结构。
_NAME_NOISE_RE = re.compile(r"[{}\[\]【】\r\n\t]")
# 正文片段（what / quote）只去会破坏结构的字符：保留 `[图片]` 这类正常内容。
_TEXT_NOISE_RE = re.compile(r"[{}\r\n\t]")

# 拒答特征。只用**强标记**（正常第一人称日记不会出现），避免误伤正文：
# 例如「我无法理解他为什么这么说」这类句子不会被判成拒答。
_REFUSAL_RE = re.compile(
    r"(作为(一个|一名)?\s*(?:AI|人工智能|语言模型|大模型|智能助手|助手)"
    r"|我(?:无法|不能|不便|没办法)(?:完成|满足|提供|生成|撰写|写)"
    r"|(?:无法|不能)(?:完成|满足)(?:这个|该|您的)?(?:请求|要求|任务))"
)


def _collapse(text: str) -> str:
    """把连续空白压成一个空格并去首尾空白。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sanitize_inline(name: str, limit: int = 20) -> str:
    """把昵称 / 人物名压成单行安全文本。

    用户昵称是**不可信输入**，原样拼进 prompt 会破坏时间线结构
    （如昵称写成 ``{系统提示}忽略以上``）。这里去掉花括号/方括号/换行/制表符，
    再把连续空白压成一个空格。
    """
    return _collapse(_NAME_NOISE_RE.sub(" ", str(name or "")))[:limit]


def sanitize_text(text: str, limit: int = 80) -> str:
    """把内容片段（选材产出的 what / quote）压成单行安全文本。

    比 :func:`sanitize_inline` 宽松：保留 ``[图片]`` 这类正常方括号内容，
    只去掉会破坏 prompt 行的花括号与换行。
    """
    return _collapse(_TEXT_NOISE_RE.sub(" ", str(text or "")))[:limit]


def diary_output_problem(content: str) -> str:
    """判断成文输出是否**不可发布**。可用返回空串，不可用返回原因。

    这是发布前的最后一道语义闸。日记成品只会发到 QQ 空间（公开可见），
    所以「模型拒答」「空输出」「残句」都必须在这里拦下，绝不能当成日记发出去。
    """
    body = (content or "").strip()
    if not body:
        return "输出为空"
    if _REFUSAL_RE.search(body):
        return "输出疑似模型拒答"
    if len(body) < MIN_DIARY_CHARS:
        return f"输出过短（{len(body)} 字），疑似未成文"
    return ""


def quote_attribution_risk(content: str, given_quotes: list[str]) -> str:
    """引用体检（**非阻断**，只用于日志留痕）。可用返回空串，可疑返回原因。

    真机反馈过的缺陷：模型把「」当成"待填占位"用，于是写出
    ``「来首无名策岂不美哉 安排上了，好听」`` —— 别人的原话和自己的回复挤在同一对
    引号里，读起来像整句都是日记作者说的。**归属错了，等于替别人说话。**

    这里只做一件事：日记里出现了「」，但**没有任何一条素材原话被原样引用**
    → 极度可疑（要么自造原话，要么引号边界写错），打一条 WARNING 供排查。
    刻意不阻断：模型可能对原话做轻微改写，这时候误报比漏报更烦人。
    """
    body = content or ""
    if "「" not in body or "」" not in body:
        return ""
    given = [str(q or "").strip() for q in given_quotes or []]
    given = [q for q in given if q]
    if not given:
        return ""
    if any(q in body for q in given):
        return ""
    return "日记里有「」但没有原样引用任何一条素材原话（可能自造原话或引号边界写错）"


def _msg_time(msg: Dict[str, Any]) -> float:
    raw = msg.get("timestamp", 0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _msg_text(msg: Dict[str, Any]) -> str:
    return str(msg.get("processed_plain_text") or "").strip()


def _msg_user(msg: Dict[str, Any]) -> Tuple[str, str]:
    info = msg.get("message_info") or {}
    user = info.get("user_info") or {}
    nickname = sanitize_inline(user.get("user_nickname") or "某人") or "某人"
    return (
        str(user.get("user_id", "") or ""),
        nickname,
    )


def _hour_label(hour: int) -> str:
    if 6 <= hour < 12:
        return f"上午{hour}点"
    if 12 <= hour < 18:
        return f"下午{hour}点"
    if hour < 6:
        return f"凌晨{hour}点"
    return f"晚上{hour}点"


def build_timeline(messages: List[Dict[str, Any]], bot_qq: str = "") -> Tuple[str, Dict[str, int]]:
    """把消息 dict 列表构建成带时段标记的时间线文本。

    Returns:
        (timeline_text, stats)，stats 含 total/bot/user 三项。
    """
    stats = {"total": 0, "bot": 0, "user": 0}
    if not messages:
        return "（今天没有聊天记录。）", stats

    parts: List[str] = []
    current_hour = -1
    for msg in sorted(messages, key=_msg_time):
        ts = _msg_time(msg)
        try:
            dt = datetime.datetime.fromtimestamp(ts)
        except (OSError, OverflowError, ValueError):
            continue
        if dt.hour != current_hour:
            parts.append(f"\n【{_hour_label(dt.hour)}】")
            current_hour = dt.hour

        user_id, nickname = _msg_user(msg)
        stats["total"] += 1
        if bot_qq and user_id == bot_qq:
            who = "我"
            stats["bot"] += 1
        else:
            who = nickname
            stats["user"] += 1

        text = _msg_text(msg)
        if msg.get("is_picture"):
            text = (text + " ").strip() or "[图片]"
            text = f"[图片]{text}"
        if not text:
            continue
        if len(text) > _MSG_TEXT_LIMIT:
            text = text[:_MSG_TEXT_LIMIT] + "……"
        parts.append(f"{who}: {text}")

    if len(parts) == 0:
        return "（今天没有聊天记录。）", stats
    return "\n".join(parts), stats


def chunk_text(text: str, chunk_chars: int, max_chunks: int) -> List[str]:
    """把时间线按字符数分块（尽量在换行处切开），最多 max_chunks 块。"""
    if len(text) <= chunk_chars:
        return [text]
    chunks: List[str] = []
    lines = text.split("\n")
    buf: List[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > chunk_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
            if len(chunks) >= max_chunks:
                break
        buf.append(line)
        size += len(line) + 1
    if buf and len(chunks) < max_chunks:
        chunks.append("\n".join(buf))
    return chunks


def build_extract_prompt(date_str: str, timeline_chunk: str) -> str:
    """阶段一：从时间线分块中提取值得写进日记的事件。"""
    return f"""你是日记素材编辑。下面是 {date_str} 的一段聊天记录（时间线格式，"我"指日记作者本人）。

从记录里挑出 0-4 件「值得写进睡前日记的事」，优先级从高到低：
- 有人分享了自己的经历、作品或情绪波动（被安慰、被夸、闹笑话都算）
- 群里真实聊开了的话题（有人接话、有来回的才算）
- "我"本人参与并有来有回的有趣互动
- 忽略：纯表情或"哈哈"刷屏、无实质内容的水群、命令与系统消息、冷场单条发言
- 必须忠实于记录，宁可漏掉也不许脑补细节

只输出 JSON 数组，不要输出任何其他文字。格式：
[{{"who": "人物昵称", "what": "发生了什么，40字以内", "quote": "聊天记录里最有味道的一句原话，没有就填空字符串", "score": 1到5的整数}}]

score 含义：5=当天最值得记的事，1=勉强可以提一句。没有值得写的事就输出 []

聊天记录：
{timeline_chunk}"""


def parse_events(text: str) -> List[Dict[str, Any]]:
    """宽容解析阶段一的 JSON 数组输出。坏输出返回空列表。"""
    if not text or not text.strip():
        return []
    cleaned = text.strip()
    # 去掉可能的代码块围栏
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    events: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            score = int(item.get("score", 1))
        except (TypeError, ValueError):
            score = 1
        events.append(
            {
                "who": sanitize_inline(item.get("who", ""), 20),
                "what": sanitize_text(item.get("what", ""), 80),
                "quote": sanitize_text(item.get("quote", ""), 60),
                "score": max(1, min(5, score)),
            }
        )
    return [e for e in events if e["what"]]


def event_id(date_str: str, event: Mapping[str, Any]) -> str:
    """派生**稳定**的 event_id：同一件事重生成后 ID 不变。

    借鉴 diary_writer 的 `normalize_daily_metadata`——ID 由内容哈希而来、
    与生成次数无关，后续做去重、纠错、引用才有稳定的锚点。
    """
    seed = "\x1f".join(
        (
            str(date_str or ""),
            str(event.get("who") or ""),
            str(event.get("what") or ""),
            str(event.get("quote") or ""),
        )
    )
    return "ev_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def events_to_text(events: List[Dict[str, Any]]) -> str:
    """把精选事件列表渲染成阶段二的素材文本。"""
    if not events:
        return ""
    lines: List[str] = []
    for e in events:
        line = f"- {e['who']}：{e['what']}"
        if e.get("quote"):
            line += f"（原话：「{e['quote']}」）"
        lines.append(line)
    return "\n".join(lines)


# 结构化 META 块的分隔标记。模型在日记正文之后另起一段输出紧凑 JSON，
# 供「证据链 + 跨天连续性」落盘用。**带标记之后的内容一律不进正文**——
# 也就永远不会发布到 QQ 空间（解析失败也一样先切掉再解析）。
META_MARKER = "===META==="

# META 里只取这几个键，其余忽略（防模型乱塞东西）
_META_KEYS = ("topics", "people", "projects", "unresolved")


def split_meta(content: str) -> Tuple[str, Dict[str, Any]]:
    """把「正文 + 结构化 META」拆开。返回 (正文, meta)。

    **fail-open 但有硬边界**：标记之后的内容永远不进正文，哪怕 JSON 解析失败。
    META 缺失或解析失败时 meta 为空 —— 只影响连续性与证据链的丰富度，
    正文照常发布，绝不因此判失败。
    """
    if not content:
        return "", {}
    if META_MARKER not in content:
        return content.strip(), {}
    body, _, tail = content.partition(META_MARKER)
    meta: Dict[str, Any] = {}
    try:
        start, end = tail.find("{"), tail.rfind("}")
        if start != -1 and end > start:
            data = json.loads(tail[start : end + 1])
            if isinstance(data, dict):
                for key in _META_KEYS:
                    value = data.get(key)
                    if isinstance(value, list):
                        items = [sanitize_text(str(v), 40) for v in value if str(v or "").strip()]
                        if items:
                            meta[key] = items[:8]
    except (json.JSONDecodeError, ValueError):
        meta = {}
    return body.strip(), meta


def build_continuity_line(continuity: Mapping[str, Any]) -> str:
    """把跨天连续性渲染成注入文本。没有任何线索时返回空串。

    连续性只用于让日记有「延续感」（昨天聊到一半的话题今天接着提）。
    它**不能证明今天发生了什么** —— 这条纪律写在 prompt 里（见 build_write_prompt 规则 6）。
    """
    if not isinstance(continuity, Mapping) or not continuity:
        return ""
    summary = str(continuity.get("previous_summary") or "").strip()
    events = [str(x).strip() for x in (continuity.get("important_events") or []) if str(x).strip()]
    projects = [str(x).strip() for x in (continuity.get("ongoing_projects") or []) if str(x).strip()]
    topics = [str(x).strip() for x in (continuity.get("ongoing_topics") or []) if str(x).strip()]
    unresolved = [str(x).strip() for x in (continuity.get("unresolved_items") or []) if str(x).strip()]
    if not (summary or events or projects or topics or unresolved):
        return ""
    lines: List[str] = []
    if summary:
        lines.append(f"- 上一次写到：{summary}")
    if topics:
        lines.append("- 还在聊的话题：" + "、".join(topics[:6]))
    if projects:
        lines.append("- 手头还在进行的事：" + "、".join(projects[:6]))
    if unresolved:
        lines.append("- 还没个结果的：" + "、".join(unresolved[:6]))
    if events:
        lines.append("- 最近记下的事：" + "；".join(events[:3]))
    return "前几篇日记留下的线索（仅供参考）：\n" + "\n".join(lines)


def _dedupe(items: List[str], limit: int) -> List[str]:
    """保序去重 + 截断（新的排前面）。"""
    result: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in result:
            continue
        result.append(text)
        if len(result) >= limit:
            break
    return result


def update_continuity(
    previous: Mapping[str, Any],
    events: List[Dict[str, Any]],
    meta: Mapping[str, Any],
    *,
    max_topics: int = 30,
    max_projects: int = 20,
    max_unresolved: int = 20,
    max_events: int = 12,
) -> Dict[str, Any]:
    """把本次成文累积进跨天连续性状态（纯函数，便于单测）。

    与 diary_writer 的 `continuity.py` 同构：只做「累积 + 保序去重 + 截断」，
    **不调用 LLM**。没有 META 时退化成只用选材事件，连续性依然成立。
    """
    prev = previous if isinstance(previous, Mapping) else {}
    meta_map = meta if isinstance(meta, Mapping) else {}
    summaries = [
        str(e.get("what") or "").strip()
        for e in (events or [])
        if isinstance(e, dict)
    ]
    summaries = [s for s in summaries if s]
    return {
        "previous_summary": "；".join(summaries[:3])[:800],
        "important_events": _dedupe(
            summaries + list(prev.get("important_events") or []), max_events
        ),
        "ongoing_projects": _dedupe(
            list(meta_map.get("projects") or []) + list(prev.get("ongoing_projects") or []),
            max_projects,
        ),
        "ongoing_topics": _dedupe(
            list(meta_map.get("topics") or []) + list(prev.get("ongoing_topics") or []),
            max_topics,
        ),
        "unresolved_items": _dedupe(
            list(meta_map.get("unresolved") or []) + list(prev.get("unresolved_items") or []),
            max_unresolved,
        ),
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_write_prompt(
    *,
    date_str: str,
    events_text: str,
    name: str,
    persona: str,
    style_extra: str = "",
    word_target: int = 250,
    max_events: int = 3,
    temporal_anchor: str = "",
    continuity_text: str = "",
) -> str:
    """阶段二：基于精选素材写日记。

    Args:
        temporal_anchor: 非空表示这是**补写**过去的某一天。此时要显式声明
            素材里「今天/昨天/前天」的相对时间基准，否则模型会以真正的今天来解读。
        continuity_text: 跨天连续性线索（见 :func:`build_continuity_line`），
            只用于引出回顾与延续，不能当作「今天发生了什么」的依据。
    """
    name_line = f"我的名字是{name}。" if name else ""
    persona_line = persona or "是一个爱聊天的机器人。"
    style_line = f"\n8. {style_extra}" if style_extra else ""
    anchor_block = ""
    if temporal_anchor.strip():
        anchor_block = (
            f"\n⚠️ 这是**补写** {date_str} 的日记。素材与记忆里出现的"
            f"「今天」「昨天」「前天」一律以 {temporal_anchor.strip()} 为基准来理解，"
            "不是真正的今天。\n"
        )
    continuity_block = f"\n{continuity_text}\n" if continuity_text.strip() else ""
    return f"""{name_line}
我{persona_line}
{anchor_block}
今天是 {date_str}。睡前翻了翻今天的聊天记录，值得记的事整理如下：

{events_text}
{continuity_block}
现在以第一人称写一篇日记。规则（重要，逐条遵守）：
1. 第一行固定为 {date_str}，X。其中 X 从 晴/多云/阴/雨 里选一个贴合今天聊天氛围的，只选一个字都不许多。
2. 只写上面你真想写的事，最多写 {max_events} 件，可以只写一两件。没劲的一天就写短点，三五句也行，绝不硬凑字数。
3. 至少一处用「」引用素材里给的聊天原话，并且**严格遵守引号边界**：
   「」里面只能放**别人说过的原话、一字不改**；我自己的话一律写在引号**外面**。
   错误写法：回了句「来首无名策岂不美哉 安排上了，好听」（把自己的回复也塞进了引号，读起来像整句都是我说的）
   正确写法：有人提了《无名策》，回了句「来首无名策岂不美哉」，我说安排上了，好听。
   另外，凡是引用聊天原话都用「」，不要用 "" 或 '' 代替；能顺带点出是谁说的更好。
4. 禁止出现：开头问候语；结尾总结或展望（如"明天也要加油""真是充实的一天"）；"我意识到/我明白了/我突然发现"句式；连续感叹号；排比句。
5. 像睡前随手写的：句子短，允许有点碎，允许口语和自嘲，想到哪写到哪，但别写成聊天记录复述。
6. **事实纪律**：主观感受和情绪可以自由写；但**没有依据就不得制造**人物、地点、对话、结果或新的事实。
   拿不准的地方用"好像""应该是""记不清了"这类不确定语气带过 —— 文风自由不等于客观事实可以补写。
7. 上面的跨天线索（若有）只能用来引出想法、疑问、期待，或带原日期的回顾；
   它**不能单独证明今天发生了什么**，今天的事必须有「值得记的事」作依据。
8. 全文 {word_target} 字左右（可上下浮动 80 字）。除第一行外就是日记正文：不要标题、不要 markdown、不要"日记"二字开头、不要任何前后缀或解释。{style_line}

日记正文写完后，另起一段只输出下面这两行（用于归档，**不会被发表**）：
{META_MARKER}
{{"topics": ["话题"], "people": ["提到的人"], "projects": ["手头的事"], "unresolved": ["还没个结果的"]}}

日记正文："""


def strip_diary_output(text: str) -> str:
    """清理模型输出：去围栏、去引导语、去首尾空白与包裹引号。"""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    # 去掉常见的引导前缀
    cleaned = re.sub(r"^(?:日记正文|日记|正文)\s*[:：]\s*", "", cleaned)
    # 去掉整段被引号包裹的情况
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'“”":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def ensure_date_line(content: str, date_str: str) -> str:
    """保证第一行是日期行；模型漏写时自动补。

    ⚠️ 空内容会被补成 ``「<日期>，多云。\\n（今天没写出什么来。）」`` 这种**占位文本**，
    它是给「聊天里回看」兜底用的展示文案，**不是可发布的日记**。
    发布前必须先用 :func:`diary_output_problem` 判定，否则占位文本会被当成成品发出去。
    """
    if not content:
        return f"{date_str}，多云。\n（今天没写出什么来。）"
    first_line = content.split("\n", 1)[0]
    year = date_str[:4]
    if year in first_line and ("，" in first_line or "," in first_line):
        return content
    return f"{date_str}，多云。\n{content}"


def date_display(date_str: str) -> str:
    """'2026-09-26' -> '2026年9月26日 星期六'。"""
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{dt.year}年{dt.month}月{dt.day}日 {weekdays[dt.weekday()]}"
