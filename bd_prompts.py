"""better-diary 纯逻辑模块：时间线构建 + 两阶段 prompt。

本模块不接触 self.ctx，全部为纯函数，可脱机单测。
消息 dict 字段参考 Host `src/plugin_runtime/host/message_utils.py:_session_message_to_dict`：
timestamp(str unix) / processed_plain_text / is_picture /
message_info.user_info.{user_id, user_nickname} / message_info.group_info.group_id
"""

import datetime
import json
import re
from typing import Any, Dict, List, Tuple

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


def build_write_prompt(
    *,
    date_str: str,
    events_text: str,
    name: str,
    persona: str,
    style_extra: str = "",
    word_target: int = 250,
    max_events: int = 3,
) -> str:
    """阶段二：基于精选素材写日记。"""
    name_line = f"我的名字是{name}。" if name else ""
    persona_line = persona or "是一个爱聊天的机器人。"
    style_line = f"\n6. {style_extra}" if style_extra else ""
    return f"""{name_line}
我{persona_line}

今天是 {date_str}。睡前翻了翻今天的聊天记录，值得记的事整理如下：

{events_text}

现在以第一人称写一篇日记。规则（重要，逐条遵守）：
1. 第一行固定为「{date_str}，X。」，X 从 晴/多云/阴/雨 里选一个贴合今天聊天氛围的，只选一个字都不许多。
2. 只写上面你真想写的事，最多写 {max_events} 件，可以只写一两件。没劲的一天就写短点，三五句也行，绝不硬凑字数。
3. 至少一处用「」原样引用素材里给的聊天原话。
4. 禁止出现：开头问候语；结尾总结或展望（如"明天也要加油""真是充实的一天"）；"我意识到/我明白了/我突然发现"句式；连续感叹号；排比句。
5. 像睡前随手写的：句子短，允许有点碎，允许口语和自嘲，想到哪写到哪，但别写成聊天记录复述。
6. 全文 {word_target} 字左右（可上下浮动 80 字）。除第一行外就是日记正文：不要标题、不要 markdown、不要"日记"二字开头、不要任何前后缀或解释。{style_line}

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
