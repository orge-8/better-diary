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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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


# 日记里的「」引用片段。正文里的「」成对出现；不成对的由正则自然忽略。
_QUOTE_SPAN_RE = re.compile(r"「([^「」]*)」")

# 归属比对时保留的字符集（中日韩汉字/假名、字母、数字）。
# 标点、空格、emoji、破折号一律丢掉 —— 它们是模型复述时最容易变样的部分。
_PLAIN_KEEP_RE = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ffa-zA-Z0-9]")


def _clip(text: str, limit: int = 30) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


def quote_attribution_risk(content: str, given_quotes: list[str]) -> str:
    """引用体检（**非阻断**，只用于日志留痕）。可用返回空串，可疑返回原因。

    真机反馈过两类缺陷：

    1. 模型把「」当成"待填占位"用，写出 ``「来首晚风岂不美哉 安排上了，好听」``
       —— 别人的原话和自己的回复挤在同一对引号里。**归属错了，等于替别人说话。**
    2. 真机 2026-10-07：一对「」里焊进了**两个不同人**的话
       （被引用人的原话 + 作者自己的回复），整句却归给了被引用人。

    两类都靠同一条判据抓：**测引号边界**。
    若某个「」的内容**包含**某条素材原话、却不等于它 → 边界画错了（越界）。
    比"有没有原样引用"更强 —— 后者只能发现"一句都没引到"，对归属错完全失明。
    刻意不阻断：模型可能对原话做轻微改写，这时候误报比漏报更烦人。
    """
    body = content or ""
    if "「" not in body or "」" not in body:
        return ""
    given = [str(q or "").strip() for q in given_quotes or []]
    given = [q for q in given if q]
    if not given:
        return ""
    # ① 引号越界：某对「」里除了一条素材原话，还混进了别的内容
    for span in (m.group(1).strip() for m in _QUOTE_SPAN_RE.finditer(body)):
        for q in given:
            if span and q != span and q in span:
                return (
                    f"引号越界：「{_clip(span)}」里除素材原话「{_clip(q)}」外还混进了别的内容"
                    "（可能跨了说话人）"
                )
    # ② 一句原话都没原样引到
    if any(q in body for q in given):
        return ""
    return "日记里有「」但没有原样引用任何一条素材原话（可能自造原话或引号边界写错）"


def _plain(text: str) -> str:
    """归到"只留中日韩/字母/数字"的比对形态。

    为什么要归一：真机消息里满是 emoji、空格、破折号，而模型写进日记时常常
    顺手丢掉 emoji、或把破折号数量写变了。逐字比对会在这些噪声上假阴性，
    所以归属核对一律先 `_plain` 再比对。
    """
    return _PLAIN_KEEP_RE.sub("", str(text or ""))


def self_quote_in_brackets(content: str, self_lines: list[str], min_len: int = 5) -> str:
    """日记里的「」装进了**作者自己说过的话** → 归属错（把自己的话当别人的引用）。

    真机 2026-10-07 的两处都属这一类：
    - ``老张…说「稿子改完了 眼睛有点酸 先躺一会儿 我也困得慌」`` —— 后半句是我自己的回复；
    - ``回了句「交班啦———— 夜班签退，白班就位」`` —— 整段都是我自己的回复。

    比"引号越界"更直接：越界要靠"素材原话被包含"来推断，而这条拿**作者本人说过的话**
    直接比对，不依赖素材里有没有抽到原话 —— 所以**降级路径（timeline_tail）也能查**。
    非阻断，只留痕。

    Args:
        self_lines: 作者本人说过的话（时间线里 ``我:`` 那几行的内容）。
        min_len: 短于此长度的原文不参与比对 —— 「嗯」「哦」这类短句容易误伤正文。
    """
    body = content or ""
    if "「" not in body or "」" not in body:
        return ""
    spans = [m.group(1) for m in _QUOTE_SPAN_RE.finditer(body)]
    plain_spans = [_plain(s) for s in spans]
    if not any(plain_spans):
        return ""
    for raw in self_lines:
        plain_self = _plain(raw)
        if len(plain_self) < min_len:
            continue
        for span, plain_span in zip(spans, plain_spans):
            if plain_span and plain_self in plain_span:
                return (
                    "引号里装了作者自己说过的话："
                    f"「{_clip(span)}」里含我的原话「{_clip(raw)}」（自己的话必须写在「」外面）"
                )
    return ""


def build_timeline_fallback_text(tail: str) -> str:
    """选材跑不通时的降级素材：原始时间线 + **格式说明**。

    ⚠️ 格式说明不能省：``build_write_prompt`` 从头到尾没解释过时间线是什么格式
    （每行 ``昵称: 内容``）。不点明「冒号前是说话人、『我』是我自己」，模型就只能靠猜，
    引号归属于是错了 —— 真机 2026-10-07 那篇就是降级路径写出来的。
    """
    return (
        "（选材环节没跑通，下面是当天聊天记录的末尾片段，从中挑你有印象的写）\n"
        "格式说明：每行是「说话人: 内容」，冒号前面是谁说的；其中「我」= 你自己说过的话，"
        "其余昵称是别人说的。\n" + str(tail or "")
    )


def timeline_speaker_lines(timeline: str) -> list[tuple[str, str]]:
    """把时间线文本拆回 ``(说话人, 内容)`` 列表，供归属核对用。

    时间线的行形如 ``昵称: 内容`` / ``我: 内容``，另有 ``【晚上21点】`` 这类时段标记
    （以 ``【`` 开头，直接跳过）。解析不出冒号的行也跳过。
    """
    out: list[tuple[str, str]] = []
    for raw_line in str(timeline or "").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("【"):
            continue
        who, sep, text = line.partition(":")
        if not sep:
            who, sep, text = line.partition("：")
        if not sep:
            continue
        text = text.strip()
        if text:
            out.append((who.strip(), text))
    return out


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


def _reply_info_from_segments(
    raw_segments: Any, bot_qq: str = ""
) -> Tuple[str, str] | None:
    """从消息 dict 的 ``raw_message`` 段列表里提取 ``(被引说话人, 被引原文)``。

    背景（2026-10-09 源码级核实）：适配器（napcat-adapter 与 1.3.0+ 统一 QQ 连接器
    逻辑同源）会把引用目标回查成 ``target_message_content`` + 真实说话人写进 reply 段，
    而 Host 的 ``process_reply_component`` 分支①把它**不加前缀**直接拼进
    ``processed_plain_text`` —— 于是「被引原文 + 我的回复」无缝黏在一条消息里，
    说话人却记成了回复者。真机 2026-10-07「稿子改完了…先躺一会儿 我也困得慌」的归属错，
    根子就在这一层。

    本函数只认**回查成功**的 reply 段（有 ``target_message_content``）；
    回查失败时段会被适配器整个丢弃，这里自然返回 ``None``，维持旧逻辑兜底。
    群名片优先于昵称；被引的若是 bot 自己（极少见），说话人记为「我」。
    """
    if not isinstance(raw_segments, (list, tuple)):
        return None
    for seg in raw_segments:
        if not isinstance(seg, Mapping) or str(seg.get("type") or "") != "reply":
            continue
        data = seg.get("data")
        if not isinstance(data, Mapping):
            return None
        quoted = str(data.get("target_message_content") or "").strip()
        if not quoted:
            return None
        sender_id = str(data.get("target_message_sender_id") or "").strip()
        if bot_qq and sender_id and sender_id == bot_qq:
            who = "我"
        else:
            who = (
                sanitize_inline(
                    str(
                        data.get("target_message_sender_cardname")
                        or data.get("target_message_sender_nickname")
                        or sender_id
                        or "某人"
                    ),
                    20,
                )
                or "某人"
            )
        return who, quoted
    return None


def build_timeline(messages: List[Dict[str, Any]], bot_qq: str = "") -> Tuple[str, Dict[str, int]]:
    """把消息 dict 列表构建成带时段标记的时间线文本。

    引用回复拆分（v1.5.1）：消息带**回查成功**的 reply 段时，把被引原文按其真实
    说话人**独立成行**（插在回复行之前 —— 被引消息必然早于回复，时序成立），
    回复行只留回复者自己的话。这样：

    - 被引原文不会因为黏在 bot 的消息里而被记成「我」说的；
    - ``self_quote_in_brackets`` 拿「我:」行做核对时，不再把别人原话当成我的话。

    被引原文若已在时间线里出现过（绝大多数引用回的是当天消息，原行本来就在），
    则只剥离、不重复插行。剥离失败（正文形态与预期不符，如带 [图片] 前缀）时
    **保守回退旧行为**，绝不因为拆分逻辑把消息弄丢。

    Returns:
        (timeline_text, stats)。stats 含 total/bot/user，另加 ``reply_split``
        （成功拆分并剥离的引用回复条数，供日志回读）。
    """
    stats = {"total": 0, "bot": 0, "user": 0, "reply_split": 0}
    if not messages:
        return "（今天没有聊天记录。）", stats

    parts: List[str] = []
    seen_plain: set[str] = set()

    def _emit(who: str, body: str) -> None:
        """渲染一行并登记其归一形态（供被引原文判重）。"""
        text = str(body or "").strip()
        if not text:
            return
        if len(text) > _MSG_TEXT_LIMIT:
            text = text[:_MSG_TEXT_LIMIT] + "……"
        parts.append(f"{who}: {text}")
        seen_plain.add(_plain(text))

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

        # 引用回复拆分：Host 分支①把被引原文裸拼在正文开头，先精确剥离。
        # startswith 必然成立（分支①返回的就是同一个字符串）；失败就保守走旧行为。
        reply = _reply_info_from_segments(msg.get("raw_message"), bot_qq)
        if reply and text.startswith(reply[1]):
            quoted_who, quoted_text = reply
            rest = text[len(quoted_text) :].strip()
            plain_quoted = _plain(quoted_text)
            if plain_quoted and plain_quoted not in seen_plain:
                _emit(quoted_who, quoted_text)
            if rest:
                _emit(who, rest)
            stats["reply_split"] += 1
            continue

        _emit(who, text)

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

从记录里挑出 0-6 件「值得写进睡前日记的事」（**多给几件没关系的，宁可多给也别漏** ——
候选少了后面就没得挑，真机 2026-10-18 只给了 2 件，结果当天最浓的一段整个没进日记）。
优先级从高到低：
- 有人**直接对"我"**说的话（尤其是带 @我 的）：被表白、被安慰、被逗、被托付，这类必须优先选
- 有人分享了自己的经历、作品或情绪波动（被安慰、被夸、闹笑话都算）
- 有人**做成了某件事**（卖了多少、拿到了什么反馈、被谁认可了）—— 有结果的比纯玩梗更值得记
- 群里真实聊开了的话题（有人接话、有来回的才算）
- "我"本人参与并有来有回的有趣互动
- 纯情绪反应（全场爆笑、集体哀嚎、齐刷刷刷同一句话）若是当天最有记忆点的，也算值得记的事
- 忽略：纯表情或"哈哈"刷屏、无实质内容的水群、命令与系统消息、冷场单条发言
- 必须忠实于记录，宁可漏掉也不许脑补细节
- 每条的"who"必须是记录里**逐字**出现的昵称，不许改写、简写、拼错

只输出 JSON 数组，不要输出任何其他文字。格式：
[{{"who": "人物昵称", "what": "发生了什么，40字以内", "quote": "聊天记录里最有味道的一句原话，没有就填空字符串", "score": 1到5的整数}}]

quote 的三条硬性要求（很重要，写错的话整篇日记会替别人说话）：
- 只能是**别人（不是"我"）**说的一句原话，**一字不改**照抄；
- 必须来自**同一个人、同一条消息**，**绝不把两条消息、或两个人说的话拼在一起**；
- 若最有味道的是「我」自己说的、或本段没有合适的别人原话，quote 一律填空字符串 ""。

score 含义：5=当天最值得记的事，1=勉强可以提一句。没有值得写的事就输出 []

聊天记录：
{timeline_chunk}"""


def parse_events_detailed(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """解析阶段一的 JSON 数组输出，并**区分「确实无事」与「解析失败」**。

    旧版 :func:`parse_events` 把 6 种情况全部折叠成 ``[]``：
    空输入 / 没有 JSON 数组 / JSON 坏 / 不是数组 / 条目全无效 / 模型真返回 []。
    前五种是**故障**，最后一种才是「确实无事」—— 混在一起时，
    真机上「选材为空，降级用时间线末尾」就永远分不清是哪种（2026-10-07 那天卡住的正是这个）。

    Returns:
        ``(events, note)``。``note`` 为 ``ok``（有事件）或人类可读的空原因。
    """
    if not text or not text.strip():
        return [], "模型返回为空"
    cleaned = text.strip()
    # 去掉可能的代码块围栏
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end <= start:
        if start != -1:
            return [], "输出里有 [ 但没有闭合的 ]（可能被截断）"
        return [], f"输出里没有 JSON 数组（{len(cleaned)} 字）"
    try:
        data = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return [], f"JSON 解析失败（截取 {end + 1 - start} 字）"
    if not isinstance(data, list):
        return [], f"输出不是数组（是 {type(data).__name__}）"
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
    valid = [e for e in events if e["what"]]
    if not valid:
        if not data:
            return [], "模型明确返回空数组（确实无事）"
        return [], f"数组 {len(data)} 条里没有有效事件（缺 what）"
    return valid, "ok"


def parse_events(text: str) -> List[Dict[str, Any]]:
    """宽容解析阶段一的 JSON 数组输出。坏输出返回空列表。

    只关心事件本身时用这个；要在日志里区分「确实无事」与「解析失败」时，
    用 :func:`parse_events_detailed`。
    """
    return parse_events_detailed(text)[0]


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


def events_to_text(events: List[Dict[str, Any]], self_labels: Iterable[str] = ()) -> str:
    """把精选事件列表渲染成阶段二的素材文本。

    ``self_labels`` 是**日记作者本人**的称呼集合（"我" + bot 昵称）。命中的事件
    **不**把 quote 渲染成「原话」形态 —— 否则素材会出现
    ``- 我：…（原话：「我自己的话」）``，与规则「「」只能放**别人**说过的原话」
    直接冲突。真机实测（2026-10-07）：模型照着素材，把作者自己的回复也塞进了「」，
    读起来像别人的话。素材与规则打架时，模型多半信素材 —— 所以要让素材自己别制造歧义。
    """
    if not events:
        return ""
    labels = {str(s).strip() for s in (self_labels or ()) if str(s).strip()}
    lines: List[str] = []
    for e in events:
        who = str(e.get("who") or "").strip()
        line = f"- {who}：{e['what']}"
        if e.get("quote"):
            if who in labels or who == "我":
                # 作者自己的话：信息照留，但**明确禁止**用「」引用它
                line += f"（我自己的话，写日记时不要用「」引用：「{e['quote']}」）"
            else:
                line += f"（原话：「{e['quote']}」）"
        lines.append(line)
    return "\n".join(lines)


# 结构化 META 块的分隔标记。模型在日记正文之后另起一段输出紧凑 JSON，
# 供「证据链 + 跨天连续性」落盘用。**带标记之后的内容一律不进正文**——
# 也就永远不会发布到 QQ 空间（解析失败也一样先切掉再解析）。
META_MARKER = "===META==="

# META 里只取这几个键，其余忽略（防模型乱塞东西）
_META_KEYS = ("topics", "people", "projects", "unresolved")


def chunk_coverage(timeline: str, chunks: List[str]) -> Dict[str, Any]:
    """分块的保真度：拼回后与原文差多少（**只出计数，不含任何正文**）。

    为什么需要它：``chunk_text`` 在 ``len(chunks) >= max_chunks`` 时直接 ``break``，
    之后的行**不进任何分块** —— 单日聊天量大时，晚间对话会整段从选材里消失，
    而日志里只看得到「选材完成: N 件」，看不出素材已经被截过。这条计数把那种
    **静默**损失变成一条可回读的告警（真机 2026-10-17 那份日记"内容有错"
    的排查结论之一就是这个）。

    实现说明：``chunk_text`` 按行切分、用 ``\\n`` 拼回，所以拼回结果要么与原文逐字节
    相同，要么是原文的一个**前缀**。用最长公共前缀定位丢失去向即可，不必做 diff。

    Returns:
        ``{"identical", "lost_lines", "lost_chars", "total_lines"}``——
        全部是计数，可直接进日志（不涉聊天原文，符合脱敏约定）。
    """
    text = str(timeline or "")
    joined = "\n".join(str(c) for c in (chunks or []))
    before = text.split("\n")
    after = joined.split("\n")
    same = 0
    while same < len(before) and same < len(after) and before[same] == after[same]:
        same += 1
    return {
        "identical": joined == text,
        "lost_lines": len(before) - same,
        "lost_chars": len(text) - len(joined),
        "total_lines": len(before),
    }


# 天气线索词（**按信号强度排序**：越靠前越能直接确定天气）
# 只做「素材里有没有提到天气」这一件事 —— 不猜、不推算、不做情感隐喻映射。
_WEATHER_STRONG = ("下大雨", "大雨", "暴雨", "下雨", "下雨了", "雨好大", "被淹", "积水", "鞋湿", "淋湿")
_WEATHER_WEAK = ("雨", "雪", "阴天", "出太阳", "太阳", "好晒", "刮风", "好冷", "好热", "闷")
_WEATHER_MAP = {
    "雨": "雨", "雪": "雪", "阴天": "阴", "出太阳": "晴", "太阳": "晴",
    "好晒": "晴", "刮风": "多云", "好冷": "多云", "好热": "晴", "闷": "多云",
}


def weather_hint(timeline: str, max_samples: int = 3) -> str:
    """从当天时间线里抽「有人真的提到过天气」的证据（v1.6.0）。

    为什么需要它：v1.0 起天气是**交给模型随文生成**的（替代早期的关键词计数猜测），
    好处是不再机械，代价是模型会凭氛围臆造 —— 真机 2026-10-18 素材里写着
    「下大雨☔️ / 学校被淹了 / 鞋湿了 / 好像可以划船了」，日记却写成「今天全天阴天」，
    与当天所有人共同经历的那场雨直接矛盾。

    这里只认**字面证据**，不做任何推断：命中强信号（下雨/淹/湿）就认定「有雨」，
    命中弱信号（太阳/雪/闷…）按词映射；都没命中返回空串，交由模型按氛围选。

    Returns:
        可直接插进写作 prompt 的一段话；空串表示「素材里没有天气线索」。
    """
    text = str(timeline or "")
    if not text.strip():
        return ""
    hits: List[str] = []
    for word in _WEATHER_STRONG:
        if word in text:
            hits.append(word)
    if hits:
        samples = "、".join(f"「{w}」" for w in hits[:max_samples])
        return f"当天明显有雨（素材里出现了 {samples}）。"
    weak_word = ""
    for word in _WEATHER_WEAK:
        if word in text:
            weak_word = word
            break
    if weak_word:
        return f"素材里提到了天气（「{weak_word}」），按这个线索来写。"
    return ""


def _evidence_coverage(term: str, evidence: str) -> tuple[int, float]:
    """术语在证据文本里的「最长命中片段长度」与「字符覆盖率」。

    为什么不用「整条术语必须出现」：``topics`` / ``unresolved`` 是模型**概括**出来的
    短语（如「全员复读猫猫梗」），逐字出现是不可能的，一刀切会把所有正常条目全杀掉
    —— 误杀比漏杀更容易摧毁这项校验的可信度。
    改用两条**可解释**的判据（满足其一即可）：
    * 至少有一段连续 **5 字以上**逐字对得上 —— 5 字以上的巧合基本不可能，
      长昵称 + 中文概括（如「秋枫叶语Acer周日上课吐槽与鞋湿」）就是靠这条留下的；
    * 或：至少有一段 **3 字以上**对得上，**且**覆盖过半。
    """
    t = str(term or "").strip()
    if not t or not evidence:
        return 0, 0.0
    longest = 0
    covered = [False] * len(t)
    for size in range(len(t), 2, -1):  # 只认 3 字及以上的片段，2 字太容易撞巧合
        for i in range(0, len(t) - size + 1):
            piece = t[i : i + size]
            if piece in evidence:
                if size > longest:
                    longest = size
                for k in range(i, i + size):
                    covered[k] = True
    ratio = sum(covered) / len(t)
    return longest, ratio


# META 里**实体类**的键：人名/项目名是专有名词，素材里怎么写就得怎么写，
# 一个字母的漂移（真机 2026-10-18 把 kugurija 写成 kugirija）就是幻觉，必须逐字命中。
_META_STRICT_KEYS = ("people", "projects")


def filter_meta_against_evidence(
    meta: Mapping[str, Any], evidence: str
) -> tuple[Dict[str, Any], List[str]]:
    """把 META 里**当天素材中找不到依据**的术语剔掉（v1.6.0）。

    为什么必须有这道确定性校验：META 不进正文、也不发布，但它会写进
    ``continuity.json``，并经由 ``get_day_digest`` 交给别的插件 —— 真机 2026-10-18
    那份 META 的 13 个术语里 **12 个是幻觉**（``FMInfinity共鸣电台``、``新歌命名``、
    ``看板娘归属权争议``、``项圈牵绳排队``…，素材里一个字都没有）。正文发完就过去了，
    META 的错却会**每天在错的基础上继续累积**。

    判据（两条都要满足，宁可漏也不可错）：
    * 至少有一段 **3 字以上**的连续片段在素材里逐字出现；
    * 术语被素材覆盖的字符 **过半**。

    Returns:
        ``(清洗后的 meta, 被丢弃的术语列表)``。丢弃列表交给调用方记日志 ——
        **抑制分支必须可见**，否则误杀会变成另一种静默。
    """
    text = str(evidence or "")
    cleaned: Dict[str, Any] = {}
    dropped: List[str] = []
    if not text.strip():
        return dict(meta or {}), []
    for key, values in (meta or {}).items():
        if not isinstance(values, list):
            continue
        strict = key in _META_STRICT_KEYS
        kept: List[str] = []
        for value in values:
            term = str(value or "").strip()
            if not term:
                continue
            if strict:
                # 实体名：整条必须逐字出现在素材里，差一个字母都不行
                if term in text:
                    kept.append(term)
                else:
                    dropped.append(f"{key}:{term}（实体名在素材里逐字未命中）")
                continue
            if len(term) < 3:
                # 术语本身不足 3 字：整条逐字命中才算有据。
                # 这里若跟着放宽，单字/双字会放进大量噪音（中文里两字撞车太容易）。
                if term in text:
                    kept.append(term)
                else:
                    dropped.append(f"{key}:{term}（短词在素材里逐字未命中）")
                continue
            longest, ratio = _evidence_coverage(term, text)
            if longest >= 5 or (longest >= 3 and ratio >= 0.5):
                kept.append(term)
            else:
                dropped.append(f"{key}:{term}（最长片段 {longest} 字 / 覆盖 {ratio:.0%}）")
        if kept:
            cleaned[key] = kept
    return cleaned, dropped


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


# ---------------------------------------------------------------- 篇幅三档

# 每次宣布「今天要写多少字」的见面礼：真人写日记不会每天等长。
# 三档 = (目标字数系数, 容差系数)：素材少就短写，素材多才摊得开。
_WORD_TIER_SCALES = {
    "short": (0.4, 0.5),
    "mid": (0.7, 0.7),
    "long": (1.0, 1.0),
}
_WORD_TIER_NAMES = {"short": "短", "mid": "中", "long": "长"}

# 短档再短也不能低于这个数（3 倍 :data:`MIN_DIARY_CHARS`）：低于它就逼近
# ``diary_output_problem`` 那条 20 字语义闸，等于天天把模型往「输出过短」上推 ——
# 生成侧的引导绝不能和发布闸打架。默认 350×0.4=140，远高于它。
_MIN_TIER_TARGET = MIN_DIARY_CHARS * 3


def choose_word_budget(
    word_target: int,
    word_tolerance: int,
    *,
    event_count: int = 0,
    material_mode: str = "events",
) -> Dict[str, Any]:
    """按素材丰富度选篇幅档位（**纯函数**，便于单测）。

    为什么不再"每天都写同样的长度"：固定 350±150 会逼着模型凑字，而凑出来的
    多半就是空转动作、复述自己、每段感受收尾这三样（v1.4.3 治过的三种病）。

    档位判据是**确定性**的 —— 同一天同样素材永远同一档，出问题能精确定位：

    ==============================  ======
    条件                            档位
    ==============================  ======
    ``material_mode == timeline_tail``（选材降级）  短
    入选事件 ≤ 1 件                 短
    入选事件 = 2 件                 中
    入选事件 ≥ 3 件                 长
    ==============================  ======

    Args:
        word_target: 配置的字数目标，作为**长档**基准。
        word_tolerance: 配置的容差，同样按比例缩放。
        event_count: 实际入选事件数。
        material_mode: 素材来源（``events`` / ``timeline_tail``）。

    Returns:
        ``{"tier", "target", "tolerance", "lo", "hi", "reason"}``，
        其中 ``tier`` 是「短/中/长」，``reason`` 供日志回读（不含聊天内容）。
    """
    base = max(1, int(word_target))
    base_tol = max(0, int(word_tolerance))
    try:
        count = max(0, int(event_count))
    except (TypeError, ValueError):
        count = 0
    mode = str(material_mode or "").strip()
    if mode == "timeline_tail":
        tier, reason = "short", "选材降级"
    elif count <= 1:
        tier, reason = "short", f"入选 {count} 件"
    elif count == 2:
        tier, reason = "mid", "入选 2 件"
    else:
        tier, reason = "long", f"入选 {count} 件"
    scale, tol_scale = _WORD_TIER_SCALES[tier]
    target = max(_MIN_TIER_TARGET, round(base * scale))
    tolerance = max(0, round(base_tol * tol_scale))
    # 下限也不许掉到语义闸以下（容差配得比目标还大时会走到这里）
    lo = max(MIN_DIARY_CHARS + 1, target - tolerance)
    hi = max(target + tolerance, lo + 1)
    return {
        "tier": _WORD_TIER_NAMES[tier],
        "target": target,
        "tolerance": tolerance,
        "lo": lo,
        "hi": hi,
        "reason": reason,
    }


# ---------------------------------------------------------------- 开场轮换

# 结构定式本身就是最大的 AI tell：每天「日期行 → 逐条转述 → 引用 → 感受收尾」，
# 哪怕每一句都写得像人，读者看三篇就认出来了。这里给 4 种开场骨架，
# 按**日期哈希**确定性轮换 —— 同一天一定同一种（可复现、可单测、出怪场能定位）。
#
# ⚠️ 文案里绝不用「」写词：引号在本插件的 prompt 里有专属含义（只能装别人的原话），
# hint 里出现「」会让模型以为那也是引用，等于给规则 3 制造歧义。
_OPENING_HINTS = (
    "从今天印象最深的那件事写起，别先交代时间地点。",
    "用今天听到的一句话开头，后面再说是什么事。",
    "先写此刻的状态（困不困、心里还挂着什么），再回头说今天的事。",
    '自由开场，但正文第一句不许出现"今天"这两个字。',
)


def opening_hint(date_str: str) -> str:
    """按日期确定性轮换开场骨架。

    同一个 ``date_str`` 永远返回同一条；不同日期大体均匀落在 4 种骨架上。
    日期缺失/异常时返回第 0 条（不抛异常 —— 开场提示不值当让成文失败）。
    """
    digest = hashlib.sha256(str(date_str or "").encode("utf-8")).hexdigest()
    return _OPENING_HINTS[int(digest[:8], 16) % len(_OPENING_HINTS)]


# ---------------------------------------------------------------- AI 腔检测

# AI 腔特征清单。**非阻断**：命中只写进存档字段 ai_tone_hits 并打一行 INFO，
# 用于观察「改了 prompt 之后，腔调有没有真的降下来」—— 这是本方案的回归指标。
#
# ⚠️ 只取**强特征**：宁可漏报也不误报。日记是给人看的，误报只会把作者的注意力
# 引到没问题的句子上。又因为要写进日志，一律只回标签、不回命中原文
# （聊天原话可能被引号捎带进正文；本插件约定日志不得出现聊天逐字原话）。
_AI_TONE_PATTERNS = (
    ("总结式收尾", re.compile(r"真是[^。！？!\n]{0,12}的一[天晚日]")),
    ("展望收尾", re.compile(r"明天也(?:要|得|会|记得|希望)")),
    ("治愈系词汇", re.compile(r"充实|治愈|小确幸|治愈感")),
    ("顿悟句式", re.compile(r"我(?:意识到|明白了|突然发现|懂得了|忽然觉得|终于明白)")),
    ("连续感叹号", re.compile(r"[!！]{2,}")),
)

# 排比：同一行里连续 3 个短分句**开头两个字完全一样**（「一遍…一遍…一遍…」）。
# 这是最典型的 AI 排比形态；正常口语不会连用三次同样的结构。
_CLAUSE_SPLIT_RE = re.compile(r"[，,、；;]")


def _parallelism_hit(content: str) -> bool:
    for line in str(content or "").split("\n"):
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(line)]
        clauses = [c for c in clauses if len(c) >= 2]
        for i in range(len(clauses) - 2):
            heads = {c[:2] for c in clauses[i : i + 3]}
            if len(heads) == 1:
                return True
    return False


def ai_tone_hits(content: str) -> List[str]:
    """检测日记正文里的 AI 腔特征。返回**命中的标签列表**（去重、稳定顺序）。

    无命中返回空列表。非阻断，调用方只做留痕与统计，不得据此判成品失败。
    """
    body = str(content or "")
    if not body.strip():
        return []
    hits: List[str] = []
    for label, pattern in _AI_TONE_PATTERNS:
        if pattern.search(body) and label not in hits:
            hits.append(label)
    if _parallelism_hit(body) and "排比句式" not in hits:
        hits.append("排比句式")
    return hits


def build_write_prompt(
    *,
    date_str: str,
    events_text: str,
    name: str,
    persona: str,
    style_extra: str = "",
    word_target: int = 350,
    word_tolerance: int = 150,
    max_events: int = 3,
    temporal_anchor: str = "",
    continuity_text: str = "",
    weather_hint_text: str = "",
) -> str:
    """阶段二：基于精选素材写日记（v1.5.0「去 AI 味」改写）。

    改写的出发点：**AI 味的来源不是单句措辞，而是四个结构性问题** ——
    规则过载（10 条编号规则让模型把算力花在"合规"上）、结构定式（日期→事件→
    引用→感受，尤其"每篇必须有引用"）、视角错位（被设定成"复盘聊天记录的
    书记员"而不是"过完了这一天的人"）、篇幅偏长（逼着模型凑字）。

    Args:
        word_target: 目标字数（正文）。**只是"最合适的长度"，不是硬指标**。
            调用方通常传 :func:`choose_word_budget` 算出来的当档目标。
        word_tolerance: 上下浮动。区间宽一点更贴合「想到哪写到哪」的基调 ——
            真机实测模型常自然写到目标之上（目标 250 时写出 341 字），
            区间太窄等于每次都在越界，逼着它回头删或硬凑。
        temporal_anchor: 非空表示这是**补写**过去的某一天。此时要显式声明
            素材里「今天/昨天/前天」的相对时间基准，否则模型会以真正的今天来解读。
        continuity_text: 跨天连续性线索（见 :func:`build_continuity_line`），
            只用于引出回顾与延续，不能当作「今天发生了什么」的依据。
        weather_hint_text: 当天天气的**字面证据**（见 :func:`weather_hint`）。
            非空时规则 1 会把天气钉死在这个线索上；空串表示素材里没提天气，
            仍按旧行为让模型选一个贴合氛围的。
    """
    name_line = f"我的名字是{name}。" if name else ""
    persona_line = persona or "是一个爱聊天的机器人。"
    tol = max(0, int(word_tolerance))
    target = max(1, int(word_target))
    lo, hi = max(1, target - tol), target + tol
    anchor_block = ""
    if temporal_anchor.strip():
        anchor_block = (
            f"\n⚠️ 这是**补写** {date_str} 的日记。素材与记忆里出现的"
            f"「今天」「昨天」「前天」一律以 {temporal_anchor.strip()} 为基准来理解，"
            "不是真正的今天。\n"
        )
    continuity_block = f"\n{continuity_text}\n" if continuity_text.strip() else ""

    # 规则区：原来 10 条编号规则吃到模型一半注意力，写出来谨慎、平、安全。
    # v1.5.0 压到 6 条 —— 但**只压表述、不删约束**：
    # 引号边界、日期行格式、事实纪律、META 块、不得硬凑，这些硬契约一条都没动；
    # v1.4.3 治过的三种病（空转动作 / 复述自己 / 每段感受收尾）并成规则 4 的一句话。
    # ⚠️ 编号**动态生成**：曾两次因为改规则数把 style_extra 的编号算错（撞号会让
    # 模型把两条规则读成一条），交给 enumerate 就再也不会错。
    # 天气：v1.6.0 起**优先采信素材里的字面证据**，不再交给模型凭氛围猜。
    # 真机 2026-10-18 素材写着「下大雨 / 学校被淹了 / 鞋湿了」，日记却写「全天阴天」，
    # 与当天所有人共同经历的那场雨直接对不上。有线索就钉死，没线索才允许按氛围选。
    if weather_hint_text.strip():
        weather_rule = (
            f"第一行固定为 {date_str}，X。\n"
            f"   ⚠️ 当天的天气**已经有据可查**：{weather_hint_text.strip()}\n"
            "   必须照这个写（线索说有雨就写 雨），不许凭感觉改成别的天气；"
            "正文里若提到天气，也必须与第一行一致。\n"
            "   只选一个字（晴/多云/阴/雨/雪），都不许多。"
        )
    else:
        weather_rule = (
            f"第一行固定为 {date_str}，X。其中 X 从 晴/多云/阴/雨 里选一个贴合今天聊天氛围的，"
            "只选一个字都不许多。"
        )
    rules = [
        weather_rule,
        f"只写上面你真想写的事，最多写 {max_events} 件，可以只写一两件。"
        "没劲的一天就写短点，三五句也行，绝不硬凑字数。",
        # 引用降级：素材给了原话就顺口引一句，没给就不引 —— "每篇必须抄一句别人原话"
        # 既不真人，又是 v1.2.7 / v1.4.5 那串引号归属病的上游诱因（为了完成指标硬凑引用）。
        # **边界约束一条不留全部保留**，只是不再下达"必须引用"的指标。
        "素材给了聊天原话就顺口引一句放进「」里，没给就不引 —— 不必为了引用而引用。\n"
        "   「」里面只能放**别人说过的原话、一字不改**；我自己的话一律写在引号**外面**。\n"
        "   错误写法：回了句「来首晚风岂不美哉 安排上了，好听」（把自己的回复也塞进了引号，"
        "读起来像整句都是我说的）\n"
        "   正确写法：有人提了《晚风》，回了句「来首晚风岂不美哉」，我说安排上了，好听。\n"
        "   **一对「」只装一个人的一句话**，绝不能把两个人的话焊在同一对引号里。\n"
        "   反例：「稿子改完了 眼睛有点酸 先躺一会儿 我也困得慌」—— 前半句是别人说的、后半句是"
        "我说的，这等于替别人发言；正确写法是别人说的进「」，我说的写引号外。\n"
        "   聊天记录里若是**「引用后回复」**（先引了别人的话、再紧跟我的回复），"
        "只引用**被引的那一句**，我的回复写在引号外面。\n"
        "   另外，凡是引用聊天原话都用「」，不要用 \"\" 或 '' 代替；能顺带点出是谁说的更好。",
        "像睡前随手写的：句子短，允许有点碎，允许口语和自嘲，想到哪写到哪，"
        "但别写成聊天记录复述。\n"
        "   三种啰嗦写法，写完通读一遍就删掉：只有动作没内容的句子；换个说法又说一遍；"
        "每段都拿我的感受收尾。",
        "**事实纪律**：主观感受和情绪可以自由写；但**没有依据就不得制造**人物、地点、对话、"
        "结果或新的事实。\n"
        "   拿不准的地方用\"好像\"\"应该是\"\"记不清了\"这类不确定语气带过 —— 文风自由不等于"
        "客观事实可以补写。\n"
        "   ⚠️ 「我问/我说」只能用在**素材真有的互动**上：素材里没写我问过，就不要写我问过"
        "（这正是上面说的\"只有动作没内容\"）。缺转场就直接写结果（「她还没给新歌取名。」），"
        "比编一个动作再让它落空强。\n"
        "   上面的跨天线索（若有）只能用来引出想法、疑问、期待，或带原日期的回顾；"
        "它**不能单独证明今天发生了什么**，今天的事必须有上面的素材作依据。\n"
        "   三条真机踩过的坑，一条都不许再犯：\n"
        "   ⚠️ **不许搬动时间**：素材里是下午发生的事就别写成「早上」，没写时段就别写时段。\n"
        "   ⚠️ **不许改写我自己的行为**：素材里我回了几次就是几次，"
        "不许写「我没再接茬」这种和素材相反的说法 —— 那等于凭空改掉事实。\n"
        "   ⚠️ **人名必须逐字照抄**素材里的写法：不许简写、音译、美化或拼错一个字母"
        "（真机把 kugurija 写成 kugirija，整段就成了在说一个不存在的人）。",
        "禁止出现：开头问候语；结尾总结或展望（如\"明天也要加油\"\"真是充实的一天\"）；"
        "\"我意识到/我明白了/我突然发现\"句式；连续感叹号；排比句。",
        f"篇幅：{lo}~{hi} 字，{target} 字上下最舒服。**字数是给你留的余地，不是任务** ——\n"
        "   宁可短一点也别硬凑；写完了觉得还差口气，就多补一句感受，别去堆形容词。"
        "除第一行外就是日记正文：不要标题、不要 markdown、不要\"日记\"二字开头、"
        "不要任何前后缀或解释。",
    ]
    if style_extra and style_extra.strip():
        rules.append(style_extra.strip())
    rules_text = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))
    # 视角：素材是"脑子里还留着的片段"，不是"整理好的聊天记录复盘"。
    # 被设定成书记员 → 输出必然是转述体；过完了一天的人才会先有印象、再有顺序。
    return f"""{name_line}
我{persona_line}
{anchor_block}
今天是 {date_str}。这一天过完了，睡前随手记一下。脑子里还留着这些片段：

{events_text}
{continuity_block}
现在以第一人称写一篇日记。
开场：{opening_hint(date_str)}
规则（重要，逐条遵守）：
{rules_text}

日记正文写完后，另起一段只输出下面这两行（用于归档，**不会被发表**）：
{META_MARKER}
{{"topics": ["话题"], "people": ["提到的人"], "projects": ["手头的事"], "unresolved": ["还没个结果的"]}}

META 的硬要求（它会跨天累积，写错会一直错下去）：
- 四个字段**只允许填当天素材里出现过的词**；想不到就留空数组 []，绝不许为了填满而编。
- people / projects 必须是素材里**逐字**出现的名字，一个字母都不许改。
- unresolved 只填**真的还没个结果**的事，已经收尾的不要写进来。

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


_WEEKDAY_CHARS = "一二三四五六日天"
_WEATHERS = ("晴", "多云", "阴", "雨")

# 完整的「年月日」才算日期行 —— 只有年月不行（「2026年过得真快」不匹配）
_FULL_DATE_RE = re.compile(r"^\s*[\[【（(「『]*\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
# 短式「9月26日」（年月日三缺一时用它，日/月必须和目标日期一致才认）
_SHORT_DATE_RE = re.compile(r"^\s*[\[【（(「『]*\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_WEEKDAY_RE = re.compile(r"^[\s（(【\[]*[星週周]\s*期?\s*[" + _WEEKDAY_CHARS + r"]")
# 天气只在「后面紧跟标点 / 空白 / 行尾」时才认，免得把「雨太大了」的正文切成天气
_WEATHER_RE = re.compile(
    r"^[\s，,、:：]*(" + "|".join(_WEATHERS) + r")(?=[\s，,、。.：:；;～~\-—）)】\]」』]|$)"
)
# 日期行收尾要清掉的残留标点
_DATE_TAIL_RE = re.compile(r"^[\s，,、。.：:；;～~\-—）)】\]」』]+")
_MD_DECOR_RE = re.compile(r"^[#>*\-\s]+")

DEFAULT_WEATHER = "多云"


def _target_ymd(date_str: str) -> tuple[int, int, int] | None:
    """从「2026年9月28日 星期一」或「2026-09-28」里取出 (年, 月, 日)。"""
    text = str(date_str or "")
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.match(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def parse_date_line(line: str, date_str: str) -> tuple[str, str] | None:
    """判断这一行是不是**目标日期**的日期行。

    是 → 返回 ``(天气, 行内剩余正文)``；不是（或写的是别的日期）→ ``None``。

    判据是**真的把日期解析出来比对**，而不是"含年份且含逗号"这种代理特征。
    真机踩坑（2026-09-28）：模型写成 ``2026年9月28日 星期一 雨``（逗号漏了），
    旧判据认不出这是日期行 → 又补一行 ``…，多云。`` →
    既出现两行日期，又把模型真实观察到的天气冲成了默认值。
    """
    ymd = _target_ymd(date_str)
    if ymd is None:
        return None
    rest = _MD_DECOR_RE.sub("", str(line or "").lstrip())
    m = _FULL_DATE_RE.match(rest)
    if m:
        if (int(m.group(1)), int(m.group(2)), int(m.group(3))) != ymd:
            return None  # 别的日期：不当日期行，交给补行逻辑
        rest = rest[m.end():]
    else:
        m = _SHORT_DATE_RE.match(rest)
        if not m:
            return None
        if (ymd[0], int(m.group(1)), int(m.group(2))) != ymd:
            return None
        rest = rest[m.end():]

    m = _WEEKDAY_RE.match(rest)
    if m:
        rest = rest[m.end():]
    weather = ""
    m = _WEATHER_RE.match(rest)
    if m:
        weather = m.group(1)
        rest = rest[m.end():]
    # 日期行该结束了：剩下来的若只是标点就丢掉，其余当正文保留（见调用方）
    return weather, _DATE_TAIL_RE.sub("", rest).strip()


def _looks_like_date_line(line: str) -> bool:
    """这一行**长得像日期行**吗（不比对具体日期）。

    用于 v1.4.3 修的缺口：``parse_date_line`` 只认**目标日期**，
    对"别的日期行"返回 None，于是 :func:`ensure_date_line` 的折叠循环一遇到
    "模型写了另一天"就 ``break`` —— 那行日期会原样留在正文里。

    真机实例（2026-09-28 那篇，补写/跨天触发时模型写了前一天的日期）：
    ```
    2026年9月28日 星期一，多云。
    2026年9月28日 星期一 雨          ← 两行日期，天气还互相冲突
    ```
    """
    rest = _MD_DECOR_RE.sub("", str(line or "").lstrip())
    return bool(_FULL_DATE_RE.match(rest) or _SHORT_DATE_RE.match(rest))


def date_line_overridden(raw: str, date_str: str) -> str:
    """返回模型写的**日期不对**的那一行日期行；全都是正确日期则返回空串。

    只用于留痕（v1.4.3「折中方案」）：成品里永远只有一行**正确日期**的日期行，
    但模型原来写的是哪一天不能就此消失 —— 落进存档字段 ``model_date_line`` 并打日志。

    扫描**所有**行而不是只看首行，因为真机有两种形态：
    - 首行就写错（``2026年9月27日 …`` 当首行）→ 就地改写成目标日期
    - 首行对、**第二行**写错（``…9月28日，多云。`` 后面又跟一行 ``…9月27日，晴。``）
      → 折叠掉第二行
    两种都要报出来；漏掉后一种就等于「折叠了但没人知道为什么」，而那正是 2026-09-28
    真机那篇的形态。

    判据是**日期真的不同**：只是格式差异（漏逗号、短式 ``9月28日``、缺句号）返回空串，
    那种情况模型写的就是当天，记下来只是噪音。短式缺年份、无法比对时也不报。
    """
    target = _target_ymd(date_str)
    if target is None:
        return ""  # 目标日期本身解析不出来 → 宁可不报也不误报
    for line in str(raw or "").split("\n"):
        stripped = line.strip()
        if not _looks_like_date_line(stripped):
            continue
        ymd = _target_ymd(stripped)
        if ymd is not None and ymd != target:
            return stripped
    return ""


def _strip_date_components(line: str) -> str:
    """剥掉一行里的「日期 + 星期 + 天气」，返回剩下的正文残留。

    用于首行是**别的日期**时：整行改写成目标日期，但那一行若还夹着正文
    （如 ``2026年9月27日，睡到中午才起`` 里的后半句），不能连正文一起丢掉。
    """
    rest = _MD_DECOR_RE.sub("", str(line or "").lstrip())
    m = _FULL_DATE_RE.match(rest) or _SHORT_DATE_RE.match(rest)
    if m:
        rest = rest[m.end():]
    m = _WEEKDAY_RE.match(rest)
    if m:
        rest = rest[m.end():]
    m = _WEATHER_RE.match(rest)
    if m:
        rest = rest[m.end():]
    return rest


def trailing_date_line(content: str, date_str: str) -> tuple[str, str] | None:
    """判断**最后一个非空行**是不是目标日期的日期行。

    真机踩坑（2026-10-08）：模型把日期行又写了一遍在**结尾**（当签名用），
    而且天气和首行不一致（首行「阴。」、尾行「晴」）——
    发布出去就是首尾两行日期 + 天气自相矛盾。旧版只看首行，结尾那行从来没人管。

    Returns:
        是 → ``(天气, 删掉该行之后的内容)``；不是 → ``None``。
        天气可能为空串（模型没写天气）。
    """
    lines = str(content or "").split("\n")
    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return None
    parsed = parse_date_line(lines[idx], date_str)
    if parsed is None:
        return None
    weather, leftover = parsed
    del lines[idx]
    if leftover:
        lines.insert(idx, leftover)
    return weather, "\n".join(lines)


def ensure_date_line(content: str, date_str: str) -> str:
    """保证第一行是日期行，并把它**规范成** ``<日期>，<天气>。``。

    五种输入都能正确处理：

    0. **结尾多写了一行日期**（v1.4.5 修，真机 2026-10-08）→ 删掉该行；
       若首行压根没有日期行，就**沿用结尾那行的天气**（别浪费模型已经写的天气）。

    1. 没写日期行 → 补一行（天气用默认值）。
    2. 写了但格式不标准（缺逗号 / 缺句号 / 用短式 ``9月28日``）→ 就地规范化，
       **保留模型自己写的天气**，不覆盖成默认值。
    3. 写了两行日期行（模型偶发）→ 折叠成一行；被折叠那行里若夹着正文，正文保留。
    4. **写了另一天的日期行**（v1.4.3 修）→ 同样折叠掉，不让它在正文里留一行
       "第二行日期"。这一行里的天气**不采用**（它属于别的日子，用了会写错天气）。

    「模型写错了日期」这件事不再靠人眼看日志发现：调用方用 :func:`date_line_overridden`
    取到被替换掉的那一行，落进存档字段 ``model_date_line``（见 plugin `_save_diary`）。

    ⚠️ 空内容会被补成 ``「<日期>，多云。\\n（今天没写出什么来。）」`` 这种**占位文本**，
    它是给「聊天里回看」兜底用的展示文案，**不是可发布的日记**。
    发布前必须先用 :func:`diary_output_problem` 判定，否则占位文本会被当成成品发出去。
    """
    if not content:
        return f"{date_str}，{DEFAULT_WEATHER}。\n（今天没写出什么来。）"
    # ① 先处理**结尾**的日期行：删掉它，并把它写的天气留作备用。
    #    模型有时把日期行当签名写在最后，留着就是"两行日期"（真机 2026-10-08）。
    tail_weather = ""
    _tail = trailing_date_line(content, date_str)
    if _tail is not None:
        tail_weather, content = _tail
    if not content.strip():
        # 整篇就只有那一行日期（罕见）→ 保留它的天气，按空内容兜底
        return f"{date_str}，{tail_weather or DEFAULT_WEATHER}。\n（今天没写出什么来。）"

    lines = content.split("\n")
    # 跳过可能存在的空行，日期行必须是**第一个非空行**
    idx = next((i for i, ln in enumerate(lines) if ln.strip()), 0)
    parsed = parse_date_line(lines[idx], date_str)
    if parsed is None and _looks_like_date_line(lines[idx]):
        # 首行是**别的日期**的日期行（v1.4.3）：直接改写成目标日期。
        # 天气不采用 —— 那是别的日子的天气，用了会把今天的天气写错。
        # 这一行的正文残留（如 ``2026年9月28日，出来看流星雨`` 里的后半句）保留。
        leftover_head = _DATE_TAIL_RE.sub("", _strip_date_components(lines[idx])).strip()
        lines[idx] = f"{date_str}，{DEFAULT_WEATHER}。"
        if leftover_head:
            lines.insert(idx + 1, leftover_head)
    elif parsed is None:
        # 没有可识别的日期行 → 补一行。若有前导空行，插在**第一个非空行之前**，
        # 而不是直接 return（曾经写成 `... if idx == 0 else content`，
        # 结果「空行开头且没写日期行」时整篇缺日期头）。
        # 天气优先用**结尾那行日期行**留下的（模型写了就别浪费），没有才用默认值。
        lines.insert(idx, f"{date_str}，{tail_weather or DEFAULT_WEATHER}。")
        return "\n".join(lines)
    else:
        weather, leftover = parsed
        lines[idx] = f"{date_str}，{weather or DEFAULT_WEATHER}。"
        if leftover:
            # 模型把正文第一句挤进了日期行 → 拆出来，别丢内容
            lines.insert(idx + 1, leftover)
    # 折叠紧随其后的日期行 —— **不论它写的是哪一天**。
    # 只认目标日期的写法会让"模型写了昨天"这种形态漏网（真机踩过）。
    while len(lines) > idx + 1 and _looks_like_date_line(lines[idx + 1]):
        nxt = parse_date_line(lines[idx + 1], date_str)
        # 同一天：沿用它的正文残留；另一天：整行丢弃（天气也不采用，那是别的日子的）
        tail = nxt[1] if nxt is not None else ""
        del lines[idx + 1]
        if tail:
            lines.insert(idx + 1, tail)
    return "\n".join(lines)


def date_display(date_str: str) -> str:
    """'2026-09-26' -> '2026年9月26日 星期六'。"""
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{dt.year}年{dt.month}月{dt.day}日 {weekdays[dt.weekday()]}"
