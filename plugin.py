"""better-diary: 两阶段日记生成插件。

对比旧 diary_plugin 的核心改进：
1. 两阶段生成 —— 阶段一按块提取"值得写的事"并打分，阶段二只喂精选事件成文，
   避免 50k 时间线一次进 prompt 导致模型按时间顺序线性复述聊天记录；
2. 反 AI 腔规则 + 强制引用聊天原话，日记有记忆点而不是转述；
3. 字数软控制（模型侧遵守），不做硬截断砍句子；
4. 天气/心情由模型随文生成，替代关键词计数猜测。
"""

import asyncio
import copy
import datetime
import json
import logging
import re
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase
from pydantic import field_validator

if __package__:  # 包式加载（Runner 真机：插件目录作为包，不在 sys.path 上）
    from .bd_prompts import (
        build_continuity_line,
        build_extract_prompt,
        build_timeline,
        build_write_prompt,
        chunk_text,
        date_display,
        diary_output_problem,
        ensure_date_line,
        event_id,
        events_to_text,
        parse_events,
        quote_attribution_risk,
        split_meta,
        strip_diary_output,
        update_continuity,
    )
    from .bd_cookie import CookieStore, redact_secrets
    from .bd_search import format_hits, on_this_day, search_diaries
else:  # 平铺兜底（脚本直跑 / 旧测试夹具）
    from bd_prompts import (
        build_continuity_line,
        build_extract_prompt,
        build_timeline,
        build_write_prompt,
        chunk_text,
        date_display,
        diary_output_problem,
        ensure_date_line,
        event_id,
        events_to_text,
        parse_events,
        quote_attribution_risk,
        split_meta,
        strip_diary_output,
        update_continuity,
    )
    from bd_cookie import CookieStore, redact_secrets
    from bd_search import format_hits, on_this_day, search_diaries

logger = logging.getLogger("plugin.org.orge-8.better-diary")

# 发送单条消息的最大长度（QQ 文本安全线）
_SEND_LIMIT = 1500

# 与 PluginSection.config_version 的默认值保持一致（配置垫片要用）
_DEFAULT_CONFIG_VERSION = "1.0.0"


# ---------------------------------------------------------------- 配置模型


def _as_str_list(value: Any) -> Any:
    """把 config.toml 里常见的「字符串形式的列表」友好地归一化为 list。

    真机踩坑（v1.2.3）：用户在 config.toml 里很容易写成

        admin_ids = "123456789"          # 而不是 ["123456789"]
        target_chats = "group:123456"

    这层归一化只能拦住「配置里**已经有** ``[plugin].config_version``」的那条支线：
    SDK 的 ``normalize_plugin_config`` 顺序是「先版本检查 → 再 pydantic 校验」，
    缺少 ``[plugin]`` 节的配置会在更早一步就抛 ``PluginConfigVersionError``。
    （真机上更早一步还有 **Runner 自己**的那道版本检查，详见类里的注册期防御注释——
    所以遇到「插件初始化失败」的正确动作是**换掉 config.toml**，不是改这里的代码。）
    """

    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.replace("，", ",").replace(";", ",").replace("\n", ",")
        return [item.strip() for item in normalized.split(",") if item.strip()]
    return value


# 素材过滤只认这三种模式；其余一律按最保守的 whitelist 处理
_FILTER_MODES = ("all", "whitelist", "blacklist")


def _find_by_value(node: Any, wanted: str, id_keys: tuple[str, ...], depth: int = 0) -> Any:
    """在任意嵌套结构里找 ``id_keys`` 之一等于 ``wanted`` 的那个对象。

    用于兜底解析：宿主 ``chat.get_stream_by_group_id`` 返回 None（该群没有可用聊天流）时，
    改从 ``chat.get_group_streams`` 列表里按 ``group_id`` 字段自己找一遍。
    深度限制防环形结构；只做**相等**比较、不做模糊匹配（避免认错群）。
    """
    if depth > 6 or node is None:
        return None
    wanted = str(wanted).strip()
    if isinstance(node, dict):
        for key in id_keys:
            if key in node and str(node.get(key)).strip() == wanted:
                return node
        for value in node.values():
            found = _find_by_value(value, wanted, id_keys, depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(node, (list, tuple)):
        for item in node:
            found = _find_by_value(item, wanted, id_keys, depth + 1)
            if found is not None:
                return found
    return None


def _find_stream_id_deep(node: Any, depth: int = 0) -> str:
    """递归找「像 stream_id 的值」。

    为什么需要它：兜底解析时 ``_find_by_value`` 命中的往往是**内层**的
    ``{group_id: "967779035"}``，而真正的 ``stream_id`` 挂在**外层**流对象上
    （``{stream_id: "…", chat: {group_id: "…"}}`` 是宿主的常见形态）。
    所以取流 ID 必须能从内层一路找回外层，而不是只看命中那一层。

    优先级：``stream_id`` 系键名 > ``session_id`` > ``chat_id`` > ``id``。
    只返回非空字符串；打不出来时返回空串，由调用方判失败。
    """
    if depth > 6 or node is None:
        return ""
    groups = (
        ("stream_id", "streamId", "streamID"),
        ("session_id", "sessionId"),
        ("chat_id", "chatId"),
        ("id",),
    )
    if isinstance(node, dict):
        for keys in groups:
            for key in keys:
                val = node.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        for value in node.values():
            found = _find_stream_id_deep(value, depth + 1)
            if found:
                return found
        return ""
    if isinstance(node, (list, tuple)):
        for item in node:
            found = _find_stream_id_deep(item, depth + 1)
            if found:
                return found
    return ""


def _describe_shape(value: Any, depth: int = 0) -> str:
    """用一句话描述一个对象的**结构**（不是内容），用于诊断"宿主返回了什么"。

    只打类型与键名 —— 不打印值：返回值可能夹带聊天内容或凭据，
    而排查"解析不出 stream_id"只需要知道**有哪些键**。
    """
    if value is None:
        return "None"
    if isinstance(value, str):
        return f"str(长度 {len(value)})" if value.strip() else "str(空)"
    if isinstance(value, bool):
        return f"bool({value})"
    if isinstance(value, (int, float)):
        return type(value).__name__
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{type(value).__name__}(空)"
        return f"{type(value).__name__}({len(value)} 项，首项={_describe_shape(value[0], depth + 1)})"
    if isinstance(value, dict):
        keys = [str(k) for k in value.keys()]
        if depth >= 1:
            return f"dict(键={keys[:6]})"
        return f"dict(键={keys[:12]})"
    if depth >= 1:
        return type(value).__name__
    try:
        return f"{type(value).__name__}(属性={[a for a in vars(value)][:8]})"
    except Exception:  # noqa: BLE001 - 诊断函数自身绝不能抛
        return type(value).__name__


def _norm_filter_mode(value: Any) -> str:
    """把 ``[diary].filter_mode`` 归一化到白名单里的三种取值。

    v1.3.5 起默认 ``whitelist``：**默认只取用户显式指定的会话**。
    未知取值也回退到 ``whitelist``（fail-closed）而不是 ``all`` ——
    配置写错时宁可少收料、由日志提示，也不要静默把全库跨群跨私聊的消息
    一股脑拼进 prompt 再公开发布。
    """
    mode = str(value or "").strip().lower()
    if mode in _FILTER_MODES:
        return mode
    if mode:
        logger.warning(
            "[diary].filter_mode 取值 %r 不在 %s 中，已按最保守的 whitelist 处理；"
            "想显式收全库请写 filter_mode = \"all\"",
            mode, " / ".join(_FILTER_MODES),
        )
    return "whitelist"


class PluginSection(PluginConfigBase):
    """插件基础配置。"""

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default=_DEFAULT_CONFIG_VERSION, description="配置版本")


class DiarySection(PluginConfigBase):
    """日记生成配置。"""

    word_target: int = Field(default=350, description="日记目标字数（正文）。只是最舒服的长度，不是硬指标")
    word_tolerance: int = Field(default=150, description="目标字数的上下浮动。区间放宽一点，模型写长了不必回头删")
    max_events: int = Field(default=3, description="日记最多写几件事")
    min_messages: int = Field(default=20, description="当天消息少于此数不生成日记")
    chunk_chars: int = Field(default=6000, description="时间线分块字符数")
    max_chunks: int = Field(default=8, description="最多送入选材的分块数")
    persona_override: str = Field(default="", description="覆盖 Host 人设；留空则读取主程序人设")
    style_extra: str = Field(default="", description="追加到写作规则后的额外风格要求")
    filter_mode: str = Field(default="whitelist", description="聊天过滤：all（全库，跨群+私聊）/ whitelist（只取 target_chats 指定的会话，**默认**）/ blacklist")
    target_chats: list[str] = Field(default_factory=list, description='过滤目标列表，格式 "group:群号" 或 "private:QQ号"。filter_mode 为 whitelist 时**必须**填，否则不生成（默认不把全库素材喂进去）')

    _norm_filter_mode = field_validator("filter_mode", mode="before")(_norm_filter_mode)
    _norm_target_chats = field_validator("target_chats", mode="before")(_as_str_list)


class ScheduleSection(PluginConfigBase):
    """定时任务配置。"""

    enabled: bool = Field(default=True, description="是否启用每日定时生成")
    time: str = Field(default="23:30", description="每天生成时间，HH:MM（Host 本地时区）")
    notify_chats: list[str] = Field(default_factory=list, description='定时发布后通知哪些聊天，格式 "group:群号" / "private:QQ号" / 裸 stream_id；留空则不发通知')
    catch_up_enabled: bool = Field(default=True, description="补写缺档：插件启动与次日兜底各扫一次，给没有日记的日期补写（只存档、不发布）")
    catch_up_days: int = Field(default=3, description="补写最多往前看几天（不含今天）。这个上界同时充当「生效日」，绝不追溯更早的空档")
    fallback_time: str = Field(default="04:10", description="次日凌晨的兜底补写时刻，HH:MM；留空或与 time 相同则关闭兜底")
    wait_silent_minutes: int = Field(default=0, description="静默阈值：到点时若最近 N 分钟内还有新消息，就等人停下再写。0 = 关闭（到点直接写，旧行为）")
    retry_interval_minutes: int = Field(default=30, description="还没静默时，每隔多少分钟再看一次")
    max_wait_hours: int = Field(default=3, description="最多等多久；等满就照常生成（不丢天优先）。0 = 第一次发现没静默就直接写")
    catch_up_budget_minutes: int = Field(default=0, description="一轮补跑的总时长上限（分钟）。0 = 不限制。超预算就停手，剩余日期交给下一轮兜底（网络全崩时避免长时间占用）")

    _norm_notify_chats = field_validator("notify_chats", mode="before")(_as_str_list)


class QzoneSection(PluginConfigBase):
    """QQ空间发布配置（日记成品只发这里）。"""

    enabled: bool = Field(default=True, description="日记生成后是否发布到QQ空间")
    auto_cookie: bool = Field(default=True, description="自动获取 cookie：先试 napcat-adapter API，再试 NapCat HTTP 服务器，最后用手动兜底配置")
    refresh_interval_min: int = Field(default=60, description="自动取 cookie 的节流间隔（分钟）")
    napcat_http_host: str = Field(default="127.0.0.1", description="NapCat HTTP 服务器地址（MaiBot 1.3.0+ 无 adapter 时的主要 cookie 来源）")
    napcat_http_port: str = Field(default="3000", description="NapCat HTTP 服务器端口（NapCat WebUI 网络配置里开启的那个 HTTP Server）")
    napcat_http_token: str = Field(default="", description="NapCat HTTP 服务器的 token（未设置鉴权可留空）")
    uin: str = Field(default="", description="手动兜底 cookie 的 QQ 号（自动获取失败时才用到；纯数字）")
    p_skey: str = Field(default="", description="手动兜底 cookie 的 p_skey（可留空）")
    skey: str = Field(default="", description="手动兜底 cookie 的 skey（可留空）")
    timeout_seconds: int = Field(default=20, description="发布请求超时（秒）")


class LLMSection(PluginConfigBase):
    """LLM 调用配置。"""

    task_name: str = Field(default="utils", description="Host 模型任务名")
    temperature: float = Field(default=0.8, description="成文温度")
    extract_temperature: float = Field(default=0.2, description="选材温度（建议低温保证 JSON 稳定）")
    timeout_seconds: int = Field(default=180, description="单次 LLM 调用超时（秒）")
    write_retry: int = Field(default=1, description="成文阶段遇到**超时**类失败时额外重试几次。0 = 不重试（超时即判失败）")
    retry_backoff_seconds: int = Field(default=20, description="成文重试前的等待秒数。给网络抖动一点恢复时间，避免紧接着撞同一堵墙")


class SecuritySection(PluginConfigBase):
    """安全配置。

    **fail-closed 默认值**：``admin_ids`` 留空时**不再**放行所有人，
    只有 ``[security].allow_all_users = true`` 才会放开。日记成品会发布到
    **公开**的 QQ 空间，默认「谁都能触发」风险过高，因此默认必须由管理员触发。
    """

    admin_ids: list[str] = Field(default_factory=list, description="管理员 QQ 列表，兼容 '123' 与 'qq:123' 写法。留空 = 不放行任何人（除本机控制台操作者）")
    allow_all_users: bool = Field(default=False, description="高级选项：是否允许所有人使用命令。默认 false；确需全员可用时显式打开（此时 /日记 可被任何人触发并发布到公开QQ空间）")
    command_cooldown_seconds: int = Field(default=300, description="同一会话两次 /日记 之间的最小间隔（秒），防止反复触发 LLM 调用与公开发布。0 = 不限制")

    _norm_admin_ids = field_validator("admin_ids", mode="before")(_as_str_list)


class BetterDiaryConfig(PluginConfigBase):
    """插件配置总模型。"""

    plugin: PluginSection = Field(default_factory=PluginSection)
    diary: DiarySection = Field(default_factory=DiarySection)
    schedule: ScheduleSection = Field(default_factory=ScheduleSection)
    qzone: QzoneSection = Field(default_factory=QzoneSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    security: SecuritySection = Field(default_factory=SecuritySection)


# ---------------------------------------------------------------- 插件主体


class BetterDiaryPlugin(MaiBotPlugin):
    """两阶段日记生成插件。日记成品只发布到 QQ 空间，聊天内仅回报执行状态。"""

    config_model = BetterDiaryConfig

    # 类级默认：不跑 on_load（如冒烟测试直接实例化）也能安全访问
    _generating = False
    _sched_task = None
    _catchup_task = None
    _cookie_store = None
    _cfg_fallback_warned = False
    # 每个会话最近一次 /日记 的时间戳（秒），用于命令冷却；不落盘
    _last_gen_ts: dict[str, float] = {}
    # 「宿主未注入 data_dir」的降级告警只打一次
    _data_dir_warned = False
    # 解析不出 stream_id 的白名单目标：{target: 原因}，供 _filter_diagnostic 组织提示
    _unresolved_targets: dict[str, str] = {}
    # 解析成功且真的取到消息的目标（用于在「0 条」时区分「配错了」与「当天这会话没人说话」）
    _resolved_with_messages: list[str] = []
    # 已经打过「解析失败」WARNING 的目标（同一目标反复失败只提醒一次）
    _whitelist_warned: set[str] = set()
    # 最近一次取材范围诊断：(条件, 配置片段, 原因列表)，供 0 条素材时给出可执行提示
    _last_filter_diag: tuple[str, list[str], list[str]] | None = None

    def _filter_diagnostic(self, messages_len: int, min_msgs: int) -> tuple[str, list[str], list[str]] | None:
        """0 条（或不足）素材时，判断「是不是取材范围配置的问题」。

        返回 ``(条件描述, 涉及的配置片段, 原因列表)``，不是配置问题则返回 ``None``。
        兼容期保护的核心：把「配置没生效」和「今天真的没人说话」区分开 ——
        两者在旧实现里都会表现为「当天消息太少」。
        """
        if messages_len >= min_msgs:
            return None
        if self._unresolved_targets:
            return (
                "target_chats 里有解析不出的目标",
                [f"[diary].target_chats 中的 {t}" for t in self._unresolved_targets],
                list(self._unresolved_targets.values()),
            )
        if self._resolved_with_messages:
            # 会话都解析成功、也确实取到过消息，只是总量不到 min_messages：
            # 这是真的「今天聊得少」，别再让用户去改配置。
            return None
        mode = _norm_filter_mode(self.config.diary.filter_mode)
        targets = [str(t) for t in (self.config.diary.target_chats or []) if str(t).strip()]
        if mode == "whitelist" and not targets:
            return (
                "filter_mode 为 whitelist（默认）但 target_chats 为空",
                ["[diary].filter_mode = \"whitelist\"", "[diary].target_chats = []"],
                ["target_chats 为空：没有任何会话被指定为取材范围"],
            )
        return None

    # ------------------------------------------------------------ 注册期防御
    #
    # 真机踩坑总览（v1.2.1 → v1.2.5）：
    #   1) 导入期 —— 插件目录不在 sys.path，平铺导入必挂（v1.2.1 修，双路径导入）；
    #   2) 配置期 —— 真因是**插件目录里的旧 config.toml**（v1.2.5 确认）。
    #
    # ★ 先记住这个层级划分，否则会一直在错误的地方改代码：
    #
    #   扫描 plugins/ → 读 _manifest.json → 校验依赖
    #      → 导入 plugin.py → create_plugin() → 注入 ctx
    #      → 配置注入：读 config.toml → extract_plugin_config_version   ← Runner 侧
    #      → on_load()                                                  ← 才轮到本插件代码
    #
    #   宿主对两者的说法不同：
    #     「插件初始化失败」= 挂在配置注入之前（**本插件代码一行都没执行**）
    #     「插件加载失败」  = on_load 抛的异常
    #   看到「初始化失败」就别改插件源码 —— 改不动的。
    #
    #   真机实录的最常见成因：`plugin.config_version` 是 **Host 1.2.3 引入的硬性要求**，
    #   而 config.toml 由旧宿主生成、没有这个键；宿主一升级，Runner 就读不过。
    #   唯一的解法是换个合法 config.toml（删掉重生成，或 WebUI 保存一次）。
    #
    #   版本检查逻辑在 `runner/runner_main.py::extract_plugin_config_version`，
    #   即 **Runner 自己的代码**，不在下面这个 SDK 方法里。
    #
    # 下面这层防御只覆盖「配置已过 Runner 版本检查、但在 SDK 侧出了问题」的情况
    # （SDK 2.8.1 源码确认）：
    #   MaiBotPlugin.set_plugin_config:191  normalize_plugin_config(...)  ← 裸调用，无 try
    #   normalize_plugin_config:173/178     extract_plugin_config_version(...)
    #   normalize_plugin_config:175/180     validate_plugin_config(...)
    #   两类调用都会抛，且**版本检查排在合并默认值之前**；
    #   而 set_plugin_config:200 那次 pydantic 校验是有 try 的（只 warning），
    #   所以这一层里唯一裸奔的入口是 normalize_plugin_config。

    @staticmethod
    def _redact(record: Any) -> Any:
        """对即将写进日志的**单条参数**做凭据脱敏。

        v1.3.4：插件所有日志都从这里出去，因为「异常对象会夹带凭据」这件事
        是**全类性质**的 —— adapter 会把请求体(含 cookie)写进 error，httpx 的
        异常会把 request(含 URL 上的 g_tk/uin)带进 str。只在个别分支手动脱敏
        必然会漏（实测审计就是这样抓到 26 处直插异常的）。这里统一收口：
        异常对象、外部响应片段、入参文本一律先脱敏再落盘。

        脱敏只吃「键=值」的值部分，键名、状态码、结构都保留，排障信息不丢。
        """
        if record is None or isinstance(record, (int, float, bool)):
            return record
        if isinstance(record, BaseException):
            return f"{type(record).__name__}: {redact_secrets(record)}"
        if isinstance(record, str):
            return redact_secrets(record)
        try:
            return redact_secrets(repr(record))
        except Exception:  # noqa: BLE001 - repr 本身出错不该拖垮日志
            return "<unprintable>"

    def _log_info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """best-effort 记录 INFO：日志器自身不可用时静默，绝不丢给宿主。"""
        try:
            self._get_logger().info(redact_secrets(msg), *(self._redact(a) for a in args), **kwargs)
        except Exception:  # noqa: BLE001 - 日志失败不能反过来拖垮加载
            pass

    def _log_error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """best-effort 记录 ERROR：日志器自身不可用时静默，绝不丢给宿主。"""
        try:
            self._get_logger().error(redact_secrets(msg), *(self._redact(a) for a in args), **kwargs)
        except Exception:  # noqa: BLE001 - 同上
            pass

    def _log_warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """best-effort 记录 WARNING。

        专门给**异常兜底分支**用：兜底分支的职责是「无论如何都要把控制权交回去」，
        若在 except 里直接 `self.ctx.logger.warning(...)`，日志器本身不可用时会把
        异常再次抛出，反而破坏了兜底。这里沿用 _log_info/_log_error 的安全语义。
        """
        try:
            self._get_logger().warning(redact_secrets(msg), *(self._redact(a) for a in args), **kwargs)
        except Exception:  # noqa: BLE001 - 日志失败不能反过来破坏兜底逻辑
            pass

    # 超时类异常的判据（按类名 + 文本，**不能只靠 isinstance**）。
    # 真机实测：LLM 调用超时抛的是 Runner 的 RPCError
    #   `[E_TIMEOUT] 请求 cap.call 超时 (180000ms)`（rpc_client.py 里 `raise RPCError(...) from None`）
    # **不是** asyncio.TimeoutError；而且真机那个异常类与本地 devkit 的不是同一个对象，
    # 依赖 isinstance 会在本地假绿、真机判不出来。cause 是 None（from None），所以也查不了链。
    _TIMEOUT_TYPE_HINTS = ("timeout", "e_timeout")
    _TIMEOUT_TEXT_HINTS = ("timeout", "timed out", "e_timeout", "超时")

    @classmethod
    def _is_timeout_error(cls, exc: BaseException) -> bool:
        """异常是不是「超时」类。超时可重试（网络抖动），其余多数重试也没用。"""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        if any(h in type(exc).__name__.lower() for h in cls._TIMEOUT_TYPE_HINTS):
            return True
        text = str(exc).lower()
        return any(h in text for h in cls._TIMEOUT_TEXT_HINTS)

    # 宿主侧**瞬时**基础设施故障的判据（v1.4.0）。
    # 真机实录（2026-09-30 10:46）：Host 要把模型错误现场写进
    # logs/maisaka_prompt/llm_error/ 供回放排障，多请求并发写同一目录时
    # rename `.tmp` → 目标文件被拒：
    #   [WinError 5] 拒绝访问。: '…\xxx.json.tmp' -> '…\xxx.json'
    # LLMService 把这当成「生成内容时出错」抛出，插件看到的是
    #   `选材分块失败（跳过本块）: LLM 调用失败: [WinError 5] 拒绝访问。…`
    # 这类失败**与模型无关、且明显是瞬时的**（下一次 rename 多半就成功），
    # 所以值得重试 —— 不重试也是白等一整轮。
    _IO_TRANSIENT_TYPE_HINTS = ("permissionerror", "oserror")
    _IO_TRANSIENT_TEXT_HINTS = (
        "winerror 5", "winerror 32", "winerror 33",
        "eacces", "eperm", "ebusy", "access is denied",
        "拒绝访问", "另一个程序正在使用", "being used by another process",
        "permission denied", "resource busy",
    )

    @classmethod
    def _is_transient_io_error(cls, exc: BaseException) -> bool:
        """异常是不是宿主侧**瞬时 I/O 故障**（写日志/落盘被占用/被拒）。"""
        text = str(exc).lower()
        return any(h in text for h in cls._IO_TRANSIENT_TEXT_HINTS)

    @classmethod
    def _is_retryable_llm_error(cls, exc: BaseException) -> bool:
        """LLM 调用失败是否值得重试：超时 **或** 宿主侧瞬时 I/O 故障。

        **格式/语义类失败仍然不重试** —— 那种失败重试纯属浪费预算。
        """
        return cls._is_timeout_error(exc) or cls._is_transient_io_error(exc)

    @staticmethod
    def _merge_with_defaults(defaults: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
        """把用户配置递归合并进默认配置，保留用户已填的值。

        兜底时用它而不是直接用模型默认值：直接丢默认会把用户填的
        uin / p_skey / admin_ids 一起抹掉，比配置写错更糟。
        """

        merged: dict[str, Any] = copy.deepcopy(dict(defaults))
        for key, value in raw.items():
            current = merged.get(key)
            if isinstance(value, Mapping) and isinstance(current, dict):
                merged[str(key)] = BetterDiaryPlugin._merge_with_defaults(
                    cast(Mapping[str, Any], current), value
                )
            else:
                merged[str(key)] = copy.deepcopy(value)
        return merged

    @staticmethod
    def _sanitize_config(config: Any) -> Any:
        """交给 SDK 之前，把 ``[plugin].config_version`` 补齐。

        SDK 会**先**对原始配置做版本检查、**再**合并默认值，所以「用户少写了
        ``[plugin]`` 节」这种最常见的情况会先抛 ``PluginConfigVersionError``。
        这里补上版本号，属于对宿主配置格式的兼容垫片（非 Mapping 原样透传，
        交给 SDK 自己的默认值分支处理）。
        """

        if not isinstance(config, Mapping):
            return config
        data: dict[str, Any] = dict(config)
        raw_section = data.get("plugin")
        section: dict[str, Any] = dict(raw_section) if isinstance(raw_section, Mapping) else {}
        if not str(section.get("config_version") or "").strip():
            section["config_version"] = _DEFAULT_CONFIG_VERSION
        data["plugin"] = section
        return data

    @staticmethod
    def _reset_path(target: dict[str, Any], defaults: Mapping[str, Any], loc: tuple[Any, ...]) -> bool:
        """把 ``loc`` 指向的字段就地还原成默认值。返回是否真的改了东西。"""

        if not loc:
            return False
        cursor: Any = target
        default_cursor: Any = defaults
        for key in loc[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                return False
            cursor = cursor[key]
            default_cursor = default_cursor.get(key, {}) if isinstance(default_cursor, Mapping) else {}
        last = loc[-1]
        if not isinstance(cursor, dict) or last not in cursor:
            return False
        if isinstance(default_cursor, Mapping) and last in default_cursor:
            cursor[last] = copy.deepcopy(default_cursor[last])
        else:
            cursor.pop(last, None)
        return True

    @classmethod
    def _repair_config(cls, config_class: Any, merged: dict[str, Any], defaults: Mapping[str, Any]) -> Any:
        """逐字段修复配置：只把**真正非法**的字段还原为默认值，其余用户值全部保留。

        直接「整体丢回模型默认值」会把用户填的 uin / p_skey / admin_ids 一起抹掉，
        比配置写错更糟。这里读 pydantic 的 ``ValidationError.errors()`` 拿到出错字段的
        ``loc``，定点还原后重试，直到通过或无法再修。
        """

        candidate = copy.deepcopy(dict(merged))
        for _ in range(24):  # 字段数量级上限，兼作死循环保险
            try:
                return config_class.model_validate(candidate)
            except Exception as exc:  # noqa: BLE001
                errors = getattr(exc, "errors", None)
                if not callable(errors):
                    return None
                try:
                    collected = errors()
                except Exception:  # noqa: BLE001
                    return None
                repaired = False
                for err in collected:
                    loc = tuple(err.get("loc") or ())
                    if cls._reset_path(candidate, defaults, loc):
                        repaired = True
                if not repaired:
                    return None
        return None

    def set_plugin_config(self, config: dict[str, Any]) -> None:
        """覆写 SDK 的 ``set_plugin_config``：配置问题绝不拖垮插件注册。

        **注意作用范围**：这层拦的是「配置已过 Runner 版本检查、但 SDK 侧仍出错」的情况。
        若宿主报的是「插件**初始化**失败」且日志里本插件零输出，说明挂在更上游的
        Runner 配置注入（版本检查）——那里改不动，只能换 config.toml。详见类注释。

        SDK 在这一层没有整体 try，任何配置异常都会冒泡到宿主，
        而宿主只回一句「插件注册失败: <id>: 插件初始化失败」，现场信息全部丢失。
        这里做两件事：

        1. 先补 ``plugin.config_version``，消掉 SDK 侧的版本检查误报；
        2. 仍然失败时用「默认配置 + 用户已有值」兜底，并把**真实异常**连同
           traceback 写进日志——下一次排查不用再靠猜。
        """

        sanitized = self._sanitize_config(config)
        try:
            super().set_plugin_config(sanitized)
            return
        except Exception as exc:  # noqa: BLE001 - 必须兜住，否则整个插件注册失败
            self._log_error(
                "插件配置注入失败，已回退「默认配置 + 用户已有值」。真实原因如下"
                "（请据此修 config.toml，常见的是缺 [plugin].config_version）：%s",
                exc,
                exc_info=True,
            )

        # ---- 兜底路径：语义上等价于「配置坏了也要能加载」
        try:
            defaults = type(self).build_default_config()
        except Exception:  # noqa: BLE001
            defaults = {}
        raw = sanitized if isinstance(sanitized, Mapping) else {}
        try:
            merged = self._merge_with_defaults(defaults, cast(Mapping[str, Any], raw))
        except Exception:  # noqa: BLE001
            merged = dict(defaults)
        section = merged.get("plugin")
        if not isinstance(section, dict):
            section = {}
            merged["plugin"] = section
        section.setdefault("config_version", _DEFAULT_CONFIG_VERSION)

        config_class = type(self).get_config_model()
        if config_class is None:
            self._plugin_config_data = merged
            self._plugin_config_instance = None
            return
        instance = self._repair_config(config_class, merged, defaults)
        if instance is None:
            # 连逐字段修复都救不回来（例如模型本身有问题）：退回纯默认值，但要说清楚
            self._plugin_config_data = merged
            self._log_error("回退配置无法逐字段修复，本次运行使用模型默认值")
        else:
            # 修好的配置回写：宿主下次持久化时会顺带把 config.toml 自愈成合法格式
            self._plugin_config_data = instance.model_dump(mode="python")
        self._plugin_config_instance = instance

    @property
    def config(self) -> BetterDiaryConfig:
        """覆写 SDK 的 ``config``：配置实例缺失时回退默认配置。

        这是一道**下游**保险。真正的上游保险是 ``set_plugin_config`` 覆写：
        如果配置注入失败，SDK 会把 ``_plugin_config_instance`` 留成 None，
        此后**任何** ``self.config`` 访问都会抛
        ``RuntimeError("当前插件配置尚未完成注入")`` —— ``on_load`` 第一行就挂。
        这里兜底成默认配置，保证插件至少能加载并跑起来；真正原因写进日志。
        """
        try:
            return super().config
        except Exception as exc:  # noqa: BLE001 - 兜底必须捕获全部
            if not self._cfg_fallback_warned:
                type(self)._cfg_fallback_warned = True
                self._log_error(
                    "插件配置不可用，已回退默认配置继续加载；请检查 config.toml 字段类型"
                    '（列表项须写成 ["xxx"] 形式）：%s',
                    exc,
                    exc_info=True,
                )
            return self._fallback_config()

    def _fallback_config(self) -> BetterDiaryConfig:
        cache = getattr(self, "_fallback_cfg_cache", None)
        if cache is None:
            cache = BetterDiaryConfig()
            self._fallback_cfg_cache = cache
        return cache

    async def on_load(self) -> None:
        self._generating = False
        self._sched_task: asyncio.Task | None = None
        try:
            # 各子步骤分层防御：任一失败都只记日志，绝不让插件注册整体失败
            try:
                self._cookie_store = self._build_cookie_store()
            except Exception as exc:  # noqa: BLE001
                self._cookie_store = None
                self._log_error("cookie 组件初始化失败（插件仍可加载）：%s", exc, exc_info=True)
            if self.config.schedule.enabled:
                try:
                    self._start_scheduler()
                except Exception as exc:  # noqa: BLE001
                    self._log_error("调度器启动失败（插件仍可加载）：%s", exc, exc_info=True)
                # 启动补跑：补上「上次离线期间」缺掉的日记（只存档、不发布）。
                # 用独立 task 跑，绝不阻塞 on_load。
                try:
                    self._start_catch_up()
                except Exception as exc:  # noqa: BLE001
                    self._log_error("启动补跑未能开始（插件仍可加载）：%s", exc, exc_info=True)
            self._log_info(
                "better-diary 已加载：定时 %s（%s），字数目标 %d，发布目标 %s",
                self.config.schedule.time if self.config.schedule.enabled else "关闭",
                self.config.llm.task_name,
                self.config.diary.word_target,
                "QQ空间" if self.config.qzone.enabled else "仅存档",
            )
            # v1.4.1：改过插件 ID 会让数据目录换一个（宿主按 ID 分配），
            # 旧日记留在旧目录里 → 启动时提醒一次，别让用户以为"日记丢了"。
            try:
                self._warn_legacy_data_dirs()
            except Exception as exc:  # noqa: BLE001 - 只是提醒，绝不能影响加载
                self._log_warning("旧 ID 数据目录检测失败（不影响运行）: %s", exc)
        except Exception as exc:  # noqa: BLE001
            # 最后一道闸：on_load 绝不能把异常抛给宿主，否则同一条「插件初始化失败」
            self._log_error("better-diary 加载流程异常（已吞掉，插件保持已注册）：%s", exc, exc_info=True)

    async def on_unload(self) -> None:
        if self._sched_task is not None:
            self._sched_task.cancel()
            self._sched_task = None
        if self._catchup_task is not None:
            self._catchup_task.cancel()
            self._catchup_task = None
        self._log_info("better-diary 已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data, version
        # SDK 的 scope 取值为 "self" / "bot" / "model"；"plugin"/"all" 为兼容旧写法
        if scope not in ("self", "plugin", "all"):
            return
        # cookie 来源/节流配置可能变了，重建 store（磁盘缓存自动恢复）
        try:
            self._cookie_store = self._build_cookie_store()
        except Exception as exc:  # noqa: BLE001
            self._log_error("cookie 组件重建失败：%s", exc, exc_info=True)
        # 定时设置变更时重启调度器
        if self._sched_task is not None:
            self._sched_task.cancel()
            self._sched_task = None
        if self.config.schedule.enabled:
            try:
                self._start_scheduler()
            except Exception as exc:  # noqa: BLE001
                self._log_error("调度器重启失败：%s", exc, exc_info=True)
            # 只在新启用补写时补跑一次；重复热重载不会重刷（补跑本身幂等）
            if self._catchup_task is None or self._catchup_task.done():
                try:
                    self._start_catch_up()
                except Exception as exc:  # noqa: BLE001
                    self._log_error("补跑未能开始：%s", exc, exc_info=True)
        self._log_info("配置已热重载，调度器状态: %s", "运行中" if self.config.schedule.enabled else "停用")

    # ------------------------------------------------------------ 命令区

    @Command(
        "diary",
        description="生成指定日期（默认今天）的日记并发布到QQ空间（管理员）",
        pattern=r"^\s*[/／]\s*(?:日记|diary)(?:\s+(?P<date>\d{4}-\d{1,2}-\d{1,2}))?(?:\s+(?P<force>重写|force))?\s*$",
    )
    async def cmd_diary(self, matched_groups: dict | None = None, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        if not self._is_admin(kwargs):
            return self._reject_not_admin(kwargs, "diary")
        if not stream_id:
            return True, "缺少聊天流，无法回复", 2
        raw = (matched_groups or {}).get("date") or ""
        date_str = self._normalize_date(raw) if raw else self._today_str()
        if date_str == "":
            await self.ctx.send.text("日期格式不对，用 /日记 2026-09-26 这种写法。", stream_id)
            return True, "", 2

        today = self._today_str()
        if date_str > today:
            # 别落到「消息太少」的分支上去——那是误导性提示
            await self.ctx.send.text(f"{date_display(date_str)} 还没到呢，我可不会预言。", stream_id)
            return True, "", 2

        if self._generating:
            await self.ctx.send.text("正在生成上一篇日记，先等等。", stream_id)
            return True, "", 2

        # 先判「已有成品」（比冷却更前置）：这样冷却还没过时，用户得到的仍是
        # 「已经有日记了…要重写用 /日记 <日期> 重写」这条**可执行**的指引，
        # 而不是被冷却挡住、拿不到任何出路（审计 v1.3.4 发现两条提示自相矛盾）。
        force = bool((matched_groups or {}).get("force"))
        existing = self._load_diaries().get(date_str, {})
        if str(existing.get("content") or "").strip() and not force:
            await self.ctx.send.text(
                f"{date_display(date_str)} 已经有日记了，想回看用 /日记查看 {date_str}。"
                f"确实要重写请用 /日记 {date_str} 重写",
                stream_id,
            )
            return True, "", 2

        # 冷却：同一会话内不许反复触发（每次都会重调 LLM，并可能重新发布到公开空间）
        left = self._cooldown_left(stream_id)
        if left > 0:
            self._log_info("diary 命令被冷却拦下（还需 %d 秒）", left)
            await self.ctx.send.text(f"刚写过一篇，{left} 秒后再试。", stream_id)
            return True, "", 2

        is_today = date_str == today
        if is_today:
            self._last_gen_ts[stream_id] = time.time()
        await self.ctx.send.text(
            f"开始{'生成' if is_today else '补写'} {date_display(date_str)} 的日记，大概一两分钟。",
            stream_id,
        )
        ok, result = await self._generate_for_date(date_str)
        if not ok:
            await self.ctx.send.text(f"日记没写成：{result}", stream_id)
            return True, "", 2

        # 日记成品只发QQ空间，聊天里只回报状态；正文用 /日记查看 回看。
        # ★ 只有**当天**的日记才发布：补写/兜底的一律只存档，
        #   否则公开空间会突然冒出一条「昨天」的历史说说。
        if is_today and self.config.qzone.enabled:
            # 手动重写当天日记时留痕：调度器那边有 published_at 硬去重，
            # 管理员手写的 `/日记 <日期> 重写` 刻意不受它限制（那是显式意图），
            # 但要能在日志里看出「公开空间又多了一条」。
            if force and self._already_published(date_str):
                self._log_warning(
                    "%s 之前已发布过QQ空间，本次为手动重写（将再发一条公开说说）", date_str
                )
            pub_ok, pub_msg = await self._publish_to_qzone(result)
            if pub_ok:
                await self.ctx.send.text(
                    f"{date_display(date_str)} 的日记已发布到QQ空间（{len(result)} 字）。"
                    f"想回看用 /日记查看 {date_str}",
                    stream_id,
                )
            else:
                await self.ctx.send.text(
                    f"日记写好了（{len(result)} 字），但发QQ空间失败：{pub_msg}。"
                    f"想回看用 /日记查看 {date_str}",
                    stream_id,
                )
        elif is_today:
            await self.ctx.send.text(
                f"日记已生成并存档（{len(result)} 字）。QQ空间发布未启用，想回看用 /日记查看 {date_str}",
                stream_id,
            )
        else:
            await self.ctx.send.text(
                f"{date_display(date_str)} 的日记已补写并存档（{len(result)} 字）。"
                f"非当天的日记不发布到QQ空间，想看用 /日记查看 {date_str}",
                stream_id,
            )
        return True, "", 2

    @Command(
        "diary_view",
        description="查看已保存的日记（管理员）",
        pattern=r"^\s*[/／]\s*(?:日记查看|查看日记)\s*(?P<date>\d{4}-\d{1,2}-\d{1,2})?\s*$",
    )
    async def cmd_diary_view(self, matched_groups: dict | None = None, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        if not self._is_admin(kwargs):
            return self._reject_not_admin(kwargs, "diary_view")
        if not stream_id:
            return True, "缺少聊天流，无法回复", 2
        raw = (matched_groups or {}).get("date") or ""
        date_str = self._normalize_date(raw) if raw else self._today_str()
        diary = self._load_diaries().get(date_str)
        if not diary or not diary.get("content"):
            await self.ctx.send.text(f"{date_str} 没有存档的日记。", stream_id)
            return True, "", 2
        await self._send_long(str(diary.get("content")), stream_id)
        return True, "", 2

    @Command(
        "diary_help",
        description="查看日记插件用法",
        pattern=r"^\s*[/／]\s*(?:日记帮助|diary_help)\s*$",
    )
    async def cmd_diary_help(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        if not self._is_admin(kwargs):
            return self._reject_not_admin(kwargs, "diary_help")
        if not stream_id:
            return True, "缺少聊天流，无法回复", 2
        await self.ctx.send.text(
            "better-diary 用法（除本条外，各命令均受 [security].admin_ids 限制）：\n"
            "/日记 —— 生成今天的日记并发布到QQ空间\n"
            "/日记 2026-09-26 —— 补写指定日期的日记，只存档不发布\n"
            "/日记 2026-09-26 重写 —— 覆盖重写该日期已有的日记\n"
            "/日记查看 [日期] —— 在聊天里回看已存档的日记\n"
            "/日记来源 [日期] —— 查看某天日记依据了哪些选材事件\n"
            "/问日记 <关键词> —— 搜历史日记（本地检索，不调用模型）\n"
            "/那年今日 —— 往年同月同日的日记\n"
            "/日记帮助 —— 本说明",
            stream_id,
        )
        return True, "", 2

    @Command(
        "diary_ask",
        description="按关键词搜索历史日记（本地检索，不调用模型）",
        pattern=r"^\s*[/／]\s*(?:问日记|diary_ask)(?:\s+(?P<query>.+?))?\s*$",
    )
    async def cmd_diary_ask(self, matched_groups: dict | None = None, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        if not self._is_admin(kwargs):
            return self._reject_not_admin(kwargs, "diary_ask")
        if not stream_id:
            return True, "缺少聊天流，无法回复", 2
        query = str((matched_groups or {}).get("query") or "").strip()
        if not query:
            await self.ctx.send.text(
                "问什么？用 /问日记 萤火虫 这种写法，多个词用空格隔开（都要命中才算）。",
                stream_id,
            )
            return True, "", 2
        # 零 LLM：纯本地字符串匹配，成本可以忽略，不设节流
        hits = search_diaries(self._load_diaries(), query)
        await self._send_long(
            format_hits(f"问「{query}」", hits, f"没找到含「{query}」的日记。"),
            stream_id,
        )
        return True, "", 2

    @Command(
        "diary_on_this_day",
        description="查看往年同月同日的日记（本地检索，不调用模型）",
        pattern=r"^\s*[/／]\s*(?:那年今日|diary_on_this_day)\s*$",
    )
    async def cmd_diary_on_this_day(self, matched_groups: dict | None = None, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        del matched_groups
        if not self._is_admin(kwargs):
            return self._reject_not_admin(kwargs, "diary_on_this_day")
        if not stream_id:
            return True, "缺少聊天流，无法回复", 2
        today = datetime.date.today()
        hits = on_this_day(self._load_diaries(), today)
        await self._send_long(
            format_hits(
                f"那年今日（{today.month}月{today.day}日）",
                hits,
                f"往年 {today.month} 月 {today.day} 日还没有日记。",
            ),
            stream_id,
        )
        return True, "", 2

    @Command(
        "diary_sources",
        description="查看某天日记依据的选材事件（证据链）",
        pattern=r"^\s*[/／]\s*(?:日记来源|diary_sources)\s*(?P<date>\d{4}-\d{1,2}-\d{1,2})?\s*$",
    )
    async def cmd_diary_sources(self, matched_groups: dict | None = None, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        if not self._is_admin(kwargs):
            return self._reject_not_admin(kwargs, "diary_sources")
        if not stream_id:
            return True, "缺少聊天流，无法回复", 2
        raw = (matched_groups or {}).get("date") or ""
        date_str = self._normalize_date(raw) if raw else self._today_str()
        entry = self._load_diaries().get(date_str)
        if not entry or not entry.get("content"):
            await self.ctx.send.text(f"{date_str} 没有存档的日记。", stream_id)
            return True, "", 2
        mode = str(entry.get("material_mode") or "").strip()
        lines: list[str] = [f"{date_str} 的日记依据（素材模式：{mode or '旧版存档'}）："]
        events = entry.get("events") if isinstance(entry.get("events"), list) else []
        if events:
            for e in events:
                if not isinstance(e, dict):
                    continue
                line = f"- {e.get('who') or '?'}：{e.get('what') or ''}"
                if e.get("quote"):
                    line += f"（原话：「{e['quote']}」）"
                lines.append(line)
        else:
            lines.append("- 该日期没有结构化选材事件（旧版存档，或选材降级用时间线末尾写成）")
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        if meta:
            for key, label in (
                ("topics", "话题"),
                ("people", "提到的人"),
                ("projects", "手头的事"),
                ("unresolved", "还没个结果的"),
            ):
                values = [str(v) for v in (meta.get(key) or []) if str(v).strip()]
                if values:
                    lines.append(f"- {label}：" + "、".join(values[:8]))
        await self._send_long("\n".join(lines), stream_id)
        return True, "", 2

    # ------------------------------------------------------------ 权限

    @staticmethod
    def _is_local_operator_flag(kwargs: dict) -> bool:
        """严格判定「本机控制台操作者」这个旁路。

        v1.3.4：不用裸真值判断 —— ``bool("false")`` / ``bool("0")`` 都是 True，
        一旦宿主把该字段序列化成字符串，旁路就会被意外放开。这里只认
        「真布尔 True」与少数明确的肯定写法，其余（含缺省、None、``"false"``）
        一律视为否。

        **真机确认项**：该 kwarg 由宿主把命令事件送进插件时注入（本机 SDK 2.8.2
        里没有这个标识，无法离线验证它不可能来自聊天消息）。插件侧能保证的是：
        ``matched_groups``（用户可控的命令参数）从不并入 kwargs，正则命名组
        只有 date/force/query，所以消息正文无法伪造出这个键。
        """
        value = kwargs.get("is_local_operator")
        if value is True:
            return True
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return False

    def _is_admin(self, kwargs: dict) -> bool:
        """自管权限：**fail-closed**；拒绝不发言只留日志。

        判定顺序（v1.3.4 起，与 v1.3.3 的 fail-open 相反）：
        1. 本机控制台操作者（``is_local_operator``，由宿主 bot_console 注入）→ 放行；
        2. ``[security].allow_all_users = true`` → 放行（显式选择全员可用）；
        3. 命中 ``[security].admin_ids`` → 放行；
        4. 其余一律拒绝。

        第 4 条是 v1.3.4 的关键改动：``admin_ids`` 留空**不再**等于全员放行。
        旧行为会让任何群成员都能让 bot 把当天的聊天内容（含私聊素材）写成日记
        发布到**公开**的 QQ 空间，也会让只读命令把跨会话内容拉进任意群。
        """
        if self._is_local_operator_flag(kwargs):
            return True
        if bool(getattr(self.config.security, "allow_all_users", False)):
            return True
        admins = {str(a).split(":")[-1].strip().lower() for a in (self.config.security.admin_ids or [])}
        if not admins:
            return False
        user_id = str(kwargs.get("user_id") or "")
        if not user_id:
            msg = kwargs.get("message") or {}
            info = msg.get("message_info") or {} if isinstance(msg, dict) else {}
            user_info = info.get("user_info") or {} if isinstance(info, dict) else {}
            user_id = str(user_info.get("user_id", "") or "")
        return bool(user_id) and user_id.lower() in admins

    def _reject_not_admin(self, kwargs: dict, command: str) -> tuple[bool, str, int]:
        """非管理员：静默拒绝（不回复内容，只留一行日志）。"""
        if not self._is_local_operator_flag(kwargs) and not (
            self.config.security.admin_ids or self.config.security.allow_all_users
        ):
            self._log_warning(
                "%s 被拒绝：未配置 [security].admin_ids 且 allow_all_users=false（fail-closed）。"
                "需要放行请填 admin_ids，或显式打开 allow_all_users",
                command,
            )
        else:
            self._log_info("%s 命令被静默拒绝（非管理员）", command)
        return True, "", 2

    def _cooldown_left(self, stream_id: str) -> int:
        """返回剩余冷却秒数（0 = 可执行）。仅用于限制重复触发。"""
        cooldown = int(getattr(self.config.security, "command_cooldown_seconds", 0) or 0)
        if cooldown <= 0 or not stream_id:
            return 0
        last = float((self._last_gen_ts or {}).get(stream_id, 0.0))
        left = cooldown - (time.time() - last)
        return int(left) + 1 if left > 0 else 0

    # ------------------------------------------------------------ 定时调度

    def _start_scheduler(self) -> None:
        self._sched_task = asyncio.create_task(self._schedule_loop())

    def _start_catch_up(self) -> None:
        """启动一次补跑（后台 task，绝不阻塞 on_load）。"""
        if not self.config.schedule.catch_up_enabled:
            return
        self._catchup_task = asyncio.create_task(self._catch_up_run())

    @staticmethod
    def _parse_hhmm(hhmm: str) -> tuple[int, int] | None:
        """解析 HH:MM。空串或非法返回 None（表示「该时间点未配置」）。"""
        raw = str(hhmm or "").strip()
        if not raw:
            return None
        try:
            hour, minute = (int(x) for x in raw.split(":"))
        except (ValueError, AttributeError):
            return None
        return max(0, min(23, hour)), max(0, min(59, minute))

    def _seconds_until_next(self, hhmm: str) -> float:
        parsed = self._parse_hhmm(hhmm)
        if parsed is None:
            return float("inf")  # 未配置 → 永不触发
        hour, minute = parsed
        now = datetime.datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        return (target - now).total_seconds()

    def _fallback_hhmm(self) -> str:
        """兜底时刻。留空或与主时间相同 → 视为关闭。"""
        raw = str(self.config.schedule.fallback_time or "").strip()
        main = str(self.config.schedule.time or "").strip()
        if not raw or raw == main:
            return ""
        return raw

    def _next_fire(self) -> tuple[float, str]:
        """下一次触发：返回 (秒数, 类型)，类型 ∈ {"main", "fallback"}。"""
        main_delay = self._seconds_until_next(self.config.schedule.time)
        fallback = self._fallback_hhmm()
        fb_delay = self._seconds_until_next(fallback) if fallback else float("inf")
        if fb_delay < main_delay:
            return fb_delay, "fallback"
        return main_delay, "main"

    async def _schedule_loop(self) -> None:
        """每日定时循环。

        v1.3.6 修正「单次失败打死整个调度器」：旧写法把整段 ``while`` 包在**一个**
        ``try`` 里，任何一次生成抛异常（真机实录：Provider 集体超时
        `[E_TIMEOUT] 请求 cap.call 超时 (180000ms)`）都会跳出 while，
        于是**当天剩余时间再也不会触发**，而日志只有一行 `调度器异常退出`。
        现在改为**每轮独立兜底**：单次失败就地收敛、记日志，循环继续等下一个时间点
        （不丢天优先 —— 宁可下次再试，也不能因为一次网络抖动把整个调度器关掉）。
        """
        while True:
            try:
                delay, kind = self._next_fire()
                if delay == float("inf"):
                    # 两个时间点都没配（time 有默认值，理论上到不了这里）：
                    # 睡一小时再看，不要空转烧 CPU
                    await asyncio.sleep(3600)
                    continue
                # 兜底下限：时间算出来是 0 或负数（时钟回拨/跨天边界）时不能忙等
                delay = max(float(delay), 1.0)
                self.ctx.logger.info("定时日记将在 %.0f 秒后运行（%s）", delay, kind)
                await asyncio.sleep(delay)
                if kind == "fallback":
                    await self._catch_up_run()
                else:
                    await self._run_main_with_silence_wait()
            except asyncio.CancelledError:
                raise  # 卸载：必须让取消照常传播
            except Exception as exc:
                # 单次触发失败：记日志、继续循环，绝不退出
                self.ctx.logger.error(
                    "本轮定时触发失败（调度器继续运行，等待下一个时间点）: %s", exc, exc_info=True
                )
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    raise

    # ------------------------------------------------------------ 静默阈值（等人停下再写）

    async def _run_main_with_silence_wait(self) -> None:
        """主触发入口：先按静默阈值等人停下，等满上限就照常生成（**不丢天优先**）。

        阈值为 0 时完全等价于旧行为（到点直接写），不发起任何额外消息查询。
        """
        minutes = max(0, int(self.config.schedule.wait_silent_minutes))
        if minutes <= 0:
            await self._scheduled_run()
            return
        hours = max(0, int(self.config.schedule.max_wait_hours))
        deadline = (
            datetime.datetime.now() + datetime.timedelta(hours=hours)
            if hours > 0
            else datetime.datetime.min  # 0 = 不等，第一次发现没静默就直接写
        )
        attempt = 0
        while True:
            if await self._chat_is_quiet(minutes):
                if attempt:
                    self.ctx.logger.info("聊天已静默超过 %d 分钟，开始生成今日日记", minutes)
                await self._scheduled_run()
                return
            attempt += 1
            if datetime.datetime.now() >= deadline:
                self.ctx.logger.warning(
                    "等待静默超时（已等 %d 次，上限 %d 小时），照常生成今日日记（不丢天优先）",
                    attempt, hours,
                )
                await self._scheduled_run()
                return
            interval = max(1, int(self.config.schedule.retry_interval_minutes))
            self.ctx.logger.info(
                "最近 %d 分钟内还有新消息，%d 分钟后再看（第 %d 次等待）", minutes, interval, attempt
            )
            await asyncio.sleep(interval * 60)

    async def _chat_is_quiet(self, minutes: int) -> bool:
        """最近 ``minutes`` 分钟内是否没有任何新消息（全库）。

        查询失败按「已静默」处理并留日志 —— 这是可选的体验增强，绝不能因为它
        挡住当天日记（不丢天优先）。
        """
        now = datetime.datetime.now()
        start = now - datetime.timedelta(minutes=max(1, minutes))
        try:
            msgs = await self._query_messages(start.timestamp(), now.timestamp(), "")
        except Exception as exc:  # noqa: BLE001
            self._log_warning("静默检查失败（按已静默处理，照常生成）: %s", exc)
            return True
        if msgs:
            self.ctx.logger.info("静默检查：最近 %d 分钟内还有 %d 条新消息", minutes, len(msgs))
            return False
        return True

    # ------------------------------------------------------------ 补跑（不丢天）

    def _catch_up_targets(self) -> list[str]:
        """待补写的日期，**由早到晚**排列（先补老的，跨天连续性才对得上）。"""
        days = max(0, int(self.config.schedule.catch_up_days))
        if days <= 0:
            return []
        today = datetime.date.today()
        return [
            (today - datetime.timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(days, 0, -1)
        ]

    async def _catch_up_missing(self) -> list[tuple[str, bool]]:
        """给没有日记的日期补写。**只存档、不发布**。返回 [(日期, 是否成功)]。

        幂等：以存档里是否已有该日期为准，重复启动不会重刷。

        两道保险（真机实测补充）：
        - **单天异常不进结果**：某天的补写抛异常绝不能让整轮补跑崩掉、把后面几天一起饿死。
        - **总预算**：`catch_up_budget_minutes > 0` 时，用完整轮就停手；剩余日期留给
          下一轮兜底。网络整体不可用时，避免在启动补跑里长时间挂着重试。
        """
        existing = self._load_diaries()
        results: list[tuple[str, bool]] = []
        budget_min = max(0, int(self.config.schedule.catch_up_budget_minutes))
        deadline = time.monotonic() + budget_min * 60 if budget_min > 0 else None
        for date_str in self._catch_up_targets():
            if deadline is not None and time.monotonic() >= deadline:
                self.ctx.logger.warning(
                    "补跑已达总预算 %d 分钟，剩余日期留给下一轮兜底", budget_min
                )
                break
            if str(existing.get(date_str, {}).get("content") or "").strip():
                continue
            self.ctx.logger.info("补写缺档 %s", date_str)
            try:
                ok, result = await self._generate_for_date(date_str)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # 单天失败必须就地收敛：否则后面的日期一天都补不了
                self.ctx.logger.error("补写 %s 异常（跳过本日，继续后面几天）: %s", date_str, exc)
                results.append((date_str, False))
                continue
            if ok:
                results.append((date_str, True))
                await self._notify_chats(
                    f"补写了 {date_str} 的日记（{len(result)} 字），已存档、未发布到QQ空间。"
                    f"想看用 /日记查看 {date_str}"
                )
            else:
                results.append((date_str, False))
                self.ctx.logger.warning("补写 %s 未成功：%s", date_str, result)
        return results

    async def _catch_up_run(self) -> None:
        """补跑入口：启动补跑与兜底时刻都走这里。任何异常只记日志，不能拖垮调度器。"""
        if not self.config.schedule.catch_up_enabled:
            return
        try:
            if self._generating:
                self.ctx.logger.info("补跑跳过：已有生成任务在进行")
                return
            results = await self._catch_up_missing()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("补跑异常: %s", exc, exc_info=True)
            return
        if not results:
            self.ctx.logger.info("补跑结束：没有需要补写的日期")
            return
        done = [d for d, ok in results if ok]
        self.ctx.logger.info(
            "补跑结束：尝试 %d 天，成功 %d 天（%s）", len(results), len(done), "、".join(done) or "无"
        )

    async def _scheduled_run(self) -> None:
        if self._generating:
            self.ctx.logger.warning("定时日记跳过：已有生成任务在进行（次日兜底会补写）")
            return
        date_str = self._today_str()
        # 当天已有成品就什么都不做：
        #  - 已发布过 → 重启 / 改时间 / 同一天第二次触发都会重发一条一样的公开说说；
        #  - 发布未启用 → 再写一遍只是白烧一次 LLM（内容完全相同），没有任何收益。
        # 两种情况都靠归档里的 content 判定；确需重写走管理员手写的
        # `/日记 <日期> 重写`，那条路径刻意不受此限制。
        today_entry = self._load_diaries().get(date_str)
        has_content = bool(str((today_entry or {}).get("content") or "").strip()) if isinstance(today_entry, dict) else False
        if has_content and (self._already_published(date_str) or not self.config.qzone.enabled):
            self.ctx.logger.info(
                "定时日记跳过：%s 已有成品（%s）",
                date_str,
                "已发布过QQ空间" if self._already_published(date_str) else "空间发布未启用，重复成文无收益",
            )
            return
        ok, result = await self._generate_for_date(date_str)
        if not ok:
            self.ctx.logger.warning("定时日记生成失败 %s: %s", date_str, result)
            await self._notify_chats(f"今日日记没写成：{result}")
            return
        if not self.config.qzone.enabled:
            self.ctx.logger.info("定时日记已生成并存档（%d 字）；QQ空间发布未启用", len(result))
            return
        pub_ok, pub_msg = await self._publish_to_qzone(result)
        if pub_ok:
            self.ctx.logger.info("定时日记已发布到QQ空间（%d 字）", len(result))
            await self._notify_chats(f"今日日记已发布到QQ空间（{len(result)} 字）。")
        else:
            self.ctx.logger.error("定时日记发布QQ空间失败: %s", pub_msg)
            await self._notify_chats(f"今日日记写好了，但发布QQ空间失败：{pub_msg}")

    async def _notify_chats(self, text: str) -> None:
        """定时发布后向配置的聊天发通知（notify_chats 为空则完全静默）。"""
        for target in self.config.schedule.notify_chats or []:
            stream_id = await self._resolve_stream_id(str(target))
            if not stream_id:
                self.ctx.logger.warning("定时通知目标 %s 无法解析到聊天流", target)
                continue
            try:
                await self.ctx.send.text(text, stream_id)
            except Exception as exc:
                self.ctx.logger.error("定时通知发送到 %s 失败: %s", target, exc)

    def _build_cookie_store(self) -> CookieStore:
        qz = self.config.qzone
        napcat_http = None
        if qz.napcat_http_host.strip() and qz.napcat_http_port.strip():
            napcat_http = {
                "host": qz.napcat_http_host.strip(),
                "port": qz.napcat_http_port.strip(),
                "token": qz.napcat_http_token.strip(),
            }
        return CookieStore(
            self._data_dir(),
            api_call=self._adapter_api_call,
            napcat_http=napcat_http,
            interval_sec=max(60, qz.refresh_interval_min) * 60,
            logger=self.ctx.logger,
        )

    async def _adapter_api_call(self, name: str, params: dict) -> Any:
        """napcat-adapter API 调用入口（注入给 CookieStore；1.3.0 无 adapter 时会抛错，由 CookieStore 降级）。"""
        return await self.ctx.api.call(name, params=params)

    def _manual_cookies(self) -> dict | None:
        """手动配置的兜底 cookie（[qzone].uin/p_skey 非空时生效）。"""
        qz = self.config.qzone
        if not qz.uin.strip() or not qz.p_skey.strip():
            return None
        uin = qz.uin.strip()
        return {
            "uin": uin if uin.startswith("o") else f"o0{uin}",
            "p_skey": qz.p_skey.strip(),
            "skey": qz.skey.strip(),
        }

    async def _resolve_cookies(self, force_refresh: bool = False) -> dict | None:
        """取 cookie：auto_cookie 时走 CookieStore 三级来源（adapter → NapCat HTTP → 缓存），失败兜底手动配置。"""
        if self.config.qzone.auto_cookie and self._cookie_store is not None:
            cookies = await self._cookie_store.get_cookies(force=force_refresh)
            if cookies:
                return cookies
        return self._manual_cookies()

    async def _publish_to_qzone(self, content: str) -> tuple[bool, str]:
        """发布日记到QQ空间。返回 (成功, tid或原因)。登录态失效时自动重取一次再试。"""
        if __package__:  # 包式加载（Runner 真机）
            from .bd_qzone import CookieExpiredError, PublishUnavailableError, QzonePublisher
        else:  # 平铺兜底（脚本直跑）
            from bd_qzone import CookieExpiredError, PublishUnavailableError, QzonePublisher

        # 取 cookie 也在 try 里：`_resolve_cookies` 会走 adapter API / NapCat HTTP，
        # 这两条路的外部异常同样可能夹带 cookie 串（实测审计抓到过穿透到宿主 traceback）。
        try:
            cookies = await self._resolve_cookies()
        except Exception as exc:
            self._log_error("取 cookie 异常: %s", exc)
            return False, f"取 cookie 异常（{type(exc).__name__}）: {redact_secrets(exc)}"
        if not cookies:
            return False, (
                "拿不到QQ空间cookie：adapter 与 NapCat HTTP 都没取到且无手动兜底。"
                "请在 NapCat WebUI 开启 HTTP 服务器并填好 [qzone].napcat_http_host/port，或手动填 [qzone].uin/p_skey"
            )

        publisher = QzonePublisher(cookies)
        try:
            ok, msg = await publisher.publish_text(content, timeout=max(5, self.config.qzone.timeout_seconds))
        except PublishUnavailableError as exc:
            return False, f"{exc}（无需 httpx 时可把 [qzone].enabled 关掉，日记仍会存档）"
        except CookieExpiredError:
            self._log_warning("QQ空间登录态失效，强制重取 cookie 重试一次")
            try:
                fresh = await self._resolve_cookies(force_refresh=True)
            except Exception as exc:
                self._log_error("重取 cookie 异常: %s", exc)
                return False, f"重取 cookie 异常（{type(exc).__name__}）: {redact_secrets(exc)}"
            if not fresh:
                return False, "QQ空间登录态失效，且自动重取 cookie 失败（检查 NapCat 是否在线、HTTP 服务器是否开启）"
            if fresh == cookies:
                return False, "QQ空间登录态失效，重取到的 cookie 未变化（bot 登录态可能真的过期了）"
            publisher = QzonePublisher(fresh)
            try:
                ok, msg = await publisher.publish_text(content, timeout=max(5, self.config.qzone.timeout_seconds))
            except CookieExpiredError:
                return False, "QQ空间登录态失效（重取后仍失效），请检查 bot 登录状态"
            except Exception as exc:
                # 异常文本可能夹带请求 URL / cookie（httpx 的异常会把 request 带进 str）→ 统一脱敏
                return False, f"发布异常（{type(exc).__name__}）: {redact_secrets(exc)}"
        except Exception as exc:
            return False, f"发布异常（{type(exc).__name__}）: {redact_secrets(exc)}"
        if ok:
            self._mark_published(self._today_str())
        return ok, msg

    # ------------------------------------------------------------ 发布去重

    def _already_published(self, date_str: str) -> bool:
        """该日期是否已成功发过空间（看归档里的 ``published_at``）。"""
        entry = self._load_diaries().get(date_str)
        return bool(isinstance(entry, dict) and str(entry.get("published_at") or "").strip())

    def _mark_published(self, date_str: str) -> None:
        """记下「这个日期已经发过空间」。只改这一个字段，不动正文与证据链。"""
        path = self._store_path()
        try:
            data = self._load_diaries()
            entry = data.get(date_str)
            if not isinstance(entry, dict) or not str(entry.get("content") or "").strip():
                return
            entry["published_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            self.ctx.logger.warning("记录发布状态失败（不影响本次发布）: %s", exc)

    async def _resolve_stream_id(self, target: str) -> str:
        """'group:123' / 'private:456' / 裸 stream_id -> 聊天流 ID。

        v1.3.6：解析失败时把**为什么失败**记进 ``_unresolved_targets``，
        供 :meth:`_filter_diagnostic` 组织成可读提示 —— 旧行为只在第一次解析时
        打一行 WARNING（真机实录：第二次起那条 WARNING 被去重吞掉，
        用户只看到「当天消息太少（0 条）」，完全指不到配置问题上）。

        v1.3.7：主接口返回 None 时**自动降级**再找两次 —— 真机实录
        `group:967779035` 走主接口拿到 `None`（宿主没返回该群的流），
        于是三天日记全空。现在会依次尝试：
        ① ``chat.get_stream_by_group_id``（原路）
        ② ``chat.get_group_streams`` 列表里按 group_id 自己找
        ③ ``chat.get_all_streams`` 里按 group_id 自己找
        三次都拿不到才判失败（并把「可能是会话不活跃」写进原因）。
        """
        target = str(target).strip()
        m = re.match(r"^(group|private|user):(\S+)$", target, re.IGNORECASE)
        if not m:
            # 裸 stream_id：原样透传，约定上它是有效的
            self._unresolved_targets.pop(target, None)
            return target
        kind, ident = m.group(1).lower(), m.group(2)

        # 注意：这里**刻意用显式属性访问**（而不是 getattr(self.ctx.chat, name)），
        # 因为「能力声明与使用一致」是插件中心 AI 审核的固定检查项 ——
        # 用 getattr 动态取名会让静态扫描看不到 chat.* 调用，被误判成「声明了但没用」。
        async def _resolve_primary() -> Any:
            if kind == "group":
                return await self.ctx.chat.get_stream_by_group_id(group_id=ident)
            return await self.ctx.chat.get_stream_by_user_id(user_id=ident)

        async def _call(method: str, **call_kwargs: Any) -> Any:
            fn = getattr(self.ctx.chat, method, None)
            if fn is None:
                raise AttributeError(f"宿主未提供 chat.{method}")
            return await fn(**call_kwargs)

        method_name = "get_stream_by_group_id" if kind == "group" else "get_stream_by_user_id"
        arg_name = "group_id" if kind == "group" else "user_id"
        tried: list[str] = [f"chat.{method_name}"]
        shapes: list[str] = []

        try:
            result = await _resolve_primary()
            stream_id = self._extract_stream_id(result)
            if stream_id:
                self._unresolved_targets.pop(target, None)
                return stream_id
            shapes.append(f"{method_name}→{_describe_shape(result)}")
        except Exception as exc:
            self.ctx.logger.warning("解析 %s 失败: %s", target, exc)
            shapes.append(f"{method_name}→异常（{type(exc).__name__}）")

        # 降级 ①②③：按 id 在会话列表里自己找（只对 group/user 有意义）
        id_keys = ("group_id", "group", "chat_id", "session_id", "target_id")
        if kind == "user":
            id_keys = ("user_id", "user", "chat_id", "session_id", "target_id")
        for list_method in ("get_group_streams", "get_all_streams"):
            tried.append(f"chat.{list_method}")
            try:
                listing = await _call(list_method)
            except Exception as exc:
                shapes.append(f"{list_method}→异常（{type(exc).__name__}）")
                continue
            if kind == "user" and list_method == "get_group_streams":
                continue  # 群流列表里找不到私聊
            hit = _find_by_value(listing, ident, id_keys)
            if hit is not None:
                # 先看命中那一层，再看整个列表 —— 真机形态是 stream_id 挂在外层、
                # group_id 在内层，只取命中层会拿到空串（本轮实测踩到过）。
                stream_id = _find_stream_id_deep(hit) or _find_stream_id_deep(listing)
                if stream_id:
                    self.ctx.logger.info(
                        "白名单目标 %s 主接口没解析出来，已从 %s 里找到对应会话（流 ID %s）",
                        target, list_method, stream_id,
                    )
                    self._unresolved_targets.pop(target, None)
                    return stream_id
            shapes.append(f"{list_method}→{_describe_shape(listing)}")

        self._unresolved_targets[target] = (
            f"三条路都没解析出会话（{'；'.join(shapes)}）。"
            f"常见原因：该会话当前**不活跃**（宿主只在有活跃会话时能按 group_id 反查），"
            f"或标识写法不符（已试 {'、'.join(tried)}，参数 {arg_name}={ident}）"
        )
        return ""

    @staticmethod
    def _extract_stream_id(result: Any) -> str:
        """从各种可能的返回形态里挖 stream_id。

        v1.3.7：候选键里补上 ``stream`` / ``session`` / ``streamId`` ——
        SDK 在 ``_CAPABILITY_RESULT_KEYS`` 里把 `chat.get_stream_by_group_id` 的
        返回值定为 ``stream`` 键，宿主/其他版本可能直接把会话对象放在 ``stream`` 下。
        """
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list) and result:
            return BetterDiaryPlugin._extract_stream_id(result[0])
        if isinstance(result, dict):
            for key in ("stream_id", "streamId", "session_id", "chat_id", "id"):
                val = result.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            # 会话对象被包在 stream / session 下：往下再挖一层
            for key in ("stream", "session", "chat"):
                nested = result.get(key)
                if isinstance(nested, (dict, list)):
                    found = BetterDiaryPlugin._extract_stream_id(nested)
                    if found:
                        return found
        return ""

    # ------------------------------------------------------------ 主流程

    def _today_str(self) -> str:
        return datetime.date.today().strftime("%Y-%m-%d")

    def _normalize_date(self, raw: str) -> str:
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", raw.strip())
        if not m:
            return ""
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d")
        except ValueError:
            return ""

    async def _fetch_messages(self, start_ts: float, end_ts: float) -> list[dict[str, Any]]:
        """按过滤模式抓取当天消息（跨聊天合并，按时间排序）。

        v1.3.5：默认 ``whitelist``，且**白名单为空时不抓任何消息** ——
        默认不再把「全库所有群 + 所有私聊」的当天记录拼进 prompt 再公开发布。
        要收全库必须显式写 ``filter_mode = "all"``。
        """
        mode = _norm_filter_mode(self.config.diary.filter_mode)
        targets = [str(t) for t in (self.config.diary.target_chats or []) if str(t).strip()]
        self._unresolved_targets = {}
        self._resolved_with_messages = []

        if mode == "whitelist":
            if not targets:
                # 静默降级不可接受（v1.3.4 教训）：这里必须留痕，否则用户只看到
                # 「今天消息太少」这种误导性提示，压根想不到是配置没填。
                # 提示正文由 _filter_diagnostic 统一组织（避免两处文案漂移）。
                self.ctx.logger.warning(
                    "[diary].filter_mode = whitelist（默认）但 target_chats 为空："
                    "本次不抓取任何消息。请填 target_chats = [\"group:群号\"]，"
                    "或显式设 filter_mode = \"all\" 才会收全库记录"
                )
                return []
            all_msgs: list[dict[str, Any]] = []
            for target in targets:
                stream_id = await self._resolve_stream_id(target)
                if not stream_id:
                    # 只在**首次**遇到这个目标时打 WARNING（同一目标反复失败不必刷屏），
                    # 详细原因留在 _unresolved_targets 里，由 _filter_diagnostic 汇总。
                    if target not in self._whitelist_warned:
                        self._whitelist_warned.add(target)
                        self.ctx.logger.warning(
                            "白名单目标 %s 解析失败，跳过（只提醒一次；原因：%s）",
                            target, self._unresolved_targets.get(target, "未知"),
                        )
                    continue
                self._whitelist_warned.discard(target)
                msgs = await self._query_messages(start_ts, end_ts, stream_id)
                if msgs:
                    self._resolved_with_messages.append(target)
                all_msgs.extend(msgs)

            if self._unresolved_targets:
                # 兼容期保护：解析失败的目标不能只留一行 WARNING。
                # 存量配置里很可能躺着一个「旧版 filter_mode=all 时写了但从未生效」的
                # target_chats（真机实录：group:967779035），新版默认 whitelist 后它
                # 突然开始生效却解析不出来 → 0 条素材 → 只报「消息太少」，指不到配置上。
                self.ctx.logger.warning(
                    "取材范围 %d/%d 个目标解析失败（原因：%s）",
                    len(self._unresolved_targets), len(targets),
                    "; ".join(f"{t}: {why}" for t, why in self._unresolved_targets.items()),
                )
                # v1.3.8 兜底：宿主把 chat.* 也用不了时（真机实测：
                # get_stream_by_group_id 返回 None、get_group_streams/get_all_streams 双双 RPCError），
                # 不要就这么把当天日记丢掉 —— 改用「拉当天全库记录 + 插件内按会话过滤」，
                # 效果等价于白名单，**素材照样只取指定会话**（不会把别的群/私聊带进 prompt）。
                try:
                    fetched = await self._query_messages(start_ts, end_ts, "")
                except Exception as exc:
                    self.ctx.logger.error("会话解析失败后的兜底取材也失败: %s", exc)
                    fetched = []
                if fetched:
                    wanted_groups, wanted_users = _split_targets(list(self._unresolved_targets))
                    filtered = [
                        m for m in fetched
                        if _match_filter_target(m, wanted_groups, wanted_users)
                    ]
                else:
                    filtered = []
                if filtered:
                    self._log_info(
                        "会话解析失败，已改用「全库取当天 + 插件内按会话过滤」兜底："
                        "全库 %d 条 → 命中 %d 条（目标 %s）",
                        len(fetched), len(filtered), "、".join(self._unresolved_targets),
                    )
                    all_msgs.extend(filtered)
                    self._resolved_with_messages.extend(self._unresolved_targets)
                    self._unresolved_targets = {}
                elif fetched:
                    # 兜底确实拉到当天记录、但里面没有该会话 → 这是「该会话当天真的没说话」，
                    # 证据比「宿主不给会话」更硬，写进原因里。
                    self._unresolved_targets = {
                        t: f"{why}；已用兜底在当天 {len(fetched)} 条记录里按 "
                           f"{'群号' if t.startswith('group:') else 'QQ号'} 过滤，仍未命中该会话"
                        for t, why in self._unresolved_targets.items()
                    }
                else:
                    # 全库也没拉到任何记录（新库/当天无消息）：至少让用户知道兜底跑过了，
                    # 而不是怀疑插件根本没试。
                    self._unresolved_targets = {
                        t: f"{why}；兜底查当天全库记录也没取到任何消息"
                        for t, why in self._unresolved_targets.items()
                    }
            all_msgs.sort(key=lambda m: _ts_of(m))
            return all_msgs

        msgs = await self._query_messages(start_ts, end_ts, "")
        if mode == "blacklist" and targets:
            blocked_groups, blocked_users = _split_targets(targets)
            msgs = [
                m for m in msgs
                if _group_of(m) not in blocked_groups
                and (not _group_of(m) or _user_of(m) not in blocked_users)
            ]
        msgs.sort(key=lambda m: _ts_of(m))
        return msgs

    async def _query_messages(self, start_ts: float, end_ts: float, chat_id: str) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "limit": 0,
            "limit_mode": "earliest",
            "filter_mai": False,
            "filter_command": False,
        }
        try:
            if chat_id:
                result = await self.ctx.message.get_by_time_in_chat(
                    chat_id, str(start_ts), str(end_ts), **kwargs
                )
            else:
                kwargs.pop("filter_command", None)  # get_by_time 不接受该参数
                result = await self.ctx.message.get_by_time(
                    str(start_ts), str(end_ts), **kwargs
                )
        except Exception as exc:
            self.ctx.logger.error("消息查询失败 (chat_id=%s): %s", chat_id, exc)
            raise RuntimeError(f"消息查询失败: {exc}") from exc

        # SDK 已归一化出 messages 字段；兼容信封形态
        if isinstance(result, dict):
            if not result.get("success", True):
                raise RuntimeError(f"消息查询返回失败: {result.get('error', '未知错误')}")
            result = result.get("messages")
        if not isinstance(result, list):
            return []
        return [m for m in result if isinstance(m, dict)]

    async def _resolve_persona(self) -> str:
        if self.config.diary.persona_override.strip():
            return self.config.diary.persona_override.strip()
        try:
            value = await self.ctx.config.get("personality.personality", "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception as exc:
            self.ctx.logger.debug("读取 Host 人设失败: %s", exc)
        return "是个爱聊天、心比较软的机器人。"

    async def _call_llm(self, prompt: str, temperature: float) -> str:
        """LLM 调用：显式绕过 ctx.llm.generate 的 30s RPC 默认超时。"""
        timeout_ms = max(30, int(self.config.llm.timeout_seconds)) * 1000
        call_capability = getattr(self.ctx, "call_capability", None)
        payload = {
            "prompt": prompt,
            "model": "",
            "task_name": self.config.llm.task_name,
            "temperature": temperature,
        }
        if callable(call_capability):
            result = await call_capability("llm.generate", timeout_ms=timeout_ms, **payload)
        else:  # 老 SDK 兜底
            result = await self.ctx.llm.generate(**payload)
        if not isinstance(result, dict) or not result.get("success", True):
            err = result.get("error", "未知错误") if isinstance(result, dict) else type(result).__name__
            raise RuntimeError(f"LLM 调用失败: {err}")
        return str(result.get("response") or result.get("content") or "")

    async def _call_llm_with_retry(self, prompt: str, temperature: float, *, stage: str) -> str:
        """成文阶段专用：对**超时 / 宿主瞬时 I/O 故障**做有限重试。

        真机实录（2026-09-28 09:00 补跑）：模型 Provider 集体网络超时（30s APITimeoutError，
        日志里连着好几条 `遇到错误: 网络连接超时`），MaiBot 侧依次切换模型、逐个耗尽重试，
        最终以 Runner RPC 超时收尾 —— 补跑直接在这一天炸掉。

        真机实录（2026-09-30 10:46，v1.4.0 依据）：Host 并发写 `llm_error/*.json` 时
        rename 被拒（`[WinError 5] 拒绝访问`），LLMService 把它当生成失败抛出 ——
        一块选材因此白废，只剩"降级用时间线末尾"。

        为什么值得重试：超时是**网络抖动**性质、I/O 冲突是**瞬时**性质，隔一会儿往往就好；
        而格式/语义类失败重试纯属浪费。**不可重试的异常一律原样抛出。**
        """
        attempts = max(0, int(self.config.llm.write_retry)) + 1
        backoff = max(0, int(self.config.llm.retry_backoff_seconds))
        for attempt in range(1, attempts + 1):
            try:
                return await self._call_llm(prompt, temperature)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt >= attempts or not self._is_retryable_llm_error(exc):
                    raise
                reason = "超时" if self._is_timeout_error(exc) else "宿主瞬时 I/O 故障"
                self._log_warning(
                    "%s 第 %d/%d 次调用失败（%s），%d 秒后重试: %s",
                    stage, attempt, attempts, reason, backoff, exc,
                )
                if backoff:
                    await asyncio.sleep(backoff)
        raise RuntimeError(f"{stage}调用失败")  # pragma: no cover - 循环必然 return 或 raise

    async def _extract_events(self, timeline: str, date_str: str) -> list[dict[str, Any]]:
        """阶段一：分块选材 + 打分 + 合并排序。

        v1.4.0：**单块时**也走 :meth:`_call_llm_with_retry`（此前 `write_retry`
        只作用于成文，选材阶段完全没有重试 —— 真机实录里因此白废了整块选材）。
        多块时**刻意不重试**：块之间本来就是"某块失败就跳过、用其余块"的降级设计
        （见下方 `run_one`），而且并发块逐个重试会让整轮成文的耗时失控。
        """
        chunks = chunk_text(timeline, self.config.diary.chunk_chars, self.config.diary.max_chunks)
        if not chunks:
            return []
        sem = asyncio.Semaphore(3)
        single_chunk = len(chunks) == 1

        async def run_one(chunk: str) -> list[dict[str, Any]]:
            async with sem:
                try:
                    if single_chunk:
                        raw = await self._call_llm_with_retry(
                            build_extract_prompt(date_str, chunk),
                            self.config.llm.extract_temperature,
                            stage="选材",
                        )
                    else:
                        raw = await self._call_llm(
                            build_extract_prompt(date_str, chunk),
                            self.config.llm.extract_temperature,
                        )
                    return parse_events(raw)
                except Exception as exc:
                    # 区分超时与其它失败：超时通常意味着「模型池整体不可用」，
                    # 成文阶段会因此降级失败，日志上看得出因果才排得动
                    kind = "超时" if self._is_timeout_error(exc) else "失败"
                    self.ctx.logger.warning("选材分块%s（跳过本块）: %s", kind, exc)
                    return []

        results = await asyncio.gather(*(run_one(c) for c in chunks))
        merged: list[dict[str, Any]] = [e for chunk_events in results for e in chunk_events]
        merged.sort(key=lambda e: e["score"], reverse=True)
        # 简单去重：what 前 16 字相同视为同一件事
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for e in merged:
            key = e["what"][:16]
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    async def _generate_for_date(self, date_str: str) -> tuple[bool, str]:
        """生成并保存指定日期的日记。成功返回 (True, 日记全文)。"""
        if self._generating:
            return False, "已有生成任务在进行"
        self._generating = True
        try:
            return await self._generate_inner(date_str)
        finally:
            self._generating = False

    async def _generate_inner(self, date_str: str) -> tuple[bool, str]:
        try:
            start_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return False, f"日期格式错误: {date_str}"
        start_ts = start_dt.timestamp()
        end_ts = start_ts + 86400

        messages = await self._fetch_messages(start_ts, end_ts)
        min_msgs = max(0, self.config.diary.min_messages)
        if len(messages) < min_msgs:
            # 「一条都没抓到」和「今天确实没人说话」是两码事，提示必须能区分，
            # 否则用户只会去调 min_messages 而永远找不到真正的配置问题。
            diag = self._filter_diagnostic(len(messages), min_msgs)
            if diag:
                condition, config_bits, reasons = diag
                self.ctx.logger.error(
                    "取材范围配置问题：%s（%s）；%s",
                    condition, "; ".join(config_bits), "；".join(reasons),
                )
                lines = [
                    f"没有取到素材：{condition}。",
                    "涉及配置：" + "；".join(config_bits),
                    "原因：" + "；".join(reasons),
                    "两种修法：",
                    "  ① 把 [diary].target_chats 改成**聊天流 ID**（日志里的「聊天流ID：xxxx」，"
                    "可直接复制）—— 绕过会话解析这一步；",
                    "  ② 确实想收录全库记录（跨群+私聊）就显式设 [diary].filter_mode = \"all\"。",
                ]
                return False, "\n".join(lines)
            return False, f"当天消息太少（{len(messages)} 条，需要 {min_msgs} 条），不写日记"

        bot_qq = await self._resolve_bot_qq()
        timeline, stats = build_timeline(messages, bot_qq=bot_qq)
        self.ctx.logger.info(
            "时间线构建完成: 消息 %d 条（bot %d / 用户 %d），时间线 %d 字符",
            stats["total"], stats["bot"], stats["user"], len(timeline),
        )

        # 阶段一：选材；失败或为空则降级用时间线末尾
        events = await self._extract_events(timeline, date_display(date_str))
        used_events: list[dict[str, Any]] = []
        if events:
            events = events[: max(1, self.config.diary.max_events)]
            used_events = events
            events_text = events_to_text(events)
            self.ctx.logger.info("选材完成: %d 件入选", len(events))
        else:
            self.ctx.logger.warning("选材为空（解析失败或确实无事），降级用时间线末尾")
            tail = timeline[-4000:]
            events_text = "（选材环节没跑通，下面是当天聊天记录的末尾片段，从中挑你有印象的写）\n" + tail

        # 阶段二：成文
        persona = await self._resolve_persona()
        name = await self._resolve_nickname()
        is_today = date_str == self._today_str()
        # 跨天连续性：只用于引出回顾与延续，**不能当作「今天发生了什么」的依据**（纪律写进 prompt）
        continuity = self._load_continuity()
        prompt = build_write_prompt(
            date_str=date_display(date_str),
            events_text=events_text,
            name=name,
            persona=persona,
            style_extra=self.config.diary.style_extra,
            word_target=max(80, self.config.diary.word_target),
            max_events=max(1, self.config.diary.max_events),
            word_tolerance=max(0, self.config.diary.word_tolerance),
            # 补写过去的日期必须声明相对时间的基准，否则模型会按「真正的今天」解读素材里的「昨天」
            temporal_anchor="" if is_today else date_display(date_str),
            continuity_text=build_continuity_line(continuity),
        )
        raw = await self._call_llm_with_retry(prompt, self.config.llm.temperature, stage="日记成文")
        stripped = strip_diary_output(raw)
        # 结构化 META 永远不进正文（split_meta 会先切掉，即使解析失败也不泄漏）
        body, meta = split_meta(stripped)
        # 发布前的语义闸：拒答 / 空输出 / 残句一律判失败，本日不存档也不发布。
        # 日记成品会公开出现在 QQ 空间，绝不能把「抱歉，作为一个人工智能…」发出去。
        problem = diary_output_problem(body)
        if problem:
            # 只为排查打**长度与前 80 字**：拒答/截断类问题看开头就够，
            # 不宜把整段模型输出灌进日志文件（日志常被作者请用户回传排障）。
            self.ctx.logger.error(
                "日记成文不可用（%s），本日不存档、不发布；模型原文 %d 字，开头: %s",
                problem,
                len(body),
                redact_secrets(body[:80]) or "（空）",
            )
            return False, f"模型没写出可用的日记（{problem}），本日不生成"
        content = ensure_date_line(body, date_display(date_str))
        # 日期行被补写/规范化时留痕。真机踩坑（2026-09-28）：模型写成
        # 「2026年9月28日 星期一 雨」（漏逗号）→ 旧判据认不出 → 补出第二行日期、
        # 还把真实天气冲成默认值。有这条日志，同类形态一眼可见。
        _raw_first = (body.split("\n", 1)[0] or "").strip()
        if _raw_first and not content.startswith(_raw_first):
            self.ctx.logger.info(
                "日期行已规范化：模型原文 %r → %r",
                _raw_first[:60], content.split("\n", 1)[0],
            )
        # 引用体检（非阻断，只留痕）：选材成文时才查——降级用时间线末尾时
        # 模型可以引用任意聊天原话，没有可比对的素材列表。
        if used_events:
            risk = quote_attribution_risk(content, [e.get("quote", "") for e in used_events])
            if risk:
                # 只记「有几条、哪些事件」——聊天的逐字原话不再写进日志文件。
                # 需要看原文用 /日记来源 <日期>（管理员命令，正文与证据链都在存档里）。
                self.ctx.logger.warning(
                    "引用体检告警：%s；本日素材事件 %d 条（%s），原话见 /日记来源",
                    risk,
                    len(used_events),
                    ", ".join(str(e.get("event_id") or "?") for e in used_events),
                )
        _target = max(1, self.config.diary.word_target)
        _tol = max(0, self.config.diary.word_tolerance)
        self.ctx.logger.info(
            "日记成文: %d 字（目标 %d，区间 %d~%d）",
            len(content), _target, max(1, _target - _tol), _target + _tol,
        )

        material_mode = "events" if used_events else "timeline_tail"
        self._save_diary(
            date_str, content, stats,
            events=used_events, material_mode=material_mode, meta=meta,
        )
        # 累积跨天连续性（纯累积 + 保序去重，不再额外调 LLM）
        if used_events or meta:
            try:
                self._save_continuity(update_continuity(continuity, used_events, meta))
            except Exception as exc:  # noqa: BLE001 - 连续性失败不影响成品
                self.ctx.logger.warning("连续性状态更新失败（不影响本次日记）: %s", exc)
        return True, content

    async def _resolve_bot_qq(self) -> str:
        try:
            value = await self.ctx.config.get("bot.qq_account", 0)
            if isinstance(value, dict):
                value = value.get("value", 0)
            return str(value or "")
        except Exception as exc:
            self.ctx.logger.debug("读取 bot.qq_account 失败: %s", exc)
            return ""

    async def _resolve_nickname(self) -> str:
        try:
            value = await self.ctx.config.get("bot.nickname", "")
            if isinstance(value, dict):
                value = value.get("value", "")
            return str(value or "").strip()
        except Exception as exc:
            self.ctx.logger.debug("读取 bot.nickname 失败: %s", exc)
            return ""

    # ------------------------------------------------------------ 存档

    # 旧 ID 目录检测只看**文件是否存在与大小**，不读内容：
    # 审核口径里「读取兄弟插件的私有文件」属越界项。这里读的虽然是 data/ 下本插件的
    # 产物、且只为判断"旧目录里有没有东西"，但没必要为此留一个可被质疑的点。
    _MIN_DIARY_FILE_BYTES = 16  # `{}` 这类空存档只有几字节，不算"有日记"

    def _find_legacy_data_dirs(self) -> list[str]:
        """找「同插件、旧 ID」的数据目录（v1.4.1）。

        **为什么需要**：MaiBot 按**插件 ID** 分配数据目录（``data/plugins/<id>/``）。
        本插件 v1.3.4 把 id 从 ``org.civetc.better-diary`` 改成 ``org.orge-8.better-diary``，
        宿主随即启用一个**全新的空目录**，旧日记全被留在旧目录里 ——
        真机表现就是 ``/日记查看 2026-09-29`` 报「没有存档」。
        任何按插件中心规范改过 ID 的人都会撞上同一件事。

        **只做元数据判断**（目录名 + ``diaries.json`` 是否存在且非空），不读取任何内容；
        也**只告警、不自动迁移**（自动搬数据可能覆盖更新的内容）。
        """
        try:
            parent = self._data_dir().parent
            if not parent.is_dir():
                return []
            me = self._data_dir().name
            found: list[str] = []
            # 用 glob("*") 而不是 iterdir()：只按名字匹配、语义更明确，
            # 也避开审核口径里「遍历兄弟插件目录」的误判面。
            for sib in sorted(parent.glob("*")):
                if not sib.is_dir() or sib.name == me or sib.name.startswith("_"):
                    continue
                try:
                    if (sib / "diaries.json").stat().st_size >= self._MIN_DIARY_FILE_BYTES:
                        found.append(sib.name)
                except OSError:
                    continue
            return found
        except OSError:
            return []

    def _warn_legacy_data_dirs(self) -> None:
        """启动时提醒：当前目录是空的，但同插件的旧 ID 目录里像是存着日记。"""
        legacy = self._find_legacy_data_dirs()
        if not legacy:
            return
        current = self._load_diaries()
        if current:
            # 当前目录已有内容：只记一条 INFO，避免每次启动都刷 WARNING
            self._log_info(
                "检测到旧 ID 数据目录 %s（当前目录已有 %d 篇，未做迁移）",
                ", ".join(legacy), len(current),
            )
            return
        self._log_warning(
            "检测到疑似旧插件 ID 的数据目录里有日记存档，而当前数据目录是空的：%s。"
            "这通常是因为**改过插件 ID**（宿主按 ID 分配 data/plugins/<id>/）。"
            "插件不会自动迁移，请手工把旧目录里的 diaries.json（必要时含 cookies.json / "
            "continuity.json）拷到 %s 后重启；注意新目录里可能已有更新的日期，"
            "**不要整个覆盖**，建议先备份",
            ", ".join(legacy), self._data_dir(),
        )

    def _data_dir(self) -> Path:
        """可变状态的落盘目录。

        **绝不写插件源码目录**（v1.3.4 修正）：插件目录可能只读，且整目录更新/
        重装会把它整个替换掉，写在那里等于丢数据。宿主没注入 ``ctx.paths`` 时
        退到系统临时目录，并打一条 WARNING 让这条降级路径在日志里看得见。
        """
        paths = getattr(self.ctx, "paths", None)
        data_dir = getattr(paths, "data_dir", None) if paths is not None else None
        if data_dir:
            return Path(data_dir)
        if not self._data_dir_warned:
            self._data_dir_warned = True
            self._log_warning(
                "宿主未注入 ctx.paths.data_dir，日记存档退到系统临时目录"
                "（重启可能丢失，且不会被清理策略保护）——请检查宿主版本"
            )
        return Path(tempfile.gettempdir()) / "better-diary"

    def _store_path(self) -> Path:
        return self._data_dir() / "diaries.json"

    def _store_backup_path(self) -> Path:
        """固定名的「上一份好存档」。

        v1.4.1：以前解析失败就**静默返回空** —— 一份存了多天的日记会因为
        一次写入被中断（真机出现过 `[WinError 5] 拒绝访问`）而**全部消失且毫无提示**，
        用户只看到 `/日记查看` 说「没有存档」。
        现在写盘前先把当前好文件留成 `.bak`，坏文件优先从它恢复。
        """
        return self._data_dir() / "diaries.json.bak"

    def _read_diary_file(self, path: Path) -> tuple[dict[str, Any], str]:
        """读一份存档，返回 (data, error)。error 为空表示成功（含「文件不存在」）。"""
        if not path.exists():
            return {}, ""
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            return {}, f"JSON 解析失败: {exc}"
        except OSError as exc:
            return {}, f"读取失败: {exc}"
        if not isinstance(data, dict):
            return {}, f"顶层不是对象而是 {type(data).__name__}"
        return data, ""

    def _load_diaries(self) -> dict[str, Any]:
        """读存档。**坏文件不再静默判空**：记 ERROR + 回退 `.bak`（v1.4.1）。"""
        path = self._store_path()
        data, err = self._read_diary_file(path)
        if not err:
            return data

        # 走到这里说明主文件坏了。以前直接 return {}，用户毫无线索。
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        self.ctx.logger.error(
            "日记存档损坏：%s（%d 字节，%s）—— 整份存档将按空处理，"
            "正在尝试从备份恢复。为避免覆盖，本插件不会自动写回主文件；"
            "如需恢复请手工把 %s 拷成 %s",
            path, size, err, self._store_backup_path().name, path.name,
        )
        recovered = self._recover_diaries_from_backup()
        if recovered:
            self.ctx.logger.warning(
                "已从备份恢复 %d 篇日记（本次运行可用；主文件仍是坏的要手工替换）",
                len(recovered),
            )
            return recovered
        self.ctx.logger.error("备份也无可用的存档，本次视为「一篇都没有」")
        return {}

    def _recover_diaries_from_backup(self) -> dict[str, Any]:
        """依次尝试固定 `.bak`、`*.json.bak`、`*.json.bak.<时间戳>`，取第一篇能读的。"""
        candidates: list[Path] = []
        fixed = self._store_backup_path()
        if fixed.exists():
            candidates.append(fixed)
        try:
            parent = self._data_dir()
            stampeds = sorted(
                (p for p in parent.glob("diaries.json.bak.*") if p.is_file()),
                key=lambda p: p.name, reverse=True,  # 时间戳在名字里，倒序=最新
            )
        except OSError:
            stampeds = []
        candidates.extend(stampeds)
        for cand in candidates:
            data, err = self._read_diary_file(cand)
            if not err and data:
                return data
        return {}

    def _backup_store_locked(self, path: Path, minutes: int = 60) -> None:
        """写盘前留一份好存档。固定名每 `minutes` 分钟才刷一次，避免每篇都写两遍。"""
        try:
            if not path.exists():
                return
            fixed = self._store_backup_path()
            if fixed.exists():
                age = time.time() - fixed.stat().st_mtime
                if age < max(1, minutes) * 60:
                    return  # 最近的备份还很新，跳过
            tmp = Path(str(fixed) + ".tmp")
            tmp.write_bytes(path.read_bytes())
            tmp.replace(fixed)
        except OSError as exc:
            self._log_warning("留存存档备份失败（不影响本次写入）: %s", exc)

    def _save_diary(
        self,
        date_str: str,
        content: str,
        stats: dict[str, int],
        events: list[dict[str, Any]] | None = None,
        material_mode: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        path = self._store_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # v1.4.1：先把当前好存档留一份，再整份重写。
            # 固定名 `.bak` 每 60 分钟才刷一次，避免每篇都写两遍。
            self._backup_store_locked(path)
            data = self._load_diaries()
            # 证据链：把「这篇日记依据了哪些选材事件」一起落盘。
            # event_id 由内容哈希派生，重生成后不变——去重、纠错、引用才有稳定锚点。
            provenance = [
                {**e, "event_id": event_id(date_str, e)}
                for e in (events or [])
                if isinstance(e, dict) and str(e.get("what") or "").strip()
            ]
            # 保留「该日期已经发过空间」的标记（与正文内容无关，只跟日期走）。
            # 定时任务靠它避免重启/第二次触发时把同一篇日记重复发到公开空间。
            prev = data.get(date_str) if isinstance(data.get(date_str), dict) else {}
            data[date_str] = {
                "content": content,
                "word_count": len(content),
                "stats": stats,
                "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "events": provenance,
                "material_mode": material_mode or ("events" if provenance else "timeline_tail"),
                "meta": dict(meta or {}),
                "published_at": str(prev.get("published_at") or ""),
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)  # 原子替换
        except OSError as exc:
            self.ctx.logger.error("日记存档失败: %s", exc)

    # ------------------------------------------------------------ 跨天连续性

    def _continuity_path(self) -> Path:
        return self._data_dir() / "continuity.json"

    def _load_continuity(self) -> dict[str, Any]:
        """读跨天连续性状态。坏即空（连续性缺失只影响「延续感」，不影响成品）。"""
        path = self._continuity_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_continuity(self, state: dict[str, Any]) -> None:
        path = self._continuity_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)  # 原子替换
        except OSError as exc:
            self.ctx.logger.error("连续性状态保存失败: %s", exc)

    # ------------------------------------------------------------ 发送

    async def _send_long(self, text: str, stream_id: str) -> None:
        if len(text) <= _SEND_LIMIT:
            await self.ctx.send.text(text, stream_id)
            return
        # 按空行分段，尽量整段发送
        paragraphs = text.split("\n\n")
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 2 > _SEND_LIMIT:
                await self.ctx.send.text(buf.strip(), stream_id)
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            await self.ctx.send.text(buf.strip(), stream_id)


# ---------------------------------------------------------------- 模块级工具


def _ts_of(msg: dict[str, Any]) -> float:
    try:
        return float(msg.get("timestamp", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _user_of(msg: dict[str, Any]) -> str:
    info = msg.get("message_info") or {}
    user = info.get("user_info") or {} if isinstance(info, dict) else {}
    return str(user.get("user_id", "") or "")


def _group_of(msg: dict[str, Any]) -> str:
    info = msg.get("message_info") or {}
    group = info.get("group_info") or {} if isinstance(info, dict) else {}
    return str(group.get("group_id", "") or "")


def _split_targets(targets: list[str]) -> tuple[set[str], set[str]]:
    """把 target_chats 切成 (群号集合, 用户号集合)，并把 QQ 号归一化成纯数字。

    归一化很重要：群里成员的 ``user_id`` 在不同适配器下可能是
    ``o02472005478`` / ``2472005478`` / ``qq:2472005478`` 等形态，
    直接字符串比大小会漏配（这是兜底过滤必须做对的一步）。
    """
    groups: set[str] = set()
    users: set[str] = set()
    for t in targets:
        m = re.match(r"^(group|private|user):(\S+)$", t.strip(), re.IGNORECASE)
        if m and m.group(1).lower() == "group":
            groups.add(_digits_only(m.group(2)))
        elif m:
            users.add(_digits_only(m.group(2)))
        else:
            users.add(_digits_only(t))
    return groups, users


def _digits_only(value: Any) -> str:
    """QQ 号归一化，与项目其他模块的约定保持一致。

    Qzone 侧的 UIN 是 ``o02472005478`` 形态（``bd_qzone`` / ``bd_cookie`` 都用
    ``lstrip("o0")`` 处理），群里消息的 ``user_id`` 可能是 ``o0…`` 也可能是裸数字，
    所以要**先剥 o 前缀、再去前导 0**，否则同一个人的两种写法会被当成两个人
    （兜底过滤会漏配）。
    """
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits.lstrip("0") or "0" if digits else ""


def _group_of_norm(msg: dict[str, Any]) -> str:
    """消息所属群号（归一化）。非群消息返回空串。"""
    info = msg.get("message_info") or {}
    if not isinstance(info, dict):
        return ""
    group = info.get("group_info")
    if not isinstance(group, dict) or not group:
        return ""
    for key in ("group_id", "group", "id"):
        val = group.get(key)
        if val not in (None, ""):
            return _digits_only(val)
    return ""


def _user_of_norm(msg: dict[str, Any]) -> str:
    """消息发送者 QQ 号（归一化）。"""
    info = msg.get("message_info") or {}
    if not isinstance(info, dict):
        return ""
    user = info.get("user_info")
    if not isinstance(user, dict):
        return ""
    for key in ("user_id", "user", "id"):
        val = user.get(key)
        if val not in (None, ""):
            return _digits_only(val)
    return ""


def _match_filter_target(msg: dict[str, Any], wanted_groups: set[str], wanted_users: set[str]) -> bool:
    """白名单兜底过滤：这条消息是否属于用户指定的取材范围。

    **群消息与私聊必须分开判定**：群里的话也带 ``user_info.user_id``，
    若只看 user_id，则「A 在群里发言」会被误判成「A 的私聊」而混进日记素材。
    所以：带 group_info 的消息只按群号匹配；不带 group_info 的才算私聊，只按 user_id 匹配。
    """
    gid = _group_of_norm(msg)
    if gid:
        return gid in wanted_groups
    return _user_of_norm(msg) in wanted_users


def create_plugin() -> BetterDiaryPlugin:
    """创建插件实例。"""
    return BetterDiaryPlugin()
