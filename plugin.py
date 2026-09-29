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
    filter_mode: str = Field(default="all", description="聊天过滤：all / whitelist / blacklist")
    target_chats: list[str] = Field(default_factory=list, description='过滤目标列表，格式 "group:群号" 或 "private:QQ号"')

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
        try:
            while True:
                delay, kind = self._next_fire()
                if delay == float("inf"):
                    # 两个时间点都没配（time 有默认值，理论上到不了这里）：
                    # 睡一小时再看，不要空转烧 CPU
                    await asyncio.sleep(3600)
                    continue
                self.ctx.logger.info("定时日记将在 %.0f 秒后运行（%s）", delay, kind)
                await asyncio.sleep(delay)
                if kind == "fallback":
                    await self._catch_up_run()
                else:
                    await self._run_main_with_silence_wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.ctx.logger.error("调度器异常退出: %s", exc, exc_info=True)

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
        """'group:123' / 'private:456' / 裸 stream_id -> 聊天流 ID。"""
        target = str(target).strip()
        m = re.match(r"^(group|private|user):(\S+)$", target, re.IGNORECASE)
        if not m:
            return target  # 已是 stream_id
        kind, ident = m.group(1).lower(), m.group(2)
        try:
            if kind == "group":
                result = await self.ctx.chat.get_stream_by_group_id(ident)
            else:
                result = await self.ctx.chat.get_stream_by_user_id(ident)
        except Exception as exc:
            self.ctx.logger.warning("解析 %s 失败: %s", target, exc)
            return ""
        return self._extract_stream_id(result)

    @staticmethod
    def _extract_stream_id(result: Any) -> str:
        """从各种可能的返回形态里挖 stream_id。"""
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list) and result:
            return BetterDiaryPlugin._extract_stream_id(result[0])
        if isinstance(result, dict):
            for key in ("stream_id", "session_id", "chat_id", "id"):
                val = result.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
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
        """按过滤模式抓取当天消息（跨聊天合并，按时间排序）。"""
        mode = (self.config.diary.filter_mode or "all").lower()
        targets = [str(t) for t in (self.config.diary.target_chats or []) if str(t).strip()]

        if mode == "whitelist":
            if not targets:
                return []
            all_msgs: list[dict[str, Any]] = []
            for target in targets:
                stream_id = await self._resolve_stream_id(target)
                if not stream_id:
                    self.ctx.logger.warning("白名单目标 %s 解析失败，跳过", target)
                    continue
                msgs = await self._query_messages(start_ts, end_ts, stream_id)
                all_msgs.extend(msgs)
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
        """成文阶段专用：**只对超时类失败**做有限重试。

        真机实录（2026-09-28 09:00 补跑）：模型 Provider 集体网络超时（30s APITimeoutError，
        日志里连着好几条 `遇到错误: 网络连接超时`），MaiBot 侧依次切换模型、逐个耗尽重试，
        最终以 Runner RPC 超时收尾 —— 补跑直接在这一天炸掉。

        为什么值得重试：Provider 超时是**网络抖动**性质，隔一会儿换一个模型往往就好了；
        而格式/语义类失败重试纯属浪费。**非超时异常一律原样抛出，不重试。**
        """
        attempts = max(0, int(self.config.llm.write_retry)) + 1
        backoff = max(0, int(self.config.llm.retry_backoff_seconds))
        for attempt in range(1, attempts + 1):
            try:
                return await self._call_llm(prompt, temperature)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt >= attempts or not self._is_timeout_error(exc):
                    raise
                self._log_warning(
                    "%s 第 %d/%d 次调用超时，%d 秒后重试: %s",
                    stage, attempt, attempts, backoff, exc,
                )
                if backoff:
                    await asyncio.sleep(backoff)
        raise RuntimeError(f"{stage}调用失败")  # pragma: no cover - 循环必然 return 或 raise

    async def _extract_events(self, timeline: str, date_str: str) -> list[dict[str, Any]]:
        """阶段一：分块选材 + 打分 + 合并排序。"""
        chunks = chunk_text(timeline, self.config.diary.chunk_chars, self.config.diary.max_chunks)
        if not chunks:
            return []
        sem = asyncio.Semaphore(3)

        async def run_one(chunk: str) -> list[dict[str, Any]]:
            async with sem:
                try:
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

    def _load_diaries(self) -> dict[str, Any]:
        path = self._store_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}  # 坏即空

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
    groups: set[str] = set()
    users: set[str] = set()
    for t in targets:
        m = re.match(r"^(group|private|user):(\S+)$", t.strip(), re.IGNORECASE)
        if m and m.group(1).lower() == "group":
            groups.add(m.group(2))
        elif m:
            users.add(m.group(2))
        else:
            users.add(t.strip())
    return groups, users


def create_plugin() -> BetterDiaryPlugin:
    """创建插件实例。"""
    return BetterDiaryPlugin()
