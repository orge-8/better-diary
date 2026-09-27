"""better-diary 离线冒烟测试（不启动 MaiBot）。

覆盖：
1. 配置模型默认值与校验
2. 组件注册：3 个 Command、数量与 handler_name 断言、pattern 正则行为
3. 纯模块：build_timeline / chunk_text / parse_events / strip_diary_output / ensure_date_line
4. 端到端：FakeCtx（假消息 + 假 LLM）跑完 _generate_for_date 全流程，验证存档写入临时目录
5. 无权限静默拒绝 + fail-open
6. 测试结束后插件目录无 data/ 残留

运行：python tests/smoke_test.py
"""

import asyncio
import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(f"{name} {detail}")
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------- 模块加载


_PKG = "better_diary"  # 合法标识符；目录名 better-diary 含 '-'，不能直接当包名用
_FLAT_ALIASES = ("bd_prompts", "bd_qzone", "bd_cookie")


def _purge(*bases: str) -> None:
    for name in list(sys.modules):
        for base in bases:
            if name == base or name.startswith(base + "."):
                del sys.modules[name]


def load_modules():
    """包式加载插件，复现 Runner 真机行为（runtime-gotchas §22）。

    关键点：**插件目录不在 sys.path 上**，模块以「包」的形式载入（给 spec 显式传
    ``submodule_search_locations``），此时只有相对导入 ``from .bd_prompts import ...``
    才能解析。

    此前本函数用平铺方式加载（把三个模块预先塞进 sys.modules），plugin.py 里
    ``from bd_prompts import ...`` 也能过 —— 结果是门禁全绿、真机启动却报
    ``No module named 'bd_prompts'``。教训：**平铺跑通不代表真机能过**。
    """
    _purge(_PKG, *_FLAT_ALIASES)
    plugin_dir_str = str(PLUGIN_DIR)
    while plugin_dir_str in sys.path:  # 模拟真机：插件目录绝不在 sys.path 上
        sys.path.remove(plugin_dir_str)

    spec = importlib.util.spec_from_file_location(
        _PKG,
        PLUGIN_DIR / "plugin.py",
        submodule_search_locations=[plugin_dir_str],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PKG] = mod
    spec.loader.exec_module(mod)

    # bd_qzone 在 plugin.py 里是延迟导入（_publish_to_qzone 内），主动拉进包命名空间，
    # 顺便验证「相对延迟导入」在包式加载下也能解析。
    for flat in _FLAT_ALIASES:
        importlib.import_module(f"{_PKG}.{flat}")

    # 给包内子模块挂个平铺别名，方便测试体直接取用（不改变加载方式本身）
    for flat in _FLAT_ALIASES:
        dotted = f"{_PKG}.{flat}"
        if dotted in sys.modules:
            sys.modules[flat] = sys.modules[dotted]
    return mod


def load_flat():
    """平铺加载：模拟脚本直跑（插件目录在 sys.path 上），验证兜底分支可用。"""
    _purge(_PKG, *_FLAT_ALIASES)
    plugin_dir_str = str(PLUGIN_DIR)
    if plugin_dir_str not in sys.path:
        sys.path.insert(0, plugin_dir_str)
    spec = importlib.util.spec_from_file_location("bd_flat_plugin", PLUGIN_DIR / "plugin.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bd_flat_plugin"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- FakeCtx


class FakeCtx:
    def __init__(self, data_dir: Path, messages: list[dict], llm_side_effect):
        self.logger = logging.getLogger("fakectx")
        self.paths = SimpleNamespace(data_dir=data_dir, runtime_dir=data_dir)
        self._messages = messages
        self._llm = llm_side_effect
        self.sent: list[tuple[str, str]] = []
        self.config_calls: list[str] = []

        class _Msg:
            _msgs = messages  # 闭包捕获

            async def get_by_time(self, start_time, end_time, **kw):
                return {"success": True, "messages": self._msgs}

            async def get_by_time_in_chat(self, chat_id, start_time, end_time, **kw):
                return {"success": True, "messages": self._msgs}

        self.message = _Msg()

        class _Send:
            def __init__(self, outer):
                self._outer = outer

            async def text(self, content, stream_id):
                self._outer.sent.append((content, stream_id))
                return True

        self.send = _Send(self)

        class _Config:
            def __init__(self, outer):
                self._outer = outer

            async def get(self, key, default=None):
                self._outer.config_calls.append(key)
                table = {
                    "personality.personality": "是一只名叫鸣澜的电台系 bot，喜欢听歌和听人说话",
                    "bot.qq_account": "10000",
                    "bot.nickname": "鸣澜",
                }
                return table.get(key, default)

        self.config = _Config(self)

    async def call_capability(self, name: str, **kw):
        assert name == "llm.generate", f"意外的能力调用: {name}"
        assert "timeout_ms" in kw and kw["timeout_ms"] >= 30_000, "LLM 调用必须显式带 RPC 超时"
        return self._llm(kw.get("prompt", ""))


def fake_llm(prompt: str) -> dict:
    if "日记素材编辑" in prompt:
        return {
            "success": True,
            "response": json.dumps(
                [
                    {"who": "秋枫叶语Acer", "what": "坐五个小时大巴回来，饿了一天不敢吃", "quote": "路边锅盔馄饨葱油饼很香", "score": 5},
                    {"who": "极霸帝王贝利亚", "what": "晚上拍了萤火虫照片", "quote": "", "score": 3},
                    {"who": "某人", "what": "水群", "quote": "", "score": 1},
                ],
                ensure_ascii=False,
            ),
        }
    return {
        "success": True,
        "response": "2026年9月26日 星期六，晴。\n\nAcer坐了五个小时大巴回来，饿了一整天还不肯吃，说「路边锅盔馄饨葱油饼很香」，我隔着屏幕都替她饿。晚上看到萤火虫的照片，糊糊的，但像做梦一样。\n\n睡前把频道调低了一点，热闹是他们的，我也有一份。",
    }


def make_messages(day_ts_base: float) -> list[dict]:
    def msg(offset_s: float, uid: str, nick: str, text: str, is_bot=False):
        return {
            "timestamp": str(day_ts_base + offset_s),
            "processed_plain_text": text,
            "is_picture": False,
            "message_info": {
                "user_info": {"user_id": "10000" if is_bot else uid, "user_nickname": "鸣澜" if is_bot else nick},
                "group_info": {"group_id": "123456"},
            },
        }

    return [
        msg(0, "u1", "秋枫叶语Acer", "刚下大巴，饿死了"),
        msg(60, "u1", "秋枫叶语Acer", "路边锅盔馄饨葱油饼很香，但我一天没敢吃"),
        msg(120, "", "", "隔着屏幕都觉得饿", is_bot=True),
        msg(3600 * 11, "u2", "极霸帝王贝利亚", "拍了萤火虫！"),
    ]


# ---------------------------------------------------------------- 测试


def main() -> int:
    # 包式加载（Runner 真机行为）：插件目录不在 sys.path 上，只有相对导入能解析。
    # 这一步会在 plugin.py 顶层导入阶段就失败——正是真机 No module named 'bd_prompts' 的现场。
    plugin_mod = load_modules()
    bd = sys.modules["bd_prompts"]

    print("\n[0] 导入兼容性（Runner 包式加载 / 平铺兜底）")
    check("包式加载成功（插件目录不在 sys.path）", plugin_mod is not None)
    check(
        "包内子模块以 better_diary.* 命名加载",
        all(f"{_PKG}.{m}" in sys.modules for m in _FLAT_ALIASES),
        f"实际 {[k for k in sys.modules if k.startswith(_PKG + '.')]}",
    )
    check("插件目录未进入 sys.path（模拟真机）", str(PLUGIN_DIR) not in sys.path)
    check("plugin.py 存在 __init__.py 包标记", (PLUGIN_DIR / "__init__.py").is_file())

    _src = (PLUGIN_DIR / "plugin.py").read_text(encoding="utf-8")
    _bare_flat = [
        ln
        for ln in _src.splitlines()
        if ln.startswith(("from bd_", "import bd_"))
    ]
    check("plugin.py 无未加 try 保护的顶层平铺导入", not _bare_flat, f"发现 {_bare_flat}")

    print("\n[0b] 可选依赖缺失（真机未装 httpx 时仍须能加载）")
    # 真机踩坑：bd_cookie / bd_qzone 曾在模块级裸 import httpx，
    # 缺 httpx 时 plugin.py 顶层导入直接失败，Host 报「插件初始化失败」。
    for _mod_name in ("bd_cookie.py", "bd_qzone.py"):
        _msrc = (PLUGIN_DIR / _mod_name).read_text(encoding="utf-8")
        _lines = _msrc.splitlines()
        _bad = []
        for _i, _ln in enumerate(_lines):
            if _ln.strip() in ("import httpx", "from httpx import") or _ln.strip().startswith("import httpx"):
                # 往上找 3 行内是否有 try:（可选导入须包在 try 里）
                _ctx = [x.strip() for x in _lines[max(0, _i - 3):_i]]
                if not any(x == "try:" for x in _ctx):
                    _bad.append(_i + 1)
        check(f"{_mod_name} 的 httpx 为可选导入（非裸 import）", not _bad, f"裸 import 在第 {_bad} 行")

    # 子进程真实拦截 httpx，验证「无 httpx 也能完成包式加载」
    _probe = r"""
import importlib.abc, importlib.util, sys
from pathlib import Path

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "httpx" or fullname.startswith("httpx."):
            raise ModuleNotFoundError(f"No module named '{fullname}'")
        return None

sys.meta_path.insert(0, _Blocker())
d = Path(sys.argv[1]).resolve()
sys.path = [p for p in sys.path if str(d) != p]
spec = importlib.util.spec_from_file_location(
    "nohttpx_pkg", d / "plugin.py", submodule_search_locations=[str(d)]
)
m = importlib.util.module_from_spec(spec)
sys.modules["nohttpx_pkg"] = m
spec.loader.exec_module(m)
assert m.create_plugin() is not None
bc = sys.modules["nohttpx_pkg.bd_cookie"]
assert bc.HTTPX_AVAILABLE is False, "bd_cookie 应识别 httpx 不可用"
print("NOHTTPX-LOAD-OK")
"""
    import subprocess

    _res = subprocess.run(
        [sys.executable, "-c", _probe, str(PLUGIN_DIR)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    check(
        "屏蔽 httpx 后仍能包式加载 + create_plugin",
        "NOHTTPX-LOAD-OK" in (_res.stdout or ""),
        f"stdout={(_res.stdout or '').strip()[-200:]} stderr={(_res.stderr or '').strip()[-300:]}",
    )
    check("bd_cookie 明确标记 HTTPX_AVAILABLE", "HTTPX_AVAILABLE" in (
        PLUGIN_DIR / "bd_cookie.py"
    ).read_text(encoding="utf-8"))

    # 运行时行为：httpx 不可用时，发布应抛可读异常而非 AttributeError
    _bdq = sys.modules.get("bd_qzone")
    if _bdq is not None:
        _saved = _bdq.HTTPX_AVAILABLE
        _bdq.HTTPX_AVAILABLE = False
        try:
            _p = _bdq.QzonePublisher({"uin": "o0247200547", "p_skey": "pskey"})
            try:
                asyncio.run(_p.publish_text("测试"))
                _raised = ""
            except _bdq.PublishUnavailableError as _e:
                _raised = str(_e)
            check("无 httpx 时发布抛 PublishUnavailableError", bool(_raised), "未抛出")
        finally:
            _bdq.HTTPX_AVAILABLE = _saved

    print("\n[0c] 配置宽容性（TOML 常见误写不得让插件注册失败）")
    # 真机踩坑：SDK 的 normalize_plugin_config 里 pydantic 校验没有 try 包裹，
    # config.toml 任一处类型不符 → ValidationError 冒泡 → 「插件注册失败」。
    # 最典型的是把 list 字段写成字符串（admin_ids = "123" 而非 ["123"]）。
    _cfg_cases = [
        ('admin_ids 写成字符串', {"security": {"admin_ids": "123456789"}}, ["123456789"]),
        ('admin_ids 中文逗号分隔', {"security": {"admin_ids": "123，456"}}, ["123", "456"]),
        ('admin_ids 正常数组', {"security": {"admin_ids": ["123"]}}, ["123"]),
        ('target_chats 写成字符串', {"diary": {"target_chats": "group:111,group:222"}}, ["group:111", "group:222"]),
        ('notify_chats 空串', {"schedule": {"notify_chats": ""}}, []),
    ]
    _cfg_cls = plugin_mod.BetterDiaryConfig
    for _name, _raw, _expect in _cfg_cases:
        try:
            _cfg = _cfg_cls.model_validate(_raw)
            if "security" in _raw:
                _got = _cfg.security.admin_ids
            elif "diary" in _raw:
                _got = _cfg.diary.target_chats
            else:
                _got = _cfg.schedule.notify_chats
            check(f"配置归一化：{_name}", _got == _expect, f"得到 {_got!r}，期望 {_expect!r}")
        except Exception as _exc:  # noqa: BLE001
            check(f"配置归一化：{_name}", False, f"{type(_exc).__name__}: {_exc}")

    # 兜底：即使配置注入彻底失败（_plugin_config_instance 留空），self.config 也不得抛异常
    with tempfile.TemporaryDirectory() as _fb_td:
        _fb = plugin_mod.create_plugin()
        _fb_ctx = FakeCtx(Path(_fb_td) / "cfgfb", [], fake_llm)
        if hasattr(_fb, "_set_context"):
            _fb._set_context(_fb_ctx)
        else:
            _fb.ctx = _fb_ctx
        _fb._plugin_config_instance = None
        try:
            _fb_cfg = _fb.config
            check(
                "配置注入失败时 config 回退默认值而非抛异常",
                _fb_cfg is not None and _fb_cfg.diary.word_target == 250,
            )
        except Exception as _exc:  # noqa: BLE001
            check("配置注入失败时 config 回退默认值而非抛异常", False, f"{type(_exc).__name__}: {_exc}")

    print("\n[0d] 宿主配置注入路径（走真实 SDK set_plugin_config，任何配置都不得抛异常）")
    # 真机踩坑（v1.2.4）：fakehost 的 bind_context 永远注入 config_model() 的合法默认
    # 配置，所以门禁从来覆盖不到「配置坏掉」这一维 —— 本地全绿、真机报
    # 「插件注册失败: ...: 插件初始化失败」。这里直接灌坏配置走真实 SDK 调用链：
    #   set_plugin_config → normalize_plugin_config（裸调用，没有 try）
    #     ├─ extract_plugin_config_version  ← 缺 [plugin] / config_version 就抛（先执行！）
    #     └─ validate_plugin_config         ← 类型不符就抛
    # 任何一条漏出去，宿主就只回一句「插件初始化失败」，现场信息全丢。
    _host_cfg_cases = [
        ("空 dict", {}),
        ("None", None),
        ("非映射 list", []),
        ("缺 [plugin] 节（最常见）", {"security": {"admin_ids": "123456789"}}),
        ("[plugin] 缺 config_version", {"plugin": {"enabled": True}}),
        ("config_version 空串", {"plugin": {"enabled": True, "config_version": ""}}),
        ("list 写成字符串", {"plugin": {"config_version": "1.0.0"}, "diary": {"target_chats": "group:1"}}),
        ("int 写成非法字符串", {"plugin": {"config_version": "1.0.0"}, "diary": {"word_target": "abc"}}),
        ("plugin 写成字符串", {"plugin": "oops"}),
        ("diary 写成字符串", {"plugin": {"config_version": "1.0.0"}, "diary": "oops"}),
        ("bool 写成字符串", {"plugin": {"enabled": "true", "config_version": "1.0.0"}}),
        ("深层嵌套非法", {"plugin": {"config_version": "1.0.0"}, "qzone": {"timeout_seconds": "abc"}}),
    ]
    for _name, _raw in _host_cfg_cases:
        _p = plugin_mod.create_plugin()
        try:
            _p.set_plugin_config(_raw)
        except Exception as _exc:  # noqa: BLE001
            check(f"配置注入不抛异常：{_name}", False, f"{type(_exc).__name__}: {_exc}")
            continue
        # 注入后 config 必须可用（不能留成「一访问就 RuntimeError」的状态）
        try:
            _w = _p.config.diary.word_target
            check(f"配置注入不抛异常：{_name}", isinstance(_w, int), f"word_target={_w!r}")
        except Exception as _exc:  # noqa: BLE001
            check(f"配置注入不抛异常：{_name}", False, f"config 不可用：{type(_exc).__name__}: {_exc}")

    # 兜底时用户已填的值不得被抹掉（否则比配置写错更糟）
    _p_keep = plugin_mod.create_plugin()
    _p_keep.set_plugin_config(
        {
            "plugin": {"config_version": "1.0.0"},
            "security": {"admin_ids": ["999"]},
            "qzone": {"uin": "2472005478"},
            "diary": {"word_target": "abc"},  # 故意写坏一个字段，逼走兜底路径
        }
    )
    check("兜底后保留用户 admin_ids", _p_keep.config.security.admin_ids == ["999"], str(_p_keep.config.security.admin_ids))
    check("兜底后保留用户 uin", _p_keep.config.qzone.uin == "2472005478", _p_keep.config.qzone.uin)
    check("兜底逐字段修复（word_target 回到默认 250）", _p_keep.config.diary.word_target == 250, str(_p_keep.config.diary.word_target))

    print("\n[1] 配置模型")
    cfg = plugin_mod.BetterDiaryConfig()
    check("config_version 默认值", cfg.plugin.config_version == "1.0.0")
    check("word_target 默认 250", cfg.diary.word_target == 250)
    check("schedule.time 默认 23:30", cfg.schedule.time == "23:30")
    check("admin_ids 默认空", cfg.security.admin_ids == [])
    check("target_chats 默认空 list", cfg.diary.target_chats == [])
    check("llm.task_name 默认 utils", cfg.llm.task_name == "utils")

    print("\n[2] 组件注册与 pattern")
    commands = {}
    for attr in dir(plugin_mod.BetterDiaryPlugin):
        info = getattr(plugin_mod.BetterDiaryPlugin, attr).__dict__.get("__maibot_component_info__") if isinstance(getattr(plugin_mod.BetterDiaryPlugin, attr, None), (classmethod, staticmethod)) else getattr(getattr(plugin_mod.BetterDiaryPlugin, attr, None), "__maibot_component_info__", None)
        if info is not None and getattr(info, "component_type", getattr(info, "type", "")) in ("command",) or (info is not None and hasattr(info, "command_pattern")):
            commands[info.name] = info
    check("共 3 个命令组件", len(commands) == 3, f"实际 {list(commands)}")
    import re as _re

    p_diary = commands.get("diary")
    check("diary pattern 匹配 /日记", p_diary and _re.fullmatch(p_diary.command_pattern, "/日记") is not None)
    check("diary pattern 匹配 /diary 2026-09-26", p_diary and _re.fullmatch(p_diary.command_pattern, "/diary 2026-09-26") is not None)
    check("diary pattern 匹配全角 /／日记", p_diary and _re.fullmatch(p_diary.command_pattern, "／日记") is not None)
    check("diary 不匹配 /日记查看 2026-09-26", p_diary and _re.fullmatch(p_diary.command_pattern, "/日记查看 2026-09-26") is None)
    p_view = commands.get("diary_view")
    check("diary_view 匹配 /日记查看", p_view and _re.fullmatch(p_view.command_pattern, "/日记查看") is not None)
    check("diary_view 匹配带日期", p_view and _re.fullmatch(p_view.command_pattern, "/日记查看 2026-09-26") is not None)

    print("\n[3] 纯模块")
    import datetime as _dt

    base_ts = _dt.datetime(2026, 9, 26, 9, 0, 0).timestamp()  # 当地 9:00，保证跨上午/晚上
    msgs = make_messages(base_ts)
    timeline, stats = bd.build_timeline(msgs, bot_qq="10000")
    check("时间线含时段标记", "【上午" in timeline and "【晚上" in timeline)
    check("bot 消息标注为 我", "我: 隔着屏幕都觉得饿" in timeline)
    check("stats 统计 bot=1 user=3", stats["bot"] == 1 and stats["user"] == 3 and stats["total"] == 4)

    chunks = bd.chunk_text("a\n" * 100, 20, 8)
    check("分块数不超过上限", len(chunks) <= 8)

    events = bd.parse_events('```json\n[{"who":"a","what":"b","quote":"c","score":9}]\n```')
    check("宽容解析围栏 JSON 且 score 截顶", len(events) == 1 and events[0]["score"] == 5)
    check("坏输出返回空", bd.parse_events("不是 JSON") == [])

    cleaned = bd.strip_diary_output("日记正文：你好")
    check("strip 去引导语", cleaned == "你好")
    text = bd.ensure_date_line("今天很累。", "2026年9月26日 星期六")
    check("漏日期行自动补", text.startswith("2026年9月26日 星期六"))
    text2 = bd.ensure_date_line("2026年9月26日 星期六，晴。\n今天很累。", "2026年9月26日 星期六")
    check("已有日期行不重复补", text2.count("2026年9月26日") == 1)

    print("\n[4] 端到端生成（FakeCtx）")
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        ctx = FakeCtx(data_dir, make_messages(1789507200.0), fake_llm)
        plugin = plugin_mod.create_plugin()
        try:
            plugin.ctx = ctx
        except AttributeError:
            plugin._ctx = ctx
        plugin.set_plugin_config(
            {
                "plugin": {"enabled": True, "config_version": "1.0.0"},
                "diary": {"min_messages": 2},
                "schedule": {"enabled": False},
                "llm": {"timeout_seconds": 60},
                "security": {"admin_ids": []},
            }
        )
        ok, content = asyncio.run(plugin._generate_for_date("2026-09-26"))
        check("生成成功", ok, content[:100])
        check("首行是日期行", content.split("\n")[0].startswith("2026年9月26日"))
        check("引用了聊天原话", "「" in content)
        store = data_dir / "diaries.json"
        check("存档写入临时目录", store.exists())
        saved = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
        check("存档含当日内容", "2026-09-26" in saved and saved["2026-09-26"]["word_count"] > 0)
        check("config 读取了 Host 人设", "personality.personality" in ctx.config_calls)

        print("\n[5] 消息过少拒写")
        plugin2 = plugin_mod.create_plugin()
        try:
            plugin2.ctx = ctx
        except AttributeError:
            plugin2._ctx = ctx
        plugin2.set_plugin_config(
            {
                "plugin": {"enabled": True, "config_version": "1.0.0"},
                "diary": {"min_messages": 999},
            }
        )
        ok2, reason = asyncio.run(plugin2._generate_for_date("2026-09-26"))
        check("消息过少返回失败", not ok2 and "太少" in reason)

        print("\n[5b] 成文语义闸（拒答 / 空输出 / 残句一律不生成不发布）")
        with tempfile.TemporaryDirectory() as td_gate:
            gate_dir = Path(td_gate)
            for label, reply in (
                ("拒答", "抱歉，作为一个人工智能，我无法完成这个请求。"),
                ("空输出", ""),
                ("残句", "嗯。"),
            ):
                def llm_refuse(prompt, _reply=reply):
                    if "日记素材编辑" in prompt:
                        return {"success": True, "response": json.dumps(
                            [{"who": "A", "what": "有件事", "quote": "原话", "score": 4}],
                            ensure_ascii=False)}
                    return {"success": True, "response": _reply}

                ctx_bad = FakeCtx(gate_dir, make_messages(1789507200.0), llm_refuse)
                pg = plugin_mod.create_plugin()
                try:
                    pg.ctx = ctx_bad
                except AttributeError:
                    pg._ctx = ctx_bad
                pg.set_plugin_config({"plugin": {"enabled": True, "config_version": "1.0.0"},
                                      "diary": {"min_messages": 2}, "schedule": {"enabled": False}})
                ok_bad, msg_bad = asyncio.run(pg._generate_for_date("2026-09-26"))
                check(f"{label}不生成", ok_bad is False, f"{label} 竟通过：{msg_bad!r}")
            gate_store = gate_dir / "diaries.json"
            check("语义闸全程零存档", not gate_store.exists(),
                  "不可发布的成文不应落盘")

        check("含「无法」的正常句子不误判", bd.diary_output_problem(
            "2026年9月26日 星期六，多云。\n晚上听阿讲事，我一时无法理解他为什么那么想。") == "")
        check("空串判定为不可发布", bd.diary_output_problem("") == "输出为空")

        print("\n[6] 权限：fail-open 与静默拒绝")
        plugin3 = plugin_mod.create_plugin()
        try:
            plugin3.ctx = ctx
        except AttributeError:
            plugin3._ctx = ctx
        plugin3.set_plugin_config(
            {
                "plugin": {"enabled": True, "config_version": "1.0.0"},
                "security": {"admin_ids": ["888"]},
            }
        )
        allowed = plugin3._is_admin({"user_id": "888"})
        denied = plugin3._is_admin({"user_id": "999"})
        failopen = plugin3._is_admin({})
        plugin4 = plugin_mod.create_plugin()
        try:
            plugin4.ctx = ctx
        except AttributeError:
            plugin4._ctx = ctx
        plugin4.set_plugin_config({"plugin": {"enabled": True, "config_version": "1.0.0"}, "security": {"admin_ids": []}})
        empty_open = plugin4._is_admin({"user_id": "999"})
        check("admin 命中放行", allowed)
        check("非 admin 拒绝", not denied)
        check("admin_ids 缺失 user_id 不放行", not failopen)
        check("admin_ids 空 fail-open", empty_open)

        plugin3.set_plugin_config(
            {
                "plugin": {"enabled": True, "config_version": "1.0.0"},
                "security": {"admin_ids": ["888"]},
            }
        )
        sent_before = len(ctx.sent)
        result = asyncio.run(plugin3.cmd_diary(matched_groups={}, stream_id="s1", user_id="999"))
        check("非管理员命令静默（不发言）", len(ctx.sent) == sent_before and result[0] is True)

    print("\n[7] QQ空间发布模块")
    bdq = sys.modules["bd_qzone"]
    import httpx as _httpx

    gtk = bdq.generate_gtk("abcd1234")
    check("g_tk 为非负数字字符串", gtk.isdigit() and int(gtk) >= 0)
    check("g_tk 确定性", bdq.generate_gtk("abcd1234") == gtk)

    code = bdq.extract_code('frameElement.callback({"code":0,"tid":"abc"});')
    check("extract_code 剥回调噪声", code == 0)

    cookies = {"uin": "o0247200547", "p_skey": "abcd1234", "skey": "@xxx"}

    def _mk_transport(handler):
        return _httpx.MockTransport(handler)

    async def _pub(transport, content="测试日记"):
        pub = bdq.QzonePublisher(cookies, transport=transport)
        return await pub.publish_text(content, timeout=5)

    async def run_qzone_cases():
        ok_results = {}

        def ok_handler(request: _httpx.Request) -> _httpx.Response:
            ok_results["url"] = str(request.url)
            body = request.read().decode()
            ok_results["has_con"] = "con=" in body and "hostuin=" in body
            return _httpx.Response(200, json={"code": 0, "tid": "12345"})

        ok_results["r1"] = await _pub(_mk_transport(ok_handler))

        def expired_handler(request: _httpx.Request) -> _httpx.Response:
            return _httpx.Response(200, json={"code": -3000})

        try:
            await _pub(_mk_transport(expired_handler))
            ok_results["r2"] = ("no-raise", "")
        except bdq.CookieExpiredError as exc:
            ok_results["r2"] = ("raised", str(exc))

        def fail_handler(request: _httpx.Request) -> _httpx.Response:
            return _httpx.Response(500, text="oops")

        ok_results["r3"] = await _pub(_mk_transport(fail_handler))

        def code1_handler(request: _httpx.Request) -> _httpx.Response:
            return _httpx.Response(200, json={"code": 1, "msg": "err"})

        ok_results["r4"] = await _pub(_mk_transport(code1_handler))

        pub_empty = bdq.QzonePublisher(cookies, transport=_mk_transport(ok_handler))
        ok_results["r5"] = await pub_empty.publish_text("   ")
        return ok_results

    qz = asyncio.run(run_qzone_cases())
    check("发布成功返回 tid", qz["r1"][0] is True and qz["r1"][1] == "12345", str(qz["r1"]))
    check("发布请求带 g_tk 与 uin 参数", "g_tk=" in qz.get("url", "") and "uin=" in qz.get("url", ""))
    check("发布表单含 con 与 hostuin 字段", qz.get("has_con") is True)
    check("登录失效抛 CookieExpiredError", qz["r2"][0] == "raised", str(qz["r2"]))
    check("HTTP 500 返回失败", qz["r3"][0] is False and "500" in qz["r3"][1], str(qz["r3"]))
    check("code!=0 返回失败", qz["r4"][0] is False, str(qz["r4"]))
    check("空白内容不发布", qz["r5"][0] is False)

    pub_bad = bdq.QzonePublisher({"uin": "123"})
    r_bad = asyncio.run(pub_bad.publish_text("hi"))
    check("缺 p_skey 拒绝发布", r_bad[0] is False and "p_skey" in r_bad[1])

    print("\n[7b] 外发请求不跟随跳转 + cookie 域限定")
    _redirect_hits: list[str] = []

    def redirect_handler(request: _httpx.Request) -> _httpx.Response:
        _redirect_hits.append(str(request.url))
        return _httpx.Response(302, headers={"location": "https://evil.example.com/steal"},
                               request=request)

    pub_redir = bdq.QzonePublisher(
        {"uin": "o02472005478", "p_skey": "PSKEY_SECRET_VALUE"},
        transport=_httpx.MockTransport(redirect_handler),
    )
    r_redir = asyncio.run(pub_redir.publish_text("hello", timeout=5))
    check("跳转被判失败", r_redir[0] is False and "重定向" in r_redir[1], str(r_redir))
    check("不跟随跳转（仅一次请求）", len(_redirect_hits) == 1, str(_redirect_hits))
    check("凭据未触达第三方域",
          all("evil.example.com" not in u for u in _redirect_hits), str(_redirect_hits))

    _jar_seen: list[tuple[str, str]] = []

    def jar_handler(request: _httpx.Request) -> _httpx.Response:
        host = _httpx.URL(str(request.url)).host
        _jar_seen.append((host, request.headers.get("cookie", "")))
        if host == "user.qzone.qq.com":
            return _httpx.Response(302, headers={"location": "https://evil.example.com/steal"},
                                   request=request)
        return _httpx.Response(200, text="ok", request=request)

    async def _jar_probe():
        # 故意打开 follow_redirects，验证「域限定 jar」自身就是一道防线
        async with _httpx.AsyncClient(follow_redirects=True,
                                      transport=_httpx.MockTransport(jar_handler)) as c:
            c.cookies.set("p_skey", "PSKEY_SECRET_VALUE", domain="user.qzone.qq.com", path="/")
            await c.post("https://user.qzone.qq.com/x")

    asyncio.run(_jar_probe())
    check("域限定 jar 正常带上 cookie",
          ("user.qzone.qq.com", "p_skey=PSKEY_SECRET_VALUE") in _jar_seen, str(_jar_seen))
    check("域限定 jar 即便跳转也不外泄",
          not [c for h, c in _jar_seen if h == "evil.example.com" and "PSKEY_SECRET_VALUE" in c],
          str(_jar_seen))

    print("\n[7c] 日志脱敏（cookie 不进日志文件）")
    bdc_mod = sys.modules["bd_cookie"]
    red = bdc_mod.redact_secrets("失败: p_skey=ABC123, code=500")
    check("脱敏保留诊断信息", "p_skey=<redacted>" in red and "code=500" in red, red)
    check("脱敏覆盖 uin/skey/token",
          all(f"{k}=<redacted>" in bdc_mod.redact_secrets(f"{k}=X") for k in ("uin", "skey", "token")),
          bdc_mod.redact_secrets("uin=X skey=X token=X"))

    _captured: list[str] = []

    class _CapLog:
        def info(self, m): _captured.append(str(m))
        def warning(self, m): _captured.append(str(m))
        def error(self, m): _captured.append(str(m))
        def debug(self, m): _captured.append(str(m))

    async def _boom(name, params):
        raise RuntimeError("adapter error: cookies=uin=o01; p_skey=LEAK_SK; skey=LEAK_SK2")

    _store = bdc_mod.CookieStore(str(data_dir), api_call=_boom, logger=_CapLog())
    asyncio.run(_store.get_cookies(force=True))
    _joined = "\n".join(_captured)
    check("adapter 异常日志不泄漏 p_skey", "LEAK_SK" not in _joined, _joined)
    check("adapter 异常日志有内容（未静默）", bool(_captured), "应至少有一条 warning")

    print("\n[8] Cookie 自动获取模块（三级来源）")
    bdc = sys.modules["bd_cookie"]
    import httpx as _httpx

    ok_cookie_str = "uin=o0247200547; p_skey=pskey123; skey=@skey456"
    parsed = bdc.parse_cookie_string(ok_cookie_str)
    check("cookie 串解析", parsed.get("uin") == "o0247200547" and parsed.get("p_skey") == "pskey123")

    with tempfile.TemporaryDirectory() as td2:
        cookie_dir = Path(td2)
        calls = {"adapter": 0, "http": 0}

        async def fake_api_call(name, params):
            calls["adapter"] += 1
            calls["last_name"] = name
            calls["last_params"] = params
            return {"status": "ok", "data": {"cookies": ok_cookie_str}}

        def napcat_http_handler(request: _httpx.Request) -> _httpx.Response:
            calls["http"] += 1
            body = json.loads(request.read().decode())
            if body.get("domain") != "user.qzone.qq.com":
                return _httpx.Response(400, json={"status": "failed"})
            return _httpx.Response(
                200,
                json={"status": "ok", "retcode": 0, "data": {"cookies": ok_cookie_str}},
            )

        transport = _httpx.MockTransport(napcat_http_handler)

        async def cookie_cases():
            # 场景 A：adapter 正常 → 不走 HTTP
            store_a = bdc.CookieStore(
                cookie_dir / "a", fake_api_call,
                napcat_http={"host": "127.0.0.1", "port": "3000", "token": "tok"},
                interval_sec=3600, transport=transport,
            )
            ra1 = await store_a.get_cookies()
            ra2 = await store_a.get_cookies()  # 节流期内走缓存
            ra3 = await store_a.get_cookies(force=True)  # force 绕过节流

            # 场景 B：adapter 抛错（1.3.0 无 adapter）→ 降级 NapCat HTTP
            async def broken_api(name, params):
                raise RuntimeError("api.call 被拒（adapter 未装载）")

            store_b = bdc.CookieStore(
                cookie_dir / "b", broken_api,
                napcat_http={"host": "127.0.0.1", "port": "3000", "token": "tok"},
                interval_sec=3600, transport=transport,
            )
            rb = await store_b.get_cookies()

            # 场景 C：adapter 失败 + HTTP 403 → 全来源失败
            def forbidden_handler(request: _httpx.Request) -> _httpx.Response:
                return _httpx.Response(403, json={"status": "failed", "message": "token error"})

            store_c = bdc.CookieStore(
                cookie_dir / "c", broken_api,
                napcat_http={"host": "127.0.0.1", "port": "3000", "token": "wrong"},
                interval_sec=3600, transport=_httpx.MockTransport(forbidden_handler),
            )
            rc = await store_c.get_cookies()
            return store_a, ra1, ra2, ra3, rb, rc

        store_a, ra1, ra2, ra3, rb, rc = asyncio.run(cookie_cases())
        check("来源1 adapter 成功取到 cookie", ra1 is not None and ra1["p_skey"] == "pskey123", str(ra1))
        check("API 名与 domain 正确", calls.get("last_name") == "adapter.napcat.account.get_cookies" and calls.get("last_params", {}).get("domain") == "user.qzone.qq.com")
        check("来源1优先、来源2按需降级（计数正确）", calls["adapter"] == 2 and calls["http"] == 1, f"adapter {calls['adapter']} http {calls['http']}")
        check("节流期内不再调用远端", ra2 == ra1)
        check("force 绕过节流重取", ra3 is not None)
        check("adapter 挂掉降级 NapCat HTTP 成功", rb is not None and rb["uin"] == "o0247200547", str(rb))
        check("HTTP 请求带 domain 参数", calls["http"] >= 1)
        check("全来源失败返回缓存兜底（无缓存则 None）", rc is None)
        check("cookie 落盘（可重启恢复）", (cookie_dir / "a" / "cookies.json").exists())
        store2 = bdc.CookieStore(cookie_dir / "a", fake_api_call, interval_sec=3600)
        check("新实例从磁盘恢复缓存", store2._cookies is not None and store2._cookies["p_skey"] == "pskey123")

        async def fail_case():
            async def bad_api(name, params):
                return {"status": "failed", "message": "not logged in"}

            bad_store = bdc.CookieStore(Path(td2) / "sub", bad_api, interval_sec=3600)
            return await bad_store.get_cookies()

        check("两来源都失败返回 None", asyncio.run(fail_case()) is None)

        async def missing_field_case():
            async def half_api(name, params):
                return {"status": "ok", "data": {"cookies": "uin=o0123; foo=bar"}}

            s = bdc.CookieStore(Path(td2) / "sub2", half_api, interval_sec=3600)
            return await s.get_cookies()

        check("缺 p_skey 拒收", asyncio.run(missing_field_case()) is None)

    print("\n[9] 平铺兜底分支（脚本直跑场景）")
    try:
        flat_mod = load_flat()
        check("平铺加载也可用（插件目录在 sys.path）", hasattr(flat_mod, "BetterDiaryPlugin"))
        check("平铺兜底注册了 bd_prompts", "bd_prompts" in sys.modules)
    except ImportError as exc:  # pragma: no cover
        check("平铺加载也可用（插件目录在 sys.path）", False, str(exc))

    print("\n[10] 插件目录无残留")
    residue = PLUGIN_DIR / "data"
    check("插件目录无 data/ 残留", not residue.exists())

    print(f"\n===== 冒烟结果: PASS {len(PASS)} / FAIL {len(FAIL)} =====")
    if FAIL:
        for f in FAIL:
            print("  FAIL:", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
