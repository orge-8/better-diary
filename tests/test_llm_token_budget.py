"""v1.7.0 · 输出 token 额度与空输出重试。

每条用例对应 2026-10-20 真机故障链条上的一个环节：

- 宿主 utils 任务 ``max_tokens=4096``，而 MiniMax-M3 是**自适应思考模型** ——
  它把思维链也算进额度。模型一决定"先想再答"，4096 就被思维链吃光，
  正文一个 token 都没轮到，插件拿到的是空串。
- 修法是按请求抬高额度，但插件必须先知道 **Host 到底接不接受 max_tokens**
  （传了不生效比不传更隐蔽），所以有一条能力探针；判定不下来就按不支持处理。
- 空输出此前既不重试也不留诊断：一次截断 = 当天日记直接判失败，
  日志里只剩一句「输出为空」，分不清是额度问题还是调用问题。

加载方式与真机一致：包式加载（submodule_search_locations），插件目录不在 sys.path 上。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
# devkit（提供 fakehost）不是插件的运行依赖，只在开发机上跑测试时用。
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

from fakehost import (  # noqa: E402
    FakeHost,
    build_context,
    get_default_config,
    load_plugin_module,
)

sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
MOD = load_plugin_module(str(PLUGIN_DIR), module_name="better_diary_token_budget")


# ---------------------------------------------------------------- 工具

_CHUNK_LINE = "[10:00] 甲: 今天发生了一件事，大家聊了一会儿。\n"
_DEFAULT_MAX_TOKENS = 16384


def _result(text: str, reasoning: str = "") -> dict:
    """仿 Host llm.generate 的返回；``reasoning`` 即思考型模型的思维链字段。"""
    return {"success": True, "response": text, "reasoning": reasoning, "model": "fake"}


def _extract_reply(label: str) -> str:
    return f'[{{"who": "甲", "what": "{label}", "quote": "原话说了一遍", "score": 4}}]'


def _is_probe(payload: dict) -> bool:
    return payload.get("max_tokens") == 1


def _is_extract(payload: dict) -> bool:
    return payload.get("temperature") == 0.2


class LLMRecorder:
    """顶替 ``_raw_llm_result``：记录每次真实发出的 payload，按脚本返回结果。"""

    def __init__(self, responder) -> None:
        self.payloads: list[dict] = []
        self._responder = responder

    async def __call__(self, payload, *, timeout_seconds=None) -> dict:
        self.payloads.append(dict(payload))
        return self._responder(dict(payload))

    @property
    def probe_calls(self) -> list[dict]:
        return [p for p in self.payloads if _is_probe(p)]

    @property
    def extract_calls(self) -> list[dict]:
        return [p for p in self.payloads if _is_extract(p)]

    @property
    def write_calls(self) -> list[dict]:
        return [
            p
            for p in self.payloads
            if p.get("temperature") == 0.8 and not _is_probe(p)
        ]


def build_plugin(responder=None, msgs=None, llm=None, diary=None):
    """构造注入了假 LLM 通道的插件实例（默认配置 + 分段覆盖）。"""
    msgs = [] if msgs is None else msgs
    plugin = MOD.create_plugin()
    host = FakeHost(
        returns={
            "message.get_by_time": msgs,
            "message.get_by_time_in_chat": msgs,
        }
    )
    ctx = build_context("org.orge-8.better-diary", rpc_call=host.rpc_call)
    plugin._set_context(ctx)
    cfg = get_default_config(MOD.BetterDiaryConfig)
    # 本文件测的是成文链路，显式选 all 让消息流进来
    cfg["diary"]["filter_mode"] = "all"
    cfg["llm"].update(llm or {})
    cfg["diary"].update(diary or {})
    plugin.set_plugin_config(cfg)
    if responder is not None:
        plugin._raw_llm_result = LLMRecorder(responder)  # type: ignore[assignment]
    return plugin


def probe_responder(behavior: str):
    """探针专用：按 ``behavior`` 模拟 Host 对 max_tokens 的三种态度。"""

    def respond(payload):
        if _is_probe(payload):
            if behavior == "supported":
                return _result("甲")  # 被 1 token 截断
            if behavior == "ignored":
                return _result("甲乙丙丁戊己庚辛壬癸")  # 完整复述 = 参数没生效
            raise RuntimeError("未知参数 max_tokens")
        return _result("正文")

    return respond


def script_responder(script: list[str], probe: str = "supported"):
    """按脚本依次出结果；**探针调用单独应答**。

    探针必须和业务调用分开：它若不单独应答，就会把脚本第一项吃掉，
    「第一次空、第二次成文」的用例会变成「第一次就成文」—— 测试全绿却根本没测到重试。
    """

    def respond(payload):
        if _is_probe(payload):
            return probe_responder(probe)(payload)
        return _result(script.pop(0)) if script else _result("")

    return respond


def constant_responder(text: str, probe: str = "supported"):
    """恒定应答（探针同样单独应答）。"""
    responder = probe_responder(probe)

    def respond(payload):
        if _is_probe(payload):
            return responder(payload)
        return _result(text)

    return respond


def make_messages(n: int):
    return [
        {
            "timestamp": str(1_700_000_000 + i),
            "processed_plain_text": f"闲聊第 {i} 句",
            "message_info": {
                "user_info": {"user_id": str(1000 + i), "user_nickname": f"用户{i}"}
            },
        }
        for i in range(n)
    ]


def timeline(n_lines: int = 600) -> str:
    """约 28 字/行 × 600 行 ≈ 16800 字 → 默认 6000 字/块切成 3 块。"""
    return _CHUNK_LINE * n_lines


# ---------------------------------------------------------------- 能力探针

def test_probe_detects_max_tokens_support():
    """被 1 token 截断 → Host 真的应用了该参数 → 之后照传。"""
    plugin = build_plugin(probe_responder("supported"))

    asyncio.run(plugin._ensure_max_tokens_probed())

    assert plugin._mt_supported is True
    assert plugin._max_tokens_arg() == {"max_tokens": _DEFAULT_MAX_TOKENS}


def test_probe_detects_ignored_max_tokens():
    """完整复述回来 → 参数被忽略 → 必须停止传参（传了不生效是更隐蔽的假象）。"""
    plugin = build_plugin(probe_responder("ignored"))

    asyncio.run(plugin._ensure_max_tokens_probed())

    assert plugin._mt_supported is False
    assert plugin._max_tokens_arg() == {}


def test_probe_treats_call_failure_as_unsupported():
    """Host 严格校验未知入参而报错 → 同样判不支持，绝不能顶着报错继续传。"""
    plugin = build_plugin(probe_responder("rejects"))

    asyncio.run(plugin._ensure_max_tokens_probed())

    assert plugin._mt_supported is False
    assert plugin._max_tokens_arg() == {}


def test_probe_warns_when_max_tokens_is_ignored(caplog):
    """不支持时必须把「该去哪里改」写进日志 —— 否则用户只会看到日记失败。"""
    plugin = build_plugin(probe_responder("ignored"))

    with caplog.at_level("WARNING"):
        asyncio.run(plugin._ensure_max_tokens_probed())

    msgs = [r.getMessage() for r in caplog.records]
    assert any("Host 未应用该参数" in m for m in msgs), msgs
    assert any("模型配置" in m for m in msgs), msgs


def test_probe_disabled_assumes_supported():
    """关掉探测就按支持处理：多传一个参数最多被忽略，不会更糟。"""
    plugin = build_plugin(probe_responder("ignored"), llm={"probe_max_tokens": False})

    asyncio.run(plugin._ensure_max_tokens_probed())

    assert plugin._max_tokens_arg() == {"max_tokens": _DEFAULT_MAX_TOKENS}


# ---------------------------------------------------------------- 实际传参

def test_max_tokens_is_sent_on_every_real_call():
    """探针判定支持 → 选材与成文都带上放大后的额度。"""
    plugin = build_plugin(probe_responder("supported"))

    async def run():
        await plugin._call_llm("选材", 0.2)
        await plugin._call_llm("成文", 0.8)

    asyncio.run(run())

    rec: LLMRecorder = plugin._raw_llm_result
    real = [p for p in rec.payloads if not _is_probe(p)]
    assert len(real) == 2
    assert all(p.get("max_tokens") == _DEFAULT_MAX_TOKENS for p in real), real


def test_probe_runs_only_once_under_concurrency():
    """三块选材并发首发，探针不能被点起三次。"""
    plugin = build_plugin(probe_responder("supported"))

    async def run():
        await asyncio.gather(*(plugin._call_llm(f"块{i}", 0.2) for i in range(3)))

    asyncio.run(run())

    rec: LLMRecorder = plugin._raw_llm_result
    assert len(rec.probe_calls) == 1, len(rec.probe_calls)


def test_zero_max_tokens_means_use_host_default():
    """max_tokens=0 → 不传、也不探测（沿用宿主配置，与 v1.6.0 行为一致）。"""
    plugin = build_plugin(probe_responder("supported"), llm={"max_tokens": 0})

    asyncio.run(plugin._call_llm("成文", 0.8))

    rec: LLMRecorder = plugin._raw_llm_result
    assert rec.write_calls, "应当有一次真实调用"
    assert all("max_tokens" not in p for p in rec.payloads), rec.payloads


def test_configured_max_tokens_is_honoured():
    """额度可配：想省就调小，不必改代码。"""
    plugin = build_plugin(probe_responder("supported"), llm={"max_tokens": 8192})

    asyncio.run(plugin._call_llm("成文", 0.8))

    rec: LLMRecorder = plugin._raw_llm_result
    assert [p["max_tokens"] for p in rec.write_calls] == [8192]


# ---------------------------------------------------------------- 空输出重试

def test_empty_output_is_retried():
    """第一次被思维链吃光、第二次直接作答 → 应当救回来。"""
    script = ["", "睡前翻了翻今天，群里在聊新出的那首歌。"]
    plugin = build_plugin(script_responder(script), llm={"retry_backoff_seconds": 0})

    raw = asyncio.run(plugin._call_llm_nonempty("p", 0.8, stage="日记成文"))

    assert raw.startswith("睡前翻了翻今天")
    # 关键：必须是两次成文调用，否则这条用例根本没走到重试
    assert len(plugin._raw_llm_result.write_calls) == 2


def test_empty_output_exhausts_retries_then_gives_up():
    """一直空 → 用完次数就停手，不能无限重跑。"""
    plugin = build_plugin(constant_responder(""), llm={"retry_backoff_seconds": 0})

    raw = asyncio.run(plugin._call_llm_nonempty("p", 0.8, stage="日记成文"))

    assert raw == ""
    rec: LLMRecorder = plugin._raw_llm_result
    # 默认 empty_retry=1 → 成文 2 次（首发 + 重试）
    assert len(rec.write_calls) == 2, len(rec.write_calls)


def test_empty_retry_can_be_turned_off():
    """empty_retry=0 → 一次空就判失败（回到 v1.6.0 行为，省调用）。"""
    plugin = build_plugin(
        constant_responder(""), llm={"empty_retry": 0, "retry_backoff_seconds": 0}
    )

    asyncio.run(plugin._call_llm_nonempty("p", 0.8, stage="日记成文"))

    rec: LLMRecorder = plugin._raw_llm_result
    assert len(rec.write_calls) == 1, len(rec.write_calls)


def test_refusal_is_not_retried():
    """拒答不重跑 —— 换一轮大概率还是拒答，纯属烧钱。"""
    refusal = "抱歉，作为一个人工智能，我无法完成这个请求。"
    plugin = build_plugin(constant_responder(refusal), llm={"retry_backoff_seconds": 0})

    asyncio.run(plugin._call_llm_nonempty("p", 0.8, stage="日记成文"))

    rec: LLMRecorder = plugin._raw_llm_result
    assert len(rec.write_calls) == 1, len(rec.write_calls)


def test_empty_retry_is_logged(caplog):
    """每一次空输出都要留痕，否则「重试救回来了」这件事在日志里是隐形的。"""
    script = ["", "成文内容"]
    plugin = build_plugin(script_responder(script), llm={"retry_backoff_seconds": 0})

    with caplog.at_level("WARNING"):
        asyncio.run(plugin._call_llm_nonempty("p", 0.8, stage="日记成文"))

    msgs = [r.getMessage() for r in caplog.records]
    assert any("第 1/2 次返回空输出" in m for m in msgs), msgs


# ---------------------------------------------------------------- 空正文诊断

def test_empty_body_is_diagnosed_with_reasoning_length(caplog):
    """空正文 + 长思维链 = 额度被思考阶段吃光，这是本次故障的指纹。"""
    plugin = build_plugin(
        lambda p: _result("甲") if _is_probe(p) else _result("", "想" * 300)
    )

    with caplog.at_level("WARNING"):
        asyncio.run(plugin._call_llm("成文", 0.8))

    msgs = [r.getMessage() for r in caplog.records]
    hit = [m for m in msgs if "LLM 返回空正文" in m]
    assert hit, msgs
    assert "推理内容 300 字" in hit[0], hit[0]
    assert f"max_tokens={_DEFAULT_MAX_TOKENS}" in hit[0], hit[0]
    assert "reasoning" in hit[0], "结果字段清单是判断 Host 返回结构的依据"


def test_full_body_is_not_diagnosed(caplog):
    """有正文就别刷日志 —— 抑制分支才需要留痕。"""
    plugin = build_plugin(constant_responder("正常的日记正文，写得挺长的。"))

    with caplog.at_level("WARNING"):
        asyncio.run(plugin._call_llm("成文", 0.8))

    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "LLM 返回空正文" in m], msgs


# ---------------------------------------------------------------- 选材空块补跑

def test_empty_extract_chunk_is_rerun():
    """三块首轮全空 → 串行补跑 → 素材应当回来了（真机 10-20 丢了块1）。"""
    counter = {"n": 0}

    def respond(payload):
        if _is_probe(payload):
            return _result("甲")
        if _is_extract(payload):
            counter["n"] += 1
            if counter["n"] <= 3:  # 首轮三块全部被思维链吃光
                return _result("")
            return _result(_extract_reply(f"第{counter['n']}件事"))
        return _result("正文")

    plugin = build_plugin(respond)

    events = asyncio.run(plugin._extract_events(timeline(600), "2026-10-20"))

    assert len(events) == 3, f"补跑后应拿回 3 件素材，实际 {len(events)}"
    assert counter["n"] == 6, f"应为 3 次首发 + 3 次补跑，实际 {counter['n']}"


def test_rerun_saved_chunk_does_not_report_empty(caplog):
    """补跑救回来的块不能再报一次「选材空块」，否则日志会误导。"""
    counter = {"n": 0}

    def respond(payload):
        if _is_probe(payload):
            return _result("甲")
        if _is_extract(payload):
            counter["n"] += 1
            if counter["n"] <= 3:
                return _result("")
            return _result(_extract_reply(f"第{counter['n']}件事"))
        return _result("正文")

    plugin = build_plugin(respond)

    with caplog.at_level("WARNING"):
        asyncio.run(plugin._extract_events(timeline(600), "2026-10-20"))

    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "选材空块" in m], msgs


def test_chunk_with_nothing_to_tell_is_not_rerun():
    """模型说「这块确实无事」是有效答案，重跑纯属烧钱还改变结果。"""
    counter = {"n": 0}

    def respond(payload):
        if _is_probe(payload):
            return _result("甲")
        if _is_extract(payload):
            counter["n"] += 1
            return _result("[]")
        return _result("正文")

    plugin = build_plugin(respond)

    events = asyncio.run(plugin._extract_events(timeline(600), "2026-10-20"))

    assert events == []
    assert counter["n"] == 3, f"确实无事不该触发补跑，实际调用 {counter['n']} 次"


def test_single_chunk_does_not_double_rerun():
    """单块时空输出已在内部重跑过，外层不能再补一轮（否则白烧一次）。"""
    counter = {"n": 0}

    def respond(payload):
        if _is_probe(payload):
            return _result("甲")
        if _is_extract(payload):
            counter["n"] += 1
            return _result("")
        return _result("正文")

    plugin = build_plugin(respond, llm={"retry_backoff_seconds": 0}, diary={"chunk_chars": 999999})

    asyncio.run(plugin._extract_events(timeline(10), "2026-10-20"))

    assert counter["n"] == 2, f"单块应为 2 次（首发 + 内部重试），实际 {counter['n']}"


# ---------------------------------------------------------------- 端到端

def test_diary_survives_one_truncated_write(tmp_path, monkeypatch):
    """整条链路：首轮成文被思维链吃光 → 重试一次 → 日记照常产出并存档。"""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    script = [
        "",
        "睡前翻了翻今天，群里在聊新出的那首歌。\n"
        "阿茶说「这个前奏一响我就跪了」，我笑出声。\n"
        "后来没人接话，就这样散了。",
    ]

    def respond(payload):
        if _is_probe(payload):
            return _result("甲")
        if _is_extract(payload):
            return _result(_extract_reply("聊了新歌"))
        return _result(script.pop(0))

    plugin = build_plugin(respond, msgs=make_messages(40), llm={"retry_backoff_seconds": 0})
    plugin._data_dir = lambda: tmp_path  # type: ignore[assignment]

    ok, result = asyncio.run(plugin._generate_for_date("2026-10-20"))

    assert ok is True, f"重试本应救回这次成文，实际失败: {result}"
    assert "前奏一响" in result
    assert plugin._load_diaries()["2026-10-20"]["content"] == result


def test_diary_still_fails_when_every_attempt_is_empty(tmp_path, monkeypatch):
    """重试救不回来时，语义闸必须照旧拦下 —— 不能因为重试过就放宽标准。"""
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    def respond(payload):
        if _is_probe(payload):
            return _result("甲")
        if _is_extract(payload):
            return _result(_extract_reply("聊了新歌"))
        return _result("", "想" * 5000)

    plugin = build_plugin(respond, msgs=make_messages(40), llm={"retry_backoff_seconds": 0})
    plugin._data_dir = lambda: tmp_path  # type: ignore[assignment]

    ok, result = asyncio.run(plugin._generate_for_date("2026-10-20"))

    assert ok is False, "全空输出绝不能当成成功成品"
    assert "输出为空" in result
    assert plugin._load_diaries() == {}, "失败当天不得落盘"
