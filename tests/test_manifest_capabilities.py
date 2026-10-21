"""manifest 与实现的一致性闸门（v1.7.1，插件中心审核打回项回归）。

背景（真机/审核实录，不是理论推演）：
    v1.7.0 送审被打回两条：
    ① ``host_application.max_version`` 写成 ``1.99.99`` ——「X.99.99」式无限兼容，
       上限没有真实依据，审核口径明确应拒绝。
    ② ``capabilities`` 漏了 ``chat.get_group_streams`` / ``chat.get_all_streams``。
       这两个能力在 ``_resolve_stream_id`` 的降级路径里被动态调用，
       **不会**出现在静态调用里，所以一直没人发现。而 Host 是按能力名**精确比对**
       授权的：未声明的调用会被直接拒绝，这两条降级路径因此**永远拿不到结果**
       （静默降级为解析失败）—— 而插件自己的注释当时还声称
       「能力声明与使用一致是固定检查项」。

为什么之前的测试没抓到：``test_audit_security.py`` 只测了「chat.* 全不可用时
白名单仍能取材」，那是**运行时容错**，不是「声明 ⊇ 使用」的**静态一致性**。
两个维度不可互相替代。

本文件三层扫描，任一层发现未声明的能力即失败：
    L1  AST 显式调用   ``self.ctx.<group>.<method>(...)``
    L2  动态 getattr   ``getattr(self.ctx.<group>, <常量|遍历变量|跨函数形参>)``
    L3  字符串兜底     代码里**带组前缀的完整能力名**字符串 ∩ SDK 已知能力全集
                      （抓 L1/L2 覆盖不到的形态：拼接、间接引用）

    L3 只匹配 ``"chat.get_all_streams"`` 这类**带组前缀的完整名**，刻意不做裸名匹配
    （``"get_all_streams"`` → 猜它是 chat 组）。实测裸名匹配误报率不可接受：
    ``.get("query")`` 会命中 ``database.query``、``"count": len(x)`` 会命中
    ``database.count``。误报比漏报更致命 —— 它会把人训练成无脑忽略红色。
    裸名形态的实际覆盖由 L2 承担（它已能穿透「for 元组 → 形参 → getattr」）。

设计上刻意避开的三个坑（都是踩过的）：
    - **误报比漏报更致命**：``ctx.logger`` 是本地日志对象、不走 RPC 授权，
      必须显式排除，否则每次跑测试都报「logger.debug 未声明」，
      人会被噪音训练成无脑忽略红色。
    - **不能手抄能力清单**：``llm.generate`` 是真实能力，却不在
      ``_CAPABILITY_RESULT_KEYS`` 里（它没有统一结果键，走单独分支）。
      所以已知全集一律**扫 SDK 源码里的 call_capability 字面量**得到。
    - **静默降级 = 假绿**：读不到 SDK 时退回内置快照，但必须显式告警。

只做静态分析，不依赖 devkit / fakehost，任何环境都能跑。
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PLUGIN_DIR / "_manifest.json"
PLUGIN_PATH = PLUGIN_DIR / "plugin.py"

# ctx 上「不是能力」的属性：这些是宿主直接注入的对象，走本地调用而非 RPC 授权，
# 不需要、也不能写进 capabilities。改动前请先确认 SDK 里它们确实不调 call_capability。
NON_CAPABILITY_CTX_ATTRS: frozenset[str] = frozenset({"logger"})

# maibot_sdk 2.8.0 的能力名快照，仅当无法扫描真实 SDK 源码时作为 L3 兜底依据。
SDK_CAPABILITY_SNAPSHOT: frozenset[str] = frozenset(
    {
        "api.call", "api.get", "api.list",
        "chat.get_all_streams", "chat.get_group_streams", "chat.get_private_streams",
        "chat.get_stream_by_group_id", "chat.get_stream_by_user_id", "chat.open_session",
        "component.get_all_plugins", "component.get_plugin_info",
        "component.list_loaded_plugins", "component.list_registered_plugins",
        "config.get", "config.get_all", "config.get_plugin",
        "database.count", "database.delete", "database.get",
        "database.query", "database.save",
        "emoji.get_all", "emoji.get_by_description", "emoji.get_count",
        "emoji.get_emotions", "emoji.get_info", "emoji.get_random",
        "frequency.get_adjust", "frequency.get_current_talk_value", "frequency.set_adjust",
        "knowledge.search",
        "llm.generate", "llm.generate_with_tools", "llm.get_available_models",
        "llm.transcribe_audio",
        "message.build_readable", "message.count_new", "message.get_by_id",
        "message.get_by_time", "message.get_by_time_in_chat", "message.get_recent",
        "person.get_id", "person.get_id_by_name", "person.get_value",
        "render.html2png",
        "send.command", "send.custom", "send.emoji", "send.forward",
        "send.hybrid", "send.image", "send.text",
        "statistics.local.message_trend", "statistics.local.model_trend",
        "statistics.local.models", "statistics.local.online_time_trend",
        "statistics.local.token_distribution", "statistics.local.token_trend",
        "statistics.local.tool_trend",
        "tool.get_definitions",
    }
)


def _load_manifest() -> dict:
    assert MANIFEST_PATH.is_file(), f"找不到 manifest: {MANIFEST_PATH}"
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _scan_sdk_capabilities() -> frozenset[str] | None:
    """扫已安装 maibot_sdk 源码里的 ``call_capability("<字面量>"`` 得到真实能力全集。

    不手抄常量表：``llm.generate`` 就不在 ``_CAPABILITY_RESULT_KEYS`` 里
    （它返回结构不统一，走单独分支），手抄会漏。
    """
    try:
        import maibot_sdk  # type: ignore
    except Exception:
        return None

    root = Path(maibot_sdk.__file__).resolve().parent
    found: set[str] = set()
    for py in root.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fname != "call_capability" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add(first.value)
    return frozenset(found) if found else None


def _known_capabilities() -> frozenset[str]:
    """已知能力全集。优先扫真实 SDK 源码，扫不到用快照并**显式告警**（不静默降级）。"""
    scanned = _scan_sdk_capabilities()
    if scanned:
        return scanned
    warnings.warn(
        "未能从 maibot_sdk 源码扫描到能力全集（未安装 maibot_sdk，或版本结构有变），"
        "三层扫描改用内置快照；快照可能滞后于你本机的 SDK 版本，请留意。",
        RuntimeWarning,
        stacklevel=1,
    )
    return SDK_CAPABILITY_SNAPSHOT


def _known_groups(known: frozenset[str]) -> set[str]:
    return {k.split(".", 1)[0] for k in known}


# --------------------------------------------------------------------------
# 扫描器
# --------------------------------------------------------------------------

def _ctx_group(node: ast.AST) -> str | None:
    """判断 node 是否为 ``self.ctx.<group>``，是则返回 group 名。"""
    if not isinstance(node, ast.Attribute):
        return None
    inner = node.value
    if not isinstance(inner, ast.Attribute) or inner.attr != "ctx":
        return None
    if not isinstance(inner.value, ast.Name) or inner.value.id != "self":
        return None
    return node.attr


def _func_index(tree: ast.AST) -> dict[str, ast.AST]:
    """函数名 -> 函数节点（含嵌套函数，重名取最后一个；这里只需要遍历用）。"""
    idx: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            idx[node.name] = node
    return idx


def _enclosing_func(tree: ast.AST, target: ast.AST):
    """返回**最内层**直接包含 target 的函数节点。

    嵌套函数（如 ``_resolve_stream_id`` 里的 ``_call``）必须取内层，
    否则形态 A 的 for 变量、形态 B 的形参回溯都会对不上。
    """
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(x is target for x in ast.walk(node)):
            continue
        if best is None or any(x is node for x in ast.walk(best)):
            best = node
    return best


def _names_from_for(scope: ast.AST, var_name: str) -> list[str]:
    """把 ``for var in ("a", "b")`` 的常量序列解成名字列表。"""
    out: list[str] = []
    for sub in ast.walk(scope):
        if isinstance(sub, ast.For) and isinstance(sub.target, ast.Name):
            if sub.target.id != var_name:
                continue
            it = sub.iter
            if isinstance(it, (ast.Tuple, ast.List, ast.Set)):
                out.extend(
                    e.value for e in it.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
    return out


def _collect_explicit(tree: ast.AST, groups: set[str]) -> dict[str, int]:
    """L1：``self.ctx.<group>.<method>(...)`` 显式调用。"""
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        group = _ctx_group(node.func.value)
        if group is None or group not in groups:
            continue
        found[f"{group}.{node.func.attr}"] = node.lineno
    return found


def _collect_dynamic(tree: ast.AST, groups: set[str]) -> dict[str, int]:
    """L2：``getattr(self.ctx.<group>, <名字>)``。

    名字可能是：常量串 / 本函数内 for 遍历变量 / 本函数形参（由调用点的 for 变量传入）。
    最后一种正是 v1.7.0 事故的实际形态 ——
    ``_call(method)`` 内部 getattr，方法名来自 ``_resolve_stream_id`` 的 for 元组。
    """
    found: dict[str, int] = {}
    funcs = _func_index(tree)

    # 先建「函数节点 -> 所在函数」的反向索引，便于把形参回溯到调用点
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            continue
        if len(node.args) not in (2, 3):
            continue
        group = _ctx_group(node.args[0])
        if group is None or group not in groups:
            continue

        owner = _enclosing_func(tree, node)
        if owner is None:
            continue
        name_arg = node.args[1]
        names: list[str] = []

        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            names = [name_arg.value]
        elif isinstance(name_arg, ast.Name):
            var = name_arg.id
            # 形态 A：本函数内 ``for var in (...)``
            names = _names_from_for(owner, var)
            # 形态 B：var 是本函数形参，回溯调用点实参
            if not names:
                positional = [a.arg for a in owner.args.args]
                if var in positional:
                    pos = positional.index(var)
                    for caller in ast.walk(tree):
                        if not isinstance(caller, ast.Call):
                            continue
                        if not isinstance(caller.func, ast.Name) or caller.func.id != owner.name:
                            continue
                        if pos >= len(caller.args):
                            continue
                        actual = caller.args[pos]
                        if isinstance(actual, ast.Constant) and isinstance(actual.value, str):
                            names.append(actual.value)
                        elif isinstance(actual, ast.Name):
                            scope = _enclosing_func(tree, caller)
                            if scope is not None:
                                names.extend(_names_from_for(scope, actual.id))
        for nm in dict.fromkeys(names):  # 去重保序
            found[f"{group}.{nm}"] = node.lineno
    return found


def _collect_string_constants(tree: ast.AST) -> set[str]:
    """代码里的字符串常量（排除 docstring）。"""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    for owner in [tree] + [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]:
        body = getattr(owner, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                out.discard(first.value.value)
    return out


def _collect_l3(tree: ast.AST, known: frozenset[str]) -> dict[str, int]:
    """L3：代码里的字符串常量 ∩ 已知能力全集，**只认带组前缀的完整名**。

    例如 ``"chat.get_all_streams"`` 会被计入；``"get_all_streams"`` 不会 ——
    裸名无法可靠归组，且 ``query`` / ``count`` / ``get`` 这类通用词满地都是，
    实测会让这条闸门每次都误报，进而被当成噪音忽略。
    """
    hits: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        raw = node.value.strip()
        if raw in known:
            hits.setdefault(raw, node.lineno)
    return hits


def _parse_version(v: str) -> tuple[int, int, int]:
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)\s*$", str(v))
    assert m, f"版本号格式应为 X.Y.Z，实际为 {v!r}"
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _scan_all() -> tuple[dict, dict, dict, frozenset]:
    src = PLUGIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    known = _known_capabilities()
    groups = _known_groups(known)
    return (
        _collect_explicit(tree, groups),
        _collect_dynamic(tree, groups),
        _collect_l3(tree, known),
        known,
    )


# --------------------------------------------------------------------------
# ① capabilities 必须覆盖实现里用到的所有能力
# --------------------------------------------------------------------------

def test_capabilities_cover_explicit_calls():
    """L1：所有 ``self.ctx.<group>.<method>()`` 显式调用都要在 manifest 里声明。"""
    declared = set(_load_manifest().get("capabilities", []))
    used, _, _, _ = _scan_all()

    assert used, "L1 一个都没扫到 —— 扫描逻辑坏了，这条闸门已经是假绿，必须修"
    missing = {k: v for k, v in used.items() if k not in declared}
    assert not missing, (
        "以下能力在实现里被显式调用，但 _manifest.json 没声明；"
        "Host 按名字精确比对授权，未声明的调用会被拒绝："
        + "；".join(f"{k} (plugin.py:{v})" for k, v in sorted(missing.items()))
    )


def test_capabilities_cover_dynamic_getattr():
    """L2：降级路径里 getattr 动态调用的能力也要声明（v1.7.0 打回项的真实形态）。"""
    declared = set(_load_manifest().get("capabilities", []))
    _, used, _, _ = _scan_all()

    assert used, (
        "L2 一个都没扫到：plugin.py 里应当存在 getattr(self.ctx.chat, ...) 动态调用。"
        "若确实已重构掉，请同步更新本文件与 _resolve_stream_id 的注释"
    )
    missing = {k: v for k, v in used.items() if k not in declared}
    assert not missing, (
        "以下能力经 getattr 动态调用，静态扫描看不到，但 Host 仍按名字精确比对授权；"
        "漏声明会让对应降级路径静默失效："
        + "；".join(f"{k} (plugin.py:{v})" for k, v in sorted(missing.items()))
    )


def test_capabilities_cover_string_constants():
    """L3 兜底：代码字符串常量 ∩ SDK 已知能力全集，也必须全部声明。"""
    declared = set(_load_manifest().get("capabilities", []))
    _, _, used, _ = _scan_all()

    missing = {k: v for k, v in used.items() if k not in declared}
    assert not missing, (
        "代码中出现了以下已知能力名但未声明（拼接/间接引用等 L1/L2 抓不到的形态）："
        + "；".join(f"{k} (plugin.py:{v})" for k, v in sorted(missing.items()))
    )


def test_fallback_paths_actually_declared():
    """直接钉死 v1.7.0 的打回项：两条降级能力必须在声明里。

    上面三条是通用扫描，这条是**针对本次事故**的显式断言 ——
    即便将来有人为了过测试去改扫描逻辑，这两条仍在。
    """
    declared = set(_load_manifest().get("capabilities", []))
    for cap in ("chat.get_group_streams", "chat.get_all_streams"):
        assert cap in declared, (
            f"{cap} 未声明：_resolve_stream_id 的降级路径动态调用它，"
            "漏声明会让该降级路径永远解析失败（v1.7.0 审核打回项）"
        )


def test_scanners_stay_effective():
    """防假绿：三层里负责主要覆盖的 L1/L3 必须有产出，否则闸门已失效。"""
    explicit, dynamic, l3, _ = _scan_all()
    assert explicit, "L1 显式调用扫描结果为空 —— 扫描逻辑坏了"
    assert l3, "L3 字符串兜底扫描结果为空 —— 扫描逻辑坏了"
    # L2 依赖当前代码形态（getattr + for 元组/形参），为空时只提示不失败：
    # 但 _resolve_stream_id 确有该形态，为空意味着有人重构了却没同步这里。
    if not dynamic:
        warnings.warn(
            "L2 未扫到动态 getattr 调用：若 _resolve_stream_id 的降级路径已重构，"
            "请同步更新 tests/test_manifest_capabilities.py 与对应注释。",
            RuntimeWarning,
            stacklevel=1,
        )


def test_no_unknown_ctx_groups():
    """ctx 上出现了既非能力组、也不在已知非能力白名单里的属性 —— 需要人工确认。

    这条是「扫描器盲区」的看门狗：新 SDK 若新增了能力组，
    或插件用了新型调用，这里会先响，而不是被静默忽略。
    """
    known = _known_capabilities()
    groups = _known_groups(known) | NON_CAPABILITY_CTX_ATTRS
    tree = ast.parse(PLUGIN_PATH.read_text(encoding="utf-8"))

    suspects: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        g = _ctx_group(node.func.value)
        if g and g not in groups:
            suspects.setdefault(g, node.lineno)
    assert not suspects, (
        f"ctx 上出现了未知属性组 {suspects}：若它是新 SDK 的能力组，"
        f"请确认扫描器能取到（当前已知组={sorted(groups)}）；"
        "若它和 logger 一样不走 RPC 授权，请加进 NON_CAPABILITY_CTX_ATTRS"
    )


def test_manifest_declares_only_known_capabilities():
    """反向：声明的能力必须是 SDK 里真实存在的名字，防止手滑写错反被静默拒绝。"""
    known = _known_capabilities()
    declared = set(_load_manifest().get("capabilities", []))
    unknown = sorted(c for c in declared if c not in known)
    assert not unknown, (
        f"manifest 声明了 SDK 里不存在的能力名：{unknown}。"
        "若确认是新版 SDK 新增能力，请确认扫描器能扫到（见 _scan_sdk_capabilities）"
    )


# --------------------------------------------------------------------------
# ② host_application.max_version 写法合规（v1.7.0 打回项）
# --------------------------------------------------------------------------

def test_max_version_is_not_open_ended():
    """``X.99.99`` 式无限兼容必须拒绝：上限要有真实依据。"""
    host = _load_manifest().get("host_application", {})
    lo, hi = _parse_version(host.get("min_version")), _parse_version(host.get("max_version"))

    assert hi[1] != 99, (
        f"max_version={host.get('max_version')} 属「X.99.99」式无限兼容写法，"
        "兼容上限没有真实依据，审核口径明确应拒绝；"
        "应改为与主程序实际版本线匹配的上限（如 1.3.99）"
    )
    assert hi[0] != 99, f"max_version={host.get('max_version')} 主版本号非法"
    assert hi >= lo, (
        f"max_version({host.get('max_version')}) 小于 min_version({host.get('min_version')})"
    )
    assert hi[0] - lo[0] <= 1, (
        f"max_version({host.get('max_version')}) 跨了 {hi[0] - lo[0]} 个主版本，"
        "声称兼容未验证的主版本线，缺乏依据"
    )


def test_max_version_covers_current_host_line():
    """上限要覆盖当前主程序实际版本线（MaiBot 1.3.x，截至 2026-10 已发布 1.3.7/1.3.8）。"""
    host = _load_manifest().get("host_application", {})
    hi = _parse_version(host.get("max_version"))
    assert hi >= (1, 3, 0), (
        f"max_version={host.get('max_version')} 未覆盖当前主程序版本线 1.3.x，"
        "装了新版 MaiBot 的用户会直接加载失败"
    )


def test_min_version_matches_config_version_requirement():
    """min_version 维持 1.2.3：``plugin.config_version`` 是 Host 1.2.3 起的硬性要求。"""
    host = _load_manifest().get("host_application", {})
    lo = _parse_version(host.get("min_version"))
    assert lo == (1, 2, 3), (
        f"min_version={host.get('min_version')} 与 README 记载的 "
        "「plugin.config_version 是 Host 1.2.3 才引入的硬性要求」不一致"
    )


# --------------------------------------------------------------------------
# ③ manifest 基本结构
# --------------------------------------------------------------------------

def test_manifest_is_valid_json_with_required_fields():
    manifest = _load_manifest()
    for field in ("manifest_version", "id", "version", "name", "description",
                  "author", "license", "host_application", "capabilities"):
        assert field in manifest, f"manifest 缺少字段: {field}"

    assert manifest["manifest_version"] == 2
    assert re.match(r"^\d+\.\d+\.\d+$", str(manifest["version"])), "version 应为 X.Y.Z"
    assert manifest["capabilities"], "capabilities 不应为空"


def test_manifest_version_matches_changelog_head():
    """manifest 版本号要与 CHANGELOG 最新条目对得上，避免发版漏改一处。"""
    version = str(_load_manifest()["version"])
    changelog = PLUGIN_DIR / "CHANGELOG.md"
    assert changelog.is_file(), "找不到 CHANGELOG.md"
    head = changelog.read_text(encoding="utf-8")[:4000]
    assert version in head, (
        f"CHANGELOG.md 开头 4000 字符里没有当前版本号 {version}，"
        "请确认发版时两处都改了"
    )
