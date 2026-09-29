"""cookie 自动获取（纯逻辑模块，不接触 self.ctx，可脱机单测）。

三级来源，逐级降级：
1. napcat-adapter API（`adapter.napcat.account.get_cookies`）—— MaiBot 1.2.x 有
   adapter 插件时可用；1.3.0 起 adapter 不兼容，此路自然失败
2. NapCat HTTP 服务器（POST /get_cookies，OneBot 标准动作）—— 与 MaiBot 版本
   无关，只需 NapCat 开一个 HTTP Server（自 Maizone cookie.py 移植）
3. 手动配置兜底（plugin.py 侧处理）

内存缓存 + 节流 + data_dir 原子落盘（tmp + os.replace + chmod 600）。
adapter api 调用以注入的 async callable 表达；napcat HTTP 用 httpx 直连
（trust_env=False，本机直连不走系统代理）。

httpx 是**可选依赖**：只有来源 2（NapCat HTTP）用到它。没有 httpx 时本模块
仍可正常导入，插件也能加载，只是该来源不可用（来源 1/3 不受影响）。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Awaitable

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - 真机未装 httpx 时走这里
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

ADAPTER_API = "adapter.napcat.account.get_cookies"
ADAPTER_DOMAIN = "user.qzone.qq.com"

# 日志脱敏：抹掉「键=值 / 键: 值」里疑似凭据的值部分。
# 真机踩坑：adapter 抛异常时会把请求的 cookie 串写进异常消息，直接
# `f"...: {e}"` 会把 p_skey 明文打进日志文件。
_SECRET_KV_RE = re.compile(
    r"((?:p_skey|skey|g_tk|gtk|uin|sid|qzonetoken|pt[a-z_]*|token|password|passwd|cookie)"
    r"\s*[=:]\s*)([^\s;,&\"'）)]+)",
    re.IGNORECASE,
)


def redact_secrets(text: Any) -> str:
    """把文本里疑似凭据的「键=值」值部分替换成 ``<redacted>``。"""
    return _SECRET_KV_RE.sub(r"\1<redacted>", str(text or ""))


def parse_cookie_string(cookie_str: str) -> dict:
    """将 'k=v; k2=v2' 形式的 cookie 字符串解析为字典。"""
    cookies = {}
    for pair in cookie_str.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        if key not in cookies:
            cookies[key] = value
    return cookies


class _NoLogger:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def debug(self, msg):
        pass


class CookieStore:
    """多级 cookie 来源 + 节流 + data_dir 落盘。"""

    def __init__(
        self,
        data_dir: str | Path,
        api_call: Callable[[str, dict], Awaitable[Any]] | None,
        napcat_http: dict | None = None,
        interval_sec: int = 3600,
        logger=None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        """
        Args:
            api_call: napcat-adapter 的 api.call 注入（不再可用时传 None）
            napcat_http: {"host": str, "port": int|str, "token": str}，None 表示未配置
            transport: 测试注入点（httpx.MockTransport）
        """
        self._data_dir = Path(data_dir)
        self._api_call = api_call
        self._napcat_http = napcat_http or None
        self.interval_sec = max(60, int(interval_sec))
        self._logger = logger or _NoLogger()
        self._transport = transport
        self._cookies: dict | None = None
        self._last_refresh_time = 0.0
        self.load_from_disk()  # 启动恢复，省一次远端调用

    # ------------------------------------------------------------ 磁盘

    def _cookies_path(self) -> Path:
        return self._data_dir / "cookies.json"

    def load_from_disk(self) -> None:
        """启动时从 data_dir 恢复 cookie（坏即空）。"""
        path = self._cookies_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("uin") and data.get("p_skey"):
                self._cookies = data
                self._logger.debug("已从磁盘恢复 cookie 缓存")
        except (json.JSONDecodeError, OSError):
            pass  # 坏即空

    def _save_to_disk(self, cookies: dict) -> None:
        """原子落盘，收紧文件权限（cookie 含登录态）。"""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            path = self._cookies_path()
            tmp_path = str(path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cookies, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except OSError as e:
            self._logger.error(f"保存 cookie 失败: {redact_secrets(e)}")

    # ------------------------------------------------------------ 来源 1：adapter API

    async def _fetch_via_adapter(self) -> dict | None:
        """napcat-adapter API（MaiBot 1.3.0 起 adapter 不兼容时自然失败）。"""
        if self._api_call is None:
            return None
        try:
            result = await self._api_call(ADAPTER_API, {"domain": ADAPTER_DOMAIN})
        except Exception as e:
            # 异常消息可能内嵌 cookie 串（adapter 会把请求体带进 error），必须脱敏
            self._logger.warning(
                f"napcat-adapter 取 cookie 失败（将尝试 NapCat HTTP）: {redact_secrets(e)}"
            )
            return None
        if (
            not isinstance(result, dict)
            or result.get("status") != "ok"
            or "cookies" not in result.get("data", {})
        ):
            # 只打结构与错误消息，不打完整 result——adapter 异常时可能把 cookie 串放进 error 字段
            status = result.get("status") if isinstance(result, dict) else type(result).__name__
            err_msg = (result.get("message") or result.get("error")) if isinstance(result, dict) else ""
            if isinstance(err_msg, dict):
                err_msg = "（结构化错误，已省略）"
            self._logger.warning(
                f"adapter 取 cookie 失败: status={status}, message={redact_secrets(err_msg)}"
            )
            return None
        return parse_cookie_string(str(result["data"]["cookies"]))

    # ------------------------------------------------------------ 来源 2：NapCat HTTP

    async def _fetch_via_napcat_http(self) -> dict | None:
        """直连 NapCat HTTP 服务器的 get_cookies（与 MaiBot 版本无关）。"""
        if not HTTPX_AVAILABLE:
            self._logger.warning("未安装 httpx，跳过 NapCat HTTP 来源（pip install httpx）")
            return None
        if not self._napcat_http or not self._napcat_http.get("host"):
            return None
        host = str(self._napcat_http["host"]).strip()
        port = str(self._napcat_http.get("port", "")).strip()
        token = str(self._napcat_http.get("token", "")).strip()
        if not host or not port:
            return None
        url = f"http://{host}:{port}/get_cookies"
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(
                timeout=30.0, trust_env=False, transport=self._transport
            ) as client:
                resp = await client.post(url, json={"domain": ADAPTER_DOMAIN}, headers=headers)
        except httpx.RequestError as e:
            self._logger.warning(f"NapCat HTTP 连接失败: {redact_secrets(e)}")
            return None
        if resp.status_code != 200:
            hint = " (Token 验证失败)" if resp.status_code == 403 else ""
            self._logger.warning(f"NapCat HTTP 返回 {resp.status_code}{hint}，确认已开启 HTTP 服务器")
            return None
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            self._logger.warning("NapCat HTTP 响应不是 JSON")
            return None
        if data.get("status") != "ok" or "cookies" not in data.get("data", {}):
            self._logger.warning(f"NapCat HTTP 取 cookie 失败: status={data.get('status')}")
            return None
        return parse_cookie_string(str(data["data"]["cookies"]))

    # ------------------------------------------------------------ 编排

    def _accept(self, parsed: dict | None) -> dict | None:
        """校验并接受一份新 cookie。"""
        if not parsed or not parsed.get("uin") or not parsed.get("p_skey"):
            if parsed:
                self._logger.error(f"cookie 缺少 uin 或 p_skey，字段: {sorted(parsed.keys())}")
            return None
        self._cookies = parsed
        self._last_refresh_time = time.time()
        self._save_to_disk(parsed)
        self._logger.info(f"获取 cookie 成功（uin={parsed['uin'].lstrip('o0')}）")
        return parsed

    async def _fetch_fresh(self) -> dict | None:
        """逐级尝试远端来源。"""
        parsed = await self._fetch_via_adapter()
        if parsed and parsed.get("uin") and parsed.get("p_skey"):
            return self._accept(parsed)
        parsed = await self._fetch_via_napcat_http()
        if parsed and parsed.get("uin") and parsed.get("p_skey"):
            return self._accept(parsed)
        return None

    async def get_cookies(self, force: bool = False) -> dict | None:
        """获取可用 cookie。

        节流期内且非 force 时用缓存。远端来源全失败时返回最后缓存
        （可能过期，调用方拿到 CookieExpiredError 后会 force 重取）。
        """
        if (
            not force
            and self._cookies
            and (time.time() - self._last_refresh_time) < self.interval_sec
        ):
            self._logger.debug("cookie 节流期内，使用缓存")
            return self._cookies

        fresh = await self._fetch_fresh()
        if fresh:
            return fresh
        return self._cookies  # 缓存兜底

    def get_age_sec(self) -> float | None:
        """当前缓存 cookie 的年龄（秒），无缓存返回 None。"""
        if self._last_refresh_time <= 0:
            return None
        return time.time() - self._last_refresh_time
