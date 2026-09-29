"""QQ空间说说发布（纯协议模块，不接触 self.ctx，可脱机单测）。

移植自 qzone-feeds/qzone_api.py 的发表说说路径（上游 Maizone），仅保留纯文本发布：
- g_tk 用 p_skey 走 5381 算法
- 发布接口 emotion_cgi_publish_v6，syn_tweet_verson 为官方拼写错误勿改
- 登录类错误码 → CookieExpiredError，由调用方提示更新 cookie

httpx 是**可选依赖**：本模块唯一用到它，且只在真正发起发布时才需要。
没有 httpx 时本模块仍可正常导入，插件能加载；调用 publish_text 会抛
PublishUnavailableError，由 plugin.py 转成可读提示。
"""

from __future__ import annotations

import json
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - 真机未装 httpx 时走这里
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

EMOTION_PUBLISH_URL = (
    "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_publish_v6"
)

# cookie 的域限定：登录态只对该域有意义。写进 client jar 时显式限定，
# 这样即便日后有人打开 follow_redirects，跳转也不会把 cookie 送到别的域。
QZONE_COOKIE_DOMAIN = "user.qzone.qq.com"

# 登录类错误码：出现即认为登录态失效
_LOGIN_ERROR_CODES = {1000000, 1000001, 1000002, 1000003, -3000, -3001, -14}


class CookieExpiredError(Exception):
    """cookie 失效（登录态丢失）。"""


class PublishUnavailableError(Exception):
    """发布所需依赖缺失（如未安装 httpx）。"""


def generate_gtk(skey: str) -> str:
    """QQ空间 g_tk 算法（5381 哈希，p_skey 版本）。"""
    hash_val = 5381
    for ch in skey:
        hash_val += (hash_val << 5) + ord(ch)
    return str(hash_val & 2147483647)


def extract_code(text: str) -> Any | None:
    """从空间响应（可能带回调噪声）中提取 code。"""
    if not text:
        return None
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "code" in data:
                return data.get("code")
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def build_publish_payload(content: str, uin: str) -> dict[str, str]:
    """构造发表说说的表单（抽出便于测试）。"""
    return {
        "syn_tweet_verson": "1",  # 官方拼写错误，勿改
        "paramstr": "1",
        "who": "1",
        "con": content,
        "feedversion": "1",
        "ver": "1",
        "ugc_right": "1",
        "to_sign": "0",
        "hostuin": uin,
        "code_version": "1",
        "format": "json",
        "qzreferrer": "https://user.qzone.qq.com/" + str(uin),
    }


class QzonePublisher:
    """纯文本说说发布器。"""

    def __init__(self, cookies: dict | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.cookies = dict(cookies or {})
        self.uin = str(self.cookies.get("uin", "")).lstrip("o0")
        self.gtk2 = generate_gtk(self.cookies["p_skey"]) if self.cookies.get("p_skey") else ""
        self._transport = transport  # 测试注入点

    async def publish_text(self, content: str, timeout: float = 20.0) -> tuple[bool, str]:
        """发表纯文本说说。成功返回 (True, tid)；失败返回 (False, 原因)。"""
        if not content.strip():
            return False, "内容为空，未发布"
        if not self.uin or not self.gtk2:
            return False, "cookies 不完整（需要 uin 与 p_skey）"
        if not HTTPX_AVAILABLE:
            raise PublishUnavailableError("未安装 httpx，无法发布（pip install httpx）")

        # 双重防线：
        # ① `follow_redirects=False` —— 本请求携带登录态 cookie，绝不自动跟随跳转。
        #    （httpx 在重定向时会无条件剥掉 Cookie 头、改用 client jar 重新派生，
        #     见 httpx/_client.py `_redirect_headers`。）
        # ② cookie 以**显式域限定**写入 client jar，而不是按请求传 `cookies=`
        #    （后者已被 httpx 标记弃用）。域限定意味着即便日后有人把跳转打开，
        #    跳转目标域也拿不到 cookie（已用 MockTransport 实测确认）。
        # ③ `trust_env=False` —— 与 bd_cookie 一致：本请求带登录态，不走系统代理，
        #    避免 HTTP_PROXY/HTTPS_PROXY 上的中间人代理看到 cookie。
        async with httpx.AsyncClient(
            follow_redirects=False, timeout=timeout, trust_env=False,
            transport=self._transport,
        ) as client:
            for key, value in self.cookies.items():
                client.cookies.set(
                    str(key), str(value), domain=QZONE_COOKIE_DOMAIN, path="/"
                )
            res = await client.request(
                method="POST",
                url=EMOTION_PUBLISH_URL,
                params={"g_tk": self.gtk2, "uin": self.uin},
                data=build_publish_payload(content, self.uin),
                headers={
                    "referer": "https://user.qzone.qq.com/" + self.uin,
                    "origin": "https://user.qzone.qq.com",
                },
            )

        if 300 <= res.status_code < 400:
            # 只回显跳转目标主机名，不回显完整 URL（可能是带参数的登录跳转）
            location = res.headers.get("location", "")
            try:
                host = httpx.URL(location).host if location else ""
            except Exception:  # noqa: BLE001 - 畸形 location 不该拖垮错误处理
                host = ""
            return False, f"空间返回重定向 HTTP {res.status_code}（{host or '未知目标'}），已拒绝跟随"

        if res.status_code != 200:
            return False, f"HTTP {res.status_code}"

        code = extract_code(res.text)
        if code is not None:
            try:
                if int(code) in _LOGIN_ERROR_CODES:
                    raise CookieExpiredError(f"code={code}")
            except (TypeError, ValueError):
                pass
        if code != 0:
            # 不回显响应体：那是外部返回的任意文本，可能夹带凭据/回显内容，
            # 而这条消息会进日志、也会发到聊天里。只报错误码。
            return False, f"空间返回 code={code}"

        try:
            data = res.json()
            tid = str(data.get("tid") or "")
        except (json.JSONDecodeError, ValueError):
            tid = ""
        return True, tid or "ok"
