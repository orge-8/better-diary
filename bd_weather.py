"""按城市取**当天真实天气**（纯协议模块，不接触 self.ctx，可脱机单测）。

为什么需要它
------------
v1.10.1 保证了「日期行每天都有天气」，但缺证据时写的是 ``[diary].weather_fallback``
这个**固定值**——连续一周都写「多云」，与真实天气无关。本模块把它换成按城市查到的
真实气象数据，同时保持既有铁律：**客观事实必须严格**。

三级来源（逐级降级，任一成功即止）
----------------------------------
1. **Open-Meteo forecast**（``api.open-meteo.com``）—— 支持 ``past_days``，
   一次请求即可覆盖「今天 + 最近若干天」，补写历史日期时拿到的仍是**那一天**的天气；
2. **Open-Meteo archive**（``archive-api.open-meteo.com``）—— 目标日期超出
   forecast 窗口时改查历史归档（实测 T-1 即可用）；
3. **wttr.in** —— 完全独立的第三方，只返回今天起 3 天。**目标日期不在返回里就
   宁可放弃**，绝不拿今天的天气冒充历史某天。

城市 → 经纬度：``[weather].latitude/longitude`` 优先；否则用 Open-Meteo geocoding
按城市名解析（``country_code`` 可限定国家，避免「北京」命中重庆的同名村镇）。

设计约束
--------
* **按日期取，不按"现在"取**：日记可能补写昨天/前天，天气必须是那一天的。
* **只输出五个标签**：晴 / 多云 / 阴 / 雨 / 雪，与日期行既有取值完全一致。
  WMO code 的映射是确定性查表，不做任何"感觉像"的推断。
* **失败即降级**：网络不通、限流、解析异常一律返回 None，由调用方退回配置兜底；
  绝不因为天气拿不到就不写日记。
* **不新增模型调用**：全部是 HTTP + 查表。
* **节流 + 落盘**：同一城市同一天只查一次（结果写 ``weather.json``），失败也有
  退避窗口，避免每次生成都去撞同一堵墙。
* **日志只记城市名与标签**，不打印响应体（外部返回的任意文本可能夹带回显内容）。

httpx 是**可选依赖**（与 bd_cookie / bd_qzone 一致）：没有它本模块仍可导入，
插件照常加载，只是取不到真实天气、退回配置兜底。
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - 真机未装 httpx 时走这里
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WTTR_URL = "https://wttr.in/"

#: 日期行只认这五个值（与 bd_prompts._WEATHERS 保持一致）。
WEATHER_LABELS = ("晴", "多云", "阴", "雨", "雪")

#: forecast 一次多带几天历史，覆盖 schedule.catch_up_days 的补写窗口。
_PAST_DAYS = 7
#: archive 比 forecast 慢，超出这个天数才值得去查归档。
_FORECAST_WINDOW_DAYS = _PAST_DAYS
#: 失败后的退避窗口（秒）。同一进程内反复生成不再重复撞网络。
_FAILURE_BACKOFF_SEC = 30 * 60
#: 城市解析结果的有效期（秒）。城市坐标不会变，缓存久一点省一次请求。
_GEO_TTL_SEC = 30 * 24 * 3600
#: 落盘最多留几天的天气，防止文件无限增长。
_CACHE_KEEP_DAYS = 30
#: wttr.in 的礼貌标识（它是社区服务，带上项目地址便于对方识别来源）。
_USER_AGENT = "better-diary/1.11 (+https://github.com/orge-8/better-diary)"

# WMO 4677 天气代码 → 五标签。查表而非推断：
#   0 晴；1 基本晴；2 局部多云；3 阴
#   45/48 雾与雾凇 —— 天空是灰白的，按「阴」记（不臆造「晴」）
#   51~67 毛毛雨/雨/冻雨 → 雨
#   71~77 雪/米雪/冰粒 → 雪
#   80~82 阵雨 → 雨；85/86 阵雪 → 雪
#   95~99 雷暴 → 雨（雷暴常伴冰雹，但日期行没有「雹」这个取值，按降水主形态记）
_WMO_LABELS: dict[int, str] = {
    0: "晴", 1: "晴", 2: "多云", 3: "阴",
    45: "阴", 48: "阴",
    51: "雨", 53: "雨", 55: "雨", 56: "雨", 57: "雨",
    61: "雨", 63: "雨", 65: "雨", 66: "雨", 67: "雨",
    71: "雪", 73: "雪", 75: "雪", 77: "雪",
    80: "雨", 81: "雨", 82: "雨", 85: "雪", 86: "雪",
    91: "雨", 92: "雨", 95: "雨", 96: "雨", 99: "雨",
}

# wttr.in 的英文描述 → 五标签。**先雪后雨**：`Light sleet` / `snow showers`
# 这类混合降水按雪记，与 bd_prompts.weather_label 的优先级一致。
_WTTR_RULES: tuple[tuple[str, str], ...] = (
    ("blizzard", "雪"), ("snow", "雪"), ("sleet", "雪"), ("ice pellets", "雪"),
    ("thunder", "雨"), ("shower", "雨"), ("rain", "雨"), ("drizzle", "雨"),
    ("overcast", "阴"), ("fog", "阴"), ("mist", "阴"), ("haze", "阴"), ("smoke", "阴"),
    ("partly cloudy", "多云"), ("cloudy", "多云"),
    ("sunny", "晴"), ("clear", "晴"),
)

_CITY_SAFE_RE = re.compile(r"^[\w\u4e00-\u9fff .,'\-]{1,60}$", re.UNICODE)


class _NoLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def wmo_label(code: Any) -> str:
    """WMO 天气代码 → 五标签；未知代码返回空串（宁可降级，也不猜）。"""
    try:
        return _WMO_LABELS.get(int(code), "")
    except (TypeError, ValueError):
        return ""


def wttr_label(desc: Any) -> str:
    """wttr.in 英文天气描述 → 五标签；识别不了返回空串。"""
    text = str(desc or "").strip().lower()
    if not text:
        return ""
    for keyword, label in _WTTR_RULES:
        if keyword in text:
            return label
    return ""


def dominant_label(labels: list[str]) -> str:
    """把一天里多个时段的天气合成**一个**当日标签。

    规则（确定性，可解释）：
    1. 出现过雪 → 雪；否则出现过雨 → 雨（降水是当天最该被记住的事实）；
    2. 无降水时取**出现最多**的天空状态，平票按 阴 > 多云 > 晴 —— 保守一档，
       不把「大半天阴、傍晚放晴」写成全天晴。
    """
    clean = [x for x in labels if x in WEATHER_LABELS]
    if not clean:
        return ""
    if "雪" in clean:
        return "雪"
    if "雨" in clean:
        return "雨"
    counts = {label: clean.count(label) for label in ("阴", "多云", "晴")}
    return max(("阴", "多云", "晴"), key=lambda label: counts[label])


def pick_wttr_day(payload: Any, date_str: str) -> str:
    """从 wttr.in 的 j1 结构里取**指定日期**的白天天气。

    只看白天时段（06:00~18:00）：日记写的是白天经历，凌晨的天气不该主导标签。
    目标日期不在返回里就返回空串 —— 这条是硬约束，防止用今天的天气冒充历史某天。
    """
    if not isinstance(payload, dict):
        return ""
    for day in payload.get("weather") or []:
        if not isinstance(day, dict) or str(day.get("date") or "") != date_str:
            continue
        labels: list[str] = []
        for hour in day.get("hourly") or []:
            if not isinstance(hour, dict):
                continue
            try:
                # wttr 的 time 是 "0"/"300"/"1200" 这种非补零写法
                hhmm = int(str(hour.get("time") or "0"))
            except (TypeError, ValueError):
                continue
            if not (600 <= hhmm <= 1800):
                continue
            desc = hour.get("weatherDesc") or []
            value = desc[0].get("value") if desc and isinstance(desc[0], dict) else ""
            label = wttr_label(value)
            if label:
                labels.append(label)
        return dominant_label(labels)
    return ""


def normalize_city(value: Any) -> str:
    """城市名归一化：去空白、限长度、挡掉会污染 URL 的字符。

    只做**字符白名单**，不做地名合法性判断 —— 合法与否交给 geocoding 回答。
    """
    text = " ".join(str(value or "").split())
    if not text or not _CITY_SAFE_RE.match(text):
        return ""
    return text[:60]


def parse_coords(lat: Any, lon: Any) -> tuple[float, float] | None:
    """校验经纬度配置；任一非法即返回 None（宁可走城市名，也不用错坐标）。"""
    try:
        flat, flon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= flat <= 90.0 and -180.0 <= flon <= 180.0):
        return None
    if flat == 0.0 and flon == 0.0:  # 「都没填」被写成 0 的常见误配
        return None
    return round(flat, 4), round(flon, 4)


def choose_geo_result(results: Any, country_code: str = "") -> dict[str, Any] | None:
    """从 geocoding 结果里挑一个坐标。

    真机风险：搜「北京」会同时命中北京直辖市与重庆/四川的同名村镇。因此
    **先按国家过滤，再按人口降序**（缺人口按 0），取最大的那个 —— 同名地方里
    人口最多的通常就是用户想写的那个城市。仍然建议直接配经纬度消除歧义。
    """
    if not isinstance(results, list):
        return None
    wanted = str(country_code or "").strip().upper()
    candidates = [
        item for item in results
        if isinstance(item, dict)
        and parse_coords(item.get("latitude"), item.get("longitude")) is not None
        and (not wanted or str(item.get("country_code") or "").strip().upper() == wanted)
    ]
    if not candidates:
        return None

    def _population(item: dict[str, Any]) -> int:
        try:
            return int(item.get("population") or 0)
        except (TypeError, ValueError):
            return 0

    best = max(candidates, key=_population)
    coords = parse_coords(best.get("latitude"), best.get("longitude"))
    if coords is None:
        return None
    return {
        "lat": coords[0],
        "lon": coords[1],
        "name": str(best.get("name") or ""),
        "admin1": str(best.get("admin1") or ""),
        "country_code": str(best.get("country_code") or ""),
        "timezone": str(best.get("timezone") or "auto"),
    }


class WeatherStore:
    """按城市 + 日期取真实天气，带落盘缓存与失败退避。"""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        city: str = "",
        latitude: Any = None,
        longitude: Any = None,
        country_code: str = "CN",
        timezone: str = "auto",
        timeout: float = 10.0,
        logger: Any = None,
        transport: Any = None,
        now_fn: Callable[[], float] | None = None,
    ):
        """
        Args:
            data_dir: 宿主注入的可变数据目录（缓存落盘处，绝不写插件源码目录）
            city: 城市名（中/英文均可）；配了经纬度时可留空
            latitude/longitude: 显式坐标，优先于城市名解析
            country_code: geocoding 的国家过滤（默认 CN），留空则不限国家
            timezone: 天气按哪个时区切「一天」；auto = 让 API 按坐标判定
            transport: 测试注入点（httpx.MockTransport）
            now_fn: 时间注入点，便于测试退避窗口
        """
        self._data_dir = Path(data_dir)
        self.city = normalize_city(city)
        self._coords = parse_coords(latitude, longitude)
        self.country_code = str(country_code or "").strip().upper()[:2]
        self.timezone = str(timezone or "auto").strip() or "auto"
        self.timeout = max(3.0, min(60.0, float(timeout or 10.0)))
        self._logger = logger or _NoLogger()
        self._transport = transport
        self._now = now_fn or time.time
        self._cache: dict[str, Any] = {}
        self._geo: dict[str, Any] | None = None
        self._failed_at: dict[str, float] = {}
        self._load_cache()

    # ------------------------------------------------------------ 配置与缓存

    @property
    def configured(self) -> bool:
        """是否配了可用的定位信息（城市名或经纬度）。"""
        return bool(self.city or self._coords)

    def describe_location(self) -> str:
        """给日志用的一句话定位描述（城市名不是凭据，可以直接打）。"""
        if self._coords:
            return f"坐标 {self._coords[0]},{self._coords[1]}"
        return self.city or "未配置"

    def _cache_path(self) -> Path:
        return self._data_dir / "weather.json"

    def _cache_key(self, date_str: str) -> str:
        place = f"{self._coords[0]},{self._coords[1]}" if self._coords else self.city
        return f"{date_str}|{place}"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return  # 坏即空：天气缓存丢了只是多查一次，不影响日记
        if isinstance(data, dict):
            days = data.get("days")
            self._cache = dict(days) if isinstance(days, dict) else {}
            geo = data.get("geo")
            if isinstance(geo, dict) and geo.get("city") == self.city:
                self._geo = geo

    def _save_cache(self) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            cutoff = (
                datetime.date.today() - datetime.timedelta(days=_CACHE_KEEP_DAYS)
            ).isoformat()
            days = {
                key: value for key, value in self._cache.items()
                if str(key).split("|", 1)[0] >= cutoff
            }
            self._cache = days
            payload = {"days": days, "geo": self._geo or {}}
            path = self._cache_path()
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, path)  # 原子替换，避免写一半被读到
        except OSError as exc:
            self._logger.warning("天气缓存落盘失败（不影响本次结果）: %s", exc)

    # ------------------------------------------------------------ 网络

    async def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        """发一次 GET 并解析 JSON。任何异常都转成 None，绝不向上抛。

        ``trust_env=False``：与 bd_cookie / bd_qzone 一致，不走系统代理。
        本机存在 MITM 根证书的环境里，走代理反而会撞上证书校验失败。
        """
        if not HTTPX_AVAILABLE:
            return None
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=False,
                transport=self._transport,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                res = await client.get(url, params=params)
            if res.status_code != 200:
                # 只打状态码：响应体是外部任意文本，可能夹带回显内容
                self._logger.warning("天气接口返回 HTTP %s（%s）", res.status_code, url)
                return None
            return res.json()
        except Exception as exc:  # noqa: BLE001 - 网络/解析全故障都只是降级
            self._logger.warning("天气接口调用失败（%s）: %s", url, type(exc).__name__)
            return None

    async def _resolve_coords(self) -> dict[str, Any] | None:
        """拿到坐标：显式配置 > 缓存 > geocoding。"""
        if self._coords:
            return {
                "lat": self._coords[0], "lon": self._coords[1],
                "name": self.city, "admin1": "", "country_code": self.country_code,
                "timezone": self.timezone,
            }
        if not self.city:
            return None
        geo = self._geo
        if (
            isinstance(geo, dict)
            and geo.get("city") == self.city
            and parse_coords(geo.get("lat"), geo.get("lon")) is not None
            and self._now() - float(geo.get("ts") or 0) < _GEO_TTL_SEC
        ):
            return geo
        payload = await self._get_json(GEOCODE_URL, {
            "name": self.city, "count": 5, "language": "zh", "format": "json",
        })
        picked = choose_geo_result(
            (payload or {}).get("results") if isinstance(payload, dict) else None,
            self.country_code,
        )
        if picked is None:
            self._logger.warning(
                "城市「%s」解析不到坐标（country_code=%s）；可在 [weather] 里直接填 latitude/longitude",
                self.city, self.country_code or "不限",
            )
            return None
        picked.update({"city": self.city, "ts": self._now()})
        self._geo = picked
        self._save_cache()
        self._logger.info(
            "城市定位：%s → %s %s（%.4f, %.4f）",
            self.city, picked["country_code"], picked["admin1"], picked["lat"], picked["lon"],
        )
        return picked

    @staticmethod
    def _label_from_daily(payload: Any, date_str: str) -> str:
        """从 Open-Meteo 的 daily 结构里取指定日期的标签。"""
        if not isinstance(payload, dict):
            return ""
        daily = payload.get("daily")
        if not isinstance(daily, dict):
            return ""
        times = daily.get("time") or []
        codes = daily.get("weather_code") or []
        if not isinstance(times, list) or not isinstance(codes, list):
            return ""
        for index, day in enumerate(times):
            if str(day) == date_str and index < len(codes):
                return wmo_label(codes[index])
        return ""

    async def _fetch_open_meteo(self, geo: dict[str, Any], date_str: str) -> tuple[str, str]:
        """Open-Meteo：先 forecast（含 past_days），未覆盖再查 archive。"""
        tz = str(geo.get("timezone") or self.timezone or "auto")
        base = {
            "latitude": geo["lat"], "longitude": geo["lon"],
            "daily": "weather_code", "timezone": tz,
        }
        payload = await self._get_json(FORECAST_URL, {
            **base, "past_days": _PAST_DAYS, "forecast_days": 2,
        })
        label = self._label_from_daily(payload, date_str)
        if label:
            return label, "open-meteo"
        # 目标日期比 forecast 窗口更早（例如手工补写很久以前的一天）才查归档。
        try:
            target = datetime.date.fromisoformat(date_str)
        except ValueError:
            return "", ""
        age = (datetime.date.today() - target).days
        if age <= _FORECAST_WINDOW_DAYS:
            return "", ""
        payload = await self._get_json(ARCHIVE_URL, {
            **base, "start_date": date_str, "end_date": date_str,
        })
        label = self._label_from_daily(payload, date_str)
        return (label, "open-meteo-archive") if label else ("", "")

    async def _fetch_wttr(self, date_str: str) -> tuple[str, str]:
        """wttr.in 兜底：只认返回里**确实有**的那一天。"""
        place = self.city or (f"{self._coords[0]},{self._coords[1]}" if self._coords else "")
        if not place:
            return "", ""
        payload = await self._get_json(WTTR_URL + place, {"format": "j1"})
        label = pick_wttr_day(payload, date_str)
        return (label, "wttr.in") if label else ("", "")

    # ------------------------------------------------------------ 对外入口

    async def get_weather(self, date_str: str) -> dict[str, Any] | None:
        """取**指定日期**的真实天气。

        Returns:
            ``{"label": "雨", "source": "open-meteo", "city": "杭州", "date": "..."}``；
            拿不到返回 ``None``（调用方据此退回配置兜底，日记照常生成）。
        """
        try:
            datetime.date.fromisoformat(str(date_str))
        except (TypeError, ValueError):
            return None
        if not self.configured:
            return None
        key = self._cache_key(date_str)
        cached = self._cache.get(key)
        if isinstance(cached, dict) and cached.get("label") in WEATHER_LABELS:
            return dict(cached)
        last_fail = self._failed_at.get(key)
        if last_fail is not None and self._now() - last_fail < _FAILURE_BACKOFF_SEC:
            return None  # 退避窗口内不再重复撞网络
        if not HTTPX_AVAILABLE:
            self._logger.warning("未安装 httpx，跳过真实天气获取（改用配置兜底）")
            return None

        result: dict[str, Any] | None = None
        geo = await self._resolve_coords()
        if geo is not None:
            label, source = await self._fetch_open_meteo(geo, date_str)
            if label:
                result = {"label": label, "source": source, "city": self.city or geo.get("name", "")}
        if result is None:
            label, source = await self._fetch_wttr(date_str)
            if label:
                result = {"label": label, "source": source, "city": self.city}
        if result is None:
            self._failed_at[key] = self._now()
            self._logger.warning(
                "%s 的真实天气没取到（%s），本次改用配置兜底；%.0f 分钟内不再重试",
                date_str, self.describe_location(), _FAILURE_BACKOFF_SEC / 60,
            )
            return None
        result["date"] = date_str
        self._cache[key] = result
        self._save_cache()
        self._logger.info(
            "真实天气：%s %s → %s（来源 %s）", date_str, self.describe_location(),
            result["label"], result["source"],
        )
        return dict(result)
