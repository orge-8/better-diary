"""v1.11.0 真实天气回归：按城市 + **按日期**取天气，失败一律降级。

全部用 httpx.MockTransport 打桩，**不联网**；真实接口连通性另有实测记录。
覆盖的关键失败模式：
- 用今天的天气冒充历史某天（补写场景的事实错误）
- 城市同名歧义（「北京」命中重庆的同名村镇）
- 网络/限流/坏 JSON 把日记一起拖垮
- 每次生成都重复撞同一堵墙（无缓存、无退避）
- 兜底天气被当成「当天真的下雨」的证据
"""

from __future__ import annotations

import asyncio
import datetime
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
for _cand in (
    os.environ.get("MAIBOT_DEVKIT_DIR", ""),
    str(PLUGIN_DIR.parent / "tools" / "maibot-devkit"),
    str(PLUGIN_DIR / "maibot-devkit"),
):
    if _cand and (Path(_cand) / "fakehost.py").is_file():
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

from fakehost import FakeHost, build_context, get_default_config, load_plugin_module  # noqa: E402

httpx = pytest.importorskip("httpx", reason="httpx 是天气模块的可选依赖")

sys.path[:] = [p for p in sys.path if str(PLUGIN_DIR) not in str(p)]
MOD = load_plugin_module(str(PLUGIN_DIR), module_name="better_diary_weather_test")
W = importlib.import_module("better_diary_weather_test.bd_weather")
BD = importlib.import_module("better_diary_weather_test.bd_prompts")
SCENE = importlib.import_module("better_diary_weather_test.bd_scene")

TODAY = datetime.date.today()
TODAY_S = TODAY.isoformat()
OLD_S = (TODAY - datetime.timedelta(days=20)).isoformat()


def _forecast_payload(days: dict[str, int]) -> dict:
    return {"daily": {"time": list(days), "weather_code": list(days.values())}}


def _wttr_payload(days: dict[str, str]) -> dict:
    return {
        "weather": [
            {
                "date": date,
                "hourly": [
                    {"time": str(hour), "weatherDesc": [{"value": desc}]}
                    for hour in (0, 600, 1200, 1800, 2100)
                ],
            }
            for date, desc in days.items()
        ]
    }


def _geo_payload(results: list[dict]) -> dict:
    return {"results": results}


def _store(tmp_path, handler, **kwargs):
    return W.WeatherStore(
        tmp_path, transport=httpx.MockTransport(handler), **kwargs
    )


# ---------------------------------------------------------------- 纯函数映射

def test_wmo_codes_map_to_the_five_allowed_labels():
    assert W.wmo_label(0) == "晴"
    assert W.wmo_label(2) == "多云"
    assert W.wmo_label(3) == "阴"
    assert W.wmo_label(45) == "阴"      # 雾：天空灰白，不臆造成晴
    assert W.wmo_label(61) == "雨"
    assert W.wmo_label(71) == "雪"
    assert W.wmo_label(85) == "雪"      # 阵雪
    assert W.wmo_label(95) == "雨"      # 雷暴按降水主形态记
    # 未知代码宁可降级，也不猜一个天气
    assert W.wmo_label(999) == ""
    assert W.wmo_label(None) == ""
    assert W.wmo_label("abc") == ""
    assert all(W.wmo_label(c) in W.WEATHER_LABELS for c in W._WMO_LABELS)


def test_english_descriptions_prefer_snow_over_rain():
    assert W.wttr_label("Light sleet") == "雪"
    assert W.wttr_label("Patchy snow possible") == "雪"
    assert W.wttr_label("Patchy rain nearby") == "雨"
    assert W.wttr_label("Partly Cloudy ") == "多云"
    assert W.wttr_label("Overcast ") == "阴"
    assert W.wttr_label("Smoky haze") == "阴"
    assert W.wttr_label("Sunny") == "晴"
    assert W.wttr_label("完全看不懂的描述") == ""


def test_dominant_label_is_conservative_and_precipitation_wins():
    assert W.dominant_label(["晴", "雨", "晴"]) == "雨"
    assert W.dominant_label(["晴", "雪", "雨"]) == "雪"
    assert W.dominant_label(["晴", "晴", "阴"]) == "晴"   # 多数决
    assert W.dominant_label(["多云", "阴"]) == "阴"       # 平票保守一档
    assert W.dominant_label([]) == ""


def test_wttr_only_uses_daytime_and_only_the_requested_date():
    payload = _wttr_payload({TODAY_S: "Sunny"})
    assert W.pick_wttr_day(payload, TODAY_S) == "晴"
    # 关键：目标日期不在返回里 → 空串，绝不拿今天冒充历史某天
    assert W.pick_wttr_day(payload, OLD_S) == ""
    night_rain = {
        "weather": [{
            "date": TODAY_S,
            "hourly": [
                {"time": "0", "weatherDesc": [{"value": "Heavy rain"}]},
                {"time": "1200", "weatherDesc": [{"value": "Sunny"}]},
            ],
        }]
    }
    assert W.pick_wttr_day(night_rain, TODAY_S) == "晴"


def test_city_and_coordinate_normalization_rejects_bad_input():
    assert W.normalize_city("  杭州 ") == "杭州"
    assert W.normalize_city("Hangzhou") == "Hangzhou"
    assert W.normalize_city("../etc/passwd") == ""
    assert W.normalize_city("a?b=1") == ""
    assert W.parse_coords("30.25", "120.17") == (30.25, 120.17)
    assert W.parse_coords(0, 0) is None          # 「都没填」写成 0 的常见误配
    assert W.parse_coords(91, 0) is None
    assert W.parse_coords("", "") is None


def test_geocoding_prefers_country_filter_then_population():
    results = [
        {"name": "北京", "admin1": "重庆市", "country_code": "CN",
         "latitude": 30.72, "longitude": 108.67},
        {"name": "北京", "admin1": "北京市", "country_code": "CN",
         "latitude": 39.90, "longitude": 116.39, "population": 18960744},
        {"name": "Beijing", "country_code": "US",
         "latitude": 40.0, "longitude": -80.0, "population": 99999999},
    ]
    picked = W.choose_geo_result(results, "CN")
    assert picked is not None and picked["admin1"] == "北京市"
    assert W.choose_geo_result(results, "JP") is None
    assert W.choose_geo_result([], "CN") is None
    assert W.choose_geo_result(None, "CN") is None


# ---------------------------------------------------------------- 取数链路

def test_city_name_resolves_then_reads_the_requested_date(tmp_path):
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host.startswith("geocoding"):
            return httpx.Response(200, json=_geo_payload([
                {"name": "杭州", "admin1": "浙江", "country_code": "CN",
                 "latitude": 30.29, "longitude": 120.16, "population": 11936010},
            ]))
        return httpx.Response(200, json=_forecast_payload({TODAY_S: 61}))

    store = _store(tmp_path, handler, city="杭州")
    got = asyncio.run(store.get_weather(TODAY_S))
    assert got is not None
    assert got["label"] == "雨" and got["source"] == "open-meteo"
    assert got["date"] == TODAY_S
    assert calls == ["geocoding-api.open-meteo.com", "api.open-meteo.com"]


def test_second_call_hits_disk_cache_without_any_request(tmp_path):
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        return httpx.Response(200, json=_forecast_payload({TODAY_S: 0}))

    first = _store(tmp_path, handler, latitude=30.25, longitude=120.17)
    assert asyncio.run(first.get_weather(TODAY_S))["label"] == "晴"
    assert hits["n"] == 1
    assert (tmp_path / "weather.json").exists()

    # 新实例（模拟重启）：应直接读盘，不再发请求
    def forbidden(request):
        raise AssertionError("命中缓存后不应再发请求")

    second = _store(tmp_path, forbidden, latitude=30.25, longitude=120.17)
    assert asyncio.run(second.get_weather(TODAY_S))["label"] == "晴"
    assert hits["n"] == 1


def test_explicit_coordinates_skip_geocoding(tmp_path):
    def handler(request):
        assert not request.url.host.startswith("geocoding"), "配了坐标就不该再解析城市"
        return httpx.Response(200, json=_forecast_payload({TODAY_S: 3}))

    store = _store(tmp_path, handler, city="杭州", latitude=39.9, longitude=116.4)
    assert asyncio.run(store.get_weather(TODAY_S))["label"] == "阴"


def test_old_date_falls_back_to_archive_api(tmp_path):
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host.startswith("archive-api"):
            return httpx.Response(200, json=_forecast_payload({OLD_S: 71}))
        # forecast 窗口里没有这一天
        return httpx.Response(200, json=_forecast_payload({TODAY_S: 0}))

    store = _store(tmp_path, handler, latitude=30.25, longitude=120.17)
    got = asyncio.run(store.get_weather(OLD_S))
    assert got is not None and got["label"] == "雪"
    assert got["source"] == "open-meteo-archive"
    assert "archive-api.open-meteo.com" in seen


def test_wttr_is_used_when_open_meteo_fails(tmp_path):
    def handler(request):
        if request.url.host.startswith("wttr"):
            return httpx.Response(200, json=_wttr_payload({TODAY_S: "Light drizzle"}))
        return httpx.Response(500, text="boom")

    store = _store(tmp_path, handler, city="Hangzhou")
    got = asyncio.run(store.get_weather(TODAY_S))
    assert got is not None and got["label"] == "雨" and got["source"] == "wttr.in"


def test_wttr_must_not_substitute_today_for_a_past_date(tmp_path):
    """核心事实约束：备源只有今天，就绝不能拿它写历史那天的天气。"""
    def handler(request):
        if request.url.host.startswith("wttr"):
            return httpx.Response(200, json=_wttr_payload({TODAY_S: "Heavy rain"}))
        return httpx.Response(500, text="boom")

    store = _store(tmp_path, handler, city="Hangzhou")
    assert asyncio.run(store.get_weather(OLD_S)) is None


def test_all_sources_failing_degrades_to_none_and_backs_off(tmp_path):
    hits = {"n": 0}
    clock = {"t": 1000.0}

    def handler(request):
        hits["n"] += 1
        raise httpx.ConnectError("network down")

    store = _store(
        tmp_path, handler, city="Hangzhou", now_fn=lambda: clock["t"],
    )
    assert asyncio.run(store.get_weather(TODAY_S)) is None
    first_round = hits["n"]
    assert first_round > 0
    # 退避窗口内：不再撞网络
    clock["t"] += 60
    assert asyncio.run(store.get_weather(TODAY_S)) is None
    assert hits["n"] == first_round
    # 窗口过后：允许再试一次
    clock["t"] += W._FAILURE_BACKOFF_SEC + 1
    assert asyncio.run(store.get_weather(TODAY_S)) is None
    assert hits["n"] > first_round


def test_malformed_json_and_unknown_code_do_not_raise(tmp_path):
    def handler(request):
        if request.url.host.startswith("api.open-meteo"):
            return httpx.Response(200, json=_forecast_payload({TODAY_S: 999}))
        return httpx.Response(200, text="<html>not json</html>")

    store = _store(tmp_path, handler, city="Hangzhou")
    assert asyncio.run(store.get_weather(TODAY_S)) is None


def test_unconfigured_or_disabled_store_never_calls_network(tmp_path):
    def forbidden(request):
        raise AssertionError("未配置定位信息时不应联网")

    store = _store(tmp_path, forbidden)
    assert store.configured is False
    assert asyncio.run(store.get_weather(TODAY_S)) is None
    assert asyncio.run(store.get_weather("不是日期")) is None


def test_cache_is_pruned_and_corrupt_cache_is_ignored(tmp_path):
    def handler(request):
        return httpx.Response(200, json=_forecast_payload({TODAY_S: 0}))

    store = _store(tmp_path, handler, latitude=30.0, longitude=120.0)
    asyncio.run(store.get_weather(TODAY_S))
    ancient = (TODAY - datetime.timedelta(days=400)).isoformat()
    store._cache[f"{ancient}|30.0,120.0"] = {"label": "晴"}
    store._save_cache()
    assert not any(key.startswith(ancient) for key in store._cache)

    (tmp_path / "weather.json").write_text("{坏文件", encoding="utf-8")
    revived = _store(tmp_path, handler, latitude=30.0, longitude=120.0)
    assert asyncio.run(revived.get_weather(TODAY_S))["label"] == "晴"


# ---------------------------------------------------------------- 与日记链路的衔接

def test_real_weather_counts_as_evidence_but_fallback_does_not():
    body = f"2026年11月14日 星期六，雨。\n外面下着雨，懒得出门。"
    timeline = "甲: 今天随便聊聊"
    # 只有兜底值时，正文写窗外天气仍属无据虚构
    assert any("天气" in r for r in SCENE.diary_fact_risks(body, timeline))
    real_hint = "2026-11-14 杭州当天的真实天气是「雨」（下雨），来自气象数据（open-meteo）。"
    assert SCENE.diary_fact_risks(body, f"{timeline}\n{real_hint}") == []


def test_prompt_separates_weather_data_from_chat_events():
    prompt = BD.build_write_prompt(
        date_str="2026年11月14日 星期六",
        events_text="- 甲：聊了一件小事",
        name="鸣澜",
        persona="测试",
        weather_hint_text="当天的真实天气是「雨」（下雨），来自气象数据。",
        weather_is_real=True,
    )
    assert "已经查实" in prompt
    assert "不是聊天记录" in prompt
    assert "绝不能编造" in prompt
    # 聊天证据路径的措辞不能被改掉
    chat_prompt = BD.build_write_prompt(
        date_str="2026年11月14日 星期六",
        events_text="- 甲：聊了一件小事",
        name="鸣澜",
        persona="测试",
        weather_hint_text="当天明显有雨（素材里出现了「下雨」）。",
    )
    assert "已经有据可查" in chat_prompt
    assert "不是聊天记录" not in chat_prompt


def test_weather_config_defaults_to_no_network():
    cfg = MOD.BetterDiaryConfig.model_validate({
        "plugin": {"enabled": True, "config_version": "1.0.0"},
    })
    assert cfg.weather.enabled is True
    assert cfg.weather.city == ""
    assert cfg.weather.country_code == "CN"
    plugin = MOD.create_plugin()
    host = FakeHost(returns={})
    plugin._set_context(build_context("org.orge-8.better-diary", rpc_call=host.rpc_call))
    plugin.set_plugin_config(get_default_config(MOD.BetterDiaryConfig))
    # 默认不配城市 → 不构造 store，也就完全不联网
    assert plugin._build_weather_store() is None


def test_weather_config_normalizes_city_and_respects_disable(tmp_path):
    cfg = MOD.BetterDiaryConfig.model_validate({
        "plugin": {"enabled": True, "config_version": "1.0.0"},
        "weather": {"city": "  杭州 ", "enabled": False},
    })
    assert cfg.weather.city == "杭州"
    plugin = MOD.create_plugin()
    host = FakeHost(returns={})
    plugin._set_context(build_context("org.orge-8.better-diary", rpc_call=host.rpc_call))
    raw = get_default_config(MOD.BetterDiaryConfig)
    raw["weather"]["city"] = "杭州"
    raw["weather"]["enabled"] = False
    plugin.set_plugin_config(raw)
    assert plugin._build_weather_store() is None
    raw["weather"]["enabled"] = True
    plugin.set_plugin_config(raw)
    store = plugin._build_weather_store()
    assert store is not None and store.configured


def test_resolve_weather_prefers_chat_evidence_over_api(tmp_path):
    plugin = MOD.create_plugin()
    host = FakeHost(returns={})
    plugin._set_context(build_context("org.orge-8.better-diary", rpc_call=host.rpc_call))
    raw = get_default_config(MOD.BetterDiaryConfig)
    raw["weather"]["city"] = "杭州"
    plugin.set_plugin_config(raw)
    plugin._data_dir = lambda: tmp_path  # type: ignore[assignment]

    def handler(request):
        raise AssertionError("聊天已有天气证据时不该再查气象数据")

    plugin._weather_store = _store(tmp_path, handler, city="杭州")
    label, hint, source = asyncio.run(
        plugin._resolve_weather(TODAY_S, "甲: 外面下雨了，鞋全湿了")
    )
    assert label == "雨" and source == "chat" and hint


def test_resolve_weather_uses_api_then_fallback(tmp_path):
    plugin = MOD.create_plugin()
    host = FakeHost(returns={})
    plugin._set_context(build_context("org.orge-8.better-diary", rpc_call=host.rpc_call))
    raw = get_default_config(MOD.BetterDiaryConfig)
    raw["weather"]["city"] = "杭州"
    raw["diary"]["weather_fallback"] = "阴"
    plugin.set_plugin_config(raw)
    plugin._data_dir = lambda: tmp_path  # type: ignore[assignment]

    def rainy(request):
        return httpx.Response(200, json=_forecast_payload({TODAY_S: 61}))

    plugin._weather_store = _store(tmp_path, rainy, city="杭州", latitude=30.0, longitude=120.0)
    label, hint, source = asyncio.run(plugin._resolve_weather(TODAY_S, "甲: 随便聊聊"))
    assert (label, source) == ("雨", "open-meteo")
    assert "下雨" in hint  # 事实校验认「下雨」这种完整说法

    # 同一城市同一天已经查到过，换实例也应命中磁盘缓存（这是期望行为）
    plugin._weather_store = _store(tmp_path, rainy, city="杭州", latitude=30.0, longitude=120.0)
    assert asyncio.run(plugin._resolve_weather(TODAY_S, "甲: 随便聊聊"))[0] == "雨"

    # 降级路径要用**干净目录**：否则读到的是上一步刚写好的缓存，测不到失败分支
    empty_dir = tmp_path / "no-cache"
    empty_dir.mkdir()

    def broken(request):
        raise httpx.ConnectError("down")

    plugin._weather_store = _store(empty_dir, broken, city="杭州", latitude=30.0, longitude=120.0)
    label, hint, source = asyncio.run(plugin._resolve_weather(TODAY_S, "甲: 随便聊聊"))
    assert (label, hint, source) == ("阴", "", "fallback")


def test_diary_archive_records_weather_provenance(tmp_path):
    plugin = MOD.create_plugin()
    host = FakeHost(returns={})
    plugin._set_context(build_context("org.orge-8.better-diary", rpc_call=host.rpc_call))
    plugin.set_plugin_config(get_default_config(MOD.BetterDiaryConfig))
    plugin._data_dir = lambda: tmp_path  # type: ignore[assignment]
    plugin._save_diary(
        TODAY_S, f"{TODAY_S}，雨。\n今天挺安静。", {},
        weather={"label": "雨", "source": "open-meteo"},
    )
    saved = plugin._load_diaries()[TODAY_S]
    assert saved["weather"] == {"label": "雨", "source": "open-meteo"}
