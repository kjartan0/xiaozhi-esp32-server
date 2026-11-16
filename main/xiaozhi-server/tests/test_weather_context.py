"""Locatie en weer in de systeemprompt (t_9fe9586d).

Achtergrond: de promptcontext renderde ``location  | local upcoming weather ``
omdat de IP-geocoder voor een Nederlands IP een lege city teruggaf en de
QWeather-demo-sleutel HTTP 401 gaf. Deze tests dekken de drie invarianten:

1. een lege geocoder-city levert nooit een lege ``local_address`` op;
2. een Open-Meteo-payload levert een Nederlands rapport met echte waarden;
3. de gerenderde prompt bevat nul CJK-tekens (t_c30a9895).

Er wordt geen netwerk gebruikt: Open-Meteo wordt vervangen door een vaste
payload en de geocoder door de echte pconline-responsvorm.
"""

import asyncio
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.utils import weather  # noqa: E402
from core.utils.location import resolve_local_address  # noqa: E402
from core.utils.prompt_manager import PromptManager  # noqa: E402
from core.utils.cache.config import CacheType  # noqa: E402
from core.utils.cache.manager import cache_manager  # noqa: E402

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

CLIENT_IP = "95.98.34.31"

# Echte responsvorm van whois.pconline.com.cn voor een Nederlands IP (HTTP 200).
PCONLINE_EMPTY_CITY = {
    "ip": CLIENT_IP,
    "pro": "",
    "proCode": "999999",
    "city": "",
    "cityCode": "0",
    "region": "",
    "addr": "",
    "regionNames": "",
    "err": "noprovince",
}

# Vaste Open-Meteo-payload (7 dagen, vanaf maandag 2026-11-16).
OPEN_METEO_PAYLOAD: Dict[str, Any] = {
    "latitude": 52.06,
    "longitude": 4.49,
    "timezone": "Europe/Amsterdam",
    "current": {
        "time": "2026-11-16T14:00",
        "temperature_2m": 8.6,
        "weather_code": 61,
        "precipitation": 0.4,
        "relative_humidity_2m": 87,
    },
    "daily": {
        "time": [
            "2026-11-16",
            "2026-11-17",
            "2026-11-18",
            "2026-11-19",
            "2026-11-20",
            "2026-11-21",
            "2026-11-22",
        ],
        "weather_code": [61, 3, 0, 2, 45, 63, 80],
        "temperature_2m_max": [11.4, 10.2, 9.1, 8.4, 7.9, 9.6, 10.8],
        "temperature_2m_min": [5.6, 4.9, 3.2, 2.8, 1.9, 4.1, 6.2],
        "precipitation_sum": [3.2, 0.4, 0.0, 0.0, 0.0, 1.8, 2.4],
        "precipitation_probability_max": [80, 20, 5, 10, 15, 60, 70],
    },
}


def make_config(**weather_overrides) -> Dict[str, Any]:
    weather_cfg = {
        "provider": "open_meteo",
        "home_location": "Zoetermeer, Nederland",
        "home_lat": 52.0575,
        "home_lon": 4.4931,
        "forecast_days": 7,
        "timezone": "Europe/Amsterdam",
    }
    weather_cfg.update(weather_overrides)
    return {
        "prompt_template": str(APP_DIR / "agent-base-prompt.txt"),
        "selected_module": {"TTS": "EdgeTTS"},
        "TTS": {"EdgeTTS": {"language": "Nederlands"}},
        "plugins": {"get_weather": weather_cfg},
    }


@pytest.fixture(autouse=True)
def _clean_caches():
    """De cache is proces-globaal; houd tests onafhankelijk van elkaar."""
    for cache_type in (CacheType.LOCATION, CacheType.WEATHER, CacheType.DEVICE_PROMPT):
        cache_manager.clear(cache_type)
    yield
    for cache_type in (CacheType.LOCATION, CacheType.WEATHER, CacheType.DEVICE_PROMPT):
        cache_manager.clear(cache_type)


class _FakeConn:
    """Net genoeg ConnectionHandler voor update_context_info."""

    def __init__(self, config: Dict[str, Any], loop, client_ip: str = CLIENT_IP):
        self.config = config
        self.loop = loop
        self.client_ip = client_ip
        self.device_id = "test-device"


def test_empty_geocoder_city_falls_back_to_home_location(monkeypatch):
    """De echte pconline-responsvorm ({"city": ""}) mag de prompt niet leegmaken."""
    import core.utils.util as util_module

    monkeypatch.setattr(
        util_module,
        "get_ip_info",
        lambda ip_addr, logger=None: dict(PCONLINE_EMPTY_CITY),
    )

    location = resolve_local_address(make_config(), CLIENT_IP)

    assert location == "Zoetermeer, Nederland"
    assert location.strip()
    assert not CJK_RE.search(location)


def test_usable_ip_city_overrides_home_and_cjk_city_is_rejected(monkeypatch):
    """Een bruikbare IP-city wint; een Chinese city mag de prompt niet vervuilen."""
    import core.utils.util as util_module

    monkeypatch.setattr(
        util_module,
        "get_ip_info",
        lambda ip_addr, logger=None: {"ip": ip_addr, "city": "Delft"},
    )
    assert resolve_local_address(make_config(), CLIENT_IP) == "Delft"

    monkeypatch.setattr(
        util_module,
        "get_ip_info",
        lambda ip_addr, logger=None: {"ip": ip_addr, "city": "杭州"},
    )
    assert resolve_local_address(make_config(), CLIENT_IP) == "Zoetermeer, Nederland"


def test_open_meteo_payload_renders_dutch_report():
    report = weather.build_weather_report("Zoetermeer", OPEN_METEO_PAYLOAD)

    # Actueel: temperatuur, Nederlandse conditie, neerslag met decimale komma.
    assert report.splitlines()[0] == (
        "Weer in Zoetermeer: nu 9°C, lichte regen, 0,4 mm neerslag, "
        "luchtvochtigheid 87%."
    )
    # Zeven korte dagregels met Nederlandse weekdagen, min/max en kans.
    assert len(report.splitlines()) == 8
    assert "ma 6°C/11°C lichte regen, 80% kans op neerslag" in report
    assert "wo 3°C/9°C helder, 5% kans op neerslag" in report
    assert "za 4°C/10°C regen, 60% kans op neerslag" in report
    for abbreviation in ("ma", "di", "wo", "do", "vr", "za", "zo"):
        assert f"\n{abbreviation} " in report
    assert not CJK_RE.search(report)


def test_wmo_codes_are_dutch_not_qweather_codes():
    """Open-Meteo levert WMO-codes (0-99); de QWeather-map (100+) is vervallen."""
    assert weather.describe_wmo(0) == "helder"
    assert weather.describe_wmo(3) == "bewolkt"
    assert weather.describe_wmo(45) == "mist"
    assert weather.describe_wmo(63) == "regen"
    assert weather.describe_wmo(71) == "lichte sneeuw"
    assert weather.describe_wmo(95) == "onweer"
    assert weather.describe_wmo(None) == "onbekend"
    for description in weather.WMO_CODE_MAP.values():
        assert not CJK_RE.search(description)


def test_weather_report_drops_city_for_prompt_context():
    """In de prompt staat de plaats al als {{local_address}}; geen dubbeling."""
    report = weather.build_weather_report(
        "Zoetermeer", OPEN_METEO_PAYLOAD, include_city=False
    )
    assert report.startswith("nu 9°C, lichte regen")
    assert "Weer in" not in report


def test_get_weather_tool_returns_dutch_report(monkeypatch):
    """De tool zelf (zonder netwerk) levert een bruikbaar Nederlands rapport."""
    from plugins_func.functions import get_weather as tool_module

    async def _fake_forecast(latitude, longitude, days=7, timezone="Europe/Amsterdam"):
        assert latitude == pytest.approx(52.0575, abs=1e-3)
        assert longitude == pytest.approx(4.4931, abs=1e-3)
        assert days == 7
        assert timezone == "Europe/Amsterdam"
        return dict(OPEN_METEO_PAYLOAD)

    monkeypatch.setattr(tool_module.weather_utils, "fetch_forecast", _fake_forecast)

    conn = _FakeConn(make_config(), loop=None)
    response = asyncio.run(
        tool_module.get_weather(conn, location="Zoetermeer", lang="nl")
    )
    report = response.result or ""

    assert report.startswith("Weer in Zoetermeer: nu 9°C, lichte regen")
    assert "0,4 mm neerslag" in report
    assert not CJK_RE.search(report)

    # Tweede aanroep komt uit de cache en doet geen nieuw verzoek.
    async def _boom(*args, **kwargs):
        raise AssertionError("fetch_forecast had niet opnieuw aangeroepen mogen worden")

    monkeypatch.setattr(tool_module.weather_utils, "fetch_forecast", _boom)
    cached = asyncio.run(
        tool_module.get_weather(conn, location="Zoetermeer", lang="nl")
    )
    assert cached.result == report


def test_deployed_config_uses_open_meteo_and_no_demo_key():
    """Geen terugkeer van de QWeather-demo-sleutel of China-host in de config."""
    config_path = APP_DIR / "data" / ".config.yaml"
    if not config_path.exists():
        pytest.skip("data/.config.yaml is niet gemount in deze omgeving")

    import yaml

    with open(config_path, encoding="utf-8") as handle:
        deployed = yaml.safe_load(handle)
    weather_cfg = weather.get_config(deployed)

    assert weather_cfg.get("provider") == "open_meteo"
    assert "api_key" not in weather_cfg
    assert "api_host" not in weather_cfg
    raw = config_path.read_text(encoding="utf-8")
    assert "a861d0d5e7bf4ee1a83d9a9e4f96d4da" not in raw
    assert "qweatherapi.com" not in raw
    assert weather_cfg.get("home_location")
    assert weather_cfg.get("home_lat") and weather_cfg.get("home_lon")


def test_enhanced_prompt_has_real_weather_and_zero_cjk(monkeypatch):
    """Volledige promptrender: echte plaats, echte temperaturen, nul CJK."""
    import core.utils.util as util_module

    monkeypatch.setattr(
        util_module,
        "get_ip_info",
        lambda ip_addr, logger=None: dict(PCONLINE_EMPTY_CITY),
    )

    async def _fake_report(config, location=None, lang="nl", include_city=True):
        return weather.build_weather_report(
            weather.city_only(location or ""), OPEN_METEO_PAYLOAD,
            include_city=include_city,
        )

    monkeypatch.setattr(weather, "build_report_for", _fake_report)

    config = make_config()
    prompt_manager = PromptManager(config)

    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    try:
        conn = _FakeConn(config, loop=loop)
        prompt_manager.context_provider.fetch_all = lambda device_id: ""
        prompt_manager.update_context_info(conn, CLIENT_IP)
        rendered = prompt_manager.build_enhanced_prompt(
            "Je bent RIG.", "test-device", CLIENT_IP
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(timeout=5)

    context_line = next(
        line for line in rendered.splitlines() if "Real-time info" in line
    )
    assert "location Zoetermeer, Nederland" in context_line
    assert "local upcoming weather" not in context_line
    assert "nu 9°C, lichte regen, 0,4 mm neerslag" in context_line
    assert "ma 6°C/11°C lichte regen" in rendered
    assert CJK_RE.findall(rendered) == []
