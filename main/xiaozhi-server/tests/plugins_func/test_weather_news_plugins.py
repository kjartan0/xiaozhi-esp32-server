"""Tests for the weather and newsnow plugin helpers.

The two plugins talk to external APIs, so the HTTP layer is replaced with a
fake client and every request is recorded. Nothing here touches the network.
"""
import httpx
import pytest

from plugins_func.functions import get_news_from_newsnow as news_module
from plugins_func.functions import get_weather as weather_module


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", json_error=None):
        self.status_code = status_code
        self.payload = payload
        self.text = text
        self.json_error = json_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeAsyncClient:
    """Stands in for httpx.AsyncClient and records what was requested."""

    requests = []
    responder = None
    init_kwargs = {}

    def __init__(self, *args, **kwargs):
        FakeAsyncClient.init_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, params=None, headers=None):
        FakeAsyncClient.requests.append({"url": url, "params": params})
        return FakeAsyncClient.responder(url, params)


@pytest.fixture
def fake_http(monkeypatch):
    """Replace httpx.AsyncClient with the recording fake."""
    FakeAsyncClient.requests = []
    FakeAsyncClient.responder = lambda url, params: FakeResponse(payload={"code": "200"})
    FakeAsyncClient.init_kwargs = {}
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return FakeAsyncClient


class FakeConn:
    def __init__(self, plugins=None):
        self.config = {"plugins": plugins or {}}
        self.client_ip = None


# ---------------------------------------------------------------------------
# Weer (Open-Meteo, keyless — t_9fe9586d)
#
# De QWeather-tests zijn vervallen met de provider: die functies bestaan niet
# meer en de gedeelde demo-sleutel gaf HTTP 401.
# ---------------------------------------------------------------------------

HOME_WEATHER_CONFIG = {
    "provider": "open_meteo",
    "home_location": "Zoetermeer, Nederland",
    "home_lat": 52.0575,
    "home_lon": 4.4931,
    "forecast_days": 7,
    "timezone": "Europe/Amsterdam",
}

OPEN_METEO_RESPONSE = {
    "current": {
        "temperature_2m": 10.2,
        "weather_code": 3,
        "precipitation": 0.0,
        "relative_humidity_2m": 65,
    },
    "daily": {
        "time": ["2026-11-16", "2026-11-17"],
        "weather_code": [61, 0],
        "temperature_2m_max": [10.4, 12.0],
        "temperature_2m_min": [4.2, 5.0],
        "precipitation_probability_max": [80, 5],
    },
}


@pytest.fixture(autouse=True)
def clear_weather_cache():
    from core.utils.cache.config import CacheType
    from core.utils.cache.manager import cache_manager

    cache_manager.clear(CacheType.WEATHER)
    yield
    cache_manager.clear(CacheType.WEATHER)


def _weather_conn():
    return FakeConn({"get_weather": dict(HOME_WEATHER_CONFIG)})


async def test_forecast_uses_open_meteo_without_api_key(fake_http):
    """Eén forecast-verzoek, geen sleutel, geen geocoding voor de eigen plaats."""
    fake_http.responder = lambda url, params: FakeResponse(payload=OPEN_METEO_RESPONSE)

    response = await weather_module.get_weather(
        _weather_conn(), location="Zoetermeer", lang="nl"
    )
    report = response.result or ""

    assert "Weer in Zoetermeer: nu 10°C, bewolkt, droog" in report
    assert "ma 4°C/10°C lichte regen, 80% kans op neerslag" in report
    assert len(fake_http.requests) == 1
    request = fake_http.requests[0]
    assert request["url"] == "https://api.open-meteo.com/v1/forecast"
    assert "key" not in (request["params"] or {})
    assert request["params"]["timezone"] == "Europe/Amsterdam"
    assert request["params"]["forecast_days"] == 7


async def test_unknown_place_returns_dutch_error_without_crashing(fake_http):
    fake_http.responder = lambda url, params: FakeResponse(payload={"results": []})

    response = await weather_module.get_weather(
        _weather_conn(), location="Onbekenddorp", lang="nl"
    )
    report = response.result or ""

    assert "Ik kan het weer voor Onbekenddorp nu niet vinden" in report
    assert fake_http.requests[0]["url"] == "https://geocoding-api.open-meteo.com/v1/search"
    assert fake_http.requests[0]["params"]["name"] == "Onbekenddorp"


async def test_forecast_http_error_returns_dutch_error(fake_http):
    fake_http.responder = lambda url, params: FakeResponse(
        status_code=503, payload={"error": True}
    )

    response = await weather_module.get_weather(
        _weather_conn(), location="Zoetermeer", lang="nl"
    )
    report = response.result or ""

    assert "probeer het later opnieuw" in report


async def test_other_city_is_geocoded_and_named(fake_http):
    """Een andere plaats dan de eigen locatie wordt gegeocodeerd en genoemd."""

    def responder(url, params):
        if "geocoding" in url:
            return FakeResponse(
                payload={
                    "results": [
                        {
                            "name": "Utrecht",
                            "latitude": 52.09,
                            "longitude": 5.12,
                            "country": "Nederland",
                        }
                    ]
                }
            )
        return FakeResponse(payload=OPEN_METEO_RESPONSE)

    fake_http.responder = responder

    response = await weather_module.get_weather(
        _weather_conn(), location="Utrecht", lang="nl"
    )
    report = response.result or ""

    assert report.startswith("Weer in Utrecht: nu 10°C")
    assert fake_http.requests[1]["params"]["latitude"] == 52.09
    assert fake_http.requests[1]["params"]["longitude"] == 5.12


# ---------------------------------------------------------------------------
# 新闻接口
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configured_url, expected_url, expected_params",
    [
        (None, "https://newsnow.busiyi.world/api/s", {"id": "thepaper"}),
        (
            "https://newsnow.busiyi.world/api/s?id=thepaper",
            "https://newsnow.busiyi.world/api/s",
            {"id": "thepaper"},
        ),
        (
            "https://newsnow.busiyi.world/api/s?id={source}",
            "https://newsnow.busiyi.world/api/s?id=thepaper",
            None,
        ),
        (
            "https://newsnow.busiyi.world/api/s",
            "https://newsnow.busiyi.world/api/s",
            {"id": "thepaper"},
        ),
        (
            "https://newsnow.busiyi.world/api/s?id=",
            "https://newsnow.busiyi.world/api/s",
            {"id": "thepaper"},
        ),
        (
            "https://newsnow.busiyi.world/api/s?id=thepaper&limit=5",
            "https://newsnow.busiyi.world/api/s",
            {"id": "thepaper", "limit": "5"},
        ),
    ],
)
def test_news_api_url_forms(configured_url, expected_url, expected_params):
    url, params = news_module._build_news_api_url("thepaper", configured_url)

    assert url == expected_url
    assert params == expected_params


async def test_news_api_url_forms_are_actually_requested(fake_http):
    """配置里写了 id 也不能拼出脏串。"""
    fake_http.responder = lambda url, params: FakeResponse(
        payload={"items": [{"id": "1", "title": "标题"}]}
    )
    conn = FakeConn({"get_news_from_newsnow": {"url": "https://newsnow.busiyi.world/api/s?id=thepaper"}})

    await news_module.fetch_news_from_api(conn, "thepaper")

    assert fake_http.requests[0]["url"] == "https://newsnow.busiyi.world/api/s"
    assert fake_http.requests[0]["params"] == {"id": "thepaper"}


async def test_news_items_must_be_a_list(fake_http):
    fake_http.responder = lambda url, params: FakeResponse(payload={"items": "oops"})

    assert await news_module.fetch_news_from_api(FakeConn(), "thepaper") == []


async def test_news_items_without_title_are_filtered_out(fake_http):
    fake_http.responder = lambda url, params: FakeResponse(
        payload={"items": [{"id": "1"}, {"id": "2", "title": ""}]}
    )

    assert await news_module.fetch_news_from_api(FakeConn(), "thepaper") == []


async def test_news_items_keep_valid_entries(fake_http):
    fake_http.responder = lambda url, params: FakeResponse(
        payload={"items": [{"id": "1", "title": ""}, {"id": "2", "title": "有效标题"}]}
    )

    items = await news_module.fetch_news_from_api(FakeConn(), "thepaper")

    assert items == [{"id": "2", "title": "有效标题"}]
