"""Weerbericht via Open-Meteo (zonder API-sleutel), Nederlands.

Vervangt de QWeather-koppeling (t_9fe9586d): die gebruikte de gedeelde
demo-sleutel van het upstream-project (sinds 2026-11 HTTP 401) en werkte
alleen betrouwbaar voor Chinese plaatsen. Open-Meteo vraagt geen sleutel,
levert WMO-weercodes en een 7-daagse verwachting in de Nederlandse tijdzone.

Alle teksten hier zijn Nederlands en bevatten geen CJK (t_c30a9895): deze
strings belanden letterlijk in de systeemprompt en in het gesproken antwoord.

Twee gebruikers:
- ``core/utils/prompt_manager.py`` -> korte samenvatting in ``<context>``;
- ``plugins_func/functions/get_weather.py`` -> antwoord op een tool-call.

Beide gebruiken dezelfde tekstvorm zodat het model één consistent beeld heeft:
één regel "nu" en daaronder zeven korte dagregels (ma/di/wo/do/vr/za/zo).
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Vaste thuisbasis van deze installatie; alleen gebruikt als de config het veld mist.
DEFAULT_HOME_LOCATION = "Zoetermeer, Nederland"
DEFAULT_HOME_LAT = 52.0575
DEFAULT_HOME_LON = 4.4931
DEFAULT_FORECAST_DAYS = 7
DEFAULT_TIMEZONE = "Europe/Amsterdam"

# Open-Meteo levert WMO-codes (0-99); de oude QWeather-codes (100+) zijn hier
# NIET bruikbaar. Zie https://open-meteo.com/en/docs (WMO Weather interpretation codes).
WMO_CODE_MAP: Dict[int, str] = {
    0: "helder",
    1: "overwegend helder",
    2: "half bewolkt",
    3: "bewolkt",
    45: "mist",
    48: "mist met rijp",
    51: "lichte motregen",
    53: "motregen",
    55: "zware motregen",
    56: "lichte ijzelmotregen",
    57: "ijzelmotregen",
    61: "lichte regen",
    63: "regen",
    65: "zware regen",
    66: "lichte ijzelregen",
    67: "ijzelregen",
    71: "lichte sneeuw",
    73: "sneeuw",
    75: "zware sneeuw",
    77: "sneeuwkorrels",
    80: "lichte buien",
    81: "buien",
    82: "zware buien",
    85: "lichte sneeuwbuien",
    86: "zware sneeuwbuien",
    95: "onweer",
    96: "onweer met hagel",
    99: "zware onweersbui met hagel",
}

# Grove terugval per range, voor codes die Open-Meteo in de toekomst toevoegt.
_WMO_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (51, 57, "motregen"),
    (61, 67, "regen"),
    (71, 77, "sneeuw"),
    (80, 82, "buien"),
    (85, 86, "sneeuwbuien"),
    (95, 99, "onweer"),
)

DUTCH_WEEKDAYS = ("ma", "di", "wo", "do", "vr", "za", "zo")


def describe_wmo(code: Any) -> str:
    """Vertaal een WMO-weercode naar een Nederlandse omschrijving."""
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "onbekend"
    if value in WMO_CODE_MAP:
        return WMO_CODE_MAP[value]
    for low, high, text in _WMO_RANGES:
        if low <= value <= high:
            return text
    return "onbekend"


def get_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Het ``plugins.get_weather`` blok uit de serverconfig."""
    plugins = (config or {}).get("plugins") or {}
    return plugins.get("get_weather") or {}


def home_location(config: Optional[Dict[str, Any]]) -> str:
    """Naam van de thuisbasis; nooit leeg."""
    name = (get_config(config).get("home_location") or "").strip()
    return name or DEFAULT_HOME_LOCATION


def city_only(location: str) -> str:
    """``Zoetermeer, Nederland`` -> ``Zoetermeer`` (voor de zin "Weer in ...")."""
    return (location or "").split(",")[0].strip() or DEFAULT_HOME_LOCATION


def home_coordinates(config: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """Breedte/lengte van de thuisbasis uit de config, met vaste terugval."""
    cfg = get_config(config)
    lat = _to_float(cfg.get("home_lat"))
    lon = _to_float(cfg.get("home_lon"))
    if lat is None or lon is None:
        return DEFAULT_HOME_LAT, DEFAULT_HOME_LON
    return lat, lon


def is_home_location(config: Optional[Dict[str, Any]], location: Optional[str]) -> bool:
    """Is ``location`` (leeg of gelijk aan de thuisbasis) de eigen locatie?"""
    if not location or not str(location).strip():
        return True
    return city_only(str(location)).lower() == city_only(home_location(config)).lower()


def forecast_days(config: Optional[Dict[str, Any]]) -> int:
    try:
        days = int(get_config(config).get("forecast_days") or DEFAULT_FORECAST_DAYS)
    except (TypeError, ValueError):
        return DEFAULT_FORECAST_DAYS
    return max(1, min(days, 16))


def timezone_name(config: Optional[Dict[str, Any]]) -> str:
    return (get_config(config).get("timezone") or DEFAULT_TIMEZONE).strip()


def cache_key(location: Optional[str], lang: str = "nl") -> str:
    """Cachetabel-sleutel voor een weerbericht (prompt en tool delen die)."""
    return f"open_meteo:{(location or '').strip().lower()}:{lang}"


def _to_float(value: Any) -> Optional[float]:
    """Zet een JSON-waarde om naar float; ``None`` als dat niet kan."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_temp(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "?"
    return f"{round(number)}°C"


def _fmt_mm(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "?"
    # Nederlandse decimale komma; het model spreekt dit natuurlijk uit.
    return f"{number:.1f}".replace(".", ",")


def _weekday_abbr(iso_date: str) -> str:
    try:
        day = datetime.date.fromisoformat(str(iso_date))
    except ValueError:
        return ""
    return DUTCH_WEEKDAYS[day.weekday()]


def build_weather_report(
    city: str, payload: Dict[str, Any], include_city: bool = True
) -> str:
    """Compacte Nederlandse tekst: nu-regel plus korte dagregels.

    ``payload`` is het Open-Meteo ``/v1/forecast`` antwoord. Onbekende velden
    worden overgeslagen in plaats van als "None" in de prompt te belanden.

    ``include_city=False`` laat de plaatsnaam weg: in de promptcontext staat
    de plaats al als ``{{local_address}}`` vlak vóór het weerbericht.
    """
    payload = payload or {}
    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    name = (city or DEFAULT_HOME_LOCATION).strip() or DEFAULT_HOME_LOCATION

    now_bits: List[str] = []
    if current.get("temperature_2m") is not None:
        now_bits.append(_fmt_temp(current.get("temperature_2m")))
    now_bits.append(describe_wmo(current.get("weather_code")))
    if current.get("precipitation") is not None:
        precip = current.get("precipitation")
        try:
            is_dry = float(precip) <= 0.0
        except (TypeError, ValueError):
            is_dry = False
        now_bits.append("droog" if is_dry else f"{_fmt_mm(precip)} mm neerslag")
    if current.get("relative_humidity_2m") is not None:
        try:
            now_bits.append(f"luchtvochtigheid {round(float(current['relative_humidity_2m']))}%")
        except (TypeError, ValueError):
            pass

    lines = [
        (f"Weer in {name}: nu " if include_city else "nu ")
        + ", ".join(bit for bit in now_bits if bit)
        + "."
    ]

    dates: List[Any] = list(daily.get("time") or [])
    codes: List[Any] = list(daily.get("weather_code") or [])
    highs: List[Any] = list(daily.get("temperature_2m_max") or [])
    lows: List[Any] = list(daily.get("temperature_2m_min") or [])
    chances: List[Any] = list(daily.get("precipitation_probability_max") or [])

    for index, iso_date in enumerate(dates):
        weekday = _weekday_abbr(iso_date)
        if not weekday:
            continue
        condition = describe_wmo(codes[index] if index < len(codes) else None)
        high = _fmt_temp(highs[index]) if index < len(highs) else "?"
        low = _fmt_temp(lows[index]) if index < len(lows) else "?"
        line = f"{weekday} {low}/{high} {condition}"
        if index < len(chances) and chances[index] is not None:
            try:
                line += f", {round(float(chances[index]))}% kans op neerslag"
            except (TypeError, ValueError):
                pass
        lines.append(line)

    return "\n".join(lines)


async def geocode_location(
    name: str, language: str = "nl"
) -> Optional[Dict[str, Any]]:
    """Zoek coördinaten bij een plaatsnaam (Open-Meteo geocoding, keyless)."""
    query = city_only(name)
    if not query:
        return None
    params = {"name": query, "count": 1, "language": language, "format": "json"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
            response = await client.get(GEOCODE_URL, params=params)
        if response.status_code != 200:
            logger.bind(tag=TAG).warning(
                f"geocoding mislukt voor {query!r}: HTTP {response.status_code}"
            )
            return None
        results = (response.json() or {}).get("results") or []
    except Exception as exc:  # noqa: BLE001 - netwerkfout mag de prompt niet breken
        logger.bind(tag=TAG).warning(f"geocoding mislukt voor {query!r}: {exc}")
        return None
    if not results:
        return None
    hit = results[0]
    latitude = _to_float(hit.get("latitude"))
    longitude = _to_float(hit.get("longitude"))
    if latitude is None or longitude is None:
        return None
    return {
        "name": hit.get("name") or query,
        "latitude": latitude,
        "longitude": longitude,
        "country": hit.get("country") or "",
    }


async def fetch_forecast(
    latitude: float,
    longitude: float,
    days: int = DEFAULT_FORECAST_DAYS,
    timezone: str = DEFAULT_TIMEZONE,
) -> Optional[Dict[str, Any]]:
    """Haal het actuele weer plus de dagverwachting op bij Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code,precipitation,relative_humidity_2m",
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max"
        ),
        "timezone": timezone,
        "forecast_days": max(1, min(int(days), 16)),
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            response = await client.get(FORECAST_URL, params=params)
        if response.status_code != 200:
            logger.bind(tag=TAG).warning(
                f"Open-Meteo forecast mislukt: HTTP {response.status_code}"
            )
            return None
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - netwerkfout mag de prompt niet breken
        logger.bind(tag=TAG).warning(f"Open-Meteo forecast mislukt: {exc}")
        return None
    return data if isinstance(data, dict) else None


async def build_report_for(
    config: Optional[Dict[str, Any]],
    location: Optional[str] = None,
    lang: str = "nl",
    include_city: bool = True,
) -> Optional[str]:
    """Volledig rapport voor ``location``; ``None`` als de plaats onbekend is.

    Zonder ``location`` (of met de eigen plaats) worden de coördinaten uit de
    config gebruikt: geen geocoding-verzoek en dus geen extra faalpunt.
    """
    days = forecast_days(config)
    tz = timezone_name(config)
    target = (location or "").strip() or home_location(config)

    if is_home_location(config, target):
        latitude, longitude = home_coordinates(config)
        display_name = city_only(home_location(config))
    else:
        geo = await geocode_location(target, language="nl")
        if not geo:
            return None
        latitude, longitude = geo["latitude"], geo["longitude"]
        display_name = city_only(geo["name"])

    payload = await fetch_forecast(latitude, longitude, days=days, timezone=tz)
    if not payload:
        return None
    return build_weather_report(display_name, payload, include_city=include_city)
