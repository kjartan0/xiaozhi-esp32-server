"""get_weather — weerbericht via Open-Meteo, Nederlands, zonder API-sleutel.

t_9fe9586d: de QWeather-koppeling is verwijderd. Die gebruikte de gedeelde
demo-sleutel van het upstream-project (``a861d0d5...``, sinds 2026-11 HTTP 401
op ``/v7/weather/now``) plus een China-only geocoder, en de fallback was een
HTML-scrape van de fxLink-pagina. Open-Meteo is keyless, levert WMO-codes en
een 7-daagse verwachting in de Nederlandse tijdzone.

Alle zichtbare tekst is Nederlands (t_c30a9895: geen CJK in prompt of antwoord).
"""

from typing import TYPE_CHECKING, Optional

from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from core.utils import weather as weather_utils
from core.utils.location import resolve_local_address

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

GET_WEATHER_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Vraagt het weer op voor een bepaalde plaats; de parameter is de plaatsnaam. "
            "Het weerbericht voor de komende 7 dagen op de eigen locatie staat al in de "
            "context; roep dit alleen aan als de gebruiker een andere plaats noemt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Plaatsnaam, bijvoorbeeld Zoetermeer",
                },
                "lang": {
                    "type": "string",
                    "description": "Taalcode voor het antwoord; de dienst antwoordt altijd in het Nederlands",
                },
            },
            "required": ["lang"],
        },
    },
}

WEATHER_CACHE_TTL = 600  # seconden; het weer verandert niet sneller


async def _resolve_target(conn, location: Optional[str] = None) -> str:
    """Plaats voor dit verzoek: expliciet genoemd, anders de eigen locatie."""
    explicit = (location or "").strip()
    if explicit:
        return explicit
    config = getattr(conn, "config", None) or {}
    return resolve_local_address(config, getattr(conn, "client_ip", None), logger)


@register_function("get_weather", GET_WEATHER_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def get_weather(
    conn: "ConnectionHandler", location: Optional[str] = None, lang: str = "nl"
):
    from core.utils.cache.manager import cache_manager, CacheType

    config = getattr(conn, "config", None) or {}
    target = await _resolve_target(conn, location)

    weather_cache_key = f"open_meteo:{target}:{lang}"
    cached_weather_report = cache_manager.get(CacheType.WEATHER, weather_cache_key)
    if cached_weather_report:
        return ActionResponse(Action.REQLLM, cached_weather_report, None)

    try:
        weather_report = await weather_utils.build_report_for(config, target, lang=lang)
    except Exception as exc:  # noqa: BLE001 - tool-call mag nooit crashen
        logger.bind(tag=TAG).error(f"weerbericht ophalen mislukt voor {target!r}: {exc}")
        weather_report = None

    if not weather_report:
        logger.bind(tag=TAG).warning(f"geen weerbericht voor {target!r}")
        return ActionResponse(
            Action.REQLLM,
            f"Ik kan het weer voor {target} nu niet vinden "
            f"(plaats onbekend of weerdienst niet bereikbaar), probeer het later opnieuw",
            None,
        )

    cache_manager.set(
        CacheType.WEATHER, weather_cache_key, weather_report, ttl=WEATHER_CACHE_TTL
    )
    return ActionResponse(Action.REQLLM, weather_report, None)
