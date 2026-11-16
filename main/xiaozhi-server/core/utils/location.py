"""Bepaal de plaats die in de systeemprompt komt (nooit leeg).

Aanleiding (t_9fe9586d): de oude route leunde volledig op de IP-geocoder
``whois.pconline.com.cn``. Die is China-only en antwoordt voor een Nederlandse
IP met HTTP 200 en ``{"city": "", "err": "noprovince"}``. De sleutel bestond
dus wel, maar de waarde was leeg: ``build_enhanced_prompt`` renderde een lege
``location`` en omdat de locatie falsy was, werd ook het weer overgeslagen.

Regel: de geconfigureerde thuisbasis is de bron van waarheid. Een IP-city mag
die alleen overschrijven als hij *bruikbaar* is (niet leeg en geen CJK), zodat
een verkeerd antwoord van de geocoder nooit een Chinese plaats in de
Nederlandse prompt kan zetten.
"""

import re
from typing import Any, Dict, Optional

from config.logger import setup_logging

from core.utils.weather import home_location

TAG = __name__

# Alles in het CJK-blok is verboden in prompt- en toolteksten (t_c30a9895).
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def is_usable_city(city: Any) -> bool:
    """Een geocoder-antwoord mag alleen gebruikt worden als het echt een stad is."""
    text = str(city or "").strip()
    if not text:
        return False
    if _CJK_RE.search(text):
        return False
    return True


def resolve_local_address(
    config: Optional[Dict[str, Any]],
    client_ip: Optional[str] = None,
    logger=None,
) -> str:
    """Plaats voor de promptcontext; valt altijd terug op de thuisbasis."""
    fallback = home_location(config)
    if not client_ip:
        return fallback

    log = logger or setup_logging()
    try:
        # Lazy import: core.utils.util is zwaar en importeert de cachemodule.
        from core.utils.util import get_ip_info

        ip_info = get_ip_info(client_ip, log) or {}
        city = ip_info.get("city")
        if is_usable_city(city):
            return str(city).strip()
        log.bind(tag=TAG).debug(
            f"IP-geocoder gaf geen bruikbare plaats voor {client_ip!r} "
            f"(city={city!r}); thuisbasis {fallback!r} gebruikt"
        )
    except Exception as exc:  # noqa: BLE001 - nooit de prompt laten breken
        log.bind(tag=TAG).warning(
            f"IP-geocoder mislukt voor {client_ip!r}: {exc}; thuisbasis {fallback!r} gebruikt"
        )
    return fallback
