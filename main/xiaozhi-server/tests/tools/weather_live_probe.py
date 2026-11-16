"""Live probe: Open-Meteo + get_weather, zonder mock en zonder API-sleutel.

Bewijst acceptatiecriterium 2 van t_9fe9586d: ``get_weather`` voor Zoetermeer
geeft een bruikbaar Nederlands rapport, zonder exception en zonder CJK.

Gebruik (in de container, met data/ gemount):
    python tests/tools/weather_live_probe.py
"""

import asyncio
import os
import re
import sys

import yaml

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from core.utils import weather  # noqa: E402

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class _FakeConn:
    """Alleen wat de tool van een connectie gebruikt."""

    def __init__(self, config, client_ip="95.98.34.31"):
        self.config = config
        self.client_ip = client_ip
        self.device_id = "weather-live-probe"


def load_config(path="/opt/xiaozhi-esp32-server/data/.config.yaml"):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


async def main():
    config = load_config()
    cfg = weather.get_config(config)
    print(f"config.get_weather = {cfg}")
    print(f"home_location = {weather.home_location(config)!r}")
    print(f"home_coordinates = {weather.home_coordinates(config)}")

    geo = await weather.geocode_location("Zoetermeer")
    print(f"geocode('Zoetermeer') = {geo}")

    report = await weather.build_report_for(config, "Zoetermeer", lang="nl")
    print("--- build_report_for('Zoetermeer') ---")
    print(report)
    assert report, "geen weerbericht"
    assert not CJK_RE.search(report), "CJK gevonden in het weerbericht"

    from plugins_func.functions.get_weather import get_weather

    response = await get_weather(
        _FakeConn(config), location="Zoetermeer", lang="nl"
    )
    tool_report = response.result or ""
    print("--- get_weather(location='Zoetermeer') ---")
    print(tool_report)
    assert tool_report, "lege tool-respons"
    assert not CJK_RE.search(tool_report), "CJK gevonden in de tool-respons"
    assert "Zoetermeer" in tool_report
    print(f"CJK-treffers: {len(CJK_RE.findall(tool_report))}")


if __name__ == "__main__":
    asyncio.run(main())
