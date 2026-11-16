"""Minimale websocket-simulator voor één echte RIG-tekstbeurt.

Er bestond geen sim-client in de repo; deze is geschreven voor t_9fe9586d om
acceptatiecriterium 3 te bewijzen: een echte eerste beurt moet
``data/.llm_warmup_firstturn.json`` schrijven met een gevulde ``<context>``-
regel (plaats + Nederlands weer). Het protocol is het gewone xiaozhi-pad:

    hello  ->  server stuurt welcome  ->  listen/detect met tekst  ->  LLM-beurt

Gebruik vanaf de host (poort 8001 -> container 8000):
    python tests/tools/ws_first_turn_sim.py --text "Hoe laat is het?"
of binnen de containernetwerk-namespace:
    python tests/tools/ws_first_turn_sim.py --url ws://127.0.0.1:8000/xiaozhi/v1/
"""

import argparse
import asyncio
import json
import time
import uuid

import websockets


def _hello_payload() -> dict:
    return {
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "features": {"mcp": False},
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60,
        },
    }


async def _connect(url: str, headers: dict):
    """Verbinden, ongeacht de websockets-versie (extra_headers vs additional_headers)."""
    try:
        return await websockets.connect(url, additional_headers=headers, open_timeout=10)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, open_timeout=10)


async def run(url: str, text: str, device_id: str, timeout: float) -> int:
    headers = {"device-id": device_id, "client-id": f"sim-{uuid.uuid4().hex[:8]}"}
    socket = await _connect(url, headers)
    print(f"[sim] verbonden met {url} als {device_id}")
    try:
        await socket.send(json.dumps(_hello_payload()))
        seen_welcome = False
        deadline = time.time() + timeout
        sent_listen = False
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=5)
            except asyncio.TimeoutError:
                if not seen_welcome and not sent_listen and time.time() > deadline - timeout + 10:
                    print("[sim] geen welcome ontvangen; toch een beurt proberen")
                    sent_listen = True
                    await _send_turn(socket, text)
                continue

            if isinstance(raw, bytes):
                continue
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                print(f"[sim] <- (tekst) {raw}")
                continue

            message_type = message.get("type")
            print(f"[sim] <- {message_type} {message.get('state', '')}".rstrip())

            if message_type == "hello":
                seen_welcome = True
                if not sent_listen:
                    sent_listen = True
                    await _send_turn(socket, text)
            elif message_type == "tts" and message.get("state") == "stop":
                print("[sim] beurt afgerond (tts stop)")
                return 0
            elif message_type == "stt" and message.get("state") == "stop":
                print("[sim] tekst ontvangen; wacht op LLM-antwoord")
        print("[sim] tijdslimiet bereikt")
        return 0
    finally:
        await socket.close()


async def _send_turn(socket, text: str) -> None:
    await socket.send(json.dumps({"type": "listen", "state": "start", "mode": "manual"}))
    await socket.send(json.dumps({"type": "listen", "state": "detect", "text": text}))
    print(f"[sim] -> tekstbeurt: {text!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:8001/xiaozhi/v1/", help="websocket-URL"
    )
    parser.add_argument("--text", default="Hoe laat is het?", help="tekst voor de beurt")
    parser.add_argument(
        "--device-id", default="sim-first-turn-t9fe9586d", help="device-id header"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="seconden")
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.text, args.device_id, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
