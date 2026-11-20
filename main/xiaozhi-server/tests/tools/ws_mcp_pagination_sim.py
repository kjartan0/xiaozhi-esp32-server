"""Websocket-simulator van een RIG die zijn MCP-tools **gepagineerd** aanbiedt.

Geschreven voor t_a2a3bf8a. De echte arm stuurt ~32 device-tools in twee
``tools/list``-pagina's (de eerste pagina eindigt met ``nextCursor``); de server
moet de lijst pas publiceren als de laatste pagina binnen is. Deze sim repliceert
dat protocol, doet meteen een tekstbeurt ("Kijk naar mij") en logt wat de server
terugstuurt, zodat je op de server kunt zien:

  * ``首轮请求前缀 … tools=N`` — moet 40 zijn (7 plugins + 32 device + direct_answer),
    nooit een gedeeltelijke lijst;
  * ``data/.llm_warmup_firstturn.json`` — moet ``self_arm_look_at_person`` bevatten;
  * of de server een ``tools/call`` voor ``self.arm.look_at_person`` stuurt in
    plaats van "工具 … 不存在" te antwoorden.

Gebruik (host, poort 8001 -> container 8000):
    python tests/tools/ws_mcp_pagination_sim.py --text "Kijk naar mij"
Trage pagina's (het echte incident: pagina 2 komt seconden later):
    python tests/tools/ws_mcp_pagination_sim.py --page2-delay 3.0
"""

import argparse
import asyncio
import json
import time
import uuid

import websockets

# De echte toolnamen van de RIG-arm (sanitized zoals de server ze ziet), in de
# volgorde waarin het device ze aanbiedt. Pagina 1 eindigde in de logs met
# nextCursor "self.arm.wag_tail", dus de split ligt na wag_tail.
DEVICE_TOOLS_PAGE_1 = [
    "self.get_device_status",
    "self.audio_speaker.set_volume",
    "self.camera.take_photo",
    "self.camera.preview",
    "self.set_power_mode",
    "self.power_off",
    "self.set_language",
    "self.dog.calibrate",
    "self.arm.teach_start",
    "self.arm.teach_stop",
    "self.arm.teach_play",
    "self.arm.teach_cancel",
    "self.arm.teach_status",
    "self.arm.idle_motion",
    "self.arm.stretch",
    "self.arm.apology",
    "self.arm.shy",
    "self.arm.coquetry",
    "self.arm.bow",
    "self.arm.feign_death",
    "self.arm.wag_tail",
]
DEVICE_TOOLS_PAGE_2 = [
    "self.arm.node",
    "self.arm.shake",
    "self.arm.x",
    "self.arm.y",
    "self.arm.z",
    "self.arm.yaw",
    "self.arm.pitch",
    "self.arm.look_at_person",
    "self.arm.roll",
    "self.laser.control",
    "self.ble.remote_control",
]


def _tool(name: str) -> dict:
    return {
        "name": name,
        "description": f"{name} (sim)",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    }


def _hello_payload() -> dict:
    return {
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "features": {"mcp": True},
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60,
        },
    }


async def _connect(url: str, headers: dict):
    try:
        return await websockets.connect(url, additional_headers=headers, open_timeout=10)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, open_timeout=10)


def _stamp() -> str:
    return f"{time.monotonic() % 1000:7.3f}s"


class Sim:
    def __init__(self, socket, text, page1_delay, page2_delay, turn_delay, linger):
        self.socket = socket
        self.text = text
        self.page1_delay = page1_delay
        self.page2_delay = page2_delay
        self.turn_delay = turn_delay
        self.linger = linger
        self.started = time.monotonic()
        self.tool_calls = []
        self.pages_sent = 0
        self.turn_sent = False
        self.done = asyncio.Event()

    async def log(self, message: str) -> None:
        print(f"[sim {_stamp()}] {message}", flush=True)

    async def send(self, payload: dict) -> None:
        await self.socket.send(json.dumps(payload))

    async def _send_mcp(self, msg_id, result) -> None:
        await self.send(
            {"type": "mcp", "payload": {"jsonrpc": "2.0", "id": msg_id, "result": result}}
        )

    async def answer_tools_list(self, has_cursor: bool) -> None:
        if has_cursor:
            await asyncio.sleep(self.page2_delay)
            self.pages_sent += 1
            await self._send_mcp(
                2, {"tools": [_tool(n) for n in DEVICE_TOOLS_PAGE_2]}
            )
            await self.log(
                f"-> tools/list pagina 2 ({len(DEVICE_TOOLS_PAGE_2)} tools, geen cursor)"
            )
        else:
            await asyncio.sleep(self.page1_delay)
            self.pages_sent += 1
            await self._send_mcp(
                2,
                {
                    "tools": [_tool(n) for n in DEVICE_TOOLS_PAGE_1],
                    "nextCursor": "self.arm.wag_tail",
                },
            )
            await self.log(
                f"-> tools/list pagina 1 ({len(DEVICE_TOOLS_PAGE_1)} tools, "
                "nextCursor=self.arm.wag_tail)"
            )

    async def send_turn_later(self) -> None:
        await asyncio.sleep(self.turn_delay)
        if self.turn_sent:
            return
        self.turn_sent = True
        await self.send({"type": "listen", "state": "start", "mode": "manual"})
        await self.send({"type": "listen", "state": "detect", "text": self.text})
        await self.log(f"-> tekstbeurt: {self.text!r}")

    async def handle_mcp(self, payload: dict) -> None:
        method = payload.get("method")
        msg_id = payload.get("id")
        if method == "initialize":
            await self._send_mcp(
                1,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "rig-sim", "version": "0.0.1"},
                },
            )
            await self.log("-> initialize antwoord")
        elif method == "tools/list":
            cursor = (payload.get("params") or {}).get("cursor")
            await self.log(f"<- tools/list (cursor={cursor!r})")
            asyncio.ensure_future(self.answer_tools_list(bool(cursor)))
        elif method == "tools/call":
            name = (payload.get("params") or {}).get("name")
            self.tool_calls.append(name)
            await self.log(f"<- tools/call {name} {payload.get('params', {}).get('arguments')}")
            await self._send_mcp(msg_id, {"content": [{"type": "text", "text": "ok"}]})

    async def run(self) -> None:
        await self.send(_hello_payload())
        turn_task = asyncio.ensure_future(self.send_turn_later())
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(self.socket.recv(), timeout=60)
                except asyncio.TimeoutError:
                    break
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                mtype = message.get("type")
                if mtype == "hello":
                    await self.log("<- hello (welcome)")
                elif mtype == "mcp":
                    await self.handle_mcp(message.get("payload") or {})
                elif mtype == "tts":
                    state = message.get("state")
                    text = (message.get("text") or "")[:80]
                    if state == "sentence_start" and text:
                        await self.log(f"<- tts: {text}")
                    if state == "stop":
                        await self.log("<- tts stop (beurt klaar)")
                        # Nog even doorluisteren: late tools/calls zijn juist interessant.
                        await asyncio.sleep(self.linger)
                        break
                elif mtype == "stt":
                    await self.log(f"<- stt {message.get('state')}")
                elif mtype == "alert":
                    await self.log(f"<- alert {message.get('status')}")
                else:
                    await self.log(f"<- {mtype} {message.get('state', '')}".rstrip())
        finally:
            turn_task.cancel()
            await self.log(f"klaar: pages={self.pages_sent} tool_calls={self.tool_calls}")


async def run(url, text, device_id, page1_delay, page2_delay, turn_delay, linger) -> int:
    headers = {"device-id": device_id, "client-id": f"sim-{uuid.uuid4().hex[:8]}"}
    socket = await _connect(url, headers)
    print(f"[sim] verbonden met {url} als {device_id}", flush=True)
    try:
        sim = Sim(socket, text, page1_delay, page2_delay, turn_delay, linger)
        await sim.run()
        return 0
    finally:
        await socket.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:8001/xiaozhi/v1/", help="websocket-URL"
    )
    parser.add_argument("--text", default="Kijk naar mij", help="tekst voor de beurt")
    parser.add_argument(
        "--device-id", default="sim-mcp-paging-t_a2a3bf8a", help="device-id header"
    )
    parser.add_argument(
        "--page1-delay", type=float, default=0.0, help="seconden voor pagina 1"
    )
    parser.add_argument(
        "--page2-delay",
        type=float,
        default=0.4,
        help="seconden tussen pagina 1 en pagina 2 (3.0 = het echte incident)",
    )
    parser.add_argument(
        "--turn-delay",
        type=float,
        default=0.2,
        help="seconden na hello waarna de tekstbeurt begint",
    )
    parser.add_argument(
        "--linger",
        type=float,
        default=1.0,
        help="seconden doorluisteren na de laatste tts (late tools/calls zien)",
    )
    args = parser.parse_args()
    return asyncio.run(
        run(
            args.url,
            args.text,
            args.device_id,
            args.page1_delay,
            args.page2_delay,
            args.turn_delay,
            args.linger,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
