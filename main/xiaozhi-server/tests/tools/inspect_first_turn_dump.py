"""Toon de <context>-regel uit data/.llm_warmup_firstturn.json + CJK-telling.

Bewijst acceptatiecriterium 3 van t_9fe9586d: na een echte eerste beurt staat
er een gevulde plaats en een Nederlands weerbericht in de promptcontext en
bevat de prompt nul CJK-tekens (t_c30a9895).

Gebruik: python tests/tools/inspect_first_turn_dump.py [pad]
"""

import json
import os
import re
import sys

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
DEFAULT_PATH = "/opt/xiaozhi-esp32-server/data/.llm_warmup_firstturn.json"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    system_text = ""
    for message in messages:
        if message.get("role") == "system":
            system_text = message.get("content") or ""
            break

    print(f"dump: {path} ({os.path.getsize(path)} bytes)")
    print(f"messages={len(messages)} tools={len(tools)}")
    print(f"CJK in system-prompt: {len(CJK_RE.findall(system_text))}")

    context_lines = []
    in_context = False
    for line in system_text.splitlines():
        if "<context>" in line:
            in_context = True
            continue
        if "</context>" in line:
            break
        if in_context:
            context_lines.append(line)

    print("--- <context> ---")
    print("\n".join(context_lines))
    print("--- einde <context> ---")

    full = json.dumps(payload, ensure_ascii=False)
    print(f"CJK in volledige payload: {len(CJK_RE.findall(full))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
