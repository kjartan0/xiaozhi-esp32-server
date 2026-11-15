"""本地 provider 预热（TTS / ASR / LLM）。

目的：把首轮对话里的"模型首次加载"开销从用户可感知的时延中挪走。

- 单独运行：  python tools/warmup_providers.py
- 服务启动时：app.py 里通过 asyncio.create_task(warmup_providers(config)) 触发，
  不阻塞启动；最多重试 3 次、间隔约 10 秒，失败只记日志（INFO/WARN），永不致命。

三个探针都按线上配置构造与真实请求一致的载荷：
  TTS  POST <url>  {"input": "Hallo.", ...}（custom 类型，{prompt_text} 替换）
  ASR  POST <base_url> multipart file=1s/16kHz 静音 WAV + model/language
  LLM  POST <base_url>/chat/completions  max_tokens=8 + chat_template_kwargs，
       请求内容是 build_probe_text() 的 >=64 token 探针（一次性 nonce + 一句自然语言
       + 填充散文）。裸的小探针（旧版 ~20 token）落在 Splash 的 17–31 行致命带里，
       会把引擎卡死 30s（ADR-022 / t_3efccbba）。

全部使用 httpx.AsyncClient，不阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
import sys
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

# 作为脚本直接运行（python tools/warmup_providers.py）时，把项目根目录加入 sys.path，
# 否则 tools/ 而不是根目录会成为 sys.path[0]，导致 import config 失败。
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from config.logger import setup_logging
from core.utils.llm_warmup import (
    WARMUP_PROBE_MIN_TOKENS,
    build_probe_text,
    estimate_probe_tokens,
)

TAG = "warmup"

TTS_PROBE_TEXT = "Hallo."
# LLM 探针的“自然语言那一句”。它只是探针主体：真正的请求内容由 build_probe_text()
# 在前面加一次性 nonce、在后面加填充散文，凑够 >=64 个未缓存 prefill 行（见 _probe_llm）。
LLM_PROBE_PROMPT = "Zeg alleen het woord hallo."
# 探针尾部的最小未缓存 prefill 行数（ADR-022 / t_3efccbba）：17–31 行是 Splash 引擎的
# 致命带（121 deaths / 1 completion），旧的 20 token 探针正好落在里面。
LLM_PROBE_MIN_TOKENS = WARMUP_PROBE_MIN_TOKENS
LLM_PROBE_MAX_TOKENS = 8
ASR_PROBE_SECONDS = 1.0
ASR_PROBE_RATE = 16000

# 单个探针的超时（秒）：ASR 首次请求会加载 GGUF 模型，给足时间
TTS_TIMEOUT = 60.0
ASR_TIMEOUT = 120.0
# LLM 探针的超时留在引擎 30s watchdog 之下：探针卡住时不要占着唯一的引擎槽位
# （与 core/utils/llm_warmup.py 的 timeout_seconds 同一理由）。
LLM_TIMEOUT = 20.0

logger = setup_logging()


def _silence_wav(seconds: float = ASR_PROBE_SECONDS, rate: int = ASR_PROBE_RATE) -> bytes:
    """生成 16 kHz / 单声道 / 16bit 静音 WAV。"""
    frames = int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<%dh" % frames, *([0] * frames)))
    return buffer.getvalue()


def _selected(config: Dict[str, Any], section: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    try:
        name = (config.get("selected_module") or {}).get(section)
        if not name:
            return None, None
        module_config = (config.get(section) or {}).get(name)
        if not isinstance(module_config, dict):
            return name, None
        return name, module_config
    except Exception:
        return None, None


def _result(label: str, ok: bool, ms: Optional[float], detail: str = "", skipped: bool = False) -> Dict[str, Any]:
    return {"provider": label, "ok": ok, "ms": None if ms is None else round(ms, 1), "detail": detail, "skipped": skipped}


async def _probe_tts(client: httpx.AsyncClient, config: Dict[str, Any]) -> Dict[str, Any]:
    name, cfg = _selected(config, "TTS")
    if not name or cfg is None:
        return _result("TTS/-", True, None, "TTS 未配置，跳过", skipped=True)
    label = f"TTS/{name}"
    if cfg.get("type") != "custom":
        return _result(label, True, None, f"type={cfg.get('type')} 不是 custom，跳过", skipped=True)

    url = cfg.get("url")
    params = cfg.get("params") or {}
    if isinstance(params, str):
        import json as _json

        try:
            params = _json.loads(params)
        except Exception:
            return _result(label, False, None, "params 无法解析")
    if not url or not isinstance(params, dict):
        return _result(label, False, None, "缺少 url 或 params")

    body = {}
    for key, value in params.items():
        if isinstance(value, str) and "{prompt_text}" in value:
            value = value.replace("{prompt_text}", TTS_PROBE_TEXT)
        body[key] = value

    started = time.monotonic()
    try:
        if str(cfg.get("method", "POST")).upper() == "POST":
            response = await client.post(url, json=body, headers=cfg.get("headers") or {}, timeout=TTS_TIMEOUT)
        else:
            response = await client.get(url, params=body, headers=cfg.get("headers") or {}, timeout=TTS_TIMEOUT)
        elapsed = (time.monotonic() - started) * 1000.0
    except Exception as exc:
        return _result(label, False, None, f"{type(exc).__name__}: {exc}")

    if response.status_code != 200:
        return _result(label, False, elapsed, f"HTTP {response.status_code}: {response.text[:200]}")
    return _result(label, True, elapsed, f"{len(response.content)} bytes")


async def _probe_asr(client: httpx.AsyncClient, config: Dict[str, Any]) -> Dict[str, Any]:
    name, cfg = _selected(config, "ASR")
    if not name or cfg is None:
        return _result("ASR/-", True, None, "ASR 未配置，跳过", skipped=True)
    label = f"ASR/{name}"
    if cfg.get("type") != "openai":
        return _result(label, True, None, f"type={cfg.get('type')} 不是 openai，跳过", skipped=True)

    url = cfg.get("base_url")
    model = cfg.get("model_name")
    if not url or not model:
        return _result(label, False, None, "缺少 base_url 或 model_name")

    data = {"model": model}
    if cfg.get("language"):
        data["language"] = cfg["language"]
    headers = {"Authorization": f"Bearer {cfg.get('api_key') or 'local'}"}

    started = time.monotonic()
    try:
        response = await client.post(
            url,
            files={"file": ("probe-silence.wav", _silence_wav(), "audio/wav")},
            data=data,
            headers=headers,
            timeout=ASR_TIMEOUT,
        )
        elapsed = (time.monotonic() - started) * 1000.0
    except Exception as exc:
        return _result(label, False, None, f"{type(exc).__name__}: {exc}")

    if response.status_code != 200:
        return _result(label, False, elapsed, f"HTTP {response.status_code}: {response.text[:200]}")
    return _result(label, True, elapsed, f"text={response.text[:60]!r}")


async def _probe_llm(client: httpx.AsyncClient, config: Dict[str, Any]) -> Dict[str, Any]:
    name, cfg = _selected(config, "LLM")
    if not name or cfg is None:
        return _result("LLM/-", True, None, "LLM 未配置，跳过", skipped=True)
    label = f"LLM/{name}"
    if cfg.get("type") != "openai":
        return _result(label, True, None, f"type={cfg.get('type')} 不是 openai，跳过", skipped=True)

    base_url = cfg.get("base_url") or cfg.get("url")
    if not base_url:
        return _result(label, False, None, "缺少 base_url")

    # 探针形状（t_3efccbba / ADR-022）：一次性 nonce + 一句自然语言 + 填充散文。
    # 旧探针是裸的一句 "Zeg alleen het woord hallo."（引擎侧 `input 20 · cached 0`），
    # 未缓存 prefill 落在 17–31 行的致命带里，每次容器启动都把 Splash native loop
    # 卡死 30s，引擎再花 ~60s 才回来。build_probe_text() 保证尾部 >=64 token，
    # 与 core/utils/llm_warmup.py 的会话预热完全同一形状。
    probe_text = build_probe_text(cfg.get("probe_text") or LLM_PROBE_PROMPT)
    probe_tokens = estimate_probe_tokens(probe_text)
    if probe_tokens < LLM_PROBE_MIN_TOKENS:  # build_probe_text 已保证；防御性二次闸门
        return _result(
            label,
            True,
            None,
            f"探针 {probe_tokens} token < {LLM_PROBE_MIN_TOKENS}，按 ADR-022 跳过",
            skipped=True,
        )

    payload: Dict[str, Any] = {
        "model": cfg.get("model_name"),
        "messages": [{"role": "user", "content": probe_text}],
        "max_tokens": LLM_PROBE_MAX_TOKENS,
        "stream": False,
    }
    chat_template_kwargs = cfg.get("chat_template_kwargs") or {}
    if isinstance(chat_template_kwargs, dict):
        payload.update(chat_template_kwargs)
    headers = {"Authorization": f"Bearer {cfg.get('api_key') or 'local'}"}

    # 一行说明请求形状：下次失败时不必再翻引擎日志才能知道发的是什么。
    logger.bind(tag=TAG).info(
        f"warmup_llm: input_tokens~{probe_tokens} chars={len(probe_text)} "
        f"nonce={probe_text.split()[1] if len(probe_text.split()) > 1 else '-'} "
        f"max_tokens={LLM_PROBE_MAX_TOKENS} stream=false model={cfg.get('model_name')}"
    )

    started = time.monotonic()
    try:
        response = await client.post(
            base_url.rstrip("/") + "/chat/completions",
            json=payload,
            headers=headers,
            timeout=LLM_TIMEOUT,
        )
        elapsed = (time.monotonic() - started) * 1000.0
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"warmup_llm: failed after {int((time.monotonic() - started) * 1000)}ms "
            f"{type(exc).__name__}: {exc} (input_tokens~{probe_tokens})"
        )
        return _result(label, False, None, f"{type(exc).__name__}: {exc}")

    if response.status_code != 200:
        return _result(label, False, elapsed, f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        body = response.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        content = (body["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        content = ""
    # 引擎侧的真实 prefill 行数（input/cached）比本地估算可信：有就一并记下来。
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    cached_tokens = (details or {}).get("cached_tokens") if isinstance(details, dict) else None
    if prompt_tokens is not None:
        logger.bind(tag=TAG).info(
            f"warmup_llm: ok input_tokens={prompt_tokens} cached_tokens={cached_tokens} "
            f"ms={int(elapsed)}"
        )
    return _result(
        label,
        True,
        elapsed,
        f"reply={content[:40]!r} input_tokens={prompt_tokens} "
        f"cached_tokens={cached_tokens} probe_tokens~{probe_tokens}",
    )


async def _run_once(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _probe_tts(client, config),
            _probe_asr(client, config),
            _probe_llm(client, config),
            return_exceptions=True,
        )
    out: List[Dict[str, Any]] = []
    for item in results:
        if isinstance(item, BaseException):
            out.append(_result("unknown", False, None, f"{type(item).__name__}: {item}"))
        else:
            out.append(item)
    return out


def _log_results(results: List[Dict[str, Any]], attempt: int, attempts: int) -> None:
    for item in results:
        if item.get("skipped"):
            logger.bind(tag=TAG).info(f"预热跳过 {item['provider']}: {item['detail']}")
        elif item["ok"]:
            logger.bind(tag=TAG).info(
                f"预热成功 {item['provider']} {item['ms']}ms（第 {attempt}/{attempts} 次）: {item['detail']}"
            )
        else:
            logger.bind(tag=TAG).warning(
                f"预热失败 {item['provider']}（第 {attempt}/{attempts} 次）: {item['detail']}"
            )


async def warmup_providers(
    config: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    delay: float = 10.0,
) -> List[Dict[str, Any]]:
    """尽力而为的预热：最多 retries 次，失败只记日志，永不抛异常。"""
    try:
        if config is None:
            from config.config_loader import load_config

            config = await load_config()
        if not isinstance(config, dict):
            return []
        retries = max(1, int(retries))
        results: List[Dict[str, Any]] = []
        for attempt in range(1, retries + 1):
            try:
                results = await _run_once(config)
            except Exception as exc:
                results = [_result("unknown", False, None, f"{type(exc).__name__}: {exc}")]
            _log_results(results, attempt, retries)
            if all(item.get("ok") for item in results):
                return results
            if attempt < retries:
                await asyncio.sleep(delay)
        return results
    except Exception as exc:  # 预热永远不能影响启动
        try:
            logger.bind(tag=TAG).warning(f"预热流程异常（已忽略）: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return []


def main() -> int:
    from config.config_loader import load_config

    config = asyncio.run(load_config())
    results = asyncio.run(warmup_providers(config))
    print("provider            ok     ms        detail")
    for item in results:
        status = "skip" if item.get("skipped") else ("ok" if item["ok"] else "FAIL")
        ms = "-" if item["ms"] is None else f"{item['ms']:.1f}"
        print(f"{item['provider']:<18} {status:<6} {ms:<9} {item['detail'][:70]}")
    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
