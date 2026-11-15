"""tools.warmup_providers 的 LLM 探针形状（t_3efccbba / ADR-022）。

回归保护：容器启动时的 LLM provider 预热曾经发送一句 20 token 的裸探针
（引擎侧 `input 20 · cached 0`），未缓存 prefill 落在 Splash 的 17–31 行致命带里，
每次 `docker compose up -d` 都把 native loop 卡死 30 s、引擎再花 ~60 s 才回来。

不触网：用假 httpx client 断言请求体与日志。
"""
import asyncio

import pytest

from core.utils import llm_warmup
from tools import warmup_providers

# 引擎日志实测的致命未缓存行区间（17–31 行 ~99% 致死）
FATAL_BAND_LOW = 17
FATAL_BAND_HIGH = 31

CONFIG = {
    "selected_module": {"LLM": "LocalOMLX"},
    "LLM": {
        "LocalOMLX": {
            "type": "openai",
            "base_url": "http://host.docker.internal:8000/v1",
            "model_name": "Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit",
            "api_key": "local",
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    },
}


class _Response:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body


class _Client:
    def __init__(self, response=None, raise_exc=None):
        self.calls = []
        self._response = response
        self._raise = raise_exc

    async def post(self, url, json=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self._raise is not None:
            raise self._raise
        return self._response


def _ok_body(prompt_tokens=118, cached_tokens=0, content="Hallo."):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def _run(client, config=None):
    return asyncio.run(warmup_providers._probe_llm(client, config or CONFIG))


def _probe_of(client):
    return client.calls[0]["json"]["messages"][-1]["content"]


def test_llm_probe_is_clear_of_the_fatal_uncached_row_band():
    client = _Client(_Response(body=_ok_body()))
    result = _run(client)
    assert result["ok"] is True
    probe = _probe_of(client)
    tokens = warmup_providers.estimate_probe_tokens(probe)
    assert tokens >= warmup_providers.LLM_PROBE_MIN_TOKENS
    # 明显高于致命带上沿，不是刚好贴边
    assert tokens >= 2 * FATAL_BAND_HIGH


def test_legacy_bare_prompt_was_in_the_fatal_band():
    """旧探针本体确实落在致命带里 —— 这就是本卡要修的东西。"""
    legacy_tokens = warmup_providers.estimate_probe_tokens(
        warmup_providers.LLM_PROBE_PROMPT
    )
    assert legacy_tokens < FATAL_BAND_LOW + 6  # 估算口径下 ~16 token，实测引擎侧 20 行
    probe_tokens = warmup_providers.estimate_probe_tokens(
        warmup_providers.build_probe_text(warmup_providers.LLM_PROBE_PROMPT)
    )
    assert probe_tokens > FATAL_BAND_HIGH * 2


def test_probe_carries_a_fresh_nonce_so_repeats_never_shrink():
    client = _Client(_Response(body=_ok_body()))
    _run(client)
    _run(client)
    first = _probe_of(client)  # calls[0] of a shared client keeps both probes below
    probes = [call["json"]["messages"][-1]["content"] for call in client.calls]
    assert len(probes) == 2
    assert probes[0] != probes[1]
    for probe in probes:
        assert probe.startswith("Warmup ")
        assert warmup_providers.estimate_probe_tokens(probe) >= warmup_providers.LLM_PROBE_MIN_TOKENS
    assert first == probes[0]


def test_short_configured_probe_is_padded_to_the_floor():
    config = {
        "selected_module": {"LLM": "LocalOMLX"},
        "LLM": {"LocalOMLX": dict(CONFIG["LLM"]["LocalOMLX"], probe_text="hallo")},
    }
    client = _Client(_Response(body=_ok_body()))
    _run(client, config)
    probe = _probe_of(client)
    assert warmup_providers.estimate_probe_tokens(probe) >= warmup_providers.LLM_PROBE_MIN_TOKENS
    assert probe.rstrip().endswith(llm_warmup.WARMUP_PROBE_FILLER.strip()[-20:])


def test_payload_keeps_model_timeout_and_thinking_flag():
    client = _Client(_Response(body=_ok_body()))
    _run(client)
    call = client.calls[0]
    payload = call["json"]
    assert call["url"] == "http://host.docker.internal:8000/v1/chat/completions"
    assert payload["model"] == "Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit"
    assert payload["max_tokens"] == warmup_providers.LLM_PROBE_MAX_TOKENS
    assert payload["stream"] is False
    assert payload["enable_thinking"] is False
    # 超时留在引擎 30s watchdog 之下
    assert call["timeout"] <= 30


def test_engine_usage_is_surfaced_in_the_result():
    client = _Client(_Response(body=_ok_body(prompt_tokens=121, cached_tokens=0)))
    result = _run(client)
    assert "input_tokens=121" in result["detail"]
    assert "cached_tokens=0" in result["detail"]


def test_probe_shape_is_logged_before_sending():
    """失败时不必翻引擎日志：形状（估算 token 数 / max_tokens / nonce）在发送前先落一行。"""
    logged = []
    sink_id = warmup_providers.logger.add(
        lambda message: logged.append(message.record["message"]), level="INFO"
    )
    try:
        client = _Client(_Response(body=_ok_body()))
        _run(client)
    finally:
        warmup_providers.logger.remove(sink_id)
    joined = "\n".join(logged)
    assert "warmup_llm: input_tokens~" in joined
    assert "max_tokens=8" in joined
    assert "nonce=" in joined


def test_http_error_keeps_the_old_failure_contract():
    client = _Client(_Response(status_code=500, body={}, text="boom"))
    result = _run(client)
    assert result["ok"] is False
    assert "HTTP 500" in result["detail"]


def test_transport_error_is_not_fatal():
    client = _Client(raise_exc=ConnectionError("peer closed connection"))
    result = _run(client)
    assert result["ok"] is False
    assert "ConnectionError" in result["detail"]


@pytest.mark.parametrize("section", ["TTS", "ASR", "LLM"])
def test_missing_or_disabled_module_still_skips(section):
    """未选中该 section 时仍然跳过而不是报错。"""
    client = _Client(_Response(body=_ok_body()))
    config = {"selected_module": {}, section: {}}
    coro = {
        "TTS": warmup_providers._probe_tts,
        "ASR": warmup_providers._probe_asr,
        "LLM": warmup_providers._probe_llm,
    }[section]
    result = asyncio.run(coro(client, config))
    assert result["skipped"] is True
