"""Tests for the per-chunk lead/trail silence trim (card t_1cdc72d5).

Background: Supertonic returns ~440-500 ms of lead silence and ~700-1000 ms of
trail silence in every synthesized chunk, and the runtime path (base.py →
audio_bytes_to_data_stream → pcm_to_data_stream → opus) delivered all of it —
measured on the real websocket stream as a ~1.0 s continuous silent run at the
seam of a 2-chunk answer.  These tests pin the trim helper's contract:

* it only ever removes lead/trail silence, never inside the chunk;
* it keeps a floor (``*_keep_ms``) so short chunks are not clipped;
* it is a no-op unless explicitly configured.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.utils.util import (
    SILENCE_TRIM_DEFAULTS,
    pcm_to_data_stream,
    trim_pcm_silence,
)

RATE = 24000


def _pcm(lead_ms=0, speech_ms=0, trail_ms=0, rate=RATE, amp=6000, freq=180.0):
    """Build 16-bit mono PCM: silence + tone + silence."""
    def silence(ms):
        return np.zeros(int(rate * ms / 1000), dtype=np.int16)

    t = np.arange(int(rate * speech_ms / 1000)) / rate
    tone = (np.sin(2 * np.pi * freq * t) * amp).astype(np.int16)
    return np.concatenate([silence(lead_ms), tone, silence(trail_ms)]).tobytes()


def test_defaults_keep_natural_pauses():
    assert SILENCE_TRIM_DEFAULTS["lead_keep_ms"] == 80
    assert SILENCE_TRIM_DEFAULTS["trail_keep_ms"] == 120


def test_trims_supertonic_shaped_padding():
    raw = _pcm(lead_ms=440, speech_ms=1000, trail_ms=730)
    trimmed, stats = trim_pcm_silence(raw, RATE, {})

    assert stats["applied"] is True
    assert stats["lead_ms"] == pytest.approx(440, abs=10)
    assert stats["trail_ms"] == pytest.approx(730, abs=10)
    # 440 - keep 80 = 360 lead cut, 730 - keep 120 = 610 trail cut
    assert stats["lead_cut_ms"] == pytest.approx(360, abs=10)
    assert stats["trail_cut_ms"] == pytest.approx(610, abs=10)
    out_ms = len(trimmed) / 2 / RATE * 1000
    assert out_ms == pytest.approx((440 + 1000 + 730) - 970, abs=10)

    # the speech itself survives untouched
    assert len(trimmed) < len(raw)
    reprofiled, restats = trim_pcm_silence(trimmed, RATE, {})
    assert restats["lead_ms"] == pytest.approx(80, abs=10)
    assert restats["trail_ms"] == pytest.approx(120, abs=10)


def test_caps_per_chunk_removal():
    raw = _pcm(lead_ms=2000, speech_ms=800, trail_ms=2000)
    _, stats = trim_pcm_silence(raw, RATE, {"lead_max_trim_ms": 400, "trail_max_trim_ms": 800})
    assert stats["lead_cut_ms"] == 400
    assert stats["trail_cut_ms"] == 800


def test_short_chunk_is_untouched():
    raw = _pcm(lead_ms=130, speech_ms=100, trail_ms=130)
    trimmed, stats = trim_pcm_silence(raw, RATE, {})
    assert stats["applied"] is False
    assert stats["reason"] == "below_min_chunk_ms"
    assert trimmed == raw


def test_all_silence_is_untouched():
    raw = np.zeros(RATE * 2, dtype=np.int16).tobytes()
    trimmed, stats = trim_pcm_silence(raw, RATE, {})
    assert stats["applied"] is False
    assert stats["reason"] == "no_voiced_frame"
    assert trimmed == raw


def test_never_removes_more_than_half_the_chunk():
    # 500 ms total, only 20 ms voiced -> trimming would leave less than half
    raw = _pcm(lead_ms=240, speech_ms=20, trail_ms=240)
    trimmed, stats = trim_pcm_silence(raw, RATE, {})
    assert stats["applied"] is False
    assert stats["reason"] == "would_remove_too_much"
    assert trimmed == raw


def test_keeps_extra_lead_when_configured():
    raw = _pcm(lead_ms=440, speech_ms=600, trail_ms=300)
    _, stats = trim_pcm_silence(raw, RATE, {"lead_keep_ms": 200, "trail_keep_ms": 200})
    assert stats["lead_cut_ms"] == pytest.approx(240, abs=10)
    assert stats["trail_cut_ms"] == pytest.approx(100, abs=10)


def test_empty_input_is_safe():
    trimmed, stats = trim_pcm_silence(b"", RATE, {})
    assert trimmed == b""
    assert stats["applied"] is False
    assert stats["reason"] == "empty"


def test_pcm_stream_without_config_is_unchanged():
    raw = _pcm(lead_ms=440, speech_ms=1000, trail_ms=730)
    frames = []
    stats = pcm_to_data_stream(raw, is_opus=False, callback=frames.append, sample_rate=RATE)
    assert stats is None
    # the stream zero-pads the final partial 60 ms frame, so compare the voiced prefix
    assert b"".join(frames)[: len(raw)] == raw


def test_pcm_stream_applies_trim_before_framing():
    raw = _pcm(lead_ms=440, speech_ms=1000, trail_ms=730)
    frames = []
    stats = pcm_to_data_stream(
        raw, is_opus=False, callback=frames.append, sample_rate=RATE, silence_trim={}
    )
    assert stats["applied"] is True
    assert b"".join(frames) == trim_pcm_silence(raw, RATE, {})[0]
    assert len(frames) == pytest.approx(len(b"".join(frames)) / (2 * RATE * 0.06), abs=1)


def test_config_resolver_is_opt_in():
    from core.providers.tts.base import TTSProviderBase

    assert TTSProviderBase._resolve_silence_trim({}) is None
    assert TTSProviderBase._resolve_silence_trim({"silence_trim": None}) is None
    assert TTSProviderBase._resolve_silence_trim({"silence_trim": {"enabled": False}}) is None
    resolved = TTSProviderBase._resolve_silence_trim(
        {"silence_trim": {"lead_keep_ms": 80, "min_chunk_ms": 400, "bogus": 1}}
    )
    assert resolved == {"lead_keep_ms": 80, "min_chunk_ms": 400}
