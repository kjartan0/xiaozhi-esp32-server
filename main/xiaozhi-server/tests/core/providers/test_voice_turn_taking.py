"""Turn-taking regression tests: ASR pre-roll retention + Silero VAD endpoint.

Pins the two knobs that stopped the robot cutting the owner off mid-sentence
and clipping the first word:

* ``core/providers/asr/base.py`` keeps 20 frames (20 x 60 ms = 1.2 s) of
  pre-roll audio instead of 10 frames (0.6 s).
* ``SileroVAD.min_silence_duration_ms`` (600 in the live config) is honoured:
  ``client_voice_stop`` must not fire before the configured amount of silence
  has elapsed since the last voiced frame.

No model files or network access are needed: the Silero ONNX session and the
wall clock are scripted so the timing assertions stay deterministic.

Running this suite requires ``main/xiaozhi-server/data/.config.yaml`` to exist
(the config package checks for it at import time). CI creates an empty one with
``printf '{}\\n' > data/.config.yaml``; do the same locally:

    cd main/xiaozhi-server && printf '{}\\n' > data/.config.yaml
    python -m pytest tests/core/providers/test_voice_turn_taking.py -v
"""
import asyncio
from collections import deque

import numpy as np
import pytest

from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

# Device frames arriving from the robot are 60 ms @ 16 kHz mono int16.
FRAME_MS = 60
FRAME_BYTES = 1920
PREROLL_FRAMES = 20


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
def _frame(index: int) -> bytes:
    """A distinguishable 60 ms PCM frame: 4-byte big-endian index + zeros."""
    return index.to_bytes(4, "big") + b"\x00" * (FRAME_BYTES - 4)


def _frame_index(frame: bytes) -> int:
    return int.from_bytes(frame[:4], "big")


class _RecordingASR(ASRProviderBase):
    """Minimal concrete provider that records what reaches the ASR layer."""

    interface_type = InterfaceType.NON_STREAM

    def __init__(self):
        super().__init__()
        self.voice_stops = []
        self.resume_points = []

    async def speech_to_text(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def handle_voice_stop(self, conn, asr_audio_task):
        self.voice_stops.append(b"".join(asr_audio_task))


class _FakeConn:
    """The subset of ConnectionHandler that receive_audio touches."""

    def __init__(self, asr):
        self.client_listen_mode = "auto"
        self.asr = asr
        self.asr_audio = []
        self.client_have_voice = False
        self.client_voice_stop = False
        self.reset_calls = 0

    def reset_audio_states(self):
        self.reset_calls += 1
        self.asr_audio.clear()
        self.client_have_voice = False
        self.client_voice_stop = False


# ---------------------------------------------------------------------------
# ASR pre-roll
# ---------------------------------------------------------------------------
def test_preroll_keeps_last_twenty_frames():
    asr = _RecordingASR()
    conn = _FakeConn(asr)

    for i in range(25):
        asyncio.run(asr.receive_audio(conn, _frame(i), audio_have_voice=False))
        # Never grows past the pre-roll window while there is no speech.
        assert len(conn.asr_audio) == min(i + 1, PREROLL_FRAMES)

    assert [_frame_index(f) for f in conn.asr_audio] == list(
        range(25 - PREROLL_FRAMES, 25)
    )
    assert asr.voice_stops == []


def test_preroll_frames_reach_asr_when_the_turn_ends():
    asr = _RecordingASR()
    conn = _FakeConn(asr)

    # 30 silent frames: only the last 20 may survive as pre-roll.
    for i in range(30):
        asyncio.run(asr.receive_audio(conn, _frame(i), audio_have_voice=False))

    # VAD hears speech; the buffer now accumulates inside the turn.
    conn.client_have_voice = True
    for i in range(30, 35):
        asyncio.run(asr.receive_audio(conn, _frame(i), audio_have_voice=True))

    # VAD declares the turn over on the next frame.
    conn.client_voice_stop = True
    asyncio.run(asr.receive_audio(conn, _frame(35), audio_have_voice=True))

    assert conn.reset_calls == 1
    assert len(asr.voice_stops) == 1

    payload = asr.voice_stops[0]
    forwarded = [
        _frame_index(payload[off : off + FRAME_BYTES])
        for off in range(0, len(payload), FRAME_BYTES)
    ]

    # 20 pre-roll frames (10..29) + 6 frames of live speech (30..35).
    assert forwarded == list(range(10, 36))
    assert forwarded[0] == 30 - PREROLL_FRAMES
    # 1.2 s of pre-roll survived, i.e. well past the old 10-frame / 0.6 s cap.
    assert len(forwarded[:PREROLL_FRAMES]) * FRAME_MS == 1200


# ---------------------------------------------------------------------------
# Silero VAD endpoint timing
# ---------------------------------------------------------------------------
class _FakeClock:
    """Stand-in for the ``time`` module inside silero.py."""

    def __init__(self):
        self.ms = 1_000_000.0

    def time(self):
        return self.ms / 1000.0


class _ScriptedSession:
    """Fake onnxruntime session whose speech probability is scripted."""

    def __init__(self, state):
        self.state = state
        self.calls = 0

    def run(self, _outputs, ort_inputs):
        self.calls += 1
        return np.array([self.state["prob"]], dtype=np.float32), ort_inputs["state"]


class _FakeVADConn:
    """The subset of ConnectionHandler that SileroVAD.is_vad touches."""

    def __init__(self):
        self.client_listen_mode = "auto"
        self.client_audio_buffer = bytearray()
        self.client_voice_window = deque(maxlen=5)
        self.client_have_voice = False
        self.client_voice_stop = False
        self.last_is_voice = False
        self.vad_last_voice_time = 0


def _build_vad(monkeypatch, silence_ms):
    from core.providers.vad import silero as silero_mod

    state = {"prob": 0.95}
    provider = object.__new__(silero_mod.VADProvider)  # skip ONNX loading
    provider.vad_threshold = 0.5
    provider.vad_threshold_low = 0.3
    provider.silence_threshold_ms = silence_ms
    provider.frame_window_threshold = 3
    provider.session = _ScriptedSession(state)

    clock = _FakeClock()
    monkeypatch.setattr(silero_mod, "time", clock)
    return provider, state, clock


def _measure_stop_delay(monkeypatch, silence_ms, max_silent_frames=40):
    """Return (stop_delay_ms, early_stop_delays, conn) for one turn."""
    provider, state, clock = _build_vad(monkeypatch, silence_ms)
    conn = _FakeVADConn()

    state["prob"] = 0.95
    last_voiced_at = clock.ms
    for _ in range(20):
        clock.ms += FRAME_MS
        if provider.is_vad(conn, b"\x00" * FRAME_BYTES):
            last_voiced_at = clock.ms

    state["prob"] = 0.02
    stop_delay = None
    early = []
    for _ in range(max_silent_frames):
        clock.ms += FRAME_MS
        provider.is_vad(conn, b"\x00" * FRAME_BYTES)
        elapsed = clock.ms - last_voiced_at
        if conn.client_voice_stop:
            if stop_delay is None:
                stop_delay = elapsed
            if elapsed < silence_ms:
                early.append(elapsed)
        elif stop_delay is not None:
            break
    return stop_delay, early, conn


def test_voice_stop_waits_for_silence_threshold(monkeypatch):
    for silence_ms in (200, 600):
        stop_delay, early, conn = _measure_stop_delay(monkeypatch, silence_ms)

        assert conn.client_have_voice is True
        assert stop_delay is not None, f"no stop detected at {silence_ms} ms"

        # The endpoint never fires before the configured silence has elapsed...
        assert early == [], f"stop fired early at {silence_ms} ms: {early}"
        assert stop_delay >= silence_ms
        # ...and the sliding window only adds a bounded lag on top of it.
        assert stop_delay <= silence_ms + 400


def test_live_600ms_threshold_stops_later_than_old_200ms(monkeypatch):
    fast, _, _ = _measure_stop_delay(monkeypatch, 200)
    slow, _, _ = _measure_stop_delay(monkeypatch, 600)

    assert fast is not None and slow is not None
    assert slow > fast
