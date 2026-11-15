"""turnbudget 计时器单元测试（含缺失阶段 / 轮外音频 / 真实调用点）。

覆盖：
* 完整四点的数值计算；
* 缺失阶段 —— 输出 null 且 missing 里带原因；
* 一轮只输出一行；
* 轮外音频（欢迎语、提示音）不产生 turnbudget 行；
* 任何异常输入都不抛异常；
* 两个真实调用点确实打点：``TTSProviderBase.to_tts_stream``（t2）与
  ``sendAudioMessage(FIRST)``（t3）。

运行（需要 main/xiaozhi-server/data/.config.yaml 存在）：

    cd main/xiaozhi-server && printf '{}\\n' > data/.config.yaml
    python -m pytest tests/core/utils/test_turn_timer.py -v
"""

import json
from types import SimpleNamespace

import pytest

from core.handle.sendAudioHandle import sendAudioMessage
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType
from core.utils import turn_timer


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _RecordingLogger:
    """记录 (tag, msg) 的最小 loguru 替身。"""

    def __init__(self):
        self.entries = []
        self._tag = None

    def bind(self, **kwargs):
        self._tag = kwargs.get("tag")
        return self

    def info(self, msg, *args):
        self.entries.append((self._tag, msg))

    def warning(self, msg, *args):
        self.entries.append((self._tag, msg))

    def error(self, msg, *args):
        self.entries.append((self._tag, msg))

    def debug(self, msg, *args):
        pass

    def turnbudget_lines(self):
        return [json.loads(msg) for tag, msg in self.entries if tag == "turnbudget"]


def _conn(session_id="probe-session"):
    return SimpleNamespace(session_id=session_id, logger=_RecordingLogger())


class _Clock:
    """可手动推进的单调时钟（毫秒）。"""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += ms
        return self


# ---------------------------------------------------------------------------
# 数值与完整路径
# ---------------------------------------------------------------------------
def test_full_turn_computes_all_stages(monkeypatch):
    conn = _conn()
    clock = _Clock(1000.0)
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)  # t0 = 1000
    clock.advance(400)
    turn_timer.mark_asr(conn, "zet de lamp aan")  # t1 = 1400
    clock.advance(400)
    turn_timer.mark_tts(conn, "Oké, ik doe het licht aan.")  # t2 = 1800
    clock.advance(250)
    turn_timer.mark_first_audio(conn, "Oké, ik doe het licht aan.")  # t3 = 2050

    lines = conn.logger.turnbudget_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line["event"] == "turnbudget"
    assert line["session_id"] == "probe-session"
    assert line["asr_ms"] == 400.0
    assert line["llm_to_tts_ms"] == 400.0
    assert line["tts_ms"] == 250.0
    assert line["total_ms"] == 1050.0
    assert line["text_len"] == len("zet de lamp aan")
    assert line["chars_first_sentence"] == len("Oké, ik doe het licht aan.")
    assert line["missing"] == []


def test_partial_turn_uses_explicit_nulls_and_reasons(monkeypatch):
    """TTS 从未被调用（例如工具直出/报错）：t2 缺失，其余照常输出。"""
    conn = _conn()
    clock = _Clock(500.0)
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)
    clock.advance(400)
    turn_timer.mark_asr(conn, "hoe laat is het")
    clock.advance(300)
    turn_timer.mark_first_audio(conn, "Het is drie uur.")

    line = conn.logger.turnbudget_lines()[0]
    assert line["asr_ms"] == 400.0
    assert line["tts_ms"] is None
    assert line["llm_to_tts_ms"] is None
    assert line["total_ms"] == 700.0
    assert line["chars_first_sentence"] == len("Het is drie uur.")
    assert any(item.startswith("t2:") for item in line["missing"])
    assert not any(item.startswith("t1:") for item in line["missing"])


def test_only_t0_present_logs_rest_as_missing(monkeypatch):
    conn = _conn()
    clock = _Clock(10.0)
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)
    clock.advance(30)
    turn_timer.mark_first_audio(conn, "Hallo.")

    line = conn.logger.turnbudget_lines()[0]
    assert line["asr_ms"] is None
    assert line["llm_to_tts_ms"] is None
    assert line["tts_ms"] is None
    assert line["total_ms"] == 30.0
    assert line["text_len"] is None
    assert [item.split(":")[0] for item in line["missing"]] == ["t1", "t2"]


# ---------------------------------------------------------------------------
# 一次性 / 轮外音频
# ---------------------------------------------------------------------------
def test_one_line_per_turn(monkeypatch):
    conn = _conn()
    clock = _Clock()
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)
    clock.advance(100)
    turn_timer.mark_asr(conn, "test")
    clock.advance(100)
    turn_timer.mark_tts(conn, "Eerste zin.")
    clock.advance(100)
    turn_timer.mark_first_audio(conn, "Eerste zin.")
    # 下一句同样是 SentenceType.FIRST（delete_audio 模式），不得再输出
    turn_timer.mark_first_audio(conn, "Tweede zin.")

    assert len(conn.logger.turnbudget_lines()) == 1


def test_audio_outside_turn_is_ignored():
    """欢迎语/提示音没有开始一轮，不应产生 turnbudget 行。"""
    conn = _conn()
    turn_timer.mark_tts(conn, "Hallo, ik ben er.")
    turn_timer.mark_first_audio(conn, "Hallo, ik ben er.")
    assert conn.logger.turnbudget_lines() == []


def test_first_write_wins_for_stage_stamps(monkeypatch):
    conn = _conn()
    clock = _Clock()
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)
    clock.advance(50)
    turn_timer.mark_tts(conn, "Eerste segment.")
    clock.advance(500)
    turn_timer.mark_tts(conn, "Tweede segment.")  # 第二次 t2 打点被忽略
    clock.advance(150)
    turn_timer.mark_first_audio(conn, "Eerste segment.")

    line = conn.logger.turnbudget_lines()[0]
    assert line["tts_ms"] == 650.0
    assert line["chars_first_sentence"] == len("Eerste segment.")


def test_text_turn_backfills_t0_t1_from_mark_asr(monkeypatch):
    """文本轮没有 ASR 阶段：mark_asr 会补一个 t0/t1，仍能输出一行。"""
    conn = _conn()
    clock = _Clock()
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.mark_asr(conn, "tekst invoer")
    clock.advance(10)
    turn_timer.mark_tts(conn, "Antwoord.")
    clock.advance(20)
    turn_timer.mark_first_audio(conn, "Antwoord.")

    line = conn.logger.turnbudget_lines()[0]
    assert line["asr_ms"] == 0.0
    assert line["text_len"] == len("tekst invoer")


def test_stale_turn_is_dropped(monkeypatch):
    """残留的旧轮（超过 MAX_TURN_AGE_MS）不会输出，也不会污染下一轮。"""
    conn = _conn()
    clock = _Clock()
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)
    clock.advance(turn_timer.MAX_TURN_AGE_MS + 5000.0)
    turn_timer.mark_first_audio(conn, "oude zin")
    assert conn.logger.turnbudget_lines() == []

    clock = _Clock(0.0)
    monkeypatch.setattr(turn_timer, "_now_ms", clock)
    turn_timer.start_turn(conn)
    clock.advance(10)
    turn_timer.mark_asr(conn, "nieuw")
    clock.advance(10)
    turn_timer.mark_tts(conn, "Nieuwe zin.")
    clock.advance(10)
    turn_timer.mark_first_audio(conn, "Nieuwe zin.")

    lines = conn.logger.turnbudget_lines()
    assert len(lines) == 1
    assert lines[0]["turn"] == 2


# ---------------------------------------------------------------------------
# 健壮性
# ---------------------------------------------------------------------------
def test_never_raises_on_hostile_conn():
    assert turn_timer.start_turn(None) is None
    turn_timer.mark_asr(None, "x")
    turn_timer.mark_tts(None, "x")
    assert turn_timer.mark_first_audio(None, "x") is None

    # 没有 logger / 没有 session_id 的对象也不能抛
    for bad_conn in (object(), SimpleNamespace()):
        turn_timer.start_turn(bad_conn)
        turn_timer.mark_asr(bad_conn, "x")
        turn_timer.mark_tts(bad_conn, "x")
        turn_timer.mark_first_audio(bad_conn, "x")


class _ExplodingLogger:
    def bind(self, **kwargs):
        raise RuntimeError("logger down")

    def info(self, *args, **kwargs):
        raise RuntimeError("logger down")


def test_logger_failure_is_swallowed(monkeypatch):
    conn = SimpleNamespace(session_id="s", logger=_ExplodingLogger())
    monkeypatch.setattr(turn_timer, "_now_ms", _Clock())
    turn_timer.start_turn(conn)
    turn_timer.mark_asr(conn, "a")
    turn_timer.mark_tts(conn, "b")
    assert turn_timer.mark_first_audio(conn, "b") is not None  # 数据仍然产出


def test_text_length_unwraps_asr_json():
    assert turn_timer.text_length(
        '{"speaker": "Kjartan", "content": "zet de lamp aan"}'
    ) == len("zet de lamp aan")
    assert turn_timer.text_length("gewone tekst") == len("gewone tekst")
    assert turn_timer.text_length(None) is None


# ---------------------------------------------------------------------------
# 真实调用点
# ---------------------------------------------------------------------------
class _ProbeTTS(TTSProviderBase):
    """最小 TTS provider：不访问网络，只验证 to_tts_stream 的 t2 打点。"""

    def __init__(self, conn):
        super().__init__({"output_dir": "tmp/"}, True)
        self.conn = conn

    async def text_to_speak(self, text, output_file):
        return None


def test_to_tts_stream_marks_t2(monkeypatch):
    conn = _conn()
    tts = _ProbeTTS(conn)
    clock = _Clock()
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)
    clock.advance(10)
    turn_timer.mark_asr(conn, "zet de lamp aan")
    clock.advance(120)
    tts.to_tts_stream("Oké, ik doe het licht aan.")

    assert conn._turn_timer.t2 is not None
    assert conn._turn_timer.chars_first_sentence == len("Oké, ik doe het licht aan.")


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)


def _audio_conn(sentence_id="sid-1"):
    conn = _conn()
    conn.websocket = _FakeWebSocket()
    conn.sentence_id = sentence_id
    conn.tts = SimpleNamespace(tts_audio_first_sentence=True)
    conn.calling = False
    conn.close_after_chat = False
    return conn


@pytest.mark.asyncio
async def test_send_audio_message_first_emits_turnbudget(monkeypatch):
    conn = _audio_conn()
    clock = _Clock()
    monkeypatch.setattr(turn_timer, "_now_ms", clock)

    turn_timer.start_turn(conn)
    clock.advance(200)
    turn_timer.mark_asr(conn, "zet de lamp aan")
    clock.advance(300)
    turn_timer.mark_tts(conn, "Oké.")
    clock.advance(300)

    await sendAudioMessage(conn, SentenceType.FIRST, None, "Oké.", "sid-1")

    lines = conn.logger.turnbudget_lines()
    assert len(lines) == 1
    assert lines[0]["tts_ms"] == 300.0
    assert lines[0]["total_ms"] == 800.0


@pytest.mark.asyncio
async def test_send_audio_message_skips_stale_sentence(monkeypatch):
    conn = _audio_conn(sentence_id="sid-2")
    monkeypatch.setattr(turn_timer, "_now_ms", _Clock())

    turn_timer.start_turn(conn)
    turn_timer.mark_asr(conn, "oud")
    turn_timer.mark_tts(conn, "Oud antwoord.")

    await sendAudioMessage(conn, SentenceType.FIRST, None, "Oud antwoord.", "sid-1")
    assert conn.logger.turnbudget_lines() == []
