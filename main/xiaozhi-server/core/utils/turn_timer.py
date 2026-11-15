"""每轮语音时延预算计时器（tag: turnbudget）。

四个打点，一轮**只输出一行** JSON 日志：

  t0  用户停止说话（Silero 判定 client_voice_stop / 文本轮进入 startToChat）
  t1  ASR 文本可用
  t2  首句文本交给 TTS provider（to_tts_stream 首次被调用）
  t3  首段音频下发给设备（sendAudioMessage 且 sentenceType == FIRST）

设计约束：
- 任何情况下都不得抛异常，也不得改变原有控制流与时序；
  所有入口都包在 try/except 里，失败即静默降级为"无数据"。
- 允许缺失阶段：缺失的时间差输出 null，并在 missing 里给出原因。
- 使用 time.monotonic()（单调时钟），可跨线程（asyncio 线程 / ASR 线程 /
  TTS 线程 / executor 线程）打点，不需要锁（写操作是幂等的"首次写入生效"）。
- 定时器挂在 conn 上（属性名 _turn_timer），不侵入 ConnectionHandler 的
  __init__；conn 上没有该属性时自动创建，创建失败则整体 no-op。
"""

import json
import time
from typing import Any, Dict, List, Optional

from config.logger import setup_logging

TAG = "turnbudget"

# 超过这个时长仍未收尾的"轮"视为残留（如 TTS 失败、连接中断），不再输出
MAX_TURN_AGE_MS = 10 * 60 * 1000.0

_ATTR = "_turn_timer"

logger = setup_logging()


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 1)


def text_length(text: Any) -> Optional[int]:
    """统计用于统计的字符数；ASR 的 {"speaker":..,"content":..} JSON 只统计 content。"""
    try:
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                if isinstance(data, dict) and isinstance(data.get("content"), str):
                    stripped = data["content"].strip()
            except Exception:
                pass
        return len(stripped)
    except Exception:
        return None


class TurnTimer:
    """单轮计时状态。首次写入生效（first-write-wins），保证一阶段只打一次点。"""

    __slots__ = (
        "t0",
        "t1",
        "t2",
        "t3",
        "turn",
        "text_len",
        "chars_first_sentence",
        "closed",
        "_turns",
    )

    def __init__(self) -> None:
        self._turns = 0
        self.reset()

    def reset(self) -> None:
        self.t0: Optional[float] = None
        self.t1: Optional[float] = None
        self.t2: Optional[float] = None
        self.t3: Optional[float] = None
        self.turn: Optional[int] = None
        self.text_len: Optional[int] = None
        self.chars_first_sentence: Optional[int] = None
        self.closed: bool = False

    # -- 打点 ---------------------------------------------------------------

    def start(self) -> "TurnTimer":
        self.reset()
        self._turns += 1
        self.turn = self._turns
        self.t0 = _now_ms()
        return self

    def mark_t1_asr(self, text: Any = None) -> None:
        if self.t1 is None:
            self.t1 = _now_ms()
        if self.text_len is None:
            self.text_len = text_length(text)

    def mark_t2_tts(self, text: Any = None) -> None:
        if self.t2 is None:
            self.t2 = _now_ms()
        if self.chars_first_sentence is None:
            self.chars_first_sentence = text_length(text)

    def mark_t3_audio(self, text: Any = None) -> None:
        if self.t3 is None:
            self.t3 = _now_ms()
        if self.chars_first_sentence is None:
            self.chars_first_sentence = text_length(text)

    # -- 输出 ---------------------------------------------------------------

    @staticmethod
    def _delta(start: Optional[float], end: Optional[float]) -> Optional[float]:
        if start is None or end is None:
            return None
        delta = end - start
        if delta < 0:
            # 打点来自不同轮/时钟异常，视为不可用
            return None
        return _round(delta)

    def payload(self, conn: Any = None) -> Dict[str, Any]:
        missing: List[str] = []
        if self.t0 is None:
            missing.append("t0:turn_start_not_marked")
        if self.t1 is None:
            missing.append("t1:asr_text_not_marked")
        if self.t2 is None:
            missing.append("t2:first_tts_dispatch_not_marked")
        if self.t3 is None:
            missing.append("t3:first_audio_not_sent")

        return {
            "event": "turnbudget",
            "session_id": getattr(conn, "session_id", None),
            "turn": self.turn,
            "asr_ms": self._delta(self.t0, self.t1),
            "llm_to_tts_ms": self._delta(self.t1, self.t2),
            "tts_ms": self._delta(self.t2, self.t3),
            "total_ms": self._delta(self.t0, self.t3),
            "text_len": self.text_len,
            "chars_first_sentence": self.chars_first_sentence,
            "missing": missing,
        }

    def age_ms(self) -> Optional[float]:
        if self.t0 is None:
            return None
        return _now_ms() - self.t0


def _emit(conn: Any, payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    try:
        conn.logger.bind(tag=TAG).info(line)
    except Exception:
        try:
            logger.bind(tag=TAG).info(line)
        except Exception:
            pass


def _timer_of(conn: Any) -> Optional[TurnTimer]:
    if conn is None:
        return None
    timer = getattr(conn, _ATTR, None)
    if isinstance(timer, TurnTimer):
        return timer
    return None


def _ensure_timer(conn: Any) -> Optional[TurnTimer]:
    if conn is None:
        return None
    timer = _timer_of(conn)
    if timer is not None:
        return timer
    try:
        timer = TurnTimer()
        setattr(conn, _ATTR, timer)
        return timer
    except Exception:
        return None


# -- 供调用点使用的函数（全部保证不抛异常） ---------------------------------


def start_turn(conn: Any) -> Optional[TurnTimer]:
    """t0：一轮开始。已在计时中则保留更早的 t0，不覆盖。"""
    try:
        timer = _ensure_timer(conn)
        if timer is None:
            return None
        if timer.t0 is not None and not timer.closed:
            return timer
        return timer.start()
    except Exception:
        return None


def mark_asr(conn: Any, text: Any = None) -> None:
    """t1：ASR 文本可用（没有 t0 的文本轮会在此补一个 t0）。"""
    try:
        timer = _ensure_timer(conn)
        if timer is None:
            return
        if timer.t0 is None or timer.closed:
            timer.start()
        timer.mark_t1_asr(text)
    except Exception:
        return


def mark_tts(conn: Any, text: Any = None) -> None:
    """t2：首句文本交给 TTS provider。无进行中的轮（如欢迎语）则忽略。"""
    try:
        timer = _timer_of(conn)
        if timer is None or timer.closed or timer.t0 is None:
            return
        timer.mark_t2_tts(text)
    except Exception:
        return


def mark_first_audio(conn: Any, text: Any = None) -> Optional[Dict[str, Any]]:
    """t3：首段音频下发。成功输出一轮唯一的 turnbudget 日志。"""
    try:
        timer = _timer_of(conn)
        if timer is None or timer.closed:
            return None
        if timer.t0 is None:
            # 轮外的音频（hello/提示音等），不产生 turnbudget 行
            timer.reset()
            return None
        age = timer.age_ms()
        if age is not None and age > MAX_TURN_AGE_MS:
            timer.reset()
            return None
        timer.mark_t3_audio(text)
        payload = timer.payload(conn)
        timer.closed = True
        _emit(conn, payload)
        return payload
    except Exception:
        return None
