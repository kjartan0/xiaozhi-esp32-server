import os
import re
import math
import uuid
import queue
import asyncio
import threading
import traceback

from core.utils import p3
from core.utils import turn_timer
from datetime import datetime
from core.utils import textUtils
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
from config.logger import setup_logging
from core.utils import opus_encoder_utils
from core.utils.tts import MarkdownCleaner, convert_percentage_to_range
from core.utils.output_counter import add_device_output
from core.handle.reportHandle import enqueue_tts_report
from core.handle.sendAudioHandle import sendAudioMessage
from core.utils.util import audio_bytes_to_data_stream, audio_to_data_stream
from core.providers.tts.dto.dto import (
    TTSMessageDTO,
    SentenceType,
    ContentType,
    InterfaceType,
)

TAG = __name__
logger = setup_logging()

# 首段词数上限的默认值与下限（见 TTSProviderBase._first_chunk_max_words）
DEFAULT_FIRST_CHUNK_MAX_WORDS = 8
MIN_FIRST_CHUNK_MAX_WORDS = 3

# 段结束标点（流式切分用，按出现位置取最早者可切处）。
# "." 与 "…" 是荷兰语/英语等拉丁语系句末的主要标记（t_78fef7de）；CJK 标点保持原样。
SENTENCE_END_PUNCTUATIONS = (
    "。",
    "？",
    "?",
    "！",
    "!",
    "；",
    ";",
    "：",
    ".",
    "…",
)
# 首段额外接受的“软”断点：逗号类可以把首包提前发出（句末标点当然也可以）。
FIRST_SENTENCE_EXTRA_PUNCTUATIONS = (
    "，",
    "~",
    "、",
    ",",
)


class TTSProviderBase(ABC):
    def __init__(self, config, delete_audio_file):
        self.interface_type = InterfaceType.NON_STREAM
        self.conn = None
        self.delete_audio_file = delete_audio_file
        self.audio_file_type = "wav"
        self.output_file = config.get("output_dir", "tmp/")
        self.tts_timeout = float(config.get("tts_timeout", 15))
        if not math.isfinite(self.tts_timeout) or self.tts_timeout <= 0:
            raise ValueError("tts_timeout must be a positive finite number")
        self.tts_text_queue = queue.Queue()
        self.tts_audio_queue = queue.Queue()
        self.tts_audio_first_sentence = True
        self.before_stop_play_files = []
        self.report_on_last = False
        # sentence_id 到文本的映射，用于流式TTS获取正确的字幕文本
        self._sentence_text_map = {}
        # 加载替换词，用于一次性正则替换
        raw_words = config.get("correct_words", [])
        self.correct_words = {}
        for item in raw_words:
            parts = item.split("|", 1)
            if len(parts) == 2:
                self.correct_words[parts[0]] = parts[1]
        # 构建正则表达式，使用最长匹配优先（排序后转义拼接）
        if self.correct_words:
            # 按key长度降序排列，长的先匹配，避免短词部分干扰
            sorted_keys = sorted(self.correct_words.keys(), key=len, reverse=True)
            pattern_str = "|".join(re.escape(k) for k in sorted_keys)
            self._correct_words_pattern = re.compile(pattern_str)
            # 构建反向替换正则，用于将TTS服务返回的替换后文本还原为原始文本（字幕显示）
            reverse_map = {v: k for k, v in self.correct_words.items()}
            sorted_reverse_keys = sorted(reverse_map.keys(), key=len, reverse=True)
            reverse_pattern_str = "|".join(re.escape(k) for k in sorted_reverse_keys)
            self._reverse_words_pattern = re.compile(reverse_pattern_str)
            self._reverse_words_map = reverse_map
            # 流式滑动窗口：按首字分组的替换词字典，用于快速查找
            self._words_by_first_char = {}
            for key in sorted_keys:  # 使用已按长度降序排列的keys，确保长词优先匹配
                first_char = key[0] if key else ""
                if first_char not in self._words_by_first_char:
                    self._words_by_first_char[first_char] = []
                self._words_by_first_char[first_char].append(key)
        else:
            self._correct_words_pattern = None
            self._reverse_words_pattern = None
            self._reverse_words_map = None

        # 流式滑动窗口：待匹配的缓存文本
        self._pending_prefix = ""
        self.tts_text_buff = []
        # 句子结束标点：只有这些标点能结束一段（其余文本等流式继续累积）。
        # "."/"…" 让荷兰语等拉丁语系的句子句末也能真正结束一段（t_78fef7de）：
        # 此前首段只能靠 first_chunk_max_words(8) 或 , ? ! ; : 切开，句末的 "."
        # 无法切分，一句 3 个词的荷兰语句子仍会被吞进 8 词上限的首段。
        self.punctuations = SENTENCE_END_PUNCTUATIONS
        self.first_sentence_punctuations = (
            FIRST_SENTENCE_EXTRA_PUNCTUATIONS + SENTENCE_END_PUNCTUATIONS
        )
        self.tts_stop_request = False
        self.processed_chars = 0
        self.is_first_sentence = True
        # 首段最大词数：标点切分超出该词数时退化为词边界切分，0 表示禁用。
        # 目的：TTS 首包受首句长度线性拖累（实测 284 字符无标点句 → 4.4s），
        # 词边界兜底把首段压到稳定长度，剩余文本照常流式切分。
        self._first_chunk_max_words = self._resolve_first_chunk_max_words(config)
        # 首尾静音裁剪（t_1cdc72d5）：Supertonic 每段固定 ~440-500ms 前导 + ~700-1000ms
        # 尾部静音，实测原样进入下发 opus 流（2 段答案缝处 ~1.0s 连续静音）。
        # 未配置该块时为 None → 行为完全不变。
        self.silence_trim = self._resolve_silence_trim(config)

    @staticmethod
    def _resolve_silence_trim(config):
        """解析 silence_trim 配置块；缺省/关闭时返回 None（不裁）。"""
        raw = (config or {}).get("silence_trim", None)
        if not raw:
            return None
        if raw is True:
            raw = {}
        if not isinstance(raw, dict):
            logger.bind(tag=TAG).warning(
                f"silence_trim 配置非法({raw!r})，按未配置处理"
            )
            return None
        if str(raw.get("enabled", "true")).lower() in ("false", "0", "no"):
            return None
        resolved = {}
        for key in (
            "lead_keep_ms",
            "trail_keep_ms",
            "lead_max_trim_ms",
            "trail_max_trim_ms",
            "min_chunk_ms",
        ):
            if key not in raw:
                continue
            try:
                value = int(raw[key])
            except (TypeError, ValueError):
                logger.bind(tag=TAG).warning(
                    f"silence_trim.{key} 配置非法({raw[key]!r})，使用默认值"
                )
                continue
            if value < 0:
                continue
            resolved[key] = value
        return resolved

    def _log_silence_trim(self, stats, text=None):
        """记录一次裁剪的实际效果（INFO 只在真的裁了时打印，便于取证）。"""
        if not stats:
            return
        if stats.get("applied"):
            logger.bind(tag=TAG).info(
                "silence_trim: lead {}ms->{}ms trail {}ms->{}ms "
                "(cut {}/{}ms of {}ms chunk) text={!r}".format(
                    stats.get("lead_ms"), (stats.get("lead_ms") or 0) - stats.get("lead_cut_ms", 0),
                    stats.get("trail_ms"), (stats.get("trail_ms") or 0) - stats.get("trail_cut_ms", 0),
                    stats.get("lead_cut_ms"), stats.get("trail_cut_ms"),
                    stats.get("duration_ms"), (text or "")[:40],
                )
            )
        else:
            logger.bind(tag=TAG).debug(
                f"silence_trim: skip reason={stats.get('reason')} text={(text or '')[:40]!r}"
            )

    @staticmethod
    def _resolve_first_chunk_max_words(config) -> int:
        """解析 first_chunk_max_words：缺省 8，最小 3，0（或负数）禁用。"""
        raw = (config or {}).get("first_chunk_max_words", None)
        if raw is None:
            return DEFAULT_FIRST_CHUNK_MAX_WORDS
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.bind(tag=TAG).warning(
                f"first_chunk_max_words 配置非法({raw!r})，回退默认值 {DEFAULT_FIRST_CHUNK_MAX_WORDS}"
            )
            return DEFAULT_FIRST_CHUNK_MAX_WORDS
        if value <= 0:
            return 0
        return max(MIN_FIRST_CHUNK_MAX_WORDS, value)

    def generate_filename(self, extension=".wav"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    def handle_opus(self, opus_data: bytes):
        logger.bind(tag=TAG).debug(f"推送数据到队列里面帧数～～ {len(opus_data)}")
        self.tts_audio_queue.put((SentenceType.MIDDLE, opus_data, None, getattr(self, 'current_sentence_id', None)))

    def handle_audio_file(self, file_audio: bytes, text):
        self.before_stop_play_files.append((file_audio, text))

    def to_tts_stream(self, text, opus_handler: Callable[[bytes], None] = None) -> None:
        # turnbudget t2：首句文本交付 TTS provider（每轮只记第一次）
        turn_timer.mark_tts(getattr(self, "conn", None), text)
        # 保留原始文本用于显示/上报
        original_text = text
        text = MarkdownCleaner.clean_markdown(text)
        # 使用正则一次性替换，避免重复遍历和部分匹配问题
        if self._correct_words_pattern:
            text = self._correct_words_pattern.sub(lambda m: self.correct_words[m.group(0)], text)
        max_repeat_time = 5
        if self.delete_audio_file:
            # 需要删除文件的直接转为音频数据
            while max_repeat_time > 0:
                try:
                    audio_bytes = asyncio.run(self.text_to_speak(text, None))
                    if audio_bytes:
                        # 使用原始文本用于显示/上报
                        self.tts_audio_queue.put((SentenceType.FIRST, None, original_text, getattr(self, 'current_sentence_id', None)))
                        stats = audio_bytes_to_data_stream(
                            audio_bytes,
                            file_type=self.audio_file_type,
                            is_opus=True,
                            callback=opus_handler,
                            sample_rate=self.conn.sample_rate,
                            opus_encoder=self.opus_encoder,
                            silence_trim=self.silence_trim,
                        )
                        self._log_silence_trim(stats, original_text)
                        break
                    else:
                        max_repeat_time -= 1
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"语音生成失败{5 - max_repeat_time + 1}次: {original_text}，错误: {e}"
                    )
                    max_repeat_time -= 1
            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"语音生成成功: {original_text}，重试{5 - max_repeat_time}次"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"语音生成失败: {original_text}，请检查网络或服务是否正常"
                )
            return None
        else:
            tmp_file = self.generate_filename()
            try:
                while not os.path.exists(tmp_file) and max_repeat_time > 0:
                    try:
                        asyncio.run(self.text_to_speak(text, tmp_file))
                    except Exception as e:
                        logger.bind(tag=TAG).warning(
                            f"语音生成失败{5 - max_repeat_time + 1}次: {original_text}，错误: {e}"
                        )
                        # 未执行成功，删除文件
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        max_repeat_time -= 1

                if max_repeat_time > 0:
                    logger.bind(tag=TAG).info(
                        f"语音生成成功: {original_text}:{tmp_file}，重试{5 - max_repeat_time}次"
                    )
                else:
                    logger.bind(tag=TAG).error(
                        f"语音生成失败: {original_text}，请检查网络或服务是否正常"
                    )
                self.tts_audio_queue.put((SentenceType.FIRST, None, original_text, getattr(self, 'current_sentence_id', None)))
                self._process_audio_file_stream(tmp_file, callback=opus_handler)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
                return None
    
    def to_tts(self, text):
        # 保留原始文本用于日志/显示
        original_text = text
        text = MarkdownCleaner.clean_markdown(text)
        if self._correct_words_pattern:
            text = self._correct_words_pattern.sub(lambda m: self.correct_words[m.group(0)], text)
        max_repeat_time = 5
        if self.delete_audio_file:
            # 需要删除文件的直接转为音频数据
            while max_repeat_time > 0:
                try:
                    audio_bytes = asyncio.run(self.text_to_speak(text, None))
                    if audio_bytes:
                        audio_datas = []
                        stats = audio_bytes_to_data_stream(
                            audio_bytes,
                            file_type=self.audio_file_type,
                            is_opus=True,
                            callback=lambda data: audio_datas.append(data),
                            sample_rate=self.conn.sample_rate,
                            silence_trim=self.silence_trim,
                        )
                        self._log_silence_trim(stats, original_text)
                        return audio_datas
                    else:
                        max_repeat_time -= 1
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"语音生成失败{5 - max_repeat_time + 1}次: {original_text}，错误: {e}"
                    )
                    max_repeat_time -= 1
            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"语音生成成功: {original_text}，重试{5 - max_repeat_time}次"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"语音生成失败: {original_text}，请检查网络或服务是否正常"
                )
            return None
        else:
            tmp_file = self.generate_filename()
            try:
                while not os.path.exists(tmp_file) and max_repeat_time > 0:
                    try:
                        asyncio.run(self.text_to_speak(text, tmp_file))
                    except Exception as e:
                        logger.bind(tag=TAG).warning(
                            f"语音生成失败{5 - max_repeat_time + 1}次: {original_text}，错误: {e}"
                        )
                        # 未执行成功，删除文件
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        max_repeat_time -= 1

                if max_repeat_time > 0:
                    logger.bind(tag=TAG).info(
                        f"语音生成成功: {original_text}:{tmp_file}，重试{5 - max_repeat_time}次"
                    )
                else:
                    logger.bind(tag=TAG).error(
                        f"语音生成失败: {original_text}，请检查网络或服务是否正常"
                    )

                return tmp_file
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
                return None

    @abstractmethod
    async def text_to_speak(self, text, output_file):
        pass

    def audio_to_pcm_data_stream(
        self, audio_file_path, callback: Callable[[Any], Any] = None
    ):
        """音频文件转换为PCM编码"""
        return audio_to_data_stream(audio_file_path, is_opus=False, callback=callback, sample_rate=self.conn.sample_rate, opus_encoder=None, silence_trim=self.silence_trim)

    def audio_to_opus_data_stream(
        self, audio_file_path, callback: Callable[[Any], Any] = None
    ):
        """音频文件转换为Opus编码"""
        return audio_to_data_stream(audio_file_path, is_opus=True, callback=callback, sample_rate=self.conn.sample_rate, opus_encoder=self.opus_encoder, silence_trim=self.silence_trim)

    def tts_one_sentence(
        self,
        conn,
        content_type,
        content_detail=None,
        content_file=None,
        sentence_id=None,
    ):
        """发送一句话"""
        if not sentence_id:
            if conn.sentence_id:
                sentence_id = conn.sentence_id
            else:
                sentence_id = str(uuid.uuid4().hex)
                conn.sentence_id = sentence_id
        # 对于单句的文本，进行分段处理
        segments = re.split(r"([。！？!?；;\n])", content_detail)
        for seg in segments:
            self.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=content_type,
                    content_detail=seg,
                    content_file=content_file,
                )
            )

    async def open_audio_channels(self, conn):
        self.conn = conn

        # 根据conn的sample_rate创建编码器，如果子类已经创建则不覆盖（IndexTTS接口返回为24kHZ-待重采样处理）
        if not hasattr(self, 'opus_encoder') or self.opus_encoder is None:
            self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
                sample_rate=conn.sample_rate, channels=1, frame_size_ms=60
            )

        # tts 消化线程
        self.tts_priority_thread = threading.Thread(
            target=self.tts_text_priority_thread, daemon=True
        )
        self.tts_priority_thread.start()

        # 音频播放 消化线程
        self.audio_play_priority_thread = threading.Thread(
            target=self._audio_play_priority_thread, daemon=True
        )
        self.audio_play_priority_thread.start()

    def store_tts_text(self, sentence_id, text):
        """存储指定 sentence_id 对应的文本，用于流式TTS获取正确的字幕文本

        Args:
            sentence_id: 会话ID
            text: 要存储的文本
        """
        if sentence_id and text:
            self._sentence_text_map[sentence_id] = text
            # 只保留最近 5 个，防止内存泄漏
            if len(self._sentence_text_map) > 5:
                oldest = next(iter(self._sentence_text_map))
                del self._sentence_text_map[oldest]

    def get_tts_text(self, sentence_id):
        """获取指定 sentence_id 对应的文本

        Args:
            sentence_id: 会话ID

        Returns:
            str: 对应的文本，如果不存在返回 None
        """
        return self._sentence_text_map.get(sentence_id)

    def clear_tts_text(self, sentence_id):
        """清除指定 sentence_id 的文本

        Args:
            sentence_id: 会话ID
        """
        if sentence_id in self._sentence_text_map:
            del self._sentence_text_map[sentence_id]

    def _restore_original_text(self, text):
        if not self._reverse_words_pattern or not text:
            return text
        return self._reverse_words_pattern.sub(
            lambda m: self._reverse_words_map[m.group(0)], text
        )

    # 这里默认是非流式的处理方式
    # 流式处理方式请在子类中重写
    def tts_text_priority_thread(self):
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                if self.conn.client_abort:
                    logger.bind(tag=TAG).info("收到打断信息，终止TTS文本处理线程")
                    continue
                # 过滤旧消息：检查sentence_id是否匹配
                if message.sentence_id != self.conn.sentence_id:
                    continue
                if message.sentence_type == SentenceType.FIRST:
                    self.current_sentence_id = message.sentence_id
                    self.tts_stop_request = False
                    self.processed_chars = 0
                    self.tts_text_buff = []
                    self.is_first_sentence = True
                    self.tts_audio_first_sentence = True
                elif ContentType.TEXT == message.content_type:
                    self.tts_text_buff.append(message.content_detail)
                    segment_text = self._get_segment_text()
                    if segment_text:
                        self.to_tts_stream(segment_text, opus_handler=self.handle_opus)
                elif ContentType.FILE == message.content_type:
                    self._process_remaining_text_stream(opus_handler=self.handle_opus)
                    tts_file = message.content_file
                    if tts_file and os.path.exists(tts_file):
                        self._process_audio_file_stream(
                            tts_file, callback=self.handle_opus
                        )
                if message.sentence_type == SentenceType.LAST:
                    self._process_remaining_text_stream(opus_handler=self.handle_opus)
                    self.tts_audio_queue.put(
                        (message.sentence_type, [], message.content_detail, message.sentence_id)
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理TTS文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )
                continue

    def _audio_play_priority_thread(self):
        # 需要上报的文本和音频列表
        enqueue_text = None
        enqueue_audio = []
        while not self.conn.stop_event.is_set():
            text = None
            try:
                try:
                    item = self.tts_audio_queue.get(timeout=0.1)
                    if len(item) == 4:
                        sentence_type, audio_datas, text, sentence_id = item
                    else:
                        sentence_type, audio_datas, text = item
                        sentence_id = None
                except queue.Empty:
                    if self.conn.stop_event.is_set():
                        break
                    continue

                if self.conn.client_abort:
                    logger.bind(tag=TAG).debug("收到打断信号，跳过当前音频数据")
                    enqueue_text, enqueue_audio = None, []
                    continue

                # 收到下一个文本开始或会话结束时进行上报
                if sentence_type is not SentenceType.MIDDLE:
                    if self.report_on_last:
                        # 累积模式：适用于全程只有一个语音流的TTS（如seed-tts-2.0）
                        # FIRST时只记录文本，音频持续累积，仅在LAST时统一上报
                        if text:
                            enqueue_text = text
                        if sentence_type == SentenceType.LAST:
                            enqueue_tts_report(self.conn, enqueue_text, enqueue_audio)
                            enqueue_audio = []
                            enqueue_text = None
                    else:
                        # 非累积模式：每个句子分别上报
                        if enqueue_text is not None:
                            enqueue_tts_report(self.conn, enqueue_text, enqueue_audio)
                        enqueue_audio = []
                        enqueue_text = text

                # 收集上报音频数据
                if isinstance(audio_datas, bytes):
                    enqueue_audio.append(audio_datas)

                # 发送音频
                future = asyncio.run_coroutine_threadsafe(
                    sendAudioMessage(self.conn, sentence_type, audio_datas, text, sentence_id),
                    self.conn.loop,
                )
                future.result()

                # 记录输出和报告
                if self.conn.max_output_size > 0 and text:
                    add_device_output(self.conn.headers.get("device-id"), len(text))

            except Exception as e:
                logger.bind(tag=TAG).error(f"audio_play_priority_thread: {text} {e}")

    async def start_session(self, session_id):
        pass

    async def finish_session(self, session_id):
        pass

    async def close(self):
        """资源清理方法"""
        self._sentence_text_map.clear()
        if hasattr(self, "ws") and self.ws:
            await self.ws.close()

    # 右引号/右括号，以及它们对应的左侧符号
    _QUOTE_PAIRS = (
        ("「", "」"),
        ("『", "』"),
        ("“", "”"),
        ('"', '"'),
        ("（", "）"),
        ("《", "》"),
        ("«", "»"),
        ("„", "“"),
        ("【", "】"),
        ("〈", "〉"),
        ("［", "］"),
        ("[", "]"),
    )
    # 注意：“（U+201C）有双重语义：中文里是开引号（对应 ”），
    # 德语 „…“ 里是闭引号；二者冲突时计数法无法兼顾，
    # 德语句子的切分只能靠“不平衡就顺延”保守处理（最终由 flush 兜底）。
    _CLOSERS = '」』”》）"»“】〉］]'
    _QUOTE_CHARS = "".join(ch for pair in _QUOTE_PAIRS for ch in pair)

    @classmethod
    def _is_quote_balanced(cls, segment: str) -> bool:
        """引号/括号是否成对闭合（计数法，〝同符引号〞用奇偶检测）。"""
        for opener, closer in cls._QUOTE_PAIRS:
            n_open = segment.count(opener)
            if opener == closer:
                if n_open % 2:
                    return False
            elif n_open > segment.count(closer):
                return False
        return True

    @staticmethod
    def _is_decimal_point(text: str, pos: int) -> bool:
        """小数点/千位分隔符不是句子边界（荷兰语 "1.000"、模型常写的 "21.5"）。

        只有紧跟在数字后面、且后一个字符是数字（或还没流到）的 "." 才被排除：
        后一个字符尚未到达时先不切分，等下一块文本；真正的句末数字（"Het is 21. Volgende"）
        后面是空格，照常切分。
        """
        if text[pos] != "." or pos == 0 or not text[pos - 1].isdigit():
            return False
        return pos + 1 >= len(text) or text[pos + 1].isdigit()

    @classmethod
    def _quote_safe_cut(cls, text: str, pos: int, punctuations) -> int:
        """避免在引号内切分（如 「わぁ！」って / “好！”他说）。

        候选位置之后紧跟的右引号和句末标点一并带走；若候选位于未闭合的引号内，
        或右引号后句子还在继续，则顺延到下一个标点；都不安全时返回 -1，等待更多文本。
        """
        if not any(ch in cls._QUOTE_CHARS for ch in text):
            # 全文不含引号/括号：不存在未闭合引号，右引号也无需携带，候选位置必然安全
            return pos
        candidates = [pos] + [
            i
            for i, ch in enumerate(text)
            if i > pos
            and ch in punctuations
            and not cls._is_decimal_point(text, i)
        ]
        for candidate in candidates:
            end = candidate + 1
            carried_closer = False
            while end < len(text) and (
                text[end] in cls._CLOSERS or text[end] in punctuations
            ):
                if text[end] in cls._CLOSERS:
                    carried_closer = True
                end += 1
            segment = text[:end]
            if not cls._is_quote_balanced(segment):
                continue
            if carried_closer and (end == len(text) or not text[end].isspace()):
                # 右引号之后句子还在继续（「…」って、“…”他说），或暂时看不到后文
                continue
            return end - 1
        return -1

    def _first_chunk_word_cut(self, text: str, max_words: int) -> int:
        """首段兜底切分：满 max_words 个完整词后按词边界切开。

        返回本次切分的结束下标（含该下标字符，即第 N 个词后的分隔符），-1 表示
        当前文本还找不到安全词边界（词未完整、或引号未闭合、或全文没有空白分词）。
        引号未闭合的候选顺延到下一个词边界，最多尝试 2×max_words 个词。
        """
        idx = 0
        count = 0
        length = len(text)
        limit = max_words * 2
        while idx < length and count < limit:
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                break
            while idx < length and not text[idx].isspace():
                idx += 1
            count += 1
            # idx 指向词后的分隔符；词完整且引号闭合才切
            if count >= max_words and idx < length and self._is_quote_balanced(text[:idx]):
                return idx
        return -1

    def _cap_first_chunk(self, text: str, punct_cut: int) -> int:
        """首段长度上限：标点落在窗口内仍用标点，否则退化为词边界切分。

        目的：TTS 首包延迟随首句长度线性增长（284 字符无标点句 → 4.4s），
        首段压到 max_words 个词后，剩余文本照常按原逻辑流式切分。
        """
        max_words = getattr(self, "_first_chunk_max_words", 0)
        if not max_words or not self.is_first_sentence:
            return punct_cut
        if punct_cut != -1 and len(re.findall(r"\S+", text[: punct_cut + 1])) <= max_words:
            return punct_cut
        word_cut = self._first_chunk_word_cut(text, max_words)
        if word_cut == -1:
            return punct_cut
        if punct_cut == -1:
            return word_cut
        return min(word_cut, punct_cut)

    def _get_segment_text(self) -> Optional[str]:
        # 合并当前全部文本并处理未分割部分
        full_text = "".join(self.tts_text_buff)
        current_text = full_text[self.processed_chars :]  # 从未处理的位置开始
        last_punct_pos = -1

        # 根据是否是第一句话选择不同的标点符号集合
        punctuations_to_use = (
            self.first_sentence_punctuations
            if self.is_first_sentence
            else self.punctuations
        )

        # 取每种标点最后出现位置的最早者作为切分点（与逐个 rfind 再取最小值等价）
        last_seen = {}
        for i, ch in enumerate(current_text):
            if ch in punctuations_to_use and not self._is_decimal_point(
                current_text, i
            ):
                last_seen[ch] = i
        last_punct_pos = min(last_seen.values(), default=-1)

        cut_pos = -1
        if last_punct_pos != -1:
            cut_pos = self._quote_safe_cut(
                current_text, last_punct_pos, punctuations_to_use
            )
        if self.is_first_sentence:
            # 首段上限：标点超出窗口时退化为词边界切分（first_chunk_max_words）
            cut_pos = self._cap_first_chunk(current_text, cut_pos)
        if cut_pos != -1:
            segment_text_raw = current_text[: cut_pos + 1]
            segment_text = textUtils.get_string_no_punctuation_or_emoji(
                segment_text_raw
            )
            self.processed_chars += len(segment_text_raw)  # 更新已处理字符位置

            # 如果是第一句话，在找到第一个逗号后，将标志设置为False
            if self.is_first_sentence:
                self.is_first_sentence = False

            return segment_text
        elif self.tts_stop_request and current_text:
            segment_text = current_text
            self.is_first_sentence = True  # 重置标志
            return segment_text
        else:
            return None

    def _process_audio_file_stream(
        self, tts_file, callback: Callable[[Any], Any]
    ) -> None:
        """处理音频文件并转换为指定格式

        Args:
            tts_file: 音频文件路径
            callback: 文件处理函数
        """
        if tts_file.endswith(".p3"):
            p3.decode_opus_from_file_stream(tts_file, callback=callback)
        elif self.conn.audio_format == "pcm":
            self.audio_to_pcm_data_stream(tts_file, callback=callback)
        else:
            self.audio_to_opus_data_stream(tts_file, callback=callback)

        if (
            self.delete_audio_file
            and tts_file is not None
            and os.path.exists(tts_file)
            and tts_file.startswith(self.output_file)
        ):
            os.remove(tts_file)

    def _process_before_stop_play_files(self):
        for audio_datas, text in self.before_stop_play_files:
            self.tts_audio_queue.put((SentenceType.MIDDLE, audio_datas, text, getattr(self, 'current_sentence_id', None)))
        self.before_stop_play_files.clear()
        self.tts_audio_queue.put((SentenceType.LAST, [], None, getattr(self, 'current_sentence_id', None)))

    def _process_remaining_text_stream(
        self, opus_handler: Callable[[bytes], None] = None
    ):
        """处理剩余的文本并生成语音

        Returns:
            bool: 是否成功处理了文本
        """
        full_text = "".join(self.tts_text_buff)
        remaining_text = full_text[self.processed_chars :]
        if remaining_text:
            segment_text = textUtils.get_string_no_punctuation_or_emoji(remaining_text)
            if segment_text:
                self.to_tts_stream(segment_text, opus_handler=opus_handler)
                self.processed_chars += len(full_text)
                return True
        return False

    def _apply_percentage_params(self, config):
        """根据子类定义的 TTS_PARAM_CONFIG 批量应用百分比参数"""
        for config_key, attr_name, min_val, max_val, base_val, transform in self.TTS_PARAM_CONFIG:
            if config_key in config:
                val = convert_percentage_to_range(config[config_key], min_val, max_val, base_val)
                setattr(self, attr_name, transform(val) if transform else val)

    def _match_stream_text(self, text):
        """流式文本滑动窗口匹配，用于处理跨分片的替换词

        Args:
            text: 输入的文本片段

        Returns:
            tuple: (确定的文本列表, 剩余待匹配的前缀)
        """
        if not self.correct_words or not text:
            return [text] if text else [], ""

        result = []
        pending = self._pending_prefix
        i = 0

        while i < len(text):
            char = text[i]

            # 尝试：pending + 当前字符 是否能匹配替换词
            test_text = pending + char

            matched = False
            # 遍历可能匹配的替换词
            candidates = self._words_by_first_char.get(pending[0], []) if pending else self._words_by_first_char.get(char, [])
            for key in candidates:
                if test_text == key:
                    # 完整匹配，替换后发送
                    result.append(self.correct_words[key])
                    pending = ""
                    matched = True
                    break
                elif key.startswith(test_text):
                    # 是替换词的前缀，继续等待
                    pending = test_text
                    matched = True
                    break

            if matched:
                i += 1
                continue

            # 没有匹配到更长的词，pending 的内容确定可以发送
            if pending:
                result.append(pending)
                pending = ""

            # 检查当前字符是否是某个替换词的开头
            if char in self._words_by_first_char:
                pending = char
            else:
                result.append(char)

            i += 1

        return result, pending

    def reset_stream_state(self):
        """重置流式处理状态，用于会话开始时清理残留状态"""
        self._pending_prefix = ""
