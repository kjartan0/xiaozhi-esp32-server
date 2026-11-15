"""首段（first chunk）长度上限：标点优先，超出窗口退化为词边界切分。

背景（kanban t_5cd58f74）：TTS 首包延迟随首句长度线性增长 —— 实测
`llm_to_tts_ms` 5 字符 681ms、110 字符 1138ms、284 字符 4418ms。
`first_chunk_max_words` 把首段压到稳定词数，剩余文本仍按原标点逻辑流式切分。
"""
import pytest

from core.providers.tts.base import (
    DEFAULT_FIRST_CHUNK_MAX_WORDS,
    MIN_FIRST_CHUNK_MAX_WORDS,
    TTSProviderBase,
)


class _TTS(TTSProviderBase):
    async def text_to_speak(self, text, output_file):
        return None


def _segments(text, step, first_sentence=True, max_words=DEFAULT_FIRST_CHUNK_MAX_WORDS):
    """按 step 字符喂流，返回切分出的所有段（含 stop 时的 flush 段）。"""
    tts = object.__new__(_TTS)
    tts.punctuations = ("。", "？", "?", "！", "!", "；", ";", "：")
    tts.first_sentence_punctuations = ("，", "~", "、", ",") + tts.punctuations
    tts.tts_text_buff, tts.processed_chars = [], 0
    tts.is_first_sentence, tts.tts_stop_request = first_sentence, False
    tts._first_chunk_max_words = max_words
    out = []
    for i in range(0, len(text), step):
        tts.tts_text_buff.append(text[i:i + step])
        segment = tts._get_segment_text()
        if segment:
            out.append(segment)
    tts.tts_stop_request = True
    rest = tts._get_segment_text()
    if rest:
        out.append(rest)
    return out


def _words(segments):
    return [w for seg in segments for w in seg.split()]


DUTCH_LONG = (
    "Ik heb gisteren de hele middag in de tuin gewerkt omdat het weer eindelijk "
    "eens een keer mooi was en daarna heb ik nog een boek gelezen op de bank."
)


@pytest.mark.parametrize("step", [1, 3, 7])
def test_long_first_sentence_is_capped_at_word_boundary(step):
    segments = _segments(DUTCH_LONG, step)
    assert len(segments[0].split()) == DEFAULT_FIRST_CHUNK_MAX_WORDS
    assert segments[0] == " ".join(DUTCH_LONG.split()[:DEFAULT_FIRST_CHUNK_MAX_WORDS])


@pytest.mark.parametrize("step", [1, 3, 7])
def test_no_text_loss_or_duplication(step):
    # 全部段按顺序拼接必须还原原文的词序列（stop flush 段保留句尾标点，故只比词干）
    segments = _segments(DUTCH_LONG, step)
    assert " ".join(_words(segments)).rstrip(".") == DUTCH_LONG.rstrip(".")


@pytest.mark.parametrize("step", [1, 4])
def test_punctuation_inside_window_is_preferred(step):
    # 逗号落在 8 词窗口内 → 仍按逗号切，不被词数上限改写
    segments = _segments("Nou ja, dat is dan een heel ander verhaal.", step)
    assert segments[0] == "Nou ja"


@pytest.mark.parametrize("step", [1, 4])
def test_punctuation_beyond_window_falls_back_to_words(step):
    # 逗号在第 12 个词之后（超出 8 词窗口）→ 首段改在词边界切开
    text = "een twee drie vier vijf zes zeven acht negen tien elf twaalf, en toen stopte het?"
    segments = _segments(text, step)
    assert segments[0] == "een twee drie vier vijf zes zeven acht"
    # 剩余文本仍按原标点逻辑流式切分（逗号不再是切点）
    assert segments[1].startswith("negen tien elf twaalf")


@pytest.mark.parametrize("step", [1, 4])
def test_only_first_chunk_is_capped(step):
    text = "een twee drie vier vijf zes zeven acht negen tien elf twaalf dertien veertien?"
    segments = _segments(text, step)
    assert segments[0] == "een twee drie vier vijf zes zeven acht"
    # 第二段不再受词数上限约束
    assert segments[1].startswith("negen tien elf twaalf dertien veertien")
    assert len(segments) == 2


@pytest.mark.parametrize("step", [1, 4])
def test_disabled_when_zero_keeps_old_behaviour(step):
    # 0 = 关闭：无标点的长句在首段阶段不切，整段 flush
    segments = _segments(DUTCH_LONG, step, max_words=0)
    assert segments == [DUTCH_LONG]


@pytest.mark.parametrize("step", [1, 4])
def test_not_applied_when_not_first_sentence(step):
    segments = _segments(DUTCH_LONG, step, first_sentence=False)
    assert segments == [DUTCH_LONG]


@pytest.mark.parametrize("step", [1, 4])
def test_word_cut_does_not_end_inside_open_quote(step):
    text = 'Hij zei "dit is echt een hele lange zin die nog even doorgaat" en toen ging hij weg.'
    segments = _segments(text, step)
    # 第 8 个词 “lange” 处引号未闭合，切点顺延到引号闭合之后（13 词）
    assert segments[0].endswith("doorgaat")
    assert len(segments[0].split()) == 13
    # 只有开引号随首段走，闭引号已被 textUtils 当作尾标点去掉
    assert segments[0].count('"') == 1
    stripped = [w.strip('"') for w in _words(segments)]
    assert " ".join(stripped).rstrip(".") == text.replace('"', "").rstrip(".")


@pytest.mark.parametrize("step", [1, 4])
def test_cjk_text_without_spaces_is_unchanged(step):
    # 无空白的 CJK 文本：找不到词边界，行为与原逻辑一致（等标点）
    assert _segments("普通の文。次の文！最後", step) == ["普通の文", "次の文", "最後"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, DEFAULT_FIRST_CHUNK_MAX_WORDS),
        ("8", 8),
        (0, 0),
        (-5, 0),
        (1, MIN_FIRST_CHUNK_MAX_WORDS),
        ("abc", DEFAULT_FIRST_CHUNK_MAX_WORDS),
    ],
)
def test_resolve_first_chunk_max_words(raw, expected):
    cfg = {} if raw is None else {"first_chunk_max_words": raw}
    assert TTSProviderBase._resolve_first_chunk_max_words(cfg) == expected
