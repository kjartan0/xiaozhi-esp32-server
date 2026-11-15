"""首段（first chunk）长度上限：标点优先，超出窗口退化为词边界切分。

背景（kanban t_5cd58f74）：TTS 首包延迟随首句长度线性增长 —— 实测
`llm_to_tts_ms` 5 字符 681ms、110 字符 1138ms、284 字符 4418ms。
`first_chunk_max_words` 把首段压到稳定词数，剩余文本仍按原标点逻辑流式切分。
"""
import pytest

from core.providers.tts.base import (
    DEFAULT_FIRST_CHUNK_MAX_WORDS,
    FIRST_SENTENCE_EXTRA_PUNCTUATIONS,
    MIN_FIRST_CHUNK_MAX_WORDS,
    SENTENCE_END_PUNCTUATIONS,
    TTSProviderBase,
)


class _TTS(TTSProviderBase):
    async def text_to_speak(self, text, output_file):
        return None


def _segments(text, step, first_sentence=True, max_words=DEFAULT_FIRST_CHUNK_MAX_WORDS):
    """按 step 字符喂流，返回切分出的所有段（含 stop 时的 flush 段）。"""
    tts = object.__new__(_TTS)
    # 用生产常量而非复制一份标点表：复制会让测试与 base.py 漂移（t_78fef7de）
    tts.punctuations = SENTENCE_END_PUNCTUATIONS
    tts.first_sentence_punctuations = (
        FIRST_SENTENCE_EXTRA_PUNCTUATIONS + SENTENCE_END_PUNCTUATIONS
    )
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
    # 0 = 关闭：首段阶段不做词数切分，整段一次输出。
    # 句末的 "." 现在是切点（t_78fef7de），所以走正常段路径而非 stop flush，
    # 尾标点被 textUtils 去掉 —— 词序列与旧行为相同，只有这个尾句号。
    segments = _segments(DUTCH_LONG, step, max_words=0)
    assert segments == [DUTCH_LONG.rstrip(".")]


@pytest.mark.parametrize("step", [1, 4])
def test_not_applied_when_not_first_sentence(step):
    # 非首段不受 8 词上限约束：只有句末 "." 一处切分
    segments = _segments(DUTCH_LONG, step, first_sentence=False)
    assert segments == [DUTCH_LONG.rstrip(".")]


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


# --- t_78fef7de: 句末 "." 现在能结束一段（此前首段只能靠 8 词上限切开） -------------
# 下面两条是真实荷兰语回答（t_f2903196/t_5e3e933e 证据里被切坏的首段）。
PERIOD_BOUNDED = [
    # 5 词首句 + 句号：旧行为切出 "Mijn favoriete kleur is blauw. Het voelt zo"（43 字符 8 词）
    (
        "Mijn favoriete kleur is blauw. Het voelt zo rustig en kalm, want blauw geeft me rust.",
        "Mijn favoriete kleur is blauw",
    ),
    # 3 词首句 + 句号：旧行为切成 "Blauw is mooi. Het geeft me rust en ruimte en"（35 字符 8 词）
    (
        "Blauw is mooi. Het geeft me rust en ruimte, vooral op een heldere dag.",
        "Blauw is mooi",
    ),
]


@pytest.mark.parametrize("step", [1, 3, 7])
@pytest.mark.parametrize("text,expected", PERIOD_BOUNDED)
def test_period_inside_window_ends_first_chunk(text, expected, step):
    segments = _segments(text, step)
    assert segments[0] == expected
    assert len(segments[0].split()) < DEFAULT_FIRST_CHUNK_MAX_WORDS


@pytest.mark.parametrize("step", [1, 4])
def test_ellipsis_ends_first_chunk(step):
    # "…" 与 "." 同义：模型偶尔用省略号收尾（证据里实测 0 次，防御性加入）。
    # 注意 textUtils 的尾标点表里没有 "…"，所以它随首段保留（"?" 同理）
    assert _segments("Nou… dat is dan een heel ander verhaal.", step)[0] == "Nou…"


@pytest.mark.parametrize("step", [1, 4])
def test_period_beyond_window_still_falls_back_to_words(step):
    # 首句超过 8 词：句号不能在窗口外提前抢切，行为必须与旧逻辑一致
    text = (
        "Ik heb gisteren de hele middag in de tuin gewerkt en daarna heb ik nog "
        "een boek gelezen op de bank."
    )
    segments = _segments(text, step)
    assert segments[0] == " ".join(text.split()[:DEFAULT_FIRST_CHUNK_MAX_WORDS])
    assert len(segments[0].split()) == DEFAULT_FIRST_CHUNK_MAX_WORDS
    assert "." not in segments[0]


@pytest.mark.parametrize("step", [1, 4])
def test_question_mark_behaviour_unchanged(step):
    # 问号一直是切点，行为不变（"Blauw is mooi?" → 14 字符 3 词；textUtils
    # 的尾标点表不含 "?"，所以问号随首段保留 —— 与 t_5e3e933e 的实测表一致）
    segments = _segments("Blauw is mooi? Ja, dat geeft me rust en ruimte.", step)
    assert segments[0] == "Blauw is mooi?"


@pytest.mark.parametrize("step", [1, 4])
def test_comma_behaviour_unchanged(step):
    # 逗号仍在 8 词窗口内优先切分（"Ik vind zelf" → 12 字符 3 词）
    segments = _segments("Ik vind zelf, blauw is mooi. Het geeft me rust.", step)
    assert segments[0] == "Ik vind zelf"


@pytest.mark.parametrize("step", [1, 3, 7])
def test_periods_do_not_lose_text(step):
    text = (
        "Mijn favoriete kleur is blauw. Het voelt zo rustig en kalm, want blauw "
        "geeft me ruimte."
    )
    flat = " ".join(_words(_segments(text, step)))
    assert flat.replace(",", "").split() == text.replace(",", "").replace(".", "").split()


@pytest.mark.parametrize("step", [1, 2, 4])
def test_decimal_point_is_not_a_boundary(step):
    # "21.5" / "1.000" 的数字点不能当句末：否则首段变成 "Het is 21"
    segments = _segments("Het is 21.5 graden buiten. Dat is warm.", step)
    assert segments[0] == "Het is 21.5 graden buiten"
    assert "21.5" in segments[0]


@pytest.mark.parametrize("step", [1, 2, 4])
def test_thousands_separator_is_not_a_boundary(step):
    segments = _segments("Dat kost 1.000 euro en dan nog wat.", step)
    assert segments[0] == "Dat kost 1.000 euro en dan nog wat"


@pytest.mark.parametrize("step", [1, 4])
def test_period_ends_later_chunks_too(step):
    # "." 加入 punctuations 后，后续段同样在句末切分（不再只靠逗号/问号/停流）
    segments = _segments("Nou, dat wist ik niet. Echt waar. Nou ja.", step)
    assert segments[0] == "Nou"
    assert segments[1] == "dat wist ik niet"
    assert segments[2] == "Echt waar"

