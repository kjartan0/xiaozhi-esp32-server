"""core.utils.llm_warmup：会话开启时预热 LLM 前缀缓存的后台调度器。

覆盖：开关、请求形状（流式 + max_tokens>=8）、单飞、300s 去重、真实轮次在途让路
（调度时 + 发送前 + 流式读取中）、静默失败、哈希稳定性、前缀审计口径一致。
不触网：用假 provider/client 断言请求参数与调度状态。
"""
import time
from types import SimpleNamespace

import pytest

from core.utils import llm_warmup


class _Chunk:
    def __init__(self, content=None, usage=None):
        self.usage = usage
        self.choices = (
            [SimpleNamespace(delta=SimpleNamespace(content=content))]
            if content is not None
            else []
        )


class _Stream:
    def __init__(self, chunks, on_iterate=None):
        self._chunks = chunks
        self._on_iterate = on_iterate
        self.closed = False

    def __iter__(self):
        if self._on_iterate is not None:
            self._on_iterate()
        for chunk in self._chunks:
            yield chunk

    def close(self):
        self.closed = True


class _Completions:
    def __init__(self, fail=False, delay=0.0, prompt_tokens=2341, cached_tokens=0, on_iterate=None):
        self.calls = []
        self.fail = fail
        self.delay = delay
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.on_iterate = on_iterate
        self.last_stream = None

    def create(self, **params):
        self.calls.append(params)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("engine down")
        usage = SimpleNamespace(
            prompt_tokens=self.prompt_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached_tokens),
        )
        stream = _Stream(
            [
                _Chunk("p"),
                _Chunk("ing"),
                _Chunk(None, usage=usage),
            ],
            on_iterate=self.on_iterate,
        )
        self.last_stream = stream
        return stream


class _Provider:
    def __init__(self, **kwargs):
        self.model_name = "test-model"
        self.chat_template_kwargs = {"enable_thinking": False}
        self.completions = _Completions(**kwargs)
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )

    def _apply_thinking_disabled(self, params):
        params.setdefault("extra_body", {})["chat_template_kwargs"] = dict(
            self.chat_template_kwargs
        )


ENABLED = {
    "llm_warmup": {
        "enable": True,
        "max_age_seconds": 300,
        "timeout_seconds": 15,
        "grace_seconds": 0,
    }
}
MESSAGES = [{"role": "system", "content": "systeem"}]
TOOLS = [{"type": "function", "function": {"name": "web_search"}}]


@pytest.fixture(autouse=True)
def _clean_state():
    llm_warmup.reset_state()
    yield
    llm_warmup.reset_state()


def _wait():
    thread = llm_warmup._LAST_THREAD
    if thread is not None:
        thread.join(timeout=5)


def test_disabled_by_default():
    provider = _Provider()
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, {}) == "disabled"
    assert provider.completions.calls == []


def test_request_shape_is_streaming_with_at_least_eight_tokens():
    """非流式 max_tokens=1 会挂死 Splash native loop，形状必须流式且 >=8 tokens。"""
    provider = _Provider(prompt_tokens=2341)
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    assert len(provider.completions.calls) == 1
    params = provider.completions.calls[0]
    assert params["model"] == "test-model"
    assert params["stream"] is True
    assert params["max_tokens"] >= 8
    assert params["temperature"] == 0
    assert params["stream_options"] == {"include_usage": True}
    assert params["tools"] == TOOLS
    assert params["timeout"] == 15
    # 前缀消息原样 + 一条一次性用户消息；探针必须是一句自然语言（单词探针会挂死引擎）
    assert params["messages"][:-1] == MESSAGES
    probe = params["messages"][-1]
    assert probe["role"] == "user"
    assert len(probe["content"].split()) >= 3
    assert probe["content"] != "ping"
    # 思考开关必须与真实轮次一致，否则 KV 前缀不通用
    assert params["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    # 流被完整读取并关闭
    assert provider.completions.last_stream.closed is True


def test_same_prefix_is_not_warmed_twice():
    provider = _Provider()
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "recent"
    _wait()
    assert len(provider.completions.calls) == 1
    # 前缀变化（工具集变化）则重新预热
    other = TOOLS + [{"type": "function", "function": {"name": "play_music"}}]
    assert llm_warmup.schedule_warmup(provider, MESSAGES, other, ENABLED) == "scheduled"
    _wait()
    assert len(provider.completions.calls) == 2


def test_single_flight():
    provider = _Provider(delay=0.3)
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "running"
    _wait()
    assert len(provider.completions.calls) == 1


def test_skips_while_real_turn_in_flight():
    provider = _Provider()
    llm_warmup.mark_turn_start()
    try:
        assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "busy"
    finally:
        llm_warmup.mark_turn_end()
    assert provider.completions.calls == []
    # 真实轮次结束后可以正常预热
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    assert len(provider.completions.calls) == 1


def test_pre_send_gate_yields_to_a_turn_that_starts_after_scheduling():
    """调度只做初筛：用户在第一毫秒开口时，发送前必须再次让路。"""
    provider = _Provider()
    config = {"llm_warmup": dict(ENABLED["llm_warmup"], grace_seconds=1.0)}
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, config) == "scheduled"
    llm_warmup.mark_turn_start()
    _wait()
    llm_warmup.mark_turn_end()
    assert provider.completions.calls == []
    # 没有预热成功 → 不留 300s 去重记录，下一轮仍可预热
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    assert len(provider.completions.calls) == 1


def test_aborts_stream_when_a_real_turn_starts():
    """流式读了一半发现用户开口：立即放弃读取、不记为已预热。"""
    provider = _Provider(on_iterate=llm_warmup.mark_turn_start)
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    llm_warmup.mark_turn_end()
    assert len(provider.completions.calls) == 1
    assert provider.completions.last_stream.closed is True
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"


def test_failure_is_silent_and_retryable():
    provider = _Provider(fail=True)
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    # 失败不抛异常、不写去重记录 → 下次仍可调度
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    assert len(provider.completions.calls) == 2


def test_unsupported_provider_without_client():
    provider = SimpleNamespace(model_name="m")
    assert (
        llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "unsupported"
    )


def test_prompt_hash_is_byte_exact_on_the_clock():
    """缓存键必须与发给引擎的字节一致：时钟不同 → 哈希不同 → 新会话重新预热。

    引擎前缀缓存逐 token 匹配；若把时间归一化后去重，新会话会被误判为“recent”而跳过
    预热，首轮永远 cached 0（t_a5523178 fix 2）。
    """
    older = [{"role": "system", "content": "Het is nu 09:15 en je bent RIG."}]
    newer = [{"role": "system", "content": "Het is nu 09:42 en je bent RIG."}]
    assert llm_warmup.prompt_hash(older, TOOLS) != llm_warmup.prompt_hash(newer, TOOLS)
    assert llm_warmup.prompt_hash(older, TOOLS) == llm_warmup.prompt_hash(
        [{"role": "system", "content": "Het is nu 09:15 en je bent RIG."}], TOOLS
    )
    # 会话内冻结时钟（Dialogue._session_hhmm）保证同一会话每一轮都是同一哈希
    other = [{"role": "system", "content": "Het is nu 09:15 en je bent iemand anders."}]
    assert llm_warmup.prompt_hash(older, TOOLS) != llm_warmup.prompt_hash(other, TOOLS)


def test_prefix_signature_matches_warmup_snapshot():
    """审计口径：真实轮次（含用户消息）的前缀指纹必须等于预热快照指纹。"""
    first_turn = list(MESSAGES) + [{"role": "user", "content": "de kat slaapt"}]
    assert llm_warmup.prefix_signature(first_turn, TOOLS) == llm_warmup.payload_signature(
        MESSAGES, TOOLS
    )


def test_turn_counter_never_goes_negative():
    llm_warmup.mark_turn_end()
    assert llm_warmup.is_busy() is False
    llm_warmup.mark_turn_start()
    assert llm_warmup.is_busy() is True
    llm_warmup.mark_turn_end()
    assert llm_warmup.is_busy() is False


# --- 引擎故障退避（t_ae32b506）---------------------------------------------
# 实测：一个 18 token 的请求把 Splash native loop 卡死后，后续 2k 预热排在它后面，
# 30s 后 watchdog 杀掉引擎，真实轮次一起 503。所以引擎一报故障，预热必须停发。


class _Boom:
    """create() 直接抛错，消息里带引擎故障标记。"""

    def __init__(self, message):
        self.message = message
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        raise RuntimeError(self.message)


def test_warmup_failure_with_engine_trouble_starts_backoff():
    provider = _Provider()
    boom = _Boom("APIError: engine_recovering; retry shortly")
    provider.client.chat.completions = boom
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    assert llm_warmup.engine_backoff_remaining() > 0
    # 退避期内不再发请求
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "engine-backoff"
    assert len(boom.calls) == 1


def test_plain_failure_does_not_start_backoff():
    """参数错之类的失败不是引擎故障，不能因此停掉预热。"""
    provider = _Provider()
    provider.client.chat.completions = _Boom("BadRequestError: 400 unknown field")
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    assert llm_warmup.engine_backoff_remaining() == 0
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"


def test_backoff_blocks_warmup_then_engine_ok_clears_it():
    llm_warmup.note_engine_trouble("Splash returned HTTP 503 runtime_unavailable")
    assert llm_warmup.schedule_warmup(_Provider(), MESSAGES, TOOLS, ENABLED) == "engine-backoff"
    llm_warmup.note_engine_ok()
    assert llm_warmup.engine_backoff_remaining() == 0
    assert llm_warmup.schedule_warmup(_Provider(), MESSAGES, TOOLS, ENABLED) == "scheduled"


def test_backoff_expires_and_is_capped():
    old_base, old_max = llm_warmup._BACKOFF_BASE, llm_warmup._BACKOFF_MAX
    llm_warmup._BACKOFF_BASE, llm_warmup._BACKOFF_MAX = 1.0, 3.0
    try:
        llm_warmup.note_engine_trouble("x")
        first = llm_warmup.engine_backoff_remaining()
        llm_warmup.note_engine_trouble("x")
        second = llm_warmup.engine_backoff_remaining()
        for _ in range(4):
            llm_warmup.note_engine_trouble("x")
        capped = llm_warmup.engine_backoff_remaining()
        assert 0.5 < first <= 1.0
        assert 1.5 < second <= 2.0
        assert 2.5 < capped <= 3.0
        assert llm_warmup.engine_backoff_streak() == 6
    finally:
        llm_warmup._BACKOFF_BASE, llm_warmup._BACKOFF_MAX = old_base, old_max


def test_trouble_classification():
    assert llm_warmup.is_engine_trouble_error(RuntimeError("native loop did not answer"))
    assert llm_warmup.is_engine_trouble_error(RuntimeError("engine stopped unexpectedly"))
    assert not llm_warmup.is_engine_trouble_error(RuntimeError("bad request"))


# --- 探针尾部 >=64 未缓存行（t_ae32b506，机理 t_d5763018）---------------------
# Splash engine 在未缓存 prefill 行数落在 18–29 行的小区间时同步编译 Metal pipeline，
# XPC 回复不返回 → 30s 后 watchdog 杀掉引擎。预热尾部本身只有 ~16 行，落在致命带边缘；
# 同一分钟内两个会话前缀相同（`{{current_time}}` 按会话冻结）时未缓存行数就只剩这个尾部。

FATAL_BAND_LOW = 18  # 实测致命区间下沿
FATAL_BAND_HIGH = 29  # 实测致命区间上沿


def _probe_of(provider):
    return provider.completions.calls[0]["messages"][-1]["content"]


def test_probe_tail_is_clear_of_the_fatal_small_row_band():
    provider = _Provider()
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, ENABLED) == "scheduled"
    _wait()
    probe = _probe_of(provider)
    tokens = llm_warmup._estimate_probe_tokens(probe)
    assert tokens >= llm_warmup.WARMUP_PROBE_MIN_TOKENS
    # 至少是致命区间上限的两倍，并且明显高于它（不是刚好贴边）
    assert tokens >= 2 * FATAL_BAND_HIGH
    # 仍然是一句自然语言 + 散文，不是单词探针
    assert len(probe.split()) >= 20


def test_short_configured_probe_text_is_padded_to_the_floor():
    """运维把 probe_text 写成 'hallo' 也不能退回到致命的小行区间。"""
    config = {"llm_warmup": dict(ENABLED["llm_warmup"], probe_text="hallo")}
    provider = _Provider()
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, config) == "scheduled"
    _wait()
    probe = _probe_of(provider)
    assert probe.startswith("Warmup ")
    assert probe.rstrip().endswith(llm_warmup.WARMUP_PROBE_FILLER.strip()[-20:])
    assert llm_warmup._estimate_probe_tokens(probe) >= llm_warmup.WARMUP_PROBE_MIN_TOKENS


def test_probe_nonce_differs_so_repeat_warmups_never_shrink_to_the_band():
    """前缀命中缓存时未缓存行数只剩探针尾部；nonce 让它每次都够长。"""
    probes = {llm_warmup.build_probe_text() for _ in range(5)}
    assert len(probes) == 5
    for probe in probes:
        assert llm_warmup._estimate_probe_tokens(probe) >= 2 * FATAL_BAND_HIGH


def test_min_probe_tokens_config_can_raise_but_not_lower_the_floor():
    config = {"llm_warmup": dict(ENABLED["llm_warmup"], min_probe_tokens=256)}
    provider = _Provider()
    assert llm_warmup.schedule_warmup(provider, MESSAGES, TOOLS, config) == "scheduled"
    _wait()
    assert llm_warmup._estimate_probe_tokens(_probe_of(provider)) >= 250
    # 低于地板的值被忽略，不会把探针缩回致命带
    assert (
        llm_warmup._estimate_probe_tokens(llm_warmup.build_probe_text(min_tokens=4))
        >= llm_warmup.WARMUP_PROBE_MIN_TOKENS
    )
