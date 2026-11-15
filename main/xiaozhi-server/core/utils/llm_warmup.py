"""进程级 LLM 前缀预热（first-turn prefix warmup）。

问题：会话建立后第一轮真实请求携带的前缀（system + few-shot + tools）不在推理引擎的
前缀缓存里时，prefill 是线性的（实测 Nail/Qwen3.6 ~1.01 ms/token，2.3k 前缀 ≈ 2.3s）。

设计约束（kanban t_5cd58f74 → t_a5523178）：
- 后台 daemon 线程执行，绝不阻塞连接初始化线程，也不被会话 await；
- 进程内单飞：同一时刻只允许一次预热；
- 缓存键 = 发给引擎的**逐字节**消息+工具哈希：`{{current_time}}` 按会话冻结（见
  Dialogue._session_hhmm），因此同一会话内前缀恒定；跨会话前缀不同，必须各自预热，
  300s 去重只在字节完全相同（重连同一会话）时生效；
- **请求形状必须流式且 max_tokens>=8**：oMLX/Splash（Nail）上非流式 max_tokens=1 会把
  native loop 挂死，30s 后 watchdog 杀掉引擎，所有并发真实轮次变成 503；
- **探针消息必须是一句自然语言**：带 8 个工具的 2.3k 前缀 + 单词探针（"ping"/"hallo"）
  同样会挂死 native loop（实测 5/5），改成一句陈述句后冷/热均正常（3/3）；
- **前缀必须是首轮真实请求的前缀**：调用方传入的 messages 必须与首轮 chat() 构造的
  messages 完全相同（同一 system、同一 few-shot、同一 tools），并记录哈希/字符数以便审计；
- 引擎 max_concurrent_requests=1：发请求前（含调度后的宽限期）再检查一次是否有真实
  轮次在途，流式生成过程中若真实轮次开始则立即放弃读取，避免占住唯一的引擎槽位；
- **引擎一旦报故障就退让（t_ae32b506）**：`engine_recovering` / `runtime_unavailable`
  / 5xx / RemoteProtocolError / 超时 都说明 native loop 已经卡住，此时再发预热只会
  排在卡死的请求后面（实测：18 token 的小请求把 loop 卡死后，2k 预热排队 30s，
  随后 watchdog 杀掉引擎）。因此记一个指数退避截止时间，期间任何预热直接跳过；
  真实轮次成功后再清除退避。预热自身失败也计入故障；
- **探针尾部必须有 >=64 个未缓存 prefill 行（t_ae32b506 / t_d5763018）**：Splash native engine
  在 transport 线程上同步编译 Metal compute pipeline，未缓存行数落在 18–29 行的小区间时
  MTLCompilerService 的 XPC 回复永远不回来，循环答不上 STATUS_REQUEST，30s 后 watchdog
  杀掉引擎（实测 18/23/29 行 3/3 致命；35/36/55/64/120 行 0.2–0.7s 正常）。预热的尾部
  （probe 句子 + 生成提示）实测只有 16 行，离致命下沿 18 只差 2 行，而且同一分钟内两个会话
  的 `{{current_time}}` 相同 → 前缀命中缓存后未缓存行数就只剩尾部。所以探针 = 一次性 nonce
  + 一句自然语言 + 填充散文，凑够 >=64 token 的尾部（填充在末尾，不影响真实首轮的前缀命中）；
- 超时（timeout_seconds）后放弃；
- 任何失败只记日志，绝不影响会话。

**当前状态（t_ae32b506，2026-11-15）：在 Splash 1.6.0-m1 上 `enable: false`。**
引擎的整段前缀缓存只对逐字节相同的提示词命中，预热之后真实首轮只能复用 2,016/2,048 行，
剩下的 32 行小 prefill 实测要 1.8 s（18–29 行直接触发 watchdog），所以预热拿不到
turn-1 的收益、还占用单槽约 2.1 s。理由与实测见 `data/.config.yaml` 的 `llm_warmup`
注释；换到前缀缓存行为正常的引擎后可以安全地重新打开（探针守卫仍然有效）。

配置（data/.config.yaml）：
    llm_warmup:
      enable: true
      max_age_seconds: 300
      timeout_seconds: 8      # 小于引擎 30s watchdog，别占着卡死的槽位
      grace_seconds: 1.0     # 调度后给用户/真实轮次让路的宽限时间
      backoff_seconds: 30    # 引擎报故障后停发预热的初始时长
      backoff_max_seconds: 300
"""

import hashlib
import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

from config.logger import setup_logging

TAG = "warmup"

logger = setup_logging()

# 预热请求的最小生成预算：Splash 上 max_tokens>=8 + stream=True 是实测安全的形状。
WARMUP_MAX_TOKENS = 8

# 预热请求末尾的“探针”用户消息。它不是前缀的一部分（前缀是它之前的 system+tools+few-shot），
# 但会决定引擎走哪条解码路径：单词级探针（"ping"、"hallo"）在带 8 个工具的 2.3k 前缀上
# 会让 Splash native loop 挂死在 prefill/首个解码周期（无 TTFT，30s 后 Engine failed），
# 已复现 5/5；一句自然的荷兰语陈述句同形状可正常返回（TTFT 2.2s，冷/热各 3/3）。
# 因此默认探针用一句话，而不是单词。
WARMUP_PROBE_TEXT = "de kat slaapt lekker op de warme bank, wat vind je daarvan"

# 探针尾部的**最小未缓存 prefill 行数**（t_ae32b506，机理见 t_d5763018）。
# Splash native engine 在 transport 线程上同步创建 Metal compute pipeline：未缓存行数落在
# 18–29 行的小区间时 MTLCompilerService 的 XPC 回复不返回，循环答不上 STATUS_REQUEST，
# 30s 后 watchdog 杀掉引擎（实测 18/23/29 行 3/3 致命，35/36/55/64/120 行 0.2–0.7s 正常）。
# 预热尾部实测 16 行（`input 2,064 · cached 2,048`），离致命下沿只差 2 行；且同一分钟内两个
# 会话的 `{{current_time}}` 相同 → 前缀命中缓存后未缓存行数就只剩这个尾部。
# 因此探针尾部必须 >=64 token（约致命区间上限的两倍）：nonce 保证重复预热也不会因前缀
# 命中缩到小行数，填充散文保证即使前缀全部命中也有 >=64 新行。
WARMUP_PROBE_MIN_TOKENS = 64
WARMUP_PROBE_CHARS_PER_TOKEN = 4
WARMUP_PROBE_TEMPLATE_TOKENS = 10
WARMUP_PROBE_FILLER = (
    "Deze technische testaanvraag controleert of het lokale taalmodel bereikbaar "
    "is en of het een gewone vraag normaal beantwoordt. De inhoud van de vraag "
    "maakt daarvoor niet uit. "
)

# 单飞锁：保护下面全部状态
_LOCK = threading.Lock()
# prompt_hash -> 上次预热成功的 monotonic 时间
_WARMED: Dict[str, float] = {}
# 是否已有一次预热在跑（进程内单飞）
_RUNNING = False
# 在途的真实 LLM 轮次计数
_INFLIGHT = 0
# 真实轮次世代号：每次 mark_turn_start() 自增。预热在发送前/读取中比较世代号，
# 变化即说明用户已经开口（哪怕该轮次已结束），本次预热让路。
_TURN_EPOCH = 0
# 预热线程句柄（仅用于测试/观测）
_LAST_THREAD: Optional[threading.Thread] = None
# 引擎故障退避（t_ae32b506）：monotonic 截止时间与连续故障次数
_ENGINE_BACKOFF_UNTIL = 0.0
_ENGINE_BACKOFF_STREAK = 0
_ENGINE_LAST_TROUBLE: Optional[str] = None
# 退避参数：遇到配置时由 _read_config 覆盖
_BACKOFF_BASE = 30.0
_BACKOFF_MAX = 300.0

# 引擎"已经卡住"的征兆。命中即退避：此时发预热只会排在卡死的 native loop 后面。
ENGINE_TROUBLE_MARKERS = (
    "engine_recovering",
    "runtime_unavailable",
    "engine stopped unexpectedly",
    "native loop did not answer",
    "recovering; retry shortly",
)

STATUS_DISABLED = "disabled"
STATUS_UNSUPPORTED = "unsupported"
STATUS_RUNNING = "running"
STATUS_RECENT = "recent"
STATUS_BUSY = "busy"
STATUS_SCHEDULED = "scheduled"
STATUS_BACKOFF = "engine-backoff"
STATUS_FAILED = "failed"


def mark_turn_start() -> None:
    """真实 LLM 请求（流式生成器）开始时调用。"""
    global _INFLIGHT, _TURN_EPOCH
    with _LOCK:
        _INFLIGHT += 1
        _TURN_EPOCH += 1


def mark_turn_end() -> None:
    """真实 LLM 请求结束时调用（生成器 finally 中）。"""
    global _INFLIGHT
    with _LOCK:
        if _INFLIGHT > 0:
            _INFLIGHT -= 1


def is_busy() -> bool:
    with _LOCK:
        return _INFLIGHT > 0


def turn_epoch() -> int:
    with _LOCK:
        return _TURN_EPOCH


def _classify_trouble(text: str) -> Optional[str]:
    low = (text or "").lower()
    for marker in ENGINE_TROUBLE_MARKERS:
        if marker in low:
            return marker
    return None


def note_engine_trouble(reason: str) -> float:
    """引擎报故障/卡住时调用（预热失败或真实轮次失败）。

    记录指数退避截止时间：退避期内任何预热都不发，避免把自己的请求排到已经卡死的
    native loop 后面 —— 那正是 t_ae32b506 里引擎被 30s watchdog 杀掉、真实轮次
    跟着变成 503 的放大路径。返回退避截止的 monotonic 时间。
    """
    global _ENGINE_BACKOFF_UNTIL, _ENGINE_BACKOFF_STREAK, _ENGINE_LAST_TROUBLE
    with _LOCK:
        _ENGINE_BACKOFF_STREAK += 1
        _ENGINE_LAST_TROUBLE = str(reason)[:200]
        delay = min(_BACKOFF_BASE * (2 ** (_ENGINE_BACKOFF_STREAK - 1)), _BACKOFF_MAX)
        _ENGINE_BACKOFF_UNTIL = time.monotonic() + delay
        streak = _ENGINE_BACKOFF_STREAK
        text = _ENGINE_LAST_TROUBLE
    logger.bind(tag=TAG).info(
        f"warmup: engine backoff seconds={delay:.0f} streak={streak} reason={text[:120]}"
    )
    return _ENGINE_BACKOFF_UNTIL


def note_engine_ok() -> None:
    """引擎正常响应后调用：清除退避，下次会话可以正常预热。"""
    global _ENGINE_BACKOFF_UNTIL, _ENGINE_BACKOFF_STREAK, _ENGINE_LAST_TROUBLE
    with _LOCK:
        had = _ENGINE_BACKOFF_STREAK
        _ENGINE_BACKOFF_UNTIL = 0.0
        _ENGINE_BACKOFF_STREAK = 0
        _ENGINE_LAST_TROUBLE = None
    if had:
        logger.bind(tag=TAG).info(f"warmup: engine backoff cleared (was streak={had})")


def engine_backoff_remaining() -> float:
    with _LOCK:
        return max(0.0, _ENGINE_BACKOFF_UNTIL - time.monotonic())


def engine_backoff_streak() -> int:
    with _LOCK:
        return _ENGINE_BACKOFF_STREAK


def _is_engine_trouble(exc: Exception) -> bool:
    """区分"引擎卡住/不可用"与"这次预热本身没意义"（例如 400 参数错）。"""
    text = f"{type(exc).__name__}: {exc}"
    if _classify_trouble(text):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    low = text.lower()
    if "timeout" in low or "timed out" in low:
        return True
    return "connection" in type(exc).__name__.lower()


def is_engine_trouble_error(exc: Exception) -> bool:
    """供真实轮次调用：这次失败是否说明引擎卡住/不可用。"""
    return _is_engine_trouble(exc)


def prompt_hash(messages: List[Dict[str, Any]], tools: Optional[List[Dict]]) -> str:
    """前缀指纹 = 引擎真正的缓存键：消息全文 + 工具定义，**逐字节**哈希。

    曾经把 HH:MM(:SS) 归一化后再哈希，好让 300s 去重跨会话生效；但引擎的前缀缓存是
    逐 token 的：`{{current_time}}` 按分钟渲染导致新会话的前缀与上一会话不同，归一化
    哈希会把新会话判为“recent”而跳过预热，于是该会话首轮永远 cached 0（t_a5523178）。
    因此缓存键必须与发给引擎的字节一致。
    """
    payload = json.dumps(
        {"messages": messages, "tools": tools or []},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def dump_payload(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]],
    tag: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """把将要发给引擎的 payload 原样落盘，供逐字节审计（`llm_warmup.dump_payload: true`）。"""
    cfg = (config or {}).get("llm_warmup") or {}
    if not cfg.get("dump_payload"):
        return None
    data_dir = ((config or {}).get("server") or {}).get("data_dir") or "data"
    path = os.path.join(data_dir, f".llm_warmup_{tag}.json")
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"tag": tag, "messages": messages, "tools": tools or []},
                handle,
                ensure_ascii=False,
                indent=1,
            )
        logger.bind(tag=TAG).info(f"warmup: dumped tag={tag} path={path}")
        return path
    except Exception as exc:  # noqa: BLE001 - 只影响可观测性
        logger.bind(tag=TAG).info(f"warmup: dump failed tag={tag} {exc}")
        return None


def payload_signature(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """首轮前缀的可审计指纹：哈希 + 消息数 + 工具数 + 序列化字符数。"""
    payload = json.dumps(
        {"messages": messages, "tools": tools or []},
        ensure_ascii=False,
        default=str,
    )
    return {
        "hash": prompt_hash(messages, tools),
        "messages": len(messages),
        "tools": len(tools or []),
        "chars": len(payload),
    }


def prefix_signature(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """真实轮次消息列表的前缀指纹：去掉末尾的用户消息后与预热快照同口径。"""
    prefix = list(messages or [])
    while prefix and _role_of(prefix[-1]) == "user":
        prefix.pop()
    return payload_signature(prefix, tools)


def _role_of(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def estimate_probe_tokens(text: str) -> int:
    """探针文本的保守 token 估算（混合荷兰语 + chat 模板自带的生成提示）。

    公开名：`tools/warmup_providers.py` 的容器启动预热必须与这里同一探针形状，
    两处各自维护估算会让“>=64 未缓存行”的守卫悄悄失配（t_3efccbba）。
    """
    return WARMUP_PROBE_TEMPLATE_TOKENS + len(text) // WARMUP_PROBE_CHARS_PER_TOKEN


# 向后兼容：历史调用点/测试仍用私有名
_estimate_probe_tokens = estimate_probe_tokens


def build_probe_text(
    base: Optional[str] = None,
    min_tokens: Optional[int] = None,
    nonce: Optional[str] = None,
) -> str:
    """构造预热探针：一次性 nonce + 一句自然语言 + 填充散文，凑够 >=64 token 尾部。

    nonce 让重复预热不会因为前缀命中缓存而把未缓存行数缩到致命的小行区间；
    填充追加在**末尾**，所以真实首轮的前缀命中（probe 之前的部分）不受影响。
    """
    floor = WARMUP_PROBE_MIN_TOKENS
    try:
        configured = int(min_tokens) if min_tokens is not None else 0
    except (TypeError, ValueError):
        configured = 0
    floor = max(floor, configured)
    if nonce is None:
        nonce = "Warmup %s %06x." % (
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            random.getrandbits(24),
        )
    text = f"{nonce} {str(base or WARMUP_PROBE_TEXT).strip()}"
    while _estimate_probe_tokens(text) < floor:
        text = text + " " + WARMUP_PROBE_FILLER
    return text


def _warmup_params(provider, messages, tools, timeout_seconds) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": getattr(provider, "model_name", None),
        "messages": messages,
        "max_tokens": WARMUP_MAX_TOKENS,
        "temperature": 0,
        # 流式：Splash 上非流式 max_tokens=1 会挂死 native loop（30s watchdog）。
        "stream": True,
        # 从同一响应里取 prompt_tokens / cached_tokens，便于审计前缀是否真的命中。
        "stream_options": {"include_usage": True},
        "timeout": timeout_seconds,
    }
    if tools:
        params["tools"] = tools
    # 与真实轮次保持一致的思考开关 / 模板参数，否则前缀 KV 不通用
    apply_thinking = getattr(provider, "_apply_thinking_disabled", None)
    if callable(apply_thinking):
        try:
            apply_thinking(params)
        except Exception as exc:  # noqa: BLE001 - 预热绝不影响会话
            logger.bind(tag=TAG).debug(f"warmup: thinking 参数注入失败 {exc}")
    chat_template_kwargs = getattr(provider, "chat_template_kwargs", None)
    if chat_template_kwargs:
        params.setdefault("extra_body", {})["chat_template_kwargs"] = (
            chat_template_kwargs
        )
    return params


def _usage_field(usage, name, default=None):
    if usage is None:
        return default
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    return default if value is None else value


def _wait_grace(grace_seconds: float, epoch: int) -> Optional[str]:
    """发送前的让路窗口：等待期间出现真实轮次/世代变化则返回跳过原因。"""
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while True:
        if is_busy():
            return "real-turn-inflight"
        if turn_epoch() != epoch:
            return "turn-epoch-changed"
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _do_warmup(provider, params, key, timeout_seconds, epoch, grace_seconds) -> None:
    global _RUNNING
    log = logger.bind(tag=TAG)
    try:
        client = getattr(provider, "client", None)
        if client is None:
            log.info(f"warmup: skip reason=no-client hash={key}")
            return
        # 调度只做初筛；这里是真正发请求前的第二次闸门（用户可能刚好开口）。
        skip = _wait_grace(grace_seconds, epoch)
        if skip is not None:
            log.info(f"warmup: skip reason={skip} hash={key} stage=pre-send")
            return

        started = time.monotonic()
        stream = client.chat.completions.create(**params)
        usage = None
        chunks = 0
        ttft_ms: Optional[int] = None
        aborted: Optional[str] = None
        try:
            for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        chunks += 1
                        if ttft_ms is None:
                            ttft_ms = int((time.monotonic() - started) * 1000)
                        # 真实轮次开始了：立刻停止读取，别占着唯一引擎槽位。
                        if is_busy():
                            aborted = "real-turn-inflight"
                            break
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - 关闭失败无关紧要
                    pass

        elapsed_ms = int((time.monotonic() - started) * 1000)
        input_tokens = _usage_field(usage, "prompt_tokens")
        details = _usage_field(usage, "prompt_tokens_details")
        cached_tokens = _usage_field(details, "cached_tokens")
        if aborted is not None:
            log.info(
                f"warmup: aborted hash={key} reason={aborted} chunks={chunks} "
                f"ttft_ms={ttft_ms} elapsed_ms={elapsed_ms}"
            )
            return
        log.info(
            f"warmup: ok hash={key} input_tokens={input_tokens} "
            f"cached_tokens={cached_tokens} chunks={chunks} ttft_ms={ttft_ms} "
            f"elapsed_ms={elapsed_ms} timeout_ms={int(timeout_seconds * 1000)}"
        )
        with _LOCK:
            _WARMED[key] = time.monotonic()
        # 引擎这次答上了：解除退避（t_ae32b506）
        note_engine_ok()
    except Exception as exc:  # noqa: BLE001 - 预热失败必须静默
        log.info(f"warmup: failed hash={key} {type(exc).__name__}: {exc}")
        # 引擎卡住/不可用就退让，别再往已经卡死的 native loop 后面排队
        if _is_engine_trouble(exc):
            note_engine_trouble(f"{type(exc).__name__}: {exc}")
    finally:
        with _LOCK:
            _RUNNING = False


def schedule_warmup(
    provider,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """调度一次前缀预热；返回便于测试/日志的状态字符串。

    provider: 已初始化的 LLM provider（需具备 .client 与 .model_name）
    messages: 首轮真实请求的前缀消息（system + few-shot），不含用户消息；
              必须与首轮 chat() 构造的消息逐字节相同
    tools:    首轮真实请求的 tools 定义
    config:   全量配置，读取 llm_warmup 开关
    """
    global _RUNNING, _LAST_THREAD

    cfg = (config or {}).get("llm_warmup") or {}
    log = logger.bind(tag=TAG)
    if not cfg.get("enable"):
        log.info("warmup: skip reason=disabled")
        return STATUS_DISABLED

    client = getattr(provider, "client", None)
    model = getattr(provider, "model_name", None)
    if client is None or not model:
        log.info("warmup: skip reason=unsupported (no openai-compatible client)")
        return STATUS_UNSUPPORTED

    try:
        max_age = float(cfg.get("max_age_seconds", 300) or 300)
    except (TypeError, ValueError):
        max_age = 300.0
    try:
        timeout_seconds = float(cfg.get("timeout_seconds", 15) or 15)
    except (TypeError, ValueError):
        timeout_seconds = 15.0
    try:
        grace_seconds = float(cfg.get("grace_seconds", 1.0) or 0.0)
    except (TypeError, ValueError):
        grace_seconds = 1.0

    global _BACKOFF_BASE, _BACKOFF_MAX
    try:
        _BACKOFF_BASE = float(cfg.get("backoff_seconds", _BACKOFF_BASE) or _BACKOFF_BASE)
    except (TypeError, ValueError):
        pass
    try:
        _BACKOFF_MAX = float(cfg.get("backoff_max_seconds", _BACKOFF_MAX) or _BACKOFF_MAX)
    except (TypeError, ValueError):
        pass

    probe_text = build_probe_text(
        cfg.get("probe_text") or WARMUP_PROBE_TEXT,
        cfg.get("min_probe_tokens"),
    )
    request_messages = list(messages) + [{"role": "user", "content": probe_text}]
    signature = payload_signature(messages, tools)
    key = signature["hash"]
    epoch = turn_epoch()

    # 引擎刚报过故障（engine_recovering / runtime_unavailable / 卡死超时）：
    # 此时发预热只会排到卡死的 native loop 后面，直接跳过（t_ae32b506）。
    remaining = engine_backoff_remaining()
    if remaining > 0:
        log.info(
            f"warmup: skip reason=engine-backoff remaining_s={remaining:.0f} "
            f"streak={engine_backoff_streak()} hash={key}"
        )
        return STATUS_BACKOFF

    with _LOCK:
        if _INFLIGHT > 0:
            log.info(f"warmup: skip reason=real-turn-inflight hash={key} stage=schedule")
            return STATUS_BUSY
        if _RUNNING:
            log.info(f"warmup: skip reason=single-flight hash={key}")
            return STATUS_RUNNING
        last = _WARMED.get(key)
        if last is not None and (time.monotonic() - last) < max_age:
            log.info(
                f"warmup: skip reason=recent hash={key} age_s={time.monotonic() - last:.1f}"
            )
            return STATUS_RECENT
        _RUNNING = True

    params = _warmup_params(provider, request_messages, tools, timeout_seconds)
    log.info(
        f"warmup: scheduled hash={key} messages={signature['messages']} "
        f"tools={signature['tools']} chars={signature['chars']} "
        f"payload_messages={len(request_messages)} max_tokens={WARMUP_MAX_TOKENS} "
        f"probe_chars={len(probe_text)} probe_tokens≈{_estimate_probe_tokens(probe_text)} "
        f"stream=true epoch={epoch} grace_ms={int(grace_seconds * 1000)}"
    )
    thread = threading.Thread(
        target=_do_warmup,
        args=(provider, params, key, timeout_seconds, epoch, grace_seconds),
        name=f"llm-warmup-{key}",
        daemon=True,
    )
    with _LOCK:
        _LAST_THREAD = thread
    thread.start()
    return STATUS_SCHEDULED


def reset_state() -> None:
    """仅供测试：清空预热状态。"""
    global _RUNNING, _INFLIGHT, _LAST_THREAD, _TURN_EPOCH
    global _ENGINE_BACKOFF_UNTIL, _ENGINE_BACKOFF_STREAK, _ENGINE_LAST_TROUBLE
    with _LOCK:
        _WARMED.clear()
        _RUNNING = False
        _INFLIGHT = 0
        _TURN_EPOCH = 0
        _LAST_THREAD = None
        _ENGINE_BACKOFF_UNTIL = 0.0
        _ENGINE_BACKOFF_STREAK = 0
        _ENGINE_LAST_TROUBLE = None
