"""设备端MCP客户端定义"""

import asyncio
import threading
import time
from concurrent.futures import Future
from core.utils.util import sanitize_tool_name
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

# t_a2a3bf8a: 设备把工具列表分页推送（nextCursor）。工具调用若落在这段窗口里，
# “工具不存在”的结论要延后有界地给出，而不是立刻回错。
PENDING_TOOL_WAIT_SECONDS = 8.0  # 单次工具调用最多为“还没收全”等多久
PAGE_STALL_SECONDS = 4.0  # 页面停止到达超过这么久就不再等


class MCPClient:
    """设备端MCP客户端，用于管理MCP状态和工具"""

    def __init__(self):
        self.tools = {}  # sanitized_name -> tool_data
        self.name_mapping = {}
        self.ready = False
        self.call_results = {}  # To store Futures for tool call responses
        # Handshake hardcodes id=1 (initialize) and id=2 (tools/list) in mcp_handler.
        # JSON-RPC 2.0 forbids reusing request ids on the same connection, and
        # compliant device MCP servers reject tools/call with id 1/2 (see #3461).
        # Official protocol sequence expects the first tools/call to use id=3.
        self.next_id = 3
        self.lock = asyncio.Lock()
        self._cached_available_tools = None  # Cache for get_available_tools

        # t_a2a3bf8a: 设备把工具列表分页推送（每页带 nextCursor）。最后一页到达前
        # self.tools 只能是半成品：重启后首轮曾只看到 39 个里的 8 个（甚至 0 个设备
        # 工具），模型于是调用一个服务器还不认识的工具，被回“工具不存在”。
        # 因此每页先暂存，收到不带 nextCursor 的最后一页时一次性**原子发布**。
        self._staged_tools = {}  # sanitized_name -> tool_data（未发布）
        self._staged_name_mapping = {}
        self.tools_complete = False  # 完整工具列表已发布
        self.tools_ready_event = threading.Event()  # 供 chat() 工作线程有界等待
        self._last_page_at = None  # 最近一页到达时间（monotonic）

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def get_available_tools(self) -> list:
        # Check if the cache is valid
        if self._cached_available_tools is not None:
            return self._cached_available_tools

        # If cache is not valid, regenerate the list
        result = []
        for tool_name, tool_data in self.tools.items():
            function_def = {
                "name": tool_name,
                "description": tool_data["description"],
                "parameters": {
                    "type": tool_data["inputSchema"].get("type", "object"),
                    "properties": tool_data["inputSchema"].get("properties", {}),
                    "required": tool_data["inputSchema"].get("required", []),
                },
            }
            result.append({"type": "function", "function": function_def})

        self._cached_available_tools = result  # Store the generated list in cache
        return result

    async def is_ready(self) -> bool:
        async with self.lock:
            return self.ready

    async def set_ready(self, status: bool):
        async with self.lock:
            self.ready = status

    async def add_tool(self, tool_data: dict):
        async with self.lock:
            sanitized_name = sanitize_tool_name(tool_data["name"])
            self.tools[sanitized_name] = tool_data
            self.name_mapping[sanitized_name] = tool_data["name"]
            self._cached_available_tools = (
                None  # Invalidate the cache when a tool is added
            )

    # ---- t_a2a3bf8a: 分页工具列表的暂存 / 发布 / 等待 ----

    def staged_tool_count(self) -> int:
        """已暂存但尚未发布的工具数量。"""
        return len(self._staged_tools)

    async def wait_for_pending_tools(
        self,
        max_wait: float = PENDING_TOOL_WAIT_SECONDS,
        stall: float = PAGE_STALL_SECONDS,
    ) -> bool:
        """工具列表还没收全时，为“工具不存在”这个结论有界地多等一会儿。

        重启后用户的第一句话常常比设备最后几页工具先到；这时直接回
        “工具不存在”是错的。这里最多等 max_wait 秒，且一旦页面停止到达
        超过 stall 秒就立刻放弃（设备卡住时不再拖时间）。
        """
        if self.tools_complete:
            return True
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + max_wait
        while not self.tools_complete:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # 用线程池里的 Event.wait 分片等待：既真正让出事件循环（别的协程能继续
            # 收页），又能被 mark_tools_complete() 立刻唤醒。
            slice_s = min(0.2, remaining)
            if await loop.run_in_executor(None, self.tools_ready_event.wait, slice_s):
                return True
            if self._last_page_at is not None:
                if time.monotonic() - self._last_page_at > stall:
                    break
        return self.tools_complete

    async def stage_tool(self, tool_data: dict):
        """暂存一页里的工具：只进 self._staged_*，不对外可见。"""
        async with self.lock:
            sanitized_name = sanitize_tool_name(tool_data["name"])
            self._staged_tools[sanitized_name] = tool_data
            self._staged_name_mapping[sanitized_name] = tool_data["name"]
            self._last_page_at = time.monotonic()

    async def publish_staged_tools(self, merge: bool = False) -> dict:
        """把所有已暂存的页一次性发布为对外工具列表（原子替换）。

        merge=True 只用于“超时兜底已发布过部分列表，后续真正的最后一页又到了”
        这一条路径，此时把新暂存的工具并进已发布的集合，避免丢掉前面的页。
        """
        async with self.lock:
            if merge:
                self.tools.update(self._staged_tools)
                self.name_mapping.update(self._staged_name_mapping)
            else:
                self.tools = self._staged_tools
                self.name_mapping = self._staged_name_mapping
            self._staged_tools = {}
            self._staged_name_mapping = {}
            self._cached_available_tools = None
            return dict(self.tools)

    def mark_tools_complete(self):
        """标记完整列表已发布，并唤醒在 chat() 工作线程里等待的首轮请求。"""
        self.tools_complete = True
        self.tools_ready_event.set()

    def wait_for_tools(self, timeout: float) -> bool:
        """阻塞等待完整工具列表（chat() 在工作线程里跑，所以用 threading.Event）。"""
        if self.tools_complete:
            return True
        return self.tools_ready_event.wait(timeout)

    async def get_next_id(self) -> int:
        async with self.lock:
            current_id = self.next_id
            self.next_id += 1
            return current_id

    async def register_call_result_future(self, id: int, future: Future):
        async with self.lock:
            self.call_results[id] = future

    async def resolve_call_result(self, id: int, result: any):
        async with self.lock:
            if id in self.call_results:
                future = self.call_results.pop(id)
                if not future.done():
                    future.set_result(result)

    async def reject_call_result(self, id: int, exception: Exception):
        async with self.lock:
            if id in self.call_results:
                future = self.call_results.pop(id)
                if not future.done():
                    future.set_exception(exception)

    async def cleanup_call_result(self, id: int):
        async with self.lock:
            if id in self.call_results:
                self.call_results.pop(id)
