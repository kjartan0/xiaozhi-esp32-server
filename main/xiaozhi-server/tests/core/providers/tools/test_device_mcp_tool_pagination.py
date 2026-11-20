"""设备 MCP 工具列表分页：完整前不发布，收全后一次性发布（t_a2a3bf8a）。

线上故障：设备把 ~40 个工具按 nextCursor 分两页推送，旧实现边收边发布，
重启后首轮只看到第一页的 8 个工具，模型调用 ``self_arm_look_at_person``
（在第二页）时服务器还不认识它，于是回了“工具不存在”。
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from core.providers.tools.base import ToolDefinition, ToolExecutor, ToolType
from core.providers.tools.device_mcp.mcp_client import MCPClient
from core.providers.tools.unified_tool_manager import ToolManager
from plugins_func.register import Action, ActionResponse


def _tool(name: str, description: str = "") -> dict:
    return {
        "name": name,
        "description": description or f"desc of {name}",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    }


PAGE_1 = ["self.get_device_status", "self.arm.wag_tail", "self.audio_speaker.set_volume"]
PAGE_2 = ["self.arm.look_at_person", "self.camera.take_photo"]


async def _sleep(seconds: float) -> None:
    """conftest 在 mock 模式把 asyncio.sleep 换成了不让出循环的桩，这里放到线程池里睡。"""
    await asyncio.get_running_loop().run_in_executor(None, time.sleep, seconds)


async def test_pages_are_staged_until_the_last_page():
    client = MCPClient()

    for name in PAGE_1:
        await client.stage_tool(_tool(name))

    # 半成品绝不能对外可见：这正是“8/39”那个故障的根源。
    assert client.staged_tool_count() == len(PAGE_1)
    assert client.tools == {}
    assert client.name_mapping == {}
    assert client.has_tool("self_get_device_status") is False
    assert client.has_tool("self_arm_look_at_person") is False
    assert client.tools_complete is False
    assert client.wait_for_tools(0.01) is False

    # 最后一页到达：两页一起、一次性发布。
    for name in PAGE_2:
        await client.stage_tool(_tool(name))
    published = await client.publish_staged_tools()
    client.mark_tools_complete()

    assert client.tools_complete is True
    assert len(published) == len(PAGE_1) + len(PAGE_2)
    assert client.has_tool("self_arm_look_at_person") is True
    assert client.has_tool("self_camera_take_photo") is True
    assert client.name_mapping["self_arm_look_at_person"] == "self.arm.look_at_person"
    # 暂存区已清空，不会在下一轮重复发布。
    assert client.staged_tool_count() == 0


async def test_first_turn_wait_wakes_up_when_last_page_arrives():
    """首轮在工作线程里等；最后一页发布后必须立刻被唤醒。"""

    client = MCPClient()
    for name in PAGE_1:
        await client.stage_tool(_tool(name))

    result = {}

    def waiter():
        started = time.monotonic()
        result["ready"] = client.wait_for_tools(2.0)
        result["waited"] = time.monotonic() - started

    worker = threading.Thread(target=waiter)
    worker.start()

    await _sleep(0.2)
    assert worker.is_alive(), "最后一页之前不应放行"
    for name in PAGE_2:
        await client.stage_tool(_tool(name))
    await client.publish_staged_tools()
    client.mark_tools_complete()

    worker.join(timeout=2.0)
    assert result["ready"] is True
    assert result["waited"] < 1.5
    assert client.has_tool("self_arm_look_at_person") is True


async def test_pending_tool_wait_gives_up_when_pages_stop_arriving():
    """页面停止到达后不再拖时间（设备卡住时工具调用要尽快下结论）。"""
    client = MCPClient()
    await client.stage_tool(_tool(PAGE_1[0]))
    client._last_page_at = time.monotonic() - 10.0  # 10s 没有新页

    started = time.monotonic()
    ready = await client.wait_for_pending_tools(max_wait=8.0, stall=4.0)
    assert ready is False
    assert time.monotonic() - started < 1.0


async def test_pending_tool_wait_returns_when_completed_midway():
    client = MCPClient()
    await client.stage_tool(_tool(PAGE_1[0]))

    async def finish_pages():
        await _sleep(0.3)
        await client.publish_staged_tools()
        client.mark_tools_complete()

    task = asyncio.ensure_future(finish_pages())
    ready = await client.wait_for_pending_tools(max_wait=8.0, stall=4.0)
    await task

    assert ready is True
    assert client.has_tool("self_get_device_status") is True


async def test_publish_merge_keeps_earlier_pages():
    """兜底先发布了部分列表后，真正的最后一页到了要 merge，不能把前面的页冲掉。"""
    client = MCPClient()
    for name in PAGE_1:
        await client.stage_tool(_tool(name))
    await client.publish_staged_tools()
    client.mark_tools_complete()

    for name in PAGE_2:
        await client.stage_tool(_tool(name))
    await client.publish_staged_tools(merge=True)

    assert client.has_tool("self_get_device_status") is True
    assert client.has_tool("self_arm_look_at_person") is True


class _FakeDeviceExecutor(ToolExecutor):
    """把 mcp_client 的工具表当成“设备工具”，用来测派发层的等待逻辑。"""

    def __init__(self, conn):
        self.conn = conn

    def get_tools(self):
        return {
            name: ToolDefinition(
                name=name,
                description={"type": "function", "function": {"name": name}},
                tool_type=ToolType.DEVICE_MCP,
            )
            for name in self.conn.mcp_client.tools
        }

    def has_tool(self, tool_name):
        return self.conn.mcp_client.has_tool(tool_name)

    async def execute(self, conn, tool_name, arguments):
        return ActionResponse(Action.REQLLM, f"executed {tool_name}", None)


async def _make_manager(conn) -> ToolManager:
    """ToolManager 的构造会调用 setup_logging()（内部 asyncio.run），只能在工作线程里建。"""
    loop = asyncio.get_running_loop()

    def build():
        manager = ToolManager(conn)
        manager.register_executor(ToolType.DEVICE_MCP, _FakeDeviceExecutor(conn))
        return manager

    return await loop.run_in_executor(None, build)


async def test_dispatch_waits_for_pages_instead_of_saying_tool_does_not_exist():
    """模型在最后几页到达前调用设备工具时，派发层要等页，不能回“工具不存在”。"""
    client = MCPClient()
    conn = SimpleNamespace(mcp_client=client)
    manager = await _make_manager(conn)

    for name in PAGE_1:  # PAGE_1 不含 look_at_person，它只在后面那一页
        await client.stage_tool(_tool(name))

    async def finish_pages():
        await _sleep(0.3)
        for name in PAGE_2:
            await client.stage_tool(_tool(name))
        await client.publish_staged_tools()
        manager.refresh_tools()
        client.mark_tools_complete()

    task = asyncio.ensure_future(finish_pages())
    result = await manager.execute_tool("self_arm_look_at_person", {})
    await task

    assert result.action == Action.REQLLM
    assert result.result == "executed self_arm_look_at_person"


async def test_dispatch_still_reports_missing_tool_when_pages_stalled():
    """页面停住后不再硬等，仍然老老实实报“工具不存在”。"""
    client = MCPClient()
    conn = SimpleNamespace(mcp_client=client)
    manager = await _make_manager(conn)

    await client.stage_tool(_tool(PAGE_1[0]))
    client._last_page_at = time.monotonic() - 10.0  # 页面已经 10s 没来

    started = time.monotonic()
    result = await manager.execute_tool("self_arm_look_at_person", {})

    assert result.action == Action.NOTFOUND
    assert result.response and "不存在" in result.response
    assert time.monotonic() - started < 2.0
