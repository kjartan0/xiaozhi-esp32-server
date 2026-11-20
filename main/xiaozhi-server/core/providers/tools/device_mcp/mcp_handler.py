"""设备端MCP客户端支持模块"""

import json
import os
import asyncio
import re
from core.utils.util import get_vision_url
from core.utils.auth import AuthToken
from config.logger import setup_logging
from .mcp_client import MCPClient
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

# t_c30a9895: Dutch tool descriptions for the device MCP tools.
# The ESP32 firmware ships its own (Chinese) descriptions over MCP. We deliberately
# do not touch or reflash the firmware: the text going into the LLM prompt is
# overridden server-side from data/tool_desc_nl.json, and tool NAMES are left
# untouched because the device protocol dispatches on them.
_DUTCH_TOOL_DESCRIPTIONS = None


def _load_dutch_tool_descriptions() -> dict:
    """Load data/tool_desc_nl.json once (data/ is bind-mounted: restart picks up edits)."""
    global _DUTCH_TOOL_DESCRIPTIONS
    if _DUTCH_TOOL_DESCRIPTIONS is not None:
        return _DUTCH_TOOL_DESCRIPTIONS

    _DUTCH_TOOL_DESCRIPTIONS = {}
    try:
        from config.config_loader import get_project_dir

        path = os.path.join(get_project_dir(), "data", "tool_desc_nl.json")
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("root must be an object")
        # Drop the "_comment" documentation key if present
        _DUTCH_TOOL_DESCRIPTIONS = {
            k: v for k, v in data.items() if isinstance(v, dict)
        }
        logger.bind(tag=TAG).info(
            f"Nederlandse tool-beschrijvingen geladen: {len(_DUTCH_TOOL_DESCRIPTIONS)}"
        )
    except FileNotFoundError:
        logger.bind(tag=TAG).warning(
            "data/tool_desc_nl.json niet gevonden; firmware-beschrijvingen blijven staan"
        )
    except Exception as e:
        logger.bind(tag=TAG).error(
            f"data/tool_desc_nl.json kon niet worden geladen: {e}"
        )
    return _DUTCH_TOOL_DESCRIPTIONS


def apply_dutch_tool_descriptions(mcp_client: "MCPClient") -> int:
    """Overwrite tool/param descriptions with Dutch; fall back to the original text."""
    overrides = _load_dutch_tool_descriptions()
    if not overrides:
        return 0

    applied = 0
    for tool_name, tool_data in mcp_client.tools.items():
        entry = overrides.get(tool_name)
        if not isinstance(entry, dict):
            continue

        description = entry.get("description")
        if isinstance(description, str) and description.strip():
            tool_data["description"] = description

        params = entry.get("params") or {}
        if isinstance(params, dict):
            properties = (tool_data.get("inputSchema") or {}).get("properties") or {}
            for param_name, param_description in params.items():
                prop = properties.get(param_name)
                if isinstance(prop, dict) and param_description:
                    prop["description"] = param_description
        applied += 1

    # The available-tools list is cached on the client; invalidate it so the next
    # get_available_tools() renders the Dutch text.
    mcp_client._cached_available_tools = None
    logger.bind(tag=TAG).info(
        f"Nederlandse tool-beschrijvingen toegepast op {applied}/{len(mcp_client.tools)} tools"
    )
    return applied

# ---- t_a2a3bf8a: 设备 MCP 工具列表分页（nextCursor）的发布与兜底 ----

# 设备收到 tools/list 后分页推送工具。若协议中途断掉（少一页），列表永远“不完整”，
# 那么设备工具就一个都用不上。因此超时后先把已收到的页发布出去（旧行为），
# 并大声记日志，等真正的最后一页到了再 merge 补齐。
TOOLS_LIST_INCOMPLETE_TIMEOUT_SECONDS = 10.0


def _localize_tool_descriptions(mcp_client: MCPClient) -> None:
    """把工具描述里的原始工具名换成模型看到的（sanitized）名字，再套荷兰语覆盖。"""
    for tool_data in mcp_client.tools.values():
        if "description" in tool_data:
            description = tool_data["description"]
            for sanitized_name, original_name in mcp_client.name_mapping.items():
                description = description.replace(original_name, sanitized_name)
            tool_data["description"] = description

    # t_c30a9895: 荷兰语描述要在名字替换之后套，覆盖文本才会生效。
    apply_dutch_tool_descriptions(mcp_client)


async def _finalise_tool_list(
    conn: "ConnectionHandler", mcp_client: MCPClient, merge: bool = False
) -> dict:
    """发布暂存的工具列表并刷新函数表；最后一步才置 ready 并唤醒首轮等待。"""
    published = await mcp_client.publish_staged_tools(merge=merge)
    _localize_tool_descriptions(mcp_client)
    await mcp_client.set_ready(True)

    if hasattr(conn, "func_handler") and conn.func_handler:
        conn.func_handler.tool_manager.refresh_tools()
        conn.func_handler.current_support_functions()

    # 顺序很重要：先刷新函数表，再唤醒首轮请求，否则被唤醒的那一轮可能读到旧缓存。
    mcp_client.mark_tools_complete()
    logger.bind(tag=TAG).info(
        f"设备MCP工具列表已发布(完整): tools={len(published)} merge={merge}"
    )
    return published


async def _tools_list_watchdog(
    conn: "ConnectionHandler",
    mcp_client: MCPClient,
    timeout: float = TOOLS_LIST_INCOMPLETE_TIMEOUT_SECONDS,
):
    """兜底：分页迟迟收不全时先发布已收到的部分，避免设备工具整场缺席。"""
    try:
        await asyncio.sleep(timeout)
        if getattr(conn, "mcp_client", None) is not mcp_client:
            return
        if mcp_client.tools_complete:
            return
        published = await _finalise_tool_list(conn, mcp_client, merge=True)
        logger.bind(tag=TAG).warning(
            f"设备MCP工具列表 {timeout:.0f}s 内未收全，先发布已收到的 {len(published)} 个"
            "（部分列表；后续页到达后会合并补齐）"
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.bind(tag=TAG).error(f"设备MCP工具列表兜底发布失败: {e}")


def _ensure_tools_list_watchdog(conn: "ConnectionHandler", mcp_client: MCPClient) -> None:
    """每个连接只起一个兜底看护任务。"""
    existing = getattr(conn, "_mcp_tools_watchdog", None)
    if existing is not None and not existing.done():
        return
    try:
        conn._mcp_tools_watchdog = asyncio.create_task(
            _tools_list_watchdog(conn, mcp_client)
        )
    except RuntimeError as e:  # 没有运行中的事件循环：不兜底也不能炸主流程
        logger.bind(tag=TAG).warning(f"无法启动设备MCP工具列表兜底任务: {e}")


async def send_mcp_message(conn: "ConnectionHandler", payload: dict):
    """Helper to send MCP messages, encapsulating common logic."""
    if not conn.features.get("mcp"):
        logger.bind(tag=TAG).warning("客户端不支持MCP，无法发送MCP消息")
        return

    message = json.dumps({"type": "mcp", "payload": payload})

    try:
        await conn.websocket.send(message)
        logger.bind(tag=TAG).debug(f"成功发送MCP消息: {message}")
    except Exception as e:
        logger.bind(tag=TAG).error(f"发送MCP消息失败: {e}")


async def handle_mcp_message(
    conn: "ConnectionHandler", mcp_client: MCPClient, payload: dict
):
    """处理MCP消息,包括初始化、工具列表和工具调用响应等"""
    logger.bind(tag=TAG).debug(f"处理MCP消息: {str(payload)[:100]}")

    if not isinstance(payload, dict):
        logger.bind(tag=TAG).error("MCP消息缺少payload字段或格式错误")
        return

    # Handle result
    if "result" in payload:
        result = payload["result"]
        msg_id = int(payload.get("id", 0))

        # Check for tool call response first
        if msg_id in mcp_client.call_results:
            logger.bind(tag=TAG).debug(
                f"收到工具调用响应，ID: {msg_id}, 结果: {result}"
            )
            await mcp_client.resolve_call_result(msg_id, result)
            return

        if msg_id == 1:  # mcpInitializeID
            logger.bind(tag=TAG).debug("收到MCP初始化响应")
            server_info = result.get("serverInfo")
            if isinstance(server_info, dict):
                name = server_info.get("name")
                version = server_info.get("version")
                logger.bind(tag=TAG).debug(
                    f"客户端MCP服务器信息: name={name}, version={version}"
                )
            return

        elif msg_id == 2:  # mcpToolsListID
            logger.bind(tag=TAG).debug("收到MCP工具列表响应")
            if isinstance(result, dict) and "tools" in result:
                tools_data = result["tools"]
                if not isinstance(tools_data, list):
                    logger.bind(tag=TAG).error("工具列表格式错误")
                    return

                logger.bind(tag=TAG).info(
                    f"客户端设备支持的工具数量: {len(tools_data)}"
                )

                # t_a2a3bf8a: 每页只暂存，最后一页才统一发布。之前的实现边收边
                # publish，首轮在分页中途就看到了半成品列表（39 个里只有 8 个），
                # 于是模型调用一个服务器还不认识的工具、被回“工具不存在”。
                for i, tool in enumerate(tools_data):
                    if not isinstance(tool, dict):
                        continue

                    name = tool.get("name", "")
                    description = tool.get("description", "")
                    input_schema = {"type": "object", "properties": {}, "required": []}

                    if "inputSchema" in tool and isinstance(tool["inputSchema"], dict):
                        schema = tool["inputSchema"]
                        input_schema["type"] = schema.get("type", "object")
                        input_schema["properties"] = schema.get("properties", {})
                        input_schema["required"] = [
                            s for s in schema.get("required", []) if isinstance(s, str)
                        ]

                    new_tool = {
                        "name": name,
                        "description": description,
                        "inputSchema": input_schema,
                    }
                    await mcp_client.stage_tool(new_tool)
                    logger.bind(tag=TAG).debug(f"客户端工具 #{i+1}: {name}")

                next_cursor = result.get("nextCursor", "") or ""
                logger.bind(tag=TAG).info(
                    f"设备MCP工具分页: 本页={len(tools_data)} 累计暂存="
                    f"{mcp_client.staged_tool_count()} nextCursor={next_cursor or '<none>'}"
                )

                if next_cursor:
                    _ensure_tools_list_watchdog(conn, mcp_client)
                    await send_mcp_tools_list_continue_request(conn, next_cursor)
                else:
                    # 最后一页：一次性发布完整列表。已因超时兜底发布过部分列表时
                    # 走 merge，避免把前面几页丢掉。
                    await _finalise_tool_list(
                        conn, mcp_client, merge=mcp_client.tools_complete
                    )
            return

    # Handle method calls (requests from the client)
    elif "method" in payload:
        method = payload["method"]
        logger.bind(tag=TAG).info(f"收到MCP客户端请求: {method}")

    elif "error" in payload:
        error_data = payload["error"]
        error_msg = error_data.get("message", "未知错误")
        logger.bind(tag=TAG).error(f"收到MCP错误响应: {error_msg}")

        msg_id = int(payload.get("id", 0))
        if msg_id in mcp_client.call_results:
            await mcp_client.reject_call_result(
                msg_id, Exception(f"MCP错误: {error_msg}")
            )


async def send_mcp_initialize_message(conn: "ConnectionHandler"):
    """发送MCP初始化消息"""

    vision_url = get_vision_url(conn.config)

    # 密钥生成token
    auth = AuthToken(conn.config["server"]["auth_key"])
    token = auth.generate_token(conn.headers.get("device-id"))

    vision = {
        "url": vision_url,
        "token": token,
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 1,  # mcpInitializeID
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {},
                "vision": vision,
            },
            "clientInfo": {
                "name": "XiaozhiClient",
                "version": "1.0.0",
            },
        },
    }
    logger.bind(tag=TAG).debug("发送MCP初始化消息")
    await send_mcp_message(conn, payload)


async def send_mcp_tools_list_request(conn: "ConnectionHandler"):
    """发送MCP工具列表请求"""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,  # mcpToolsListID
        "method": "tools/list",
    }
    logger.bind(tag=TAG).debug("发送MCP工具列表请求")
    await send_mcp_message(conn, payload)


async def send_mcp_tools_list_continue_request(conn: "ConnectionHandler", cursor: str):
    """发送带有cursor的MCP工具列表请求"""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,  # mcpToolsListID (same ID for continuation)
        "method": "tools/list",
        "params": {"cursor": cursor},
    }
    logger.bind(tag=TAG).info(f"发送带cursor的MCP工具列表请求: {cursor}")
    await send_mcp_message(conn, payload)


async def call_mcp_tool(
    conn: "ConnectionHandler",
    mcp_client: MCPClient,
    tool_name: str,
    args: str = "{}",
    timeout: int = 30,
):
    """
    调用指定的工具，并等待响应
    """
    if not await mcp_client.is_ready():
        raise RuntimeError("MCP客户端尚未准备就绪")

    if not mcp_client.has_tool(tool_name):
        # t_a2a3bf8a: 分页还没收全时不要断言“不存在”（等页的重试在
        # ToolManager.execute_tool 里，那里才持有完整工具表）。
        if not mcp_client.tools_complete:
            raise ValueError(f"工具 {tool_name} 不存在（设备MCP工具列表尚未接收完整）")
        raise ValueError(f"工具 {tool_name} 不存在")

    tool_call_id = await mcp_client.get_next_id()
    result_future = asyncio.Future()
    await mcp_client.register_call_result_future(tool_call_id, result_future)

    # 处理参数
    try:
        if isinstance(args, str):
            # 确保字符串是有效的JSON
            if not args.strip():
                arguments = {}
            else:
                try:
                    # 尝试直接解析
                    arguments = json.loads(args)
                except json.JSONDecodeError:
                    # 如果解析失败，尝试合并多个JSON对象
                    try:
                        # 使用正则表达式匹配所有JSON对象
                        json_objects = re.findall(r"\{[^{}]*\}", args)
                        if len(json_objects) > 1:
                            # 合并所有JSON对象
                            merged_dict = {}
                            for json_str in json_objects:
                                try:
                                    obj = json.loads(json_str)
                                    if isinstance(obj, dict):
                                        merged_dict.update(obj)
                                except json.JSONDecodeError:
                                    continue
                            if merged_dict:
                                arguments = merged_dict
                            else:
                                raise ValueError(f"无法解析任何有效的JSON对象: {args}")
                        else:
                            raise ValueError(f"参数JSON解析失败: {args}")
                    except Exception as e:
                        logger.bind(tag=TAG).error(
                            f"参数JSON解析失败: {str(e)}, 原始参数: {args}"
                        )
                        raise ValueError(f"参数JSON解析失败: {str(e)}")
        elif isinstance(args, dict):
            arguments = args
        else:
            raise ValueError(f"参数类型错误，期望字符串或字典，实际类型: {type(args)}")

        # 确保参数是字典类型
        if not isinstance(arguments, dict):
            raise ValueError(f"参数必须是字典类型，实际类型: {type(arguments)}")

    except Exception as e:
        if not isinstance(e, ValueError):
            raise ValueError(f"参数处理失败: {str(e)}")
        raise e

    actual_name = mcp_client.name_mapping.get(tool_name, tool_name)
    payload = {
        "jsonrpc": "2.0",
        "id": tool_call_id,
        "method": "tools/call",
        "params": {"name": actual_name, "arguments": arguments},
    }

    logger.bind(tag=TAG).info(f"发送客户端mcp工具调用请求: {actual_name}，参数: {args}")
    await send_mcp_message(conn, payload)

    try:
        # Wait for response or timeout
        raw_result = await asyncio.wait_for(result_future, timeout=timeout)
        logger.bind(tag=TAG).info(
            f"客户端mcp工具调用 {actual_name} 成功，原始结果: {raw_result}"
        )

        if isinstance(raw_result, dict):
            if raw_result.get("isError") is True:
                error_msg = raw_result.get(
                    "error", "工具调用返回错误，但未提供具体错误信息"
                )
                raise RuntimeError(f"工具调用错误: {error_msg}")

            content = raw_result.get("content")
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and "text" in content[0]:
                    # 直接返回文本内容，不进行JSON解析
                    return content[0]["text"]
        # 如果结果不是预期的格式，将其转换为字符串
        return str(raw_result)
    except asyncio.TimeoutError:
        await mcp_client.cleanup_call_result(tool_call_id)
        raise TimeoutError("工具调用请求超时")
    except Exception as e:
        await mcp_client.cleanup_call_result(tool_call_id)
        raise e
