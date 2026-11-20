import httpx
from config.logger import setup_logging
from plugins_func.register import (
    register_function,
    ToolType,
    ActionResponse,
    Action,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

_DEFAULT_DESCRIPTION = (
    "Online zoekhulpmiddel. Gebruik dit als de gebruiker iets wilt opzoeken of iets "
    "actueels wil weten waarvoor je online moet zoeken."
)

WEB_SEARCH_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": _DEFAULT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Zoekterm of vraag",
                }
            },
            "required": ["query"],
        },
    },
}


async def _search_metaso(api_key: str, query: str, max_results: int) -> str:
    """调用秘塔搜索API"""
    url = "https://metaso.cn/api/v1/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "size": max_results,
        "stream": False,
        "scope": "webpage",
        "includeSummary": True,
        "includeRawContent": False,
        "conciseSnippet": False,
    }
    logger.bind(tag=TAG).debug(f"秘塔搜索请求 | URL: {url} | payload: {payload}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=3.0)) as client:
        response = await client.post(url, json=payload, headers=headers)
    data = response.json()
    logger.bind(tag=TAG).debug(f"秘塔搜索响应 | status: {response.status_code}")

    webpages = data.get("webpages", [])
    if not webpages:
        return "Geen relevante zoekresultaten gevonden."

    lines = ["Online zoekresultaten:"]
    for i, item in enumerate(webpages, 1):
        title = item.get("title", "zonder titel")
        snippet = item.get("summary", "")
        date = item.get("date", "")
        lines.append(f"{i}. Titel: {title}")
        if date:
            lines.append(f"   Datum: {date}")
        if snippet:
            lines.append(f"   Samenvatting: {snippet}")

    return "\n".join(lines)


async def _search_tavily(api_key: str, query: str, max_results: int) -> str:
    """调用Tavily搜索API"""
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": "advanced",
    }
    logger.bind(tag=TAG).debug(f"Tavily搜索请求 | URL: {url} | payload: {payload}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=3.0)) as client:
        response = await client.post(url, json=payload, headers=headers)
    data = response.json()
    logger.bind(tag=TAG).debug(f"Tavily搜索响应 | status: {response.status_code} | data: {data}")

    results = data.get("results", [])
    if not results:
        return "Geen relevante zoekresultaten gevonden."

    answer = data.get("answer", "")
    lines = [f"Online zoekresultaten:\nSamenvatting: {answer}"]
    # for i, item in enumerate(results, 1):
    #     title = item.get("title", "zonder titel")
    #     summary = item.get("content", "")
    #     lines.append(f"{i}. Titel: {title}")
    #     if summary:
    #         lines.append(f"   Samenvatting: {summary}")

    return "\n".join(lines)


async def _search_serply(api_key: str, query: str, max_results: int) -> str:
    """调用Serply搜索API"""
    url = "https://api.serply.io/v1/search"
    headers = {
        "X-Api-Key": api_key,
        "User-Agent": "xiaozhi-esp32-server",
    }
    # Serply单页最多返回10条，且返回条数可能略多于num，需要本地再截断一次
    params = {"q": query, "num": min(max_results, 10)}
    logger.bind(tag=TAG).debug(f"Serply搜索请求 | URL: {url} | params: {params}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=3.0)) as client:
        response = await client.get(url, params=params, headers=headers)
    data = response.json()
    logger.bind(tag=TAG).debug(f"Serply搜索响应 | status: {response.status_code}")

    if response.status_code != 200:
        detail = data.get("detail", "") if isinstance(data, dict) else ""
        logger.bind(tag=TAG).error(
            f"Serply搜索失败 | status: {response.status_code} | detail: {detail}"
        )
        return "Zoekverzoek mislukt, controleer of de API-sleutel klopt."

    results = data.get("results", [])
    if not results:
        return "Geen relevante zoekresultaten gevonden."

    lines = ["Online zoekresultaten:"]
    for i, item in enumerate(results[:max_results], 1):
        title = item.get("title", "zonder titel")
        snippet = item.get("description", "")
        lines.append(f"{i}. Titel: {title}")
        if snippet:
            lines.append(f"   Samenvatting: {snippet}")

    return "\n".join(lines)


@register_function("web_search", WEB_SEARCH_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def web_search(conn: "ConnectionHandler", query: str = None):
    logger.bind(tag=TAG).info(f"web_search 被调用 | query={query}")
    if not query:
        return ActionResponse(Action.REQLLM, "Geef een zoekterm of vraag op.", None)

    web_search_config = conn.config.get("plugins", {}).get("web_search", {})
    provider = web_search_config.get("provider", "").lower()
    max_results = int(web_search_config.get("max_results", 3))
    logger.bind(tag=TAG).info(f"web_search 配置 | provider={provider} | max_results={max_results} | config_keys={list(web_search_config.keys())}")

    api_key = web_search_config.get("api_key", "")
    if not api_key:
        return ActionResponse(
            Action.REQLLM,
            "Zoeken is niet geconfigureerd: er ontbreekt een API-sleutel in de configuratie.",
            None,
        )

    try:
        if provider == "metaso":
            result_text = await _search_metaso(api_key, query, max_results)
        elif provider == "tavily":
            result_text = await _search_tavily(api_key, query, max_results)
        elif provider == "serply":
            result_text = await _search_serply(api_key, query, max_results)
        else:
            return ActionResponse(
                Action.REQLLM,
                f"Zoeken is niet geconfigureerd of de zoekbron is ongeldig (nu: {provider}); controleer de configuratie.",
                None,
            )
        logger.bind(tag=TAG).info(f"搜索结果组装完成:\n{result_text}")
    except httpx.TimeoutException:
        logger.bind(tag=TAG).error("联网搜索请求超时")
        result_text = "Het zoekverzoek is verlopen, probeer het later opnieuw."
    except httpx.HTTPStatusError as e:
        logger.bind(tag=TAG).error(f"联网搜索请求失败: {e}")
        result_text = "Het zoekverzoek mislukte, probeer het later opnieuw."
    except Exception as e:
        logger.bind(tag=TAG).error(f"联网搜索异常: {e}")
        result_text = "Er ging iets mis bij het zoeken, probeer het later opnieuw."

    return ActionResponse(Action.REQLLM, result_text, None)
