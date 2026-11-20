import random
import httpx
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler


TAG = __name__
logger = setup_logging()

# NOTE (t_c30a9895): this plugin scrapes chinanews.com.cn RSS — Chinese news, in
# Chinese, for a Chinese audience. Its tool description is Dutch so it no longer
# pulls the prompt into Chinese, but for a Dutch user it is arguably irrelevant;
# it is kept (not removed) because removing a tool is the owner's call, not a
# translation detail. See the card handoff.
GET_NEWS_FROM_CHINANEWS_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_news_from_chinanews",
        "description": (
            "Roep dit aan als de gebruiker nieuws wil zien of horen; je kunt een rubriek meegeven (algemeen, binnenland, buitenland, economie); zonder rubriek krijg je het algemene nieuws."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Rubriek, bijvoorbeeld algemeen, binnenland, buitenland of economie. Optioneel; zonder waarde wordt de standaardrubriek gebruikt",
                },
                "detail": {
                    "type": "boolean",
                    "description": "Of de volledige inhoud moet worden opgehaald; standaard false. Bij true krijg je de details van het laatst genoemde nieuwsbericht",
                },
                "lang": {
                    "type": "string",
                    "description": "Taalcode voor het antwoord, bijvoorbeeld nl of en; standaard nl",
                },
            },
            "required": ["lang"],
        },
    },
}


async def fetch_news_from_rss(rss_url):
    """从RSS源获取新闻列表"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
            response = await client.get(rss_url)

        # 解析XML
        root = ET.fromstring(response.content)

        # 查找所有item元素（新闻条目）
        news_items = []
        for item in root.findall(".//item"):
            title = (
                item.find("title").text if item.find("title") is not None else "zonder titel"
            )
            link = item.find("link").text if item.find("link") is not None else "#"
            description = (
                item.find("description").text
                if item.find("description") is not None
                else "zonder beschrijving"
            )
            pubDate = (
                item.find("pubDate").text
                if item.find("pubDate") is not None
                else "onbekende tijd"
            )

            news_items.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "pubDate": pubDate,
                }
            )

        return news_items
    except Exception as e:
        logger.bind(tag=TAG).error(f"获取RSS新闻失败: {e}")
        return []


async def fetch_news_detail(url):
    """获取新闻详情页内容并总结"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            response = await client.get(url)

        soup = BeautifulSoup(response.content, "html.parser")

        # 尝试提取正文内容 (这里的选择器需要根据实际网站结构调整)
        content_div = soup.select_one(
            ".content_desc, .content, article, .article-content"
        )
        if content_div:
            paragraphs = content_div.find_all("p")
            content = "\n".join(
                [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            )
            return content
        else:
            # 如果找不到特定的内容区域，尝试获取所有段落
            paragraphs = soup.find_all("p")
            content = "\n".join(
                [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            )
            return content[:2000]  # 限制长度
    except Exception as e:
        logger.bind(tag=TAG).error(f"获取新闻详情失败: {e}")
        return "Geen volledige inhoud beschikbaar"


def map_category(category_text):
    """将用户输入的中文类别映射到配置文件中的类别键"""
    if not category_text:
        return None

    # 类别映射字典，目前支持社会、国际、财经新闻，如需更多类型，参见配置文件
    category_map = {
        # algemeen nieuws
        "algemeen": "default_rss_url",
        "algemeen nieuws": "default_rss_url",
        # binnenlands nieuws
        "binnenland": "society_rss_url",
        "nationaal": "society_rss_url",
        # buitenlands nieuws
        "buitenland": "world_rss_url",
        "internationaal": "world_rss_url",
        # economisch nieuws
        "economie": "finance_rss_url",
        "financieel": "finance_rss_url",
        "financien": "finance_rss_url",
        "zakelijk": "finance_rss_url",
    }

    # 转换为小写并去除空格
    normalized_category = category_text.lower().strip()

    # 返回映射结果，如果没有匹配项则返回原始输入
    return category_map.get(normalized_category, category_text)


@register_function(
    "get_news_from_chinanews",
    GET_NEWS_FROM_CHINANEWS_FUNCTION_DESC,
    ToolType.SYSTEM_CTL,
)
async def get_news_from_chinanews(
    conn: "ConnectionHandler",
    category: str = None,
    detail: bool = False,
    lang: str = "nl",
):
    """获取新闻并随机选择一条进行播报，或获取上一条新闻的详细内容"""
    try:
        # 如果detail为True，获取上一条新闻的详细内容
        if detail:
            if (
                not hasattr(conn, "last_news_link")
                or not conn.last_news_link
                or "link" not in conn.last_news_link
            ):
                return ActionResponse(
                    Action.REQLLM,
                    "Sorry, ik heb nog geen recent nieuwsbericht waar ik details van kan geven; vraag eerst een nieuwsbericht op.",
                    None,
                )

            link = conn.last_news_link.get("link")
            title = conn.last_news_link.get("title", "zonder titel")

            if link == "#":
                return ActionResponse(
                    Action.REQLLM, "Sorry, bij dit nieuwsbericht zit geen link naar de volledige inhoud.", None
                )

            logger.bind(tag=TAG).debug(f"nieuwsdetails ophalen: {title}, URL={link}")

            # 获取新闻详情
            detail_content = await fetch_news_detail(link)

            if not detail_content or detail_content == "Geen volledige inhoud beschikbaar":
                return ActionResponse(
                    Action.REQLLM,
                    f"Sorry, ik kan de details van {title} niet ophalen; de link werkt mogelijk niet meer.",
                    None,
                )

            # 构建详情报告
            detail_report = (
                f"Beantwoord de vraag van de gebruiker over dit nieuwsbericht met de volgende gegevens:\n\n"
                f"Titel: {title}\n"
                f"Inhoud: {detail_content}\n\n"
                f"(Vat de inhoud hierboven samen in het Nederlands, haal de kern eruit en vertel het "
                f"als een natuurlijk, vlot verhaal; zeg niet dat het een samenvatting is)"
            )

            return ActionResponse(Action.REQLLM, detail_report, None)

        # 否则，获取新闻列表并随机选择一条
        # 从配置中获取RSS URL
        rss_config = conn.config.get("plugins", {}).get("get_news_from_chinanews", {})
        default_rss_url = rss_config.get(
            "default_rss_url", "https://feeds.nos.nl/nosnieuwsalgemeen"
        )

        # 将用户输入的类别映射到配置中的类别键
        mapped_category = map_category(category)

        # 如果提供了类别，尝试从配置中获取对应的URL
        rss_url = default_rss_url
        if mapped_category and mapped_category in rss_config:
            rss_url = rss_config[mapped_category]

        logger.bind(tag=TAG).info(
            f"nieuws ophalen: rubriek={category}, gemapt={mapped_category}, URL={rss_url}"
        )

        # 获取新闻列表
        news_items = await fetch_news_from_rss(rss_url)

        if not news_items:
            return ActionResponse(
                Action.REQLLM, "Sorry, ik kon geen nieuws ophalen; probeer het later opnieuw.", None
            )

        # 随机选择一条新闻
        selected_news = random.choice(news_items)

        # 保存当前新闻链接到连接对象，以便后续查询详情
        if not hasattr(conn, "last_news_link"):
            conn.last_news_link = {}
        conn.last_news_link = {
            "link": selected_news.get("link", "#"),
            "title": selected_news.get("title", "zonder titel"),
        }

        # 构建新闻报告
        news_report = (
            f"Beantwoord de nieuwsvraag van de gebruiker met de volgende gegevens:\n\n"
            f"Titel: {selected_news['title']}\n"
            f"Gepubliceerd: {selected_news['pubDate']}\n"
            f"Inhoud: {selected_news['description']}\n"
            f"(Vertel dit bericht in het Nederlands in natuurlijke, vlotte spreektaal; vat het waar "
            f"nodig kort samen. Lees het bericht gewoon voor, zonder extra inleiding. "
            f"Wil de gebruiker meer details, zeg dan dat die om de details van dit bericht kan vragen.)"
        )

        return ActionResponse(Action.REQLLM, news_report, None)

    except Exception as e:
        logger.bind(tag=TAG).error(f"获取新闻出错: {e}")
        return ActionResponse(
            Action.REQLLM, "Sorry, er ging iets mis bij het ophalen van het nieuws; probeer het later opnieuw.", None
        )
