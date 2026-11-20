"""
系统提示词管理器模块
负责管理和更新系统提示词，包括快速初始化和异步增强功能
"""

import os
import asyncio
import hashlib
import threading
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from config.logger import setup_logging
from jinja2 import Template

TAG = __name__

WEEKDAY_MAP = {
    "Monday": "maandag",
    "Tuesday": "dinsdag",
    "Wednesday": "woensdag",
    "Thursday": "donderdag",
    "Friday": "vrijdag",
    "Saturday": "zaterdag",
    "Sunday": "zondag",
}

EMOJI_List = [
    "😶",
    "🙂",
    "😆",
    "😂",
    "😔",
    "😠",
    "😭",
    "😍",
    "😳",
    "😲",
    "😱",
    "🤔",
    "😉",
    "😎",
    "😌",
    "🤤",
    "😘",
    "😏",
    "😴",
    "😜",
    "🙄",
]


class PromptManager:
    """系统提示词管理器，负责管理和更新系统提示词"""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or setup_logging()
        self.base_prompt_template = None
        self.last_update_time = 0

        # 导入全局缓存管理器
        from core.utils.cache.manager import cache_manager, CacheType

        self.cache_manager = cache_manager
        self.CacheType = CacheType

        # 初始化上下文源
        from core.utils.context_provider import ContextDataProvider

        self.context_provider = ContextDataProvider(config, self.logger)
        self.context_data = {}

        self._load_base_template()

    def _load_base_template(self):
        """加载基础提示词模板"""
        try:
            template_path = self.config.get("prompt_template", None)
            if not template_path:
                template_path = "agent-base-prompt.txt"
            cache_key = f"prompt_template:{template_path}"

            # 先从缓存获取
            cached_template = self.cache_manager.get(self.CacheType.CONFIG, cache_key)
            if cached_template is not None:
                self.base_prompt_template = cached_template
                self.logger.bind(tag=TAG).debug("从缓存加载基础提示词模板")
                return

            # 缓存未命中，从文件读取
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    template_content = f.read()

                # 存入缓存（CONFIG类型默认不自动过期，需要手动失效）
                self.cache_manager.set(
                    self.CacheType.CONFIG, cache_key, template_content
                )
                self.base_prompt_template = template_content
                self.logger.bind(tag=TAG).debug("成功加载基础提示词模板并缓存")
            else:
                self.logger.bind(tag=TAG).warning(f"未找到{template_path}文件")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"加载提示词模板失败: {e}")

    def get_quick_prompt(self, user_prompt: str, device_id: str = None) -> str:
        """快速获取系统提示词（使用用户配置）"""
        device_cache_key = f"device_prompt:{device_id}"
        cached_device_prompt = self.cache_manager.get(
            self.CacheType.DEVICE_PROMPT, device_cache_key
        )
        if cached_device_prompt is not None:
            self.logger.bind(tag=TAG).debug(f"使用设备 {device_id} 的缓存提示词")
            return cached_device_prompt
        else:
            self.logger.bind(tag=TAG).debug(
                f"设备 {device_id} 无缓存提示词，使用传入的提示词"
            )

        # 使用传入的提示词并缓存（如果有设备ID）
        if device_id:
            self.cache_manager.set(
                self.CacheType.DEVICE_PROMPT, device_cache_key, user_prompt
            )
            self.logger.bind(tag=TAG).debug(f"设备 {device_id} 的提示词已缓存")

        self.logger.bind(tag=TAG).info(f"使用快速提示词: {user_prompt[:50]}...")
        return user_prompt

    def _get_current_time_info(self) -> tuple:
        """获取当前时间信息"""
        from .current_time import (
            get_current_date,
            get_current_weekday,
            get_current_lunar_date,
        )

        today_date = get_current_date()
        today_weekday = get_current_weekday()
        # t_c30a9895: the lunar date line was removed from agent-base-prompt.txt;
        # the value is kept as an empty string so the render() contract is stable.
        lunar_date = ""

        return today_date, today_weekday, lunar_date

    def _get_location_info(self, client_ip: str) -> str:
        """Plaats voor de prompt; nooit leeg (t_9fe9586d).

        Vroeger stond hier alleen de IP-city van een China-only geocoder. Voor
        een Nederlands IP is dat een lege string, waardoor de prompt een lege
        ``location`` kreeg en het weer helemaal overgeslagen werd. Nu is de
        geconfigureerde thuisbasis de terugval en mag een IP-city alleen
        overschrijven als hij bruikbaar is.
        """
        from core.utils.location import resolve_local_address

        cache_key = client_ip or "no-client-ip"
        try:
            cached_location = self.cache_manager.get(self.CacheType.LOCATION, cache_key)
            # Een lege waarde uit een eerdere deploy nooit als cache-hit zien.
            if cached_location:
                return cached_location

            location = resolve_local_address(self.config, client_ip, self.logger)
            self.cache_manager.set(self.CacheType.LOCATION, cache_key, location)
            return location
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"locatie bepalen mislukt: {e}")
            return resolve_local_address(self.config, None, self.logger)

    def _get_weather_info(self, conn: "ConnectionHandler", location: str) -> str:
        """Weerbericht voor de promptcontext, Nederlands, via Open-Meteo."""
        from core.utils import weather as weather_utils

        cache_key = weather_utils.cache_key(location, "nl")
        try:
            cached_weather = self.cache_manager.get(self.CacheType.WEATHER, cache_key)
            if cached_weather:
                return cached_weather

            # Lokaal async rapport bouwen. Windows ProactorEventLoop ondersteunt
            # run_coroutine_threadsafe().result() niet; daarom call_soon_threadsafe
            # + threading.Event (blokkeert alleen deze thread, niet de event loop).
            result_holder = []
            exception_holder = []
            event = threading.Event()

            async def _call():
                try:
                    result_holder.append(
                        await weather_utils.build_report_for(
                            self.config, location, lang="nl", include_city=False
                        )
                    )
                except Exception as e:
                    exception_holder.append(e)
                finally:
                    event.set()

            conn.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_call()))
            if not event.wait(timeout=5):
                raise TimeoutError("weerbericht ophalen duurde te lang")
            if exception_holder:
                raise exception_holder[0]
            weather_report = result_holder[0] if result_holder else None
            if not weather_report:
                raise RuntimeError(f"geen weerbericht voor {location!r}")

            self.cache_manager.set(self.CacheType.WEATHER, cache_key, weather_report)
            return weather_report

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"weerbericht ophalen mislukt: {e}")
            return ""

    def update_context_info(self, conn, client_ip: str):
        """同步更新上下文信息"""
        try:
            local_address = ""
            wants_context = bool(
                self.base_prompt_template
                and (
                    "local_address" in self.base_prompt_template
                    or "weather_info" in self.base_prompt_template
                )
            )
            if wants_context:
                # Zonder client_ip geldt de thuisbasis: de locatie is nooit leeg,
                # dus het weer wordt niet meer overgeslagen (t_9fe9586d).
                local_address = self._get_location_info(client_ip)

            if (
                self.base_prompt_template
                and "weather_info" in self.base_prompt_template
                and local_address
            ):
                # 获取天气信息（使用全局缓存）
                self._get_weather_info(conn, local_address)

            # 获取配置的上下文数据
            if hasattr(conn, "device_id") and conn.device_id:
                if (
                    self.base_prompt_template
                    and "dynamic_context" in self.base_prompt_template
                ):
                    self.context_data = self.context_provider.fetch_all(conn.device_id)
                else:
                    self.context_data = ""

            self.logger.bind(tag=TAG).debug(f"上下文信息更新完成")

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"更新上下文信息失败: {e}")

    def build_enhanced_prompt(
        self, user_prompt: str, device_id: str, client_ip: str = None, *args, **kwargs
    ) -> str:
        """构建增强的系统提示词"""
        if not self.base_prompt_template:
            return user_prompt

        # 缓存键须覆盖所有渲染输入：emoji 开关、设备 prompt 变更否则会命中过期提示词
        emoji_enabled = kwargs.get("emoji_enabled", True)
        prompt_hash = hashlib.md5(user_prompt.encode("utf-8")).hexdigest()[:8]
        enhanced_cache_key = (
            f"enhanced_prompt:{device_id}:{emoji_enabled}:{prompt_hash}"
        )
        cached_prompt = self.cache_manager.get(
            self.CacheType.DEVICE_PROMPT, enhanced_cache_key
        )
        if cached_prompt is not None:
            return cached_prompt

        try:
            # 获取最新的时间信息（不缓存）
            today_date, today_weekday, lunar_date = self._get_current_time_info()

            # 获取缓存的上下文信息
            local_address = ""
            weather_info = ""
            from core.utils import weather as weather_utils

            location_cache_key = client_ip or "no-client-ip"
            # 获取位置信息（从全局缓存）
            local_address = (
                self.cache_manager.get(self.CacheType.LOCATION, location_cache_key) or ""
            )
            if not local_address:
                # Cache leeg of een oude lege waarde: altijd een echte plaats.
                local_address = self._get_location_info(client_ip)

            # 获取天气信息（从全局缓存）
            if local_address:
                weather_info = (
                    self.cache_manager.get(
                        self.CacheType.WEATHER,
                        weather_utils.cache_key(local_address, "nl"),
                    )
                    or ""
                )
            if not weather_info:
                # Nooit een leeg "local upcoming weather" in de prompt laten staan.
                weather_info = "niet beschikbaar"

            # 获取TTS选择的语言，默认值为中文
            # t_c30a9895: the fallback is Dutch, not 中文 — an unset TTS language
            # used to inject the string "中文" into the prompt's "Answer in
            # {{language}}" rule.
            language = (
                self.config.get("TTS", {})
                .get(self.config.get("selected_module", {}).get("TTS", ""), {})
                .get("language")
                or "Nederlands"
            )
            self.logger.bind(tag=TAG).debug(f"获取到选择的语言: {language}")

            # 替换模板变量
            template = Template(self.base_prompt_template)
            enhanced_prompt = template.render(
                base_prompt=user_prompt,
                current_time="{{current_time}}",
                today_date=today_date,
                today_weekday=today_weekday,
                lunar_date=lunar_date,
                local_address=local_address,
                weather_info=weather_info,
                emojiList=EMOJI_List,
                device_id=device_id,
                client_ip=client_ip,
                dynamic_context=self.context_data,
                language=language,
                *args,
                **kwargs,
            )
            self.cache_manager.set(
                self.CacheType.DEVICE_PROMPT,
                enhanced_cache_key,
                enhanced_prompt,
                ttl=300,
            )
            self.logger.bind(tag=TAG).debug(
                f"构建增强提示词成功，长度: {len(enhanced_prompt)}"
            )
            return enhanced_prompt

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"构建增强提示词失败: {e}")
            return user_prompt
