"""
时间工具模块
提供统一的时间获取功能
"""

import cnlunar
from datetime import datetime

WEEKDAY_MAP = {
    "Monday": "maandag",
    "Tuesday": "dinsdag",
    "Wednesday": "woensdag",
    "Thursday": "donderdag",
    "Friday": "vrijdag",
    "Saturday": "zaterdag",
    "Sunday": "zondag",
}


def get_current_time() -> str:
    """
    获取当前时间字符串 (格式: HH:MM)
    """
    return datetime.now().strftime("%H:%M")


def get_current_date() -> str:
    """
    获取今天日期字符串 (格式: YYYY-MM-DD)
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_current_weekday() -> str:
    """
    获取今天星期几
    """
    now = datetime.now()
    return WEEKDAY_MAP[now.strftime("%A")]


def get_current_lunar_date() -> str:
    """获取农历日期字符串

    NOTE (t_c30a9895): returns "" deliberately. cnlunar only emits Chinese
    calendar names (二零二六年十月初八), and this line was injected into the LLM
    system prompt — Chinese text in the prompt pulled the spoken answer towards
    Chinese. The lunar line was dropped from agent-base-prompt.txt instead of
    being translated, because a Chinese-lunar date is not something the Dutch
    prompt context needs.
    """
    return ""


def get_current_time_info() -> tuple:
    """
    获取当前时间信息
    返回: (当前时间字符串, 今天日期, 今天星期, 农历日期)
    """
    current_time = get_current_time()
    today_date = get_current_date()
    today_weekday = get_current_weekday()
    lunar_date = get_current_lunar_date()
    
    return current_time, today_date, today_weekday, lunar_date
