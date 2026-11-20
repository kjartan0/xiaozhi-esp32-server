from datetime import datetime
import cnlunar
from plugins_func.register import register_function, ToolType, ActionResponse, Action

# NOTE (t_c30a9895): get_lunar is a Chinese-calendar tool (cnlunar) and for a Dutch
# household it is arguably irrelevant. Its descriptions are translated anyway so the
# LLM prompt stays Dutch, and the tool is kept (not removed) — removal is the owner's
# call, flagged in the handoff. The VALUES it returns are Chinese calendar names
# (干支, 生肖, 宜忌) because cnlunar has no Dutch/English data source for them; the
# Dutch instruction line below tells the model to answer in Dutch regardless.
get_lunar_function_desc = {
    "type": "function",
    "function": {
        "name": "get_lunar",
        "description": (
            "Vraagt Chinese kalender-/almanakinformatie op voor een datum "
            "(jaarcyclus, zonneperiodes, dierenriem, gunstige en ongunstige dagen enz.), "
            "standaard vandaag. Voor een simpele vraag als 'welke datum is het vandaag' "
            "gebruik je de context, niet deze tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "De datum om op te vragen, als YYYY-MM-DD, bijvoorbeeld 2024-01-01. Zonder waarde wordt de huidige datum gebruikt",
                },
                "query": {
                    "type": "string",
                    "description": "Wat je wilt opvragen, bijvoorbeeld de maankalenderdatum, de jaarcyclus, feestdagen, zonneperiodes, het dierenriemteken, sterrenbeeld, of gunstige/ongunstige dagen",
                },
            },
            "required": [],
        },
    },
}


@register_function("get_lunar", get_lunar_function_desc, ToolType.WAIT)
def get_lunar(date=None, query=None):
    """
    用于获取当前的阴历/农历，和天干地支、节气、生肖、星座、八字、宜忌等黄历信息
    """
    from core.utils.cache.manager import cache_manager, CacheType

    # 如果提供了日期参数，则使用指定日期；否则使用当前日期
    if date:
        try:
            now = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return ActionResponse(
                Action.REQLLM,
                f"Ongeldige datumnotatie, gebruik YYYY-MM-DD, bijvoorbeeld: 2024-01-01",
                None,
            )
    else:
        now = datetime.now()

    current_date = now.strftime("%Y-%m-%d")

    # 如果 query 为 None，则使用默认文本
    if query is None:
        query = "het Chinese kalenderjaar en de maankalenderdatum"

    # 尝试从缓存获取农历信息
    lunar_cache_key = f"lunar_info_{current_date}"
    cached_lunar_info = cache_manager.get(CacheType.LUNAR, lunar_cache_key)
    if cached_lunar_info:
        return ActionResponse(Action.REQLLM, cached_lunar_info, None)

    # The label text is Dutch (it is read by the model); the almanac values stay as
    # cnlunar produces them.
    response_text = (
        f"Beantwoord de vraag van de gebruiker met de onderstaande gegevens over {query}.\n"
        "Antwoord in het Nederlands, ook als hieronder Chinese kalendertermen staan; "
        "noem die termen alleen als de gebruiker er zelf naar vraagt.\n"
        "Chinese kalendergegevens:\n"
    )

    lunar = cnlunar.Lunar(now, godType="8char")
    response_text += (
        "%s jaar, %s maand, %s\n" % (lunar.lunarYearCn, lunar.lunarMonthCn[:-1], lunar.lunarDayCn)
        + "Jaarcyclus: %s jaar, %s maand, %s dag\n"
        % (lunar.year8Char, lunar.month8Char, lunar.day8Char)
        + "Dierenriem: jaar van de %s\n" % (lunar.chineseYearZodiac)
        + "Acht tekens: %s\n"
        % (
            " ".join(
                [lunar.year8Char, lunar.month8Char, lunar.day8Char, lunar.twohour8Char]
            )
        )
        + "Feestdagen vandaag: %s\n"
        % (
            ",".join(
                filter(
                    None,
                    (
                        lunar.get_legalHolidays(),
                        lunar.get_otherHolidays(),
                        lunar.get_otherLunarHolidays(),
                    ),
                )
            )
        )
        + "Zonneperiode vandaag: %s\n" % (lunar.todaySolarTerms)
        + "Volgende zonneperiode: %s %s-%s-%s\n"
        % (
            lunar.nextSolarTerm,
            lunar.nextSolarTermYear,
            lunar.nextSolarTermDate[0],
            lunar.nextSolarTermDate[1],
        )
        + "Zonneperiodes dit jaar: %s\n"
        % (
            ", ".join(
                [
                    f"{term}({date[0]}-{date[1]})"
                    for term, date in lunar.thisYearSolarTermsDic.items()
                ]
            )
        )
        + "Dierenriemconflict: %s\n" % (lunar.chineseZodiacClash)
        + "Sterrenbeeld: %s\n" % (lunar.starZodiac)
        + "Nayin: %s\n" % lunar.get_nayin()
        + "Taboes van de dag: %s\n" % (lunar.get_pengTaboo(delimit=", "))
        + "Dagdienst: %s\n" % lunar.get_today12DayOfficer()[0]
        + "Daggeest: %s(%s)\n"
        % (lunar.get_today12DayOfficer()[1], lunar.get_today12DayOfficer()[2])
        + "28 sterrenbeelden: %s\n" % lunar.get_the28Stars()
        + "Richting geluksgoden: %s\n" % " ".join(lunar.get_luckyGodsDirection())
        + "Vandaag: %s\n" % lunar.get_fetalGod()
        + "Gunstig: %s\n" % ",".join(lunar.goodThing[:10])
        + "Ongunstig: %s\n" % ",".join(lunar.badThing[:10])
        + "(Standaard worden alleen het kalenderjaar en de maankalenderdatum gegeven; "
        "de gunstige/ongunstige dagen alleen als de gebruiker daarom vraagt)"
    )

    # 缓存农历信息
    cache_manager.set(CacheType.LUNAR, lunar_cache_key, response_text)

    return ActionResponse(Action.REQLLM, response_text, None)
