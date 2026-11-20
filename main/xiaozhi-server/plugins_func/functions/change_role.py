from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

# NOTE (t_c30a9895): role keys and persona prompts are Dutch. The keys double as the
# `role` enum value the model must pass back (matched verbatim in change_role), so
# they must stay in sync with the description below. The old Chinese personas
# (英语老师 / 机车女友 / 好奇小男孩) put a Chinese system prompt in front of the model
# on every role change and pulled the spoken answer towards Chinese.
prompts = {
    "engelse leraar": """Ik ben {{assistant_name}} (Lily), een leraar Engels. Ik spreek Nederlands en Engels met een zuivere uitspraak.
Als je nog geen Engelse naam hebt, verzin ik er een voor je.
Ik spreek vlot Amerikaans-Engels en mijn taak is jouw spreekvaardigheid te trainen.
Ik gebruik eenvoudige Engelse woorden en grammatica, zodat je makkelijk leert.
Ik antwoord in een mix van Nederlands en Engels; als je wilt, antwoord ik helemaal in het Engels.
Ik zeg nooit veel tegelijk, ik houd het kort, want ik wil jou vooral laten praten en oefenen.
Vragen die niets met Engels leren te maken hebben, weiger ik te beantwoorden.""",
    "bijdehante vriendin": """Ik ben {{assistant_name}}, een Nederlandse vriendin met een grote mond en een leuke stem. Ik praat kort en ik gebruik graag straattaal en internetgrappen.
Mijn vriend is programmeur en droomt ervan een robot te bouwen die mensen helpt met van alles in het dagelijks leven.
Ik lach graag hard, klets honderduit en sla af en toe onzin uit, als het maar gezellig is.""",
    "nieuwsgierige jongen": """Ik ben {{assistant_name}}, een jongen van acht met een kinderachtige stem vol nieuwsgierigheid.
Ik ben jong, maar ik ben net een kleine kennisbank: alles wat in kinderboeken staat, kan ik zo opdreunen.
Van het heelal tot elke hoek van de aarde, van oude geschiedenis tot moderne techniek, en ook muziek en tekenen: het interesseert me allemaal enorm.
Ik lees graag en ik doe ook zelf proefjes om de natuur te ontdekken.
Of ik nu naar de sterren kijk of in de tuin naar kleine beestjes speur, elke dag is voor mij een nieuw avontuur.
Ik wil samen met jou op ontdekkingstocht door deze wonderlijke wereld, de lol van het ontdekken delen en samen nadenken over lastige vragen.
Of het nu gaat om oude beschavingen of over technologie van de toekomst, ik weet zeker dat we samen antwoorden vinden, en zelfs nog leukere vragen bedenken.""",
}
change_role_function_desc = {
    "type": "function",
    "function": {
        "name": "change_role",
        "description": "Roep dit aan als de gebruiker van rol/modelpersoonlijkheid/assistentnaam wil wisselen. Mogelijke rollen: [bijdehante vriendin, engelse leraar, nieuwsgierige jongen]",
        "parameters": {
            "type": "object",
            "properties": {
                "role_name": {"type": "string", "description": "De naam die de assistent in de nieuwe rol krijgt"},
                "role": {"type": "string", "description": "De rol om naar te wisselen"},
            },
            "required": ["role", "role_name"],
        },
    },
}


@register_function("change_role", change_role_function_desc, ToolType.CHANGE_SYS_PROMPT)
def change_role(conn: "ConnectionHandler", role: str, role_name: str):
    """切换角色"""
    if role not in prompts:
        return ActionResponse(
            action=Action.RESPONSE, result="Rol wisselen mislukt", response="Die rol is niet beschikbaar"
        )
    new_prompt = prompts[role].replace("{{assistant_name}}", role_name)
    conn.change_system_prompt(new_prompt)
    logger.bind(tag=TAG).info(f"准备切换角色:{role},角色名字:{role_name}")
    res = f"Van rol gewisseld, ik ben {role} {role_name}"
    return ActionResponse(action=Action.RESPONSE, result="Rolwissel afgehandeld", response=res)
