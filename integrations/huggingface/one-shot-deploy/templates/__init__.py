from .chatbot import CHATBOT_TEMPLATE
from .comparison import COMPARISON_TEMPLATE
from .dashboard import DASHBOARD_TEMPLATE
from .form_wizard import FORM_WIZARD_TEMPLATE

TEMPLATES = {
    "chatbot": CHATBOT_TEMPLATE,
    "comparison": COMPARISON_TEMPLATE,
    "dashboard": DASHBOARD_TEMPLATE,
    "form_wizard": FORM_WIZARD_TEMPLATE,
}


def get_template(template_type: str) -> str:
    return TEMPLATES.get(template_type, CHATBOT_TEMPLATE)
