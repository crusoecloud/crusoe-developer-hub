from config import settings

NO_STRICT_TOOLS = ("nvidia/",)


def model_for(spec: str):
    provider, _, name = spec.partition(":")
    if provider != "crusoe":
        return spec
    try:
        from pydantic_ai.models.crusoe import CrusoeModel
    except ImportError:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIChatModel(name, provider=OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.crusoe_api_key))
    model = CrusoeModel(name)
    if name.startswith(NO_STRICT_TOOLS):
        return CrusoeModel(name, profile={**dict(model.profile), "openai_supports_strict_tool_definition": False,
                                         "supports_forced_tool_choice": False})
    return model
