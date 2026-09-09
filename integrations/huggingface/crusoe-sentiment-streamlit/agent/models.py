"""Model routing and LLM client factory (OpenRouter)."""
from langchain_openai import ChatOpenAI
from config import settings

# OpenRouter model IDs
MODEL_GEMMA       = "google/gemma-3-12b-it"
MODEL_DEEPSEEK_V3 = "deepseek/deepseek-chat-v3-0324"
MODEL_LLAMA       = "meta-llama/llama-3.3-70b-instruct"
MODEL_QWEN        = "qwen/qwen3-235b-a22b"
MODEL_DEEPSEEK_R1 = "deepseek/deepseek-r1-0528"
MODEL_KIMI        = "moonshotai/kimi-k2"
MODEL_GPT_OSS     = "openai/gpt-4o"          # gpt-oss-120b fallback


def _llm(model: str, temperature: float = 0.1, max_tokens: int = 512) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://crusoe.ai",
            "X-Title": "Crusoe Sentiment Agent",
        },
    )


def _token_count(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text) // 4


SARCASM_SIGNALS = {"lol", "sure", "right", "totally", "definitely", "wow", "/s", "🙄", "😂"}


def route_model(source: str, body: str, title: str = "") -> str:
    text = f"{title} {body}".lower()
    if sum(1 for s in SARCASM_SIGNALS if s in text) >= 2:
        return MODEL_DEEPSEEK_R1
    if source == "twitter":
        return MODEL_GEMMA
    if source == "reddit":
        return MODEL_DEEPSEEK_V3
    if source == "news":
        return MODEL_QWEN if _token_count(text) >= 2000 else MODEL_LLAMA
    return MODEL_DEEPSEEK_V3


def analysis_llm(model_id: str) -> ChatOpenAI:
    return _llm(model_id, temperature=0.1, max_tokens=512)

def synthesis_llm() -> ChatOpenAI:
    return _llm(MODEL_GPT_OSS, temperature=0.3, max_tokens=2048)

def trend_llm() -> ChatOpenAI:
    return _llm(MODEL_KIMI, temperature=0.2, max_tokens=1024)
