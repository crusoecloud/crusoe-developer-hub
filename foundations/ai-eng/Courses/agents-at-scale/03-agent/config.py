from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    crusoe_api_key: str
    llm_base_url: str = "https://api.inference.crusoecloud.com/v1"
    model: str = "zai-org/GLM-5.3"
    fallback_models: list[str] = ["deepseek-ai/DeepSeek-V4.1-Flash", "openai/gpt-oss-120b"]
    request_timeout_s: float = 120
    max_retries: int = 4
    max_steps: int = 25
    max_concurrency: int = 4
    context_window: int = 128000
    results_dir: str = "results/runs"


settings = Settings()
client = AsyncOpenAI(api_key=settings.crusoe_api_key, base_url=settings.llm_base_url,
                     timeout=settings.request_timeout_s, max_retries=settings.max_retries)
