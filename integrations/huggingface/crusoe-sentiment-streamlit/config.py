import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "crusoe-sentiment-bot/1.0"

    TWITTER_BEARER_TOKEN: str = ""

    GNEWS_API_KEY: str = ""
    NEWSAPI_KEY: str = ""

    DATABASE_URL: str = "sqlite:///crusoe_sentiment.db"

    MAX_POSTS_PER_SOURCE: int = 50

    @property
    def db_url(self) -> str:
        url = self.DATABASE_URL
        # Heroku uses postgres:// prefix; SQLAlchemy needs postgresql+pg8000://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+pg8000://", 1)
        elif url.startswith("postgresql://") and "pg8000" not in url:
            url = url.replace("postgresql://", "postgresql+pg8000://", 1)
        return url


settings = Settings()
