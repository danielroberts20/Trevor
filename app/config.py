"""
Central configuration. All environment variables are read here.
Other modules import `settings` from this file rather than reading os.environ directly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM provider selection
    llm_provider: str = Field(alias="LLM_PROVIDER")  # "ollama" | "openai" | "anthropic"

    # Ollama
    ollama_base_url: str = Field(alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(alias="OLLAMA_MODEL")
    ollama_embed_model: str = Field(alias="OLLAMA_EMBED_MODEL")

    # OpenAI
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(alias="OPENAI_MODEL")

    # Anthropic
    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(alias="ANTHROPIC_MODEL")

    # Storage paths (inside container)
    db_path: str = Field(alias="DB_PATH")
    chroma_path: str = Field(alias="CHROMA_PATH")

    # Ingestion filter
    travel_start_date: str = Field(alias="TRAVEL_START_DATE")

    # API security
    trevor_api_key: str = Field(alias="TREVOR_API_KEY")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
