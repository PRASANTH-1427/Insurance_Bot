from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "InsureAssist AI"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    debug: bool = False
    database_url: str = "sqlite:///./insureassist.db"

    # LLM / Groq settings
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("APP_NAME", "InsureAssist AI"),
            api_prefix=os.getenv("API_PREFIX", "/api/v1"),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            database_url=os.getenv("DATABASE_URL", "sqlite:///./insureassist.db"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
