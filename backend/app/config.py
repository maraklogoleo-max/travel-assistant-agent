from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    amap_api_key: str = Field(default="", validation_alias=AliasChoices("AMAP_WEB_API_KEY", "AMAP_API_KEY"))
    database_path: Path = Field(default=Path("./data/weather_helper.db"))
    frontend_origin: str = "http://localhost:3000"
    timezone: str = "Asia/Shanghai"
    amap_timeout_seconds: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
