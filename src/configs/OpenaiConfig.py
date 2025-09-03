from src.configs.ConfigBase import ConfigBase

from pydantic_settings import SettingsConfigDict

class OpenaiConfig(ConfigBase):
    model_config = SettingsConfigDict(
        env_prefix = "OPENAI_"
    )
    
    api_key: str
    base_url: str
    model_name: str
    
    temperature: float = 0.5
    request_timeout: int = 20
    max_tokens: int = 300