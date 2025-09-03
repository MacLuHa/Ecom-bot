from pydantic_settings import BaseSettings
from pydantic import Field

from src.configs.OpenaiConfig import OpenaiConfig

class LlmConfig(BaseSettings):
    
    openai: OpenaiConfig = Field(default_factory=OpenaiConfig)
    
    @classmethod
    def load(cls) -> "LlmConfig":
        return cls()
    