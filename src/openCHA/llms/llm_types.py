from enum import Enum


class LLMType(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MEDGEMMA = "medgemma"
