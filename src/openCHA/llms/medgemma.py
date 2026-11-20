import os
import re
from typing import Any, Dict
from pydantic import Field, model_validator
from openCHA.llms.deepseek import DeepSeekLLM

OLLAMA_URL = "http://localhost:11434/v1"


class MedGemmaLLM(DeepSeekLLM):
    models: Dict[str, int] = {"medgemma1.5:4b": 128000}
    api_key: str = Field(default="EMPTY", exclude=True)
    base_url: str = OLLAMA_URL
    default_model: str = "medgemma1.5:4b"
    max_tokens: int = 2000
    timeout: int = 300

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        values.setdefault("base_url", os.getenv("MEDGEMMA_BASE_URL", OLLAMA_URL))
        values["api_key"] = values.get("medgemma_api_key") or os.getenv("MEDGEMMA_API_KEY", "EMPTY")
        return values

    def _parse_response(self, response: Dict[str, Any]) -> str:
        text = super()._parse_response(response)
        # remove o bloco de raciocínio (completo ou cortado pelo limite de tokens)
        text = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
        text = re.sub(r"<unused94>.*$", "", text, flags=re.DOTALL)
        return text.strip()
    
    def generate(self, query: str, **kwargs: Any) -> str:
        # o modelo "pensa" antes de responder; com poucos tokens a resposta some
        kwargs["max_tokens"] = max(kwargs.get("max_tokens", self.max_tokens), 2000)
        return super().generate(query, **kwargs)