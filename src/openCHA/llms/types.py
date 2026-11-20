from typing import Dict
from typing import Type

from openCHA.llms import AntropicLLM
from openCHA.llms import BaseLLM
from openCHA.llms import LLMType
from openCHA.llms import OpenAILLM
from openCHA.llms import GeminiLLM
from openCHA.llms.deepseek import DeepSeekLLM
from openCHA.llms.medgemma import MedGemmaLLM

LLM_TO_CLASS: Dict[LLMType, Type[BaseLLM]] = {
    LLMType.OPENAI: OpenAILLM,
    LLMType.ANTHROPIC: AntropicLLM,
    LLMType.DEEPSEEK: DeepSeekLLM,
    LLMType.GEMINI: GeminiLLM,
    LLMType.MEDGEMMA: MedGemmaLLM,
}
