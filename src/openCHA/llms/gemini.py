import logging
from typing import Any, Dict, List, Optional
from openCHA.llms import BaseLLM
from openCHA.utils import get_from_dict_or_env
from pydantic import model_validator, Field

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """
    Implementação da API Gemini do Google usando o SDK oficial.

    Documentação oficial: https://ai.google.dev/gemini-api/docs/get-started/python

    Modelos suportados:
        - gemini-2.0-flash-exp: Modelo experimental mais recente
        - gemini-1.5-pro: Modelo Pro com alta capacidade
        - gemini-1.5-flash: Modelo rápido e eficiente
        - gemini-1.0-pro: Modelo Pro da primeira geração

    Exemplos de uso:
        >>> llm = GeminiLLM()
        >>> resposta = llm.generate("Explique o que é inteligência artificial")
        >>>
        >>> # Com parâmetros customizados
        >>> resposta = llm.generate(
        ...     "Escreva um poema",
        ...     model_name="gemini-pro",
        ...     temperature=1.0,
        ...     max_tokens=500
        ... )
        >>>
        >>> # Com histórico de conversa
        >>> resposta = llm.generate(
        ...     "Continue a história",
        ...     conversation_history=[
        ...         {"role": "user", "parts": ["Era uma vez..."]},
        ...         {"role": "model", "parts": ["Um reino distante..."]}
        ...     ]
        ... )
    """

    models: Dict[str, int] = {
        "gemini-3.5-flash-lite": 512,          # Modelo lite rápido
    }

    llm_module: Any = Field(default=None, exclude=True)
    api_key: str = Field(default="", exclude=True)
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None
    default_model: str = "gemini-3.5-flash-lite"

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Valida que a API key do Gemini e o SDK estão disponíveis.
        Carrega de values dict ou variável de ambiente GEMINI_API_KEY.
        """
        gemini_api_key = get_from_dict_or_env(
            values, "gemini_api_key", "GEMINI_API_KEY"
        )

        if not gemini_api_key:
            raise ValueError(
                "API key do Gemini não encontrada. "
                "Configure a variável de ambiente GEMINI_API_KEY "
                "ou passe 'gemini_api_key' no construtor."
            )

        values["api_key"] = gemini_api_key

        # Importa e configura o SDK do Google
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            values["llm_module"] = genai
            logger.info("GeminiLLM inicializado com sucesso")
            logger.debug(f"Modelos disponíveis: {list(values.get('models', {}).keys())}")
        except ImportError:
            raise ValueError(
                "Pacote 'google-generativeai' não encontrado. "
                "Instale com: pip install google-generativeai"
            )
        except Exception as e:
            raise ValueError(f"Erro ao configurar Gemini SDK: {str(e)}")

        return values

    def get_model_names(self) -> List[str]:
        """
        Retorna a lista de modelos suportados.

        Returns:
            List[str]: Lista com nomes dos modelos disponíveis
        """
        return list(self.models.keys())

    def get_max_context_length(self, model_name: str) -> int:
        """
        Retorna o tamanho máximo do contexto para um modelo específico.

        Args:
            model_name: Nome do modelo

        Returns:
            int: Número máximo de tokens de contexto
        """
        return self.models.get(model_name, 32768)

    def _validate_model(self, model_name: str) -> None:
        """
        Valida se o modelo especificado é suportado.

        Args:
            model_name: Nome do modelo a validar

        Raises:
            ValueError: Se o modelo não for suportado
        """
        if model_name not in self.get_model_names():
            raise ValueError(
                f"Modelo '{model_name}' não é suportado. "
                f"Modelos válidos: {', '.join(self.get_model_names())}"
            )

    def _validate_parameters(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int]
    ) -> None:
        """
        Valida os parâmetros de geração.

        Args:
            max_tokens: Número máximo de tokens
            temperature: Valor de temperatura
            top_p: Valor de top_p
            top_k: Valor de top_k (opcional)

        Raises:
            ValueError: Se os parâmetros forem inválidos
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens deve ser maior que 0")

        if not 0 <= temperature <= 2:
            raise ValueError("temperature deve estar entre 0 e 2")

        if not 0 <= top_p <= 1:
            raise ValueError("top_p deve estar entre 0 e 1")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k deve ser maior que 0 ou None")

    def _parse_response(self, response: Any) -> str:
        """
        Extrai o texto da resposta do Gemini.

        O SDK do Gemini pode retornar diferentes estruturas dependendo
        do tipo de resposta e configurações de segurança.

        Args:
            response: Objeto de resposta do Gemini

        Returns:
            str: Texto gerado pelo modelo

        Raises:
            ValueError: Se não conseguir extrair texto da resposta
        """
        try:
            # Tenta acessar o texto diretamente
            if hasattr(response, "text") and response.text:
                return response.text

            # Tenta acessar via candidates
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, "content") and candidate.content:
                    # Extrai texto das parts
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        parts_text = []
                        for part in candidate.content.parts:
                            if hasattr(part, "text"):
                                parts_text.append(part.text)
                        if parts_text:
                            return "".join(parts_text)

                # Fallback para text direto no candidate
                if hasattr(candidate, "text"):
                    return candidate.text

            # Se chegou aqui, verifica se foi bloqueado por segurança
            if hasattr(response, "prompt_feedback"):
                feedback = response.prompt_feedback
                if hasattr(feedback, "block_reason"):
                    raise ValueError(
                        f"Conteúdo bloqueado por segurança: {feedback.block_reason}"
                    )

            logger.warning("Não foi possível extrair texto da resposta")
            logger.debug(f"Estrutura da resposta: {dir(response)}")

            raise ValueError(
                "Não foi possível extrair texto da resposta do Gemini. "
                "A resposta pode estar vazia ou bloqueada por filtros de segurança."
            )

        except AttributeError as e:
            logger.error(f"Erro ao acessar atributos da resposta: {e}")
            raise ValueError(f"Estrutura de resposta inesperada: {str(e)}")

    def _prepare_prompt(
        self,
        query: str,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:
        """
        Prepara o prompt e histórico para o Gemini.

        Args:
            query: Pergunta ou prompt do usuário
            system_instruction: Instruções do sistema (opcional)
            conversation_history: Histórico formatado para Gemini (opcional)
                Formato: [{"role": "user"/"model", "parts": ["texto"]}]

        Returns:
            tuple: (conteúdo_para_enviar, histórico_ou_None)
        """
        # Se há histórico, adiciona a query atual
        if conversation_history:
            # Cria uma cópia para não modificar o original
            history = conversation_history.copy()
            history.append({"role": "user", "parts": [query]})
            logger.debug(f"Usando histórico com {len(history)} mensagens")
            return None, history

        # Sem histórico, envia apenas a query
        logger.debug("Gerando resposta sem histórico de conversa")
        return query, None

    def generate(self, query: str, **kwargs: Any) -> str:
        """
        Gera uma resposta usando o modelo Gemini.

        Args:
            query: Pergunta ou prompt do usuário
            **kwargs: Parâmetros opcionais:
                - model_name (str): Nome do modelo (padrão: "gemini-1.5-flash")
                - max_tokens (int): Limite de tokens de saída (padrão: 2048)
                - temperature (float): Controle de criatividade 0-2 (padrão: 0.7)
                - top_p (float): Núcleo de amostragem 0-1 (padrão: 1.0)
                - top_k (int): Top-k amostragem (padrão: None)
                - system_instruction (str): Instruções do sistema
                - conversation_history (List[Dict]): Histórico no formato Gemini
                - stop_sequences (List[str]): Sequências de parada

        Returns:
            str: Resposta gerada pelo modelo

        Raises:
            ValueError: Se parâmetros forem inválidos
            RuntimeError: Se houver erro na comunicação com a API
        """
        # Extrai e valida parâmetros
        model_name = kwargs.get("model_name", self.default_model)
        self._validate_model(model_name)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)

        self._validate_parameters(max_tokens, temperature, top_p, top_k)

        system_instruction = kwargs.get("system_instruction", None)
        conversation_history = kwargs.get("conversation_history", None)
        stop_sequences = kwargs.get("stop_sequences", None)

        # Log do modelo e tokens sendo utilizados
        print("here", max_tokens, model_name)

        # Prepara o prompt
        content, history = self._prepare_prompt(
            query,
            system_instruction,
            conversation_history
        )

        # Configura o modelo
        model_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if top_k is not None:
            model_config["top_k"] = top_k

        if stop_sequences:
            model_config["stop_sequences"] = stop_sequences

        logger.info(f"Gerando resposta com modelo: {model_name}")
        logger.debug(f"Parâmetros: {model_config}")

        try:
            # Cria instância do modelo
            model_kwargs = {"model_name": model_name}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            model = self.llm_module.GenerativeModel(**model_kwargs)

            # Gera conteúdo (APENAS UMA VEZ)
            if history:
                # Com histórico, usa chat
                chat = model.start_chat(history=history[:-1])
                response = chat.send_message(
                    history[-1]["parts"][0],
                    generation_config=model_config
                )
            else:
                # Sem histórico, geração direta
                response = model.generate_content(
                    content,
                    generation_config=model_config
                )

            result = self._parse_response(response)
            logger.info("Resposta gerada com sucesso")
            logger.debug(f"Tamanho da resposta: {len(result)} caracteres")

            return result

        except ValueError as e:
            # Erros de validação ou parsing já estão formatados
            logger.error(f"Erro de validação: {e}")
            raise

        except Exception as e:
            error_msg = f"Erro ao gerar conteúdo com Gemini: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Tenta identificar erros comuns
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str:
                error_msg = "Limite de requisições excedido. Tente novamente mais tarde."
            elif "api key" in error_str:
                error_msg = "Erro de autenticação. Verifique sua API key do Gemini."
            elif "invalid argument" in error_str:
                error_msg = f"Argumento inválido na requisição: {str(e)}"

            raise RuntimeError(error_msg)
