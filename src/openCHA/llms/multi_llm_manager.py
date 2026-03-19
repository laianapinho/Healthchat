import time
import logging
import hashlib
from typing import Dict, Any, List, Optional, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps

from openCHA.llms import initialize_llm, LLMType
from openCHA.orchestrator import Orchestrator
from openCHA.planners import PlannerType
from openCHA.datapipes import DatapipeType
from openCHA.response_generators import ResponseGeneratorType

# NOVO
from openCHA.evaluation import ResponseEvaluator

logger = logging.getLogger(__name__)


class LLMFullResponse(TypedDict):
    content: Optional[str]
    time_ms: Optional[float]
    error: Optional[str]
    model_name: str
    timestamp: float
    tokens_estimate: Optional[int]
    planning_time_ms: Optional[float]
    generation_time_ms: Optional[float]


class MultiLLMResultFull(TypedDict):
    responses: Dict[str, Optional[str]]
    times: Dict[str, Optional[float]]
    planning_times: Dict[str, Optional[float]]
    generation_times: Dict[str, Optional[float]]
    errors: Dict[str, Optional[str]]

    # NOVO
    evaluations: Dict[str, Optional[Dict[str, Any]]]

    metadata: Dict[str, Any]


def retry_on_failure(max_retries: int = 2, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:

                    last_exception = e
                    error_msg = str(e).lower()

                    if any(x in error_msg for x in ["invalid", "unauthorized", "forbidden"]):
                        raise

                    if attempt < max_retries:

                        wait_time = delay * (2 ** attempt)

                        logger.warning(
                            f"Tentativa {attempt + 1}/{max_retries + 1} falhou: {e}. "
                            f"Tentando novamente em {wait_time}s..."
                        )

                        time.sleep(wait_time)

                    else:
                        logger.error(f"Todas as {max_retries + 1} tentativas falharam")

            raise last_exception

        return wrapper

    return decorator


class MultiLLMManager:

    def __init__(
        self,
        enable_cache: bool = False,
        default_timeout: int = 120,
        max_workers: int = 3,
        enable_retry: bool = True,
        retry_attempts: int = 2,
        restrict_to_health_only: bool = False,
        use_llm_classifier: bool = False
    ):

        logger.info("🔧 Inicializando MultiLLMManager...")

        self.enable_cache = enable_cache
        self.default_timeout = default_timeout
        self.max_workers = max_workers
        self.enable_retry = enable_retry
        self.retry_attempts = retry_attempts

        self.restrict_to_health_only = restrict_to_health_only
        self.use_llm_classifier = use_llm_classifier

        self._cache: Dict[str, Dict[str, Any]] = {}

        # NOVO
        self.evaluator = ResponseEvaluator()

        self.available_models = {
            "chatgpt": LLMType.OPENAI,
            "deepseek": LLMType.DEEPSEEK,
            "gemini": LLMType.GEMINI,
        }

        self.models: Dict[str, LLMType] = {}

        self._initialize_models()

        logger.info(
            f"✅ MultiLLMManager inicializado com {len(self.models)} modelos: "
            f"{', '.join(self.models.keys())}"
        )

    def _initialize_models(self) -> None:

        logger.info("Inicializando modelos...")

        test_query = "What is 2 + 2?"

        for name, llm_type in self.available_models.items():

            try:

                llm = initialize_llm(llm_type)

                test_response = llm.generate(
                    test_query,
                    max_tokens=50,
                    temperature=0
                )

                if test_response and len(test_response.strip()) > 0:

                    self.models[name] = llm_type

                    logger.info(f"✅ {name.upper()} inicializado")

                else:

                    logger.warning(f"⚠️ {name.upper()} resposta vazia")

            except Exception as e:

                logger.warning(
                    f"⚠️ {name.upper()} falhou: {type(e).__name__}: {e}"
                )

        if not self.models:
            raise RuntimeError(
                "❌ Nenhum modelo foi inicializado"
            )

    def get_available_models(self) -> List[str]:

        return list(self.models.keys())

    def clear_cache(self) -> None:

        self._cache.clear()

        logger.info("Cache limpo")

    def _estimate_tokens(self, text: str) -> int:

        return len(text) // 4 if text else 0

    def _create_orchestrator_for_model(self, model_type: LLMType) -> Orchestrator:

        return Orchestrator.initialize(
            planner_llm=model_type,
            planner_name=PlannerType.TREE_OF_THOUGHT,
            datapipe_name=DatapipeType.MEMORY,
            promptist_name="",
            response_generator_llm=model_type,
            response_generator_name=ResponseGeneratorType.BASE_GENERATOR,
            available_tasks=[],
            previous_actions=[],
            verbose=False,
            restrict_to_health_only=False,
        )

    def _generate_with_model_orchestrated(
        self,
        name: str,
        model_type: LLMType,
        query: str,
        timeout: int,
        **kwargs
    ) -> LLMFullResponse:

        start_time = time.time()

        try:

            cache_key = hashlib.md5(f"{name}:{query}".encode()).hexdigest()

            if self.enable_cache and cache_key in self._cache:

                cached = self._cache[cache_key]

                return {
                    "content": cached["content"],
                    "time_ms": cached.get("time_ms", 0.0),
                    "error": None,
                    "model_name": name,
                    "timestamp": time.time(),
                    "tokens_estimate": self._estimate_tokens(cached["content"]),
                    "planning_time_ms": cached.get("planning_time_ms", 0.0),
                    "generation_time_ms": cached.get("generation_time_ms", 0.0),
                }

            def generate():

                orchestrator = self._create_orchestrator_for_model(model_type)

                response, timings = orchestrator.run(
                    query=query,
                    meta=[],
                    history="",
                    use_history=False,
                    return_timings=True,
                    **kwargs
                )

                planning = float(timings.get("planning_time_ms", 0.0))
                generation = float(timings.get("generation_time_ms", 0.0))

                return response, planning, generation

            if self.enable_retry:

                generate = retry_on_failure(
                    max_retries=self.retry_attempts
                )(generate)

            with ThreadPoolExecutor(max_workers=1) as executor:

                future = executor.submit(generate)

                response, planning, generation = future.result(timeout=timeout)

            elapsed = round((time.time() - start_time) * 1000, 2)

            if self.enable_cache and response:

                self._cache[cache_key] = {
                    "content": response,
                    "planning_time_ms": planning,
                    "generation_time_ms": generation,
                    "time_ms": elapsed,
                }

            return {
                "content": response,
                "time_ms": elapsed,
                "error": None,
                "model_name": name,
                "timestamp": time.time(),
                "tokens_estimate": self._estimate_tokens(response),
                "planning_time_ms": planning,
                "generation_time_ms": generation,
            }

        except Exception as e:

            elapsed = round((time.time() - start_time) * 1000, 2)

            return {
                "content": None,
                "time_ms": elapsed,
                "error": str(e),
                "model_name": name,
                "timestamp": time.time(),
                "tokens_estimate": 0,
                "planning_time_ms": 0.0,
                "generation_time_ms": 0.0,
            }

    def generate_all_with_orchestration(
        self,
        query: str,
        models: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        parallel: bool = True,
        **kwargs
    ) -> MultiLLMResultFull:

        if models is None:
            selected = self.models
        else:
            selected = {k: v for k, v in self.models.items() if k in models}

        timeout = timeout or self.default_timeout

        start_total = time.time()

        if parallel and len(selected) > 1:

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

                futures = {
                    name: executor.submit(
                        self._generate_with_model_orchestrated,
                        name,
                        model,
                        query,
                        timeout,
                        **kwargs
                    )
                    for name, model in selected.items()
                }

                raw_results = {
                    name: future.result()
                    for name, future in futures.items()
                }

        else:

            raw_results = {
                name: self._generate_with_model_orchestrated(
                    name,
                    model,
                    query,
                    timeout,
                    **kwargs
                )
                for name, model in selected.items()
            }

        total_time = round((time.time() - start_total) * 1000, 2)

        # NOVO — avaliação das respostas

        evaluations: Dict[str, Optional[Dict[str, Any]]] = {}

        for name, res in raw_results.items():

            if res["content"] and not res["error"]:

                evaluation = self.evaluator.evaluate(
                    query=query,
                    response=res["content"]
                )

                evaluations[name] = evaluation.model_dump()

            else:

                evaluations[name] = None

        result: MultiLLMResultFull = {

            "responses": {n: r["content"] for n, r in raw_results.items()},
            "times": {n: r["time_ms"] for n, r in raw_results.items()},
            "planning_times": {n: r["planning_time_ms"] for n, r in raw_results.items()},
            "generation_times": {n: r["generation_time_ms"] for n, r in raw_results.items()},
            "errors": {n: r["error"] for n, r in raw_results.items()},

            # NOVO
            "evaluations": evaluations,

            "metadata": {
                "total_time_ms": total_time,
                "parallel_execution": parallel,
                "models_count": len(selected),
                "success_count": sum(1 for r in raw_results.values() if r["content"]),
                "failed_count": sum(1 for r in raw_results.values() if r["error"]),
                "total_tokens_estimate": sum(r["tokens_estimate"] for r in raw_results.values()),
                "query_length": len(query),
                "timestamp": time.time(),
                "execution_type": "full_orchestration",
            },
        }

        return result
