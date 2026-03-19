from typing import List, Optional

from .schemas import EvaluationResult
from .metrics import (
    evaluate_completeness,
    evaluate_relevance,
    evaluate_safety_rules,
)


class ResponseEvaluator:
    def evaluate(
        self,
        query: str,
        response: str,
        chat_history: Optional[List] = None,
        expected_topics: Optional[List[str]] = None,
        clinical_context: Optional[object] = None,
    ) -> EvaluationResult:
        completeness = evaluate_completeness(
            query=query,
            response=response,
            expected_topics=expected_topics,
        )

        relevance = evaluate_relevance(
            query=query,
            response=response,
        )

        safety = evaluate_safety_rules(
            query=query,
            response=response,
        )

        final_score = round(
            (
                completeness.score * 0.35 +
                relevance.score * 0.30 +
                safety.score * 0.35
            ),
            3,
        )

        return EvaluationResult(
            completeness=completeness,
            relevance=relevance,
            safety=safety,
            context_adherence={
                "score": 0.0,
                "passed": True,
                "details": ["Métrica de contexto removida temporariamente"],
            },
            final_score=final_score,
            meta={
                "weights": {
                    "completeness": 0.35,
                    "relevance": 0.30,
                    "safety": 0.35,
                }
            }
        )
