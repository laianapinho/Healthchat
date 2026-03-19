from typing import List, Optional

from .schemas import EvaluationResult, ClinicalContext
from .metrics import (
    evaluate_completeness,
    evaluate_relevance,
    evaluate_safety_rules,
    evaluate_context_adherence,
)


class ResponseEvaluator:
    def evaluate(
        self,
        query: str,
        response: str,
        chat_history: Optional[List] = None,
        expected_topics: Optional[List[str]] = None,
        clinical_context: Optional[ClinicalContext] = None,
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

        context_adherence = evaluate_context_adherence(
            response=response,
            clinical_context=clinical_context,
            chat_history=chat_history,
        )

        final_score = round(
            (
                completeness.score * 0.30 +
                relevance.score * 0.25 +
                safety.score * 0.30 +
                context_adherence.score * 0.15
            ),
            3,
        )

        return EvaluationResult(
            completeness=completeness,
            relevance=relevance,
            safety=safety,
            context_adherence=context_adherence,
            final_score=final_score,
            meta={
                "weights": {
                    "completeness": 0.30,
                    "relevance": 0.25,
                    "safety": 0.30,
                    "context_adherence": 0.15,
                }
            }
        )
