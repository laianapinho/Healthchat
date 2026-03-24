from typing import List, Optional
from .schemas import EvaluationResult
from .metrics import (
    evaluate_completeness,
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

        safety = evaluate_safety_rules(
            query=query,
            response=response,
        )

        final_score = round(
            (
                completeness.score * 0.50 +
                safety.score * 0.50
            ),
            3,
        )

        return EvaluationResult(
            completeness=completeness,
            safety=safety,
            final_score=final_score,
            meta={
                "weights": {
                    "completeness": 0.50,
                    "safety": 0.50,
                }
            },
        )
