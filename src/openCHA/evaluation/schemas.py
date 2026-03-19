from typing import List, Dict, Any
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    score: float
    passed: bool
    details: List[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    completeness: MetricResult
    relevance: MetricResult
    safety: MetricResult
    final_score: float
    meta: Dict[str, Any] = Field(default_factory=dict)
