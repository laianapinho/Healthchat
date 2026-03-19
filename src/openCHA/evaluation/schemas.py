from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    score: float
    passed: bool
    details: List[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    completeness: MetricResult
    relevance: MetricResult
    safety: MetricResult
    context_adherence: MetricResult
    final_score: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class ClinicalContext(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    vital_signs: Dict[str, Any] = Field(default_factory=dict)
    red_flags: List[str] = Field(default_factory=list)
