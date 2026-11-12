"""Pydantic v2 data contracts for LLM outputs, OPM decisions, and calibration metrics."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMSampleOutput(BaseModel):
    """Structured output contract for a single stochastic Monte Carlo LLM completion."""

    model_config = ConfigDict(extra="ignore")

    canonical_answer: str = Field(
        ...,
        description="Concise, normalized canonical answer string or 'UNANSWERABLE'.",
    )
    raw_reasoning: str = Field(
        default="",
        description="Step-by-step verification against the retrieved context.",
    )
    context_entailed: bool = Field(
        ...,
        description="True if and only if the retrieved context explicitly entails the answer.",
    )
    confidence_claim: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Self-reported verbalized confidence in [0.0, 1.0].",
    )

    @field_validator("confidence_claim", mode="before")
    @classmethod
    def clamp_confidence(cls, value: Any) -> float:
        """Ensure confidence_claim is coerced to a valid float in [0.0, 1.0]."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    @field_validator("canonical_answer", mode="before")
    @classmethod
    def clean_canonical_answer(cls, value: Any) -> str:
        """Ensure canonical_answer is always a non-empty stripped string."""
        text = str(value or "").strip()
        return text if text else "UNANSWERABLE"


class DecisionAction(str, Enum):
    """Action space A = {Accept, Abstain} for the Optimal Predictor Machine."""

    ACCEPT = "Accept"
    ABSTAIN = "Abstain"


class MEUResult(BaseModel):
    """Complete decision-theoretic output of the OPM Engine for a single query."""

    p_correct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated epistemic probability p = P(theta_1 | q, C, I).",
    )
    eu_accept: float = Field(
        ...,
        description="Expected utility EU(a_accept | p).",
    )
    eu_abstain: float = Field(
        ...,
        description="Expected utility EU(a_abstain | p).",
    )
    optimal_action: DecisionAction = Field(
        ...,
        description="Action maximizing expected utility: Accept or Abstain.",
    )
    decision_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Critical probability threshold p* above which Accept is optimal.",
    )
    modal_answer: str = Field(
        ...,
        description="Modal canonical candidate answer y* across Monte Carlo samples.",
    )
    final_response: str = Field(
        default="",
        description="User-facing response (modal_answer on Accept, fallback message on Abstain).",
    )
    context_chunks: List[str] = Field(
        default_factory=list,
        description="Top-k retrieved text chunks provided as evidence.",
    )
    top_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum cosine similarity s_max between query and retrieved chunks.",
    )
    mc_agreement: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Raw Monte Carlo agreement rate f_MC before retrieval gating.",
    )
    samples: List[LLMSampleOutput] = Field(
        default_factory=list,
        description="Individual stochastic completions collected during Monte Carlo sampling.",
    )


class EvalQueryRecord(BaseModel):
    """Benchmark evaluation record from data/golden_eval.jsonl."""

    id: str = Field(..., description="Unique benchmark query identifier.")
    domain: str = Field(..., description="Knowledge corpus domain identifier.")
    query: str = Field(..., description="Natural language evaluation question.")
    ground_truth_answer: str = Field(
        ..., description="Reference answer or 'UNANSWERABLE'."
    )
    is_answerable: bool = Field(
        ...,
        description="True if the corpus contains the answer, False if adversarial/absent.",
    )


class CalibrationBin(BaseModel):
    """Empirical calibration summary for one confidence interval bin."""

    bin_range: str = Field(
        ..., description="String representation of interval, e.g. '(0.8, 1.0]'."
    )
    count: int = Field(
        ..., ge=0, description="Number of queries falling into this bin."
    )
    empirical_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Observed fraction of true answers in bin."
    )
    mean_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Average predicted probability p in bin."
    )
    calibration_error: float = Field(
        ..., ge=0.0, le=1.0, description="Absolute gap |acc(B_b) - conf(B_b)|."
    )


class EvaluationReport(BaseModel):
    """Aggregate benchmark report with calibration metrics and utility comparison."""

    total_queries: int = Field(..., ge=0)
    brier_score: float = Field(..., ge=0.0, le=1.0)
    ece: float = Field(..., ge=0.0, le=1.0)
    bins: List[CalibrationBin] = Field(default_factory=list)
    opm_total_utility: float = Field(...)
    naive_total_utility: float = Field(...)
    utility_gain: float = Field(...)
    query_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-query evaluation breakdown for telemetry and visualization.",
    )
