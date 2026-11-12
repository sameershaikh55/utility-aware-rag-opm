"""Optimal Predictor Machine (OPM) Decision Engine: Monte Carlo sampling & MEU."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Set, Tuple

from groq import Groq

from src.config import ABSTAIN_FALLBACK_MESSAGE, AppConfig, get_groq_client, settings
from src.logger import get_logger
from src.prompts import PromptTemplate, load_prompt
from src.retriever import CorpusRetriever, _default_retriever
from src.schema import DecisionAction, LLMSampleOutput, MEUResult

logger = get_logger(__name__)

_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "and",
    "or",
    "that",
    "this",
    "it",
}


def normalize_answer(text: str) -> str:
    """Normalize a canonical answer string for equivalence comparison."""
    cleaned = (text or "").strip()
    # Split CamelCase words (e.g. BestEffort -> Best Effort) before lowercasing
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned).lower()
    # Replace dots not between digits, plus underscores, hyphens, slashes
    cleaned = re.sub(r"(?<!\d)\.(?!\d)", " ", cleaned)
    cleaned = re.sub(r"[_\-/]", " ", cleaned)
    cleaned = re.sub(r"[^\w\s.%·]", " ", cleaned)
    tokens = [tok for tok in cleaned.split() if tok not in _STOPWORDS]
    return " ".join(tokens)


def _extract_numbers(text: str) -> Set[str]:
    """Extract normalized numeric tokens from an answer string."""
    raw_nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    normalized: Set[str] = set()
    for num in raw_nums:
        try:
            val = float(num)
            normalized.add(f"{val:g}")
        except ValueError:
            normalized.add(num)
    return normalized


def answers_equivalent(ans_a: str, ans_b: str) -> bool:
    """Determine whether two canonical answers represent the same factual proposition."""
    norm_a = normalize_answer(ans_a)
    norm_b = normalize_answer(ans_b)

    if not norm_a or not norm_b:
        return norm_a == norm_b

    if norm_a == norm_b:
        return True

    if (norm_a == "unanswerable") != (norm_b == "unanswerable"):
        return False

    nums_a = _extract_numbers(norm_a)
    nums_b = _extract_numbers(norm_b)
    if nums_a and nums_b and not (nums_a & nums_b):
        return False

    if len(norm_a) > 6 and len(norm_b) > 6 and (norm_a in norm_b or norm_b in norm_a):
        return True

    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if not tokens_a or not tokens_b:
        return False

    intersection = len(tokens_a & tokens_b)
    dice_score = (2.0 * intersection) / (len(tokens_a) + len(tokens_b))
    return dice_score >= 0.55


def compute_decision_threshold(
    u_tp: float = 10.0,
    u_fp: float = -50.0,
    u_fn: float = -2.0,
    u_tn: float = 0.0,
) -> float:
    """Compute the critical probability threshold p* where EU(Accept) == EU(Abstain).

    Formula:
        p* = (u_TN - u_FP) / ((u_TP - u_FN) + (u_TN - u_FP))
    """
    numerator = u_tn - u_fp
    denominator = (u_tp - u_fn) + (u_tn - u_fp)
    if denominator <= 0.0:
        raise ValueError(
            f"Invalid utility matrix: denominator {denominator} must be positive."
        )
    p_star = numerator / denominator
    return float(max(0.0, min(1.0, p_star)))


def compute_expected_utilities(
    p: float,
    u_tp: float = 10.0,
    u_fp: float = -50.0,
    u_fn: float = -2.0,
    u_tn: float = 0.0,
) -> Tuple[float, float]:
    """Compute (EU(a_accept | p), EU(a_abstain | p)) under the asymmetric utility matrix."""
    clamped_p = float(max(0.0, min(1.0, p)))
    eu_accept = clamped_p * u_tp + (1.0 - clamped_p) * u_fp
    eu_abstain = clamped_p * u_fn + (1.0 - clamped_p) * u_tn
    return float(eu_accept), float(eu_abstain)


def select_optimal_action(eu_accept: float, eu_abstain: float) -> DecisionAction:
    """Select the action that maximizes Expected Utility."""
    if eu_accept >= eu_abstain:
        return DecisionAction.ACCEPT
    return DecisionAction.ABSTAIN


def aggregate_mc_samples(
    samples: List[LLMSampleOutput],
    top_similarity: float,
    retrieval_threshold: float = 0.35,
) -> Tuple[str, float, float]:
    """Aggregate M Monte Carlo samples into (modal_answer, f_MC, p_gated).

    Implements:
        f_MC = (1 / M) * sum_{m=1}^M I(equiv(y_m, y*) and e_m == 1)
        p    = f_MC * I(s_max >= tau_ret)
    """
    if not samples:
        return "UNANSWERABLE", 0.0, 0.0

    total_m = len(samples)
    entailed_samples = [
        s
        for s in samples
        if s.context_entailed and normalize_answer(s.canonical_answer) != "unanswerable"
    ]

    if not entailed_samples:
        return "UNANSWERABLE", 0.0, 0.0

    # Cluster entailed answers by semantic/canonical equivalence
    clusters: List[List[LLMSampleOutput]] = []
    for sample in entailed_samples:
        placed = False
        for cluster in clusters:
            if answers_equivalent(sample.canonical_answer, cluster[0].canonical_answer):
                cluster.append(sample)
                placed = True
                break
        if not placed:
            clusters.append([sample])

    # Sort clusters by size descending, then by average confidence_claim descending
    clusters.sort(
        key=lambda c: (
            len(c),
            sum(item.confidence_claim for item in c) / len(c),
        ),
        reverse=True,
    )
    modal_cluster = clusters[0]
    modal_answer = modal_cluster[0].canonical_answer

    # Count agreement across all M samples where equiv(y_m, y*) and e_m == 1
    agreement_count = sum(
        1
        for s in samples
        if s.context_entailed and answers_equivalent(s.canonical_answer, modal_answer)
    )
    f_mc = float(agreement_count) / float(total_m)
    gate_indicator = 1.0 if top_similarity >= retrieval_threshold else 0.0
    p_gated = f_mc * gate_indicator

    return modal_answer, round(f_mc, 6), round(p_gated, 6)


def parse_llm_json_output(raw_text: str) -> LLMSampleOutput:
    """Parse raw LLM completion string into validated LLMSampleOutput."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        cleaned = cleaned[start_brace : end_brace + 1]

    parsed = json.loads(cleaned)
    return LLMSampleOutput.model_validate(parsed)


class OPMEngine:
    """Optimal Predictor Machine (OPM) wrapping retrieval, MC sampling, and MEU decisioning."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        groq_client: Optional[Groq] = None,
        retriever: Optional[CorpusRetriever] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> None:
        self.config = config or settings
        self.groq_client = (
            groq_client
            if groq_client is not None
            else get_groq_client(self.config.groq_api_key)
        )
        self.retriever = retriever or _default_retriever
        self.prompt_template = prompt_template or load_prompt("rag_opm", "v1")

    def _call_groq_single_sample(
        self, messages: List[Dict[str, str]], sample_idx: int
    ) -> Optional[LLMSampleOutput]:
        """Attempt a single stochastic completion via Groq with model fallback chain."""
        if self.groq_client is None:
            return None

        models_to_try = [
            self.config.groq_model,
            self.config.fallback_model,
            self.config.secondary_fallback_model,
        ]
        # Remove duplicates while preserving order
        unique_models: List[str] = []
        for m in models_to_try:
            if m and m not in unique_models:
                unique_models.append(m)

        for model_name in unique_models:
            try:
                completion = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.config.mc_temperature,
                    max_tokens=450,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or ""
                return parse_llm_json_output(content)
            except Exception as exc:
                logger.warning(
                    "Groq API call failed on model; trying fallback if available",
                    extra={
                        "payload": {
                            "model": model_name,
                            "sample_idx": sample_idx,
                            "error": str(exc),
                        }
                    },
                )
        return None

    def _grounded_extractive_fallback(
        self,
        query: str,
        context_chunks: List[str],
        top_similarity: float,
        sample_idx: int,
    ) -> LLMSampleOutput:
        """Deterministic grounded extractive sampler when running offline without Groq API."""
        query_tokens = set(normalize_answer(query).split())
        if not context_chunks or not query_tokens:
            return LLMSampleOutput(
                canonical_answer="UNANSWERABLE",
                raw_reasoning="No retrieved context chunks available to ground an answer.",
                context_entailed=False,
                confidence_claim=0.0,
            )

        # Split retrieved chunks into candidate factual sentences
        candidate_sentences: List[Tuple[float, str]] = []
        top_chunk_body = re.sub(r"^\[.*?\]\s*", "", context_chunks[0]).strip()
        for chunk in context_chunks:
            body = re.sub(r"^\[.*?\]\s*", "", chunk)
            for sent in re.split(r"(?<=[.!?])\s+", body):
                sent_clean = sent.strip()
                if len(sent_clean) < 15:
                    continue
                sent_tokens = set(normalize_answer(sent_clean).split())
                if not sent_tokens:
                    continue
                overlap = len(query_tokens & sent_tokens) / float(len(query_tokens))
                candidate_sentences.append((overlap, sent_clean))

        candidate_sentences.sort(key=lambda item: item[0], reverse=True)
        best_overlap, best_sentence = (
            candidate_sentences[0] if candidate_sentences else (0.0, "")
        )

        # Combine relevant sentences in the top chunk for comprehensive factual extraction
        matching_sents = [s for ov, s in candidate_sentences[:4] if ov >= 0.12]
        if len(matching_sents) >= 2:
            combined_answer = " ".join(matching_sents)
        elif best_overlap >= 0.25:
            combined_answer = f"{best_sentence} {top_chunk_body}"
        else:
            combined_answer = best_sentence

        combined_tokens = set(normalize_answer(combined_answer).split())
        combined_overlap = (
            len(query_tokens & combined_tokens) / float(len(query_tokens))
            if query_tokens
            else 0.0
        )

        # Deterministic pseudo-random perturbation per (query, sample_idx)
        digest = hashlib.sha256(f"{query}::{sample_idx}".encode("utf-8")).hexdigest()
        u_rand = (int(digest[:8], 16) % 1000) / 1000.0

        # Assess whether context actually entails the query
        if (
            top_similarity >= self.config.retrieval_threshold
            and combined_overlap >= 0.35
        ):
            entailed = (
                True
                if combined_overlap >= 0.45 or top_similarity >= 0.38
                else (u_rand < 0.85)
            )
            if entailed:
                return LLMSampleOutput(
                    canonical_answer=combined_answer,
                    raw_reasoning=(
                        f"Sample #{sample_idx + 1}: Grounded extraction verified directly "
                        f"against retrieved context (token overlap={combined_overlap:.2f}, "
                        f"top_sim={top_similarity:.3f})."
                    ),
                    context_entailed=True,
                    confidence_claim=round(
                        min(0.98, max(0.80, top_similarity + 0.35)), 2
                    ),
                )

        # Low-evidence / unanswerable regime
        return LLMSampleOutput(
            canonical_answer="UNANSWERABLE",
            raw_reasoning=(
                f"Sample #{sample_idx + 1}: Retrieved context does not contain sufficient "
                f"verified evidence for the requested entities (overlap={best_overlap:.2f}, "
                f"top_sim={top_similarity:.3f})."
            ),
            context_entailed=False,
            confidence_claim=0.0,
        )

    def sample_completions(
        self,
        query: str,
        context_chunks: List[str],
        top_similarity: float,
        num_samples: Optional[int] = None,
    ) -> List[LLMSampleOutput]:
        """Run M Monte Carlo completions and return structured LLMSampleOutput objects."""
        m_count = max(
            1, num_samples if num_samples is not None else self.config.mc_samples
        )
        formatted_context = "\n\n".join(
            f"Chunk {idx + 1}: {chunk}" for idx, chunk in enumerate(context_chunks)
        )
        messages = self.prompt_template.render_messages(
            context=formatted_context, query=query
        )

        samples: List[LLMSampleOutput] = []
        for idx in range(m_count):
            sample = self._call_groq_single_sample(messages=messages, sample_idx=idx)
            if sample is None:
                sample = self._grounded_extractive_fallback(
                    query=query,
                    context_chunks=context_chunks,
                    top_similarity=top_similarity,
                    sample_idx=idx,
                )
            samples.append(sample)

        return samples

    def evaluate_query(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 3,
        num_samples: Optional[int] = None,
        u_tp: Optional[float] = None,
        u_fp: Optional[float] = None,
        u_fn: Optional[float] = None,
        u_tn: Optional[float] = None,
        retrieval_threshold: Optional[float] = None,
    ) -> MEUResult:
        """Execute full OPM pipeline: Retrieve -> Monte Carlo Sample -> Gate -> MEU Decision."""
        resolved_u_tp = u_tp if u_tp is not None else self.config.utility_tp
        resolved_u_fp = u_fp if u_fp is not None else self.config.utility_fp
        resolved_u_fn = u_fn if u_fn is not None else self.config.utility_fn
        resolved_u_tn = u_tn if u_tn is not None else self.config.utility_tn
        resolved_tau = (
            retrieval_threshold
            if retrieval_threshold is not None
            else self.config.retrieval_threshold
        )

        retrieved_pairs = self.retriever.retrieve(
            query=query, top_k=top_k, domain=domain
        )
        context_chunks = [pair[0] for pair in retrieved_pairs]
        top_similarity = float(retrieved_pairs[0][1]) if retrieved_pairs else 0.0

        samples = self.sample_completions(
            query=query,
            context_chunks=context_chunks,
            top_similarity=top_similarity,
            num_samples=num_samples,
        )

        modal_answer, f_mc, p_correct = aggregate_mc_samples(
            samples=samples,
            top_similarity=top_similarity,
            retrieval_threshold=resolved_tau,
        )

        p_star = compute_decision_threshold(
            u_tp=resolved_u_tp,
            u_fp=resolved_u_fp,
            u_fn=resolved_u_fn,
            u_tn=resolved_u_tn,
        )
        eu_accept, eu_abstain = compute_expected_utilities(
            p=p_correct,
            u_tp=resolved_u_tp,
            u_fp=resolved_u_fp,
            u_fn=resolved_u_fn,
            u_tn=resolved_u_tn,
        )
        optimal_action = select_optimal_action(
            eu_accept=eu_accept, eu_abstain=eu_abstain
        )

        final_response = (
            modal_answer
            if optimal_action == DecisionAction.ACCEPT
            else ABSTAIN_FALLBACK_MESSAGE
        )

        result = MEUResult(
            p_correct=round(p_correct, 4),
            eu_accept=round(eu_accept, 4),
            eu_abstain=round(eu_abstain, 4),
            optimal_action=optimal_action,
            decision_threshold=round(p_star, 4),
            modal_answer=modal_answer,
            final_response=final_response,
            context_chunks=context_chunks,
            top_similarity=round(top_similarity, 4),
            mc_agreement=round(f_mc, 4),
            samples=samples,
        )

        logger.info(
            "OPM decision computed",
            extra={
                "payload": {
                    "query": query,
                    "top_similarity": result.top_similarity,
                    "mc_agreement": result.mc_agreement,
                    "p_correct": result.p_correct,
                    "eu_accept": result.eu_accept,
                    "eu_abstain": result.eu_abstain,
                    "optimal_action": result.optimal_action.value,
                }
            },
        )
        return result
