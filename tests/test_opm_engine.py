"""Unit tests for the OPM Decision Engine, Monte Carlo aggregation, and MEU math."""

from __future__ import annotations

import math

import pytest

from src.config import ABSTAIN_FALLBACK_MESSAGE
from src.opm_engine import (
    OPMEngine,
    aggregate_mc_samples,
    answers_equivalent,
    compute_decision_threshold,
    compute_expected_utilities,
    select_optimal_action,
)
from src.schema import DecisionAction, LLMSampleOutput


def test_critical_decision_threshold_derivation() -> None:
    """Verify p* = 50 / 62 ≈ 0.8064516 for default (+10, -50, -2, 0) utilities."""
    p_star = compute_decision_threshold(u_tp=10.0, u_fp=-50.0, u_fn=-2.0, u_tn=0.0)
    expected = 50.0 / 62.0
    assert math.isclose(p_star, expected, rel_tol=1e-9)

    # At p = p*, EU(Accept) must equal EU(Abstain)
    eu_acc, eu_abs = compute_expected_utilities(
        p=p_star, u_tp=10.0, u_fp=-50.0, u_fn=-2.0, u_tn=0.0
    )
    assert math.isclose(eu_acc, eu_abs, abs_tol=1e-9)


def test_expected_utilities_and_action_selection() -> None:
    """Verify MEU selects Abstain at p=0.80 (< 0.8065) and Accept at p=1.0."""
    # At p = 0.80 (4/5 MC agreement):
    # EU(Accept) = 0.8 * 10 + 0.2 * (-50) = 8 - 10 = -2.0
    # EU(Abstain) = 0.8 * (-2) + 0.2 * 0 = -1.6
    # Since -1.6 > -2.0, Abstain is strictly optimal!
    eu_acc_80, eu_abs_80 = compute_expected_utilities(
        p=0.80, u_tp=10.0, u_fp=-50.0, u_fn=-2.0, u_tn=0.0
    )
    assert math.isclose(eu_acc_80, -2.0, abs_tol=1e-9)
    assert math.isclose(eu_abs_80, -1.6, abs_tol=1e-9)
    assert select_optimal_action(eu_acc_80, eu_abs_80) == DecisionAction.ABSTAIN

    # At p = 1.0 (5/5 MC agreement):
    # EU(Accept) = 10.0, EU(Abstain) = -2.0 -> Accept
    eu_acc_100, eu_abs_100 = compute_expected_utilities(
        p=1.0, u_tp=10.0, u_fp=-50.0, u_fn=-2.0, u_tn=0.0
    )
    assert math.isclose(eu_acc_100, 10.0, abs_tol=1e-9)
    assert math.isclose(eu_abs_100, -2.0, abs_tol=1e-9)
    assert select_optimal_action(eu_acc_100, eu_abs_100) == DecisionAction.ACCEPT


def test_monte_carlo_aggregation_and_retrieval_gate() -> None:
    """Verify f_MC calculation and retrieval threshold gating I(s_max >= tau_ret)."""
    samples = [
        LLMSampleOutput(
            canonical_answer="12.0V nominal operating voltage and 4.1 N·m stall torque",
            raw_reasoning="Entailed by section 1.",
            context_entailed=True,
            confidence_claim=0.95,
        )
        for _ in range(4)
    ] + [
        LLMSampleOutput(
            canonical_answer="UNANSWERABLE",
            raw_reasoning="Uncertain.",
            context_entailed=False,
            confidence_claim=0.0,
        )
    ]

    # Case 1: s_max = 0.62 >= 0.35 (Gate passes)
    modal, f_mc, p_gated = aggregate_mc_samples(
        samples=samples, top_similarity=0.62, retrieval_threshold=0.35
    )
    assert "12.0V" in modal
    assert math.isclose(f_mc, 0.80, abs_tol=1e-6)
    assert math.isclose(p_gated, 0.80, abs_tol=1e-6)

    # Case 2: s_max = 0.21 < 0.35 (Gate blocks belief to 0.0)
    _, f_mc_blocked, p_blocked = aggregate_mc_samples(
        samples=samples, top_similarity=0.21, retrieval_threshold=0.35
    )
    assert math.isclose(f_mc_blocked, 0.80, abs_tol=1e-6)
    assert math.isclose(p_blocked, 0.0, abs_tol=1e-6)


def test_answers_equivalent_normalization() -> None:
    """Verify canonical equivalence handles minor phrasing differences."""
    a = "The nominal operating voltage is 12.0V and maximum stall torque is 4.1 N·m."
    b = "12.0V nominal operating voltage and 4.1 N·m maximum stall torque"
    c = "24.0V nominal operating voltage and 8.5 N·m maximum stall torque"
    assert answers_equivalent(a, b) is True
    assert answers_equivalent(a, c) is False


def test_opm_engine_end_to_end_accept_and_abstain() -> None:
    """Verify OPMEngine accepts grounded corpus queries and abstains on unanswerable queries."""
    engine = OPMEngine(groq_client=None)

    # Answerable query present in robotics_hardware.md
    res_accept = engine.evaluate_query(
        query="What is the nominal operating voltage and maximum stall torque of the Dynamixel XM430-W350-T servo?"
    )
    assert res_accept.top_similarity >= 0.35
    assert res_accept.p_correct == 1.0
    assert res_accept.optimal_action == DecisionAction.ACCEPT
    assert "12.0" in res_accept.final_response
    assert "4.1" in res_accept.final_response

    # Adversarial / unanswerable query absent from corpus
    res_abstain = engine.evaluate_query(
        query="What is the pediatric intravenous amiodarone loading dosage for refractory supraventricular tachycardia?"
    )
    assert res_abstain.p_correct == 0.0
    assert res_abstain.optimal_action == DecisionAction.ABSTAIN
    assert res_abstain.final_response == ABSTAIN_FALLBACK_MESSAGE
