"""Unit tests for 5-bin probability calibration, ECE, Brier Score, and Utility Gain."""

from __future__ import annotations

import math

import pytest

from src.eval import (
    compute_brier_score,
    compute_calibration_bins,
    compute_realized_utilities,
)
from src.schema import DecisionAction


def test_brier_score_extremes_and_midpoint() -> None:
    """Verify Brier Score calculation for perfect, worst, and 0.5 predictions."""
    # Perfect calibration & accuracy: BS = 0.0
    assert math.isclose(
        compute_brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]), 0.0, abs_tol=1e-9
    )

    # Completely wrong confident predictions: BS = 1.0
    assert math.isclose(
        compute_brier_score([0.0, 1.0], [1, 0]), 1.0, abs_tol=1e-9
    )

    # Uniform 0.5 predictions: (0.5 - 1)^2 = 0.25
    assert math.isclose(
        compute_brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]), 0.25, abs_tol=1e-9
    )


def test_five_bin_calibration_and_ece() -> None:
    """Verify exact 5-bin partitioning and ECE formula."""
    # 5 predictions, one in each bin:
    # Bin 1 [0.0, 0.2]: p=0.1, y=0 -> acc=0.0, conf=0.1, err=0.1
    # Bin 2 (0.2, 0.4]: p=0.3, y=0 -> acc=0.0, conf=0.3, err=0.3
    # Bin 3 (0.4, 0.6]: p=0.5, y=1 -> acc=1.0, conf=0.5, err=0.5
    # Bin 4 (0.6, 0.8]: p=0.7, y=1 -> acc=1.0, conf=0.7, err=0.3
    # Bin 5 (0.8, 1.0]: p=0.9, y=1 -> acc=1.0, conf=0.9, err=0.1
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    outcomes = [0, 0, 1, 1, 1]

    bins, ece = compute_calibration_bins(probs, outcomes)
    assert len(bins) == 5
    assert [b.bin_range for b in bins] == [
        "[0.0, 0.2]",
        "(0.2, 0.4]",
        "(0.4, 0.6]",
        "(0.6, 0.8]",
        "(0.8, 1.0]",
    ]
    assert all(b.count == 1 for b in bins)

    expected_ece = (0.1 + 0.3 + 0.5 + 0.3 + 0.1) / 5.0
    assert math.isclose(ece, expected_ece, abs_tol=1e-6)


def test_realized_utility_comparison_opm_vs_naive() -> None:
    """Verify OPM avoids the -50 hallucination penalty that Naive RAG suffers on theta=0."""
    # On an answerable grounded query (theta = 1) where OPM accepts:
    opm_tp, naive_tp = compute_realized_utilities(DecisionAction.ACCEPT, theta=1)
    assert opm_tp == 10.0
    assert naive_tp == 10.0

    # On an unanswerable query (theta = 0) where OPM abstains and Naive accepts:
    opm_tn, naive_fp = compute_realized_utilities(DecisionAction.ABSTAIN, theta=0)
    assert opm_tn == 0.0
    assert naive_fp == -50.0
    assert (opm_tn - naive_fp) == 50.0
