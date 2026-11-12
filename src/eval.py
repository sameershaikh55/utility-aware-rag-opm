"""Benchmark evaluation runner, 5-bin empirical calibration, ECE, and Brier score."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tabulate import tabulate

from src.config import AppConfig, settings
from src.io_utils import read_jsonl, write_json_atomic
from src.logger import get_logger
from src.opm_engine import (
    OPMEngine,
    _extract_numbers,
    answers_equivalent,
    normalize_answer,
)
from src.schema import (
    CalibrationBin,
    DecisionAction,
    EvalQueryRecord,
    EvaluationReport,
)

logger = get_logger(__name__)

BIN_SPECIFICATIONS: List[Tuple[str, float, float, bool]] = [
    ("[0.0, 0.2]", 0.0, 0.2, True),
    ("(0.2, 0.4]", 0.2, 0.4, False),
    ("(0.4, 0.6]", 0.4, 0.6, False),
    ("(0.6, 0.8]", 0.6, 0.8, False),
    ("(0.8, 1.0]", 0.8, 1.0, False),
]


def compute_brier_score(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> float:
    """Compute the Brier Score: BS = (1/N) * sum_{i=1}^N (p_i - y_i)^2."""
    if len(probabilities) != len(outcomes):
        raise ValueError("Length of probabilities and outcomes must match.")
    n_total = len(probabilities)
    if n_total == 0:
        return 0.0

    sq_error_sum = sum(
        (float(p) - float(y)) ** 2 for p, y in zip(probabilities, outcomes)
    )
    return round(sq_error_sum / float(n_total), 6)


def compute_calibration_bins(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> Tuple[List[CalibrationBin], float]:
    """Partition predictions into 5 confidence bins and compute Expected Calibration Error (ECE).

    Bins:
        [0.0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    ECE:
        sum_{b=1}^5 (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("Length of probabilities and outcomes must match.")

    n_total = len(probabilities)
    bins: List[CalibrationBin] = []
    ece_accumulator = 0.0

    for label, low, high, include_low in BIN_SPECIFICATIONS:
        bin_probs: List[float] = []
        bin_outcomes: List[int] = []

        for p_raw, y_raw in zip(probabilities, outcomes):
            p_val = float(max(0.0, min(1.0, p_raw)))
            y_val = int(bool(y_raw))
            in_bin = (low <= p_val <= high) if include_low else (low < p_val <= high)
            if in_bin:
                bin_probs.append(p_val)
                bin_outcomes.append(y_val)

        count = len(bin_probs)
        if count > 0:
            emp_acc = sum(bin_outcomes) / float(count)
            mean_conf = sum(bin_probs) / float(count)
            cal_err = abs(emp_acc - mean_conf)
            if n_total > 0:
                ece_accumulator += (float(count) / float(n_total)) * cal_err
        else:
            emp_acc = 0.0
            mean_conf = 0.0
            cal_err = 0.0

        bins.append(
            CalibrationBin(
                bin_range=label,
                count=count,
                empirical_accuracy=round(emp_acc, 4),
                mean_confidence=round(mean_conf, 4),
                calibration_error=round(cal_err, 4),
            )
        )

    return bins, round(ece_accumulator, 6)


def evaluate_ground_truth_entailment(
    modal_answer: str, ground_truth_answer: str, is_answerable: bool
) -> int:
    """Determine true state of nature theta in {1, 0} for candidate answer y*.

    - If the query is unanswerable (is_answerable=False), no factual proposition
      is grounded in the corpus, so theta = 0 (accepting incurs FP penalty).
    - If the query is answerable (is_answerable=True), theta = 1 if and only if
      modal_answer is non-empty, not 'UNANSWERABLE', and matches key facts/numbers
      in ground_truth_answer.
    """
    if not is_answerable:
        return 0

    norm_modal = normalize_answer(modal_answer)
    norm_gt = normalize_answer(ground_truth_answer)
    if not norm_modal or norm_modal == "unanswerable":
        return 0

    if answers_equivalent(modal_answer, ground_truth_answer):
        return 1

    gt_tokens = set(norm_gt.split())
    modal_tokens = set(norm_modal.split())
    if not gt_tokens:
        return 0

    gt_numbers = _extract_numbers(ground_truth_answer)
    modal_numbers = _extract_numbers(modal_answer)
    if gt_numbers and not (gt_numbers & modal_numbers):
        return 0

    recall = len(gt_tokens & modal_tokens) / float(len(gt_tokens))
    return 1 if recall >= 0.35 else 0


def compute_realized_utilities(
    action: DecisionAction,
    theta: int,
    u_tp: float = 10.0,
    u_fp: float = -50.0,
    u_fn: float = -2.0,
    u_tn: float = 0.0,
) -> Tuple[float, float]:
    """Return (opm_realized_utility, naive_realized_utility) for a single query.

    Naive RAG always selects Accept regardless of evidence.
    """
    naive_utility = u_tp if theta == 1 else u_fp
    if action == DecisionAction.ACCEPT:
        opm_utility = u_tp if theta == 1 else u_fp
    else:
        opm_utility = u_fn if theta == 1 else u_tn
    return float(opm_utility), float(naive_utility)


def load_golden_benchmark(
    path: Optional[Path] = None, limit: Optional[int] = None
) -> List[EvalQueryRecord]:
    """Load and validate EvalQueryRecord items from data/golden_eval.jsonl."""
    target = path or settings.golden_eval_path
    raw_rows = read_jsonl(target)
    records = [EvalQueryRecord.model_validate(row) for row in raw_rows]
    if limit is not None and limit > 0:
        return records[:limit]
    return records


def run_evaluation(
    limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    engine: Optional[OPMEngine] = None,
    config: Optional[AppConfig] = None,
) -> EvaluationReport:
    """Run the golden benchmark through OPMEngine and compute calibration & utility metrics."""
    cfg = config or settings
    opm = engine or OPMEngine(config=cfg)
    records = load_golden_benchmark(cfg.golden_eval_path, limit=limit)

    probabilities: List[float] = []
    outcomes: List[int] = []
    opm_total_utility = 0.0
    naive_total_utility = 0.0
    query_details: List[Dict[str, Any]] = []

    for record in records:
        meu_res = opm.evaluate_query(query=record.query)
        theta = evaluate_ground_truth_entailment(
            modal_answer=meu_res.modal_answer,
            ground_truth_answer=record.ground_truth_answer,
            is_answerable=record.is_answerable,
        )
        opm_u, naive_u = compute_realized_utilities(
            action=meu_res.optimal_action,
            theta=theta,
            u_tp=cfg.utility_tp,
            u_fp=cfg.utility_fp,
            u_fn=cfg.utility_fn,
            u_tn=cfg.utility_tn,
        )

        probabilities.append(meu_res.p_correct)
        outcomes.append(theta)
        opm_total_utility += opm_u
        naive_total_utility += naive_u

        query_details.append(
            {
                "id": record.id,
                "domain": record.domain,
                "query": record.query,
                "is_answerable": record.is_answerable,
                "ground_truth_answer": record.ground_truth_answer,
                "modal_answer": meu_res.modal_answer,
                "top_similarity": meu_res.top_similarity,
                "mc_agreement": meu_res.mc_agreement,
                "p_correct": meu_res.p_correct,
                "theta": theta,
                "eu_accept": meu_res.eu_accept,
                "eu_abstain": meu_res.eu_abstain,
                "optimal_action": meu_res.optimal_action.value,
                "opm_utility": opm_u,
                "naive_utility": naive_u,
            }
        )

    brier = compute_brier_score(probabilities, outcomes)
    bins, ece = compute_calibration_bins(probabilities, outcomes)
    utility_gain = opm_total_utility - naive_total_utility

    report = EvaluationReport(
        total_queries=len(records),
        brier_score=round(brier, 4),
        ece=round(ece, 4),
        bins=bins,
        opm_total_utility=round(opm_total_utility, 2),
        naive_total_utility=round(naive_total_utility, 2),
        utility_gain=round(utility_gain, 2),
        query_details=query_details,
    )

    if output_path is not None:
        write_json_atomic(output_path, report.model_dump(mode="json"))
        logger.info(
            "Saved evaluation report to disk",
            extra={"payload": {"output_path": str(output_path)}},
        )

    return report


def print_evaluation_summary(report: EvaluationReport) -> None:
    """Print formatted tables of per-query results, calibration bins, and aggregate KPIs."""
    print("\n" + "=" * 88)
    print("  OPTIMAL PREDICTOR MACHINE (OPM) -- EMPIRICAL CALIBRATION & UTILITY REPORT")
    print("=" * 88)

    if report.query_details:
        query_rows = [
            [
                d["id"],
                d["domain"],
                "Yes" if d["is_answerable"] else "No",
                f"{d['top_similarity']:.3f}",
                f"{d['p_correct']:.2f}",
                d[" theta"] if " theta" in d else d["theta"],
                d["optimal_action"],
                f"{d['opm_utility']:+.1f}",
                f"{d['naive_utility']:+.1f}",
            ]
            for d in report.query_details
        ]
        print("\n[1] Per-Query Decision & Utility Breakdown:")
        print(
            tabulate(
                query_rows,
                headers=[
                    "ID",
                    "Domain",
                    "Answerable",
                    "s_max",
                    "p(theta_1)",
                    "theta_1",
                    "Action",
                    "OPM Util",
                    "Naive Util",
                ],
                tablefmt="github",
            )
        )

    bin_rows = [
        [
            b.bin_range,
            b.count,
            f"{b.empirical_accuracy:.4f}",
            f"{b.mean_confidence:.4f}",
            f"{b.calibration_error:.4f}",
        ]
        for b in report.bins
    ]
    print("\n[2] 5-Bin Empirical Probability Calibration Table:")
    print(
        tabulate(
            bin_rows,
            headers=[
                "Confidence Bin",
                "Count |B_b|",
                "Empirical Acc",
                "Mean Conf",
                "|Acc - Conf|",
            ],
            tablefmt="github",
        )
    )

    kpi_rows = [
        ["Total Evaluated Queries (N)", report.total_queries],
        ["Brier Score (BS)", f"{report.brier_score:.4f}"],
        ["Expected Calibration Error (ECE)", f"{report.ece:.4f}"],
        ["OPM Cumulative Realized Utility", f"{report.opm_total_utility:+.2f}"],
        ["Naive RAG Cumulative Realized Utility", f"{report.naive_total_utility:+.2f}"],
        ["Net Decision-Theoretic Utility Gain", f"{report.utility_gain:+.2f}"],
    ]
    print("\n[3] Aggregate Telemetry & Examiner KPIs:")
    print(tabulate(kpi_rows, headers=["Metric", "Value"], tablefmt="github"))
    print("=" * 88 + "\n")


def main() -> None:
    """CLI entrypoint for `src.eval`."""
    parser = argparse.ArgumentParser(
        description="Run OPM RAG empirical calibration and utility benchmark."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of benchmark queries to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_eval.json",
        help="Output filepath for the JSON EvaluationReport.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    report = run_evaluation(limit=args.limit, output_path=out_path)
    print_evaluation_summary(report)


if __name__ == "__main__":
    main()
