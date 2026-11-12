"""Streamlit interactive dashboard for the Utility-Aware RAG Evaluator (OPM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.config import ABSTAIN_FALLBACK_MESSAGE, settings
from src.eval import load_golden_benchmark, run_evaluation
from src.io_utils import read_json
from src.opm_engine import (
    OPMEngine,
    compute_decision_threshold,
)
from src.schema import DecisionAction, EvaluationReport


def _load_or_run_report() -> Optional[EvaluationReport]:
    """Load an existing evaluation report from results/ if available."""
    baseline_path = settings.results_dir / "baseline_eval.json"
    smoke_path = settings.results_dir / "smoke_eval.json"

    for candidate in (baseline_path, smoke_path):
        if candidate.exists():
            try:
                raw = read_json(candidate)
                return EvaluationReport.model_validate(raw)
            except Exception:
                continue
    return None


def main() -> None:
    """Render the Streamlit OPM RAG Evaluator application."""
    st.set_page_config(
        page_title="Utility-Aware RAG Evaluator (OPM)",
        layout="wide",
    )

    st.title("Utility-Aware RAG Evaluator — Optimal Predictor Machine (OPM)")
    st.caption(
        "Bayesian Decision Theory, "
        "Grounded Monte Carlo Probability Elicitation & Maximum Expected Utility (MEU)"
    )

    # -------------------------------------------------------------------------
    # Sidebar: Asymmetric Utility Matrix & OPM Hyperparameters
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Decision Matrix & Gate")
        st.markdown("Configure asymmetric utility payoffs $U(a, \\theta)$:")

        u_tp = st.slider(
            "True Positive Utility (u_TP: Accept, θ₁)",
            min_value=1.0,
            max_value=50.0,
            value=float(settings.utility_tp),
            step=1.0,
        )
        u_fp = st.slider(
            "False Positive Utility (u_FP: Accept, θ₀)",
            min_value=-100.0,
            max_value=-1.0,
            value=float(settings.utility_fp),
            step=1.0,
        )
        u_fn = st.slider(
            "False Negative Utility (u_FN: Abstain, θ₁)",
            min_value=-25.0,
            max_value=0.0,
            value=float(settings.utility_fn),
            step=0.5,
        )
        u_tn = st.slider(
            "True Negative Utility (u_TN: Abstain, θ₀)",
            min_value=-5.0,
            max_value=10.0,
            value=float(settings.utility_tn),
            step=0.5,
        )

        p_star = compute_decision_threshold(u_tp=u_tp, u_fp=u_fp, u_fn=u_fn, u_tn=u_tn)
        st.info(f"**Critical Decision Threshold ($p^*$):** `{p_star:.4f}`")

        st.divider()
        st.subheader("Sampling & Retrieval")
        tau_ret = st.slider(
            "Retrieval Cosine Gate (τ_ret)",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.retrieval_threshold),
            step=0.05,
        )
        mc_samples = st.slider(
            "Monte Carlo Samples (M)",
            min_value=1,
            max_value=10,
            value=int(settings.mc_samples),
            step=1,
        )

        mode_label = (
            f"Live Groq API (`{settings.groq_model}`)"
            if settings.has_valid_api_key
            else "Offline Grounded Extractive Simulation"
        )
        st.caption(f"**Execution Mode:** {mode_label}")

    tab_sandbox, tab_telemetry = st.tabs(
        ["Live OPM Sandbox", "Calibration & Benchmark Telemetry"]
    )

    # -------------------------------------------------------------------------
    # Tab 1: Live OPM Sandbox
    # -------------------------------------------------------------------------
    with tab_sandbox:
        st.subheader("Interactive Decision-Theoretic RAG Query Sandbox")

        benchmark_records = load_golden_benchmark()
        preset_options = ["Custom User Query"] + [
            f"[{r.id} | {r.domain}] {r.query}" for r in benchmark_records
        ]
        selected_preset = st.selectbox(
            "Select a Preset Benchmark Query or Custom Input:", preset_options
        )

        default_query = (
            "What is the nominal operating voltage and maximum stall torque of the Dynamixel XM430-W350-T servo?"
            if selected_preset == "Custom User Query"
            else selected_preset.split("] ", 1)[1]
        )

        col_q, col_dom = st.columns([3, 1])
        with col_q:
            user_query = st.text_input("Incoming User Query (q):", value=default_query)
        with col_dom:
            domain_choice = st.selectbox(
                "Domain Corpus Filter:",
                options=[
                    "All Domains",
                    "robotics_hardware",
                    "university_policy",
                    "medical_triage",
                ],
            )

        selected_domain = None if domain_choice == "All Domains" else domain_choice

        if st.button("Execute OPM Decision Engine", type="primary"):
            engine = OPMEngine()
            with st.spinner(
                "Running dense retrieval, Monte Carlo sampling, and MEU optimization..."
            ):
                result = engine.evaluate_query(
                    query=user_query,
                    domain=selected_domain,
                    top_k=3,
                    num_samples=mc_samples,
                    u_tp=u_tp,
                    u_fp=u_fp,
                    u_fn=u_fn,
                    u_tn=u_tn,
                    retrieval_threshold=tau_ret,
                )

            # Verdict Banner
            if result.optimal_action == DecisionAction.ACCEPT:
                st.success(
                    f"### ✅ OPTIMAL ACTION: ACCEPT ($EU = {result.eu_accept:+.2f} \\ge {result.eu_abstain:+.2f}$)\n"
                    f"**Verified Response ($y^*$):** {result.final_response}"
                )
            else:
                st.error(
                    f"### OPTIMAL ACTION: ABSTAIN ($EU = {result.eu_abstain:+.2f} > {result.eu_accept:+.2f}$)\n"
                    f"**Fallback Intercept:** {ABSTAIN_FALLBACK_MESSAGE}\n\n"
                    f"*(Suppressed Candidate Answer: `{result.modal_answer}`)*"
                )

            # Key Epistemic & Utility Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Top Cosine Similarity (s_max)", f"{result.top_similarity:.4f}")
            m2.metric("MC Agreement (f_MC)", f"{result.mc_agreement:.2%}")
            m3.metric("Gated Belief p(θ₁|q,C)", f"{result.p_correct:.4f}")
            m4.metric("EU(Accept | p)", f"{result.eu_accept:+.2f}")
            m5.metric("EU(Abstain | p)", f"{result.eu_abstain:+.2f}")

            # Retrieved Context Chunks
            st.markdown("#### Top-k Retrieved Context Chunks ($C$)")
            gate_passed = result.top_similarity >= tau_ret
            st.caption(
                f"Retrieval Gate $\\mathbb{{I}}(s_{{\\max}} \\ge \\tau_{{\\text{{ret}}}})$: "
                f"**{'PASSED (1.0)' if gate_passed else 'BLOCKED (0.0)'}** "
                f"($s_{{\\max}} = {result.top_similarity:.4f}$ vs $\\tau_{{\\text{{ret}}}} = {tau_ret:.2f}$)"
            )
            for idx, chunk_text in enumerate(result.context_chunks, start=1):
                st.code(f"Chunk #{idx}: {chunk_text}", language="markdown")

            # Monte Carlo Samples Table
            st.markdown(
                f"#### Stochastic Monte Carlo Completions ($M = {len(result.samples)}$)"
            )
            sample_rows: List[Dict[str, Any]] = []
            for idx, s in enumerate(result.samples, start=1):
                sample_rows.append(
                    {
                        "Sample #": idx,
                        "Entailed (e_m)": "✅ True"
                        if s.context_entailed
                        else "❌ False",
                        "Confidence Claim": f"{s.confidence_claim:.2f}",
                        "Canonical Answer": s.canonical_answer,
                        "Reasoning": s.raw_reasoning,
                    }
                )
            st.dataframe(sample_rows, use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 2: Calibration & Benchmark Telemetry
    # -------------------------------------------------------------------------
    with tab_telemetry:
        st.subheader("Empirical Probability Calibration & Cumulative Utility Telemetry")

        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            run_now = st.button("Run Full 18-Query Benchmark Now")

        if run_now:
            with st.spinner("Evaluating all 18 golden benchmark queries..."):
                report = run_evaluation(
                    limit=None,
                    output_path=settings.results_dir / "baseline_eval.json",
                )
        else:
            report = _load_or_run_report()

        if report is None:
            st.warning(
                "No saved evaluation report found in `results/`. Click **Run Full 18-Query Benchmark Now** "
                "or run `make eval` / `make eval-smoke` from the CLI."
            )
        else:
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Queries (N)", report.total_queries)
            k2.metric("Brier Score (BS)", f"{report.brier_score:.4f}")
            k3.metric("Expected Cal. Error (ECE)", f"{report.ece:.4f}")
            k4.metric("OPM Total Utility", f"{report.opm_total_utility:+.1f}")
            k5.metric("Naive RAG Utility", f"{report.naive_total_utility:+.1f}")
            k6.metric("Net Utility Gain", f"{report.utility_gain:+.1f}")

            st.markdown("#### 5-Bin Confidence Calibration Breakdown")
            bin_table = [
                {
                    "Confidence Bin": b.bin_range,
                    "Query Count |B_b|": b.count,
                    "Empirical Accuracy acc(B_b)": b.empirical_accuracy,
                    "Mean Confidence conf(B_b)": b.mean_confidence,
                    "Calibration Gap |acc - conf|": b.calibration_error,
                }
                for b in report.bins
            ]
            st.dataframe(bin_table, use_container_width=True)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown(
                    "##### Reliability Diagram (Empirical Accuracy vs. Mean Confidence)"
                )
                chart_data = {
                    "Bin": [b.bin_range for b in report.bins],
                    "Empirical Accuracy": [b.empirical_accuracy for b in report.bins],
                    "Mean Confidence": [b.mean_confidence for b in report.bins],
                }
                st.bar_chart(
                    data=chart_data,
                    x="Bin",
                    y=["Empirical Accuracy", "Mean Confidence"],
                )

            with col_chart2:
                st.markdown("##### Realized Utility Comparison (OPM RAG vs. Naive RAG)")
                util_data = {
                    "System": ["OPM Decision-Theoretic RAG", "Naive Unconstrained RAG"],
                    "Cumulative Realized Utility": [
                        report.opm_total_utility,
                        report.naive_total_utility,
                    ],
                }
                st.bar_chart(
                    data=util_data,
                    x="System",
                    y="Cumulative Realized Utility",
                )

            if report.query_details:
                st.markdown("#### Per-Query Benchmark Telemetry")
                st.dataframe(report.query_details, use_container_width=True)


if __name__ == "__main__":
    main()
