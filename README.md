# Utility-Aware RAG Evaluator (Optimal Predictor Machine - OPM)

**Architecture:** Decision-Theoretic Retrieval-Augmented Generation with Grounded Monte Carlo Probability Elicitation & Maximum Expected Utility (MEU)

---

## 1. Theoretical Foundation

Standard Retrieval-Augmented Generation (RAG) pipelines behave as unconstrained text generators that frequently hallucinate when retrieved context is incomplete, ambiguous, or absent. This repository wraps the generative model inside an **Optimal Predictor Machine (OPM)** governed by Bayesian decision theory.

### 1.1 Decision Space & States of Nature
- **Incoming Query & Context:** For a query $q$ and top-$k$ retrieved chunks $C = \{c_1, \dots, c_k\}$, let $s_{\max} = \max_i \text{sim}(q, c_i)$ denote the top cosine similarity score.
- **States of Nature ($\Theta = \{\theta_1, \theta_0\}$):**
  - $\theta_1$: The modal candidate answer $y^*$ is factually entailed by $C$ and correct.
  - $\theta_0$: The modal candidate answer $y^*$ is ungrounded, fabricated, or hallucinated.
- **Action Space ($\mathcal{A} = \{a_{\text{accept}}, a_{\text{abstain}}\}$):**
  - $a_{\text{accept}}$: Present $y^*$ to the user.
  - $a_{\text{abstain}}$: Intercept $y^*$ and return a deterministic safety fallback:
    > `"The available context does not contain sufficient verified information to answer this query safely."`

### 1.2 Probability Elicitation via Grounded Monte Carlo Sampling
To estimate the epistemic probability $p = P(\theta_1 \mid q, C, I)$:
1. Draw $M = 5$ stochastic completions $\{y_1, \dots, y_M\}$ at temperature $T = 0.7$ via Groq (`openai/gpt-oss-20b` with automatic fallback to `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`).
2. Each sample outputs a structured JSON matching `LLMSampleOutput` with a normalized `canonical_answer` and boolean `context_entailed` flag $e_m \in \{0, 1\}$.
3. Compute the Monte Carlo self-consistency agreement rate for the modal answer $y^*$:
   $$f_{\text{MC}} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\big(\text{equiv}(y_m, y^*) \land e_m = 1\big)$$
4. Gate the belief by the retrieval relevance threshold $\tau_{\text{ret}} = 0.35$:
   $$p = f_{\text{MC}} \cdot \mathbb{I}(s_{\max} \ge \tau_{\text{ret}})$$

### 1.3 Asymmetric Utility Matrix & Critical Decision Threshold
Given utilities $u_{\text{TP}} = +10$, $u_{\text{FP}} = -50$, $u_{\text{FN}} = -2$, and $u_{\text{TN}} = 0$:
$$EU(a_{\text{accept}} \mid p) = p \cdot u_{\text{TP}} + (1 - p) \cdot u_{\text{FP}}$$
$$EU(a_{\text{abstain}} \mid p) = p \cdot u_{\text{FN}} + (1 - p) \cdot u_{\text{TN}}$$

The agent chooses $a^* = \arg\max_{a \in \mathcal{A}} EU(a \mid p)$, yielding the critical probability threshold:
$$p^* = \frac{u_{\text{TN}} - u_{\text{FP}}}{(u_{\text{TP}} - u_{\text{FN}}) + (u_{\text{TN}} - u_{\text{FP}})} = \frac{50}{12 + 50} = \frac{50}{62} \approx 0.8065$$

---

## 2. Repository Structure

```text
rag-opm/
├── .env                <-- Create this file and add your GROQ_API_KEY
├── .env.example        <-- Template for environment variables
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
├── prompts/
│   └── rag_opm/
│       └── v1.yaml
├── data/
│   ├── corpus/
│   │   ├── robotics_hardware.md
│   │   ├── university_policy.md
│   │   └── medical_triage.md
│   └── golden_eval.jsonl
├── results/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   ├── logger.py
│   ├── schema.py
│   ├── prompts.py
│   ├── retriever.py
│   ├── opm_engine.py
│   ├── eval.py
│   └── app.py
└── tests/
    ├── __init__.py
    ├── test_prompts.py
    ├── test_opm_engine.py
    └── test_calibration.py
```

---

## 3. Quickstart & Execution

### 3.1 Environment Setup
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY (or run in offline deterministic simulation mode)
make install
```

### 3.2 Run Unit & Mathematical Verification Suite
```bash
make test
```

### 3.3 Build & Verify Vector Index
```bash
make index
```

### 3.4 Run Smoke & Full Calibration Evaluations
```bash
make eval-smoke
make eval
```

### 3.5 Launch Interactive Streamlit Dashboard
```bash
make ui
```
