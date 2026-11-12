"""Central configuration and environment singleton for the OPM RAG system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

# Load .env from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

ABSTAIN_FALLBACK_MESSAGE = (
    "The available context does not contain sufficient verified information "
    "to answer this query safely."
)


@dataclass(frozen=False)
class AppConfig:
    """Application-wide configuration parameters and decision-theoretic constants."""

    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip()
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    )
    fallback_model: str = field(
        default_factory=lambda: os.getenv(
            "FALLBACK_MODEL", "llama-3.3-70b-versatile"
        ).strip()
    )
    secondary_fallback_model: str = field(
        default_factory=lambda: os.getenv(
            "SECONDARY_FALLBACK_MODEL", "llama-3.1-8b-instant"
        ).strip()
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper().strip()
    )

    # Asymmetric Utility Matrix U(a, theta)
    utility_tp: float = field(
        default_factory=lambda: float(os.getenv("UTILITY_TP", "10.0"))
    )
    utility_fp: float = field(
        default_factory=lambda: float(os.getenv("UTILITY_FP", "-50.0"))
    )
    utility_fn: float = field(
        default_factory=lambda: float(os.getenv("UTILITY_FN", "-2.0"))
    )
    utility_tn: float = field(
        default_factory=lambda: float(os.getenv("UTILITY_TN", "0.0"))
    )

    # Monte Carlo & Retrieval Hyperparameters
    mc_samples: int = field(
        default_factory=lambda: int(os.getenv("MC_SAMPLES", "5"))
    )
    mc_temperature: float = field(
        default_factory=lambda: float(os.getenv("MC_TEMPERATURE", "0.7"))
    )
    retrieval_threshold: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_THRESHOLD", "0.35"))
    )

    # Filesystem paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    corpus_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "corpus")
    golden_eval_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "golden_eval.jsonl"
    )
    prompts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")

    @property
    def has_valid_api_key(self) -> bool:
        """Return True if a non-placeholder Groq API key is configured."""
        return bool(
            self.groq_api_key
            and self.groq_api_key != "your_groq_api_key_here"
            and len(self.groq_api_key) > 10
        )


# Global singleton instance
settings = AppConfig()


def get_groq_client(api_key: Optional[str] = None) -> Optional[Groq]:
    """Instantiate and return a Groq client if a valid API key is available."""
    key = (api_key if api_key is not None else settings.groq_api_key).strip()
    if not key or key == "your_groq_api_key_here":
        return None
    return Groq(api_key=key)
