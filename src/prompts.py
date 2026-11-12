"""YAML prompt loader and template renderer for versioned RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable container for a versioned YAML prompt specification."""

    version: str
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str

    def render_messages(self, context: str, query: str) -> List[Dict[str, str]]:
        """Render OpenAI/Groq-compatible chat messages with interpolated context and query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty when rendering prompt messages.")

        formatted_user = self.user_prompt_template.format(
            context=context.strip() if context else "No relevant context retrieved.",
            query=query.strip(),
        )
        return [
            {"role": "system", "content": self.system_prompt.strip()},
            {"role": "user", "content": formatted_user.strip()},
        ]


class PromptManager:
    """Loads and caches YAML prompt templates from the prompts/ directory."""

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self.prompts_dir = (prompts_dir or settings.prompts_dir).resolve()
        self._cache: Dict[str, PromptTemplate] = {}

    def load(self, task_name: str = "rag_opm", version: str = "v1") -> PromptTemplate:
        """Load and validate a YAML prompt file by task name and version."""
        cache_key = f"{task_name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        yaml_path = self.prompts_dir / task_name / f"{version}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt file not found at {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle)

        if not isinstance(raw_data, dict):
            raise ValueError(f"Prompt YAML at {yaml_path} must deserialize to a mapping.")

        required_keys = {"version", "name", "system_prompt", "user_prompt_template"}
        missing_keys = required_keys - set(raw_data.keys())
        if missing_keys:
            raise KeyError(
                f"Prompt YAML at {yaml_path} is missing required keys: {sorted(missing_keys)}"
            )

        template = PromptTemplate(
            version=str(raw_data["version"]),
            name=str(raw_data["name"]),
            description=str(raw_data.get("description", "")).strip(),
            system_prompt=str(raw_data["system_prompt"]),
            user_prompt_template=str(raw_data["user_prompt_template"]),
        )
        self._cache[cache_key] = template
        logger.debug(
            "Loaded prompt template",
            extra={"payload": {"task": task_name, "version": version, "path": str(yaml_path)}},
        )
        return template


# Default module-level convenience loader
_default_manager = PromptManager()


def load_prompt(task_name: str = "rag_opm", version: str = "v1") -> PromptTemplate:
    """Load a prompt template using the default PromptManager singleton."""
    return _default_manager.load(task_name=task_name, version=version)
