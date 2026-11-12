"""Unit tests for YAML prompt management and variable interpolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prompts import PromptManager, load_prompt


def test_load_default_rag_opm_prompt() -> None:
    """Verify v1.yaml loads with expected metadata and instructions."""
    template = load_prompt("rag_opm", "v1")
    assert template.version == "1.0"
    assert template.name == "rag_opm_grounded_qa"
    assert "context_entailed" in template.system_prompt
    assert "canonical_answer" in template.system_prompt
    assert "{context}" in template.user_prompt_template
    assert "{query}" in template.user_prompt_template


def test_render_messages_interpolates_context_and_query() -> None:
    """Verify system and user messages are rendered with context and query."""
    template = load_prompt("rag_opm", "v1")
    messages = template.render_messages(
        context="Operating voltage is 12.0V DC and stall torque is 4.1 N·m.",
        query="What is the operating voltage?",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "12.0V DC" in messages[1]["content"]
    assert "What is the operating voltage?" in messages[1]["content"]


def test_render_messages_rejects_empty_query() -> None:
    """Verify empty or whitespace-only queries raise ValueError."""
    template = load_prompt("rag_opm", "v1")
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        template.render_messages(context="Some context", query="   ")


def test_prompt_manager_missing_file_raises(tmp_path: Path) -> None:
    """Verify requesting a non-existent prompt version raises FileNotFoundError."""
    manager = PromptManager(prompts_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.load("rag_opm", "v99")
