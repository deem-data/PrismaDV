"""The ICDBench codex prompt document.

Rendering rules live in `workflow_prismadv.utils.prompt_docs`, shared with the
EIDBench rows so an edit to how documents are read reaches all three. The wording
is separate because the setting is: an ICDBench case is a code snippet, a small
sample table and one target column, not a whole program or pipeline.
"""
from __future__ import annotations

from pathlib import Path

from workflow_prismadv.utils.prompt_docs import (
    COMMENT_PATTERN,
    PLACEHOLDER_PATTERN,
    load_prompt_template as _load,
    render_prompt as _render,
)

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPT_PATH = PROMPTS_DIR / "codex.md"

# Without these the prompt has no subject: the agent would be asked to constrain
# a snippet it was never shown, or a column it was never named. Every other
# ICDBench baseline names the target column too.
REQUIRED_PLACEHOLDERS = frozenset({"CODE", "DATA_SAMPLE", "TARGET_COLUMN"})

__all__ = [
    "COMMENT_PATTERN",
    "PLACEHOLDER_PATTERN",
    "PROMPTS_DIR",
    "PROMPT_PATH",
    "REQUIRED_PLACEHOLDERS",
    "load_prompt_template",
    "render_prompt",
]


def load_prompt_template(path: Path | str | None = None) -> str:
    """Read the checked-in document, or one passed in its place."""
    return _load(path or PROMPT_PATH)


def render_prompt(values: dict[str, str], path: Path | str | None = None) -> str:
    """Substitute `{{NAME}}` placeholders in the ICDBench prompt document."""
    return _render(values, path or PROMPT_PATH, REQUIRED_PLACEHOLDERS)
