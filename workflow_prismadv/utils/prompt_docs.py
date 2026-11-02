from __future__ import annotations

import re
from pathlib import Path

PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")

# Notes to the prompt's author, stripped before the agent sees anything.
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def load_prompt_template(path: Path | str) -> str:
    """Read a prompt document, minus the notes written for its author."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return COMMENT_PATTERN.sub("", path.read_text(encoding="utf-8")).strip() + "\n"


def render_prompt(values: dict[str, str], path: Path | str,
                  required: frozenset[str] = frozenset()) -> str:
    template = load_prompt_template(path)
    found = set(PLACEHOLDER_PATTERN.findall(template))
    name = Path(path).name
    if unknown := sorted(found - set(values)):
        raise ValueError(
            f"{name} uses placeholder(s) nothing supplies: {unknown}. "
            f"Available placeholders: {sorted(values)}"
        )
    if missing := sorted(required - found):
        raise ValueError(
            f"{name} no longer mentions {missing}, so the agent would be asked to write "
            f"constraints for something it is never shown. Restore the placeholder."
        )
    rendered = template
    for placeholder, value in values.items():
        rendered = rendered.replace("{{" + placeholder + "}}", str(value))
    return rendered
