from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPLICATE_PART = re.compile(r"^r(\d+)$")
FOLDED_SUFFIX = re.compile(r"#r(\d+)$")


def normalize_replicate(replicate) -> int:
    """None, 0, '' and 1 all mean the first replicate."""
    return max(int(replicate or 1), 1)


def replicate_part(replicate) -> str:
    """The filename segment for a replicate: '' for the first, 'rN' after that."""
    replicate = normalize_replicate(replicate)
    return f"r{replicate}" if replicate > 1 else ""


def parse_replicate(stem: str | Path) -> int:
    """Replicate encoded in a filename, 1 when there is none."""
    stem = Path(stem).stem if isinstance(stem, Path) or stem.endswith(".yaml") else stem
    for part in stem.split("--")[2:]:
        match = REPLICATE_PART.fullmatch(part)
        if match:
            return int(match.group(1))
    return 1


def replicate_of_artifact(path: str | Path, raw: dict | None = None) -> int:
    """The artifact's own record wins over its filename; both default to 1."""
    if isinstance(raw, dict) and raw.get("replicate"):
        return normalize_replicate(raw["replicate"])
    return parse_replicate(Path(path))


def artifact_stem(method: str, model: str, replicate=None, *, timestamp: bool = True,
                  extra: Iterable[str] = ()) -> str:
    """`method--model[--extra...][--rN][--timestamp]`."""
    parts = [method, model, *extra]
    part = replicate_part(replicate)
    if part:
        parts.append(part)
    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "--".join(parts)


def artifact_filename(method: str, model: str, replicate=None, *, timestamp: bool = True,
                      extra: Iterable[str] = ()) -> str:
    return artifact_stem(method, model, replicate, timestamp=timestamp, extra=extra) + ".yaml"


def existing_artifacts(directory: Path, method: str, model: str, replicate=None) -> list[Path]:
    """Artifacts of this method, model and replicate, oldest first."""
    wanted = normalize_replicate(replicate)
    found = []
    for path in sorted(Path(directory).glob(f"{method}--{model}--*.yaml")):
        parts = path.stem.split("--")
        if len(parts) < 2 or parts[0] != method or parts[1] != model:
            continue
        if parse_replicate(path.stem) == wanted:
            found.append(path)
    return found


def fold_replicate(name: str, replicate) -> str:
    """`gpt-5` -> `gpt-5#r2` for replicate 2; unchanged for the first replicate."""
    replicate = normalize_replicate(replicate)
    if replicate == 1 or FOLDED_SUFFIX.search(name):
        return name
    return f"{name}#r{replicate}"


def split_replicate(name: str) -> tuple[str, int]:
    """`gpt-5#r2` -> ('gpt-5', 2); `gpt-5` -> ('gpt-5', 1)."""
    match = FOLDED_SUFFIX.search(name)
    if not match:
        return name, 1
    return name[:match.start()], int(match.group(1))


REPLICATE_HELP = (
    "Which repeat of this exact configuration to run (default 1). Replicate 1 is "
    "the existing single run; 2, 3, ... write beside it as new samples for a "
    "variance study and are skipped only if that same replicate already exists."
)


def add_script_argument(parser) -> None:
    """`--script`: restrict a synth run to some script stems, e.g. general_task_11."""
    parser.add_argument("--script", action="append", default=None, dest="script_stems",
                        help="Run only this script stem (repeatable), e.g. general_task_11. "
                             "Default: every script of the subtask.")


def select_scripts(script_path_list, script_stems=None) -> list:
    """The scripts a runner loops over, narrowed to `script_stems` when given."""
    if not script_stems:
        return list(script_path_list)
    wanted = set(script_stems)
    chosen = [path for path in script_path_list if path.stem in wanted]
    missing = wanted - {path.stem for path in chosen}
    if missing:
        raise ValueError(f"no script matches {sorted(missing)}; "
                         f"available: {sorted(p.stem for p in script_path_list)}")
    return chosen


def add_replicate_argument(parser) -> None:
    parser.add_argument("--replicate", type=int, default=1, help=REPLICATE_HELP)
