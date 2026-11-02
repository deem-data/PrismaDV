"""When a method alarms on a batch: the one rule for every method and benchmark.
"""
from __future__ import annotations

from pathlib import Path

import yaml

LOAD = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
NOT_RUN = "Unable to instantiate constraint"


def load(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=LOAD) or {}


def _failed(entries) -> bool:
    return any(e.get("status") is False and e.get("level", "error") == "error"
               and NOT_RUN not in (e.get("reason_if_failed") or "") for e in entries or [])


def alarm(doc: dict) -> bool:
    """True when a validation result rejects its batch."""
    if "pass" in doc:                                   # AutoTest and tfdv on EIDBench-synth
        return not doc["pass"]
    if doc.get("tables"):                               # EIDBench-real: table -> column -> {"code": [...]}
        return any(_failed((body or {}).get("code")) for cols in doc["tables"].values()
                   for body in (cols or {}).values())
    if "results" in doc:                                # EIDBench-synth: column -> {"code": [...]}
        return any(_failed(col.get("code")) for col in doc["results"].values())
    return (doc.get("summary") or {}).get("failed_error", 0) > 0     # AutoTest and tfdv on EIDBench-real
