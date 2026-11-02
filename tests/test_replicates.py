"""The shared replicate convention: filenames, folding, and discovery."""
from pathlib import Path

from workflow_prismadv.utils.replicates import (
    artifact_filename,
    existing_artifacts,
    fold_replicate,
    normalize_replicate,
    parse_replicate,
    replicate_of_artifact,
    split_replicate,
)


def test_first_replicate_keeps_the_bare_name():
    assert normalize_replicate(None) == normalize_replicate(0) == normalize_replicate(1) == 1
    name = artifact_filename("prismadv", "gpt-5", None)
    parts = name[:-5].split("--")
    assert parts[:2] == ["prismadv", "gpt-5"] and len(parts) == 3
    assert parse_replicate(name) == 1


def test_later_replicates_sit_before_the_timestamp():
    name = artifact_filename("few_shot", "gemini-2.5-pro", 3)
    parts = name[:-5].split("--")
    assert parts[:3] == ["few_shot", "gemini-2.5-pro", "r3"]
    assert len(parts[-1]) == 15 and parse_replicate(name) == 3
    # The synth codex layout, effort and no timestamp, parses the same way.
    assert parse_replicate("codex--gpt-5.6-sol--medium--r2") == 2
    assert parse_replicate("codex--gpt-5.6-sol--medium") == 1
    # An `r` inside the model name is not a replicate.
    assert parse_replicate("single_shot--r1-model--20260902_120000") == 1


def test_artifact_record_beats_filename():
    assert replicate_of_artifact(Path("x/prismadv--gpt-5--20260902_120000.yaml"), {"replicate": 2}) == 2
    assert replicate_of_artifact(Path("x/prismadv--gpt-5--r3--20260902_120000.yaml"), {}) == 3


def test_fold_and_split():
    assert fold_replicate("gpt-5", 1) == "gpt-5"
    assert fold_replicate("gpt-5", 2) == "gpt-5#r2"
    assert fold_replicate("gpt-5#r2", 2) == "gpt-5#r2"
    assert split_replicate("gpt-5.6-sol@medium#r3") == ("gpt-5.6-sol@medium", 3)
    assert split_replicate("gpt-5") == ("gpt-5", 1)


def test_existing_artifacts_are_per_replicate(tmp_path):
    for name in ("prismadv--gpt-5--20260901_000000.yaml",
                 "prismadv--gpt-5--r2--20260902_000000.yaml",
                 "prismadv--gpt-5-mini--20260901_000000.yaml",
                 "few_shot--gpt-5--20260901_000000.yaml"):
        (tmp_path / name).write_text("constraints: {}\n")
    assert [p.name for p in existing_artifacts(tmp_path, "prismadv", "gpt-5", 1)] == [
        "prismadv--gpt-5--20260901_000000.yaml"]
    assert [p.name for p in existing_artifacts(tmp_path, "prismadv", "gpt-5", 2)] == [
        "prismadv--gpt-5--r2--20260902_000000.yaml"]
    assert existing_artifacts(tmp_path, "prismadv", "gpt-5", 3) == []
    # `gpt-5` must not match `gpt-5-mini`.
    assert len(existing_artifacts(tmp_path, "prismadv", "gpt-5", 1)) == 1

