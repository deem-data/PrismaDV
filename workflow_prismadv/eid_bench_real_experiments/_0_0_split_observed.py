#!/usr/bin/env python3
"""Split each EIDBench-real example into a disjoint observed half and clean half.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import oyaml as yaml
import pandas as pd

from prismadv.utils import get_project_root

EXAMPLE_IDS = ["bank_marketing_analysis", "healthy_diet_dashboard", "omop_cdm_databricks"]
OBSERVED_FRACTION = 0.6
DEFAULT_SEED = 1


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _canon_country(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def split_bank_marketing(tables: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.Series]:
    """Row-level split; the single table has no key to preserve."""
    frame = tables["bank_marketing"]
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(frame))
    cut = int(len(frame) * OBSERVED_FRACTION)
    mask = pd.Series(False, index=frame.index)
    mask.iloc[order[:cut]] = True
    return {"bank_marketing": mask}


def split_omop(tables: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.Series]:
    """Patient-level split; every child table follows its PATIENT."""
    patients = tables["patients"]
    ids = patients["Id"].dropna().unique()
    rng = np.random.RandomState(seed)
    ids = rng.permutation(ids)
    observed_ids = set(ids[: int(len(ids) * OBSERVED_FRACTION)])

    masks = {"patients": patients["Id"].isin(observed_ids)}
    for name, frame in tables.items():
        if name == "patients":
            continue
        masks[name] = frame["PATIENT"].isin(observed_ids)
    return masks


def split_healthy_diet(tables: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.Series]:
    """Country-level split across the fact table and both lookups.

    The fact table joins country_codes on the numeric code and
    countries_by_continents on the country name, so the country is the only unit
    that keeps both joins inside one half.
    """
    price = tables["price_of_healthy_diet"]
    codes = tables["country_codes"]
    continents = tables["countries_by_continents"]

    price_key = pd.to_numeric(price["country_code"], errors="coerce")
    codes_key = pd.to_numeric(codes["Numeric"], errors="coerce")

    all_codes = sorted(set(price_key.dropna()) | set(codes_key.dropna()))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_codes)
    observed_codes = set(shuffled[: int(len(shuffled) * OBSERVED_FRACTION)])

    price_names = _canon_country(price["country"])
    name_of_code = dict(zip(price_key, price_names))
    observed_names = {name_of_code[c] for c in observed_codes if c in name_of_code}

    linked = set(name_of_code.values())
    continent_names = _canon_country(continents["Country"])
    orphans = sorted(set(continent_names) - linked)
    rng_orphan = np.random.RandomState(seed + 1)
    shuffled_orphans = rng_orphan.permutation(orphans) if orphans else []
    observed_names |= set(shuffled_orphans[: int(len(shuffled_orphans) * OBSERVED_FRACTION)])

    return {
        "price_of_healthy_diet": price_key.isin(observed_codes),
        "country_codes": codes_key.isin(observed_codes),
        "countries_by_continents": continent_names.isin(observed_names),
    }


SPLITTERS = {
    "bank_marketing_analysis": split_bank_marketing,
    "omop_cdm_databricks": split_omop,
    "healthy_diet_dashboard": split_healthy_diet,
}


def check_bank_marketing(halves: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    return []


def check_omop(halves: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    """Referential integrity has to survive, or the three scripts stop joining."""
    lines = []
    for half, tables in halves.items():
        patient_ids = set(tables["patients"]["Id"])
        encounter_ids = set(tables["encounters"]["Id"])
        for name, frame in sorted(tables.items()):
            if name == "patients":
                continue
            to_patient = frame["PATIENT"].isin(patient_ids).mean() if len(frame) else float("nan")
            note = f"{half:8s} {name:14s} PATIENT->patients {to_patient:7.1%}"
            if "ENCOUNTER" in frame.columns:
                to_encounter = frame["ENCOUNTER"].isin(encounter_ids).mean() if len(frame) else float("nan")
                note += f"   ENCOUNTER->encounters {to_encounter:7.1%}"
            lines.append(note)
    return lines


def check_healthy_diet(halves: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    """Both joins the cleaning script performs have to stay as covered as today."""
    lines = []
    for half, tables in halves.items():
        price = tables["price_of_healthy_diet"]
        codes_key = set(pd.to_numeric(tables["country_codes"]["Numeric"], errors="coerce").dropna())
        continent_names = set(_canon_country(tables["countries_by_continents"]["Country"]))
        if not len(price):
            lines.append(f"{half:8s} price half is empty")
            continue
        to_codes = pd.to_numeric(price["country_code"], errors="coerce").isin(codes_key).mean()
        to_continents = _canon_country(price["country"]).isin(continent_names).mean()
        lines.append(f"{half:8s} price->country_codes {to_codes:7.1%}   "
                     f"price->countries_by_continents {to_continents:7.1%}")
    return lines


CHECKS = {
    "bank_marketing_analysis": check_bank_marketing,
    "omop_cdm_databricks": check_omop,
    "healthy_diet_dashboard": check_healthy_diet,
}


def input_layout(example_root: Path) -> dict[str, str]:
    """The manifest's own names for the four bundle directories."""
    manifest = yaml.safe_load((example_root / "manifest.yaml").read_text())
    return manifest["input_layout"]


def clean_input_dir(example_root: Path) -> Path | None:
    """Where the runner reads the clean bundle from, when it is not tables/.

    omop points `clean_input_dir` straight at `clean_tables_dir`, so writing the
    tables already satisfies it; the other two keep a separate `input/` that has
    to hold the same CSVs.
    """
    layout = input_layout(example_root)
    input_dir = example_root / layout["clean_input_dir"]
    tables_dir = example_root / layout["clean_tables_dir"]
    return None if input_dir == tables_dir else input_dir


def source_dir(example_root: Path) -> Path:
    """Where the full tables live: the staged raw copy once one exists."""
    staged = example_root / "files" / "raw" / "tables"
    return staged if staged.exists() else example_root / "files" / "clean" / "tables"


def load_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    return {path.stem: _read(path) for path in sorted(tables_dir.glob("*.csv"))}


def split_example(example_id: str, project_root: Path, seed: int) -> dict[str, Any]:
    example_root = project_root / "benchmarks" / "EIDBench-real" / example_id
    tables_dir = source_dir(example_root)
    tables = load_tables(tables_dir)
    if not tables:
        raise FileNotFoundError(f"no CSV tables under {tables_dir}")

    masks = SPLITTERS[example_id](tables, seed)
    halves = {
        "observed": {name: frame[masks[name]].reset_index(drop=True) for name, frame in tables.items()},
        "clean": {name: frame[~masks[name]].reset_index(drop=True) for name, frame in tables.items()},
    }
    return {
        "example_root": example_root,
        "tables_dir": tables_dir,
        "source": tables,
        "halves": halves,
    }


def report(example_id: str, result: dict[str, Any]) -> None:
    source, halves = result["source"], result["halves"]
    print(f"\n########## {example_id}")
    print(f"  source: {result['tables_dir'].relative_to(get_project_root())}")
    print(f"  {'table':30s} {'source':>9s} {'observed':>12s} {'clean':>12s}")
    for name in sorted(source):
        total = len(source[name])
        n_obs, n_clean = len(halves['observed'][name]), len(halves['clean'][name])
        share_obs = f"{n_obs} ({n_obs / total:.0%})" if total else "0"
        share_clean = f"{n_clean} ({n_clean / total:.0%})" if total else "0"
        flag = "" if n_obs and n_clean else "   <-- EMPTY HALF"
        print(f"  {name:30s} {total:9d} {share_obs:>12s} {share_clean:>12s}{flag}")
        if n_obs + n_clean != total:
            print(f"     ROWS LOST: {total - n_obs - n_clean}")
    for line in CHECKS[example_id](halves):
        print(f"    {line}")

    root = get_project_root()
    input_dir = clean_input_dir(result["example_root"])
    if input_dir is None:
        print("    clean input: manifest points clean_input_dir at clean_tables_dir, "
              "nothing extra to mirror")
    else:
        present = sorted(p.name for p in input_dir.glob("*.csv")) if input_dir.exists() else []
        print(f"    clean input: mirror {len(halves['clean'])} tables -> "
              f"{input_dir.relative_to(root)}  (holds {len(present)} csv today)")


def write_example(example_id: str, result: dict[str, Any]) -> None:
    """Stage the full tables under files/raw, then write both halves."""
    example_root = result["example_root"]
    clean_tables = example_root / "files" / "clean" / "tables"
    raw_tables = example_root / "files" / "raw" / "tables"
    if not raw_tables.exists():
        raw_tables.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(clean_tables, raw_tables)
        print(f"  staged full tables -> {raw_tables}")

    for half, tables in result["halves"].items():
        target = example_root / "files" / half / "tables"
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(target / f"{name}.csv", index=False)
        for extra in raw_tables.iterdir():
            if extra.is_dir() and not (target / extra.name).exists():
                shutil.copytree(extra, target / extra.name)
        print(f"  wrote {len(tables)} tables -> {target}")

    input_dir = clean_input_dir(example_root)
    if input_dir is not None:
        input_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in result["halves"]["clean"].items():
            frame.to_csv(input_dir / f"{name}.csv", index=False)
        print(f"  mirrored {len(result['halves']['clean'])} tables -> {input_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example-id", action="append", choices=EXAMPLE_IDS, default=None,
                        help="Restrict to one or more examples (default: all).")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Split seed (default: {DEFAULT_SEED}, matching EIDBench-synth).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the split and write nothing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root or get_project_root()
    for example_id in (args.example_id or EXAMPLE_IDS):
        result = split_example(example_id, project_root, args.seed)
        report(example_id, result)
        if not args.dry_run:
            write_example(example_id, result)
    if args.dry_run:
        print("\ndry run: nothing written. Corrupted bundles and expected outputs "
              "must be regenerated after a real write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
