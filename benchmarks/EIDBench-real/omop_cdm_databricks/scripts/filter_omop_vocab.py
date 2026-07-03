"""Filter the Athena OMOP vocab dump to only the concepts referenced by our Synthea sample.

One-shot offline preprocessing step. Run from the example root once:

    uv run python scripts/filter_omop_vocab.py

It expects the full Athena dump under ``/tmp/omop_vocab_full/`` (download with
``curl https://hls-eng-data-public.s3.amazonaws.com/data/rwe/omop-vocabs/<table>.csv.gz``)
and writes the filtered subset to ``files/clean/tables/vocab/``.

Strategy:
1. Collect distinct source codes from the 4 Synthea code-bearing tables, paired with the
   vocabulary that the ETL maps them through (see 4-omop531-etl-synthea.sql joins):
     - conditions.CODE    → SNOMED
     - medications.CODE   → RxNorm
     - procedures.CODE    → SNOMED  (Synthea uses SNOMED for procedure codes)
     - observations.CODE  → LOINC
2. Filter CONCEPT to rows where (vocabulary_id, concept_code) matches a source pair.
3. Follow "Maps to" relationships in CONCEPT_RELATIONSHIP from those source concepts.
4. Keep the source concepts and the Maps-to targets in CONCEPT.
5. Keep CONCEPT_RELATIONSHIP rows whose "maps to" both endpoints live in the keep set.
6. Ship VOCABULARY / DOMAIN / CONCEPT_CLASS / RELATIONSHIP verbatim (all tiny).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
VOCAB_FULL = Path("/tmp/omop_vocab_full")
OUTPUT_DIR = EXAMPLE_DIR / "files" / "clean" / "tables" / "vocab"
INPUTS_DIR = EXAMPLE_DIR / "files" / "clean" / "tables"

SOURCE_TO_VOCAB = {
    "conditions": ("CODE", "SNOMED"),
    "medications": ("CODE", "RxNorm"),
    "procedures": ("CODE", "SNOMED"),
    "observations": ("CODE", "LOINC"),
}

# Standard OMOP administrative concept IDs the ETL and the downstream CHF
# cohort script reference but which are never source codes in Synthea, so
# they don't make it into the Maps-to closure. Hardcode them so person /
# visit_occurrence joins to `concept` resolve.
ADMIN_CONCEPT_IDS: set[str] = {
    # Gender
    "8507",       # MALE
    "8532",       # FEMALE
    # Visit concepts the Synthea ETL writes into visit_occurrence.visit_concept_id
    "9201",       # Inpatient Visit
    "9202",       # Outpatient Visit
    "9203",       # Emergency Room Visit (CHF outcome cohort)
    "44818518",   # Visit (Visit Type Concept)
    # Race
    "8527",       # White
    "8516",       # Black or African American
    "8657",       # American Indian or Alaska Native
    "8557",       # Native Hawaiian or Other Pacific Islander
    "8515",       # Asian
    # Ethnicity
    "38003563",   # Hispanic or Latino
    "38003564",   # Not Hispanic or Latino
}


def collect_synthea_codes() -> dict[str, set[str]]:
    """Return {vocabulary_id: {code, ...}} for the codes in our Synthea CSVs."""
    codes_by_vocab: dict[str, set[str]] = {}
    for table, (col, vocab) in SOURCE_TO_VOCAB.items():
        df = pd.read_csv(INPUTS_DIR / f"{table}.csv", usecols=[col], dtype=str)
        vals = set(df[col].dropna().astype(str).unique())
        codes_by_vocab.setdefault(vocab, set()).update(vals)
        print(f"{table}: {len(vals)} distinct codes → {vocab}")
    print(f"total: {sum(len(v) for v in codes_by_vocab.values())} codes across {len(codes_by_vocab)} vocabs")
    return codes_by_vocab


def load_full_concept() -> pd.DataFrame:
    print("loading CONCEPT.csv.gz (~30s, ~5M rows)...")
    concept = pd.read_csv(VOCAB_FULL / "CONCEPT.csv.gz", dtype=str, na_filter=False, on_bad_lines="skip")
    print(f"  CONCEPT rows: {len(concept):,}")
    return concept


def select_source_concepts(concept: pd.DataFrame, codes_by_vocab: dict[str, set[str]]) -> pd.DataFrame:
    mask = pd.Series(False, index=concept.index)
    for vocab, codes in codes_by_vocab.items():
        mask |= (concept["vocabulary_id"] == vocab) & (concept["concept_code"].isin(codes))
    matched = concept[mask].copy()
    print(f"  matched source CONCEPT rows: {len(matched):,}")
    return matched


def load_full_relationship() -> pd.DataFrame:
    print("loading CONCEPT_RELATIONSHIP.csv.gz (~60s, ~40M rows)...")
    rel = pd.read_csv(VOCAB_FULL / "CONCEPT_RELATIONSHIP.csv.gz", dtype=str, na_filter=False, on_bad_lines="skip")
    print(f"  CONCEPT_RELATIONSHIP rows: {len(rel):,}")
    return rel


def write_gz(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{name}.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    print(f"wrote {out.relative_to(EXAMPLE_DIR)}  ({len(df):,} rows, {out.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    codes_by_vocab = collect_synthea_codes()

    concept_all = load_full_concept()
    source_concept = select_source_concepts(concept_all, codes_by_vocab)
    source_ids = set(source_concept["concept_id"].tolist())

    rel_all = load_full_relationship()
    mapsto = rel_all["relationship_id"].str.lower() == "maps to"
    rel_from_src = rel_all[mapsto & rel_all["concept_id_1"].isin(source_ids)]
    print(f"  'maps to' edges from source concepts: {len(rel_from_src):,}")

    target_ids = set(rel_from_src["concept_id_2"].tolist())
    keep_ids = source_ids | target_ids | ADMIN_CONCEPT_IDS
    print(f"keep_ids (source + Maps-to targets + admin concepts): {len(keep_ids):,}")

    rel_keep_mask = (
        mapsto
        & rel_all["concept_id_1"].isin(keep_ids)
        & rel_all["concept_id_2"].isin(keep_ids)
    )
    rel_keep = rel_all[rel_keep_mask].copy()
    print(f"keeping {len(rel_keep):,} CONCEPT_RELATIONSHIP rows ('maps to' within keep_ids)")

    # CONCEPT_ANCESTOR + ingredient expansion: the drug_era ETL joins
    # concept_ancestor.descendant_concept_id = drug_exposure.drug_concept_id
    # and pulls the Ingredient-class ancestor. The Maps-to closure above does
    # not include those ancestor concepts (they live one hierarchy hop away),
    # so we widen keep_ids to include every ancestor of any current keep_id.
    print("loading CONCEPT_ANCESTOR.csv.gz (~90s, ~80M rows)...")
    anc = pd.read_csv(VOCAB_FULL / "CONCEPT_ANCESTOR.csv.gz", dtype=str, na_filter=False,
                      on_bad_lines="skip")
    print(f"  CONCEPT_ANCESTOR rows: {len(anc):,}")
    desc_in = anc["descendant_concept_id"].isin(keep_ids)
    ancestor_ids_for_drugs = set(anc.loc[desc_in, "ancestor_concept_id"].tolist())
    print(f"  ancestors of keep_ids: {len(ancestor_ids_for_drugs):,}")
    keep_ids_expanded = keep_ids | ancestor_ids_for_drugs
    # Refilter CONCEPT to include the ancestor concepts too.
    concept_keep = concept_all[concept_all["concept_id"].isin(keep_ids_expanded)].copy()
    print(f"  CONCEPT (after ancestor expansion): {len(concept_keep):,} rows")
    anc_keep_mask = (
        anc["descendant_concept_id"].isin(keep_ids_expanded)
        & anc["ancestor_concept_id"].isin(keep_ids_expanded)
    )
    anc_keep = anc[anc_keep_mask].copy()
    print(f"  keeping {len(anc_keep):,} CONCEPT_ANCESTOR rows (both endpoints in expanded keep_ids)")
    write_gz(concept_keep, "CONCEPT")
    write_gz(rel_keep, "CONCEPT_RELATIONSHIP")
    write_gz(anc_keep, "CONCEPT_ANCESTOR")

    # Tiny tables: ship in full so the OMOP setup notebook can populate them
    # without surprises.
    for tiny in ("VOCABULARY", "DOMAIN", "CONCEPT_CLASS", "RELATIONSHIP"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dst = OUTPUT_DIR / f"{tiny}.csv.gz"
        shutil.copyfile(VOCAB_FULL / f"{tiny}.csv.gz", dst)
        print(f"copied {dst.relative_to(EXAMPLE_DIR)}  ({dst.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
