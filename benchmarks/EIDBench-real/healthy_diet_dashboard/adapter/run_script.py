#!/usr/bin/env python3
"""CLI runner for the Healthy Diet Dashboard EIDBench-real adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from case import run_script  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-id", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_script(args.script_id, args.input.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
