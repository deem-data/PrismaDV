from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from diskcache import Cache, FanoutCache

SHARDS = 16                     # as dspy.clients.cache.Cache opens its FanoutCache
LATENCY = "latency"
KEEP_VERSIONS = 2


def _open(directory: Path) -> tuple[FanoutCache, Cache]:
    return FanoutCache(str(directory), shards=SHARDS), Cache(str(directory / LATENCY))


def seed_local(store: Path, local: Path) -> set:
    """Copy the store's current version into `local`; return the keys it holds."""
    current = store / "current"
    if current.exists():
        shutil.copytree(current.resolve(), local, dirs_exist_ok=True)
    responses, latency = _open(local)
    keys = set(responses) | {("latency", k) for k in latency}
    responses.close()
    latency.close()
    return keys


def save_delta(store: Path, local: Path, seed_keys: set, job: str) -> int:
    """Write the entries `local` holds beyond `seed_keys` to <store>/deltas/<job>; return how many."""
    responses, latency = _open(local)
    out = store / "deltas" / job
    new_responses, new_latency = _open(out)
    n = 0
    for k in responses:
        if k not in seed_keys:
            new_responses.set(k, responses.get(k))
            n += 1
    for k in latency:
        if ("latency", k) not in seed_keys:
            new_latency.set(k, latency.get(k))
    for c in (responses, latency, new_responses, new_latency):
        c.close()
    return n


def merge(store: Path) -> None:
    deltas = sorted(p for p in (store / "deltas").glob("*") if p.is_dir())
    if not deltas:
        print("no deltas to merge")
        return
    version = store / "versions" / time.strftime("%Y%m%d_%H%M%S")
    current = store / "current"
    if current.exists():
        shutil.copytree(current.resolve(), version)
    responses, latency = _open(version)
    for delta in deltas:
        d_responses, d_latency = _open(delta)
        added = 0
        for k in d_responses:
            if k not in responses:
                responses.set(k, d_responses.get(k))
                added += 1
        for k in d_latency:
            if k not in latency:
                latency.set(k, d_latency.get(k))
        d_responses.close()
        d_latency.close()
        print(f"{delta.name}: {added} new entries")
    responses.close()
    latency.close()
    link = store / "current.new"
    link.unlink(missing_ok=True)
    link.symlink_to(version.relative_to(store))
    os.replace(link, current)
    (store / "merged").mkdir(exist_ok=True)
    for delta in deltas:
        shutil.move(str(delta), store / "merged" / delta.name)
    for old in sorted((store / "versions").iterdir())[:-KEEP_VERSIONS]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"current -> {version.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["merge"])
    ap.add_argument("--store", type=Path, default=os.environ.get("DSPY_CACHE_STORE"))
    a = ap.parse_args(argv)
    if a.store is None:
        ap.error("--store or DSPY_CACHE_STORE is required")
    merge(a.store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
