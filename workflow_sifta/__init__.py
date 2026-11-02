"""SIFTA and its GEPA baselines.
"""
import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_cache_dir = Path(tempfile.mkdtemp(prefix="dspy_cache_"))
os.environ["DSPY_CACHEDIR"] = str(_cache_dir)
_store = Path(os.environ["DSPY_CACHE_STORE"]) if os.environ.get("DSPY_CACHE_STORE") else None
_seed_keys = None
if _store is not None:
    from workflow_sifta.runtime import cache_store

    _seed_keys = cache_store.seed_local(_store, _cache_dir)
    print(f"[cache] {len(_seed_keys)} entries copied from {_store}/current", file=sys.stderr)

import dspy  # noqa: E402

dspy.configure_cache(enable_disk_cache=_store is not None, enable_memory_cache=True, disk_cache_dir=str(_cache_dir))

from workflow_sifta.runtime.usage import alias_cache_keys  # noqa: E402

alias_cache_keys(dspy.cache)

_finished = False


def _finish() -> None:
    """Save the entries this run added to the shared store (when there is one) and remove the local cache."""
    global _finished
    if _finished:
        return
    _finished = True
    if _store is not None:
        try:
            from workflow_sifta.runtime.usage import close_latency_store

            close_latency_store()
            if getattr(dspy.cache, "disk_cache", None) is not None:
                dspy.cache.disk_cache.close()
            name = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}_{time.strftime('%Y%m%d_%H%M%S')}"
            n = cache_store.save_delta(_store, _cache_dir, _seed_keys, name)
            print(f"[cache] {n} new entries saved to {_store}/deltas/{name}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 -- a failed save must not hide the run's own result
            print(f"[cache] saving new entries failed: {e!r}", file=sys.stderr)
    shutil.rmtree(_cache_dir, ignore_errors=True)


atexit.register(_finish)


def hard_exit(code: int = 0) -> None:
    """End the process once a run's outputs are written. PyDeequ's py4j callback thread is not a daemon and keeps a
    finished process alive (workflow_prismadv/utils/spark_cleanup.py), so the drivers end with os._exit."""
    _finish()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _exit_on_uncaught(exc_type, exc, tb):
    # The same thread keeps a crashed driver alive too: print the traceback and end the process.
    sys.__excepthook__(exc_type, exc, tb)
    hard_exit(1)


sys.excepthook = _exit_on_uncaught
