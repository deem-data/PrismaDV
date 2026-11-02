"""Shut a Spark session down so the Python process can exit.
"""
from __future__ import annotations

from typing import Any


def close_spark_session(spark_session: Any) -> None:
    if spark_session is None:
        return
    try:
        spark_session.sparkContext._gateway.close()
    finally:
        spark_session.stop()
