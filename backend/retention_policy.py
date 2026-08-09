"""
retention_policy.py — Week 2: automatic evidence retention/purge.

Runs a background daemon thread that periodically deletes evidence clips
older than a configurable retention window, preventing unbounded disk
growth in the evidence vault.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

DEFAULT_EVIDENCE_DIR = "evidence_vault"
DEFAULT_RETENTION_SECONDS = 86400  # 24 hours
DEFAULT_CHECK_INTERVAL = 3600      # scan every hour


def _purge_old_files(directory: Path, retention_seconds: int) -> int:
    """Delete files older than retention_seconds; return count removed."""
    cutoff = time.time() - retention_seconds
    removed = 0
    if not directory.is_dir():
        return 0
    for entry in directory.iterdir():
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                os.remove(entry)
                removed += 1
        except OSError:
            continue
    return removed


def start_auto_purge_thread(
    directory: str | Path = DEFAULT_EVIDENCE_DIR,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    check_interval: int = DEFAULT_CHECK_INTERVAL,
) -> threading.Thread:
    """
    Start a daemon thread that scans `directory` and deletes files older
    than `retention_seconds`, re-scanning every `check_interval` seconds.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def _loop() -> None:
        while True:
            removed = _purge_old_files(directory, retention_seconds)
            if removed:
                print(
                    f"[retention] purged {removed} expired file(s) from {directory}",
                    flush=True,
                )
            time.sleep(max(1, int(check_interval)))

    thread = threading.Thread(target=_loop, daemon=True, name="auto-purge")
    thread.start()
    return thread
