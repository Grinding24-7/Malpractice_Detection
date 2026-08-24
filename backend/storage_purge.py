"""
storage_purge.py — Week 7: Background storage purging daemon.

Responsibilities:
    1. Monitor total disk usage in evidence_vault/.
    2. If usage exceeds MAX_STORAGE_GB, auto-delete oldest LOW_SEVERITY clips.
    3. Enforce age-based retention: purge unreviewed clips older than RETENTION_DAYS.
    4. Preserve bookmarked/flagged clips regardless of age or disk pressure.

Runs as a daemon thread started during FastAPI lifespan.  Never blocks
the real-time video stream.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
VAULT_DIR = BACKEND_DIR / "evidence_vault"
CLIPS_DIR = VAULT_DIR / "clips"
METADATA_DIR = VAULT_DIR / "metadata"

MAX_STORAGE_GB: float = 10.0          # max vault size in GB
RETENTION_DAYS: int = 30              # purge unreviewed clips older than 30 days
CHECK_INTERVAL_MINUTES: int = 15      # scan every 15 minutes
SEVERITY_PRIORITY = {"low": 0, "medium": 1, "high": 2}  # lower = deleted first


@dataclass
class PurgeStats:
    """Summary of a single purge cycle."""
    timestamp: str
    disk_usage_gb: float
    vault_size_gb: float
    clips_deleted: int
    bytes_freed: int
    reason: str


class StoragePurgeDaemon:
    """
    Background daemon that prevents evidence vault disk exhaustion.

    Runs a periodic scan thread that:
    - Checks total vault size against MAX_STORAGE_GB
    - Deletes oldest LOW_SEVERITY clips when over limit
    - Enforces age-based retention for unreviewed clips
    - Preserves bookmarked/flagged clips

    Usage:
        daemon = StoragePurgeDaemon()
        daemon.start()
        # ... later ...
        daemon.stop()
    """

    def __init__(
        self,
        vault_dir: Path = VAULT_DIR,
        max_storage_gb: float = MAX_STORAGE_GB,
        retention_days: int = RETENTION_DAYS,
        check_interval_minutes: int = CHECK_INTERVAL_MINUTES,
    ) -> None:
        self._vault_dir = vault_dir
        self._clips_dir = vault_dir / "clips"
        self._metadata_dir = vault_dir / "metadata"
        self._max_bytes = int(max_storage_gb * 1024**3)
        self._retention_seconds = retention_days * 86400
        self._check_interval = check_interval_minutes * 60
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._purge_log: list[PurgeStats] = []

    def start(self) -> None:
        """Start the background purge daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="storage-purge"
        )
        self._thread.start()
        print(
            f"[purge] daemon started (max={self._max_bytes / 1024**3:.1f} GB, "
            f"retention={self._retention_seconds // 86400} days, "
            f"interval={self._check_interval // 60} min)",
            flush=True,
        )

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def run_once(self) -> Optional[PurgeStats]:
        """
        Execute a single purge cycle manually (for testing/admin).

        Returns the PurgeStats if a purge was performed, or None if
        no action was needed.
        """
        return self._purge_cycle()

    # -- Internal loop -------------------------------------------------------

    def _run(self) -> None:
        """Main daemon loop: sleep, then run a purge cycle."""
        while not self._stop_event.is_set():
            self._purge_cycle()
            self._stop_event.wait(timeout=self._check_interval)

    def _purge_cycle(self) -> Optional[PurgeStats]:
        """Run one full purge cycle: disk check + age retention."""
        vault_size = self._vault_size_bytes()
        disk_ok = vault_size <= self._max_bytes

        if disk_ok:
            # Even if disk is fine, enforce age-based retention
            age_purged = self._purge_by_age()
            if age_purged > 0:
                new_size = self._vault_size_bytes()
                stats = PurgeStats(
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    disk_usage_gb=new_size / 1024**3,
                    vault_size_gb=new_size / 1024**3,
                    clips_deleted=age_purged,
                    bytes_freed=vault_size - new_size,
                    reason="age_retention",
                )
                self._purge_log.append(stats)
                return stats
            return None

        # Disk pressure: delete oldest LOW_SEVERITY clips first
        freed = 0
        deleted = 0
        for clip_info in self._sorted_clips_by_severity():
            if vault_size - freed <= self._max_bytes:
                break
            if clip_info["severity"] != "low":
                continue
            if clip_info.get("bookmarked", False):
                continue
            clip_size = self._delete_clip(clip_info)
            freed += clip_size
            deleted += 1

        # If still over limit, try medium severity
        if vault_size - freed > self._max_bytes:
            for clip_info in self._sorted_clips_by_severity():
                if vault_size - freed <= self._max_bytes:
                    break
                if clip_info["severity"] != "medium":
                    continue
                if clip_info.get("bookmarked", False):
                    continue
                clip_size = self._delete_clip(clip_info)
                freed += clip_size
                deleted += 1

        new_size = vault_size - freed
        stats = PurgeStats(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            disk_usage_gb=new_size / 1024**3,
            vault_size_gb=new_size / 1024**3,
            clips_deleted=deleted,
            bytes_freed=freed,
            reason="disk_pressure",
        )
        self._purge_log.append(stats)
        print(
            f"[purge] disk pressure: freed {freed / 1024**2:.1f} MB "
            f"({deleted} clips), vault now {new_size / 1024**3:.2f} GB",
            flush=True,
        )
        return stats

    # -- Age-based retention -------------------------------------------------

    def _purge_by_age(self) -> int:
        """Delete clips older than RETENTION_DAYS that aren't bookmarked."""
        cutoff = time.time() - self._retention_seconds
        deleted = 0

        for meta_file in self._metadata_dir.glob("*.json"):
            try:
                meta = json.loads(meta_file.read_text())
                created = meta.get("timestamp", "")
                if not created:
                    continue

                # Parse ISO timestamp
                from datetime import datetime, timezone
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if dt.timestamp() > cutoff:
                        continue  # not old enough
                except ValueError:
                    continue

                # Skip bookmarked clips
                if meta.get("bookmarked", False):
                    continue

                # Skip reviewed high-severity clips
                if meta.get("reviewed") and meta.get("severity") == "high":
                    continue

                event_id = meta.get("event_id", meta_file.stem)
                self._delete_clip({"event_id": event_id})
                deleted += 1

            except (json.JSONDecodeError, OSError):
                continue

        return deleted

    # -- Helpers -------------------------------------------------------------

    def _vault_size_bytes(self) -> int:
        """Total size of all files in the vault directory."""
        total = 0
        for directory in [self._clips_dir, self._metadata_dir]:
            if not directory.exists():
                continue
            for f in directory.iterdir():
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        continue
        return total

    def _sorted_clips_by_severity(self) -> list[dict]:
        """Load all clip metadata, sorted by severity (low first) then age."""
        clips = []
        if not self._metadata_dir.exists():
            return clips

        for meta_file in self._metadata_dir.glob("*.json"):
            try:
                meta = json.loads(meta_file.read_text())
                sev = meta.get("severity", "low")
                sev_rank = SEVERITY_PRIORITY.get(sev, 0)
                clips.append({
                    "event_id": meta.get("event_id", meta_file.stem),
                    "severity": sev,
                    "severity_rank": sev_rank,
                    "timestamp": meta.get("timestamp", ""),
                    "bookmarked": meta.get("bookmarked", False),
                    "reviewed": meta.get("reviewed", False),
                })
            except (json.JSONDecodeError, OSError):
                continue

        # Sort: low severity first, then oldest first
        clips.sort(key=lambda c: (c["severity_rank"], c["timestamp"]))
        return clips

    def _delete_clip(self, clip_info: dict) -> int:
        """Delete a clip's MP4 and JSON files. Returns bytes freed."""
        event_id = clip_info.get("event_id", "")
        freed = 0

        clip_path = self._clips_dir / f"{event_id}.mp4"
        meta_path = self._metadata_dir / f"{event_id}.json"

        for path in [clip_path, meta_path]:
            if path.exists():
                try:
                    freed += path.stat().st_size
                    path.unlink()
                except OSError:
                    pass

        return freed

    # -- Query ---------------------------------------------------------------

    def get_stats(self) -> dict:
        """Current vault statistics."""
        vault_bytes = self._vault_size_bytes()
        clip_count = len(list(self._clips_dir.glob("*.mp4"))) if self._clips_dir.exists() else 0
        meta_count = len(list(self._metadata_dir.glob("*.json"))) if self._metadata_dir.exists() else 0
        return {
            "vault_size_gb": round(vault_bytes / 1024**3, 3),
            "max_storage_gb": round(self._max_bytes / 1024**3, 1),
            "clip_count": clip_count,
            "metadata_count": meta_count,
            "retention_days": self._retention_seconds // 86400,
            "check_interval_minutes": self._check_interval // 60,
            "recent_purges": [
                {
                    "timestamp": s.timestamp,
                    "clips_deleted": s.clips_deleted,
                    "bytes_freed_mb": round(s.bytes_freed / 1024**2, 2),
                    "reason": s.reason,
                }
                for s in self._purge_log[-10:]
            ],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_daemon: Optional[StoragePurgeDaemon] = None


def get_purge_daemon() -> StoragePurgeDaemon:
    global _daemon
    if _daemon is None:
        _daemon = StoragePurgeDaemon()
    return _daemon
