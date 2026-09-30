"""List every rule's pose id and confidence, flagging any that the margin gate
would make permanently unreachable.

The confidence gate in `_apply_stability_gate` drops any anomaly whose rule
confidence is below `perf_config.ANOMALY_MIN_MARGIN`.  A rule authored with a
confidence of 0.65 can therefore never be published, no matter how long it
persists.  This lists those cases so they can be re-authored deliberately
rather than discovered as a silent dead rule.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import POSE_MAP  # noqa: E402

_SOURCE = pathlib.Path(__file__).resolve().parent.parent / (
    "desktop_app/core/pose_classifier.py"
)


def main() -> int:
    text = _SOURCE.read_text(encoding="utf-8")
    # Matches  return ("P07", 0.85)
    found = re.findall(r'\(\s*"(P\d{2})"\s*,\s*([0-9]*\.?[0-9]+)\s*\)', text)

    best: dict[str, float] = {}
    for pose_id, conf in found:
        value = float(conf)
        # Keep the highest confidence each pose can be emitted with.
        if pose_id not in best or value > best[pose_id]:
            best[pose_id] = value

    margin = perf_config.ANOMALY_MIN_MARGIN
    print(f"ANOMALY_MIN_MARGIN = {margin}")
    print(f"ANOMALY_MIN_STREAK_FRAMES = {perf_config.ANOMALY_MIN_STREAK_FRAMES}")
    print()
    print(f"{'pose':<6} {'conf':>6}  {'name':<26} status")
    print("-" * 72)

    dead, gated = [], []
    for pose_id in sorted(best):
        conf = best[pose_id]
        name = POSE_MAP.get(pose_id, "?")
        if pose_id in perf_config.IMMEDIATE_POSES:
            status = "immediate (not gated)"
        elif conf < margin:
            status = "UNREACHABLE - conf below margin"
            dead.append(pose_id)
        else:
            status = "gated"
            gated.append(pose_id)

        print(f"{pose_id:<6} {conf:>6.2f}  {name:<26} {status}")

    print()
    if dead:
        print(f"PROBLEM: {len(dead)} rule(s) can never fire: {', '.join(dead)}")
        print("Each needs either a higher authored confidence or to be added")
        print("to IMMEDIATE_POSES, which should be reserved for room-level")
        print("facts (absent student, extra person).")
        return 1

    print(f"OK: all {len(gated)} gated rules are reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
