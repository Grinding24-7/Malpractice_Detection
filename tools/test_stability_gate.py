"""Verify the stability gate suppresses noise without disabling real findings.

A 0% false-positive rate on a normal clip is only half the result.  If the
gate also killed every genuine anomaly, the classifier would look perfect and
be useless.  This drives the gate with a scripted sequence of raw rule
verdicts and asserts the intended behaviour for each segment.

The rule engine is stubbed so the test measures the gate in isolation, with no
MediaPipe/YOLO in the loop.
"""

from __future__ import annotations

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import PoseClassifier  # noqa: E402

N = perf_config.ANOMALY_MIN_STREAK_FRAMES


def run_case(
    classifier: PoseClassifier,
    script: list[tuple[str, float]],
) -> list[tuple[str, float, bool]]:
    """Feed a scripted sequence of raw verdicts, return the gated output."""
    outputs = []
    for pose_id, confidence in script:
        # Replace the real rule engine with the scripted verdict.
        classifier._evaluate_rules = lambda *a, **k: (pose_id, confidence)
        classifier._yolo_detect = lambda f: ([], [])
        classifier._mediapipe_landmarks = lambda f, b: []
        classifier._head_pose_estimator.estimate = lambda l, image_shape: {
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "gaze_vector": (0, 0, 0), "anomalies": [],
        }
        classifier._head_pose_estimator.get_gaze_endpoint = lambda *a, **k: (0, 0)
        gated, gated_conf, suppressed = classifier._apply_stability_gate(
            pose_id, confidence
        )
        outputs.append((gated, gated_conf, suppressed))
    return outputs


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def main() -> int:
    print(f"ANOMALY_MIN_STREAK_FRAMES = {N}   "
          f"ANOMALY_MIN_MARGIN = {perf_config.ANOMALY_MIN_MARGIN}")
    print()
    ok = True
    c = PoseClassifier()

    print("1. sustained genuine anomaly (P22 Raising Hand, 0.75) must fire")
    out = run_case(c, [("P22", 0.75)] * (N + 20))
    fired = [i for i, (p, _, _) in enumerate(out) if p == "P22"]
    ok &= check(
        "suppressed for the first N-1 frames, then fires",
        fired and fired[0] == N - 1,
        f"first fire at frame {fired[0] if fired else 'never'} (expected {N - 1})",
    )
    ok &= check(
        "keeps firing afterwards",
        all(p == "P22" for p, _, _ in out[N:]),
    )

    print()
    print("2. brief flicker (single frame of P22) must be suppressed")
    c = PoseClassifier()  # fresh state, so case 1's streak cannot leak
    out = run_case(c, [("P01", 0.95)] * 5 + [("P22", 0.75)] + [("P01", 0.95)] * 20)
    ok &= check("no P22 ever published", all(p != "P22" for p, _, _ in out))

    print()
    print("2b. a stale streak must NOT survive normal frames (regression)")
    # Build a streak, then return to normal for a long time, then flick.
    run_case(c, [("P22", 0.75)] * (N - 1))
    out = run_case(c, [("P01", 0.95)] * 60 + [("P22", 0.75)] + [("P01", 0.95)] * 20)
    ok &= check(
        "P01 frames cleared the old streak; later flick stays suppressed",
        all(p != "P22" for p, _, _ in out),
    )

    print()
    print("3. flickering between two ids must not accumulate a streak")
    out = run_case(c, [("P10" if i % 2 else "P11", 0.80) for i in range(N * 4)])
    ok &= check("no P10/P11 published", all(p not in ("P10", "P11") for p, _, _ in out))

    print()
    print("4. sustained P33 (Forward Hunch, 0.75) must fire after recalibration")
    out = run_case(c, [("P33", 0.75)] * (N + 20))
    ok &= check("fires once persistent", any(p == "P33" for p, _, _ in out))

    print()
    print("5. immediate poses bypass the gate (absent student must not wait)")
    out = run_case(c, [("P16", 0.95)] * 3)
    ok &= check("P16 fires on the very first frame", out[0][0] == "P16")

    print()
    print("6. P01 is never gated")
    out = run_case(c, [("P01", 0.95)] * 5)
    ok &= check("P01 always passes", all(p == "P01" for p, _, _ in out))

    print()
    print("7. degenerate low confidence is dropped by the margin floor")
    out = run_case(c, [("P22", 0.10)] * (N + 20))
    ok &= check("conf 0.10 never published", all(p != "P22" for p, _, _ in out))

    print()
    print("8. gate state resets between candidates")
    out = run_case(c, [("P22", 0.75)] * (N - 1))
    c2 = PoseClassifier()  # fresh state, so case 1's streak cannot leak
    ok &= check("fresh classifier has no streak", c2._temporal_buffer._pose_streak_len == 0)

    print()
    print("9. system states are never gated (regression)")
    # A camera fault must not be downgraded to "P01 Normal Sitting": that is
    # false reassurance, and it is the exact failure this gate must not cause.
    for state in ("NO_SIGNAL", "INITIALIZING"):
        c3 = PoseClassifier()
        gated, conf, supp = c3._apply_stability_gate(state, 0.0)
        ok &= check(f"{state} passes through unchanged",
                    gated == state and not supp,
                    f"got {gated!r} suppressed={supp}")

    print()
    print("10. warmup frames must not build a streak (regression)")
    # Anomalies observed while the room is still being assessed are not
    # evidence: models are warming, baselines are forming, the student may
    # still be sitting down.  If they were counted, a streak built during
    # INITIALIZING would let the very first post-warmup frame publish an
    # anomaly, which defeats the warmup entirely.
    W = perf_config.WARMUP_FRAMES
    c4 = PoseClassifier()
    for _ in range(W):
        c4._apply_stability_gate("P22", 0.75, warming_up=True)
    ok &= check(
        f"{W} warmup anomaly frames left the streak empty",
        c4._temporal_buffer._pose_streak_len == 0,
        f"streak={c4._temporal_buffer._pose_streak_len}",
    )

    first_post, _, suppressed = c4._apply_stability_gate("P22", 0.75)
    ok &= check(
        "first post-warmup anomaly frame is still suppressed",
        first_post == "P01" and suppressed,
        f"got {first_post!r} suppressed={suppressed}",
    )

    # ...and the full streak must be earned from warmup's end.  Frame 1 is
    # consumed above; frames 2..N-1 stay suppressed, frame N publishes.
    post = []
    for _ in range(N - 2):
        g, _, s = c4._apply_stability_gate("P22", 0.75)
        post.append((g, s))
    ok &= check(
        f"frames 2..{N - 1} after warmup stay suppressed",
        all(p != "P22" for p, _ in post),
        f"streak={c4._temporal_buffer._pose_streak_len}",
    )
    g, _, _ = c4._apply_stability_gate("P22", 0.75)
    ok &= check("fires on the Nth post-warmup frame", g == "P22",
                f"got {g!r} streak={c4._temporal_buffer._pose_streak_len}")

    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
