"""QA sweep: neutral filter, desk-object fusion thread, and stability gate wiring.

Verifies the three live-demo safety mechanisms end to end without a camera:

1. check_neutral_alignment recognises writing / thinking / glance, and the
   classify() pipeline suppresses P14 and P31_PHONE_USABILITY into P01 with
   the neutral_suppressed flag the green P00 HUD verdict keys off.
2. The desk-object model resolves desk_objects.pt when present and falls back
   safely to COCO-80 (yolo11n.pt) when not; the concurrent DeskProximityThread
   runs throttled 416 px inference on its own QThread and never blocks the
   inference/display loop (snapshot handoff is lock-guarded and non-blocking).
3. The 12-frame streak / 0.60 margin gate constants are in force (behaviour
   covered exhaustively by tools/test_stability_gate.py).

Run:  python tools/test_neutral_and_fusion_qa.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import (  # noqa: E402
    NEUTRAL_GLANCE,
    NEUTRAL_THINKING,
    NEUTRAL_WRITING,
    NEUTRAL_LABELS,
    PoseClassifier,
    _NEUTRAL_SUPPRESSES,
    check_neutral_alignment,
    _TemporalBuffer,
)
from desktop_app.core.object_context import (  # noqa: E402
    OBJ_CLASS_MAP,
    ObjectContext,
    resolve_object_model,
)
from desktop_app.core.hud_renderer import render_hud  # noqa: E402

MP_NOSE, MP_MOUTH_LEFT, MP_MOUTH_RIGHT = 0, 9, 10
MP_L_SHOULDER, MP_R_SHOULDER = 11, 12
MP_L_ELBOW, MP_R_ELBOW = 13, 14
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_HIP, MP_R_HIP = 23, 24


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def landmarks33(
    wrist_x: float = 250.0, wrist_y: float = 500.0, mouth_y: float = 150.0
) -> list[tuple[float, float, float]]:
    """A minimal valid 33-point pose: visible shoulders, hips, wrists, mouth."""
    lms = [(250.0, 120.0, 0.9)] * 33  # nose-ish default everywhere
    lms[MP_NOSE] = (250.0, 120.0, 0.9)
    lms[3] = (230.0, 110.0, 0.9)   # left eye outer
    lms[6] = (270.0, 110.0, 0.9)   # right eye outer
    lms[MP_MOUTH_LEFT] = (240.0, mouth_y, 0.9)
    lms[MP_MOUTH_RIGHT] = (260.0, mouth_y, 0.9)
    lms[MP_L_SHOULDER] = (150.0, 200.0, 0.9)
    lms[MP_R_SHOULDER] = (350.0, 200.0, 0.9)
    lms[MP_L_ELBOW] = (160.0, 350.0, 0.9)
    lms[MP_R_ELBOW] = (340.0, 350.0, 0.9)
    lms[MP_L_WRIST] = (wrist_x, wrist_y, 0.9)
    lms[MP_R_WRIST] = (wrist_x + 60.0, wrist_y, 0.9)
    lms[MP_L_HIP] = (170.0, 600.0, 0.9)
    lms[MP_R_HIP] = (330.0, 600.0, 0.9)
    return lms


def stub_classifier(
    classifier: PoseClassifier,
    raw_verdict: tuple[str, float],
    pitch: float,
    landmarks_fn,
) -> None:
    """Pin every heavyweight input so classify() measures the logic under test."""
    classifier._evaluate_rules = lambda *a, **k: raw_verdict
    classifier._yolo_detect = lambda f: ([(100, 50, 400, 700)],
                                         [np.zeros((17, 3))])
    classifier._mediapipe_landmarks = lambda f, b: [landmarks_fn()]
    classifier._detect_objects = lambda f, b: []
    classifier._head_pose_estimator.estimate = (
        lambda lms, image_shape: {
            "yaw": 0.0, "pitch": pitch, "roll": 0.0,
            "gaze_vector": (0, 0, 0), "anomalies": [],
        }
    )
    classifier._head_pose_estimator.get_gaze_endpoint = lambda *a, **k: (0, 0)


FRAME = np.full((720, 1280, 3), 128, dtype=np.uint8)
BBOX = [100, 50, 400, 700]


def main() -> int:
    ok = True

    # ── 1. check_neutral_alignment unit behaviour ──────────────────────
    print("1. check_neutral_alignment recognises the three benign states")

    temporal = _TemporalBuffer()
    # Feed pen-stroke oscillation so the writing test has history.
    for i in range(6):
        wx = 250.0 + (4.0 if i % 2 else 0.0)
        check_neutral_alignment(
            landmarks33(wx), 0.0, -25.0, 0.0, [], bbox=BBOX, temporal=temporal
        )
    state = check_neutral_alignment(
        landmarks33(252.0), 0.0, -25.0, 0.0, [], bbox=BBOX, temporal=temporal
    )
    ok &= check("writing posture -> NEUTRAL_WRITING",
                state == NEUTRAL_WRITING, f"got {state!r}")

    state = check_neutral_alignment(
        landmarks33(wrist_x=255.0, wrist_y=310.0, mouth_y=300.0),
        0.0, 0.0, 0.0, [], bbox=BBOX, temporal=_TemporalBuffer(),
    )
    ok &= check("wrist at cheek, no device -> NEUTRAL_THINKING",
                state == NEUTRAL_THINKING, f"got {state!r}")

    state = check_neutral_alignment(
        landmarks33(wrist_x=255.0, wrist_y=310.0, mouth_y=300.0),
        0.0, 0.0, 0.0,
        [{"label": "cell_phone", "confidence": 0.9, "box": [240, 300, 270, 340]}],
        bbox=BBOX, temporal=_TemporalBuffer(),
    )
    ok &= check("device at the cheek wrist kills NEUTRAL_THINKING",
                state is None, f"got {state!r}")

    fresh = _TemporalBuffer()
    state = check_neutral_alignment(
        landmarks33(wrist_x=120.0, wrist_y=300.0),
        0.0, 20.0, 0.0, [], bbox=BBOX, temporal=fresh,
    )
    ok &= check("brief upward pitch -> NEUTRAL_GLANCE",
                state == NEUTRAL_GLANCE, f"got {state!r}")

    state = check_neutral_alignment(
        landmarks33(wrist_x=120.0, wrist_y=300.0),
        0.0, 0.0, 0.0, [], bbox=BBOX, temporal=fresh,
    )
    ok &= check("leaving the glance band clears the state",
                state is None, f"got {state!r}")

    print()
    print("2. suppression contract: only P14 and P31_PHONE_USABILITY")
    ok &= check("_NEUTRAL_SUPPRESSES is exactly {P14, P31_PHONE_USABILITY}",
                _NEUTRAL_SUPPRESSES == {"P14", "P31_PHONE_USABILITY"},
                f"got {sorted(_NEUTRAL_SUPPRESSES)}")
    ok &= check("presence verdicts are never suppressible",
                not (_NEUTRAL_SUPPRESSES & {"P16", "P35"}))
    ok &= check("every neutral state has a HUD label",
                set(NEUTRAL_LABELS) == {NEUTRAL_WRITING, NEUTRAL_THINKING,
                                        NEUTRAL_GLANCE})

    # ── 3. end-to-end classify(): P14 suppressed by writing ────────────
    print()
    print("3. classify(): sustained P14 collapses to P01 while writing")
    c = PoseClassifier()
    c._frames_seen = perf_config.WARMUP_FRAMES  # past warmup
    results = []
    for i in range(15):
        wx = 250.0 + (4.0 if i % 2 else 0.0)
        stub_classifier(c, ("P14", 0.82), pitch=-25.0,
                        landmarks_fn=lambda wx=wx: landmarks33(wx))
        results.append(c.classify(FRAME))
    last = results[-1]
    ok &= check("final verdict is P01, not P14",
                last["pose_id"] == "P01", f"got {last['pose_id']!r}")
    ok &= check("neutral_state is NEUTRAL_WRITING",
                last.get("neutral_state") == NEUTRAL_WRITING,
                f"got {last.get('neutral_state')!r}")
    ok &= check("neutral_suppressed flag is set",
                bool(last.get("neutral_suppressed")))
    ok &= check("published confidence is the neutral floor (0.95)",
                abs(last["confidence"] - 0.95) < 1e-6,
                f"got {last['confidence']}")
    ok &= check("P14 never leaked on any frame",
                all(r["pose_id"] != "P14" for r in results))

    print()
    print("4. classify(): P31_PHONE_USABILITY suppressed by writing")
    c2 = PoseClassifier()
    c2._frames_seen = perf_config.WARMUP_FRAMES
    phone = {"label": "cell_phone", "confidence": 0.8,
             "box": [240, 480, 270, 520]}
    c2.set_proximity({"objects": [phone], "phone": phone,
                      "ts": time.time()})
    results = []
    for i in range(15):
        wx = 250.0 + (4.0 if i % 2 else 0.0)
        stub_classifier(c2, ("P01", 0.95), pitch=-25.0,
                        landmarks_fn=lambda wx=wx: landmarks33(wx))
        results.append(c2.classify(FRAME))
    last = results[-1]
    ok &= check("proximity snapshot fused into the verdict",
                any(r.get("phone_proximity") for r in results))
    ok &= check("final verdict is P01 with neutral_suppressed",
                last["pose_id"] == "P01"
                and bool(last.get("neutral_suppressed")),
                f"got {last['pose_id']!r} suppressed={last.get('neutral_suppressed')}")
    ok &= check("P31 never leaked on any frame",
                all(r["pose_id"] != "P31_PHONE_USABILITY" for r in results))

    print()
    print("5. control: P14 without a neutral state still publishes (gate ok)")
    c3 = PoseClassifier()
    c3._frames_seen = perf_config.WARMUP_FRAMES
    results = []
    for _ in range(15):
        stub_classifier(c3, ("P14", 0.82), pitch=0.0,
                        landmarks_fn=lambda: landmarks33(120.0, 300.0))
        results.append(c3.classify(FRAME))
    fired = [i for i, r in enumerate(results) if r["pose_id"] == "P14"]
    ok &= check("P14 fires once its streak is earned",
                bool(fired) and fired[0] == perf_config.ANOMALY_MIN_STREAK_FRAMES - 1,
                f"first fire at frame {fired[0] if fired else 'never'}")

    print()
    print("6. HUD: neutral result renders the green P00 path without error")
    try:
        hud_frame = render_hud(FRAME.copy(), last, 30.0, 10.0)
        ok &= check("render_hud completed on a neutral result",
                    hud_frame is not None and hud_frame.shape == FRAME.shape)
    except Exception as exc:  # pragma: no cover - failure path
        ok &= check("render_hud completed on a neutral result", False, str(exc))
    ok &= check("HUD inputs are present (pose_id P01 + neutral_state)",
                last["pose_id"] == "P01" and bool(last["neutral_state"])
                and bool(last["neutral_label"]))

    # ── 7. desk-object model fallback ──────────────────────────────────
    print()
    print("7. desk-object model: custom weights win, COCO-80 is the fallback")
    resolved = resolve_object_model(
        perf_config.OBJ_CUSTOM_MODEL_FILE, perf_config.OBJ_MODEL_DIRS
    )
    fallback = resolve_object_model(
        perf_config.OBJ_FALLBACK_MODEL_FILE, perf_config.OBJ_MODEL_DIRS
    )
    if resolved is not None:
        ok &= check("desk_objects.pt present -> custom model wins",
                    resolved[1] is True, f"got {resolved}")
    else:
        ok &= check("desk_objects.pt absent -> resolves to None, no crash",
                    resolved is None)
    ok &= check("COCO fallback weights resolve",
                fallback is not None and fallback[1] is False
                and fallback[0].name.endswith("yolo11n.pt"),
                f"got {fallback}")
    ok &= check("fallback label set is COCO {book,cell_phone,laptop}",
                set(OBJ_CLASS_MAP.values()) == {"book", "cell_phone", "laptop"})

    ctx = ObjectContext()
    ctx.load(
        perf_config.OBJ_CUSTOM_MODEL_FILE,
        perf_config.OBJ_FALLBACK_MODEL_FILE,
        perf_config.OBJ_MODEL_DIRS,
    )
    ok &= check("ObjectContext.load() succeeds (no exception)", True)
    if resolved is None:
        ok &= check("loaded model is flagged NOT custom (COCO path)",
                    ctx.is_custom is False)
        ok &= check("class map fell back to COCO subset",
                    set(ctx.class_map.values()) == {"book", "cell_phone", "laptop"})
    ok &= check("proximity throttle is 416 px / 1-in-3 frames",
                perf_config.OBJ_IMGSZ == 416
                and perf_config.PROX_EVERY_N_FRAMES == 3,
                f"imgsz={perf_config.OBJ_IMGSZ} every_n={perf_config.PROX_EVERY_N_FRAMES}")
    ok &= check("phone thresholds are 120 px / conf > 0.6",
                perf_config.PHONE_WRIST_DIST_PX == 120.0
                and perf_config.PHONE_MIN_CONF == 0.6)

    # ── 8. DeskProximityThread: non-blocking handoff ───────────────────
    print()
    print("8. DeskProximityThread runs off the display loop and hands off safely")
    from PyQt6.QtCore import QCoreApplication
    from desktop_app.core.inference_worker import DeskProximityThread

    app = QCoreApplication.instance() or QCoreApplication([])

    class _StubCamera:
        frame: np.ndarray | None = None

        def get_latest_frame(self):
            return self.frame

    cam = _StubCamera()
    prox = DeskProximityThread(cam)  # type: ignore[arg-type]
    ok &= check("initial snapshot is a valid empty dict",
                prox.snapshot() == {"objects": [], "phone": None, "ts": 0.0}
                or (set(prox.snapshot()) == {"objects", "phone", "ts"}
                    and prox.snapshot()["objects"] == []))

    t0 = time.monotonic()
    prox.start()
    started = prox.wait(8000)  # model load + first loop, bounded
    ok &= check("thread starts, loads weights, and is running",
                prox.isRunning(), f"wait_ok={started}")
    # snapshot() must never block the caller, even mid-detect.
    t0 = time.monotonic()
    snap = prox.snapshot()
    snap_ms = (time.monotonic() - t0) * 1000.0
    ok &= check("snapshot() handoff is non-blocking",
                snap_ms < 50.0, f"{snap_ms:.2f} ms")

    cam.frame = FRAME
    prox.update_candidate(BBOX, ((250.0, 500.0),))
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        s = prox.snapshot()
        if s["ts"] > 0.0:
            break
        time.sleep(0.05)
    s = prox.snapshot()
    ok &= check("snapshot published after a real frame",
                s["ts"] > 0.0 and isinstance(s["objects"], list),
                f"objects={len(s['objects'])} phone={s['phone']}")

    t0 = time.monotonic()
    prox.stop()
    stop_ms = (time.monotonic() - t0) * 1000.0
    ok &= check("stop() joins promptly (no deadlock)",
                not prox.isRunning() and stop_ms < 5500,
                f"{stop_ms:.0f} ms")
    app.processEvents()

    # The classifier consumes the same snapshot shape without touching the thread.
    c4 = PoseClassifier()
    c4.set_proximity(s)
    ok &= check("classifier accepts the thread's snapshot shape",
                c4._proximity is s or c4._proximity == s)

    # ── 9. stability gate constants in force ───────────────────────────
    print()
    print("9. stability gate constants (behaviour: tools/test_stability_gate.py)")
    ok &= check("ANOMALY_MIN_STREAK_FRAMES == 12",
                perf_config.ANOMALY_MIN_STREAK_FRAMES == 12,
                f"got {perf_config.ANOMALY_MIN_STREAK_FRAMES}")
    ok &= check("ANOMALY_MIN_MARGIN == 0.60",
                perf_config.ANOMALY_MIN_MARGIN == 0.60,
                f"got {perf_config.ANOMALY_MIN_MARGIN}")
    ok &= check("WARMUP_FRAMES == 30", perf_config.WARMUP_FRAMES == 30)
    gate = PoseClassifier()
    g, _, supp = gate._apply_stability_gate("P22", 0.75)
    ok &= check("single-frame anomaly is suppressed by the gate",
                g == "P01" and supp)

    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
