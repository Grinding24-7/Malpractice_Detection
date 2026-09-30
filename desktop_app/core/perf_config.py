"""
perf_config.py — Centralised performance tuning for the inference pipeline.

Every hot-path constant lives here so the accuracy/latency trade-off can be
tuned in one place without hunting through the pipeline code.

Rationale for the defaults
--------------------------
The target deployment is CCTV / proctoring, where subjects are typically
*far from the camera*.  Shrinking the YOLO input resolution would speed up
inference but meaningfully hurts recall on small or distant people, so
POSE_IMGSZ stays at the model's native 640.  The latency budget is instead
reclaimed from work that is redundant rather than inaccurate:

  * Object detection is a *second* YOLO pass costing as much as the first.
    It only needs to catch a phone or book, not a precise box, so it runs on
    a reduced input size and is throttled to every Nth anomalous frame.
  * Inference is rate-limited to a target FPS.  The camera buffer is a lossy
    single-frame deque, so surplus frames are dropped regardless; capping
    the rate returns CPU to the UI thread instead of spinning at 100%.
  * MediaPipe runs on per-person crops, so only the crop is converted to
    RGB rather than the whole frame.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# YOLO pose detection
# ---------------------------------------------------------------------------

# Input size for YOLO11n-pose.  640 is the model's native training size.
# Lowering it speeds up inference but hurts recall on distant CCTV subjects.
POSE_IMGSZ: int = int(os.environ.get("IP_POSE_IMGSZ", "640"))

# Cap on persons reported per frame.  Exam rooms rarely exceed a handful of
# students; a low cap keeps post-processing (and MediaPipe crops) bounded.
POSE_MAX_DET: int = int(os.environ.get("IP_POSE_MAX_DET", "8"))

# ---------------------------------------------------------------------------
# Object detection (phone / book / laptop)
# ---------------------------------------------------------------------------

# Which weights to load, and where to look for them.
#
# COCO-80 cannot express the objects that actually matter in a written exam:
# paper, pen, pencil, notebook, exam sheet and desk are all absent, so a
# "no object detected" result from the default model is not evidence that the
# desk was clear.  That gap is a property of the label set, not of the
# threshold -- probing the stock clip at conf 0.01 still yields only "person".
#
# A custom desk model is therefore preferred when one is present, and its own
# class names are used directly, so training one requires no code change.
# Measured on the stock clip, yolo11m was 3.6x the latency of yolo11n
# (63 ms vs 17 ms) for zero additional desk detections, so the larger stock
# model is NOT used; see tools/bench_object_models.py.
OBJ_CUSTOM_MODEL_FILE: str = os.environ.get(
    "IP_OBJ_MODEL_CUSTOM", "desk_objects.pt"
)
OBJ_FALLBACK_MODEL_FILE: str = os.environ.get("IP_OBJ_MODEL_FALLBACK", "yolo11n.pt")

# Directories searched for the weights, in priority order.  A custom model is
# looked up in all of them before falling back to the stock COCO weights.
OBJ_MODEL_DIRS: tuple[str, ...] = (
    os.environ.get("IP_OBJ_MODEL_DIR", "backend"),
    "models/desk",
    "backend",
)

# Desk crops are small, so a 640 input upscales them and wastes cycles.
# 416 is ample for classifying "is there a phone on this desk".
OBJ_IMGSZ: int = int(os.environ.get("IP_OBJ_IMGSZ", "416"))

# Minimum confidence before a detection is reported.
OBJ_CONF_THRESHOLD: float = float(os.environ.get("IP_OBJ_CONF", "0.35"))

# Run object detection on at most 1 in N anomalous frames.  A sustained
# anomaly (e.g. 30 s of head turned) would otherwise pay the extra model
# pass on every single frame.  Detections persist between passes, so throttling
# costs latency, not recall.
OBJ_DETECT_EVERY_N_FRAMES: int = int(os.environ.get("IP_OBJ_EVERY_N", "8"))

# ── Desk region ──────────────────────────────────────────────────────────
# The old search region was the person's own bounding box, lower 60%: 20% of
# the frame, which excluded everything to the left and right of the torso
# (measured 27% and 33% respectively).  A phone placed beside the student, a
# book set down to one side, or anything on a shared desk was therefore
# unreachable by construction.  The region now runs the full width of the
# frame, from a fixed height down to the bottom edge, so the desk in front of
# the student is covered.
OBJ_REGION_TOP_RATIO: float = float(
    os.environ.get("IP_OBJ_REGION_TOP", "0.45")
)
# Retained only as an upper bound on how much the region narrows for a very
# wide scene; 1.0 disables narrowing entirely.
OBJ_LATERAL_SHOULDER_WIDTHS: float = float(
    os.environ.get("IP_OBJ_LATERAL_SW", "1.5")
)

# ── Object persistence ───────────────────────────────────────────────────
# Object detection is throttled, so a detection is carried forward rather than
# vanishing between passes.  Entries older than this are dropped.  At the
# 1-in-8 throttle and 10 inference FPS this is ~1.3 s of memory.
OBJ_PERSIST_FRAMES: int = int(os.environ.get("IP_OBJ_PERSIST", "40"))

# ── Track association ────────────────────────────────────────────────────
# How a fresh detection is matched to an existing track.  Matching on the
# horizontal centre alone was wrong twice over: it merged two same-class
# objects at the same x but different heights (a phone in a hand and a book
# on the desk), and it let two fresh detections claim one track, silently
# losing a second object.  A candidate must now agree in BOTH axes, or
# overlap the stored box outright.
#
# A detection matches when either:
#   * box overlap (IoU) reaches OBJ_TRACK_MIN_IOU, which handles the common
#     case of a static object detected again on a later throttled pass; or
#   * both centre shifts are within the tolerance below, which handles a
#     small shift of a jittering box or a slowly moved object.
#
# Tolerances are fractions of the frame, so they hold at any resolution:
# a phone sliding a few centres in 0.8 s stays the same object, while a
# second phone 30 cm away does not.
OBJ_TRACK_MIN_IOU: float = float(os.environ.get("IP_OBJ_TRACK_IOU", "0.05"))
OBJ_TRACK_MAX_SHIFT_X: float = float(os.environ.get("IP_OBJ_TRACK_DX", "0.15"))
OBJ_TRACK_MAX_SHIFT_Y: float = float(os.environ.get("IP_OBJ_TRACK_DY", "0.15"))

# ── Person-object spatial relations ──────────────────────────────────────
# An object existing is not misconduct.  What matters is whether the student
# can reach it and is plausibly using it, so every detection is classified
# against the nearest person before it can escalate anything.
#
# "in_hand":   box centre falls within this multiple of shoulder width of a
#              wrist landmark.
OBJ_IN_HAND_SHOULDER_WIDTHS: float = float(
    os.environ.get("IP_OBJ_IN_HAND_SW", "0.5")
)
# "reachable": vertically above the wrist and horizontally aligned.  Uses the
# same shoulder-width normalisation, so it holds at any resolution.
OBJ_REACHABLE_SHOULDER_WIDTHS: float = float(
    os.environ.get("IP_OBJ_REACHABLE_SW", "1.2")
)

# A phone/book/laptop only escalates when its spatial relation supports use.
# Without this, a laptop belonging to the proctor on the far side of the room
# would be reported as the student using a device.
OBJ_REQUIRE_SPATIAL_AGREEMENT: bool = os.environ.get(
    "IP_OBJ_REQUIRE_SPATIAL", "1"
) not in ("0", "false", "False")

# Confidence floor for the spatial gate.  Spatial agreement is necessary but
# not sufficient, so a weak detection cannot escalate on position alone.
OBJ_SPATIAL_MIN_CONFIDENCE: float = float(
    os.environ.get("IP_OBJ_SPATIAL_CONF", "0.50")
)

# Which labels escalate to which verdict, once the spatial gate is cleared,
# and the confidence the published finding is reported at.
#
# This is explicit rather than a label test in the escalation code, because a
# custom desk model names 13 classes and each one needs a decision:
#   * devices escalate to "Unusual Object";
#   * a book escalates to "Note Peeking";
#   * paper, pens, calculators and the rest deliberately do NOT escalate --
#     holding exam paper, writing, or using an allowed calculator is normal
#     exam behaviour.  The false-accusation cost of escalating those is far
#     higher than the missed-cheating cost, because they are present in every
#     exam frame.
#
# The reported floor sits above ANOMALY_MIN_MARGIN so an object-driven
# finding cannot be dropped by the gate it just cleared.
OBJ_ESCALATE_TO: dict[str, tuple[str, float]] = {
    "cell_phone": ("P38", 0.80),
    "phone": ("P38", 0.80),
    "laptop": ("P38", 0.80),
    "tablet": ("P38", 0.80),
    "headphones": ("P38", 0.80),
    "book": ("P13", 0.75),
}

# ---------------------------------------------------------------------------
# MediaPipe PoseLandmarker
# ---------------------------------------------------------------------------

# Model variant.  Measured on this workload (per-person desk crop, CPU):
#     heavy  ~89 ms   mean visibility 0.40
#     full   ~46 ms   mean visibility 0.43   <-- default: ~2x faster, equal recall
#     lite   ~37 ms   mean visibility 0.45
# "heavy" buys nothing detectable here and costs ~2x latency, so "full" is
# the default.  Override to "heavy" for maximum model capacity.
MP_MODEL_VARIANT: str = os.environ.get("IP_MP_MODEL", "full")

# Maximum poses MediaPipe will return per crop.  The classifier reads only
# pose_landmarks[0] for each person box, so 1 is sufficient.
MP_NUM_POSES: int = int(os.environ.get("IP_MP_NUM_POSES", "1"))

# ---------------------------------------------------------------------------
# Inference scheduling
# ---------------------------------------------------------------------------

# Target inference rate.  0 disables the cap (runs flat out).
# 10 FPS gives comfortable headroom for anomaly detection while leaving the
# UI thread the CPU it needs for smooth 60 FPS rendering.
TARGET_INFERENCE_FPS: float = float(os.environ.get("IP_TARGET_FPS", "10.0"))

# Warn when a single classify() call exceeds this.  Set above the realistic
# ceiling so the console is not spammed on every frame.
WARN_THRESHOLD_MS: float = float(os.environ.get("IP_WARN_MS", "120.0"))

# Only emit the slow-frame warning once per this many frames.
WARN_RATE_LIMIT_FRAMES: int = 30

# ---------------------------------------------------------------------------
# Head pitch thresholds (P06 / P07 / P18)
# ---------------------------------------------------------------------------
# These are expressed as a fraction of torso length (nose -> hip), NOT as
# absolute pixels.  Absolute pixel thresholds are meaningless for CCTV: the
# same student produces a 240 px gap on a 720p webcam and a 60 px gap on a
# 4K ceiling mount.  Measured on a normally-seated subject at 720p, the
# nose-to-shoulder gap is ~-0.28 torso lengths, so a threshold anywhere near
# -40 px flags 100% of frames as "Head Up".
#
# Sign convention: negative => nose above the shoulder line (head up),
#                   positive => nose below the shoulder line (head down).
#
# NOTE: the "head up" and "head down" boundaries are deliberately conservative
# to avoid false accusations.  They should be re-calibrated against labelled
# looking-up / looking-down footage for your own camera geometry.

# Head Down (P06): nose has dropped below the shoulder line.
HEAD_DOWN_RATIO: float = float(os.environ.get("IP_HEAD_DOWN_RATIO", "0.05"))

# Head Up (P07): nose raised well clear of the shoulders.
HEAD_UP_RATIO: float = float(os.environ.get("IP_HEAD_UP_RATIO", "-0.40"))

# Head on desk (P18): nose far below the shoulder line.  Must be more extreme
# than HEAD_DOWN_RATIO, and is evaluated first so it is not shadowed by P06.
HEAD_ON_DESK_RATIO: float = float(os.environ.get("IP_HEAD_DESK_RATIO", "0.18"))

# Pitch rules are skipped when the torso is too short to normalise against
# (tiny / far-away subjects where MediaPipe extrapolates badly).
MIN_TORSO_PX: float = float(os.environ.get("IP_MIN_TORSO_PX", "30.0"))

# ---------------------------------------------------------------------------
# Image-plane scale reference
# ---------------------------------------------------------------------------
# Shoulder width is used as the scale reference for horizontal / vertical
# offsets.  It is chosen over torso length because MediaPipe reliably reports
# the shoulders (visibility ~0.9) while hips are frequently extrapolated well
# outside the frame (visibility ~0.0), which corrupts any hip-derived scale.
#
# Measured live at 720p: shoulder width 462 px (437-486, i.e. +/-6% drift).
MIN_SCALE_PX: float = float(os.environ.get("IP_MIN_SCALE_PX", "30.0"))

# ── Calibrated against live measurements ─────────────────────────────────
# Body turn (P10/P11) is NOT thresholded on the raw ratio.  The old absolute
# pixel test fired on 18% of a normal subject; normalising it to shoulder
# widths did not fix that, because the ratio is self-scaling — turning the
# torso foreshortens the shoulders, so the denominator shrinks exactly as the
# numerator grows.  Measured on a known-normal clip the ratio spanned
# -0.86..+0.56 (p05..p95) and the false-positive rate barely moved as the cut
# was raised: 0.20 -> 41.7%, 0.50 -> 33.3%.  So the rule fires on a robust
# z-score against the student's own running mean, with this many robust
# sigmas required.  5.0 is deliberately strict; a real torso turn is far
# outside a still student's normal band.
BODY_TURN_Z: float = float(os.environ.get("IP_BODY_TURN_Z", "5.0"))

# ── Normalised from the original pixel thresholds, NOT yet calibrated ─────
# These preserve the original author's intended sensitivity (30 px / 462 px
# typical shoulder width) while removing the resolution dependence.  Re-tune
# them against labelled footage before relying on them.
RAISE_HAND_RATIO: float = float(os.environ.get("IP_RAISE_HAND_RATIO", "0.07"))
HAND_UNDER_DESK_RATIO: float = float(
    os.environ.get("IP_HAND_UNDER_DESK_RATIO", "0.07")
)

# ── Forward hunch / backward lean (P32 / P33) ───────────────────────────
# The PnP pitch estimate is jittery: on a normally-seated student the measured
# pitch was median 21 deg, p95 31 deg, straddling the old 25 deg cut, so
# ~11% of frames were reported as "Forward Hunch".  The deviation is now taken
# from a per-student running baseline rather than from absolute zero, and a
# much larger excursion is required before reporting.
PITCH_DEVIATION_RATIO: float = float(
    os.environ.get("IP_PITCH_DEV_RATIO", "1.0")
)
HUNCH_PITCH_DEG: float = float(os.environ.get("IP_HUNCH_DEG", "35.0"))
LEAN_BACK_PITCH_DEG: float = float(os.environ.get("IP_LEAN_BACK_DEG", "-35.0"))

# ---------------------------------------------------------------------------
# Temporal smoothing / confidence gate
# ---------------------------------------------------------------------------
# Single-frame threshold crossings used to be reported immediately, so one
# noisy landmark frame became a proctoring event.  On a known-normal clip this
# produced a 50% false-positive rate.  An anomaly must now persist for
# ANOMALY_MIN_STREAK_FRAMES consecutive frames before it is reported, and it
# must clear ANOMALY_MIN_MARGIN to be published at all.

# Frames to withhold any verdict for after the classifier starts.  The models
# are still warming, the student may still be sitting down or reaching for the
# keyboard, and the pitch/turn baselines have not been established.  At 10
# inference FPS, 30 frames is ~3 s.  This matters most for "Absent Seat",
# which bypasses the gate because presence is treated as a fact about the
# room: without warmup, frame 1 accuses a student who has simply not settled
# yet, which is the single worst failure this system can produce.
WARMUP_FRAMES: int = int(os.environ.get("IP_WARMUP_FRAMES", "30"))

# ── Frame luminance validation ───────────────────────────────────────────
# A camera that is unplugged, lens-capped or fed a corrupt file keeps
# returning structurally valid frames: right dimensions, right dtype, all
# black.  Downstream that is indistinguishable from an empty room, so the
# classifier would report "Absent Seat" and the room would look proctored
# while nothing is actually being seen.  Both the capture watchdog and the
# classifier check frames against this, so the threshold lives here rather
# than being duplicated.
#
# 2.0 was chosen from measurements: a real exam clip averages 153.7, a
# genuinely dim room still measures well above 5, and the dead 1-hour clip
# measures 1.6.
BLACK_LUMA_THRESHOLD: float = float(
    os.environ.get("IP_BLACK_LUMA", "2.0")
)

# Consecutive frames the same pose id must hold before it is reported.
# At 10 inference FPS this is ~1.5 s of sustained behaviour.
ANOMALY_MIN_STREAK_FRAMES: int = int(
    os.environ.get("IP_ANOMALY_STREAK", "12")
)

# Poses reported immediately, regardless of streak or margin.  These are
# presence/absence facts that must not be delayed or suppressed: a missing
# student or an extra person in frame is the most serious finding there is.
IMMEDIATE_POSES: frozenset[str] = frozenset(
    {"P16", "P17", "P35", "P39"}
)

# Sanity floor, NOT a tuned filter.  The per-rule confidences in
# pose_classifier.py are hand-authored estimates, not calibrated
# probabilities, and they range from 0.65 (P20) to 0.95 (P16).  Setting this
# above 0.65 does not select "strong" findings -- it makes the low-authoring
# rules unreachable (a 0.80 margin silently disabled P07, P08, P09, P19, P20,
# P21, P22, P30, P32, P33 and P36).  So it sits just below the lowest authored
# value, where it still catches a degenerate 0.0/NaN verdict, while the streak
# gate does the actual noise rejection.
#
# Run `python tools/audit_rule_confidence.py` after editing any rule to confirm
# nothing has become unreachable.
ANOMALY_MIN_MARGIN: float = float(os.environ.get("IP_ANOMALY_MARGIN", "0.60"))


def describe() -> str:
    """Return a one-line summary of the active tuning (for startup logs).

    The desk-object settings are included because they are the ones most
    often adjusted per deployment (resolution, throttle, spatial gate,
    which labels escalate), and an operator reading the log should be able
    to see which of them are in force without opening this file.
    """
    return (
        f"perf: pose_imgsz={POSE_IMGSZ} obj_imgsz={OBJ_IMGSZ} "
        f"obj_every_n={OBJ_DETECT_EVERY_N_FRAMES} "
        f"obj_conf={OBJ_CONF_THRESHOLD} "
        f"region_top={OBJ_REGION_TOP_RATIO} "
        f"persist={OBJ_PERSIST_FRAMES} "
        f"track=(iou {OBJ_TRACK_MIN_IOU} dx {OBJ_TRACK_MAX_SHIFT_X} "
        f"dy {OBJ_TRACK_MAX_SHIFT_Y}) "
        f"spatial_gate={OBJ_REQUIRE_SPATIAL_AGREEMENT}"
        f"@{OBJ_SPATIAL_MIN_CONFIDENCE} "
        f"escalations={len(OBJ_ESCALATE_TO)} "
        f"obj_model={OBJ_CUSTOM_MODEL_FILE}->{OBJ_FALLBACK_MODEL_FILE} "
        f"mp={MP_MODEL_VARIANT} target_fps={TARGET_INFERENCE_FPS} "
        f"warn_ms={WARN_THRESHOLD_MS} warmup={WARMUP_FRAMES} "
        f"streak={ANOMALY_MIN_STREAK_FRAMES} margin={ANOMALY_MIN_MARGIN}"
    )
