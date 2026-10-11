"""End-to-end test of the custom desk-model path with a REAL checkpoint.

tools/test_object_context.py proves class adoption from a stub with a
``names`` dict.  That is one step short of the contract that matters: a
trained desk_objects.pt must load, be recognised as custom, adopt all desk
classes (including ``exam_paper`` at id 0), win over the stock COCO
weights, and survive an actual detect() pass.

This builds a tiny untrained model with nc=len(DESK_CLASSES) in a temp
directory and runs
it through ObjectContext.load()/detect(), so the test exercises the same
code path a trained model will -- without needing labelled data or a GPU.

It also pins the contract between models/desk/desk_classes.py and
perf_config.OBJ_ESCALATE_TO: every DEVICE class must escalate, no AMBIENT
class may.  Those two files are edited in different sessions, and a desk
class added later without a decision would otherwise be silently shown but
never acted on.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import sys
import tempfile

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.object_context import (  # noqa: E402
    OBJ_CLASS_MAP, ObjectContext,
)
from models.desk.desk_classes import (  # noqa: E402
    AMBIENT_CLASSES, DESK_CLASSES, DEVICE_CLASSES,
)

STOCK_WEIGHTS = _ROOT / "backend" / "yolo11n.pt"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return condition


def build_desk_checkpoint(path: pathlib.Path) -> None:
    """Create a real (untrained) YOLO checkpoint with len(DESK_CLASSES)
    classes at *path*.

    Built from the stock architecture YAML with nc overridden, then named
    exactly as models/desk/desk_classes.py defines them -- the same layout
    train_desk_objects.py promotes to models/desk/desk_objects.pt.
    """
    import torch
    from ultralytics.nn.tasks import DetectionModel

    try:
        model = DetectionModel(
            cfg="yolo11n.yaml", ch=3, nc=len(DESK_CLASSES), verbose=False
        )
    except Exception:
        # Older/newer ultralytics releases move the 11-series YAMLs around;
        # locate it explicitly rather than failing the whole test on a path.
        import ultralytics

        cfg = (
            pathlib.Path(ultralytics.__file__).parent
            / "cfg" / "models" / "11" / "yolo11n.yaml"
        )
        model = DetectionModel(
            cfg=str(cfg), ch=3, nc=len(DESK_CLASSES), verbose=False
        )

    model.names = {i: name for i, name in enumerate(DESK_CLASSES)}
    # Same shape ultralytics writes when training, so the loader takes the
    # checkpoint branch it will take for a real promoted model.
    torch.save({"model": model, "epoch": -1, "best_fitness": None}, path)


def test_custom_checkpoint_loads() -> bool:
    """1. a desk_objects.pt is adopted as custom, with all desk classes"""
    print("1. real checkpoint loads as a custom desk model")
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        ckpt = tmp_path / "desk_objects.pt"
        build_desk_checkpoint(ckpt)
        ok &= check(f"synthetic checkpoint written (nc={len(DESK_CLASSES)})",
                    ckpt.is_file())

        ctx = ObjectContext()
        model = ctx.load("desk_objects.pt", "yolo11n.pt", (tmp,))
        ok &= check("model loaded", model is not None)
        ok &= check("recognised as a custom desk model", ctx.is_custom)
        ok &= check(
            f"path resolves to the checkpoint ({ctx.model_path})",
            ctx.model_path.endswith("desk_objects.pt"),
        )

        got = set(ctx.class_map.values())
        expected = set(DESK_CLASSES)
        ok &= check(
            f"all {len(DESK_CLASSES)} desk classes adopted "
            f"({len(got)} found)",
            got == expected,
            f"missing={sorted(expected - got)} extra={sorted(got - expected)}",
        )
        ok &= check(
            "exam_paper at id 0 kept (a person-class exclusion by id "
            "would drop it)",
            ctx.class_map.get(0) == "exam_paper",
            str(ctx.class_map.get(0)),
        )
        ok &= check(
            "no person class in the label set",
            "person" not in got,
        )

        # Every label the model reports must be a known desk class, so a
        # typo in a data-config cannot surface as an unknown verdict.
        ok &= check(
            "every adopted label is a declared desk class",
            all(v in DESK_CLASSES for v in ctx.class_map.values()),
        )
    return ok


def test_precedence_and_fallback() -> bool:
    """2. custom wins over stock; stock alone still works"""
    print()
    print("2. precedence: desk_objects.pt wins, stock is the fallback")
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        build_desk_checkpoint(tmp_path / "desk_objects.pt")
        if not STOCK_WEIGHTS.is_file():
            print("  [SKIP] backend/yolo11n.pt absent -- fallback untested")
            return True
        shutil.copy2(STOCK_WEIGHTS, tmp_path / "yolo11n.pt")

        custom_ctx = ObjectContext()
        custom_ctx.load("desk_objects.pt", "yolo11n.pt", (tmp,))
        ok &= check(
            "custom desk model preferred when both are present",
            custom_ctx.is_custom and len(custom_ctx.class_map) == len(DESK_CLASSES),
            f"is_custom={custom_ctx.is_custom} "
            f"classes={len(custom_ctx.class_map)}",
        )

        # Remove the custom model: the stock weights must then load and
        # report the COCO map, NOT a stale desk-class map.
        (tmp_path / "desk_objects.pt").unlink()
        stock_ctx = ObjectContext()
        stock_ctx.load("desk_objects.pt", "yolo11n.pt", (tmp,))
        ok &= check("falls back to stock weights", stock_ctx.model is not None)
        ok &= check(
            "stock model is not flagged custom",
            stock_ctx.is_custom is False,
        )
        ok &= check(
            "stock model reports the COCO label set, not desk classes",
            stock_ctx.class_map == OBJ_CLASS_MAP,
            str(sorted(stock_ctx.class_map.values())),
        )
    return ok


def test_detect_pass_with_custom_model() -> bool:
    """3. detect() runs the custom model end-to-end without crashing"""
    print()
    print("3. detect() pass through the custom model")
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        build_desk_checkpoint(tmp_path / "desk_objects.pt")

        ctx = ObjectContext()
        ctx.load("desk_objects.pt", "yolo11n.pt", (tmp,))

        # Empty frame, no people: the model still runs, and anything it
        # reports must be labelled with a desk class.  A randomly initialised
        # model on black pixels should find nothing, but the assertion is on
        # the labels rather than the count so the test does not depend on
        # what untrained weights happen to do.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx.set_frame_context(frame.shape, [], None, None, 0.0)
        dets = ctx.detect(frame)
        ok &= check(
            f"detect() completed on a blank frame ({len(dets)} detections)",
            isinstance(dets, list),
        )
        ok &= check(
            "any detection carries a declared desk-class label",
            all(d.get("label") in DESK_CLASSES for d in dets),
            str([d.get("label") for d in dets]),
        )
        ok &= check(
            "throttle counter advanced (detect really ran)",
            ctx.detect_counter >= 1,
            str(ctx.detect_counter),
        )
    return ok


def test_desk_class_escalation_contract() -> bool:
    """4. every desk class has a decision: devices escalate, ambient never do"""
    print()
    print("4. desk classes vs OBJ_ESCALATE_TO (cross-module contract)")
    ok = True

    unmapped_devices = sorted(DEVICE_CLASSES - set(perf_config.OBJ_ESCALATE_TO))
    ok &= check(
        "every DEVICE class is configured to escalate",
        not unmapped_devices,
        f"undecided: {unmapped_devices}",
    )

    ambient_leaking = sorted(AMBIENT_CLASSES & set(perf_config.OBJ_ESCALATE_TO))
    ok &= check(
        "no AMBIENT class escalates (paper/notebook are normal exam "
        "behaviour)",
        not ambient_leaking,
        f"would accuse every student: {ambient_leaking}",
    )

    # And the map's own entries must all be real desk classes or the two
    # legacy COCO names, so nothing points at a label nobody can emit.
    known = set(DESK_CLASSES) | {"phone", "book"}
    unknown = sorted(set(perf_config.OBJ_ESCALATE_TO) - known)
    ok &= check(
        "every escalation target names a class that can actually be emitted",
        not unknown,
        f"unknown: {unknown}",
    )

    # Behavioural spot-check with a desk-class label from the loaded map.
    held = {
        "label": "cell_phone", "confidence": 0.9,
        "box": [0, 0, 10, 10],
        "relation": "in_hand", "in_hand": True, "reachable": True,
    }
    ok &= check(
        "desk-class device at the wrist escalates",
        ObjectContext.escalation([held]) is not None,
    )
    ok &= check(
        "desk-class exam paper at the wrist does not",
        ObjectContext.escalation([dict(held, label="exam_paper")]) is None,
    )
    return ok


def main() -> int:
    ok = True
    ok &= test_custom_checkpoint_loads()
    ok &= test_precedence_and_fallback()
    ok &= test_detect_pass_with_custom_model()
    ok &= test_desk_class_escalation_contract()
    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
