"""Training config for the custom desk-object detector.

Run `python -m models.desk.desk_classes` to print the class list this config
references.  Weights are written to models/desk/desk_objects.pt, which is
exactly where perf_config.OBJ_MODEL_DIRS looks, and the stem starts with
"desk_" so the loader recognises it as custom and adopts its own class names
without any code change.

The stock yolo11n.pt is fine for phones and laptops.  It cannot see paper, pen
or notebook, so this model is what makes a clear-desk claim defensible.
"""

from __future__ import annotations

import pathlib

# Written here; models/desk is searched before backend/ at load time.
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
WEIGHTS = OUTPUT_DIR / "desk_objects.pt"

# Expected dataset layout (Ultralytics YOLO detection format):
#   data/desk/
#     images/{train,val}/<any name>.jpg
#     labels/{train,val}/<same name>.txt   one line per object:
#         <class_id> <x_center> <y_center> <width> <height>   normalised 0-1
#   desk.yaml:
#     path: ../data/desk
#     train: images/train
#     val: images/val
#     names: {0: exam_paper, ...}   # must match models/desk/desk_classes.py
DATA_YAML = pathlib.Path(__file__).resolve().parents[2] / "data" / "desk" / "desk.yaml"

# Start from the stock COCO weights rather than from scratch: the desk
# classes are visually ordinary, and the COCO model already knows what a
# phone, book and laptop look like under the lighting and camera angles
# common to exam rooms.
BASE_MODEL = "yolo11n.pt"

TRAIN_KWARGS = {
    "data": str(DATA_YAML),
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "device": "cpu",
    "project": str(OUTPUT_DIR / "runs"),
    "name": "desk_objects",
    "exist_ok": True,
    # Desk objects are small and often partly occluded by a hand, so the
    # default 0.7 box/cls loss balance is raised for classification and the
    # objectness gain lowered, which favours many small objects over a few
    # large confident ones.
    "box": 7.5,
    "cls": 1.0,
    "obj": 3.0,
    # A held-out validation split is required; without one the reported
    # mAP is a measure of memorisation.
    "val": True,
    "patience": 25,          # stop early if validation stops improving
    "workers": 2,
    "seed": 0,
    "deterministic": True,
    "verbose": True,
}

# Recall, not mAP, is the quantity that matters here: a missed exam_paper is a
# missed cheating event, while a false positive is a false accusation.  The
# acceptance floor below is deliberately conservative -- it should be raised
# as labelled data grows, not lowered to make a first run pass.
ACCEPTANCE = {
    "min_precision": 0.80,
    "min_recall": 0.70,
    "min_map50": 0.60,
    "note": (
        "Do not evaluate on the stock sample_exam.mp4: it contains no "
        "detectable desk objects at any confidence, so it cannot separate a "
        "good desk model from an empty one. A labelled val split is the only "
        "valid test."
    ),
}
