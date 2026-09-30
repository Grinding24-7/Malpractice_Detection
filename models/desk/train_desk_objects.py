"""Train the custom desk-object detector.

    python -m models.desk.train_desk_objects

Requires a labelled dataset laid out as described in desk_config.py.  The
script refuses to start without it, because the alternative -- training on
whatever images happen to be present and reporting a confident mAP for a model
that learned nothing -- is worse than not running at all.

Validation results are checked against ACCEPTANCE before the weights are
promoted to desk_objects.pt, so an undertrained model cannot silently become
the production detector.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.desk import desk_config  # noqa: E402
from models.desk.desk_classes import DESK_CLASSES  # noqa: E402


def _fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def check_dataset() -> pathlib.Path:
    """Validate the dataset layout and the class-name contract."""
    yaml_path = desk_config.DATA_YAML
    if not yaml_path.is_file():
        _fail(
            f"{yaml_path} not found.\n"
            "Create the dataset first:\n"
            "  data/desk/images/{train,val}/\n"
            "  data/desk/labels/{train,val}/   (YOLO format .txt per image)\n"
            f"  {yaml_path.relative_to(pathlib.Path(__file__).resolve().parents[2])}"
            " with a 'names:' map matching\n"
            "  models/desk/desk_classes.py exactly."
        )

    root = yaml_path.parent
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            _fail(f"missing {img_dir}")
        if not lbl_dir.is_dir():
            _fail(f"missing {lbl_dir}")
        imgs = [p for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        counts[split] = len(imgs)
        if not imgs:
            _fail(f"{img_dir} contains no images")
        missing = [
            p for p in imgs
            if not (lbl_dir / f"{p.stem}.txt").is_file()
        ]
        if missing:
            _fail(
                f"{len(missing)} image(s) in {img_dir} have no label file, "
                f"e.g. {missing[0].name}.\n"
                "Every training image needs a .txt label file, even if it is "
                "empty (an empty file means 'no desk objects visible')."
            )

    if counts["val"] < 10:
        _fail(
            f"only {counts['val']} validation images; at least 10 are needed "
            "for the acceptance check to mean anything"
        )
    if counts["val"] >= counts["train"]:
        _fail(
            f"validation set ({counts['val']}) is not smaller than the "
            f"training set ({counts['train']})"
        )
    return root


def check_class_contract(root: pathlib.Path) -> None:
    """The yaml 'names' map must match desk_classes.py exactly.

    A mismatch here is the most damaging kind of silent error available: the
    model trains happily and every class id means something else at inference.
    """
    text = (root / "desk.yaml").read_text(encoding="utf-8")
    listed: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not stripped or stripped.startswith("#"):
                if in_names and stripped.startswith("path:"):
                    break
                continue
            body = stripped.strip("{} ").rstrip(",")
            if ":" not in body:
                break
            key, _, val = body.partition(":")
            try:
                listed[int(key.strip().strip("{}"))] = val.strip().strip("\"'")
            except ValueError:
                break

    if not listed:
        _fail(
            "could not parse a 'names:' map from desk.yaml.\n"
            "Expected, e.g.:\n"
            "  names:\n"
            "    0: exam_paper\n"
            f"  Classes defined: {DESK_CLASSES}"
        )

    mismatch = [
        (i, listed.get(i), name)
        for i, name in enumerate(DESK_CLASSES)
        if listed.get(i) != name
    ]
    if mismatch:
        _fail(
            "desk.yaml names do not match models/desk/desk_classes.py:\n"
            + "\n".join(
                f"  id {i}: yaml={got!r} expected={want!r}"
                for i, got, want in mismatch
            )
        )
    print(f"class contract verified: {len(DESK_CLASSES)} classes")


def main() -> int:
    from ultralytics import YOLO

    print("custom desk-object training")
    print(f"  classes : {len(DESK_CLASSES)} ({', '.join(DESK_CLASSES)})")
    print(f"  data    : {desk_config.DATA_YAML}")
    print(f"  output  : {desk_config.WEIGHTS}")
    print()

    root = check_dataset()
    check_class_contract(root)

    model = YOLO(desk_config.BASE_MODEL)
    results = model.train(**desk_config.TRAIN_KWARGS)

    best = getattr(results, "save_dir", None)
    best_pt = pathlib.Path(best) / "weights" / "best.pt" if best else None
    if not best_pt or not best_pt.is_file():
        _fail(f"training finished but {best_pt} was not produced")

    metrics = _extract_metrics(results)
    print()
    print("validation metrics")
    for key, val in metrics.items():
        print(f"  {key:<14} {val:.3f}")

    floors = desk_config.ACCEPTANCE
    shortfalls = [
        f"{k}: {metrics.get(k, 0):.3f} < {v}"
        for k, v in floors.items()
        if k.startswith("min_") and metrics.get(k, 0.0) < v
    ]
    if shortfalls:
        print()
        print("ACCEPTANCE NOT MET -- weights NOT promoted:")
        for s in shortfalls:
            print(f"  {s}")
        print()
        print("The trained weights were left in place for inspection:")
        print(f"  {best_pt}")
        print("Add labelled data, then re-run. Do not lower the floors to")
        print("force a pass: that trades a missed cheating event for a")
        print("false accusation, which is the worse error here.")
        return 1

    shutil.copy2(best_pt, desk_config.WEIGHTS)
    print()
    print(f"ACCEPTANCE MET -- promoted to {desk_config.WEIGHTS}")
    print("Restart the app (or the backend worker) to pick up the new model.")
    return 0


def _extract_metrics(results) -> dict[str, float]:
    """Pull precision/recall/mAP out of the validation results defensively."""
    out: dict[str, float] = {}
    box = getattr(results, "box", None)
    if box is None:
        return out
    out["min_precision"] = float(getattr(box, "mp", 0.0))
    out["min_recall"] = float(getattr(box, "mr", 0.0))
    out["min_map50"] = float(getattr(box, "map50", 0.0))
    if hasattr(box, "map"):
        out["map50_95"] = float(getattr(box, "map"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
