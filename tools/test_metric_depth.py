"""Metric 3D depth regression suite.

Verifies the solvePnP depth path that replaced the pixel-domain phone
threshold:

1. The nominal 1280x720 intrinsics scale correctly for other resolutions.
2. Synthetic face and torso projections recover the known metric depth.
3. Wrist-to-object distance is measured in metres on the candidate plane.
4. ``annotate_metric_distances`` marks a device as in-hand only inside the
   0.15 m usability radius.
5. The live classifier fails closed when depth exists but an object could
   not be measured, and uses the legacy pixel path only when the depth solve
   is entirely unavailable.

Run:  python tools/test_metric_depth.py
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from desktop_app.core import metric_depth, perf_config  # noqa: E402
from desktop_app.core.pose_classifier import PoseClassifier  # noqa: E402

FRAME_SHAPE = (720, 1280, 3)


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return bool(condition)


def base_landmarks() -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 0.0)] * 33


def face_landmarks(
    tvec: tuple[float, float, float],
    wrist: tuple[float, float] | None = (640.0, 360.0),
) -> list[tuple[float, float, float]]:
    """Project the 5-point face model, plus an optional visible wrist."""
    intrinsics = metric_depth.camera_matrix(FRAME_SHAPE)
    image, _ = cv2.projectPoints(
        metric_depth.FACE_PNP_MODEL_3D_M,
        np.zeros((3, 1)),
        np.array(tvec, dtype=np.float64).reshape(3, 1),
        intrinsics,
        None,
    )
    points = image.reshape(-1, 2)
    lms = base_landmarks()
    for idx, point in zip((0, 3, 6, 9, 10), points):
        lms[idx] = (float(point[0]), float(point[1]), 1.0)
    if wrist is not None:
        lms[15] = (float(wrist[0]), float(wrist[1]), 1.0)
    return lms


def torso_landmarks(
    tvec: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Project the 5-point torso fallback model."""
    intrinsics = metric_depth.camera_matrix(FRAME_SHAPE)
    image, _ = cv2.projectPoints(
        metric_depth.TORSO_MODEL_3D_M,
        np.zeros((3, 1)),
        np.array(tvec, dtype=np.float64).reshape(3, 1),
        intrinsics,
        None,
    )
    points = image.reshape(-1, 2)
    lms = base_landmarks()
    for idx, point in zip((11, 12, 23, 24, 0), points):
        lms[idx] = (float(point[0]), float(point[1]), 1.0)
    return lms


def test_camera_matrix() -> bool:
    print()
    print("1. camera intrinsics scale from the 1280x720 contract")
    hd = metric_depth.camera_matrix(FRAME_SHAPE)
    ok = True
    ok &= check("fx = fy = 1000 at 1280x720",
                float(hd[0, 0]) == 1000.0 and float(hd[1, 1]) == 1000.0,
                f"fx={hd[0, 0]} fy={hd[1, 1]}")
    ok &= check("principal point is (640, 360)",
                float(hd[0, 2]) == 640.0 and float(hd[1, 2]) == 360.0,
                f"cx={hd[0, 2]} cy={hd[1, 2]}")
    half = metric_depth.camera_matrix((432, 768, 3))
    ok &= check("768x432 scales fx/fy to 600",
                float(half[0, 0]) == 600.0 and float(half[1, 1]) == 600.0,
                f"fx={half[0, 0]} fy={half[1, 1]}")
    ok &= check("768x432 scales principal point to (384, 216)",
                float(half[0, 2]) == 384.0 and float(half[1, 2]) == 216.0,
                f"cx={half[0, 2]} cy={half[1, 2]}")
    return ok


def test_pnp_depth() -> bool:
    print()
    print("2. synthetic face/torso landmarks recover metric depth")
    ok = True
    for tvec in ((0.02, 0.03, 0.6), (0.0, 0.0, 1.0), (-0.10, 0.05, 2.5)):
        result = metric_depth.estimate_candidate_depth(
            face_landmarks(tvec), FRAME_SHAPE
        )
        ok &= check(
            f"face depth {tvec[2]:.1f} m",
            result["success"]
            and abs(float(result["depth_m"]) - tvec[2]) < 0.01
            and result["method"] == "face",
            f"got {result['depth_m']} ({result['method']}, "
            f"{result['reprojection_error_px']} px)",
        )
    for tvec in ((0.0, 0.0, 0.8), (0.0, 0.0, 1.5), (0.0, 0.0, 3.0)):
        result = metric_depth.estimate_candidate_depth(
            torso_landmarks(tvec), FRAME_SHAPE
        )
        ok &= check(
            f"torso depth {tvec[2]:.1f} m",
            result["success"]
            and abs(float(result["depth_m"]) - tvec[2]) < 0.01
            and result["method"] == "torso",
            f"got {result['depth_m']} ({result['method']}, "
            f"{result['reprojection_error_px']} px)",
        )
    missing = metric_depth.estimate_candidate_depth([], FRAME_SHAPE)
    ok &= check("no landmarks -> solve fails closed",
                missing["success"] is False and missing["depth_m"] is None)
    return ok


def test_wrist_object_distance() -> bool:
    print()
    print("3. wrist-to-object distance is metric on the depth plane")
    lms = face_landmarks((0.0, 0.0, 1.0), wrist=(640.0, 360.0))
    depth_info = metric_depth.estimate_candidate_depth(lms, FRAME_SHAPE)
    ok = True
    ok &= check("depth solve available for distance test",
                depth_info["success"] and depth_info["depth_m"] == 1.0)

    # At 1 m and fx=1000, 100 px is 0.10 m.  Box [690,340,790,380] is
    # centred at (740, 360), exactly 100 px right of the wrist.
    measured = metric_depth.wrist_object_distance_m(
        [690, 340, 790, 380], lms, 1.0, depth_info["camera_matrix"],
        FRAME_SHAPE,
    )
    ok &= check("100 px at 1 m -> 0.10 m",
                measured is not None and abs(measured[0] - 0.10) < 0.005,
                f"got {None if measured is None else round(measured[0], 4)}")

    # 200 px -> 0.20 m, beyond the 0.15 m usability radius.
    measured = metric_depth.wrist_object_distance_m(
        [790, 340, 890, 380], lms, 1.0, depth_info["camera_matrix"],
        FRAME_SHAPE,
    )
    ok &= check("200 px at 1 m -> 0.20 m",
                measured is not None and abs(measured[0] - 0.20) < 0.005,
                f"got {None if measured is None else round(measured[0], 4)}")

    ok &= check("no depth -> not measured",
                metric_depth.wrist_object_distance_m(
                    [690, 340, 790, 380], lms, None
                ) is None)
    ok &= check("no landmarks -> not measured",
                metric_depth.wrist_object_distance_m(
                    [690, 340, 790, 380], [], 1.0
                ) is None)
    return ok


def test_annotation() -> bool:
    print()
    print("4. annotate_metric_distances applies the 0.15 m in-hand gate")
    lms = face_landmarks((0.0, 0.0, 1.0), wrist=(640.0, 360.0))
    depth_info = metric_depth.estimate_candidate_depth(lms, FRAME_SHAPE)
    ok = True
    ok &= check("threshold is 0.15 m", perf_config.PHONE_WRIST_DIST_M == 0.15)

    near = {"label": "cell_phone", "confidence": 0.9,
            "box": [690, 340, 790, 380]}
    far = {"label": "cell_phone", "confidence": 0.9,
           "box": [790, 340, 890, 380]}
    metric_depth.annotate_metric_distances(
        [near, far], lms, depth_info, FRAME_SHAPE
    )
    ok &= check("near 0.10 m is in_hand",
                near.get("wrist_dist_m") is not None
                and float(near["wrist_dist_m"]) < 0.15
                and near.get("in_hand") is True,
                f"dist={near.get('wrist_dist_m')}")
    ok &= check("far 0.20 m is not in_hand",
                far.get("wrist_dist_m") is not None
                and float(far["wrist_dist_m"]) > 0.15
                and far.get("in_hand") is False,
                f"dist={far.get('wrist_dist_m')}")

    untouched = {"label": "notebook", "confidence": 0.9,
                 "box": [690, 340, 790, 380]}
    metric_depth.annotate_metric_distances(
        [untouched], lms, depth_info, FRAME_SHAPE
    )
    ok &= check("documents are not wrist-annotated",
                "wrist_dist_m" not in untouched)
    no_depth = {"label": "cell_phone", "confidence": 0.9,
                "box": [690, 340, 790, 380]}
    metric_depth.annotate_metric_distances(
        [no_depth], lms, {"success": False, "depth_m": None,
                          "camera_matrix": None}, FRAME_SHAPE
    )
    ok &= check("no depth leaves the object unannotated",
                "wrist_dist_m" not in no_depth)
    return ok


def test_classifier_gate() -> bool:
    print()
    print("5. classifier P31 only with a measured <0.15 m wrist distance")
    lms = face_landmarks((0.0, 0.0, 1.0), wrist=(640.0, 360.0))
    depth_info = metric_depth.estimate_candidate_depth(lms, FRAME_SHAPE)
    classifier = PoseClassifier.__new__(PoseClassifier)
    classifier._last_landmarks = lms
    classifier._last_depth_info = depth_info

    def phone(box: list[int], **extra) -> dict:
        obj = {"label": "cell_phone", "confidence": 0.9,
               "hits": perf_config.PROX_MIN_HITS, "box": box}
        obj.update(extra)
        return obj

    near = phone([690, 340, 790, 380])
    metric_depth.annotate_metric_distances(
        [near], lms, depth_info, FRAME_SHAPE
    )
    ok = True
    ok &= check("measured 0.10 m phone -> usability hit",
                classifier._phone_proximity_hit([near]) is not None)
    far = phone([790, 340, 890, 380])
    metric_depth.annotate_metric_distances(
        [far], lms, depth_info, FRAME_SHAPE
    )
    ok &= check("measured 0.20 m phone -> no hit",
                classifier._phone_proximity_hit([far]) is None)
    unmeasured = phone([690, 340, 790, 380])
    classifier._last_depth_info = depth_info
    ok &= check("depth exists but object unmeasured -> fail closed",
                classifier._phone_proximity_hit([unmeasured]) is None)

    classifier._last_depth_info = {
        "success": False, "depth_m": None, "camera_matrix": None,
    }
    pixel_near = phone([630, 350, 650, 370])
    ok &= check("no depth solve -> legacy pixel fallback still works",
                classifier._phone_proximity_hit([pixel_near]) is not None)
    return ok


def main() -> int:
    ok = True
    ok &= test_camera_matrix()
    ok &= test_pnp_depth()
    ok &= test_wrist_object_distance()
    ok &= test_annotation()
    ok &= test_classifier_gate()
    print()
    print("=" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
