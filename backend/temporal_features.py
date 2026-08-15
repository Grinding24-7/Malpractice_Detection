"""
temporal_features.py — Week 4: spatial-temporal feature extraction & anomaly sequences.

Pipeline stage 3 of the Intelligent Exam Malpractice Detection stack:

    Week 1-2: pose detection + per-frame heuristics        (detector.py / feature_extractor.py)
    Week 3:   multi-student ByteTrack + per-track buffers  (app.py)
    Week 4:   THIS module — temporal sequence features.

Responsibilities
---------------
1. ``TemporalFeatureExtractor`` — turns a ``(T, 17, 2)`` window of normalised
   2D COCO-17 keypoints into per-frame ``(T, F)`` feature tensors plus a
   window-level ``(S,)`` summary vector.  Velocities / accelerations are 1st /
   2nd-order backward temporal differences; joint angles are computed for six
   body vectors (head, arms, torso, spine).  Every numeric step is a batched
   NumPy operation — there are no Python loops over time steps or keypoints.

2. Sequence builder + dataset saver — batches raw windows into ``(B, T, F)``
   PyTorch tensors and persists ``.pt`` / ``.npz`` artefacts ready for
   LSTM / GRU / ST-GCN training.

3. ``HeuristicBaseline`` — rule-based early flagging (head turning, hand
   reaching / note passing) evaluated on the current window snapshot in well
   under 1 ms, so it never adds streaming latency to the Week 3 pipeline.

4. ``PoseWindowManager`` — per-``track_id`` sliding-window pose buffer that
   mirrors the Week 3 ``collections.deque(maxlen=T)`` RAM-buffer pattern and
   is filled incrementally with O(1) appends from the 5 FPS inference path.

Coordinate convention
---------------------
Keypoints must be NORMALISED to the image plane: ``x in [0, 1]`` (image
width), ``y in [0, 1]`` (image height).  Consequently velocities /
displacements below are expressed in image-width units per frame, and every
geometric quantity is scale-invariant w.r.t. camera zoom / subject distance.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 keypoint schema (YOLO11n-pose output order)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Head keypoints used for the head-turn proxy (lateral displacement centroid).
HEAD_KEYPOINTS: np.ndarray = np.asarray(
    [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR], dtype=np.int64
)

# Virtual keypoints appended to the (17, 2) skeleton for geometric features.
NECK = 17    # midpoint of the two shoulders
HIP_MID = 18  # midpoint of the two hips

# Body vectors whose per-frame angle / angular displacement become features.
VECTOR_NAMES: list[str] = [
    "head", "left_arm", "right_arm", "left_torso", "right_torso", "spine",
]
BODY_VECTORS: np.ndarray = np.asarray(
    [
        (NECK, NOSE),                # head (yaw proxy)
        (LEFT_SHOULDER, LEFT_WRIST),  # left arm (reach)
        (RIGHT_SHOULDER, RIGHT_WRIST),  # right arm (reach)
        (LEFT_SHOULDER, LEFT_HIP),    # left torso
        (RIGHT_SHOULDER, RIGHT_HIP),  # right torso
        (NECK, HIP_MID),              # spine
    ],
    dtype=np.int64,
)

# Resulting tensor dimensions (see FEATURE_LAYOUT / SUMMARY_LAYOUT).
FEATURE_DIM: int = 17 * 2 + 17 * 2 + 17 * 2 + len(BODY_VECTORS) * 2 + 1 + 2
SUMMARY_DIM: int = 17 * 2 + 17 + 17 * 2 + 17 + 1 + 1

__all__ = [
    "TemporalFeatureExtractor",
    "PoseWindowManager",
    "HeuristicBaseline",
    "SequenceDatasetWriter",
    "PoseSequenceDataset",
    "build_sequence_tensor",
    "save_dataset",
    "load_dataset",
    "normalize_keypoints",
    "make_synthetic_sequence",
    "FEATURE_DIM",
    "SUMMARY_DIM",
    "KEYPOINT_NAMES",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _require_torch():
    """Lazily import PyTorch so pure-NumPy paths stay importable without it."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the sequence-tensor / dataset features. "
            "Install it, e.g. via the project venv."
        ) from exc
    return torch


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into [-pi, pi] (shortest signed arc)."""
    return np.arctan2(np.sin(angles), np.cos(angles))


def normalize_keypoints(
    keypoints: np.ndarray, width: float, height: float
) -> Optional[np.ndarray]:
    """
    Normalise one person's keypoints to the image plane.

    Args:
        keypoints: (17, 3) array in COCO order, columns [x, y, confidence],
            or (17, 2) if confidence is unavailable.
        width: frame width in pixels (> 0).
        height: frame height in pixels (> 0).

    Returns:
        (17, 2) float32 array with x in [0, 1] and y in [0, 1], or None when
        fewer than 3 keypoints have meaningful confidence (degenerate track).
    """
    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.shape[-1] >= 3:
        valid = kpts[..., 2] > 0.1
    else:
        valid = np.ones(kpts.shape[-2], dtype=bool)

    if kpts.shape[-2] != 17 or kpts.shape[-1] < 2:
        return None

    xy = kpts[..., :2].copy()
    xy[..., 0] /= float(width)
    xy[..., 1] /= float(height)
    xy[~valid] = 0.0  # mask out low-confidence joints
    if int(valid.sum()) < 3:
        return None
    return np.asarray(xy, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Spatial-temporal feature extraction
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor:
    """
    Vectorized spatial-temporal feature extraction over a pose window.

    All methods accept either a single window ``(T, 17, 2)`` or a batch of
    windows ``(B, T, 17, 2)``; intermediate arrays keep the leading batch dims
    so the full stack of transforms is executed by NumPy without Python loops.

    Per-frame feature layout ``(T, F)``, ``F = 117``:
        [0:34)     17x2 normalised keypoints
        [34:68)    17x2 velocity      (p_t - p_{t-1}, frame 0 zero-padded)
        [68:102)   17x2 acceleration  (v_t - v_{t-1}, frames 0-1 zero-padded)
        [102:108)  6 joint angles     (radians, 6 body vectors)
        [108:114)  6 angular displacements vs frame 0 (wrapped radians)
        [114]      head centroid lateral displacement vs frame 0 (x only)
        [115:117)  |wrist velocity| left / right

    Window-level summary ``(S,)``, ``S = 104``:
        [0:34)     mean |velocity| per keypoint coordinate
        [34:51)    max |acceleration| per joint
        [51:85)    temporal variance per keypoint coordinate
        [85:102)   total path length per joint
        [102]      total motion (mean speed across all joints / time)
        [103]      max |head angular displacement|
    """

    def __init__(self, window_size: int = 30, n_keypoints: int = 17) -> None:
        self.window_size = int(window_size)
        self.n_keypoints = int(n_keypoints)

    # -- kinematics ---------------------------------------------------------

    def velocities(self, sequence: np.ndarray) -> np.ndarray:
        """
        1st-order backward temporal difference, zero-padded at t=0.

        Args:
            sequence: (..., T, 17, 2) normalised keypoints.

        Returns:
            (..., T, 17, 2) where out[t] = sequence[t] - sequence[t-1] and
            out[0] = 0.
        """
        seq = np.asarray(sequence, dtype=np.float32)
        diff = np.diff(seq, axis=-3)  # (..., T-1, 17, 2)
        pad = np.zeros_like(seq[..., :1, :, :])
        return np.concatenate([pad, diff], axis=-3)

    def accelerations(self, sequence: np.ndarray) -> np.ndarray:
        """
        2nd-order backward temporal difference (Δ²p = Δp_t - Δp_{t-1}).

        Args:
            sequence: (..., T, 17, 2) normalised keypoints.

        Returns:
            (..., T, 17, 2) where out[t] = p_t - 2 p_{t-1} + p_{t-2} for
            t >= 2 and 0 otherwise.
        """
        seq = np.asarray(sequence, dtype=np.float32)
        vel = np.diff(seq, axis=-3)          # (..., T-1, 17, 2)
        acc = np.diff(vel, axis=-3)          # (..., T-2, 17, 2)
        pad = np.zeros_like(seq[..., :2, :, :])
        return np.concatenate([pad, acc], axis=-3)

    # -- geometry -----------------------------------------------------------

    def _augment(self, sequence: np.ndarray) -> np.ndarray:
        """
        Append virtual neck and hip-midpoint keypoints to the skeleton.

        Args:
            sequence: (..., T, 17, 2).

        Returns:
            (..., T, 19, 2) with indices 17 = neck, 18 = hip midpoint.
        """
        seq = np.asarray(sequence, dtype=np.float32)
        neck = (seq[..., LEFT_SHOULDER, :] + seq[..., RIGHT_SHOULDER, :]) / 2.0
        hip_mid = (seq[..., LEFT_HIP, :] + seq[..., RIGHT_HIP, :]) / 2.0
        return np.concatenate(
            [seq, neck[..., None, :], hip_mid[..., None, :]], axis=-2
        )

    def joint_angles(
        self, sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Angles and angular displacements of the six body vectors.

        Args:
            sequence: (..., T, 17, 2) normalised keypoints.

        Returns:
            angles: (..., T, 6) vector angles in radians (image x-axis frame).
            angular_displacement: (..., T, 6) wrapped angle minus the angle
                at t=0, i.e. the temporal rotation of each body vector.
        """
        aug = self._augment(sequence)
        start = aug[..., BODY_VECTORS[:, 0], :]  # (..., T, 6, 2)
        end = aug[..., BODY_VECTORS[:, 1], :]    # (..., T, 6, 2)
        vec = end - start
        angles = np.arctan2(vec[..., 1], vec[..., 0])  # (..., T, 6)
        ref = angles[..., 0:1, :]                      # (..., 1, 6)
        displacement = _wrap_angles(angles - ref)
        return angles, displacement

    # -- feature assembly ---------------------------------------------------

    def extract_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Vectorized per-frame features for a batch of windows.

        Args:
            sequences: (B, T, 17, 2) or (T, 17, 2) normalised keypoints.

        Returns:
            (B, T, F) float32 feature tensor with F = FEATURE_DIM = 117.
        """
        seq = np.asarray(sequences, dtype=np.float32)
        batch = seq.shape[:-3]
        T = seq.shape[-3]
        if seq.ndim == 3:
            seq = seq[None]
            batch = seq.shape[:-3]

        vel = self.velocities(seq)     # (B, T, 17, 2)
        acc = self.accelerations(seq)  # (B, T, 17, 2)
        angles, ang_disp = self.joint_angles(seq)  # (B, T, 6) each

        head = seq[..., HEAD_KEYPOINTS, :].mean(axis=-2)  # (B, T, 2)
        head_dx = head[..., 0] - head[..., 0:1, 0]        # (B, T)

        vel_raw = np.diff(seq, axis=-3)  # (B, T-1, 17, 2)
        wrist_speed = np.linalg.norm(
            vel_raw[..., [LEFT_WRIST, RIGHT_WRIST], :], axis=-1
        )  # (B, T-1, 2)
        wrist_speed = np.concatenate(
            [np.zeros_like(wrist_speed[..., :1, :]), wrist_speed], axis=-2
        )  # (B, T, 2)

        features = np.concatenate(
            [
                seq.reshape(*batch, T, -1),      # 34
                vel.reshape(*batch, T, -1),      # 34
                acc.reshape(*batch, T, -1),      # 34
                angles,                          # 6
                ang_disp,                        # 6
                head_dx[..., None],              # 1
                wrist_speed,                     # 2
            ],
            axis=-1,
        )
        return features.astype(np.float32)

    def extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Per-frame features for a single window.

        Args:
            sequence: (T, 17, 2) normalised keypoints.

        Returns:
            (T, F) float32 feature tensor.
        """
        out = self.extract_batch(sequence)
        return out[0] if out.ndim == 3 else out

    # -- window-level summary ----------------------------------------------

    def summarize_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Aggregate motion statistics across each window (see class docstring).

        Args:
            sequences: (B, T, 17, 2) or (T, 17, 2).

        Returns:
            (B, SUMMARY_DIM) float32 summary vectors (S = 104).
        """
        seq = np.asarray(sequences, dtype=np.float32)
        if seq.ndim == 3:
            seq = seq[None]
        B = seq.shape[0]

        vel = np.abs(np.diff(seq, axis=-3))         # (B, T-1, 17, 2)
        acc = np.abs(np.diff(seq, n=2, axis=-3))    # (B, T-2, 17, 2)

        mean_speed_comp = vel.mean(axis=-3).reshape(B, -1)          # 34
        max_acc_joint = acc.max(axis=-3).max(axis=-1)               # 17
        spatial_var_comp = seq.var(axis=-3).reshape(B, -1)          # 34
        path_len_joint = vel.sum(axis=-3).mean(axis=-1)             # 17
        total_motion = vel.reshape(B, -1).mean(axis=1)              # 1

        _, ang_disp = self.joint_angles(seq)        # (B, T, 6)
        head_yaw_max = np.abs(ang_disp[..., 0]).max(axis=-1)        # 1

        summary = np.concatenate(
            [
                mean_speed_comp,
                max_acc_joint,
                spatial_var_comp,
                path_len_joint,
                total_motion[:, None],
                head_yaw_max[:, None],
            ],
            axis=1,
        )
        return summary.astype(np.float32)

    def summarize(self, sequence: np.ndarray) -> np.ndarray:
        """Aggregate motion statistics for a single window -> (SUMMARY_DIM,)."""
        return self.summarize_batch(sequence)[0]


# ---------------------------------------------------------------------------
# 2. Sequence tensor builder + dataset saver
# ---------------------------------------------------------------------------

def build_sequence_tensor(
    windows: list[np.ndarray],
    extractor: Optional[TemporalFeatureExtractor] = None,
    labels: Optional[list[int]] = None,
):
    """
    Stack raw windows into a model-ready (B, T, F) PyTorch tensor.

    Args:
        windows: list of (T, 17, 2) normalised keypoint windows.
        extractor: feature extractor; defaults to a fresh TemporalFeatureExtractor.
        labels: optional integer labels aligned with `windows`.

    Returns:
        X: (B, T, F) float32 torch.Tensor; if `labels` is given, returns
        (X, y) where y is a (B,) int64 torch.Tensor.
    """
    if not windows:
        raise ValueError("build_sequence_tensor requires at least one window")
    torch = _require_torch()
    extractor = extractor or TemporalFeatureExtractor()
    X = extractor.extract_batch(np.stack(windows))  # (B, T, F)
    Xt = torch.from_numpy(X)
    if labels is not None:
        y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        return Xt, y
    return Xt


def save_dataset(
    path: str | Path,
    windows: list[np.ndarray],
    labels: Optional[list[int]] = None,
    extractor: Optional[TemporalFeatureExtractor] = None,
    fmt: str = "pt",
) -> Path:
    """
    Persist labelled temporal sequences for model training.

    Args:
        path: destination file (.pt or .npz depending on `fmt`).
        windows: list of (T, 17, 2) windows.
        labels: optional integer class per window.
        extractor: feature extractor (fresh one used if omitted).
        fmt: "pt" (torch.save) or "npz" (np.savez_compressed).

    Returns:
        Path of the written artefact.

    The payload always contains:
        X:       (B, T, F) float32 features
        summary: (B, SUMMARY_DIM) float32 window-level statistics
        labels:  (B,) int64 (when `labels` provided)
        meta:    {window_size, feature_dim, summary_dim} for .pt
    """
    path = Path(path)
    extractor = extractor or TemporalFeatureExtractor()
    X = extractor.extract_batch(np.stack(windows))
    summaries = extractor.summarize_batch(np.stack(windows))
    label_arr = (
        np.asarray(labels, dtype=np.int64) if labels is not None else None
    )
    meta = {
        "window_size": int(X.shape[1]),
        "feature_dim": int(X.shape[-1]),
        "summary_dim": int(summaries.shape[-1]),
        "n_keypoints": 17,
    }

    if fmt == "pt":
        torch = _require_torch()
        payload: dict = {
            "X": torch.from_numpy(X),
            "summary": torch.from_numpy(summaries),
            "meta": meta,
        }
        if label_arr is not None:
            payload["labels"] = torch.from_numpy(label_arr)
        torch.save(payload, str(path))
    elif fmt == "npz":
        data: dict = {
            "X": X,
            "summary": summaries,
            "window_size": meta["window_size"],
            "feature_dim": meta["feature_dim"],
            "summary_dim": meta["summary_dim"],
        }
        if label_arr is not None:
            data["labels"] = label_arr
        np.savez_compressed(path, **data)
    else:
        raise ValueError(f"unknown dataset format: {fmt!r}")
    return path


def load_dataset(path: str | Path) -> dict:
    """
    Load a dataset saved by :func:`save_dataset`.

    Returns:
        dict with keys "X" ((B, T, F) torch.Tensor), "summary"
        ((B, S) torch.Tensor), optional "labels" ((B,) torch.Tensor), and
        "meta" (dict).
    """
    path = Path(path)
    if path.suffix == ".pt":
        torch = _require_torch()
        return torch.load(str(path), map_location="cpu", weights_only=True)
    torch = _require_torch()
    data = np.load(str(path))
    payload = {
        "X": torch.from_numpy(data["X"]),
        "summary": torch.from_numpy(data["summary"]),
        "meta": {
            "window_size": int(data["window_size"]),
            "feature_dim": int(data["feature_dim"]),
            "summary_dim": int(data["summary_dim"]),
        },
    }
    if "labels" in data:
        payload["labels"] = torch.from_numpy(data["labels"])
    return payload


class PoseSequenceDataset:
    """
    Duck-typed torch.utils.data.Dataset over saved sequence tensors.

    Compatible with ``torch.utils.data.DataLoader`` (only ``__len__`` and
    ``__getitem__`` are required).  Samples are returned as ``(X_i,)`` plus
    optional ``(labels_i,)`` and ``(summary_i,)``.

    Args:
        X: (B, T, F) feature tensor.
        labels: optional (B,) int64 class tensor.
        summaries: optional (B, S) window-level summary tensor.
    """

    def __init__(self, X, labels=None, summaries=None) -> None:
        self.X = X
        self.labels = labels
        self.summaries = summaries

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        item = (self.X[idx],)
        if self.labels is not None:
            item += (self.labels[idx],)
        if self.summaries is not None:
            item += (self.summaries[idx],)
        return item


class SequenceDatasetWriter:
    """
    Incremental (append-style) labelled-sequence collector.

    Adds windows in O(1) amortised time and flushes buffered samples to disk
    in bounded .pt batches, so recording from the live Week 3 inference loop
    never blocks the camera stream.

    Args:
        path: directory to write ``sequence_dataset_<n>.pt`` batches into.
        extractor: feature extractor used at flush time.
        max_pending: samples buffered in RAM before an automatic flush.
    """

    def __init__(
        self,
        path: str | Path,
        extractor: Optional[TemporalFeatureExtractor] = None,
        max_pending: int = 32,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._extractor = extractor or TemporalFeatureExtractor()
        self._max_pending = int(max_pending)
        self._pending: list[tuple[np.ndarray, int]] = []
        self._index = 0
        self._lock = threading.Lock()

    def add(self, sequence: np.ndarray, label: int) -> None:
        """Buffer one (sequence, label) pair; auto-flush when full."""
        with self._lock:
            self._pending.append(
                (np.asarray(sequence, dtype=np.float32), int(label))
            )
            if len(self._pending) >= self._max_pending:
                self._flush_locked()

    def flush(self) -> None:
        """Force-write any buffered samples to disk."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        windows, labels = zip(*self._pending)
        self._pending.clear()
        dest = self.path / f"sequence_dataset_{self._index:05d}.pt"
        save_dataset(
            dest,
            list(windows),
            labels=list(labels),
            extractor=self._extractor,
            fmt="pt",
        )
        self._index += 1

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


# ---------------------------------------------------------------------------
# 3. Rule-based heuristic baseline (early malpractice flagging)
# ---------------------------------------------------------------------------

@dataclass
class HeuristicBaseline:
    """
    Fast, stateless rule baseline computed on a window snapshot.

    Runs entirely on ``(T, 17, 2)`` arrays (typically the current deque
    contents of ``PoseWindowManager``) and is cheap enough to call every 5 FPS
    inference tick (< 1 ms) — it never adds streaming latency.

    Head turning (θ_head / head_turn_min_frames):
        persistent lateral displacement of the head keypoints (nose, eyes,
        ears) beyond `head_turn_theta` image-width units for more than
        `head_turn_min_frames` consecutive frames.  The neck->nose vector
        rotation is reported as an auxiliary, translation-invariant signal.

    Hand reaching / note passing:
        rapid outward trajectory of either wrist relative to the student's
        bounding-box centre: outward distance growth beyond `hand_reach_theta`
        sustained for `hand_reach_min_frames` consecutive frames.
    """

    head_turn_theta: float = 0.03       # lateral head-centroid shift (image width units)
    head_turn_theta_deg: float = 30.0   # |neck->nose angle| displacement threshold
    head_turn_min_frames: int = 10      # sustained frames required to flag
    hand_reach_theta: float = 0.08      # wrist outward growth (image width units)
    hand_reach_min_frames: int = 8      # sustained frames required to flag

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _validate(sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 3 or seq.shape[1:] != (17, 2):
            raise ValueError(f"expected (T, 17, 2), got shape {seq.shape}")
        return seq

    @staticmethod
    def max_consecutive_run(mask: np.ndarray) -> int:
        """
        Length of the longest consecutive True-run in a boolean mask.

        Vectorized via the transition points of a padded copy — no Python
        loop over the time axis.

        Args:
            mask: (T,) boolean array.

        Returns:
            Maximum run length in frames (0 when no True entries exist).
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0:
            return 0
        padded = np.concatenate(([False], mask, [False]))
        switches = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(switches == 1)
        ends = np.flatnonzero(switches == -1)
        runs = ends - starts
        return int(runs.max()) if runs.size else 0

    # -- detectors ----------------------------------------------------------

    def head_turn(
        self, sequence: np.ndarray, reference: Optional[np.ndarray] = None
    ) -> dict[str, object]:
        """
        Persistent lateral head movement over a window.

        Args:
            sequence: (T, 17, 2) normalised keypoints.
            reference: optional (2,) resting head centroid; defaults to the
                centroid at t=0 (streaming callers may pass a slow EMA base).

        Returns:
            dict with keys: head_centroid (T,2), lateral_disp (T,),
            mask_lateral (T,) bool, max_run (int),
            ang_disp (T,), mask_angle (T,) bool, max_run_angle (int).
        """
        seq = self._validate(sequence)
        head = seq[:, HEAD_KEYPOINTS, :].mean(axis=1)  # (T, 2)
        if reference is None:
            reference = head[0]
        lateral = head[:, 0] - float(reference[0])  # (T,)
        mask_lateral = np.abs(lateral) > self.head_turn_theta
        max_run = self.max_consecutive_run(mask_lateral)

        neck = (seq[:, LEFT_SHOULDER, :] + seq[:, RIGHT_SHOULDER, :]) / 2.0
        vec = seq[:, NOSE, :] - neck
        ang = np.arctan2(vec[:, 1], vec[:, 0])
        ang_disp = _wrap_angles(ang - ang[0])
        mask_angle = np.abs(ang_disp) > np.radians(self.head_turn_theta_deg)
        max_run_angle = self.max_consecutive_run(mask_angle)

        return {
            "head_centroid": head,
            "lateral_disp": lateral,
            "mask_lateral": mask_lateral,
            "max_run": max_run,
            "ang_disp": ang_disp,
            "mask_angle": mask_angle,
            "max_run_angle": max_run_angle,
        }

    def hand_reach(
        self, sequence: np.ndarray, bbox_center: Optional[np.ndarray] = None
    ) -> dict[str, object]:
        """
        Rapid outward wrist trajectory relative to the bounding-box centre.

        Args:
            sequence: (T, 17, 2) normalised keypoints.
            bbox_center: optional (2,) xy centre of the student's bounding
                box; when omitted the frame-0 shoulder midpoint is used as a
                stable body-centre proxy.

        Returns:
            dict with keys: outward (T,) max outward growth across wrists,
            mask (T,) bool, max_run (int), dist (2, T), center (2,).
        """
        seq = self._validate(sequence)
        if bbox_center is None:
            center = (seq[0, LEFT_SHOULDER, :] + seq[0, RIGHT_SHOULDER, :]) / 2.0
        else:
            center = np.asarray(bbox_center, dtype=np.float32)
            if center.shape != (2,):
                raise ValueError("bbox_center must be a (2,) xy pair")

        wrists = seq[:, [LEFT_WRIST, RIGHT_WRIST], :]  # (T, 2, 2)
        dist = np.linalg.norm(wrists - center[None, None, :], axis=-1)  # (T, 2)
        outward = dist - dist[:1]                        # (T, 2)
        max_outward = outward.max(axis=1)                # (T,)
        mask = max_outward > self.hand_reach_theta
        max_run = self.max_consecutive_run(mask)

        return {
            "outward": max_outward,
            "mask": mask,
            "max_run": max_run,
            "dist": dist,
            "center": center,
        }

    # -- public entry point -------------------------------------------------

    def evaluate(
        self,
        sequence: np.ndarray,
        bbox_center: Optional[np.ndarray] = None,
        reference: Optional[np.ndarray] = None,
    ) -> dict[str, object]:
        """
        Evaluate both baseline rules on one window.

        Args:
            sequence: (T, 17, 2) normalised keypoints.
            bbox_center: optional (2,) bounding-box centre (hand reaching).
            reference: optional (2,) resting head centroid (head turning).

        Returns:
            dict with boolean flags `head_turn`, `hand_reach`, `anomalous`
            plus diagnostic statistics (max runs, displacements).
        """
        head = self.head_turn(sequence, reference=reference)
        hand = self.hand_reach(sequence, bbox_center=bbox_center)

        head_turn = int(head["max_run"]) >= self.head_turn_min_frames
        hand_reach = int(hand["max_run"]) >= self.hand_reach_min_frames

        return {
            "head_turn": bool(head_turn),
            "head_turn_max_run": int(head["max_run"]),
            "head_turn_max_lateral": float(np.abs(head["lateral_disp"]).max()),
            "head_turn_max_yaw_deg": float(
                np.degrees(np.abs(head["ang_disp"])).max()
            ),
            "hand_reach": bool(hand_reach),
            "hand_reach_max_run": int(hand["max_run"]),
            "hand_reach_max_outward": float(np.abs(hand["outward"]).max()),
            "anomalous": bool(head_turn or hand_reach),
        }


# ---------------------------------------------------------------------------
# 4. Week 3 buffer integration — per-track sliding pose windows
# ---------------------------------------------------------------------------

@dataclass
class PoseWindowManager:
    """
    Thread-safe sliding-window pose buffer indexed by ByteTrack candidate_id.

    Mirrors the Week 3 RAM-buffer pattern (``collections.deque(maxlen=T)``)
    but stores normalised ``(17, 2)`` keypoint frames instead of encoded
    JPEGs.  ``push`` is an O(1) deque append, so feeding the 5 FPS inference
    path adds no measurable streaming latency; the < 1 ms baseline evaluation
    reads a ``np.stack`` snapshot of the deque.

    Args:
        window_size: number of pose frames kept per track (default 30 =
            ~6 s of history at the 5 FPS inference cadence).
    """

    window_size: int = 30
    _buffers: dict[int, deque[np.ndarray]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, track_id: int, keypoints_xy: np.ndarray) -> None:
        """
        Append one normalised (17, 2) keypoint frame to a track's window.

        O(1) — never copies the deque, never allocates beyond the deque's
        internal ring.  A brand-new track_id lazily creates its buffer.
        """
        frame = np.asarray(keypoints_xy, dtype=np.float32)
        if frame.shape != (17, 2):
            raise ValueError(f"expected (17, 2) frame, got {frame.shape}")
        with self._lock:
            self._buffers.setdefault(
                track_id, deque(maxlen=self.window_size)
            ).append(frame)

    def window(self, track_id: int) -> Optional[np.ndarray]:
        """
        Snapshot a track's current pose window.

        Returns:
            (T, 17, 2) float32 array (T = number of frames buffered, up to
            `window_size`), or None when the track has no buffered frames.
        """
        with self._lock:
            buf = self._buffers.get(track_id)
            if not buf:
                return None
            return np.stack(list(buf))

    def is_ready(self, track_id: int) -> bool:
        """True once a track has buffered a full (window_size-long) window."""
        with self._lock:
            buf = self._buffers.get(track_id)
            return buf is not None and len(buf) >= self.window_size

    def ready_count(self) -> int:
        """Number of tracks with a full window (for telemetry)."""
        with self._lock:
            return sum(
                1 for buf in self._buffers.values() if len(buf) >= self.window_size
            )

    def active_ids(self) -> list[int]:
        """Live track ids currently held in memory."""
        with self._lock:
            return list(self._buffers.keys())

    def drop(self, track_id: int) -> None:
        """Discard one track's window (used by stale-candidate GC)."""
        with self._lock:
            self._buffers.pop(track_id, None)

    def prune(self, keep_ids) -> int:
        """
        Drop every track not present in `keep_ids`; return count removed.

        Called by Week 3's garbage-collection sweep so pose buffers never
        outlive their ByteTrack candidates.
        """
        keep = {int(i) for i in keep_ids}
        with self._lock:
            stale = [cid for cid in self._buffers if cid not in keep]
            for cid in stale:
                self._buffers.pop(cid, None)
            return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffers)

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()


# ---------------------------------------------------------------------------
# Synthetic sequence generator (smoke tests)
# ---------------------------------------------------------------------------

def _canonical_pose() -> np.ndarray:
    """A plausible seated student pose in normalised image coordinates."""
    pts = {
        NOSE: (0.500, 0.180),
        LEFT_EYE: (0.506, 0.162), RIGHT_EYE: (0.494, 0.162),
        LEFT_EAR: (0.532, 0.184), RIGHT_EAR: (0.468, 0.184),
        LEFT_SHOULDER: (0.550, 0.320), RIGHT_SHOULDER: (0.450, 0.320),
        LEFT_ELBOW: (0.620, 0.500), RIGHT_ELBOW: (0.380, 0.500),
        LEFT_WRIST: (0.680, 0.660), RIGHT_WRIST: (0.320, 0.660),
        LEFT_HIP: (0.520, 0.620), RIGHT_HIP: (0.480, 0.620),
        LEFT_KNEE: (0.550, 0.820), RIGHT_KNEE: (0.450, 0.820),
        LEFT_ANKLE: (0.550, 0.950), RIGHT_ANKLE: (0.450, 0.950),
    }
    pose = np.zeros((17, 2), dtype=np.float32)
    for idx, (x, y) in pts.items():
        pose[idx] = (x, y)
    return pose


def make_synthetic_sequence(
    behavior: str = "normal", window_size: int = 30, seed: int = 0
) -> np.ndarray:
    """
    Generate a synthetic (window_size, 17, 2) pose sequence.

    Args:
        behavior: "normal" (small jitter), "head_turn" (sustained lateral
            head sweep from frame 12), or "hand_reach" (rapid right-wrist
            outward swing from frame 15).
        window_size: number of frames.
        seed: RNG seed for reproducibility.

    Returns:
        (window_size, 17, 2) float32 normalised keypoints.
    """
    rng = np.random.default_rng(seed)
    base = _canonical_pose()
    seq = np.repeat(base[None, :, :], window_size, axis=0).copy()
    seq += rng.normal(0.0, 0.004, size=seq.shape).astype(np.float32)

    if behavior == "head_turn":
        frames = np.arange(window_size)
        shift = np.where(
            frames >= 12, np.maximum(frames - 12, 0) * 0.045, 0.0
        )
        seq[:, HEAD_KEYPOINTS, 0] += shift[:, None]
    elif behavior == "hand_reach":
        frames = np.arange(window_size)
        f = np.where(frames >= 15, np.maximum(frames - 15, 0) * 0.03, 0.0)
        seq[:, RIGHT_WRIST, 0] -= f  # lateral outward swing away from body centre
    elif behavior != "normal":
        raise ValueError(f"unknown behavior: {behavior!r}")
    return np.asarray(seq, dtype=np.float32)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _run_smoke_tests() -> None:
    import tempfile

    T = 30
    normal = make_synthetic_sequence("normal", T)
    turn = make_synthetic_sequence("head_turn", T)
    reach = make_synthetic_sequence("hand_reach", T)

    extractor = TemporalFeatureExtractor(window_size=T)
    baseline = HeuristicBaseline()

    # -- 1. per-frame + summary feature extraction --------------------------
    f_norm = extractor.extract_features(normal)
    assert f_norm.shape == (T, FEATURE_DIM), f_norm.shape
    s_norm = extractor.summarize(normal)
    assert s_norm.shape == (SUMMARY_DIM,), s_norm.shape

    f_batch = extractor.extract_batch(np.stack([normal, turn, reach]))
    assert f_batch.shape == (3, T, FEATURE_DIM), f_batch.shape
    assert np.allclose(f_batch[0], f_norm), "batch/single extraction mismatch"
    print(f"[test] per-frame features (T,F)={f_norm.shape}, summary (S,)={s_norm.shape}")

    # -- 2. baseline heuristics ---------------------------------------------
    r_norm = baseline.evaluate(normal)
    r_turn = baseline.evaluate(turn)
    r_reach = baseline.evaluate(reach)
    assert not r_norm["anomalous"], f"normal falsely flagged: {r_norm}"
    assert r_turn["head_turn"], f"head turn missed: {r_turn}"
    assert r_reach["hand_reach"], f"hand reach missed: {r_reach}"
    print(f"[test] normal      -> {r_norm}")
    print(f"[test] head_turn   -> {r_turn}")
    print(f"[test] hand_reach  -> {r_reach}")

    # -- 3. sequence tensor + dataset round-trip -----------------------------
    windows = [normal, turn, reach]
    labels = [0, 1, 1]
    X, y = build_sequence_tensor(windows, extractor=extractor, labels=labels)
    assert X.shape == (3, T, FEATURE_DIM), X.shape
    assert y.shape == (3,), y.shape

    with tempfile.TemporaryDirectory() as tmp:
        pt_path = save_dataset(
            Path(tmp) / "seq.pt", windows, labels, extractor=extractor, fmt="pt"
        )
        data_pt = load_dataset(pt_path)
        assert data_pt["X"].shape == (3, T, FEATURE_DIM)
        assert data_pt["labels"].tolist() == [0, 1, 1]

        npz_path = save_dataset(
            Path(tmp) / "seq.npz", windows, labels, extractor=extractor, fmt="npz"
        )
        data_npz = load_dataset(npz_path)
        assert data_npz["X"].shape == (3, T, FEATURE_DIM)
        print(f"[test] dataset round-trip OK (.pt={pt_path.name}, .npz={npz_path.name})")

        # -- 4. incremental writer + torch dataset --------------------------
        writer = SequenceDatasetWriter(
            Path(tmp) / "live", extractor=extractor, max_pending=4
        )
        for i in range(10):
            writer.add(windows[i % 3], labels[i % 3])
        writer.flush()
        batches = sorted((Path(tmp) / "live").glob("*.pt"))
        assert batches, "writer produced no batches"
        ds = PoseSequenceDataset(data_pt["X"], data_pt.get("labels"))
        assert len(ds) == 3 and ds[0][0].shape == (T, FEATURE_DIM)
        print(f"[test] writer produced {len(batches)} batch(es); dataset len={len(ds)}")

    # -- 5. PoseWindowManager streaming integration --------------------------
    mgr = PoseWindowManager(window_size=T)
    assert not mgr.is_ready(7)
    for t in range(T):
        mgr.push(7, turn[t])  # O(1) append per inference tick
    assert mgr.is_ready(7)
    assert mgr.window(7).shape == (T, 17, 2)
    r_stream = baseline.evaluate(mgr.window(7))
    assert r_stream["head_turn"], "streaming window not flagged"
    assert mgr.prune(keep_ids={}) == 1
    assert len(mgr) == 0
    print("[test] PoseWindowManager streaming window flagged head_turn")

    print("\nALL WEEK 4 SMOKE TESTS PASSED")


if __name__ == "__main__":
    _run_smoke_tests()
