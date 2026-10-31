"""
stgcn.py — Spatio-Temporal Graph Convolutional Network for skeleton-based
action recognition in exam malpractice detection.

Detects multi-person interactions (passing papers, looking at neighbour's
screen, collaboration) by modelling skeleton joint dynamics over a temporal
window using graph convolutions.

COCO 17-keypoint skeleton topology:
    Edges connect anatomically adjacent joints (nose→eyes→ears→shoulders→
    elbows→wrists, shoulders→hips→knees→ankles).

Architecture:
    3 × SpatialTemporalBlock(C_in, C_out, V=17)
    → Global average pooling over T and V
    → FC → num_classes

Input:  (batch, C=3, T=30, V=17)  — 30 frames × 17 joints × (x, y, conf)
Output: (batch, num_classes)       — logits for [normal, interaction, ...]
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from backend.runtime import autocast, is_cuda, optimize_cuda, resolve_device
except ImportError:  # pragma: no cover - direct module execution
    from runtime import (  # type: ignore[no-redef]
        autocast,
        is_cuda,
        optimize_cuda,
        resolve_device,
    )


# -----------------------------------------------------------------------
# COCO 17-keypoint skeleton graph adjacency
# -----------------------------------------------------------------------

# Each tuple (i, j) represents a bidirectional edge between keypoints.
_COCO_EDGES = [
    (0, 1), (0, 2),       # nose ↔ left_eye, nose ↔ right_eye
    (1, 3), (2, 4),       # left_eye ↔ left_ear, right_eye ↔ right_ear
    (0, 5), (0, 6),       # nose ↔ left_shoulder, nose ↔ right_shoulder
    (5, 6),               # left_shoulder ↔ right_shoulder
    (5, 7), (7, 9),       # left_shoulder ↔ left_elbow ↔ left_wrist
    (6, 8), (8, 10),      # right_shoulder ↔ right_elbow ↔ right_wrist
    (5, 11), (6, 12),     # shoulders ↔ hips
    (11, 12),             # left_hip ↔ right_hip
    (11, 13), (13, 15),   # left_hip ↔ left_knee ↔ left_ankle
    (12, 14), (14, 16),   # right_hip ↔ right_knee ↔ right_ankle
]

NUM_JOINTS = 17
NUM_CLASSES = 4  # normal, interaction, paper_pass, phone_pass


def _build_adjacency(num_joints: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Build a symmetric adjacency matrix with self-loops."""
    adj = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


ADJACENCY = _build_adjacency(NUM_JOINTS, _COCO_EDGES)


# -----------------------------------------------------------------------
# Graph Convolution layer
# -----------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """Spectral graph convolution: H' = σ(A · H · W).

    A: (V, V) adjacency matrix (fixed, not learned)
    H: (batch, C, T, V) input features
    W: (C_in, C_out) learnable weight matrix
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch, C_in, T, V)
            adj: (V, V) adjacency matrix
        Returns:
            (batch, C_out, T, V)
        """
        # x: (B, C, T, V) -> (B, T, V, C) -> matmul with adj
        # Support = adj @ x  (aggregate neighbor features)
        # x @ weight + bias
        x_perm = x.permute(0, 2, 3, 1)  # (B, T, V, C_in)
        support = torch.matmul(adj, x_perm)  # (B, T, V, C_in)
        out = torch.matmul(support, self.weight) + self.bias  # (B, T, V, C_out)
        return out.permute(0, 3, 1, 2)  # (B, C_out, T, V)


# -----------------------------------------------------------------------
# Spatial Temporal Block
# -----------------------------------------------------------------------

class SpatialTemporalBlock(nn.Module):
    """One block = GraphConv (spatial) + 1D Conv (temporal) + residual + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.spatial_conv = GraphConvolution(in_channels, out_channels)
        # Temporal convolution: kernel=9, padding=4 to preserve T
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(9, 1), padding=(4, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual projection if channels change
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.spatial_conv(x, adj)
        out = self.temporal_conv(out)
        out = self.bn(out)
        out = self.relu(out + res)
        return out


# -----------------------------------------------------------------------
# ST-GCN Model
# -----------------------------------------------------------------------

class STGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network for skeleton action recognition.

    Architecture:
        SpatialTemporalBlock(3, 64)  → 64 channels
        SpatialTemporalBlock(64, 64) → 64 channels
        SpatialTemporalBlock(64, 128) → 128 channels
        Global average pool over (T, V)
        FC → num_classes
    """

    def __init__(
        self,
        num_joints: int = NUM_JOINTS,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_classes = num_classes

        self.block1 = SpatialTemporalBlock(in_channels, 64)
        self.block2 = SpatialTemporalBlock(64, 64)
        self.block3 = SpatialTemporalBlock(64, 128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch, 3, T, V) — skeleton sequence
            adj: (V, V) adjacency matrix
        Returns:
            (batch, num_classes) logits
        """
        out = self.block1(x, adj)
        out = self.block2(out, adj)
        out = self.block3(out, adj)
        out = self.global_pool(out)  # (B, 128, 1, 1)
        out = out.view(out.size(0), -1)  # (B, 128)
        out = self.fc(out)  # (B, num_classes)
        return out


# -----------------------------------------------------------------------
# Wrapper for inference with automatic device selection
# -----------------------------------------------------------------------

class STGCNClassifier:
    """High-level wrapper: loads model, runs inference on skeleton sequences.

    Usage:
        clf = STGCNClassifier()
        result = clf.predict(skeleton_sequence)  # returns (label, confidence)
    """

    LABELS = ["normal", "interaction", "paper_pass", "phone_pass"]

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.device = resolve_device("auto")
        optimize_cuda(self.device)
        self.model = STGCN().to(self.device)
        self.model.eval()
        self._amp = is_cuda(self.device)

        # Pre-built adjacency matrix as tensor
        self.adj = torch.tensor(ADJACENCY, dtype=torch.float32, device=self.device)

        if model_path is not None:
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self._has_weights = True
            except Exception:
                self._has_weights = False
        else:
            self._has_weights = False

    @torch.inference_mode()
    def predict(self, skeleton_sequence: np.ndarray) -> tuple[str, float]:
        """Classify a skeleton sequence.

        Args:
            skeleton_sequence: (T, V, 3) array — T frames, V joints, (x, y, conf)
                               Normalised coordinates expected.

        Returns:
            (label, confidence) tuple.
        """
        if not self._has_weights:
            return ("normal", 0.0)

        # Prepare tensor: (1, 3, T, V).  from_numpy avoids an intermediate
        # Python list; non_blocking overlaps the copy with the first kernel.
        seq = torch.from_numpy(
            np.ascontiguousarray(skeleton_sequence, dtype=np.float32)
        ).to(self.device, non_blocking=is_cuda(self.device))
        if seq.ndim == 3:
            seq = seq.permute(2, 0, 1).unsqueeze(0)  # (1, C, T, V)
        elif seq.ndim == 4:
            seq = seq.unsqueeze(0)

        with autocast(self.device, enabled=self._amp):
            logits = self.model(seq, self.adj)
        probs = F.softmax(logits.float(), dim=-1)
        conf, idx = probs.max(dim=-1)
        label = self.LABELS[idx.item()] if idx.item() < len(self.LABELS) else "unknown"
        return (label, conf.item())

    @torch.no_grad()
    def predict_batch(self, sequences: list[np.ndarray]) -> list[tuple[str, float]]:
        """Classify multiple skeleton sequences."""
        return [self.predict(seq) for seq in sequences]
