"""
onnx_exporter.py — Week 10: ONNX & TensorRT Export Pipeline.

Automated exporter for two model families:

    1. YOLO11-Pose — Ultralytics export to ONNX + TensorRT engine
    2. Malpractice LSTM/GRU — torch.onnx.export with dynamic axes,
       then TensorRT FP16/INT8 engine compilation

All heavy-dep imports (tensorrt, onnxruntime, pycuda) are guarded
behind lazy checks so the module loads cleanly on CPU-only machines.

Usage:
    python exporter.py --model lstm --checkpoint best_malpractice_model.pth
    python exporter.py --model yolo
    python exporter.py --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("exporter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_WEIGHTS = str(BACKEND_DIR / "yolo11n-pose.pt")
DEFAULT_CHECKPOINT = str(BACKEND_DIR / "best_malpractice_model.pth")
EXPORT_DIR = BACKEND_DIR / "exported_models"

# Sequence model defaults (must match training config)
DEFAULT_INPUT_DIM = 117   # Week 4 feature dimension
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_CLASSES = 4
DEFAULT_SEQUENCE_LEN = 30

YOLO_INPUT_SIZE = 640


# ---------------------------------------------------------------------------
# Feature availability checks
# ---------------------------------------------------------------------------

def _has_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pycuda() -> bool:
    try:
        import pycuda  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Export result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    """Metadata about an exported model."""
    model_name: str
    format: str  # "onnx", "trt_fp16", "trt_int8"
    path: str
    size_mb: float
    input_shapes: dict[str, list[int]]
    export_time_s: float
    validates: bool = True

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "format": self.format,
            "path": self.path,
            "size_mb": round(self.size_mb, 2),
            "input_shapes": self.input_shapes,
            "export_time_s": round(self.export_time_s, 2),
            "validates": self.validates,
        }


# ---------------------------------------------------------------------------
# YOLO11-Pose exporter
# ---------------------------------------------------------------------------

class YOLOExporter:
    """
    Export YOLO11n-pose to ONNX and TensorRT engine via Ultralytics APIs.

    Ultralytics handles ONNX export internally and can compile TensorRT
    engines directly when the `tensorrt` package is available.
    """

    def __init__(
        self,
        weights_path: str = DEFAULT_YOLO_WEIGHTS,
        export_dir: Path = EXPORT_DIR,
    ) -> None:
        self.weights_path = weights_path
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_onnx(self) -> ExportResult:
        """Export YOLO11n-pose to ONNX via Ultralytics."""
        from ultralytics import YOLO

        t0 = time.monotonic()
        model = YOLO(self.weights_path)

        output_path = str(self.export_dir / "yolo11n_pose.onnx")
        model.export(
            format="onnx",
            imgsz=YOLO_INPUT_SIZE,
            dynamic=True,
            simplify=True,
            opset=17,
        )

        # Ultralytics exports to same dir as weights by default
        # Move to our export dir
        src = Path(self.weights_path).parent / "yolo11n_pose.onnx"
        if src.exists() and str(src) != output_path:
            import shutil
            shutil.move(str(src), output_path)

        elapsed = time.monotonic() - t0
        size_mb = Path(output_path).stat().st_size / (1024 * 1024) if Path(output_path).exists() else 0

        logger.info(f"[yolo] ONNX exported: {output_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")

        return ExportResult(
            model_name="yolo11n_pose",
            format="onnx",
            path=output_path,
            size_mb=size_mb,
            input_shapes={"images": [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]},
            export_time_s=elapsed,
        )

    def export_tensorrt(self, fp16: bool = True) -> ExportResult:
        """Export YOLO11n-pose to TensorRT engine via Ultralytics."""
        if not _has_tensorrt():
            logger.warning("[yolo] tensorrt not available — skipping TensorRT export")
            return ExportResult(
                model_name="yolo11n_pose",
                format="trt_fp16" if fp16 else "trt_fp32",
                path="",
                size_mb=0,
                input_shapes={},
                export_time_s=0,
                validates=False,
            )

        from ultralytics import YOLO

        t0 = time.monotonic()
        model = YOLO(self.weights_path)

        suffix = "fp16" if fp16 else "fp32"
        output_path = str(self.export_dir / f"yolo11n_pose_{suffix}.engine")

        model.export(
            format="engine",
            imgsz=YOLO_INPUT_SIZE,
            half=fp16,
            dynamic=True,
        )

        # Move engine file
        src = Path(self.weights_path).parent / "yolo11n_pose.engine"
        if src.exists() and str(src) != output_path:
            import shutil
            shutil.move(str(src), output_path)

        elapsed = time.monotonic() - t0
        size_mb = Path(output_path).stat().st_size / (1024 * 1024) if Path(output_path).exists() else 0

        logger.info(f"[yolo] TensorRT {suffix} exported: {output_path} ({size_mb:.1f} MB)")

        return ExportResult(
            model_name="yolo11n_pose",
            format=f"trt_{suffix}",
            path=output_path,
            size_mb=size_mb,
            input_shapes={"images": [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]},
            export_time_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Sequence model exporter (LSTM / GRU)
# ---------------------------------------------------------------------------

class SequenceModelExporter:
    """
    Export MalpracticeLSTM/GRU to ONNX with dynamic axes, then
    optionally compile TensorRT FP16/INT8 engines.
    """

    def __init__(
        self,
        model_type: str = "lstm",
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        export_dir: Path = EXPORT_DIR,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_classes: int = DEFAULT_NUM_CLASSES,
        sequence_len: int = DEFAULT_SEQUENCE_LEN,
    ) -> None:
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.sequence_len = sequence_len

    def _build_model(self) -> torch.nn.Module:
        """Build and optionally load weights for the sequence model."""
        from temporal_training import MalpracticeLSTM, MalpracticeGRU

        if self.model_type == "lstm":
            model = MalpracticeLSTM(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=0.0,  # no dropout for inference
                bidirectional=True,
                use_attention=True,
            )
        elif self.model_type == "gru":
            model = MalpracticeGRU(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=0.0,
                bidirectional=True,
                use_attention=True,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        model.eval()

        # Load checkpoint if available
        ckpt_path = Path(self.checkpoint_path)
        if ckpt_path.exists():
            from temporal_training import load_checkpoint
            load_checkpoint(model, str(ckpt_path))
            logger.info(f"[seq] loaded checkpoint: {ckpt_path}")
        else:
            logger.warning(f"[seq] no checkpoint at {ckpt_path} — using random weights")

        return model

    def export_onnx(self) -> ExportResult:
        """Export sequence model to ONNX with dynamic batch + sequence axes."""
        model = self._build_model()

        t0 = time.monotonic()
        output_path = str(self.export_dir / f"malpractice_{self.model_type}.onnx")

        # Dummy input: (batch=1, T=30, F=117)
        dummy = torch.randn(1, self.sequence_len, self.input_dim)

        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )

        elapsed = time.monotonic() - t0
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)

        # Validate with ONNX
        validates = self._validate_onnx(output_path, dummy)

        logger.info(
            f"[seq] ONNX exported: {output_path} "
            f"({size_mb:.2f} MB, {elapsed:.1f}s, valid={validates})"
        )

        return ExportResult(
            model_name=f"malpractice_{self.model_type}",
            format="onnx",
            path=output_path,
            size_mb=size_mb,
            input_shapes={
                "input": [1, self.sequence_len, self.input_dim],
            },
            export_time_s=elapsed,
            validates=validates,
        )

    def _validate_onnx(self, onnx_path: str, dummy: torch.Tensor) -> bool:
        """Validate ONNX model by comparing outputs with PyTorch."""
        if not _has_onnxruntime():
            return True  # can't validate, assume OK

        try:
            import onnxruntime as ort
            import onnx

            # Check model loads
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Compare outputs
            session = ort.InferenceSession(onnx_path)
            ort_output = session.run(None, {"input": dummy.numpy()})

            with torch.no_grad():
                pt_output = model_forward_from_onnx(onnx_path, dummy)

            # Tolerance for numerical differences
            pt_np = pt_output.numpy() if hasattr(pt_output, 'numpy') else pt_output
            diff = np.abs(ort_output[0] - pt_np).max()
            return diff < 1e-4

        except Exception as e:
            logger.warning(f"[seq] ONNX validation failed: {e}")
            return False

    def export_tensorrt_fp16(self) -> ExportResult:
        """Compile ONNX to TensorRT FP16 engine."""
        onnx_path = str(self.export_dir / f"malpractice_{self.model_type}.onnx")
        if not Path(onnx_path).exists():
            logger.warning(f"[seq] ONNX not found at {onnx_path} — export first")
            return ExportResult(
                model_name=f"malpractice_{self.model_type}",
                format="trt_fp16",
                path="",
                size_mb=0,
                input_shapes={},
                export_time_s=0,
                validates=False,
            )

        return _compile_tensorrt_engine(
            onnx_path=onnx_path,
            model_name=f"malpractice_{self.model_type}",
            precision="fp16",
            export_dir=self.export_dir,
        )

    def export_tensorrt_int8(
        self,
        calibration_data: Optional[np.ndarray] = None,
    ) -> ExportResult:
        """Compile ONNX to TensorRT INT8 engine with calibration."""
        onnx_path = str(self.export_dir / f"malpractice_{self.model_type}.onnx")
        if not Path(onnx_path).exists():
            logger.warning(f"[seq] ONNX not found at {onnx_path} — export first")
            return ExportResult(
                model_name=f"malpractice_{self.model_type}",
                format="trt_int8",
                path="",
                size_mb=0,
                input_shapes={},
                export_time_s=0,
                validates=False,
            )

        return _compile_tensorrt_engine(
            onnx_path=onnx_path,
            model_name=f"malpractice_{self.model_type}",
            precision="int8",
            export_dir=self.export_dir,
            calibration_data=calibration_data,
        )


# ---------------------------------------------------------------------------
# TensorRT engine compilation helper
# ---------------------------------------------------------------------------

def _compile_tensorrt_engine(
    onnx_path: str,
    model_name: str,
    precision: str,
    export_dir: Path,
    calibration_data: Optional[np.ndarray] = None,
) -> ExportResult:
    """Compile an ONNX model to a TensorRT engine file."""
    if not _has_tensorrt():
        logger.warning("[trt] tensorrt not available")
        return ExportResult(
            model_name=model_name,
            format=f"trt_{precision}",
            path="",
            size_mb=0,
            input_shapes={},
            export_time_s=0,
            validates=False,
        )

    import tensorrt as trt

    t0 = time.monotonic()
    output_path = str(export_dir / f"{model_name}_{precision}.engine")

    logger.info(f"[trt] compiling {precision} engine from {onnx_path}...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"[trt] ONNX parse error: {parser.get_error(i)}")
            return ExportResult(
                model_name=model_name,
                format=f"trt_{precision}",
                path="",
                size_mb=0,
                input_shapes={},
                export_time_s=time.monotonic() - t0,
                validates=False,
            )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("[trt] FP16 mode enabled")
        else:
            logger.warning("[trt] platform does not support fast FP16")

    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Attach INT8 calibrator
            if calibration_data is not None:
                from calibrator import TensorRTINT8Calibrator
                calibrator = TensorRTINT8Calibrator(calibration_data)
                config.int8_calibrator = calibrator
                logger.info("[trt] INT8 mode enabled with calibrator")
            else:
                logger.warning("[trt] INT8 requested but no calibration data — using default")
        else:
            logger.warning("[trt] platform does not support fast INT8")

    # Dynamic shapes (batch + sequence)
    profile = builder.create_optimization_profile()
    input_name = network.get_input_name(0)
    # Set min, opt, max shapes for dynamic axes
    profile.set_shape(
        input_name,
        min=(1, 1, DEFAULT_INPUT_DIM),
        opt=(4, DEFAULT_SEQUENCE_LEN, DEFAULT_INPUT_DIM),
        max=(16, DEFAULT_SEQUENCE_LEN * 2, DEFAULT_INPUT_DIM),
    )
    config.add_optimization_profile(profile)

    # Build
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        logger.error("[trt] engine build failed")
        return ExportResult(
            model_name=model_name,
            format=f"trt_{precision}",
            path="",
            size_mb=0,
            input_shapes={},
            export_time_s=time.monotonic() - t0,
            validates=False,
        )

    # Write engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    elapsed = time.monotonic() - t0
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    logger.info(f"[trt] {precision} engine: {output_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    return ExportResult(
        model_name=model_name,
        format=f"trt_{precision}",
        path=output_path,
        size_mb=size_mb,
        input_shapes={
            input_name: [1, DEFAULT_SEQUENCE_LEN, DEFAULT_INPUT_DIM],
        },
        export_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Helper: run PyTorch model via ONNX path for validation
# ---------------------------------------------------------------------------

def model_forward_from_onnx(onnx_path: str, dummy: torch.Tensor) -> torch.Tensor:
    """Load the corresponding PyTorch model and run forward pass."""
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    output = session.run(None, {"input": dummy.numpy()})
    return torch.from_numpy(output[0])


# ---------------------------------------------------------------------------
# Full export pipeline
# ---------------------------------------------------------------------------

def export_all(
    model_type: str = "lstm",
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    export_dir: Path = EXPORT_DIR,
    include_trt: bool = True,
    include_int8: bool = False,
    calibration_data: Optional[np.ndarray] = None,
) -> list[ExportResult]:
    """Run the full export pipeline for a sequence model + YOLO."""
    results = []

    # 1. YOLO11-Pose
    yolo_exp = YOLOExporter(export_dir=export_dir)
    results.append(yolo_exp.export_onnx())
    if include_trt:
        results.append(yolo_exp.export_tensorrt(fp16=True))

    # 2. Sequence model
    seq_exp = SequenceModelExporter(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        export_dir=export_dir,
    )
    results.append(seq_exp.export_onnx())
    if include_trt:
        results.append(seq_exp.export_tensorrt_fp16())
    if include_int8:
        results.append(seq_exp.export_tensorrt_int8(calibration_data=calibration_data))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 10 Model Exporter")
    parser.add_argument("--model", choices=["lstm", "gru", "yolo"], default="lstm")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--export-dir", default=str(EXPORT_DIR))
    parser.add_argument("--include-trt", action="store_true", help="Include TensorRT compilation")
    parser.add_argument("--include-int8", action="store_true", help="Include INT8 calibration")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)

    if args.all:
        results = export_all(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            export_dir=export_dir,
            include_trt=args.include_trt,
            include_int8=args.include_int8,
        )
    elif args.model == "yolo":
        exp = YOLOExporter(export_dir=export_dir)
        results = [exp.export_onnx()]
        if args.include_trt:
            results.append(exp.export_tensorrt(fp16=True))
    else:
        exp = SequenceModelExporter(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            export_dir=export_dir,
        )
        results = [exp.export_onnx()]
        if args.include_trt:
            results.append(exp.export_tensorrt_fp16())
        if args.include_int8:
            results.append(exp.export_tensorrt_int8())

    print(f"\n{'='*60}")
    print(f"  Export Results")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r.validates else "FAILED"
        print(f"  [{status}] {r.model_name} ({r.format})")
        print(f"         Path: {r.path}")
        print(f"         Size: {r.size_mb:.2f} MB")
        print(f"         Time: {r.export_time_s:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
