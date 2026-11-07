"""Small shared vision-geometry helpers.

Ultralytics derives its internal tensor shape from the aspect ratio of the
input image.  Changing that shape triggers a one-off CUDA/cuDNN autotune that
can cost seconds.  Live crops (desk bands, candidate ROIs) change shape every
frame, so both desk-object passes letterbox onto a fixed square canvas before
calling the model and undo the transform afterwards.

:class:`LetterboxBuffer` owns that canvas for the lifetime of a worker.  The
resize writes directly into the canvas ROI (OpenCV accepts a strided ``dst``),
so the steady state performs one resize and no heap allocation.  Callers must
consume the returned canvas before the next ``apply`` on the same buffer.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def configure_opencv_threads(num_threads: int | None = None) -> int:
    """Pin OpenCV's internal thread pool, defaulting to four workers.

    OpenCV otherwise creates one worker per core for every resize/colour
    conversion.  Those workers compete with PyTorch's own intra-op pool and
    with the Qt UI thread, which on a small machine shows up as latency
    spikes rather than throughput.  ``IP_CV_THREADS`` overrides the default;
    ``0`` restores OpenCV's automatic policy.
    """
    if num_threads is None:
        try:
            num_threads = int(os.environ.get("IP_CV_THREADS", "4"))
        except ValueError:
            num_threads = 4
    num_threads = max(0, int(num_threads))
    cv2.setNumThreads(num_threads)
    return num_threads


class LetterboxBuffer:
    """Reusable square canvas for :func:`letterbox_square`."""

    __slots__ = ("size", "pad_value", "canvas")

    def __init__(self, size: int, *, pad_value: int = 114) -> None:
        self.size = int(size)
        self.pad_value = int(pad_value)
        # ``uint8`` matches every BGR frame/crop in this project.  If a caller
        # supplies another dtype, letterbox_square falls back to a fresh
        # canvas rather than silently truncating values.
        self.canvas = np.full(
            (self.size, self.size, 3), self.pad_value, dtype=np.uint8
        )

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox *image* into the owned canvas and return the transform."""
        return letterbox_square(
            image,
            self.size,
            pad_value=self.pad_value,
            canvas=self.canvas,
        )


def letterbox_square(
    image: np.ndarray,
    size: int,
    *,
    pad_value: int = 114,
    canvas: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int, int]:
    """Resize *image* onto a fixed ``size x size`` canvas.

    Returns ``(canvas, scale, pad_x, pad_y)``.  ``scale`` maps original
    coordinates to canvas coordinates; ``pad_x/pad_y`` are the top-left
    offsets of the resized image inside the canvas.

    When *canvas* is supplied it must be a writable ``(size, size, 3)``
    ``uint8`` array owned by the caller.  It is reused in place: the padding
    is reset and the resize writes directly into the interior ROI.  The
    returned array is the same object, so consume it before the next call.
    """
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image, 1.0, 0, 0

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] != 3:
        raise ValueError(f"unsupported image shape: {image.shape}")

    scale = min(size / float(width), size / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    reuse = (
        canvas is not None
        and canvas.shape == (size, size, 3)
        and canvas.dtype == image.dtype
        and canvas.flags["C_CONTIGUOUS"]
        and canvas.flags["WRITEABLE"]
    )
    if not reuse:
        canvas = np.full((size, size, 3), pad_value, dtype=image.dtype)
    else:
        # Reset only the padding.  The interior is fully overwritten by the
        # resize below, so clearing the whole canvas would be wasted bandwidth.
        if pad_y > 0:
            canvas[:pad_y, :, :] = pad_value
        if pad_y + new_h < size:
            canvas[pad_y + new_h:, :, :] = pad_value
        if pad_x > 0:
            canvas[pad_y:pad_y + new_h, :pad_x, :] = pad_value
        if pad_x + new_w < size:
            canvas[pad_y:pad_y + new_h, pad_x + new_w:, :] = pad_value

    destination = canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    try:
        # OpenCV accepts a strided destination view; this is the zero-copy
        # path.  Older/odd builds may reject it, so fall back to a resize that
        # allocates rather than failing the frame.
        cv2.resize(
            image,
            (new_w, new_h),
            dst=destination,
            interpolation=cv2.INTER_LINEAR,
        )
    except cv2.error:
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    return canvas, scale, pad_x, pad_y


def unletterbox_box(
    box: np.ndarray | list[float] | tuple[float, ...],
    scale: float,
    pad_x: int,
    pad_y: int,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> list[int]:
    """Map one ``xyxy`` box from canvas space back to full-frame pixels."""
    if scale <= 0:
        scale = 1.0
    x1 = int(round((float(box[0]) - pad_x) / scale)) + int(offset_x)
    y1 = int(round((float(box[1]) - pad_y) / scale)) + int(offset_y)
    x2 = int(round((float(box[2]) - pad_x) / scale)) + int(offset_x)
    y2 = int(round((float(box[3]) - pad_y) / scale)) + int(offset_y)
    if frame_width is not None:
        x1 = max(0, min(x1, int(frame_width)))
        x2 = max(0, min(x2, int(frame_width)))
    if frame_height is not None:
        y1 = max(0, min(y1, int(frame_height)))
        y2 = max(0, min(y2, int(frame_height)))
    return [x1, y1, x2, y2]


__all__ = [
    "LetterboxBuffer",
    "configure_opencv_threads",
    "letterbox_square",
    "unletterbox_box",
]
