# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the IntelliProctor desktop application.

Build with::

    python tools/build_release.py --clean

The release script runs PyInstaller against this spec and then creates the
in-bundle shortcut (and, optionally, a Desktop shortcut).

Layout notes
------------
PyInstaller 6 places collected data in ``dist/IntelliProctor/_internal``.
The application resolves its resources from ``Path(__file__)``, which points
into that directory at runtime, so ``assets``, ``models`` and ``backend`` are
copied there under the same names.  The executable is deliberately a one-dir
build: PyTorch, CUDA, MediaPipe and PyQt6 have large native libraries, and a
one-file archive would add a slow extraction step to every launch.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).resolve()
APP_NAME = "IntelliProctor"

datas = []
binaries = []
hiddenimports = []


def _add_metadata(distribution: str) -> None:
    """Copy package metadata when the distribution is installed."""
    try:
        datas.extend(copy_metadata(distribution))
    except Exception:
        # Optional metadata must never make an otherwise valid build fail.
        pass


def _add_file(path: Path, destination: str, *, required: bool = False) -> None:
    if path.is_file():
        datas.append((str(path), destination))
        return
    message = f"IntelliProctor spec: missing {'required ' if required else ''}asset: {path}"
    print(message)
    if required:
        raise SystemExit(message)


# ---------------------------------------------------------------------------
# Third-party model runtimes
# ---------------------------------------------------------------------------
# Ultralytics resolves task/tracker modules dynamically, and MediaPipe loads
# its task graph through native bindings.  collect_all captures the Python
# submodules, data files and shared libraries that static import analysis
# cannot see.
for package in ("ultralytics", "mediapipe"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

for distribution in (
    "ultralytics",
    "mediapipe",
    "torch",
    "torchvision",
    "scikit-learn",
    "PyQt6",
    "opencv-python",
    "numpy",
    "pandas",
    "joblib",
    "lap",
):
    _add_metadata(distribution)

hiddenimports += [
    "backend.runtime",
    "lap",
    # RandomForest artifacts are unpickled at runtime.  These are the
    # scikit-learn modules the fitted tree ensemble needs, and naming them
    # explicitly keeps a future PyInstaller hook change from silently
    # disabling the P01/P20 classifier.
    "sklearn.ensemble._forest",
    "sklearn.tree._classes",
    "sklearn.tree._tree",
    "sklearn.utils._typedefs",
]

# ---------------------------------------------------------------------------
# Application resources
# ---------------------------------------------------------------------------
# Branding.  The executable embeds the .ico; the copy in the bundle lets
# QApplication use the same image for the window/taskbar icon.
_add_file(ROOT / "assets" / "IntelliProctor.ico", "assets")

# MediaPipe pose landmarkers.  The app selects a variant through
# IP_MP_MODEL_VARIANT (default: heavy).
for task_file in sorted((ROOT / "assets").glob("pose_landmarker_*.task")):
    datas.append((str(task_file), "assets"))

# The trained P01/P20 RandomForest and its committed metrics sidecar.
_add_file(ROOT / "models" / "pose_rf_classifier.pkl", "models", required=True)
_add_file(
    ROOT / "models" / "pose_rf_classifier.pkl.metrics.json",
    "models",
)

# SCUT-HEAD one-class head detector used for seat occupancy/anchors.
_add_file(
    ROOT / "models" / "head" / "scut_head_yolo11n.pt",
    "models/head",
    required=True,
)

# Pose weights and the COCO fallback used for desk objects.
_add_file(ROOT / "backend" / "yolo11n-pose.pt", "backend", required=True)
_add_file(ROOT / "backend" / "yolo11n.pt", "backend")

# Adopt a custom desk model automatically when one has been trained locally.
_add_file(ROOT / "models" / "desk" / "desk_objects.pt", "models/desk")

# ---------------------------------------------------------------------------
# Analysis / bundle
# ---------------------------------------------------------------------------
a = Analysis(
    [str(ROOT / "desktop_app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # The desktop entry point does not import the web dashboard or the
        # training/evaluation stack.  Excluding them keeps the bundle smaller
        # without touching any runtime import of the Qt application.
        "flask",
        "fastapi",
        "starlette",
        "uvicorn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

icon_file = ROOT / "assets" / "IntelliProctor.ico"
version_file = ROOT / "assets" / "IntelliProctor_version.txt"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_file) if icon_file.is_file() else None,
    version=str(version_file) if version_file.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
