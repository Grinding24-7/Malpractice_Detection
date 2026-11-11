#!/usr/bin/env python3
"""Build IntelliProctor with PyInstaller and create the release shortcuts.

Usage
-----
Build the bundle and create the shortcut inside it::

    python tools/build_release.py --clean

Also create a shortcut on the current user's Desktop::

    python tools/build_release.py --clean --desktop

The Desktop shortcut is opt-in so a build never modifies a user's Desktop
unless explicitly requested.  The bundle shortcut is always refreshed because
it is a generated artefact.

Build prerequisites are listed in ``requirements-build.txt``.  Runtime model
files are intentionally not committed (see ``.gitignore``), so a release build
requires the locally trained weights and MediaPipe ``.task`` files to exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from create_shortcut import (  # noqa: E402  (path bootstrap above)
    DEFAULT_DESCRIPTION,
    DEFAULT_EXE,
    DEFAULT_NAME,
    ShortcutResult,
    create_shortcut,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the IntelliProctor PyInstaller bundle and create its "
            "Windows shortcut(s)."
        )
    )
    parser.add_argument(
        "--spec",
        default=str(REPO_ROOT / "IntelliProctor.spec"),
        help="PyInstaller spec file (default: IntelliProctor.spec)",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"application/shortcut name (default: {DEFAULT_NAME})",
    )
    parser.add_argument(
        "--exe",
        default=str(DEFAULT_EXE),
        help=f"packaged executable (default: {DEFAULT_EXE})",
    )
    parser.add_argument(
        "--icon",
        default=None,
        help=(
            "optional shortcut icon file (default: the icon embedded in the "
            "packaged executable, which keeps the shortcut portable)"
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="pass --clean to PyInstaller",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="only create shortcuts for an existing executable",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="also create a shortcut on the current user's Desktop",
    )
    parser.add_argument(
        "--no-bundle-shortcut",
        action="store_true",
        help="do not create the shortcut inside the deployment bundle",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing Desktop shortcut",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the build and shortcut plan without changing anything",
    )
    return parser


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyInstaller is not installed. Run:\n"
            f"  {Path(sys.executable).name} -m pip install "
            "-r requirements-build.txt"
        ) from exc


def _run_pyinstaller(spec: Path, *, clean: bool, dry_run: bool) -> None:
    if not spec.is_file():
        raise FileNotFoundError(f"PyInstaller spec not found: {spec}")

    command = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if clean:
        command.append("--clean")
    command.append(str(spec))

    printable = " ".join(command)
    if dry_run:
        print(f"Would run: {printable}")
        return

    print(f"Running: {printable}")
    started = time.perf_counter()
    process = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"PyInstaller exited with code {process.returncode} "
            f"after {elapsed:.1f}s"
        )
    print(f"PyInstaller finished in {elapsed:.1f}s")


def _report(result: ShortcutResult) -> None:
    if result.dry_run:
        action = "Would create"
    elif result.created:
        action = "Created"
    else:
        action = "Kept existing"
    print(f"{action} shortcut: {result.path}")
    print(f"  target: {result.target}")
    print(f"  icon:   {result.icon}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = Path(args.spec).expanduser().resolve()
    exe = Path(args.exe).expanduser().resolve()
    icon = Path(args.icon).expanduser().resolve() if args.icon else None

    try:
        if icon is not None and not icon.is_file():
            raise FileNotFoundError(f"shortcut icon not found: {icon}")
        if not args.skip_build:
            _ensure_pyinstaller()
            _run_pyinstaller(spec, clean=args.clean, dry_run=args.dry_run)

        if not exe.is_file():
            raise FileNotFoundError(
                f"packaged executable not found: {exe}\n"
                "Run the build first, or pass --exe with an existing bundle."
            )

        if not args.no_bundle_shortcut:
            # The in-bundle shortcut is generated output, so refreshing it is
            # safe and keeps a rebuilt bundle internally consistent.
            result = create_shortcut(
                exe=exe,
                name=args.name,
                dest="bundle",
                icon=icon,
                description=DEFAULT_DESCRIPTION,
                force=True,
                dry_run=args.dry_run,
            )
            _report(result)

        if args.desktop:
            result = create_shortcut(
                exe=exe,
                name=args.name,
                dest="desktop",
                icon=icon,
                description=DEFAULT_DESCRIPTION,
                force=args.force,
                dry_run=args.dry_run,
            )
            _report(result)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
