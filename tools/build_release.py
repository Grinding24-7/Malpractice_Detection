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
import hashlib
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_VERSION = "1.0.0"
PORTABLE_FILES = (
    ("Install IntelliProctor.cmd", "Install IntelliProctor.cmd"),
    ("README-release.txt", "README.txt"),
)
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
        "--zip",
        action="store_true",
        help="create a distributable ZIP under dist/release/",
    )
    parser.add_argument(
        "--split-size-mb",
        type=int,
        default=1900,
        help=(
            "split the release ZIP into parts no larger than this many MiB "
            "(default: 1900, below GitHub's 2 GiB asset limit)"
        ),
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


def _stage_portable_files(exe: Path, *, dry_run: bool) -> None:
    """Copy the first-run helper and user README into the bundle root.

    These files sit next to the executable (not inside ``_internal``) so an
    ordinary user can find them after extracting the archive.  The generated
    ``IntelliProctor.lnk`` is deliberately not part of the distributable
    archive: a .lnk stores an absolute path and is only valid on the build
    machine.  ``Install IntelliProctor.cmd`` creates a fresh shortcut at the
    extracted location on the target machine.
    """
    for source_name, dest_name in PORTABLE_FILES:
        source = REPO_ROOT / "assets" / source_name
        if not source.is_file():
            raise FileNotFoundError(f"portable release asset not found: {source}")
        destination = exe.parent / dest_name
        if dry_run:
            print(f"Would stage: {destination}")
            continue
        shutil.copy2(source, destination)
        print(f"Staged: {destination}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(paths: list[Path], release_dir: Path) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (release_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {release_dir / 'SHA256SUMS.txt'}")


def _split_archive(archive: Path, part_size: int, release_dir: Path) -> list[Path]:
    """Split *archive* into byte-exact parts and write a combine helper."""
    parts: list[Path] = []
    with archive.open("rb") as source:
        index = 1
        while True:
            part = archive.with_name(f"{archive.name}.part{index:03d}")
            written = 0
            with part.open("wb") as destination:
                while written < part_size:
                    chunk = source.read(min(8 * 1024 * 1024, part_size - written))
                    if not chunk:
                        break
                    destination.write(chunk)
                    written += len(chunk)
            if written == 0:
                part.unlink(missing_ok=True)
                break
            parts.append(part)
            print(f"  part {index:03d}: {part.name} ({written / (1024 ** 3):.2f} GiB)")
            index += 1

    archive.unlink()
    combine = release_dir / "Combine IntelliProctor parts.cmd"
    copy_args = " + ".join(f'"{part.name}"' for part in parts)
    combine.write_text(
        "\r\n".join(
            [
                "@echo off",
                "setlocal",
                'cd /d "%~dp0"',
                f'if exist "{archive.name}" del "{archive.name}"',
                f'copy /b {copy_args} "{archive.name}"',
                "if errorlevel 1 (",
                "    echo [ERROR] Could not combine the archive parts.",
                "    pause",
                "    exit /b 1",
                ")",
                "echo.",
                f"echo Created {archive.name}",
                "echo Extract that ZIP, then run Install IntelliProctor.cmd.",
                "pause",
                "",
            ]
        ),
        encoding="utf-8",
        newline="",
    )
    print(f"Wrote: {combine}")
    return parts


def _create_release_zip(exe: Path, *, split_size_mb: int, dry_run: bool) -> None:
    release_dir = REPO_ROOT / "dist" / "release"
    archive = release_dir / f"IntelliProctor-{APP_VERSION}-windows-x64.zip"
    if dry_run:
        print(f"Would create release archive: {archive}")
        return

    release_dir.mkdir(parents=True, exist_ok=True)
    archive.unlink(missing_ok=True)
    file_count = 0
    started = time.perf_counter()
    print(f"Creating release archive: {archive}")
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as bundle:
        for path in sorted(exe.parent.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(exe.parent)
            # The build-machine .lnk contains an absolute path.  The portable
            # installer creates the correct shortcut on the target machine.
            if relative.name.lower() == "intelliproctor.lnk":
                continue
            bundle.write(path, Path(exe.parent.name) / relative)
            file_count += 1
            if file_count % 500 == 0:
                print(f"  packed {file_count} files...")

    size = archive.stat().st_size
    elapsed = time.perf_counter() - started
    print(
        f"Created: {archive} "
        f"({size / (1024 ** 3):.2f} GiB, {file_count} files, {elapsed:.1f}s)"
    )

    split_size = max(256, split_size_mb) * 1024 * 1024
    if size > split_size:
        print(f"Splitting into parts of at most {split_size_mb} MiB...")
        parts = _split_archive(archive, split_size, release_dir)
        _write_checksums(parts, release_dir)
        print("Upload every .partNNN file and the combine helper together.")
    else:
        _write_checksums([archive], release_dir)


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

        _stage_portable_files(exe, dry_run=args.dry_run)

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

        if args.zip:
            _create_release_zip(
                exe,
                split_size_mb=args.split_size_mb,
                dry_run=args.dry_run,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
