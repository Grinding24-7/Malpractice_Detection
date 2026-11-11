#!/usr/bin/env python3
"""Create a launch shortcut for the IntelliProctor desktop application.

The helper is intentionally dependency-free: it uses only the Python standard
library plus the PowerShell that ships with Windows.  On Windows it writes a
real ``.lnk`` through ``WScript.Shell`` so the result behaves exactly like a
shortcut created from Explorer.  Linux and macOS fall back to a ``.desktop``
file and a ``.command`` launcher respectively.

Typical use::

    python tools/create_shortcut.py
    python tools/create_shortcut.py --dest desktop --force
    python tools/create_shortcut.py --dest path --dest-dir release/

The default target is ``dist/IntelliProctor/IntelliProctor.exe``.  The default
destination is the deployment bundle (the executable's own folder), so a build
never modifies the user's Desktop unless ``--dest desktop`` is requested.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXE = REPO_ROOT / "dist" / "IntelliProctor" / "IntelliProctor.exe"
DEFAULT_NAME = "IntelliProctor"
DEFAULT_DESCRIPTION = (
    "IntelliProctor - AI-assisted exam proctoring and malpractice detection"
)

_WINDOWS_INVALID_NAME_CHARS = set('<>:"/\\|?*')


@dataclass(frozen=True)
class ShortcutResult:
    """Outcome of a shortcut request."""

    path: Path
    target: Path
    icon: Path
    created: bool
    dry_run: bool = False

    @property
    def status(self) -> str:
        if self.dry_run:
            return "dry-run"
        return "created" if self.created else "exists"


def _resolve_existing_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_target(exe: str | Path) -> Path:
    target = _resolve_existing_file(exe, "executable")
    if os.name == "nt" and target.suffix.lower() != ".exe":
        raise ValueError(f"Windows shortcut target must be a .exe: {target}")
    return target


def _resolve_icon(icon: str | Path | None, exe: Path) -> Path:
    # An icon embedded in the executable is always available, so it is the
    # safest default.  Passing --icon is useful for a separately branded .ico.
    if icon is None:
        return exe
    return _resolve_existing_file(icon, "icon")


def _validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("shortcut name must not be empty")
    if name in {".", ".."}:
        raise ValueError("shortcut name must not be '.' or '..'")
    if any(char in _WINDOWS_INVALID_NAME_CHARS for char in name):
        raise ValueError(
            "shortcut name contains a character Windows cannot use in a "
            f"file name: {name!r}"
        )
    # Do not silently create "IntelliProctor.lnk.lnk".
    for suffix in (".lnk", ".desktop", ".command"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)].rstrip()
            break
    if not name:
        raise ValueError("shortcut name must not consist only of an extension")
    return name


def _powershell() -> str:
    path = shutil.which("powershell") or shutil.which("pwsh")
    if not path:
        raise RuntimeError("PowerShell is required to create a Windows .lnk")
    return path


def _run_powershell_command(command: str, *, timeout: int = 30) -> str:
    process = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(f"PowerShell failed: {detail}")
    return process.stdout.strip()


def _ps_quote(value: str) -> str:
    """Quote a string for a single-quoted PowerShell literal."""
    return "'" + value.replace("'", "''") + "'"


def _windows_desktop_dir() -> Path:
    try:
        raw = _run_powershell_command(
            "[Environment]::GetFolderPath("
            "[Environment+SpecialFolder]::DesktopDirectory)"
        )
    except Exception:
        raw = ""
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Desktop"


def _linux_desktop_dir() -> Path:
    # Respect the XDG user-dirs setting when it exists; it may point at a
    # localised folder such as ~/Bureau or ~/Escritorio.
    config = Path.home() / ".config" / "user-dirs.dirs"
    try:
        for line in config.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("XDG_DESKTOP_DIR="):
                value = line.split("=", 1)[1].strip().strip('"')
                return Path(os.path.expandvars(value)).expanduser()
    except OSError:
        pass
    return Path.home() / "Desktop"


def _desktop_dir() -> Path:
    if os.name == "nt":
        return _windows_desktop_dir()
    if sys.platform.startswith("linux"):
        return _linux_desktop_dir()
    return Path.home() / "Desktop"


def _resolve_destination(
    dest: str,
    exe: Path,
    dest_dir: str | Path | None,
) -> Path:
    if dest == "bundle":
        return exe.parent
    if dest == "desktop":
        return _desktop_dir()
    if dest == "path":
        if not dest_dir:
            raise ValueError("--dest-dir is required when --dest path is used")
        return Path(dest_dir).expanduser().resolve()
    raise ValueError(f"unknown destination: {dest!r}")


def _shortcut_extension() -> str:
    if os.name == "nt":
        return ".lnk"
    if sys.platform.startswith("linux"):
        return ".desktop"
    return ".command"


def _create_windows_shortcut(
    shortcut: Path,
    target: Path,
    working_dir: Path,
    icon: Path,
    description: str,
    arguments: str,
) -> None:
    payload = {
        "path": str(shortcut),
        "target": str(target),
        "working_directory": str(working_dir),
        "icon_location": f"{icon},0",
        "description": description,
        "arguments": arguments,
    }

    with tempfile.TemporaryDirectory(prefix="intelliproctor_shortcut_") as tmp:
        payload_path = Path(tmp) / "shortcut.json"
        script_path = Path(tmp) / "create_shortcut.ps1"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        script_path.write_text(
            "\n".join(
                [
                    "$ErrorActionPreference = 'Stop'",
                    f"$payload = Get-Content -LiteralPath "
                    f"{_ps_quote(str(payload_path))} -Raw -Encoding UTF8 "
                    "| ConvertFrom-Json",
                    "$folder = Split-Path -Parent $payload.path",
                    "if (-not (Test-Path -LiteralPath $folder)) {",
                    "    New-Item -ItemType Directory -Path $folder "
                    "-Force | Out-Null",
                    "}",
                    "$wsh = New-Object -ComObject WScript.Shell",
                    "try {",
                    "    $shortcut = $wsh.CreateShortcut($payload.path)",
                    "    $shortcut.TargetPath = $payload.target",
                    "    $shortcut.WorkingDirectory = "
                    "$payload.working_directory",
                    "    $shortcut.IconLocation = $payload.icon_location",
                    "    $shortcut.Description = $payload.description",
                    "    $shortcut.WindowStyle = 1",
                    "    if ($payload.arguments) {",
                    "        $shortcut.Arguments = $payload.arguments",
                    "    }",
                    "    $shortcut.Save()",
                    "} finally {",
                    "    if ($shortcut) { $shortcut = $null }",
                    "    [System.Runtime.InteropServices.Marshal]"
                    "::ReleaseComObject($wsh) | Out-Null",
                    "}",
                    "Write-Output $payload.path",
                ]
            ),
            encoding="utf-8-sig",
        )
        process = subprocess.run(
            [
                _powershell(),
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(f"could not create Windows shortcut: {detail}")
    if not shortcut.is_file():
        raise RuntimeError(
            f"PowerShell reported success but no shortcut exists: {shortcut}"
        )


def _create_linux_shortcut(
    shortcut: Path,
    target: Path,
    working_dir: Path,
    icon: Path,
    description: str,
    arguments: str,
) -> None:
    exec_line = shlex.quote(str(target))
    if arguments:
        exec_line += " " + arguments
    shortcut.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Version=1.0",
                "Type=Application",
                "Name=IntelliProctor",
                f"Comment={description}",
                f"Exec={exec_line}",
                f"Path={working_dir}",
                f"Icon={icon}",
                "Terminal=false",
                "Categories=Education;Utility;",
                "StartupNotify=true",
                "StartupWMClass=IntelliProctor",
                "",
            ]
        ),
        encoding="utf-8",
    )
    shortcut.chmod(0o755)


def _create_macos_shortcut(
    shortcut: Path,
    target: Path,
    working_dir: Path,
    description: str,
    arguments: str,
) -> None:
    command = f"exec {shlex.quote(str(target))}"
    if arguments:
        command += " " + arguments
    shortcut.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                f"# {description}",
                f"cd {shlex.quote(str(working_dir))}",
                command,
                "",
            ]
        ),
        encoding="utf-8",
    )
    shortcut.chmod(0o755)


def create_shortcut(
    exe: str | Path = DEFAULT_EXE,
    *,
    name: str = DEFAULT_NAME,
    dest: str = "bundle",
    dest_dir: str | Path | None = None,
    icon: str | Path | None = None,
    description: str = DEFAULT_DESCRIPTION,
    arguments: str = "",
    working_dir: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> ShortcutResult:
    """Create (or report) an IntelliProctor launch shortcut.

    Parameters
    ----------
    exe:
        Path to the packaged executable.  It must already exist.
    name:
        Base shortcut name, without an extension.
    dest:
        ``"bundle"`` (next to the executable), ``"desktop"``, or ``"path"``.
    dest_dir:
        Directory used with ``dest="path"``.
    icon:
        Icon file.  Defaults to *exe*, which uses the icon embedded in the
        executable.
    description:
        Shortcut comment/tooltip.
    arguments:
        Optional command-line arguments stored in the shortcut.
    working_dir:
        Working directory for the launched process.  Defaults to *exe*'s
        parent so relative model paths resolve from the deployment folder.
    force:
        Replace an existing shortcut.  Without it an existing file is left
        untouched and ``created`` is ``False``.
    dry_run:
        Validate and print the planned action without writing anything.
    """

    target = _resolve_target(exe)
    clean_name = _validate_name(name)
    destination = _resolve_destination(dest, target, dest_dir)
    icon_path = _resolve_icon(icon, target)
    work_path = (
        Path(working_dir).expanduser().resolve()
        if working_dir is not None
        else target.parent
    )

    shortcut = destination / f"{clean_name}{_shortcut_extension()}"
    exists = shortcut.exists()
    if exists and not force:
        return ShortcutResult(
            path=shortcut,
            target=target,
            icon=icon_path,
            created=False,
            dry_run=dry_run,
        )

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            _create_windows_shortcut(
                shortcut,
                target,
                work_path,
                icon_path,
                description,
                arguments,
            )
        elif sys.platform.startswith("linux"):
            _create_linux_shortcut(
                shortcut,
                target,
                work_path,
                icon_path,
                description,
                arguments,
            )
        else:
            _create_macos_shortcut(
                shortcut,
                target,
                work_path,
                description,
                arguments,
            )

    return ShortcutResult(
        path=shortcut,
        target=target,
        icon=icon_path,
        created=True,
        dry_run=dry_run,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Windows .lnk (or a Linux/macOS launcher) for the "
            "packaged IntelliProctor executable."
        )
    )
    parser.add_argument(
        "--exe",
        default=str(DEFAULT_EXE),
        help=f"packaged executable (default: {DEFAULT_EXE})",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"shortcut name without extension (default: {DEFAULT_NAME})",
    )
    parser.add_argument(
        "--dest",
        choices=("bundle", "desktop", "path"),
        default="bundle",
        help=(
            "where to write the shortcut (default: bundle, i.e. the "
            "executable's folder)"
        ),
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="destination directory; required with --dest path",
    )
    parser.add_argument(
        "--icon",
        default=None,
        help="icon file; defaults to the executable's embedded icon",
    )
    parser.add_argument(
        "--description",
        default=DEFAULT_DESCRIPTION,
        help="shortcut comment/tooltip",
    )
    parser.add_argument(
        "--arguments",
        default="",
        help="optional command-line arguments stored in the shortcut",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="working directory for the launched process (default: exe folder)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing shortcut",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and print the planned shortcut without writing it",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the result as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = create_shortcut(
            exe=args.exe,
            name=args.name,
            dest=args.dest,
            dest_dir=args.dest_dir,
            icon=args.icon,
            description=args.description,
            arguments=args.arguments,
            working_dir=args.working_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "status": result.status,
        "path": str(result.path),
        "target": str(result.target),
        "icon": str(result.icon),
        "created": result.created,
        "dry_run": result.dry_run,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if result.dry_run:
        print(f"Would create shortcut: {result.path}")
    elif result.created:
        print(f"Created shortcut: {result.path}")
    else:
        print(
            f"Shortcut already exists: {result.path} "
            "(use --force to replace it)"
        )
    print(f"  target: {result.target}")
    print(f"  icon:   {result.icon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
