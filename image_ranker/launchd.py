from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .config import Settings


LOCAL_LABEL = "com.image-ranker.local"
REMOTE_LABEL = "com.image-ranker.remote"
JOBS_LABEL = "com.image-ranker.jobs"
LABELS = (LOCAL_LABEL, REMOTE_LABEL, JOBS_LABEL)
LAUNCHCTL = Path("/bin/launchctl")
REMOTE_ALLOWED_ORIGIN = "https://lumen-ranker.vercel.app"


class LaunchAgentError(RuntimeError):
    """Raised when a LaunchAgent cannot be safely configured."""


@dataclass(frozen=True)
class LaunchAgent:
    label: str
    arguments: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path
    stdout: Path
    stderr: Path
    keep_alive: bool = False
    start_interval: int | None = None

    @property
    def filename(self) -> str:
        return f"{self.label}.plist"

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "Label": self.label,
            "ProgramArguments": list(self.arguments),
            "EnvironmentVariables": dict(self.environment),
            "WorkingDirectory": str(self.working_directory),
            "StandardOutPath": str(self.stdout),
            "StandardErrorPath": str(self.stderr),
            "RunAtLoad": True,
            "ProcessType": "Background",
            # launchd applies this to logs and every file created by a child.
            "Umask": 0o077,
        }
        if self.keep_alive:
            payload["KeepAlive"] = True
            payload["ThrottleInterval"] = 10
        if self.start_interval is not None:
            payload["StartInterval"] = self.start_interval
            payload["LowPriorityIO"] = True
        return payload

    def render(self) -> bytes:
        return plistlib.dumps(self.payload(), fmt=plistlib.FMT_XML, sort_keys=True)


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, check=False, capture_output=True, text=True)


def default_agents_directory() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def default_token_file(settings: Settings) -> Path:
    return settings.data / "remote-token"


def default_executable(settings: Settings) -> Path:
    return settings.root / ".venv" / "bin" / "image-ranker"


def build_agents(
    settings: Settings,
    executable: Path,
    token_file: Path,
    *,
    jobs_interval: int = 900,
) -> tuple[LaunchAgent, LaunchAgent, LaunchAgent]:
    """Build deterministic per-user agents without reading or changing the host."""
    if jobs_interval <= 0:
        raise ValueError("jobs_interval must be positive")
    executable = executable.expanduser().resolve()
    token_file = token_file.expanduser().resolve()
    logs = settings.data / "logs"
    common = {
        "IMAGE_RANKER_ROOT": str(settings.root),
        "IMAGE_RANKER_DATA": str(settings.data),
        "PYTHONUNBUFFERED": "1",
    }
    return (
        LaunchAgent(
            label=LOCAL_LABEL,
            arguments=(str(executable), "serve"),
            environment={
                **common,
                "IMAGE_RANKER_HOST": "127.0.0.1",
                "IMAGE_RANKER_PORT": "8787",
            },
            working_directory=settings.root,
            stdout=logs / "local.stdout.log",
            stderr=logs / "local.stderr.log",
            keep_alive=True,
        ),
        LaunchAgent(
            label=REMOTE_LABEL,
            arguments=(str(executable), "serve", "--remote"),
            environment={
                **common,
                "IMAGE_RANKER_ALLOWED_ORIGIN": REMOTE_ALLOWED_ORIGIN,
                "IMAGE_RANKER_REMOTE_PORT": "8788",
                "IMAGE_RANKER_REMOTE_TOKEN_FILE": str(token_file),
            },
            working_directory=settings.root,
            stdout=logs / "remote.stdout.log",
            stderr=logs / "remote.stderr.log",
            keep_alive=True,
        ),
        LaunchAgent(
            label=JOBS_LABEL,
            arguments=(str(executable), "jobs"),
            environment=common,
            working_directory=settings.root,
            stdout=logs / "jobs.stdout.log",
            stderr=logs / "jobs.stderr.log",
            start_interval=jobs_interval,
        ),
    )


def _require_macos(system: str | None = None) -> None:
    if (system or platform.system()) != "Darwin":
        raise LaunchAgentError("LaunchAgents are only supported on macOS")


def _validate_executable(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise LaunchAgentError(
            f"Image Ranker executable is missing or not executable: {path}. "
            "Create .venv and install the project first."
        )
    return path


def _validate_token_file(path: Path, uid: int) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink():
        raise LaunchAgentError("Remote token file must not be a symbolic link")
    try:
        token_stat = candidate.stat()
    except FileNotFoundError as exc:
        raise LaunchAgentError(f"Remote token file does not exist: {candidate}") from exc
    mode = stat.S_IMODE(token_stat.st_mode)
    if not stat.S_ISREG(token_stat.st_mode):
        raise LaunchAgentError("Remote token path must be a regular file")
    if token_stat.st_uid != uid:
        raise LaunchAgentError("Remote token file must be owned by the current user")
    if mode not in (0o400, 0o600):
        raise LaunchAgentError("Remote token file must be owner-readable and private (0400 or 0600)")
    return candidate.resolve(strict=True)


def _prepare_private_logs(logs: Path, agents: Iterable[LaunchAgent]) -> None:
    if logs.is_symlink():
        raise LaunchAgentError("Private logs directory must not be a symbolic link")
    logs.mkdir(parents=True, mode=0o700, exist_ok=True)
    logs.chmod(0o700)
    for agent in agents:
        for path in (agent.stdout, agent.stderr):
            if path.is_symlink():
                raise LaunchAgentError(f"LaunchAgent log must not be a symbolic link: {path}")
            if path.exists():
                if not path.is_file():
                    raise LaunchAgentError(f"LaunchAgent log must be a regular file: {path}")
                path.chmod(0o600)


def _service_target(label: str, uid: int) -> str:
    return f"gui/{uid}/{label}"


def _domain_target(uid: int) -> str:
    return f"gui/{uid}"


def _loaded(label: str, uid: int, runner: Runner) -> bool:
    result = runner((str(LAUNCHCTL), "print", _service_target(label, uid)))
    return result.returncode == 0


def _command_error(action: str, result: subprocess.CompletedProcess[str]) -> LaunchAgentError:
    detail = (result.stderr or result.stdout or "unknown launchctl error").strip()
    return LaunchAgentError(f"launchctl {action} failed: {detail}")


def _bootout(label: str, uid: int, runner: Runner) -> bool:
    if not _loaded(label, uid, runner):
        return False
    result = runner((str(LAUNCHCTL), "bootout", _service_target(label, uid)))
    if result.returncode != 0 and _loaded(label, uid, runner):
        raise _command_error("bootout", result)
    return True


def _bootstrap(path: Path, uid: int, runner: Runner) -> None:
    result = runner((str(LAUNCHCTL), "bootstrap", _domain_target(uid), str(path)))
    if result.returncode != 0:
        raise _command_error("bootstrap", result)


def _atomic_write(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise LaunchAgentError(f"Refusing to replace symbolic link: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def install_agents(
    settings: Settings,
    *,
    executable: Path | None = None,
    token_file: Path | None = None,
    agents_directory: Path | None = None,
    jobs_interval: int = 900,
    runner: Runner = _run,
    uid: int | None = None,
    system: str | None = None,
) -> dict[str, str]:
    """Install and load all agents, reloading only configurations that changed."""
    _require_macos(system)
    current_uid = os.getuid() if uid is None else uid
    resolved_executable = _validate_executable(executable or default_executable(settings))
    resolved_token = _validate_token_file(token_file or default_token_file(settings), current_uid)
    agents = build_agents(
        settings, resolved_executable, resolved_token, jobs_interval=jobs_interval
    )
    _prepare_private_logs(settings.data / "logs", agents)
    destination = (agents_directory or default_agents_directory()).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    for agent in agents:
        path = destination / agent.filename
        if path.is_symlink():
            raise LaunchAgentError(f"Refusing to read or replace symbolic link: {path}")
        rendered = agent.render()
        previous = path.read_bytes() if path.exists() else None
        changed = previous != rendered
        was_loaded = _loaded(agent.label, current_uid, runner)
        if not changed and was_loaded:
            path.chmod(0o644)
            results[agent.label] = "unchanged"
            continue

        if was_loaded:
            _bootout(agent.label, current_uid, runner)
        if changed:
            _atomic_write(path, rendered)
        else:
            path.chmod(0o644)
        try:
            _bootstrap(path, current_uid, runner)
        except LaunchAgentError:
            if changed:
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_write(path, previous)
            if was_loaded and previous is not None:
                _bootstrap(path, current_uid, runner)
            raise
        results[agent.label] = "reloaded" if was_loaded else "installed"
    return results


def uninstall_agents(
    *,
    agents_directory: Path | None = None,
    runner: Runner = _run,
    uid: int | None = None,
    system: str | None = None,
) -> dict[str, str]:
    """Unload known agents before removing their plists; private data is untouched."""
    _require_macos(system)
    current_uid = os.getuid() if uid is None else uid
    destination = (agents_directory or default_agents_directory()).expanduser().resolve()
    results: dict[str, str] = {}
    for label in LABELS:
        unloaded = _bootout(label, current_uid, runner)
        path = destination / f"{label}.plist"
        existed = path.exists() or path.is_symlink()
        if existed:
            path.unlink()
        if unloaded and existed:
            results[label] = "unloaded and removed"
        elif unloaded:
            results[label] = "unloaded"
        elif existed:
            results[label] = "removed"
        else:
            results[label] = "not installed"
    return results


def agent_status(
    *,
    agents_directory: Path | None = None,
    runner: Runner = _run,
    uid: int | None = None,
    system: str | None = None,
) -> dict[str, dict[str, bool]]:
    _require_macos(system)
    current_uid = os.getuid() if uid is None else uid
    destination = (agents_directory or default_agents_directory()).expanduser().resolve()
    return {
        label: {
            "installed": (destination / f"{label}.plist").is_file(),
            "loaded": _loaded(label, current_uid, runner),
        }
        for label in LABELS
    }


def add_cli_parser(subparsers: argparse._SubParsersAction) -> None:
    launchd = subparsers.add_parser(
        "launchd", help="manage private per-user macOS background services"
    )
    actions = launchd.add_subparsers(dest="launchd_action", required=True)
    install = actions.add_parser("install", help="install and load all LaunchAgents")
    install.add_argument(
        "--executable",
        type=Path,
        help="absolute image-ranker executable (default: repo .venv)",
    )
    install.add_argument(
        "--token-file",
        type=Path,
        help="existing private remote token file (default: data/remote-token)",
    )
    install.add_argument(
        "--jobs-interval",
        type=int,
        default=900,
        help="seconds between one-shot maintenance runs (default: 900)",
    )
    actions.add_parser("uninstall", help="safely unload and remove all LaunchAgents")
    actions.add_parser("status", help="show whether each LaunchAgent is installed and loaded")
    actions.add_parser("render", help="print the generated LaunchAgent templates without installing")


def run_cli(args: argparse.Namespace, settings: Settings) -> None:
    if args.launchd_action == "install":
        result = install_agents(
            settings,
            executable=args.executable,
            token_file=args.token_file,
            jobs_interval=args.jobs_interval,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.launchd_action == "uninstall":
        print(json.dumps(uninstall_agents(), indent=2, sort_keys=True))
    elif args.launchd_action == "status":
        print(json.dumps(agent_status(), indent=2, sort_keys=True))
    elif args.launchd_action == "render":
        agents = build_agents(
            settings,
            default_executable(settings),
            default_token_file(settings),
        )
        for index, agent in enumerate(agents):
            if index:
                print()
            print(f"# {default_agents_directory() / agent.filename}")
            print(agent.render().decode(), end="")
