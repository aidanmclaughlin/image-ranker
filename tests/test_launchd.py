import os
import plistlib
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from image_ranker.config import Settings
from image_ranker.launchd import (
    JOBS_LABEL,
    LABELS,
    LOCAL_LABEL,
    REMOTE_LABEL,
    REMOTE_ALLOWED_ORIGIN,
    LaunchAgentError,
    agent_status,
    build_agents,
    install_agents,
    uninstall_agents,
)


def settings_for(root: Path) -> Settings:
    data = root / "private-data"
    return Settings(
        root=root,
        data=data,
        images=data / "images",
        models=data / "models",
        database=data / "ranker.sqlite3",
        host="127.0.0.1",
        port=8787,
    )


def completed(arguments, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(arguments, returncode, stdout, stderr)


class FakeLaunchctl:
    def __init__(self):
        self.loaded = set()
        self.calls = []
        self.fail_bootout_for = None
        self.fail_bootstrap_once_for = None

    def __call__(self, arguments):
        arguments = tuple(arguments)
        self.calls.append(arguments)
        action = arguments[1]
        if action == "print":
            label = arguments[2].rsplit("/", 1)[-1]
            return completed(arguments, 0 if label in self.loaded else 113)
        if action == "bootstrap":
            with Path(arguments[3]).open("rb") as handle:
                label = plistlib.load(handle)["Label"]
            if label == self.fail_bootstrap_once_for:
                self.fail_bootstrap_once_for = None
                return completed(arguments, 5, stderr="invalid plist")
            self.loaded.add(label)
            return completed(arguments)
        if action == "bootout":
            label = arguments[2].rsplit("/", 1)[-1]
            if label == self.fail_bootout_for:
                return completed(arguments, 5, stderr="permission denied")
            self.loaded.discard(label)
            return completed(arguments)
        raise AssertionError(f"Unexpected launchctl action: {action}")


class LaunchAgentTests(unittest.TestCase):
    def prepare(self, root: Path):
        settings = settings_for(root)
        executable = root / ".venv" / "bin" / "image-ranker"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        token = settings.data / "remote-token"
        token.parent.mkdir(parents=True)
        token.write_text("never-put-this-in-a-plist", encoding="utf-8")
        token.chmod(0o600)
        return settings, executable, token

    def test_templates_use_fixed_ports_absolute_paths_and_private_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, executable, token = self.prepare(Path(directory))
            agents = build_agents(settings, executable, token, jobs_interval=321)
            payloads = {agent.label: plistlib.loads(agent.render()) for agent in agents}

            local = payloads[LOCAL_LABEL]
            self.assertEqual(local["ProgramArguments"], [str(executable.resolve()), "serve"])
            self.assertEqual(local["EnvironmentVariables"]["IMAGE_RANKER_PORT"], "8787")
            self.assertTrue(local["KeepAlive"])

            remote = payloads[REMOTE_LABEL]
            self.assertEqual(
                remote["ProgramArguments"], [str(executable.resolve()), "serve", "--remote"]
            )
            self.assertEqual(
                remote["EnvironmentVariables"]["IMAGE_RANKER_REMOTE_TOKEN_FILE"],
                str(token.resolve()),
            )
            self.assertEqual(remote["EnvironmentVariables"]["IMAGE_RANKER_REMOTE_PORT"], "8788")
            self.assertEqual(
                remote["EnvironmentVariables"]["IMAGE_RANKER_ALLOWED_ORIGIN"],
                REMOTE_ALLOWED_ORIGIN,
            )
            self.assertNotIn("never-put-this-in-a-plist", remote["EnvironmentVariables"].values())

            jobs = payloads[JOBS_LABEL]
            self.assertEqual(jobs["ProgramArguments"], [str(executable.resolve()), "jobs"])
            self.assertEqual(jobs["StartInterval"], 321)
            self.assertNotIn("KeepAlive", jobs)

            for payload in payloads.values():
                self.assertEqual(payload["Umask"], 0o077)
                self.assertEqual(
                    payload["EnvironmentVariables"]["IMAGE_RANKER_ROOT"], str(settings.root)
                )
                self.assertEqual(Path(payload["StandardOutPath"]).parent, settings.data / "logs")
                self.assertEqual(Path(payload["StandardErrorPath"]).parent, settings.data / "logs")
                self.assertTrue(Path(payload["ProgramArguments"][0]).is_absolute())

    def test_install_is_idempotent_and_preserves_token_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, executable, token = self.prepare(root)
            destination = root / "LaunchAgents"
            launchctl = FakeLaunchctl()
            logs = settings.data / "logs"
            logs.mkdir()
            existing_log = logs / "local.stdout.log"
            existing_log.write_text("old log", encoding="utf-8")
            existing_log.chmod(0o644)

            first = install_agents(
                settings,
                executable=executable,
                token_file=token,
                agents_directory=destination,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )
            bootstrap_count = sum(call[1] == "bootstrap" for call in launchctl.calls)
            second = install_agents(
                settings,
                executable=executable,
                token_file=token,
                agents_directory=destination,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )

            self.assertEqual(set(first), set(LABELS))
            self.assertTrue(all(value == "installed" for value in first.values()))
            self.assertTrue(all(value == "unchanged" for value in second.values()))
            self.assertEqual(sum(call[1] == "bootstrap" for call in launchctl.calls), bootstrap_count)
            self.assertEqual(stat.S_IMODE(token.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((settings.data / "logs").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(existing_log.stat().st_mode), 0o600)
            for label in LABELS:
                self.assertEqual(
                    stat.S_IMODE((destination / f"{label}.plist").stat().st_mode), 0o644
                )

    def test_only_changed_agent_is_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, executable, token = self.prepare(root)
            destination = root / "LaunchAgents"
            launchctl = FakeLaunchctl()
            install_agents(
                settings,
                executable=executable,
                token_file=token,
                agents_directory=destination,
                jobs_interval=900,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )
            launchctl.calls.clear()

            result = install_agents(
                settings,
                executable=executable,
                token_file=token,
                agents_directory=destination,
                jobs_interval=1200,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )

            self.assertEqual(result[LOCAL_LABEL], "unchanged")
            self.assertEqual(result[REMOTE_LABEL], "unchanged")
            self.assertEqual(result[JOBS_LABEL], "reloaded")
            changed_actions = [call[1] for call in launchctl.calls if JOBS_LABEL in call[-1]]
            self.assertIn("bootout", changed_actions)

    def test_rejects_public_token_without_changing_its_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, executable, token = self.prepare(root)
            token.chmod(0o644)
            launchctl = FakeLaunchctl()

            with self.assertRaisesRegex(LaunchAgentError, "private"):
                install_agents(
                    settings,
                    executable=executable,
                    token_file=token,
                    agents_directory=root / "LaunchAgents",
                    runner=launchctl,
                    uid=os.getuid(),
                    system="Darwin",
                )

            self.assertEqual(stat.S_IMODE(token.stat().st_mode), 0o644)
            self.assertEqual(launchctl.calls, [])

    def test_uninstall_unloads_before_removal_and_keeps_private_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, executable, token = self.prepare(root)
            destination = root / "LaunchAgents"
            launchctl = FakeLaunchctl()
            install_agents(
                settings,
                executable=executable,
                token_file=token,
                agents_directory=destination,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )
            launchctl.calls.clear()

            result = uninstall_agents(
                agents_directory=destination,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )

            self.assertTrue(all(value == "unloaded and removed" for value in result.values()))
            self.assertEqual(launchctl.loaded, set())
            self.assertFalse(any(destination.glob("*.plist")))
            self.assertTrue(token.exists())
            for label in LABELS:
                actions = [call[1] for call in launchctl.calls if label in call[-1]]
                self.assertEqual(actions[:2], ["print", "bootout"])

    def test_failed_unload_keeps_plist_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, executable, token = self.prepare(root)
            destination = root / "LaunchAgents"
            launchctl = FakeLaunchctl()
            install_agents(
                settings,
                executable=executable,
                token_file=token,
                agents_directory=destination,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )
            launchctl.fail_bootout_for = LOCAL_LABEL

            with self.assertRaisesRegex(LaunchAgentError, "bootout"):
                uninstall_agents(
                    agents_directory=destination,
                    runner=launchctl,
                    uid=os.getuid(),
                    system="Darwin",
                )

            self.assertTrue((destination / f"{LOCAL_LABEL}.plist").exists())

    def test_failed_reload_restores_previous_plist_and_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, executable, token = self.prepare(root)
            destination = root / "LaunchAgents"
            launchctl = FakeLaunchctl()
            install_agents(
                settings,
                executable=executable,
                token_file=token,
                agents_directory=destination,
                jobs_interval=900,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )
            jobs_plist = destination / f"{JOBS_LABEL}.plist"
            previous = jobs_plist.read_bytes()
            launchctl.fail_bootstrap_once_for = JOBS_LABEL

            with self.assertRaisesRegex(LaunchAgentError, "bootstrap"):
                install_agents(
                    settings,
                    executable=executable,
                    token_file=token,
                    agents_directory=destination,
                    jobs_interval=1200,
                    runner=launchctl,
                    uid=os.getuid(),
                    system="Darwin",
                )

            self.assertEqual(jobs_plist.read_bytes(), previous)
            self.assertIn(JOBS_LABEL, launchctl.loaded)

    def test_status_reports_loaded_and_installed_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            launchctl = FakeLaunchctl()
            launchctl.loaded.add(LOCAL_LABEL)
            (destination / f"{REMOTE_LABEL}.plist").write_text("placeholder", encoding="utf-8")

            status = agent_status(
                agents_directory=destination,
                runner=launchctl,
                uid=os.getuid(),
                system="Darwin",
            )

            self.assertEqual(status[LOCAL_LABEL], {"installed": False, "loaded": True})
            self.assertEqual(status[REMOTE_LABEL], {"installed": True, "loaded": False})


if __name__ == "__main__":
    unittest.main()
