from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from api.config import settings


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
LOG_DIR = ROOT / "logs" / "training"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class TrainingManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._presets: dict[str, dict[str, Any]] = {
            "classifier_cifar10": {
                "title": "Classifier · CIFAR-10",
                "task": "train",
                "args": ["--train-dataset", "cifar10", "--epochs", "30"],
                "description": "Baseline classifier for quick local training.",
            },
            "classifier_food101": {
                "title": "Classifier · Food-101",
                "task": "train",
                "args": ["--train-dataset", "food101", "--epochs", "20", "--image-size", "224", "--batch-size", "32"],
                "description": "Large transfer-learning run for a stronger image classifier.",
            },
            "colorizer_stl10": {
                "title": "Colorizer · STL-10",
                "task": "train-colorizer",
                "args": ["--colorize-dataset", "stl10", "--epochs", "50", "--limit", "12000"],
                "description": "Default colorizer training pipeline.",
            },
            "colorizer_oxford_pet": {
                "title": "Colorizer · Oxford-IIIT Pet",
                "task": "train-colorizer",
                "args": ["--colorize-dataset", "oxford_iiit_pet", "--epochs", "40", "--limit", "20000", "--batch-size", "24"],
                "description": "Heavier colorizer run for a better model on a larger dataset.",
            },
        }

    def list_presets(self) -> list[dict[str, Any]]:
        return [
            {"id": preset_id, **preset, "runner": self._runner(), "runner_label": self._runner_label()}
            for preset_id, preset in self._presets.items()
        ]

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        hydrated = [self._hydrate_job(job) for job in jobs]
        return sorted(hydrated, key=lambda item: item["started_at"], reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return self._hydrate_job(job) if job else None

    def start(self, preset_id: str) -> dict[str, Any]:
        preset = self._presets.get(preset_id)
        if not preset:
            raise ValueError(f"Unknown preset: {preset_id}")

        with self._lock:
            for job in self._jobs.values():
                if job["preset"] == preset_id and job["status"] == "running":
                    raise RuntimeError("This training preset is already running")

        job_id = uuid.uuid4().hex[:10]
        started_at = datetime.utcnow().isoformat()
        run_label = f"{started_at[:19].replace(':', '-')}_{preset_id}_{job_id}"
        log_path = LOG_DIR / f"{run_label}.log"
        progress_path = LOG_DIR / f"{run_label}.progress.json"
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        command, runner_meta = self._build_command(
            preset=preset,
            progress_path=progress_path,
            env=env,
        )

        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        job = {
            "id": job_id,
            "preset": preset_id,
            "title": preset["title"],
            "description": preset["description"],
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "return_code": None,
            "pid": process.pid,
            "command": command,
            "runner": runner_meta["runner"],
            "runner_label": runner_meta["runner_label"],
            "log_path": str(log_path.relative_to(ROOT)).replace('\\', '/'),
            "progress_path": str(progress_path.relative_to(ROOT)).replace('\\', '/'),
            "progress_path_wsl": runner_meta.get("progress_path_wsl"),
        }

        with self._lock:
            self._jobs[job_id] = job

        watcher = threading.Thread(
            target=self._watch_process,
            args=(job_id, process, log_file),
            daemon=True,
        )
        watcher.start()
        return job

    def _watch_process(
        self,
        job_id: str,
        process: subprocess.Popen,
        log_file,
    ) -> None:
        return_code = process.wait()
        log_file.close()
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["return_code"] = return_code
            job["finished_at"] = datetime.utcnow().isoformat()
            job["status"] = "completed" if return_code == 0 else "failed"

    def _runner(self) -> str:
        runner = settings.training_runner.strip().lower()
        return runner or "native"

    def _runner_label(self) -> str:
        if self._runner() == "wsl":
            distro = settings.training_wsl_distribution.strip()
            return f"wsl:{distro}" if distro else "wsl"
        return "native"

    def _build_command(
        self,
        preset: dict[str, Any],
        progress_path: Path,
        env: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        if self._runner() == "wsl":
            return self._build_wsl_command(preset, progress_path)
        env["TRAINING_PROGRESS_FILE"] = str(progress_path)
        return (
            [
                str(PYTHON),
                "scripts/run_local.py",
                preset["task"],
                *preset["args"],
            ],
            {
                "runner": "native",
                "runner_label": "native",
            },
        )

    def _build_wsl_command(
        self,
        preset: dict[str, Any],
        progress_path: Path,
    ) -> tuple[list[str], dict[str, str]]:
        wsl_project_dir = settings.training_wsl_project_dir.strip() or self._guess_wsl_project_dir()
        wsl_python = settings.training_wsl_python.strip() or ".venv/bin/python"
        distro = settings.training_wsl_distribution.strip()
        progress_path_wsl = f"{wsl_project_dir}/logs/training/{progress_path.name}"

        inner_command = " ".join(
            [
                f"cd {shlex.quote(wsl_project_dir)}",
                "&& mkdir -p logs/training",
                f"&& PYTHONUTF8=1 PYTHONIOENCODING=utf-8 TRAINING_PROGRESS_FILE={shlex.quote(progress_path_wsl)}",
                shlex.quote(wsl_python),
                "scripts/run_local.py",
                preset["task"],
                *(shlex.quote(arg) for arg in preset["args"]),
            ]
        )

        command = ["wsl.exe"]
        if distro:
            command.extend(["-d", distro])
        command.extend(["--", "bash", "-lc", inner_command])
        return (
            command,
            {
                "runner": "wsl",
                "runner_label": f"wsl:{distro}" if distro else "wsl",
                "progress_path_wsl": progress_path_wsl,
            },
        )

    def _guess_wsl_project_dir(self) -> str:
        drive, tail = os.path.splitdrive(str(ROOT))
        if drive and tail:
            normalized_tail = tail.replace('\\', '/')
            return f"/mnt/{drive[0].lower()}{normalized_tail}"
        return str(ROOT).replace('\\', '/')

    def _hydrate_job(self, job: dict[str, Any]) -> dict[str, Any]:
        item = dict(job)
        progress = self._read_progress(job)
        if progress is not None:
            if item["status"] != "running" and progress.get("status") == "running":
                progress["status"] = item["status"]
            item["progress"] = progress
        log_tail = self._read_log_tail(ROOT / job["log_path"])
        if log_tail:
            item["log_tail"] = log_tail
        return item

    def _read_progress(self, job: dict[str, Any]) -> dict[str, Any] | None:
        progress_path = ROOT / job["progress_path"]
        if progress_path.exists():
            try:
                return json.loads(progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None

        progress_path_wsl = job.get("progress_path_wsl")
        if job.get("runner") == "wsl" and progress_path_wsl:
            payload = self._read_wsl_text(progress_path_wsl)
            if payload:
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return None
        return None

    def _read_log_tail(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        tail_count = max(1, settings.training_log_tail_lines)
        return lines[-tail_count:]

    def _read_wsl_text(self, wsl_path: str) -> str | None:
        distro = settings.training_wsl_distribution.strip()
        command = ["wsl.exe"]
        if distro:
            command.extend(["-d", distro])
        command.extend([
            "--",
            "bash",
            "-lc",
            f"if [ -f {shlex.quote(wsl_path)} ]; then cat {shlex.quote(wsl_path)}; fi",
        ])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except OSError:
            return None
        output = completed.stdout.strip()
        return output or None


training_manager = TrainingManager()
