#!/usr/bin/env python3
"""Collect a reproducible, secret-free execution environment manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "huggingface-hub",
    "safetensors",
    "tokenizers",
)
SOURCE_DATA = (
    "data/deep_chal_math_train.csv",
    "data/deep_chal_math_leaderboard.csv",
)


def run(command: list[str]) -> dict[str, Any]:
    """Run a read-only probe and return its output without raising."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_details() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment probe
        return {"import_error": repr(exc)}

    details: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        details["bf16_supported"] = torch.cuda.is_bf16_supported()
        details["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return details


def vast_details() -> dict[str, Any]:
    probe = run(["vast-capabilities", "metrics,packages"])
    if probe["returncode"] != 0:
        return {"probe": probe}
    try:
        raw = json.loads(probe["stdout"])
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "probe": probe}

    instance = raw.get("instance", {})
    selected_instance_keys = (
        "container_id",
        "workspace",
        "workspace_is_volume",
        "image",
        "image_tag",
        "template_id",
        "template_name",
    )
    return {
        "image": raw.get("image"),
        "instance": {key: instance.get(key) for key in selected_instance_keys if key in instance},
        "hardware": raw.get("hardware"),
        "python_environments": raw.get("python_environments"),
    }


def collect(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    now_utc = datetime.now(timezone.utc)
    disk = shutil.disk_usage("/workspace" if Path("/workspace").exists() else repo_root)
    data_hashes = {
        relative: sha256(repo_root / relative)
        for relative in SOURCE_DATA
        if (repo_root / relative).is_file()
    }

    return {
        "schema_version": 1,
        "collected_at_utc": now_utc.isoformat(),
        "collected_at_kst": now_utc.astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
        "timezone": {"utc": "UTC", "local": "Asia/Seoul"},
        "repo_root": str(repo_root),
        "workspace": os.environ.get("WORKSPACE", "/workspace"),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "os_release": platform.freedesktop_os_release() if sys.platform == "linux" else {},
            "kernel": run(["uname", "-a"]),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "packages": package_versions(),
        "torch": torch_details(),
        "gpu": {
            "nvidia_smi_query": run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "nvidia_smi": run(["nvidia-smi"]),
            "nvcc": run(["nvcc", "--version"]),
        },
        "cpu": {"logical_count": os.cpu_count(), "lscpu": run(["lscpu"])},
        "memory": {"free_bytes": run(["free", "-b"])},
        "disk": {"path": "/workspace", "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "vast": vast_details(),
        "git": {
            "head": run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
            "status_short": run(["git", "-C", str(repo_root), "status", "--short"]),
            "diff_stat": run(["git", "-C", str(repo_root), "diff", "--stat"]),
        },
        "source_data_sha256": data_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = collect(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
