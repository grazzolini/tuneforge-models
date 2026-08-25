"""Assemble a deterministic, checksummed Hugging Face upload directory."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

from .build import copy_release_inputs
from .integrity import digest
from .spec import (
    CREMA_MEMBERS,
    CREMA_TAG_COMMIT,
    CREMA_WHEEL_SHA256,
    CREMA_WHEEL_SIZE,
    MODEL_FILENAME,
    OPSET,
    OUTPUTS,
    STATE_FILENAME,
)


def assemble(
    build_directory: Path, release_directory: Path, repository: Path, validation_report: Path
) -> dict[str, Any]:
    """Stage canonical bytes and write manifest/checksums."""
    validation = json.loads(validation_report.read_text(encoding="utf-8"))
    artifacts = {
        "model": digest(build_directory / MODEL_FILENAME).__dict__,
        "state": digest(build_directory / STATE_FILENAME).__dict__,
    }
    if validation.get("status") != "passed" or not any(
        candidate.get("artifacts") == artifacts for candidate in validation.get("candidates", [])
    ):
        raise ValueError("build-not-covered-by-passed-validation")
    copy_release_inputs(build_directory, release_directory)
    shutil.copyfile(
        repository / "models/crema-0.2.0/MODEL_CARD.md",
        release_directory / "README.md",
    )
    license_directory = release_directory / "LICENSES"
    license_directory.mkdir()
    shutil.copyfile(
        repository / "LICENSES/crema-0.2.0-BSD-2-Clause.txt",
        license_directory / "crema-0.2.0-BSD-2-Clause.txt",
    )
    build_metadata = json.loads(
        (build_directory / "build-metadata.json").read_text(encoding="utf-8")
    )
    files = {}
    for path in sorted(release_directory.rglob("*")):
        if path.is_file():
            files[path.relative_to(release_directory).as_posix()] = digest(path).__dict__
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model": {"name": "crema", "version": "0.2.0", "format": "onnx", "opset": OPSET},
        "source": {
            "wheel": {"size": CREMA_WHEEL_SIZE, "sha256": CREMA_WHEEL_SHA256},
            "members": {
                name: {"size": size, "sha256": sha256}
                for name, (size, sha256) in CREMA_MEMBERS.items()
            },
            "upstream_repository": "https://github.com/bmcfee/crema",
            "upstream_tag_commit": CREMA_TAG_COMMIT,
            "trained_by_tuneforge": False,
            "training_manifest": "not-provided-upstream",
        },
        "conversion": {
            "tensorflow_version": build_metadata["tensorflow_version"],
            "tf2onnx_version": build_metadata["tf2onnx_version"],
            "onnx_version": build_metadata["onnx_version"],
            "outputs": build_metadata["outputs"],
        },
        "runtime": {
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "provider": "CPUExecutionProvider",
            "outputs": [name for name, _ in OUTPUTS],
        },
        "files": files,
    }
    (release_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksummed = [
        MODEL_FILENAME,
        STATE_FILENAME,
        "README.md",
        "LICENSES/crema-0.2.0-BSD-2-Clause.txt",
        "manifest.json",
    ]
    (release_directory / "SHA256SUMS").write_text(
        "".join(f"{digest(release_directory / name).sha256}  {name}\n" for name in checksummed),
        encoding="utf-8",
    )
    return manifest
