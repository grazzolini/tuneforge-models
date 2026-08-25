from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tuneforge_models.integrity import digest
from tuneforge_models.release import assemble
from tuneforge_models.spec import MODEL_FILENAME, STATE_FILENAME


def test_release_contains_only_publishable_data(
    tmp_path: Path, crema_runtime_state: dict[str, Any]
) -> None:
    repository = tmp_path / "repository"
    (repository / "models/crema-0.2.0").mkdir(parents=True)
    (repository / "LICENSES").mkdir()
    (repository / "models/crema-0.2.0/MODEL_CARD.md").write_text("card\n")
    (repository / "LICENSES/crema-0.2.0-BSD-2-Clause.txt").write_text("license\n")
    build = tmp_path / "build"
    build.mkdir()
    (build / MODEL_FILENAME).write_bytes(b"model")
    (build / STATE_FILENAME).write_text(json.dumps(crema_runtime_state))
    (build / "build-metadata.json").write_text(
        json.dumps(
            {
                "tensorflow_version": "2.15.1",
                "tf2onnx_version": "1.16.1",
                "onnx_version": "1.17.0",
                "outputs": [],
            }
        )
    )
    release = tmp_path / "release"
    validation = tmp_path / "validation.json"
    artifacts = {
        "model": digest(build / MODEL_FILENAME).__dict__,
        "state": digest(build / STATE_FILENAME).__dict__,
    }
    validation.write_text(
        json.dumps({"status": "passed", "candidates": [{"artifacts": artifacts}]})
    )

    manifest = assemble(build, release, repository, validation)

    assert manifest["model"]["opset"] == 18
    assert manifest["source"]["wheel"]["size"] == 5_893_392
    assert manifest["source"]["members"]["crema/models/chord/model.h5"]["size"] == 6_441_860
    assert manifest["source"]["trained_by_tuneforge"] is False
    assert manifest["runtime"]["onnxruntime_version"] == "1.29.0"
    assert (release / "manifest.json").is_file()
    assert (release / "SHA256SUMS").is_file()
    assert not list(release.rglob("*.pkl"))
    assert not list(release.rglob("*.h5"))
    assert not list(release.rglob("*.npz"))


def test_release_rejects_unvalidated_candidate(tmp_path: Path) -> None:
    (tmp_path / MODEL_FILENAME).write_bytes(b"model")
    (tmp_path / STATE_FILENAME).write_text("{}")
    report = tmp_path / "validation.json"
    report.write_text('{"status":"passed","candidates":[]}')
    try:
        assemble(tmp_path, tmp_path / "release", tmp_path, report)
    except ValueError as error:
        assert str(error) == "build-not-covered-by-passed-validation"
    else:
        raise AssertionError("unvalidated candidate was accepted")
