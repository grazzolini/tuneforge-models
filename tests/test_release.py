from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tuneforge_models.integrity import digest
from tuneforge_models.music_core import notation_identity
from tuneforge_models.release import assemble
from tuneforge_models.spec import MODEL_FILENAME, STATE_FILENAME


def test_release_contains_only_publishable_data(
    tmp_path: Path, crema_runtime_state: dict[str, Any]
) -> None:
    repository = tmp_path / "repository"
    (repository / "models/crema").mkdir(parents=True)
    (repository / "LICENSES").mkdir()
    (repository / "models/crema/MODEL_CARD.md").write_text("card\n")
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
                "notation_core": notation_identity(),
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
        json.dumps({"status": "passed", "notation_core": notation_identity(),
                    "candidates": [{"artifacts": artifacts, "notation_core": notation_identity()}]})
    )

    manifest = assemble(build, release, repository, validation)

    assert manifest["conversion"]["notation_core"] == notation_identity()
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


@pytest.mark.parametrize("mismatched_record", ("build", "report", "candidate"))
def test_release_rejects_core_identity_mismatch_before_staging(
    tmp_path: Path, mismatched_record: str
) -> None:
    identity = notation_identity()
    mismatched = dict(identity, revision="0" * 40)
    build = tmp_path / "build"
    build.mkdir()
    (build / MODEL_FILENAME).write_bytes(b"model")
    (build / STATE_FILENAME).write_text("{}")
    (build / "build-metadata.json").write_text(json.dumps({
        "notation_core": mismatched if mismatched_record == "build" else identity,
    }))
    artifacts = {
        "model": digest(build / MODEL_FILENAME).__dict__,
        "state": digest(build / STATE_FILENAME).__dict__,
    }
    report = tmp_path / "validation.json"
    report.write_text(json.dumps({
        "status": "passed",
        "notation_core": mismatched if mismatched_record == "report" else identity,
        "candidates": [{
            "artifacts": artifacts,
            "notation_core": mismatched if mismatched_record == "candidate" else identity,
        }],
    }))
    release = tmp_path / "release"
    with pytest.raises(ValueError, match="notation-core-release-mismatch"):
        assemble(build, release, tmp_path, report)
    assert not release.exists()
