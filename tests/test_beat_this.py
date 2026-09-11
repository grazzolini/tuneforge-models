from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuneforge_models.beat_this_build import build
from tuneforge_models.beat_this_release import _metadata, release
from tuneforge_models.beat_this_spec import MODEL_FILENAME


def test_build_rejects_corrupt_checkpoint_before_creating_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"corrupt")
    output = tmp_path / "build"

    with pytest.raises(ValueError, match="^artifact-integrity-failed$"):
        build(checkpoint, output, "working-tree")

    assert not output.exists()


def test_candidate_metadata_must_bind_exact_artifact(tmp_path: Path) -> None:
    (tmp_path / MODEL_FILENAME).write_bytes(b"candidate")
    (tmp_path / "build-metadata.json").write_text(json.dumps({"artifact": {}}))

    with pytest.raises(ValueError, match="^candidate-metadata-failed$"):
        _metadata(tmp_path)


def test_release_rejects_existing_output_before_validation(tmp_path: Path) -> None:
    output = tmp_path / "release"
    output.mkdir()

    with pytest.raises(ValueError, match="^output-already-exists$"):
        release(tmp_path, tmp_path, tmp_path, output, tmp_path)
