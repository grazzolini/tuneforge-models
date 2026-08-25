from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tuneforge_models.integrity import digest, verify_file
from tuneforge_models.spec import FileDigest


def test_digest_and_verify_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")
    expected = FileDigest(8, hashlib.sha256(b"verified").hexdigest())

    assert digest(artifact) == expected
    verify_file(artifact, expected)


def test_verify_file_sanitizes_failure(tmp_path: Path) -> None:
    private_name = tmp_path / "private-song-name.bin"
    private_name.write_bytes(b"changed")

    with pytest.raises(ValueError, match="^artifact-integrity-failed$") as error:
        verify_file(private_name, FileDigest(7, "0" * 64))

    assert "private-song-name" not in str(error.value)
