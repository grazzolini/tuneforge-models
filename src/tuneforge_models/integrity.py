"""Artifact integrity and safe archive extraction helpers."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from .spec import CREMA_MEMBERS, CREMA_WHEEL_SHA256, CREMA_WHEEL_SIZE, FileDigest


def digest(path: Path) -> FileDigest:
    """Return streaming SHA-256 and byte size."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
    return FileDigest(size=size, sha256=hasher.hexdigest())


def verify_file(path: Path, expected: FileDigest) -> None:
    """Reject missing or changed input without including its path in errors."""
    if not path.is_file():
        raise ValueError("missing-artifact")
    if digest(path) != expected:
        raise ValueError("artifact-integrity-failed")


def extract_verified_crema_members(wheel: Path, output: Path) -> dict[str, Path]:
    """Verify the wheel, then extract only pinned model members."""
    verify_file(wheel, FileDigest(CREMA_WHEEL_SIZE, CREMA_WHEEL_SHA256))
    output.mkdir(parents=True, exist_ok=False)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(wheel) as archive:
        for member, (size, sha256) in CREMA_MEMBERS.items():
            info = archive.getinfo(member)
            if info.file_size != size:
                raise ValueError("crema-member-size-failed")
            value = archive.read(info)
            if hashlib.sha256(value).hexdigest() != sha256:
                raise ValueError("crema-member-integrity-failed")
            destination = output / Path(member).name
            destination.write_bytes(value)
            extracted[member] = destination
    return extracted
