"""Validate the two-family Hugging Face repository layout."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CREMA = "crema"
BEAT_THIS = "beat-this"

CREMA_FILES = frozenset(
    {
        "README.md",
        "manifest.json",
        "SHA256SUMS",
        "crema-0.2.0-opset18.onnx",
        "crema-0.2.0-runtime-state.json",
        "LICENSES/crema-0.2.0-BSD-2-Clause.txt",
    }
)
BEAT_THIS_FILES = frozenset(
    {
        "README.md",
        "manifest.json",
        "SHA256SUMS",
        "beat-this-small0.pte",
        "validation.json",
        "LICENSES/beat-this-1.1.0-MIT.txt",
        "LICENSES/executorch-1.4.0-BSD-3-Clause.txt",
    }
)
FAMILY_FILES = {CREMA: CREMA_FILES, BEAT_THIS: BEAT_THIS_FILES}
TARGET_PREFIXES = {CREMA: "crema/", BEAT_THIS: "beat-this/"}
TARGET_FILES = frozenset(
    {
        ".gitattributes",
        "README.md",
        *(
            prefix + name
            for family, prefix in TARGET_PREFIXES.items()
            for name in FAMILY_FILES[family]
        ),
    }
)


def read_tree(root: Path) -> dict[str, bytes]:
    """Read a snapshot while excluding Hugging Face's local cache metadata."""
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        if path.is_symlink():
            raise ValueError(f"snapshot-contains-symlink:{relative.as_posix()}")
        if path.is_file():
            files[relative.as_posix()] = path.read_bytes()
    return files


def write_tree(files: Mapping[str, bytes], root: Path) -> None:
    """Write a validated in-memory layout to an empty directory."""
    if root.exists() and any(root.iterdir()):
        raise ValueError("target-directory-is-not-empty")
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def verify_target_inventory(names: Iterable[str]) -> None:
    """Reject any remote path outside the exact versionless layout."""
    if set(names) != TARGET_FILES:
        raise ValueError("target-remote-inventory-mismatch")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _family(tree: Mapping[str, bytes], family: str) -> dict[str, bytes]:
    prefix = TARGET_PREFIXES[family]
    return {
        name.removeprefix(prefix): data for name, data in tree.items() if name.startswith(prefix)
    }


def verify_family(files: Mapping[str, bytes], family: str) -> None:
    """Require one exact family release with self-contained relative checksums."""
    expected = FAMILY_FILES[family]
    if set(files) != expected:
        raise ValueError(f"{family}-file-set-mismatch")

    checksums: dict[str, str] = {}
    for line in files["SHA256SUMS"].decode("utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or name.startswith("/")
            or ".." in Path(name).parts
            or name in checksums
        ):
            raise ValueError(f"{family}-invalid-checksum-entry")
        checksums[name] = digest
    checksummed = expected - {"SHA256SUMS"}
    if set(checksums) != checksummed or any(
        checksums[name] != _digest(files[name]) for name in checksummed
    ):
        raise ValueError(f"{family}-checksum-mismatch")

    manifest: dict[str, Any] = json.loads(files["manifest.json"])
    entries = manifest.get("files")
    manifest_files = expected - {"manifest.json", "SHA256SUMS"}
    if not isinstance(entries, dict) or set(entries) != manifest_files:
        raise ValueError(f"{family}-manifest-file-set-mismatch")
    for name in manifest_files:
        if entries[name] != {"size": len(files[name]), "sha256": _digest(files[name])}:
            raise ValueError(f"{family}-manifest-digest-mismatch")


def verify_target(tree: Mapping[str, bytes]) -> None:
    """Require the exact shared-card plus two-family target layout."""
    if set(tree) != TARGET_FILES:
        raise ValueError("target-layout-file-set-mismatch")
    for family in FAMILY_FILES:
        verify_family(_family(tree, family), family)


def publisher_overlay(
    current: Mapping[str, bytes], family: str, release: Mapping[str, bytes], hub_card: bytes
) -> dict[str, bytes]:
    """Overlay one owned family and the shared card while preserving its sibling."""
    verify_target(current)
    verify_family(release, family)
    updated = dict(current)
    updated["README.md"] = hub_card
    prefix = TARGET_PREFIXES[family]
    for name, data in release.items():
        updated[prefix + name] = data
    verify_target(updated)
    sibling = BEAT_THIS if family == CREMA else CREMA
    if _family(updated, sibling) != _family(current, sibling):
        raise ValueError("publisher-modified-sibling-family")
    return updated
