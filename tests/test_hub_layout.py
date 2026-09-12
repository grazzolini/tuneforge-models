from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tuneforge_models.hub_layout import (
    BEAT_THIS,
    BEAT_THIS_FILES,
    CREMA,
    CREMA_FILES,
    publisher_overlay,
    read_tree,
    verify_family,
    verify_target,
    verify_target_inventory,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release(family: str) -> dict[str, bytes]:
    names = CREMA_FILES if family == CREMA else BEAT_THIS_FILES
    files = {
        name: f"{family}:{name}\n".encode() for name in names - {"manifest.json", "SHA256SUMS"}
    }
    manifest = {
        "model": {"name": family, "version": "unchanged"},
        "files": {name: {"size": len(data), "sha256": _sha(data)} for name, data in files.items()},
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    files["SHA256SUMS"] = "".join(
        f"{_sha(files[name])}  {name}\n" for name in sorted(names - {"SHA256SUMS"})
    ).encode()
    return files


def _target() -> dict[str, bytes]:
    target = {".gitattributes": b"attributes\n", "README.md": b"shared card\n"}
    for family in (CREMA, BEAT_THIS):
        target.update({f"{family}/{name}": data for name, data in _release(family).items()})
    verify_target(target)
    return target


def test_family_checksums_reject_corruption() -> None:
    release = _release(CREMA)
    verify_family(release, CREMA)

    release["crema-0.2.0-opset18.onnx"] = b"corrupt"
    with pytest.raises(ValueError, match="checksum-mismatch"):
        verify_family(release, CREMA)


def test_publisher_overlay_preserves_sibling() -> None:
    current = _target()
    replacement = _release(CREMA)

    updated = publisher_overlay(current, CREMA, replacement, b"updated catalog\n")

    assert updated["README.md"] == b"updated catalog\n"
    assert {name: data for name, data in updated.items() if name.startswith("beat-this/")} == {
        name: data for name, data in current.items() if name.startswith("beat-this/")
    }


def test_remote_inventory_rejects_unrelated_files() -> None:
    target = _target()
    verify_target_inventory(target)

    with pytest.raises(ValueError, match="target-remote-inventory-mismatch"):
        verify_target_inventory({*target, ".cache/extra"})


def test_read_tree_ignores_only_hugging_face_sidecars_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_bytes(b"card")
    sidecar = tmp_path / ".cache/huggingface/download"
    sidecar.mkdir(parents=True)
    (sidecar / "metadata").write_bytes(b"local")
    assert read_tree(tmp_path) == {"README.md": b"card"}

    (tmp_path / "linked").symlink_to(tmp_path / "README.md")
    with pytest.raises(ValueError, match="snapshot-contains-symlink:linked"):
        read_tree(tmp_path)


def test_repository_cards_match_family_and_catalog_roles() -> None:
    repository = Path(__file__).parents[1]
    catalog = (repository / "models/HUB_README.md").read_text()
    crema = (repository / "models/crema/MODEL_CARD.md").read_text()
    beat_this = (repository / "models/beat-this/MODEL_CARD.md").read_text()

    frontmatter = catalog.split("---", 2)[1]
    assert "license:" not in frontmatter
    assert "library_name:" not in frontmatter
    assert "ONNX Runtime 1.29.0 CPU" in catalog
    assert "ExecuTorch 1.4.0 XNNPACK" in catalog
    assert "library_name: onnxruntime" in crema
    assert "CPU chord recognition" in crema
    assert "library_name: executorch" in beat_this
