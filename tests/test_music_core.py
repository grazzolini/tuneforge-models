from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tuneforge_models import music_core
from tuneforge_models.runtime_state import decode_timeline

# Independent semantic expectations; vocabulary alone comes from the shared fixture.
INTERVALS = {
    "7": (0, 4, 7, 10), "aug": (0, 4, 8), "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10), "maj": (0, 4, 7),
    "maj6": (0, 4, 7, 9), "maj7": (0, 4, 7, 11), "min": (0, 3, 7),
    "min6": (0, 3, 7, 9), "min7": (0, 3, 7, 10), "minmaj7": (0, 3, 7, 11),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7),
}
DEGREES = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@pytest.mark.parametrize("label", music_core.crema_labels())
def test_all_shared_labels_parse_with_independent_expectations(label: str) -> None:
    actual = music_core.parse_crema(label)
    if label in {"N", "X"}:
        assert actual == {"kind": "no_chord" if label == "N" else "unknown",
                          "root_pitch_class": None, "quality": None}
        return
    root, quality = label.split(":")
    assert actual == {"kind": "chord", "root_pitch_class": PITCHES.index(root),
                      "quality": {"maj": "major", "min": "minor", "min7": "m7"}.get(
                          quality, quality)}


@pytest.mark.parametrize("quality", INTERVALS)
@pytest.mark.parametrize("bass_index", range(13))
@pytest.mark.parametrize("root", ("C", "D#", "F#"))
def test_decode_preserves_raw_degree_labels_for_every_bass_class(
    crema_runtime_state: dict[str, Any], quality: str, bass_index: int, root: str
) -> None:
    label = f"{root}:{quality}"
    tag = np.zeros((4, 170), dtype=np.float32)
    tag[:, crema_runtime_state["decoder"]["labels"].index(label)] = 1
    bass = np.zeros((4, 13), dtype=np.float32)
    bass[:, bass_index] = 1
    relative = (bass_index - PITCHES.index(root)) % 12 if bass_index < 12 else 0
    expected = label
    if relative and relative in INTERVALS[quality]:
        expected += f"/{DEGREES[relative]}"
    assert decode_timeline(tag, bass, crema_runtime_state) == [{
        "start_seconds": 0.0, "end_seconds": 5 * 4096 / 44100,
        "label": expected, "confidence": 1.0,
    }]


@pytest.mark.parametrize(("label", "normalized"), [
    ("C:maj7/3", "Cmaj7/E"), ("D:min7/5", "Dm7/A"),
    ("F#:dim7/6", "F#dim7/D#"), ("A#:7/b7", "A#7/G#"),
])
def test_normalized_display_is_separate_from_raw_timeline(label: str, normalized: str) -> None:
    import tuneforge_music_core

    response: object = tuneforge_music_core.call({"operation": "backendParseChord", "input": label})
    assert isinstance(response, dict)
    assert response["display_label"] == normalized


@pytest.fixture
def identity_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    source = music_core.ROOT
    for relative in (Path("models/crema/build-spec.json"),
                     music_core.CORE / "pyproject.toml", music_core.CORE / "src/abi.cpp",
                     music_core.CORE / "test/fixtures/crema-labels-v1.json"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    (root / music_core.SUBMODULE / ".git").write_text("gitdir: unused\n")
    identity = music_core.notation_identity()

    def git(directory: Path, *args: str) -> str:
        if args[0] == "config":
            return (music_core.SUBMODULE.as_posix() if args[-1].endswith(".path")
                    else str(identity["repository"]))
        if args[0] == "ls-files":
            return f"160000 {identity['revision']} 0 third_party/tuneforge"
        if args[0] == "rev-parse":
            return str(identity["revision"])
        return ""

    monkeypatch.setattr(music_core, "_git", git)
    return root


@pytest.mark.parametrize(("mutation", "error"), [
    ("missing", "submodule-missing"), ("pin", "revision-mismatch"),
    ("fixture", "fixture-mismatch"), ("package", "package-mismatch"),
    ("contract", "contract-mismatch"),
])
def test_identity_rejects_changed_inputs(
    identity_checkout: Path, mutation: str, error: str
) -> None:
    root = identity_checkout
    if mutation == "missing":
        (root / music_core.SUBMODULE / ".git").unlink()
    elif mutation in {"pin", "fixture", "package", "contract"}:
        path = root / "models/crema/build-spec.json"
        data = json.loads(path.read_text())
        identity = data["notation_core"]
        if mutation == "pin":
            identity["revision"] = "0" * 40
        elif mutation == "fixture":
            identity["fixture"]["sha256"] = "0" * 64
        elif mutation == "package":
            identity["package_version"] = "0.2.0"
        else:
            identity["api_version"] = 2
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        music_core.notation_identity(root)


@pytest.mark.parametrize("status", (" M src/core.cpp", "M  src/core.cpp", "?? surprise.py"))
def test_identity_rejects_tracked_staged_and_untracked_changes(
    identity_checkout: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    original = music_core._git
    monkeypatch.setattr(music_core, "_git", lambda root, *args: (
        status if args[0] == "status" else original(root, *args)))
    with pytest.raises(ValueError, match="submodule-dirty"):
        music_core.notation_identity(identity_checkout)


@pytest.mark.parametrize(("operation", "value"), [("parse", []), ("contains", 1), ("degree", None)])
def test_wrong_native_response_fails_closed(
    monkeypatch: pytest.MonkeyPatch, operation: str, value: object
) -> None:
    monkeypatch.setattr(music_core, "_binding", lambda: SimpleNamespace(call=lambda request: value))
    with pytest.raises(ValueError, match="response-mismatch"):
        if operation == "parse":
            music_core.parse_crema("C:maj")
        elif operation == "contains":
            music_core.crema_contains("C:maj", 4)
        else:
            music_core.degree_name(4)


def test_prepare_does_not_import_installed_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tuneforge_music_core", None)
    original = subprocess.run

    def run(command: list[str], **kwargs: Any) -> Any:
        if command[0] == sys.executable:
            assert command[-2:] == ["--fetch-sources", "--offline"]
            return None
        return original(command, **kwargs)

    monkeypatch.setattr("tuneforge_models.music_core.subprocess.run", run)
    assert music_core.prepare(offline=True) == music_core.notation_identity()
