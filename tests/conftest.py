from __future__ import annotations

import hashlib
from typing import Any

import pytest

_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_QUALITIES = (
    "min",
    "maj",
    "dim",
    "aug",
    "min6",
    "maj6",
    "min7",
    "maj7",
    "7",
    "dim7",
    "hdim7",
    "minmaj7",
    "sus2",
    "sus4",
)
_CLASSES_SHA256 = "e319b684db4725df87ab52c8c7b6df46508af23077c4b0a7dc662a6cbe6228c1"


@pytest.fixture
def crema_runtime_state() -> dict[str, Any]:
    """Return a fresh fixed Crema decoder state for each test."""
    labels = sorted(
        ["N", "X", *(f"{pitch}:{quality}" for pitch in _PITCHES for quality in _QUALITIES)]
    )
    classes_sha256 = hashlib.sha256("\n".join(labels).encode()).hexdigest()
    assert len(labels) == 170
    assert classes_sha256 == _CLASSES_SHA256
    return {
        "schema_version": 1,
        "decoder": {
            "sample_rate": 44_100,
            "hop_length": 4_096,
            "labels": labels,
            "classes_sha256": classes_sha256,
            "transition": {
                "encoding": "uniform-off-diagonal",
                "shape": [170, 170],
                "diagonal": 0.95,
                "off_diagonal": 0.05 / 169,
            },
        },
    }
