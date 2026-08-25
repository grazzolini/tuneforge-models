from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tuneforge_models.runtime_state import decode_timeline, load_runtime_state


def test_data_only_state_rejects_pickle(tmp_path: Path) -> None:
    state = tmp_path / "state.pkl"
    state.write_bytes(b"not loaded")

    with pytest.raises(ValueError, match="runtime-state-must-be-json"):
        load_runtime_state(state)


def test_load_and_decode_inversion(
    tmp_path: Path, crema_runtime_state: dict[str, Any]
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(crema_runtime_state), encoding="utf-8")
    state = load_runtime_state(state_path)
    tag = np.zeros((4, 170), dtype=np.float32)
    chord_index = state["decoder"]["labels"].index("C:maj")
    tag[:, chord_index] = 1.0
    bass = np.zeros((4, 13), dtype=np.float32)
    bass[:, 4] = 1.0

    timeline = decode_timeline(tag, bass, state)

    assert timeline == [
        {
            "start_seconds": 0.0,
            "end_seconds": 5 * 4_096 / 44_100,
            "label": "C:maj/3",
            "confidence": 1.0,
        }
    ]


def test_decode_merges_viterbi_frames_deterministically(
    crema_runtime_state: dict[str, Any]
) -> None:
    state = crema_runtime_state
    tag = np.full((6, 170), 1e-8, dtype=np.float32)
    chord_index = state["decoder"]["labels"].index("C:maj")
    no_chord_index = state["decoder"]["labels"].index("N")
    tag[:, chord_index] = 0.9999983
    tag[3:, no_chord_index] = 0.999999
    tag[3:, chord_index] = 1e-8
    tag /= tag.sum(axis=1, keepdims=True)
    bass = np.full((6, 13), 1 / 13, dtype=np.float32)

    first = decode_timeline(tag, bass, state)
    second = decode_timeline(tag, bass, state)

    assert first == second
    assert [segment["label"] for segment in first] == ["C:maj", "N"]
