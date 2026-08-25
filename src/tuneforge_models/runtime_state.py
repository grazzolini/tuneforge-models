"""Trusted build-time export and pickle-free runtime decoding."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .integrity import verify_file
from .spec import CREMA_MEMBERS, FileDigest

_QUALITY_INTERVALS = {
    "7": {0, 4, 7, 10},
    "aug": {0, 4, 8},
    "dim": {0, 3, 6},
    "dim7": {0, 3, 6, 9},
    "hdim7": {0, 3, 6, 10},
    "maj": {0, 4, 7},
    "maj6": {0, 4, 7, 9},
    "maj7": {0, 4, 7, 11},
    "min": {0, 3, 7},
    "min6": {0, 3, 7, 9},
    "min7": {0, 3, 7, 10},
    "minmaj7": {0, 3, 7, 11},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
}
_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_DEGREES = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
_CLASSES_SHA256 = "e319b684db4725df87ab52c8c7b6df46508af23077c4b0a7dc662a6cbe6228c1"


def _load_trusted_pump(pump_path: Path) -> Any:
    size, sha256 = CREMA_MEMBERS["crema/models/chord/pump.pkl"]
    verify_file(pump_path, FileDigest(size, sha256))
    with pump_path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - exact hash verified above


def export_runtime_state(pump_path: Path, output: Path) -> dict[str, Any]:
    """Deserialize only the exact pinned pump and export fixed data.

    Pickle remains a trusted, isolated build input. This function must never be
    used in TuneForge or against downloaded/unverified state.
    """
    _, sha256 = CREMA_MEMBERS["crema/models/chord/pump.pkl"]
    pump = _load_trusted_pump(pump_path)
    feature = pump["cqt"]
    decoder = pump["chord_tag"]
    labels = [str(value) for value in decoder.encoder.classes_.tolist()]
    classes_sha256 = hashlib.sha256("\n".join(labels).encode()).hexdigest()
    if classes_sha256 != _CLASSES_SHA256:
        raise ValueError("unexpected-decoder-classes")
    transition = np.asarray(decoder.transition, dtype=np.float64)
    diagonal = np.diag(transition)
    off_diagonal = transition[~np.eye(len(labels), dtype=bool)]
    if not np.all(diagonal == diagonal[0]) or not np.all(off_diagonal == off_diagonal[0]):
        raise ValueError("unsupported-transition-encoding")
    state: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "name": "crema",
            "version": "0.2.0",
            "pump_sha256": sha256,
        },
        "preprocessing": {
            "type": "hcqt-magnitude",
            "sample_rate": int(feature.sr),
            "hop_length": int(feature.hop_length),
            "fmin": float(feature.fmin),
            "harmonics": [int(value) for value in feature.harmonics],
            "octaves": int(feature.n_octaves),
            "oversample": int(feature.over_sample),
            "log_amplitude": bool(feature.log),
            "convolution_layout": str(feature.conv),
            "dtype": str(feature.dtype),
            "output_shape": [None, 216, 2],
        },
        "decoder": {
            "type": "viterbi-discriminative",
            "sample_rate": int(decoder.sr),
            "hop_length": int(decoder.hop_length),
            "vocabulary": str(decoder.vocab),
            "labels": labels,
            "classes_sha256": classes_sha256,
            "sparse": bool(decoder.sparse),
            "transition": {
                "encoding": "uniform-off-diagonal",
                "shape": [len(labels), len(labels)],
                "diagonal": float(diagonal[0]),
                "off_diagonal": float(off_diagonal[0]),
            },
            "p_init": None,
            "p_state": None,
        },
    }
    output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def trusted_reference_timeline(
    pump_path: Path,
    tag: npt.NDArray[np.float32],
    bass: npt.NDArray[np.float32],
) -> list[dict[str, Any]]:
    """Decode with Crema/pumpp inside the trusted conversion environment."""
    import librosa  # type: ignore[import-not-found]
    import mir_eval  # type: ignore[import-not-found]
    from scipy.stats import gmean  # type: ignore[import-untyped]

    pump = _load_trusted_pump(pump_path)
    decoder = pump["chord_tag"]
    annotation = decoder.inverse(tag)
    timeline: list[dict[str, Any]] = []
    for observation in annotation.pop_data():
        start, end = librosa.time_to_frames(
            [observation.time, observation.time + observation.duration],
            sr=decoder.sr,
            hop_length=decoder.hop_length,
        )
        label = str(observation.value)
        if label not in {"N", "X"}:
            mean_bass = gmean(bass[start : end + 1])
            bass_index = int(np.argmax(mean_bass))
            root_index, pitches, _ = mir_eval.chord.encode(label)
            relative = (bass_index - root_index) % 12 if bass_index < 12 else 0
            if relative and pitches[relative]:
                label = f"{label}/{_DEGREES[relative]}"
        timeline.append(
            {
                "start_seconds": float(observation.time),
                "end_seconds": float(observation.time + observation.duration),
                "label": label,
                "confidence": float(observation.confidence),
            }
        )
    return timeline


def load_runtime_state(path: Path) -> dict[str, Any]:
    """Load and structurally validate data-only runtime state."""
    if path.suffix != ".json":
        raise ValueError("runtime-state-must-be-json")
    state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    decoder = state["decoder"]
    labels = decoder["labels"]
    transition = decoder["transition"]
    classes_sha256 = hashlib.sha256("\n".join(labels).encode()).hexdigest()
    if (
        state["schema_version"] != 1
        or len(labels) != 170
        or classes_sha256 != decoder["classes_sha256"]
        or classes_sha256 != _CLASSES_SHA256
    ):
        raise ValueError("unsupported-runtime-state")
    if transition["shape"] != [170, 170] or transition["encoding"] != "uniform-off-diagonal":
        raise ValueError("invalid-transition-state")
    return state


def _transition(state: dict[str, Any]) -> npt.NDArray[np.float64]:
    transition = state["decoder"]["transition"]
    count = len(state["decoder"]["labels"])
    matrix = np.full((count, count), transition["off_diagonal"], dtype=np.float64)
    np.fill_diagonal(matrix, transition["diagonal"])
    return matrix


def _viterbi(
    probabilities: npt.NDArray[np.float32], state: dict[str, Any]
) -> npt.NDArray[np.int64]:
    count = probabilities.shape[1]
    tiny = np.finfo(probabilities.dtype).tiny
    log_prob = np.log(probabilities + tiny) - math.log(1.0 / count)
    log_transition = np.log(_transition(state) + tiny)
    values = np.zeros(probabilities.shape, dtype=np.float64)
    pointers = np.zeros(probabilities.shape, dtype=np.int64)
    values[0] = log_prob[0] + math.log(1.0 / count)
    for frame in range(1, probabilities.shape[0]):
        candidates = values[frame - 1][:, np.newaxis] + log_transition
        pointers[frame] = np.argmax(candidates, axis=0)
        values[frame] = log_prob[frame] + candidates[pointers[frame], np.arange(count)]
    path = np.zeros(probabilities.shape[0], dtype=np.int64)
    path[-1] = int(np.argmax(values[-1]))
    for frame in range(probabilities.shape[0] - 2, -1, -1):
        path[frame] = pointers[frame + 1, path[frame + 1]]
    return path


def decode_timeline(
    tag: npt.NDArray[np.float32],
    bass: npt.NDArray[np.float32],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decode tag/bass heads using only NumPy and exported state."""
    labels = state["decoder"]["labels"]
    hop = int(state["decoder"]["hop_length"])
    sample_rate = int(state["decoder"]["sample_rate"])
    path = _viterbi(tag, state)
    changes = np.where(path[1:] != path[:-1])[0]
    ends = np.unique(np.append(changes, len(path)))
    lengths = np.diff(np.append(-1, ends))
    starts = np.cumsum(np.append(0, lengths))[:-1]
    timeline: list[dict[str, Any]] = []
    for start, length in zip(starts, lengths, strict=True):
        end = int(start + length)
        label_index = int(path[start])
        label = labels[label_index]
        confidence = float(np.mean(tag[start : end + 1, label_index]))
        if label not in {"N", "X"}:
            root, quality = label.split(":", 1)
            root_index = _PITCHES.index(root)
            stabilized = np.maximum(bass[start : end + 1], np.finfo(np.float32).tiny)
            mean_log = np.mean(np.log(stabilized), axis=0)
            bass_index = int(np.argmax(np.exp(mean_log)))
            relative = (bass_index - root_index) % 12 if bass_index < 12 else 0
            if relative and relative in _QUALITY_INTERVALS[quality]:
                label = f"{label}/{_DEGREES[relative]}"
        timeline.append(
            {
                "start_seconds": float(start * hop / sample_rate),
                "end_seconds": float(end * hop / sample_rate),
                "label": label,
                "confidence": confidence,
            }
        )
    return timeline
