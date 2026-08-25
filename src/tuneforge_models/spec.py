"""Immutable Crema 0.2.0 source and release specification."""

from __future__ import annotations

from dataclasses import dataclass

CREMA_WHEEL_FILENAME = "crema-0.2.0-py3-none-any.whl"
CREMA_WHEEL_SIZE = 5_893_392
CREMA_WHEEL_SHA256 = "b2787afd0367463438ca2b9b2944c490308eee1f307e5796078ec540a6281484"
CREMA_TAG_COMMIT = "051c91697fd16856a0a1019cc06ee1f11fb52c5f"
CREMA_MEMBERS = {
    "crema/models/chord/model.h5": (
        6_441_860,
        "08b80e5b648e743c89284e9bc0b12b993dad1129157a75e0de70e076b0b8a235",
    ),
    "crema/models/chord/model_spec.pkl": (
        4_776,
        "3769565e5d3f4d3c590a15d92755b8633b0c5fc280c3e76847f875dd5959374c",
    ),
    "crema/models/chord/pump.pkl": (
        256_233,
        "73034e05996ed1f5979a2bfbd63c5061f074f57a0670e26b8a71808181845ec6",
    ),
}
OUTPUTS = (
    ("chord_tag", 170),
    ("chord_pitch", 12),
    ("chord_root", 13),
    ("chord_bass", 13),
)
MODEL_FILENAME = "crema-0.2.0-opset18.onnx"
STATE_FILENAME = "crema-0.2.0-runtime-state.json"
OPSET = 18
MAX_CONFIDENCE_DIFFERENCE = 0.10


@dataclass(frozen=True)
class FileDigest:
    """Size and SHA-256 for one artifact."""

    size: int
    sha256: str
