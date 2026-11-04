from __future__ import annotations

import hashlib
from typing import Any

import pytest

from tuneforge_models.music_core import crema_labels


@pytest.fixture
def crema_runtime_state() -> dict[str, Any]:
    """Return a fresh fixed Crema decoder state for each test."""
    labels = crema_labels()
    classes_sha256 = hashlib.sha256("\n".join(labels).encode()).hexdigest()
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
