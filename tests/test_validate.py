from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tuneforge_models.validate import (
    compare_output_head,
    compare_timelines,
    require_repeat_equal,
)


def test_ort_session_disables_telemetry_artifact(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["ORT_DISABLE_TELEMETRY"] = "0"
    code = (
        "import os; import tuneforge_models.validate as v; "
        "from onnxruntime.datasets import get_example; "
        "assert os.environ['ORT_DISABLE_TELEMETRY'] == '1'; "
        "v.ort.InferenceSession(get_example('mul_1.onnx'), providers=['CPUExecutionProvider'])"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, check=True)
    assert not (tmp_path / ":memory:.ses").exists()


def test_raw_tensor_drift_is_diagnostic_when_argmax_matches() -> None:
    expected = np.asarray([[[0.9, 0.1]]], dtype=np.float32)
    actual = np.asarray([[[0.8, 0.2]]], dtype=np.float32)

    report = compare_output_head("head", 2, actual, expected)

    assert report["argmax_exact"] is True
    assert report["maximum_difference"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("actual", "error"),
    [
        (np.asarray([[[0.1, 0.9]]], dtype=np.float32), "head-argmax-parity-failed"),
        (np.asarray([[[0.9, 0.1, 0.0]]], dtype=np.float32), "head-shape-parity-failed"),
    ],
)
def test_output_semantic_failures_remain_blocking(
    actual: np.ndarray[Any, Any], error: str
) -> None:
    expected = np.asarray([[[0.9, 0.1]]], dtype=np.float32)

    with pytest.raises(ValueError, match=error):
        compare_output_head("head", 2, actual, expected)


def test_repeat_drift_remains_blocking() -> None:
    first = [np.asarray([1.0], dtype=np.float32)]
    second = [np.asarray([0.9], dtype=np.float32)]

    with pytest.raises(ValueError, match="run-to-run-drift"):
        require_repeat_equal(first, second)


def test_timeline_semantics_and_confidence_remain_blocking() -> None:
    expected = [
        {"start_seconds": 0.0, "end_seconds": 1.0, "label": "C:maj7/E", "confidence": 0.8}
    ]
    semantic_drift = [dict(expected[0], label="C:maj7")]
    confidence_drift = [dict(expected[0], confidence=0.699)]

    with pytest.raises(ValueError, match="timeline-semantic-parity-failed"):
        compare_timelines(expected, semantic_drift)
    with pytest.raises(ValueError, match="confidence-parity-failed"):
        compare_timelines(expected, confidence_drift)
