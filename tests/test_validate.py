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


@pytest.mark.parametrize("mismatched_candidate", ("first", "second"))
def test_validation_rejects_candidate_core_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatched_candidate: str
) -> None:
    from tuneforge_models import validate
    from tuneforge_models.music_core import notation_identity

    identity = notation_identity()

    def candidate(path: Path) -> dict[str, Any]:
        candidate_identity = dict(identity)
        if path.name == mismatched_candidate:
            candidate_identity["revision"] = "0" * 40
        return {"notation_core": candidate_identity}

    monkeypatch.setattr(validate, "_validate_candidate", candidate)
    report = tmp_path / "validation.json"
    with pytest.raises(ValueError, match="notation-core-validation-mismatch"):
        validate.validate_builds(tmp_path / "first", tmp_path / "second", report)
    assert not report.exists()


@pytest.mark.parametrize("recorded_identity", (None, {"revision": "0" * 40}))
def test_candidate_rejects_build_metadata_core_identity_mismatch(
    tmp_path: Path, crema_runtime_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    recorded_identity: dict[str, str] | None,
) -> None:
    import json
    from types import SimpleNamespace

    from tuneforge_models import validate
    from tuneforge_models.runtime_state import decode_timeline
    from tuneforge_models.spec import OUTPUTS, STATE_FILENAME

    outputs = [np.full((1, 4, classes), 1 / classes, dtype=np.float32)
               for _, classes in OUTPUTS]
    outputs[0][:] = 0
    outputs[0][:, :, crema_runtime_state["decoder"]["labels"].index("N")] = 1
    np.savez(tmp_path / "reference.npz",
             **{f"reference_{index}": value for index, value in enumerate(outputs)})
    (tmp_path / STATE_FILENAME).write_text(json.dumps(crema_runtime_state))
    (tmp_path / "reference-timeline.json").write_text(json.dumps(
        decode_timeline(outputs[0][0], outputs[3][0], crema_runtime_state)))
    (tmp_path / "build-metadata.json").write_text(json.dumps({"notation_core": recorded_identity}))
    monkeypatch.setattr("tuneforge_models.validate.onnx.load", lambda path: SimpleNamespace(
        graph=SimpleNamespace(output=[None] * 4)))
    monkeypatch.setattr(
        "tuneforge_models.validate.onnx.checker.check_model", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(validate, "_run", lambda candidate: outputs)
    with pytest.raises(ValueError, match="notation-core-build-mismatch"):
        validate._validate_candidate(tmp_path)
