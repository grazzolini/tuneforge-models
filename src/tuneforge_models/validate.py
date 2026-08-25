"""ONNX-only graph, tensor, timeline, determinism, and release validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import onnxruntime as ort  # type: ignore[import-untyped]

from .integrity import digest
from .runtime_state import decode_timeline, load_runtime_state
from .spec import (
    MAX_CONFIDENCE_DIFFERENCE,
    MODEL_FILENAME,
    OUTPUTS,
    STATE_FILENAME,
)


def validate_builds(first: Path, second: Path, report_path: Path) -> dict[str, Any]:
    """Require two independent candidates to satisfy all parity gates."""
    reports = [_validate_candidate(path) for path in (first, second)]
    first_output = _run(first)
    second_output = _run(second)
    cross_max = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(first_output, second_output, strict=True)
    )
    cross_argmax = all(
        np.array_equal(np.argmax(left, axis=-1), np.argmax(right, axis=-1))
        for left, right in zip(first_output, second_output, strict=True)
    )
    if not cross_argmax:
        raise ValueError("independent-build-semantic-drift")
    if (first / STATE_FILENAME).read_bytes() != (second / STATE_FILENAME).read_bytes():
        raise ValueError("runtime-state-drift")
    result = {
        "status": "passed",
        "candidate_count": 2,
        "candidate_onnx_sha256": [digest(path / MODEL_FILENAME).sha256 for path in (first, second)],
        "onnx_bytes_identical": (
            (first / MODEL_FILENAME).read_bytes() == (second / MODEL_FILENAME).read_bytes()
        ),
        "cross_build_maximum_tensor_difference": cross_max,
        "cross_build_argmax_exact": cross_argmax,
        "candidates": reports,
    }
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _run(candidate: Path) -> list[np.ndarray[Any, Any]]:
    reference = np.load(candidate / "reference.npz")
    session = ort.InferenceSession(
        str(candidate / MODEL_FILENAME), providers=["CPUExecutionProvider"]
    )
    result = session.run(None, {session.get_inputs()[0].name: reference["input"]})
    return cast(list[np.ndarray[Any, Any]], result)


def _validate_candidate(candidate: Path) -> dict[str, Any]:
    model = onnx.load(candidate / MODEL_FILENAME)
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.output) != len(OUTPUTS):
        raise ValueError("output-count-mismatch")
    reference = np.load(candidate / "reference.npz")
    actual_first = _run(candidate)
    actual_second = _run(candidate)
    require_repeat_equal(actual_first, actual_second)
    output_reports: list[dict[str, Any]] = []
    maximum_differences: list[float] = []
    for index, ((name, classes), actual) in enumerate(zip(OUTPUTS, actual_first, strict=True)):
        expected = reference[f"reference_{index}"]
        report = compare_output_head(name, classes, actual, expected)
        maximum = report["maximum_difference"]
        maximum_differences.append(maximum)
        output_reports.append(report)
    state = load_runtime_state(candidate / STATE_FILENAME)
    expected_timeline = json.loads(
        (candidate / "reference-timeline.json").read_text(encoding="utf-8")
    )
    actual_timeline = decode_timeline(actual_first[0][0], actual_first[3][0], state)
    confidence_delta = compare_timelines(expected_timeline, actual_timeline)
    metadata = json.loads((candidate / "build-metadata.json").read_text(encoding="utf-8"))
    crafted_timeline = decode_timeline(reference["crafted_tag"], reference["crafted_bass"], state)
    compare_timelines(metadata["crafted_decoder_timeline"], crafted_timeline)
    return {
        "artifacts": {
            "model": digest(candidate / MODEL_FILENAME).__dict__,
            "state": digest(candidate / STATE_FILENAME).__dict__,
        },
        "maximum_tensor_difference": max(maximum_differences),
        "argmax_exact": all(row["argmax_exact"] for row in output_reports),
        "output_heads": output_reports,
        "segment_count": len(actual_timeline),
        "crafted_decoder_segment_count": len(crafted_timeline),
        "semantic_timeline_exact": True,
        "maximum_confidence_difference": confidence_delta,
        "run_to_run_drift": False,
    }


def compare_output_head(
    name: str,
    classes: int,
    actual: np.ndarray[Any, Any],
    expected: np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Enforce shape/argmax parity while reporting raw float drift diagnostically."""
    if actual.shape != expected.shape or actual.shape[-1] != classes:
        raise ValueError(f"{name}-shape-parity-failed")
    maximum = float(np.max(np.abs(actual - expected)))
    argmax = np.array_equal(np.argmax(actual, axis=-1), np.argmax(expected, axis=-1))
    if not argmax:
        raise ValueError(f"{name}-argmax-parity-failed")
    return {
        "name": name,
        "shape": list(actual.shape),
        "maximum_difference": maximum,
        "argmax_exact": argmax,
    }


def require_repeat_equal(
    first: list[np.ndarray[Any, Any]], second: list[np.ndarray[Any, Any]]
) -> None:
    """Reject any run-to-run tensor drift."""
    if any(
        not np.array_equal(left, right)
        for left, right in zip(first, second, strict=True)
    ):
        raise ValueError("run-to-run-drift")


def compare_timelines(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> float:
    """Require semantic equality and enforce the confidence-delta gate."""
    if len(actual) != len(expected):
        raise ValueError("segment-count-parity-failed")
    semantic_keys = ("start_seconds", "end_seconds", "label")
    if any(
        any(left[key] != right[key] for key in semantic_keys)
        for left, right in zip(expected, actual, strict=True)
    ):
        raise ValueError("timeline-semantic-parity-failed")
    confidence_delta = max(
        (
            abs(left["confidence"] - right["confidence"])
            for left, right in zip(expected, actual, strict=True)
        ),
        default=0.0,
    )
    if confidence_delta > MAX_CONFIDENCE_DIFFERENCE:
        raise ValueError("confidence-parity-failed")
    return confidence_delta
