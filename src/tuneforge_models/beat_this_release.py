"""Validate two Beat This exports and stage one self-contained model family."""

from __future__ import annotations

import json
import math
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .beat_this_build import input_tensor, load_exportable
from .beat_this_spec import (
    ANDROID_AAR_COORDINATE,
    ANDROID_AAR_SHA256,
    ANDROID_AAR_SIZE,
    ANDROID_ABI,
    ANDROID_API,
    ATOL,
    BACKEND,
    BEAT_THIS_LICENSE_FILENAME,
    BEAT_THIS_VERSION,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    EXECUTORCH_LICENSE_FILENAME,
    EXECUTORCH_VERSION,
    FEATURES,
    MAX_FRAMES,
    MIN_FRAMES,
    MODEL_FILENAME,
    RTOL,
    STRUCTURED_BEAT_COUNT,
    STRUCTURED_DOWNBEAT_COUNT,
    STRUCTURED_FRAMES,
    TORCH_VERSION,
    UPSTREAM_REPOSITORY,
    UPSTREAM_TAG_COMMIT,
    VALIDATION_FRAMES,
)
from .integrity import digest, verify_file
from .spec import FileDigest


def _structured_input() -> Any:
    """Recreate the synthetic music fixture without storing audio."""
    soxr = import_module("soxr")
    torch = import_module("torch")
    inference = import_module("beat_this.inference")

    sample_rate = 48_000
    chords = (
        (261.6256, 329.6276, 391.9954),
        (349.2282, 440.0000, 523.2511),
        (391.9954, 493.8833, 587.3295),
        (261.6256, 329.6276, 391.9954),
    )

    def sample(index: int) -> float:
        seconds = index / sample_rate
        chord = chords[min(int(seconds // 4), 3)]
        position = seconds % 4
        fade = min(1.0, position / 0.02, (4 - position) / 0.02)
        harmonic = sum(math.sin(2 * math.pi * hz * seconds) for hz in chord) / 3
        beat = seconds % 0.5
        click = math.exp(-beat * 80) * math.sin(2 * math.pi * 1800 * beat)
        return max(-1.0, min(1.0, 0.34 * harmonic * max(0.0, fade) + 0.18 * click))

    count = sample_rate * 16
    pcm = np.fromiter(
        (round(sample(index) * 32767) for index in range(count)),
        dtype=np.int16,
        count=count,
    )
    signal = soxr.resample(pcm.astype(np.float64) / 32768.0, sample_rate, 22_050)
    spectrogram = inference.LogMelSpect()(torch.tensor(signal, dtype=torch.float32))
    chunks, starts = inference.split_piece(spectrogram, MAX_FRAMES, 6, True)
    if tuple(spectrogram.shape) != (801, FEATURES) or list(starts) != [-6]:
        raise ValueError("structured-fixture-regression")
    value = chunks[0].unsqueeze(0)
    if tuple(value.shape) != (1, STRUCTURED_FRAMES, FEATURES):
        raise ValueError("structured-fixture-regression")
    return value


def _outputs(values: Any, frames: int) -> tuple[Any, Any]:
    if not isinstance(values, tuple | list) or len(values) != 2:
        raise ValueError("output-contract-failed")
    result = tuple(value.detach().cpu() for value in values)
    if any(tuple(value.shape) != (1, frames) for value in result):
        raise ValueError("output-contract-failed")
    return result


def _timeline(values: tuple[Any, Any]) -> tuple[list[float], list[float]]:
    Postprocessor = import_module("beat_this.model.postprocessor").Postprocessor

    beats, downbeats = Postprocessor(type="minimal", fps=50)(values[0][0], values[1][0])
    return [float(value) for value in beats], [float(value) for value in downbeats]


def _metadata(candidate: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(
        (candidate / "build-metadata.json").read_text(encoding="utf-8")
    )
    expected = {
        "source_checkpoint": {"size": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256},
        "precision": "float32",
        "backend": BACKEND,
        "pre_export_warmup": [1, MAX_FRAMES, FEATURES],
        "range_constraints": "{s11: VR[13, 1500]}",
    }
    environment = value.get("environment", {})
    if (
        any(value.get(name) != expected_value for name, expected_value in expected.items())
        or environment.get("beat_this") != BEAT_THIS_VERSION
        or environment.get("torch") != TORCH_VERSION
        or environment.get("executorch") != EXECUTORCH_VERSION
        or value.get("rotary_cache_modules", 0) < 1
        or value.get("xnnpack_delegates", 0) < 1
        or value.get("artifact") != digest(candidate / MODEL_FILENAME).__dict__
    ):
        raise ValueError("candidate-metadata-failed")
    return value


def _validate(first: Path, second: Path, checkpoint: Path) -> dict[str, Any]:
    torch = import_module("torch")
    Runtime = import_module("executorch.runtime").Runtime

    verify_file(checkpoint, FileDigest(CHECKPOINT_SIZE, CHECKPOINT_SHA256))
    candidates = (first, second)
    metadata = [_metadata(candidate) for candidate in candidates]
    if (first / MODEL_FILENAME).read_bytes() != (second / MODEL_FILENAME).read_bytes():
        raise ValueError("candidate-reproducibility-failed")

    structured = _structured_input()
    cases = [(f"linspace-{frames}", input_tensor(frames)) for frames in VALIDATION_FRAMES]
    cases.append(("synthetic-music", structured))
    eager = load_exportable(checkpoint)
    with torch.inference_mode():
        references = [(name, _outputs(eager(value), value.shape[1])) for name, value in cases]
    reference_timeline = _timeline(references[-1][1])
    if tuple(map(len, reference_timeline)) != (
        STRUCTURED_BEAT_COUNT,
        STRUCTURED_DOWNBEAT_COUNT,
    ):
        raise ValueError("structured-fixture-event-regression")

    reports = []
    for candidate, candidate_metadata in zip(candidates, metadata, strict=True):
        # Program._data owns the byte buffer borrowed by the native runtime.
        program = Runtime.get().load_program((candidate / MODEL_FILENAME).read_bytes())
        method = program.load_method("forward")
        if method is None:
            raise ValueError("forward-method-missing")
        rows = []
        maximum = 0.0
        for (name, value), (_, expected) in zip(cases, references, strict=True):
            actual = _outputs(method.execute((value,)), value.shape[1])
            heads = {}
            for head, left, right in zip(("beat", "downbeat"), expected, actual, strict=True):
                delta = np.abs(left.numpy().astype(np.float64) - right.numpy().astype(np.float64))
                if not np.allclose(left.numpy(), right.numpy(), rtol=RTOL, atol=ATOL):
                    raise ValueError("host-parity-failed")
                heads[head] = float(np.max(delta, initial=0.0))
                maximum = max(maximum, heads[head])
            rows.append({"case": name, "frames": int(value.shape[1]), "maximum_delta": heads})
        structured_actual = _outputs(method.execute((structured,)), STRUCTURED_FRAMES)
        if _timeline(structured_actual) != reference_timeline:
            raise ValueError("structured-timeline-parity-failed")
        reports.append(
            {
                "artifact": digest(candidate / MODEL_FILENAME).__dict__,
                "build_metadata": digest(candidate / "build-metadata.json").__dict__,
                "exporter_revision": candidate_metadata["exporter_revision"],
                "maximum_delta": maximum,
                "cases": rows,
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "checkpoint": {"size": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256},
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "candidate_bytes_identical": True,
        "candidates": reports,
        "structured_timeline": {
            "beat_times": STRUCTURED_BEAT_COUNT,
            "downbeat_times": STRUCTURED_DOWNBEAT_COUNT,
            "exact": True,
        },
    }


def release(
    first: Path, second: Path, checkpoint: Path, output: Path, repository: Path
) -> dict[str, Any]:
    """Validate exact candidates, then stage candidate A with provenance and checksums."""
    if output.exists():
        raise ValueError("output-already-exists")
    validation = _validate(first, second, checkpoint)
    output.mkdir(parents=True)
    licenses = output / "LICENSES"
    licenses.mkdir()
    shutil.copyfile(first / MODEL_FILENAME, output / MODEL_FILENAME)
    shutil.copyfile(repository / "models/beat-this-small0/MODEL_CARD.md", output / "README.md")
    for name in (BEAT_THIS_LICENSE_FILENAME, EXECUTORCH_LICENSE_FILENAME):
        shutil.copyfile(repository / "LICENSES" / name, licenses / name)
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_metadata = _metadata(first)
    files = {
        path.relative_to(output).as_posix(): digest(path).__dict__
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "model": {"name": "beat-this", "variant": "small0", "format": "pte"},
        "source": {
            "version": BEAT_THIS_VERSION,
            "repository": UPSTREAM_REPOSITORY,
            "tag_commit": UPSTREAM_TAG_COMMIT,
            "checkpoint_url": CHECKPOINT_URL,
            "checkpoint": validation["checkpoint"],
            "license": "MIT",
            "trained_by_tuneforge": False,
            "training_manifest": "not-provided-upstream",
        },
        "conversion": {
            "exporter_revision": build_metadata["exporter_revision"],
            "environment": build_metadata["environment"],
            "backend": BACKEND,
            "pre_export_warmup": build_metadata["pre_export_warmup"],
            "xnnpack_delegates": build_metadata["xnnpack_delegates"],
        },
        "contract": {
            "input": {"dtype": "float32", "shape": [1, "frames", FEATURES]},
            "frames": {"minimum": MIN_FRAMES, "maximum": MAX_FRAMES},
            "outputs": ["beat", "downbeat"],
            "tolerances": validation["tolerances"],
        },
        "consumer_runtime": {
            "coordinate": ANDROID_AAR_COORDINATE,
            "aar": {"size": ANDROID_AAR_SIZE, "sha256": ANDROID_AAR_SHA256},
            "api": ANDROID_API,
            "abi": ANDROID_ABI,
        },
        "validation": {"status": "passed", "report": "validation.json"},
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = [*files, "manifest.json"]
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest(output / name).sha256}  {name}\n" for name in sorted(names)),
        encoding="utf-8",
    )
    return manifest
