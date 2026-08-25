"""Isolated, reproducible Crema TensorFlow-to-ONNX build."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .integrity import digest, extract_verified_crema_members
from .runtime_state import export_runtime_state, load_runtime_state, trusted_reference_timeline
from .spec import CREMA_MEMBERS, MODEL_FILENAME, OPSET, OUTPUTS, STATE_FILENAME

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def build(wheel: Path, output: Path) -> dict[str, Any]:
    """Build one candidate and its private validation references."""
    if output.exists():
        raise ValueError("output-already-exists")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="tuneforge-crema-source-") as temporary:
        members = extract_verified_crema_members(wheel, Path(temporary) / "source")
        _verify_installed_crema(wheel, members)
        state_path = output / STATE_FILENAME
        export_runtime_state(members["crema/models/chord/pump.pkl"], state_path)
        metadata = _convert(output / MODEL_FILENAME, output / "reference.npz")
        reference = np.load(output / "reference.npz")
        state = load_runtime_state(state_path)
        crafted_tag, crafted_bass = _crafted_decoder_inputs(state)
        np.savez_compressed(
            output / "reference.npz",
            **{name: reference[name] for name in reference.files},
            crafted_tag=crafted_tag,
            crafted_bass=crafted_bass,
        )
        timeline = trusted_reference_timeline(
            members["crema/models/chord/pump.pkl"],
            reference["reference_0"][0],
            reference["reference_3"][0],
        )
        metadata["crafted_decoder_timeline"] = trusted_reference_timeline(
            members["crema/models/chord/pump.pkl"], crafted_tag, crafted_bass
        )
    (output / "reference-timeline.json").write_text(
        json.dumps(timeline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["artifacts"] = {
        name: digest(output / name).__dict__ for name in (MODEL_FILENAME, STATE_FILENAME)
    }
    (output / "build-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _crafted_decoder_inputs(state: dict[str, Any]) -> tuple[np.ndarray[Any, Any], ...]:
    labels = state["decoder"]["labels"]
    tag = np.full((12, len(labels)), 0.01 / (len(labels) - 1), dtype=np.float32)
    bass = np.full((12, 13), 0.01 / 12, dtype=np.float32)
    for segment, (label, bass_index) in enumerate((("C:maj7", 4), ("D:min7", 9), ("G:7", 11))):
        frames = slice(segment * 4, segment * 4 + 4)
        tag[frames, labels.index(label)] = 0.99
        bass[frames, bass_index] = 0.99
    return tag, bass


def _verify_installed_crema(wheel: Path, members: dict[str, Path]) -> None:
    installed_distribution = importlib.metadata.distribution("crema")
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.startswith("crema/"):
                continue
            installed_file = Path(str(installed_distribution.locate_file(info.filename)))
            if not installed_file.is_file() or installed_file.read_bytes() != archive.read(info):
                raise ValueError("installed-crema-does-not-match-source")
    installed = Path(str(installed_distribution.locate_file("crema/models/chord")))
    for member, (size, sha256) in CREMA_MEMBERS.items():
        source = members[member]
        if digest(source).__dict__ != {"size": size, "sha256": sha256}:
            raise ValueError("source-integrity-failed")
        installed_member = installed / Path(member).name
        if not installed_member.is_file() or digest(installed_member) != digest(source):
            raise ValueError("installed-crema-does-not-match-source")


def _convert(onnx_output: Path, reference_output: Path) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    import onnx
    import tensorflow as tf  # type: ignore[import-untyped]
    import tf2onnx  # type: ignore[import-not-found]
    from crema.models.chord import ChordModel  # type: ignore[import-not-found]

    tf.keras.utils.set_random_seed(474)
    model = ChordModel().model
    if tuple(model.output_names) != tuple(name for name, _ in OUTPUTS):
        raise ValueError("unexpected-output-order")
    signature = (tf.TensorSpec((None, None, 216, 2), tf.float32, name="cqt_mag"),)

    @tf.function(input_signature=signature)  # type: ignore[misc]
    def serving(cqt_mag: Any) -> tuple[Any, ...]:
        return tuple(model(cqt_mag, training=False))

    model_proto, _ = tf2onnx.convert.from_function(
        serving,
        input_signature=signature,
        opset=OPSET,
        output_path=str(onnx_output),
    )
    onnx.checker.check_model(model_proto, full_check=True)
    rng = np.random.default_rng(474)
    tensor = rng.random((1, 73, 216, 2), dtype=np.float32)
    references = [output.numpy() for output in serving(tf.convert_to_tensor(tensor))]
    for (name, classes), value in zip(OUTPUTS, references, strict=True):
        if value.shape != (1, 73, classes):
            raise ValueError(f"unexpected-{name}-shape")
    np.savez_compressed(
        reference_output,
        input=tensor,
        **{f"reference_{index}": value for index, value in enumerate(references)},
    )
    return {
        "schema_version": 1,
        "tensorflow_version": tf.__version__,
        "tf2onnx_version": tf2onnx.__version__,
        "onnx_version": importlib.metadata.version("onnx"),
        "opset": OPSET,
        "parameter_count": int(model.count_params()),
        "input": {"name": "cqt_mag", "shape": [None, None, 216, 2], "dtype": "float32"},
        "outputs": [
            {
                "semantic_name": name,
                "onnx_name": model_proto.graph.output[index].name,
                "shape": [None, None, classes],
            }
            for index, (name, classes) in enumerate(OUTPUTS)
        ],
    }


def copy_release_inputs(build_directory: Path, release_directory: Path) -> None:
    """Copy only publishable, pickle-free inputs from a validated build."""
    load_runtime_state(build_directory / STATE_FILENAME)
    release_directory.mkdir(parents=True, exist_ok=False)
    for name in (MODEL_FILENAME, STATE_FILENAME):
        shutil.copyfile(build_directory / name, release_directory / name)
