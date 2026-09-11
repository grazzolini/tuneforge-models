"""Pinned Beat This small0 export to dynamic FP32 ExecuTorch/XNNPACK."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from importlib import import_module
from pathlib import Path
from typing import Any

from .beat_this_spec import (
    BACKEND,
    BEAT_THIS_VERSION,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    EXECUTORCH_VERSION,
    FEATURES,
    MAX_FRAMES,
    MIN_FRAMES,
    MODEL_FILENAME,
    TORCH_VERSION,
    TRACE_FRAMES,
)
from .integrity import digest, verify_file
from .spec import FileDigest


def input_tensor(frames: int) -> Any:
    torch = import_module("torch")

    return torch.linspace(-1.0, 1.0, frames * FEATURES, dtype=torch.float32).reshape(
        1, frames, FEATURES
    )


def load_exportable(checkpoint: Path) -> Any:
    torch = import_module("torch")
    load_model = import_module("beat_this.inference").load_model

    class ExportableBeatThis(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.model = load_model(str(checkpoint), device="cpu")

        def forward(self, spectrogram: Any) -> tuple[Any, Any]:
            result = self.model(spectrogram)
            return result["beat"], result["downbeat"]

    return ExportableBeatThis().eval()


def _environment() -> dict[str, str]:
    torch = import_module("torch")

    actual = {
        "python": platform.python_version(),
        "beat_this": importlib.metadata.version("beat-this"),
        "torch": torch.__version__.split("+", 1)[0],
        "executorch": importlib.metadata.version("executorch"),
    }
    expected = {
        "beat_this": BEAT_THIS_VERSION,
        "torch": TORCH_VERSION,
        "executorch": EXECUTORCH_VERSION,
    }
    if any(actual[name] != version for name, version in expected.items()):
        raise ValueError("unexpected-beat-this-build-environment")
    return actual


def build(checkpoint: Path, output: Path, exporter_revision: str) -> dict[str, Any]:
    """Verify the checkpoint, warm the unchanged model, and export one candidate."""
    if output.exists():
        raise ValueError("output-already-exists")
    if exporter_revision != "working-tree" and re.fullmatch(
        r"[0-9a-f]{40}", exporter_revision
    ) is None:
        raise ValueError("invalid-exporter-revision")
    verify_file(checkpoint, FileDigest(CHECKPOINT_SIZE, CHECKPOINT_SHA256))
    environment = _environment()

    torch = import_module("torch")
    XnnpackPartitioner = import_module(
        "executorch.backends.xnnpack.partition.xnnpack_partitioner"
    ).XnnpackPartitioner
    to_edge_transform_and_lower = import_module(
        "executorch.exir"
    ).to_edge_transform_and_lower
    RotaryEmbedding = import_module("rotary_embedding_torch").RotaryEmbedding

    model = load_exportable(checkpoint)
    with torch.inference_mode():
        model(torch.zeros((1, MAX_FRAMES, FEATURES), dtype=torch.float32))
    rotary = [module for module in model.modules() if isinstance(module, RotaryEmbedding)]
    if not rotary or any(module.cached_freqs_seq_len < MAX_FRAMES for module in rotary):
        raise ValueError("maximum-shape-warmup-failed")

    frames = torch.export.Dim("frames", min=MIN_FRAMES, max=MAX_FRAMES)
    exported = torch.export.export(
        model, (input_tensor(TRACE_FRAMES),), dynamic_shapes=({1: frames},), strict=True
    )
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    nodes = list(edge.exported_program().graph_module.graph.nodes)
    delegates = sum("executorch_call_delegate" in str(node.target) for node in nodes)
    if delegates == 0:
        raise ValueError("xnnpack-delegation-missing")

    output.mkdir(parents=True)
    model_path = output / MODEL_FILENAME
    model_path.write_bytes(edge.to_executorch().buffer)
    metadata = {
        "schema_version": 1,
        "exporter_revision": exporter_revision,
        "source_checkpoint": {"size": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256},
        "environment": environment,
        "precision": "float32",
        "backend": BACKEND,
        "pre_export_warmup": [1, MAX_FRAMES, FEATURES],
        "rotary_cache_modules": len(rotary),
        "range_constraints": str(exported.range_constraints),
        "xnnpack_delegates": delegates,
        "artifact": digest(model_path).__dict__,
    }
    (output / "build-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
