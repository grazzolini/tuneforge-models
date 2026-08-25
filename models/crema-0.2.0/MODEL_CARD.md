---
license: bsd-2-clause
library_name: onnxruntime
pipeline_tag: audio-classification
tags:
  - chord-recognition
  - music-information-retrieval
  - onnx
---

# Crema 0.2.0 ONNX for TuneForge

ONNX conversion of the pretrained chord-recognition model distributed in
[Crema 0.2.0](https://github.com/bmcfee/crema/tree/0.2.0). TuneForge did not
train this model. This repository supplies a reproducible format conversion,
fixed inference state, integrity manifest, and validation tooling.

## Intended use

Experimental CPU chord recognition in local TuneForge builds. TensorFlow Crema
remains TuneForge's default while this conversion receives day-to-day testing.
The model is not intended for safety-critical or unrelated audio classification.

## Inputs and outputs

- Input: float32 HCQT magnitude, shape `[batch, frames, 216, 2]`, sampled at
  44,100 Hz with hop length 4,096.
- Outputs, in order: `chord_tag` (170 classes), `chord_pitch` (12),
  `chord_root` (13), and `chord_bass` (13).
- Runtime state: reviewed JSON containing fixed HCQT parameters, the ordered
  chord vocabulary, and compact transition probabilities. Runtime loading does
  not use pickle.

## Conversion and validation

The verified Crema wheel and embedded H5 weights are converted with TensorFlow
2.15.1, tf2onnx 1.16.1, ONNX 1.17.0, and opset 18. ONNX Runtime 1.29.0 uses only
`CPUExecutionProvider`.

Each release build validates two independent conversions. Required gates:

- reported TensorFlow/ONNX tensor difference for numerical diagnostics;
- exact argmax parity across all four heads;
- exact normalized labels, boundaries, extensions, and inversions;
- exact normalized timelines, which provide identical input to TuneForge's
  deterministic scorer;
- maximum aligned confidence difference `0.10`;
- zero run-to-run timeline drift.

The release manifest records actual artifact hashes and sizes. Rebuilding may
produce a different ONNX byte hash. Exact argmax and timeline behavior, not
raw floating-point or byte identity, form the reproducibility gate.

A separate sanitized private checkpoint compared six scorer results directly
and found zero mismatches. Repository CI does not import or invoke TuneForge's
scorer.

## Source, authorship, and provenance

- Upstream author: Brian McFee.
- Crema version: 0.2.0; tag commit
  `051c91697fd16856a0a1019cc06ee1f11fb52c5f`.
- Verified wheel SHA-256:
  `b2787afd0367463438ca2b9b2944c490308eee1f307e5796078ec540a6281484`.
- Embedded H5 SHA-256:
  `08b80e5b648e743c89284e9bc0b12b993dad1129157a75e0de70e076b0b8a235`.

Crema's distribution includes the pretrained model but does not provide a
machine-readable per-track training manifest. Upstream publications and source
documentation describe evaluation and development using established chord
annotation corpora; this conversion cannot independently reconstruct or audit
the complete training set. Do not infer dataset ownership or broader rights
from this model card.

## License

Crema ships a BSD 2-Clause license covering its distribution, including the
packaged model files. PyPI metadata labels version 0.2.0 as ISC, conflicting
with the byte-identical license files in the wheel, source distribution, and
upstream repository. This release preserves the exact shipped BSD notice in
`LICENSES/crema-0.2.0-BSD-2-Clause.txt`.

TuneForge's MIT license does not relicense Crema or this converted artifact.
