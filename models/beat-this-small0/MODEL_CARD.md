---
license: mit
library_name: executorch
pipeline_tag: audio-classification
tags:
  - beat-tracking
  - downbeat-tracking
  - music-information-retrieval
  - executorch
  - xnnpack
---

# Beat This 1.1.0 small0 FP32 for TuneForge

Dynamic ExecuTorch conversion of the pretrained `small0` beat/downbeat model from
[Beat This 1.1.0](https://github.com/CPJKU/beat_this/tree/v1.1.0). TuneForge did not
train or quantize this model.

## Intended use

On-device beat and downbeat inference for TuneForge Android. Desktop-compatible
log-mel preprocessing, chunk aggregation, and minimal peak postprocessing remain
application responsibilities. This artifact does not accept audio samples directly.

## Contract

- Input: float32 spectrogram `[1, N, 128]`, where `N` is dynamic from 13 through
  1500 frames.
- Outputs, in order: named `beat` and `downbeat` float32 logit heads, each `[1, N]`.
- Runtime: ExecuTorch 1.4.0, XNNPACK, Android arm64-v8a.
- Numerical gate: eager PyTorch parity at `rtol=atol=1e-4`; structured beat and
  downbeat timelines must match exactly on the synthetic validation case.

The exporter performs one eager zero-input call at `[1,1500,128]` before strict
dynamic `torch.export`. It does not pad inputs, replace weights, quantize, or use a
static-shape substitute.

## Reproduction and validation

Source checkpoint `small0` is 8,451,101 bytes with SHA-256
`6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b`.
The source package is beat-this 1.1.0 at tag commit
`ad7974846029835307ba19a3d5cefbf40b243041`. Export uses Torch 2.13.0 and
ExecuTorch 1.4.0.

Each release requires byte-identical independent exports, then compares one reused
ExecuTorch method with eager PyTorch at frame lengths 13, 14, 33, 34, 812 through
814, 1488, 1489, 1499, and 1500. A deterministic 16-second synthetic music fixture
must also retain exactly 33 beats and 9 downbeats. The release manifest binds the
selected artifact and host report. The PTE does not bundle the Android runtime;
TuneForge application packaging supplies that runtime separately.

## Authorship, training data, and license

Beat This authors: Francesco Foscarin, Jan Schlüter, and Gerhard Widmer. Upstream
1.1.0 ships an MIT license from the Institute of Computational Perception, JKU Linz,
and offers the named pretrained checkpoints for download and model reuse. Current
upstream documentation explicitly describes published model weights as MIT licensed;
that sentence was not present in the tagged 1.1.0 README. TuneForge preserves the
exact 1.1.0 MIT notice and records this evidence distinction.

Upstream documents training across multiple music datasets and publishes annotation
and spectrogram resources separately. It does not provide one complete
machine-readable per-track training manifest with this checkpoint. Some source audio
has separate copyright or Creative Commons restrictions. This conversion does not
claim ownership of training data or grant rights to those datasets.

ExecuTorch is BSD 3-Clause licensed. Its Android AAR is a runtime dependency, not
embedded in the PTE. Android packaging must review the AAR's resolved transitive
dependency notices separately.
