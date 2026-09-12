---
tags:
  - music-information-retrieval
  - audio
---

# TuneForge model catalog

Versioned, checksummed model releases maintained for
[TuneForge](https://github.com/grazzolini/tuneforge). TuneForge did not train these models.

| Family | Task | Format | Runtime | License | Files |
| --- | --- | --- | --- | --- | --- |
| Crema 0.2.0 | Chord recognition | ONNX opset 18 | ONNX Runtime 1.29.0 CPU | BSD 2-Clause | [`crema/`](./crema/README.md) |
| Beat This 1.1.0 `small0` | Beat and downbeat tracking | ExecuTorch PTE FP32 | ExecuTorch 1.4.0 XNNPACK | MIT; runtime BSD 3-Clause | [`beat-this/`](./beat-this/README.md) |

Each family directory contains its model card, release manifest, licenses, and
family-relative `SHA256SUMS`. The Crema release also contains its fixed runtime state. The Beat
This PTE does not bundle ExecuTorch; application distributions supply the runtime separately.

The Git source, reproducible conversion tooling, and publication workflows live in the
[TuneForge models repository](https://github.com/grazzolini/tuneforge-models). New model releases
use maintainer-approved workflows, short-lived OIDC credentials, anonymous byte verification,
and artifact provenance attestations.
