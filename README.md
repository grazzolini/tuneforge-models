# TuneForge Models

Source, provenance, and release tooling for machine-learning models used by
[TuneForge](https://github.com/grazzolini/tuneforge).

## Model families

Crema 0.2.0 ONNX is publicly available from the
[TuneForge model Hub](https://huggingface.co/grazzolini/tuneforge-models). TuneForge uses the
converted Crema model with ONNX Runtime as the sole Advanced Chords implementation while
preserving the `crema-advanced` engine identity. TuneForge packages the pinned 2.2 MB model and
runtime state whenever Advanced Chords is enabled.

Beat This 1.1.0 `small0` FP32 export tooling targets ExecuTorch 1.4.0/XNNPACK
with a dynamic `[1,N,128]` input contract for `N=13..1500`. Its generated PTE is publicly
available from the model Hub and stays out of Git. Publication requires maintainer approval,
two-build host validation, anonymous download verification, and provenance attestation.

This Git repository contains source, provenance records, validation, and
publication tooling rather than model binaries. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for both model records.

## Publication

Maintainers publish each family from `main` by manually dispatching its workflow.
Both workflows require the protected `model-publication` environment, rebuild and
validate two candidates, authenticate to Hugging Face with a short-lived OIDC
credential, overlay only their family plus the shared Hub catalog, preserve the sibling family's
exact bytes, verify the immutable revision through an anonymous download, and attest the verified
files. The model Hub exposes the shared catalog at its root and keeps each maintained release in
the versionless `crema/` and `beat-this/` family directories.

## License

Repository-authored source and documentation are licensed under the
[MIT License](LICENSE), unless a file says otherwise. Third-party materials keep
their original licenses and attribution. The MIT License does not relicense
Crema, its model files, or any future conversion output derived from them.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing model sources,
conversion work, or release artifacts. Follow the approved publication gate
before adding model files or datasets, and never commit copyrighted audio.
Source, provenance, and release tooling for TuneForge models.
