# TuneForge Models

Source, provenance, and release tooling for machine-learning models used by
[TuneForge](https://github.com/grazzolini/tuneforge).

## Current status

Crema 0.2.0 ONNX is publicly available from
[Hugging Face](https://huggingface.co/grazzolini/tuneforge-models/tree/65af18f49af5101267fd28f15ac8c452d98b8e3d)
at immutable revision `65af18f49af5101267fd28f15ac8c452d98b8e3d`. TuneForge uses the converted
Crema model with ONNX Runtime as the sole Advanced Chords implementation while preserving the `crema-advanced`
engine identity. TuneForge packages the pinned 2.2 MB model and runtime state whenever Advanced
Chords is enabled. This Git repository
contains source, provenance records, validation, and publication tooling rather
than model binaries. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the Crema 0.2.0 record.

## Publication

Maintainers publish from `main` by manually dispatching the `Publish Crema ONNX`
GitHub Actions workflow. It rebuilds and validates two candidates, authenticates
to Hugging Face with a short-lived OIDC credential, updates the current Hub head,
verifies the resulting immutable revision through an anonymous download, and
records GitHub build-provenance attestations for the verified files.

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
