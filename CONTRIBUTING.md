# Contributing

Thanks for helping improve TuneForge's model provenance and release process.

## Before opening a change

- Search existing issues and pull requests.
- Use an issue form for a new model or conversion proposal.
- Document the exact upstream source, version or commit, artifact checksums,
  license files, model authors, training-data provenance, and redistribution
  terms.
- Do not upload model binaries, datasets, copyrighted audio, credentials, or
  generated conversion output before maintainers approve the publication gate.

## Pull requests

1. Create a focused branch named `<type>/<brief-description>`.
2. Keep one concern and preferably one commit per pull request.
3. Use Conventional Commits: `<type>(<optional scope>): <subject>`.
4. Include a non-empty commit body describing the important change.
5. Run `bash scripts/check-repository.sh` and report the command honestly.

Allowed commit types are `feat`, `fix`, `perf`, `refactor`, `docs`, `test`,
`build`, `ci`, `chore`, and `revert`. Commit and pull-request title lines must
not exceed 100 characters.

## Local validation

After [source-checkout setup](README.md#local-setup-and-notation-core), run:

```sh
uv run --frozen ruff check .
uv run --frozen mypy
uv run --frozen pytest
bash scripts/check-repository.sh
```

Keep `--reinstall-package tuneforge-music-core` on dev syncs and isolated conversion runs:
the native source can change while its package version stays unchanged.

```sh
uv run --frozen --isolated --reinstall-package tuneforge-music-core \
  --no-group dev --group convert tuneforge-models build \
  --wheel /path/to/crema-0.2.0-py3-none-any.whl --output /path/to/new-build
```

After an intentional core pin change, rerun preparation before syncing. Preparation verifies the
clean registered pin and never updates Git; `--prepare --offline` requires verified cached or
vendored sources.

Only the submodule's `packages/music-core` is consumed. [The build spec](models/crema/build-spec.json)
pins its source, contract and vocabulary identity, recorded in build, validation and conversion
metadata. Runtime-state JSON and model tensors keep their format. Crema/pumpp/mir_eval remains the
independent conversion oracle; core binaries stay out of model releases.

## Licensing boundary

The repository's MIT License applies only to repository-authored materials
unless a file states otherwise. Preserve third-party licenses and attribution.
Do not claim that MIT relicenses upstream models or derived conversion output.

Model publication requires all of the following:

- authoritative source and immutable artifact checksum;
- complete author and license attribution;
- documented training-data provenance;
- confirmed rights for redistribution and conversion output;
- reproducible conversion and validation evidence;
- maintainer approval of the publication gate.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.
