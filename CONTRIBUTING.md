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
