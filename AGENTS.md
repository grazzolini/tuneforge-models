# AGENTS.md

Guidance for automated coding agents working in this repository.

## Scope

This repository holds source, provenance records, validation, and release
tooling for TuneForge model artifacts. Model binaries stay out of Git; approved
artifacts publish only through the gated external workflow.

## Hard rules

1. Do not add or publish model binaries, converted weights, datasets,
   copyrighted audio, or release artifacts without an explicitly approved
   publication gate.
2. Preserve exact upstream licenses, copyright notices, checksums, authorship,
   training-data provenance, and redistribution terms.
3. The repository MIT License does not relicense Crema or derived conversion
   output.
4. Treat pickle, HDF5, and model formats as untrusted input. Do not deserialize
   untrusted artifacts outside a reviewed, isolated validation path.
5. Pin GitHub Actions to full commit SHAs and keep workflow permissions read-only
   unless a documented job requires more.
6. Never bypass checks, signing, review, or artifact provenance gates.

## Workflow

- Read before editing and keep changes narrowly scoped.
- Branch names: `<type>/<brief-description>`.
- Prefer one signed commit per pull request; amend follow-up changes.
- Commit header: `<type>(<optional scope>): <subject>`.
- Include a short non-empty commit body. Lines must be at most 100 characters.
- Run `bash scripts/check-repository.sh` before opening a pull request.
- Use synthetic fixtures only. Never commit copyrighted audio.

## Stop and ask

Stop before any model upload, release, publication, license reinterpretation,
training-data use, or change that weakens a provenance or integrity gate.
