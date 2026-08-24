# Security Policy

## Supported versions

This repository ships from `main`. Only the latest commit on `main` receives
security fixes.

## In scope

- Malicious or unexpectedly executable model or serialized-data artifacts
- Unsafe deserialization in conversion or validation tooling
- Artifact substitution, checksum bypass, or provenance tampering
- Compromised dependencies or GitHub Actions workflows
- Secrets or private data committed to the repository
- Release-process weaknesses that could publish an unreviewed artifact

Licensing or provenance concerns without a security impact may use the model
proposal issue form. Do not attach model files, datasets, copyrighted audio, or
private material to any report.

## Reporting a vulnerability

Do **not** open a public issue for security problems.

- Use GitHub private vulnerability reporting:
  <https://github.com/grazzolini/tuneforge-models/security/advisories/new>
- Or contact `@grazzolini` directly through the GitHub profile.

Include a description, impact, reproduction steps, affected commit or artifact
checksum, and suggested mitigation when available. There is no bounty program.
