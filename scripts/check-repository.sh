#!/usr/bin/env bash

set -euo pipefail

readonly CREMA_LICENSE_SHA256="81858cdf4828dc809be65331411ceee9641e896a2fa43a703d331a03410c2dad"

readonly REQUIRED_FILES=(
  ".github/CODEOWNERS"
  ".github/ISSUE_TEMPLATE/bug_report.yml"
  ".github/ISSUE_TEMPLATE/config.yml"
  ".github/ISSUE_TEMPLATE/model_proposal.yml"
  ".github/PULL_REQUEST_TEMPLATE.md"
  ".github/dependabot.yml"
  ".github/workflows/ci.yml"
  ".gitignore"
  ".husky/commit-msg"
  "AGENTS.md"
  "CODE_OF_CONDUCT.md"
  "CONTRIBUTING.md"
  "LICENSE"
  "LICENSES/crema-0.2.0-BSD-2-Clause.txt"
  "README.md"
  "SECURITY.md"
  "THIRD_PARTY_NOTICES.md"
  "commitlint.config.cjs"
  "commitlint.pr-title.config.cjs"
  "package.json"
  "pnpm-lock.yaml"
  "scripts/check-repository.sh"
  "scripts/commitlint.test.mjs"
)

compute_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d ' ' -f 1
  else
    echo "Neither sha256sum nor shasum is available." >&2
    return 1
  fi
}

if (( $# != 0 )); then
  echo "usage: $0" >&2
  exit 2
fi

for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required repository file is missing: ${path}" >&2
    exit 1
  fi
  if ! git ls-files --error-unmatch -- "${path}" >/dev/null 2>&1; then
    echo "Required repository file is not tracked: ${path}" >&2
    exit 1
  fi
done

actual_license_sha256="$(compute_sha256 LICENSES/crema-0.2.0-BSD-2-Clause.txt)"
if [[ "${actual_license_sha256}" != "${CREMA_LICENSE_SHA256}" ]]; then
  echo "Crema 0.2.0 license text does not match the verified upstream file." >&2
  exit 1
fi

if git grep -nI -E '[[:blank:]]+$' -- .; then
  echo "Tracked text files contain trailing whitespace." >&2
  exit 1
fi

echo "Repository policy checks passed."
