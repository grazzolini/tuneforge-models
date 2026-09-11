#!/usr/bin/env bash

set -euo pipefail

readonly CREMA_LICENSE_SHA256="81858cdf4828dc809be65331411ceee9641e896a2fa43a703d331a03410c2dad"
readonly BEAT_THIS_LICENSE_SHA256="909ab6549794a18e9bb243aacfadda4a5f308436fc2846f350755c53c4f06ae1"
readonly EXECUTORCH_LICENSE_SHA256="c58707ea5d5c0ee17af7e16f1377e4d31ad1533492ab0bf9b87ebd9f718f9e7b"

readonly REQUIRED_FILES=(
  ".github/CODEOWNERS"
  ".github/ISSUE_TEMPLATE/bug_report.yml"
  ".github/ISSUE_TEMPLATE/config.yml"
  ".github/ISSUE_TEMPLATE/model_proposal.yml"
  ".github/PULL_REQUEST_TEMPLATE.md"
  ".github/dependabot.yml"
  ".github/workflows/ci.yml"
  ".github/workflows/publish-beat-this-small0.yml"
  ".gitignore"
  ".husky/commit-msg"
  "AGENTS.md"
  "CODE_OF_CONDUCT.md"
  "CONTRIBUTING.md"
  "LICENSE"
  "LICENSES/crema-0.2.0-BSD-2-Clause.txt"
  "LICENSES/beat-this-1.1.0-MIT.txt"
  "LICENSES/executorch-1.4.0-BSD-3-Clause.txt"
  "models/beat-this-small0/MODEL_CARD.md"
  "models/beat-this-small0/build-spec.json"
  "models/crema-0.2.0/MODEL_CARD.md"
  "models/crema-0.2.0/build-spec.json"
  "pyproject.toml"
  "README.md"
  "SECURITY.md"
  "THIRD_PARTY_NOTICES.md"
  "commitlint.config.cjs"
  "commitlint.pr-title.config.cjs"
  "package.json"
  "pnpm-lock.yaml"
  "scripts/check-repository.sh"
  "scripts/commitlint.test.mjs"
  "src/tuneforge_models/__init__.py"
  "src/tuneforge_models/build.py"
  "src/tuneforge_models/beat_this_build.py"
  "src/tuneforge_models/beat_this_release.py"
  "src/tuneforge_models/beat_this_spec.py"
  "src/tuneforge_models/cli.py"
  "src/tuneforge_models/integrity.py"
  "src/tuneforge_models/release.py"
  "src/tuneforge_models/runtime_state.py"
  "src/tuneforge_models/spec.py"
  "src/tuneforge_models/validate.py"
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

actual_beat_this_license_sha256="$(compute_sha256 LICENSES/beat-this-1.1.0-MIT.txt)"
if [[ "${actual_beat_this_license_sha256}" != "${BEAT_THIS_LICENSE_SHA256}" ]]; then
  echo "Beat This 1.1.0 license text does not match the verified upstream file." >&2
  exit 1
fi

actual_executorch_license_sha256="$(compute_sha256 LICENSES/executorch-1.4.0-BSD-3-Clause.txt)"
if [[ "${actual_executorch_license_sha256}" != "${EXECUTORCH_LICENSE_SHA256}" ]]; then
  echo "ExecuTorch 1.4.0 license text does not match the verified upstream file." >&2
  exit 1
fi

if git grep -nI -E '[[:blank:]]+$' -- .; then
  echo "Tracked text files contain trailing whitespace." >&2
  exit 1
fi

if git ls-files -z | grep -zE '\.(ckpt|h5|onnx|pkl|npz|pte)$' >/dev/null; then
  echo "Generated model, weight, pickle, or reference artifact is tracked." >&2
  exit 1
fi

echo "Repository policy checks passed."
