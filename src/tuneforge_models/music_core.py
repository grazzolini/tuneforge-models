"""Verified source-checkout identity and typed access to the shared notation core."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
SUBMODULE = Path("third_party/tuneforge")
CORE = SUBMODULE / "packages/music-core"


class CremaChord(TypedDict):
    kind: str
    root_pitch_class: int | None
    quality: str | None


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.PIPE
    ).strip()


def notation_identity(root: Path = ROOT) -> dict[str, Any]:
    """Verify the registered gitlink, clean checkout, package, contract and fixture."""
    expected: dict[str, Any] = json.loads(
        (root / "models/crema/build-spec.json").read_text(encoding="utf-8")
    )["notation_core"]
    checkout = root / SUBMODULE
    core = root / CORE
    if not (checkout / ".git").is_file():
        raise ValueError("music-core-submodule-missing")
    if (
        _git(root, "config", "-f", ".gitmodules", "--get", "submodule.third_party/tuneforge.path")
        != SUBMODULE.as_posix()
        or _git(root, "config", "-f", ".gitmodules", "--get", "submodule.third_party/tuneforge.url")
        != expected["repository"]
    ):
        raise ValueError("music-core-submodule-registration-mismatch")
    link = _git(root, "ls-files", "--stage", "--", SUBMODULE.as_posix()).split()
    if (
        link != ["160000", expected["revision"], "0", SUBMODULE.as_posix()]
        or _git(checkout, "rev-parse", "HEAD") != expected["revision"]
    ):
        raise ValueError("music-core-revision-mismatch")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("music-core-submodule-dirty")
    package = tomllib.loads((core / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if package["name"] != expected["package"] or package["version"] != expected["package_version"]:
        raise ValueError("music-core-package-mismatch")
    abi = (core / "src/abi.cpp").read_text(encoding="utf-8")
    for name in ("api", "abi", "schema", "corpus"):
        match = re.search(rf"tf_music_core_{name}_version\(\) noexcept \{{ return (\d+); \}}", abi)
        if match is None or int(match[1]) != expected[f"{name}_version"]:
            raise ValueError("music-core-contract-mismatch")
    fixture_bytes = (core / expected["fixture"]["path"]).read_bytes()
    fixture = json.loads(fixture_bytes)
    if (
        hashlib.sha256(fixture_bytes).hexdigest() != expected["fixture"]["sha256"]
        or fixture["classes_sha256"] != expected["fixture"]["classes_sha256"]
        or len(fixture["labels"]) != 170
        or hashlib.sha256("\n".join(fixture["labels"]).encode()).hexdigest()
        != expected["fixture"]["classes_sha256"]
    ):
        raise ValueError("music-core-fixture-mismatch")
    return expected


def crema_labels() -> list[str]:
    """Load the shared model vocabulary, preserving its historical oracle provenance."""
    identity = notation_identity()
    fixture = json.loads((ROOT / CORE / identity["fixture"]["path"]).read_text(encoding="utf-8"))
    return cast(list[str], fixture["labels"])


def prepare(*, offline: bool = False, root: Path = ROOT) -> dict[str, Any]:
    """Acquire only verified locked sources; never initialize or update the submodule."""
    if sys.version_info[:2] != (3, 11):
        raise ValueError("music-core-preparation-requires-python-3.11")
    identity = notation_identity(root)
    command = [sys.executable, str(root / CORE / "tools/prepare_sources.py"), "--fetch-sources"]
    if offline:
        command.append("--offline")
    subprocess.run(command, check=True)
    notation_identity(root)
    return identity


@lru_cache(maxsize=1)
def _binding() -> ModuleType:
    import tuneforge_music_core

    identity = notation_identity()
    if importlib.metadata.version("tuneforge-music-core") != identity["package_version"]:
        raise ValueError("music-core-installed-package-mismatch")
    if any(
        getattr(tuneforge_music_core, f"{name.upper()}_VERSION") != identity[f"{name}_version"]
        for name in ("api", "abi", "schema", "corpus")
    ):
        raise ValueError("music-core-installed-contract-mismatch")
    return tuneforge_music_core


def parse_crema(label: str) -> CremaChord:
    value: object = _binding().call({"operation": "nativeParseCrema", "input": label})
    if not isinstance(value, dict):
        raise ValueError("music-core-parse-response-mismatch")
    kind, root, quality = value.get("kind"), value.get("root_pitch_class"), value.get("quality")
    if kind in {"no_chord", "unknown"} and root is None and quality is None:
        return {"kind": kind, "root_pitch_class": None, "quality": None}
    if kind == "chord" and type(root) is int and 0 <= root < 12 and isinstance(quality, str):
        return {"kind": kind, "root_pitch_class": root, "quality": quality}
    raise ValueError("music-core-parse-response-mismatch")


def crema_contains(label: str, relative: int) -> bool:
    value: object = _binding().call(
        {"operation": "nativeCremaContains", "input": label, "relative": relative}
    )
    if type(value) is not bool:
        raise ValueError("music-core-contains-response-mismatch")
    return value


def degree_name(relative: int) -> str:
    value: object = _binding().call({"operation": "nativeDegreeName", "input": relative})
    if not isinstance(value, str):
        raise ValueError("music-core-degree-response-mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline and not args.prepare:
        parser.error("--offline requires --prepare")
    print(json.dumps(prepare(offline=args.offline) if args.prepare else notation_identity()))


if __name__ == "__main__":
    main()
