"""Download Beat This small0 from upstream or its checksum-identical maintainer mirror."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .beat_this_spec import CHECKPOINT_SHA256, CHECKPOINT_SIZE, CHECKPOINT_URL
from .integrity import verify_file
from .spec import FileDigest

MIRROR_URL = "https://f0k.de/pubs/2024_ismir/beat_this-small0.ckpt"
# curl transport, TLS, redirect, and HTTP failures; local write/configuration errors abort.
_SOURCE_ERRORS = frozenset({5, 6, 7, 16, 18, 22, 28, 35, 47, 52, 55, 56, 60, 92, 95, 96, 97})


def download_checkpoint(destination: Path) -> None:
    """Publish only verified bytes, preserving an existing destination on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".beat-this-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "checkpoint.ckpt"
        for url in (CHECKPOINT_URL, MIRROR_URL):
            candidate.unlink(missing_ok=True)
            print(f"Downloading Beat This checkpoint: {url}", flush=True)
            try:
                result = subprocess.run(
                    [
                        "curl", "--fail", "--location", "--silent", "--show-error",
                        "--proto", "=https", "--proto-redir", "=https",
                        "--connect-timeout", "15", "--max-time", "60",
                        "--retry", "3", "--retry-max-time", "90",
                        "--output", str(candidate), url,
                    ],
                    check=False,
                    timeout=180,
                )
            except subprocess.TimeoutExpired:
                continue
            if result.returncode in _SOURCE_ERRORS:
                continue
            result.check_returncode()
            verify_file(candidate, FileDigest(CHECKPOINT_SIZE, CHECKPOINT_SHA256))
            candidate.replace(destination)
            return
    raise RuntimeError("checkpoint-sources-unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    download_checkpoint(parser.parse_args().destination)


if __name__ == "__main__":
    main()
