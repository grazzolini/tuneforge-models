from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from tuneforge_models import checkpoint_download as downloader
from tuneforge_models.beat_this_spec import CHECKPOINT_URL

PAYLOAD = b"synthetic checkpoint bytes"


@pytest.mark.parametrize(
    ("codes", "payload", "error", "existing"),
    [
        ([0], PAYLOAD, None, False),
        ([22, 0], PAYLOAD, None, False),
        ([28, 0], PAYLOAD, None, True),
        ([-1, 0], PAYLOAD, None, False),
        ([22, 7], PAYLOAD, "checkpoint-sources-unavailable", False),
        ([22, 7], PAYLOAD, "checkpoint-sources-unavailable", True),
        ([0], b"x" * len(PAYLOAD), "artifact-integrity-failed", False),
        ([0], PAYLOAD[:-1], "artifact-integrity-failed", True),
        ([22, 0], b"x" * len(PAYLOAD), "artifact-integrity-failed", False),
        ([23], PAYLOAD, "local-error", True),
    ],
)
def test_download_source_selection_and_atomic_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codes: list[int],
    payload: bytes,
    error: str | None,
    existing: bool,
) -> None:
    monkeypatch.setattr(downloader, "CHECKPOINT_SIZE", len(PAYLOAD))
    monkeypatch.setattr(downloader, "CHECKPOINT_SHA256", hashlib.sha256(PAYLOAD).hexdigest())
    destination = tmp_path / "source/checkpoint.ckpt"
    if existing:
        destination.parent.mkdir()
        destination.write_bytes(b"previous verified bytes")
    calls: list[str] = []

    def fake_curl(
        command: list[str], *, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[0] == "curl"
        assert not check
        assert timeout == 180
        for flag, value in (
            ("--connect-timeout", "15"), ("--max-time", "60"),
            ("--retry", "3"), ("--retry-max-time", "90"),
            ("--proto", "=https"), ("--proto-redir", "=https"),
        ):
            assert command[command.index(flag) + 1] == value
        assert "--fail" in command
        assert "--location" in command
        candidate = Path(command[command.index("--output") + 1])
        assert candidate.parent.parent == destination.parent
        assert not candidate.exists()
        assert destination.read_bytes() == b"previous verified bytes" if existing else (
            not destination.exists()
        )
        code = codes[len(calls)]
        calls.append(command[-1])
        if code == -1:
            raise subprocess.TimeoutExpired(command, timeout)
        candidate.write_bytes(payload if code == 0 else b"partial transfer")
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(subprocess, "run", fake_curl)
    if error == "local-error":
        with pytest.raises(subprocess.CalledProcessError) as failure:
            downloader.download_checkpoint(destination)
        assert failure.value.returncode == 23
    elif error:
        with pytest.raises((ValueError, RuntimeError), match=f"^{error}$"):
            downloader.download_checkpoint(destination)
    else:
        downloader.download_checkpoint(destination)

    assert calls == [CHECKPOINT_URL, downloader.MIRROR_URL][:len(codes)]
    if error:
        assert destination.read_bytes() == b"previous verified bytes" if existing else (
            not destination.exists()
        )
    else:
        assert destination.read_bytes() == PAYLOAD
    assert list(destination.parent.iterdir()) == ([destination] if destination.exists() else [])
