"""Streaming downloads with a byte cap, resume, and checksums.

A transfer writes a partial file beside the destination and renames it only
after the checksum, when one was requested, matches. A cancellation leaves the
partial file in place so a later call can resume. The acquirer does not delete
a completed source file to free space.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from forcing_atlas.budget import assert_room
from forcing_atlas.constants import DEFAULT_CHUNK_BYTES
from forcing_atlas.errors import AcquisitionError, BudgetError, CancelledError


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError("acquisition cancelled")


@dataclass(frozen=True, slots=True)
class DownloadResult:
    path: Path
    byte_count: int
    sha256: str
    resumed_from: int
    url: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(DEFAULT_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(DEFAULT_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stream_download(
    url: str,
    dest: Path,
    *,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout_s: float = 60.0,
    cancel: CancellationToken | None = None,
    headers: dict[str, str] | None = None,
) -> DownloadResult:
    """Download `url` to `dest` without holding the body in memory.

    `max_bytes` is the stage cap for this file. The project cap is checked as
    well, using `max_bytes` as the worst-case addition when the destination
    does not yet exist.
    """

    if max_bytes <= 0:
        raise BudgetError("max_bytes must be positive")
    if chunk_bytes <= 0:
        raise AcquisitionError("chunk_bytes must be positive")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    resumed = partial.stat().st_size if partial.exists() else 0
    if dest.exists() and not partial.exists():
        # A completed file is reused only when the requested checksum matches.
        if expected_sha256 is None and expected_md5 is None:
            return DownloadResult(dest, dest.stat().st_size, _sha256_file(dest), 0, url)
        sha = _sha256_file(dest)
        md5 = _md5_file(dest) if expected_md5 else ""
        sha_ok = expected_sha256 is None or sha == expected_sha256
        md5_ok = expected_md5 is None or md5 == expected_md5
        if sha_ok and md5_ok:
            return DownloadResult(dest, dest.stat().st_size, sha, 0, url)
        raise AcquisitionError(f"existing file {dest} does not match the pinned checksum")

    projected = max(0, max_bytes - resumed)
    assert_room(projected)

    request_headers = {"User-Agent": "forcing-atlas/0.1"}
    if headers:
        request_headers.update(headers)
    if resumed:
        request_headers["Range"] = f"bytes={resumed}-"
    request = urllib.request.Request(url, headers=request_headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.URLError as exc:
        raise AcquisitionError(f"download failed for {url}: {exc}") from exc

    status = getattr(response, "status", 200)
    if resumed and status == 200:
        # The server ignored the range. Restart so the partial is not concatenated twice.
        response.close()
        partial.unlink()
        return stream_download(
            url,
            dest,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
            expected_md5=expected_md5,
            chunk_bytes=chunk_bytes,
            timeout_s=timeout_s,
            cancel=cancel,
            headers=headers,
        )
    if status not in (200, 206):
        response.close()
        raise AcquisitionError(f"download of {url} returned HTTP {status}")

    content_length = response.headers.get("Content-Length")
    incoming = int(content_length) if content_length and content_length.isdigit() else None
    total_expected = (resumed + incoming) if (incoming is not None and status == 206) else incoming
    if total_expected is not None and total_expected > max_bytes:
        response.close()
        raise BudgetError(f"{url} announces {total_expected} bytes, above the stage cap {max_bytes}")

    written = resumed
    try:
        with partial.open("ab" if resumed else "wb") as handle:
            while True:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                block = response.read(chunk_bytes)
                if not block:
                    break
                written += len(block)
                if written > max_bytes:
                    raise BudgetError(f"download exceeded {max_bytes} bytes at {written}")
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    except CancelledError:
        raise
    finally:
        response.close()

    sha = _sha256_file(partial)
    if expected_sha256 is not None and sha != expected_sha256:
        raise AcquisitionError(
            f"sha256 mismatch for {url}: got {sha}, expected {expected_sha256}. Partial file kept."
        )
    if expected_md5 is not None:
        md5 = _md5_file(partial)
        if md5 != expected_md5:
            raise AcquisitionError(
                f"md5 mismatch for {url}: got {md5}, expected {expected_md5}. Partial file kept."
            )
    os.replace(partial, dest)
    sidecar = dest.with_suffix(dest.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "url": url,
                "bytes": written,
                "sha256": sha,
                "md5": expected_md5,
                "resumed_from": resumed,
            },
            indent=2,
        )
    )
    return DownloadResult(dest, written, sha, resumed, url)


def read_range(url: str, start: int, end_inclusive: int, *, timeout_s: float = 60.0) -> bytes:
    """HTTP byte range. `end_inclusive` follows the HTTP Range header."""

    if end_inclusive < start:
        raise AcquisitionError("empty byte range")
    length = end_inclusive - start + 1
    if length > 8_000_000:
        raise BudgetError(f"refusing a range of {length} bytes in one request")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "forcing-atlas/0.1", "Range": f"bytes={start}-{end_inclusive}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            if status not in (200, 206):
                raise AcquisitionError(f"range request returned HTTP {status}")
            payload = response.read(length + 1)
    except urllib.error.URLError as exc:
        raise AcquisitionError(f"range request failed: {exc}") from exc
    if len(payload) > length:
        raise AcquisitionError("range response is longer than requested")
    return payload
