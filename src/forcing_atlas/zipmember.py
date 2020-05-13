"""Read selected members out of a remote zip without storing the archive.

The central directory is fetched from the tail of the file. Each requested
member is fetched by its local header offset and inflated if the method is
store or deflate. The CRC-32 in the central directory is checked against the
bytes we actually decoded. This is the bounded path for the ClimateBench
CMIP6.zip, which is larger than the slice we need.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass

from forcing_atlas.errors import AcquisitionError


@dataclass(frozen=True, slots=True)
class ZipMember:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def _find_eocd(blob: bytes) -> int:
    signature = b"PK\x05\x06"
    index = blob.rfind(signature)
    if index < 0 or index + 22 > len(blob):
        raise AcquisitionError("zip end-of-central-directory signature not in the fetched tail")
    return index


def parse_eocd(tail: bytes, tail_start: int) -> tuple[int, int, int]:
    """Return (entry_count, central_directory_size, central_directory_offset)."""

    index = _find_eocd(tail)
    _disk, _disk2, _entries, entries, size, offset, comment_len = struct.unpack(
        "<HHHHIIH", tail[index + 4 : index + 22]
    )
    if comment_len:
        # A comment does not move the central directory. It only has to fit in the tail.
        if index + 22 + comment_len > len(tail):
            raise AcquisitionError("zip comment extends past the fetched tail")
    if entries == 0xFFFF or offset == 0xFFFFFFFF or size == 0xFFFFFFFF:
        raise AcquisitionError("zip64 archives are not supported by this reader")
    if tail_start > offset:
        raise AcquisitionError("fetched tail does not contain the central directory offset")
    return entries, size, offset


def parse_central_directory(blob: bytes) -> list[ZipMember]:
    members: list[ZipMember] = []
    pos = 0
    while pos + 46 <= len(blob):
        if blob[pos : pos + 4] != b"PK\x01\x02":
            break
        (
            _ver_made,
            _ver_need,
            _flag,
            method,
            _time,
            _date,
            crc,
            compressed,
            uncompressed,
            name_len,
            extra_len,
            comment_len,
            _disk,
            _iattr,
            _eattr,
            local_offset,
        ) = struct.unpack("<HHHHHHIIIHHHHHII", blob[pos + 4 : pos + 46])
        name = blob[pos + 46 : pos + 46 + name_len].decode("utf-8")
        members.append(
            ZipMember(name, method, crc, compressed, uncompressed, local_offset)
        )
        pos += 46 + name_len + extra_len + comment_len
    if not members:
        raise AcquisitionError("central directory contained no members")
    return members


def decode_member(local_blob: bytes, member: ZipMember) -> bytes:
    """Decode a local-header blob that begins at the member's local header."""

    if local_blob[:4] != b"PK\x03\x04":
        raise AcquisitionError(f"{member.name} is missing a local zip header")
    method, _time, _date, crc, compressed, uncompressed, name_len, extra_len = struct.unpack(
        "<HHHIIIHH", local_blob[8:30]
    )
    if method != member.method:
        raise AcquisitionError(f"{member.name} local method {method} disagrees with the directory")
    # Bit 3 of the general-purpose flag puts the sizes in a data descriptor.
    # The local header then carries zeros, and the central directory is the size source.
    if compressed == 0:
        compressed = member.compressed_size
    if uncompressed == 0:
        uncompressed = member.uncompressed_size
    data_start = 30 + name_len + extra_len
    payload = local_blob[data_start : data_start + compressed]
    if len(payload) != compressed:
        raise AcquisitionError(
            f"{member.name} compressed payload is {len(payload)} bytes, directory says {compressed}"
        )
    if method == 0:
        decoded = payload
    elif method == 8:
        decoded = zlib.decompress(payload, -15)
    else:
        raise AcquisitionError(f"{member.name} uses zip method {method}, which this reader does not inflate")
    if len(decoded) != uncompressed and uncompressed != 0:
        raise AcquisitionError(
            f"{member.name} inflated to {len(decoded)} bytes, directory says {uncompressed}"
        )
    actual_crc = binascii.crc32(decoded) & 0xFFFFFFFF
    expected = member.crc32 & 0xFFFFFFFF
    if actual_crc != expected:
        raise AcquisitionError(
            f"{member.name} crc32 {actual_crc:08x} does not match directory {expected:08x}"
        )
    if crc and (crc & 0xFFFFFFFF) != expected:
        raise AcquisitionError(f"{member.name} local crc disagrees with the central directory")
    return decoded


def member_map(members: list[ZipMember]) -> dict[str, ZipMember]:
    return {member.name: member for member in members}


def select_members(members: list[ZipMember], names: list[str]) -> list[ZipMember]:
    lookup = member_map(members)
    chosen: list[ZipMember] = []
    for name in names:
        key = name
        if key not in lookup:
            # ClimateBench stores paths as CMIP6/<file>.
            alt = f"CMIP6/{name}"
            if alt in lookup:
                key = alt
            else:
                matches = [candidate for candidate in lookup if candidate.endswith("/" + name) or candidate.endswith(name)]
                if len(matches) != 1:
                    raise AcquisitionError(f"zip member {name} not found")
                key = matches[0]
        chosen.append(lookup[key])
    return chosen
