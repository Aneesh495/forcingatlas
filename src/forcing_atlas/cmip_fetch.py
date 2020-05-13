"""Pull named ClimateBench annual fields out of CMIP6.zip without storing the zip.

The public archive's matched experiments are ssp245 and piControl. Historical
and other ScenarioMIP pathways are not in that zip, so this fetch does not
pretend they were downloaded.
"""

from __future__ import annotations

import struct
from pathlib import Path

from forcing_atlas.acquire import read_span
from forcing_atlas.budget import assert_room
from forcing_atlas.catalog import CMIP6_TAS_MEMBERS, cmip6_member_name, sources_by_id
from forcing_atlas.errors import AcquisitionError, BudgetError
from forcing_atlas.zipmember import decode_member, parse_central_directory, parse_eocd, select_members


def fetch_tas_members(destination: Path, names: list[str] | None = None, *, max_member_bytes: int = 40_000_000) -> list[Path]:
    source = sources_by_id()["climatebench-cmip6-zip"]
    if source.byte_size is None:
        raise AcquisitionError("CMIP6 zip size is not pinned")
    size = source.byte_size
    tail_start = size - 262144
    tail = read_span(source.url, tail_start, size - 1)
    _entries, cd_size, cd_offset = parse_eocd(tail, tail_start)
    directory = read_span(source.url, cd_offset, cd_offset + cd_size - 1)
    wanted = names or [cmip6_member_name(entry) for entry in CMIP6_TAS_MEMBERS[:6]]
    members = select_members(parse_central_directory(directory), wanted)
    written: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    for member in members:
        if member.uncompressed_size > max_member_bytes:
            raise BudgetError(f"{member.name} uncompressed size {member.uncompressed_size} exceeds {max_member_bytes}")
        assert_room(member.uncompressed_size)
        header = read_span(source.url, member.local_header_offset, member.local_header_offset + 29)
        if header[:4] != b"PK\x03\x04":
            raise AcquisitionError(f"{member.name} local header is missing")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        compressed = member.compressed_size
        end = member.local_header_offset + 30 + name_len + extra_len + compressed - 1
        blob = read_span(source.url, member.local_header_offset, end)
        payload = decode_member(blob, member)
        target = destination / Path(member.name).name
        target.write_bytes(payload)
        written.append(target)
    return written


def describe_calendars(paths: list[Path]) -> list[dict[str, object]]:
    """Read the calendar attribute and time span. The filename years are not trusted."""

    import xarray as xr

    rows: list[dict[str, object]] = []
    for path in paths:
        dataset = xr.open_dataset(path)
        try:
            time = dataset["time"]
            calendar = str(time.attrs.get("calendar") or dataset["time"].encoding.get("calendar") or "missing")
            units = str(time.attrs.get("units") or time.encoding.get("units") or "")
            rows.append(
                {
                    "file": path.name,
                    "calendar": calendar,
                    "time_units": units,
                    "variable": "tas" if "tas" in dataset else ",".join(dataset.data_vars),
                    "sizes": {key: int(value) for key, value in dataset.sizes.items()},
                }
            )
        finally:
            dataset.close()
    return rows
