"""Parsers, cubes, zip members, and a resumable local download."""

from __future__ import annotations

import math
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from forcing_atlas.catalog import PINNED, sources_by_id
from forcing_atlas.constants import EARTH_RADIUS_M, JULIAN_YEAR_SECONDS
from forcing_atlas.cube import read_cube, write_cube
from forcing_atlas.errors import AcquisitionError, ContractError
from forcing_atlas.identity import Semantics
from forcing_atlas.readers import (
    compare_component_and_aggregate_totals,
    energy_uptake_from_inventory,
    read_berkeley_land_ocean,
    read_energy_inventory,
    read_erf_aggregates,
    read_erf_components,
    read_ghg_concentrations,
    read_gwp_emissions,
    read_hadcrut_ensemble,
    read_hadcrut_summary,
    read_pages2k_summary,
)
from forcing_atlas.tables import parse_delimited
from forcing_atlas.zipmember import decode_member, parse_central_directory, parse_eocd, select_members


HADCRUT = """Time,Anomaly (deg C),Lower confidence limit (2.5%),Upper confidence limit (97.5%)
1850,-0.4,-0.6,-0.2
1851,NA,-0.5,-0.1
"""

ENSEMBLE = """Time,Fraction of area represented,Coverage uncertainty (1 sigma),Realization 1,Realization 2
1850,0.5,0.05,-0.4,-0.6
1851,0.5,0.05,-0.2,-0.4
"""

BERKELEY = """% Estimated Jan 1951-Dec 1980 global mean temperature (C)
%   Using air temperature above sea ice:   14.100 +/- 0.020
%   Using water temperature below sea ice: 14.700 +/- 0.020
% Global Average Temperature Anomaly with Sea Ice Temperature Inferred from Air Temperatures
% Year, Month,  Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.
  1850     1    -0.5  0.4       NaN    NaN       NaN    NaN       NaN    NaN       NaN    NaN
  1850     2    -0.1  0.4       NaN    NaN       NaN    NaN       NaN    NaN       NaN    NaN
% Global Average Temperature Anomaly with Sea Ice Temperature Inferred from Water Temperatures
% Year, Month,  Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.
  1850     1    -0.4  0.4       NaN    NaN       NaN    NaN       NaN    NaN       NaN    NaN
  1850     2    -0.2  0.4       NaN    NaN       NaN    NaN       NaN    NaN       NaN    NaN
"""

PAGES = """## reconstruction, not a thermometer
Year\tCowtan & Way instrumental target\tFull ensemble median\tFull ensemble 2.5th percentile\tFull ensemble 97.5th percentile\tCowtan & Way instrumental target 31-year filtered\t31-year filtered full ensemble median\t31-year filtered full ensemble 2.5th percentile\t31-year filtered full ensemble 97.5th percentile
1\tNA\t-0.30\t-0.70\t0.10\tNA\tNA\tNA\tNA
1850\t-0.40\t-0.20\t-0.50\t0.05\tNA\tNA\tNA\tNA
"""

ERF = """time,solar,volcanic,aerosol-radiation_interactions,aerosol-cloud_interactions,CO2,CH4
1750,0.10,0.20,0.00,0.00,0.00,0.05
1751,0.10,-1.00,-0.20,-0.30,0.01,0.05
"""

ERF_AGG = """time,CO2,aerosol,solar,volcanic,total
1750.5,0.00,0.00,0.10,0.20,0.35
1751.5,0.01,-0.50,0.10,-1.00,-1.34
"""

GHG = """YYYY,CO2,CH4,N2O
1750,278.3,729.2,270.1
1751,NA,730.0,270.1
"""

EMIS = """time,timebound_lower,timebound_upper,F-gases,N2O,CH4,CO2-LULUCF,CO2-FFI
1750.5,1750,1751,0.0,0.2,1.0,NA,NA
2022.5,2022,2023,1.8,3.0,9.0,4.3,36.3
"""

INVENTORY = """time,timebound_lower,timebound_upper,atmosphere,land,ocean_0-700m_error,ocean_full-depth,Total
1971.5,1971,1972,0.0,0.0,1.0,0.0,0.0
1972.5,1972,1973,0.0,0.0,1.0,10.0,16.0
"""


def test_missing_marker_is_not_zero() -> None:
    table = parse_delimited("year,value\n1850,NA\n1851,0\n")
    assert np.isnan(table.column("value")[0])
    assert table.column("value")[1] == 0.0
    with pytest.raises(ContractError):
        parse_delimited("year,value\n1850,1\n1851,2,3\n")


def test_hadcrut_summary_quantiles() -> None:
    source = sources_by_id()["hadcrut5-analysis-global-annual"]
    record = read_hadcrut_summary(HADCRUT, source)
    assert record.semantics is source.semantics
    assert record.baseline_id == "1961-1990"
    assert np.isnan(record.values[1])
    assert record.extras["lower_2_5"][0] == pytest.approx(-0.6)
    assert "one standard deviation" in record.notes[0]


def test_ensemble_members_are_not_a_second_product() -> None:
    source = sources_by_id()["hadcrut5-analysis-global-ensemble-annual"]
    record = read_hadcrut_ensemble(ENSEMBLE, source)
    assert record.extras["members"].shape == (2, 2)
    assert record.values[0] == pytest.approx(-0.5)
    assert "one analysis" in record.notes[-1]


def test_berkeley_keeps_two_sea_ice_treatments() -> None:
    source = sources_by_id()["berkeley-land-ocean-complete"]
    records = read_berkeley_land_ocean(BERKELEY, source)
    air = records["air_above_ice"]
    water = records["water_below_ice"]
    assert air.baseline_id == "1951-1980"
    assert np.isnan(air.values[0])  # only two months, so the calendar year is missing
    assert air.extras["months_in_year"][0] == 2
    assert air.extras["absolute_climatology_c"][0] == pytest.approx(14.1)
    assert water.extras["monthly_anomaly"][0] == pytest.approx(-0.4)
    assert air.values is not water.values


def test_pages_proxy_is_separate_from_the_instrumental_column() -> None:
    source = sources_by_id()["pages2k-2019-ensemble-summary"]
    split = read_pages2k_summary(PAGES, source)
    assert split["proxy"].semantics is Semantics.PROXY_RECONSTRUCTION
    assert split["instrumental_target"].semantics is Semantics.OBSERVATIONAL_ANALYSIS
    assert split["instrumental_target"].source_id == "CowtanWay"
    assert np.isnan(split["instrumental_target"].values[0])
    assert split["proxy"].values[0] == pytest.approx(-0.30)
    assert split["proxy"].values[1] == pytest.approx(-0.20)


def test_erf_groups_partition_the_total_and_match_the_aggregate_year() -> None:
    component_source = sources_by_id()["igcc-erf-components-1750-2023"]
    aggregate_source = sources_by_id()["igcc-erf-aggregates-1750-2023"]
    components = read_erf_components(ERF, component_source)
    aggregates = read_erf_aggregates(ERF_AGG, aggregate_source)
    rebuilt = (
        components.extras["erf_co2"]
        + components.extras["erf_aerosol"]
        + components.extras["erf_solar"]
        + components.extras["erf_volcanic"]
        + components.extras["erf_other"]
    )
    assert rebuilt == pytest.approx(components.values)
    assert components.values[1] == pytest.approx(0.10 - 1.00 - 0.20 - 0.30 + 0.01 + 0.05)
    report = compare_component_and_aggregate_totals(components, aggregates, tolerance_w_m2=1e-9)
    assert report["within_tolerance"] == 1.0
    assert aggregates.years.tolist() == [1750, 1751]


def test_concentrations_reject_nonpositive_and_keep_gaps() -> None:
    source = sources_by_id()["igcc-ghg-concentrations"]
    gases = read_ghg_concentrations(GHG, source)
    assert gases["co2"].unit == "ppm_CO2"
    assert np.isnan(gases["co2"].values[1])
    assert gases["ch4"].unit == "ppb_CH4"
    with pytest.raises(ContractError):
        read_ghg_concentrations("YYYY,CO2,CH4,N2O\n1750,0,1,1\n", source)


def test_gwp_emissions_leave_early_years_missing() -> None:
    source = sources_by_id()["igcc-ghg-emissions-gwp100"]
    record = read_gwp_emissions(EMIS, source)
    assert record.variable == "co2e_gwp100_emissions"
    assert np.isnan(record.values[0])
    assert record.values[1] == pytest.approx(36.3)
    assert "carbon-cycle" in record.notes[1]


def test_inventory_difference_uses_the_joule_identity() -> None:
    source = sources_by_id()["igcc-energy-inventory"]
    stock = read_energy_inventory(INVENTORY, source)
    assert stock.variable == "heat_content"
    assert stock.unit == "ZJ"
    rate = energy_uptake_from_inventory(stock)
    assert rate.variable == "energy_uptake_rate"
    assert np.isnan(rate.values[0])
    area = 4.0 * math.pi * EARTH_RADIUS_M ** 2
    expected = (16.0 * 1.0e21) / JULIAN_YEAR_SECONDS / area
    assert rate.values[1] == pytest.approx(expected, rel=1e-12)


def test_cube_round_trip_preserves_nan_and_detects_corruption(tmp_path: Path) -> None:
    values = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
    path = tmp_path / "cube"
    write_cube(path, values, {"semantics": "observation", "variable": "tas_anomaly"})
    loaded = read_cube(path)
    assert np.isnan(loaded.values[0, 1])
    assert loaded.values[1, 0] == pytest.approx(2.0)
    chunk = next((path).glob("chunk-*.f32"))
    blob = bytearray(chunk.read_bytes())
    blob[0] ^= 0xFF
    chunk.write_bytes(blob)
    with pytest.raises(ContractError):
        read_cube(path)


def test_zip_member_crc_and_selection(tmp_path: Path) -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    payload = b"annual-field-bytes"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("CMIP6/tas_example.nc", payload)
    data = buffer.getvalue()
    tail_start = max(0, len(data) - 4096)
    entries, size, offset = parse_eocd(data[tail_start:], tail_start)
    assert entries == 1
    members = parse_central_directory(data[offset : offset + size])
    chosen = select_members(members, ["tas_example.nc"])
    member = chosen[0]
    local = data[member.local_header_offset :]
    assert decode_member(local, member) == payload
    tampered = ZipTamper(member)
    with pytest.raises(AcquisitionError):
        decode_member(local, tampered)


class ZipTamper:
    def __init__(self, member) -> None:
        self.name = member.name
        self.method = member.method
        self.crc32 = (member.crc32 + 1) & 0xFFFFFFFF
        self.compressed_size = member.compressed_size
        self.uncompressed_size = member.uncompressed_size
        self.local_header_offset = member.local_header_offset


def test_download_resumes_and_checksums(tmp_path: Path) -> None:
    from forcing_atlas.acquire import stream_download
    import hashlib
    from http.server import BaseHTTPRequestHandler

    payload = b"0123456789abcdef" * 50

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            header = self.headers.get("Range")
            if header:
                start = int(header.split("=", 1)[1].split("-", 1)[0])
                body = payload[start:]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            else:
                body = payload
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/series.bin"
        dest = tmp_path / "out" / "series.bin"
        partial = dest.with_name(dest.name + ".partial")
        partial.parent.mkdir(parents=True)
        partial.write_bytes(payload[:20])
        digest = hashlib.sha256(payload).hexdigest()
        result = stream_download(url, dest, max_bytes=len(payload), expected_sha256=digest)
        assert result.resumed_from == 20
        assert dest.read_bytes() == payload
        assert not partial.exists()
    finally:
        server.shutdown()


def test_catalog_keeps_semantics_apart() -> None:
    by_semantics = {source.semantics for source in PINNED}
    assert Semantics.OBSERVATION in by_semantics
    assert Semantics.PROXY_RECONSTRUCTION in by_semantics
    assert Semantics.SOURCE_SIMULATION in by_semantics
    assert Semantics.ENERGY_INVENTORY in by_semantics
    inventory = sources_by_id()["igcc-energy-inventory"]
    assert inventory.unit == "ZJ"
    assert inventory.variable == "heat_content"
