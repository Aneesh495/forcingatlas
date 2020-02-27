"""Readers for the pinned text products.

Each reader returns a typed record whose semantics and unit are already
checked against the variable contract. A proxy file that also contains an
instrumental target returns that target as a separate record. The energy
inventory stays in zettajoules. A flux is a different function and a different
variable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from forcing_atlas.catalog import PinnedSource
from forcing_atlas.conversions import zj_per_year_to_w_m2
from forcing_atlas.errors import ContractError, UnitError
from forcing_atlas.identity import Semantics
from forcing_atlas.tables import NumericTable, parse_berkeley_sections, parse_delimited
from forcing_atlas.variables import get_contract


@dataclass(slots=True)
class AnnualRecord:
    dataset_id: str
    semantics: Semantics
    source_id: str
    variable: str
    unit: str
    baseline_id: str | None
    years: np.ndarray
    values: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.years = np.asarray(self.years)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.years.shape != self.values.shape:
            raise ContractError(f"{self.dataset_id} years and values differ in shape")
        if self.years.ndim != 1:
            raise ContractError(f"{self.dataset_id} is not a flat annual record")
        get_contract(self.variable)
        if self.variable in {"tas_anomaly", "sst_anomaly"} and not self.baseline_id:
            raise ContractError(f"{self.dataset_id} anomaly is missing a baseline")
        if np.isinf(self.values).any():
            raise ContractError(f"{self.dataset_id} contains infinite values")


def _require_source(source: PinnedSource, variable: str, semantics: Semantics) -> None:
    if source.variable != variable and source.dataset_id.startswith("igcc-ghg-concentrations"):
        # The concentration file's catalog variable is CO2. Other gases are read
        # from the same file under their own variable names.
        return
    if source.semantics is not semantics:
        raise ContractError(
            f"{source.dataset_id} is {source.semantics.value}, reader asked for {semantics.value}"
        )


def read_hadcrut_summary(text: str, source: PinnedSource) -> AnnualRecord:
    _require_source(source, "tas_anomaly", source.semantics)
    if source.baseline_id != "1961-1990":
        raise ContractError("HadCRUT summary reader expects the 1961-1990 baseline")
    table = parse_delimited(text)
    years = table.column("Time")
    anomaly = table.column("Anomaly (deg C)")
    lower = table.column("Lower confidence limit (2.5%)")
    upper = table.column("Upper confidence limit (97.5%)")
    if np.any(~np.isfinite(years)):
        raise ContractError("HadCRUT time column contains a missing year")
    # The published interval is a pair of quantiles, not a standard deviation.
    crossed = np.isfinite(lower) & np.isfinite(upper) & (upper < lower)
    if np.any(crossed):
        raise ContractError("HadCRUT upper confidence limit falls below the lower limit")
    return AnnualRecord(
        source.dataset_id,
        source.semantics,
        source.source_id,
        "tas_anomaly",
        "degC_anomaly",
        "1961-1990",
        years.astype(np.int64),
        anomaly,
        {"lower_2_5": lower, "upper_97_5": upper},
        notes=("limits are the 2.5 and 97.5 percent levels of the HadCRUT summary, not one standard deviation",),
    )


def read_hadcrut_ensemble(text: str, source: PinnedSource) -> AnnualRecord:
    if source.dataset_id != "hadcrut5-analysis-global-ensemble-annual":
        raise ContractError("ensemble reader is bound to the pinned HadCRUT global ensemble file")
    table = parse_delimited(text)
    years = table.column("Time").astype(np.int64)
    coverage = table.column("Fraction of area represented")
    coverage_sigma = table.column("Coverage uncertainty (1 sigma)")
    member_names = [name for name in table.columns if name.startswith("Realization ")]
    if len(member_names) < 2:
        raise ContractError("HadCRUT ensemble file does not contain realizations")
    members = np.column_stack([table.column(name) for name in member_names])
    # The ensemble mean is a statistic of these members. It is not a second observation.
    with np.errstate(invalid="ignore"):
        member_mean = np.nanmean(members, axis=1)
    return AnnualRecord(
        source.dataset_id,
        Semantics.OBSERVATIONAL_ANALYSIS,
        "HadCRUT5",
        "tas_anomaly",
        "degC_anomaly",
        "1961-1990",
        years,
        member_mean,
        {
            "members": members,
            "coverage_fraction": coverage,
            "coverage_uncertainty_1sigma": coverage_sigma,
        },
        notes=(
            "values are the mean of the uncertainty ensemble, stored so a caller can see it is not an independent product",
            "coverage uncertainty is one standard deviation and is not the summary 2.5 to 97.5 percent interval",
            f"{members.shape[1]} realizations belong to one analysis",
        ),
    )


_AIR_LABEL = "air temperature above sea ice"
_WATER_LABEL = "water temperature below sea ice"
_ABS_RE = re.compile(
    r"Using (?P<which>air temperature above sea ice|water temperature below sea ice):\s*"
    r"(?P<value>[-+0-9.]+)\s*\+/-\s*(?P<unc>[-+0-9.]+)"
)


def read_berkeley_land_ocean(text: str, source: PinnedSource) -> dict[str, AnnualRecord]:
    """Return the two sea-ice treatments, keyed by `air_above_ice` and `water_below_ice`.

    Monthly rows are kept in `extras['monthly']`. The annual column in the file
    is a centered 12-month mean reported on a month, so it is not used as a
    calendar-year mean. Calendar-year means are computed here from the monthly
    anomalies with equal month weights only after checking that all 12 months
    exist. Berkeley's own annual column is retained under a different key.
    """

    if source.source_id != "BerkeleyEarth":
        raise ContractError("Berkeley reader received a different source")
    comments, sections = parse_berkeley_sections(text)
    absolutes: dict[str, tuple[float, float]] = {}
    for comment in comments:
        match = _ABS_RE.search(comment.replace("%", " "))
        if match:
            absolutes[match.group("which")] = (float(match.group("value")), float(match.group("unc")))
    if _AIR_LABEL not in absolutes or _WATER_LABEL not in absolutes:
        raise ContractError("Berkeley header is missing one of the two absolute climatology lines")
    key_by_phrase = {
        "Sea Ice Temperature Inferred from Air Temperatures": "air_above_ice",
        "Sea Ice Temperature Inferred from Water Temperatures": "water_below_ice",
    }
    # The published headings differ slightly. Match on the distinctive words.
    records: dict[str, AnnualRecord] = {}
    for heading, table in sections:
        if "Air Temperatures" in heading:
            key = "air_above_ice"
            phrase = _AIR_LABEL
        elif "Water Temperatures" in heading or "sea-surface" in heading.lower():
            key = "water_below_ice"
            phrase = _WATER_LABEL
        else:
            raise ContractError(f"unrecognized Berkeley section {heading}")
        if key in records:
            raise ContractError(f"duplicate Berkeley section {key}")
        annual_years, annual_values, counts = _berkeley_calendar_years(table)
        absolute_value, absolute_unc = absolutes[phrase]
        records[key] = AnnualRecord(
            f"{source.dataset_id}-{key}",
            Semantics.OBSERVATIONAL_ANALYSIS,
            "BerkeleyEarth",
            "tas_anomaly",
            "degC_anomaly",
            "1951-1980",
            annual_years,
            annual_values,
            {
                "months_in_year": counts,
                "monthly_year": table.column("year"),
                "monthly_month": table.column("month"),
                "monthly_anomaly": table.column("monthly_anomaly"),
                "monthly_uncertainty_95": table.column("monthly_uncertainty_95"),
                "file_annual_anomaly": table.column("annual_anomaly"),
                "absolute_climatology_c": np.array([absolute_value]),
                "absolute_climatology_uncertainty_c": np.array([absolute_unc]),
            },
            notes=(
                "baseline is January 1951 to December 1980",
                "monthly uncertainty is the file's 95 percent interval",
                "absolute climatology is a global scalar for that baseline, not a spatial field",
                "the ocean portion uses HadSST, so this is not independent of HadCRUT over the ocean",
                f"section key {key_by_phrase.get(heading, key)}",
            ),
        )
    if set(records) != {"air_above_ice", "water_below_ice"}:
        raise ContractError(f"Berkeley sections resolved to {sorted(records)}")
    return records


def _berkeley_calendar_years(table: NumericTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = table.column("year").astype(np.int64)
    months = table.column("month").astype(np.int64)
    values = table.column("monthly_anomaly")
    unique = np.unique(years)
    out_years = []
    out_values = []
    out_counts = []
    for year in unique.tolist():
        mask = years == year
        present = months[mask]
        samples = values[mask]
        if len(set(present.tolist())) != present.size:
            raise ContractError(f"Berkeley year {year} repeats a month")
        finite = present[np.isfinite(samples)]
        # A calendar-year mean is emitted only when all 12 months are finite.
        # Otherwise the year is present with a NaN so the gap stays visible.
        out_years.append(int(year))
        out_counts.append(int(np.isfinite(samples).sum()))
        if finite.size == 12 and set(finite.tolist()) == set(range(1, 13)):
            out_values.append(float(np.mean(samples)))
        else:
            out_values.append(float("nan"))
    return (
        np.asarray(out_years, dtype=np.int64),
        np.asarray(out_values, dtype=np.float64),
        np.asarray(out_counts, dtype=np.int64),
    )


def read_pages2k_summary(text: str, source: PinnedSource) -> dict[str, AnnualRecord]:
    """Split the reconstruction from the instrumental target stored beside it."""

    if source.semantics is not Semantics.PROXY_RECONSTRUCTION:
        raise ContractError("PAGES2k summary is a proxy reconstruction")
    table = parse_delimited(text, delimiter="\t", comment_prefixes=("#",))
    year_name = table.columns[0]
    years = table.column(year_name).astype(np.int64)

    def raw_column(fragment: str) -> np.ndarray:
        matches = [
            name
            for name in table.columns
            if fragment in name.lower() and "filtered" not in name.lower()
        ]
        if len(matches) != 1:
            raise ContractError(f"PAGES2k column matching {fragment!r} found {matches}")
        return table.column(matches[0])

    instrumental = raw_column("instrumental target")
    median = raw_column("ensemble median")
    low = raw_column("2.5")
    high = raw_column("97.5")
    proxy = AnnualRecord(
        source.dataset_id,
        Semantics.PROXY_RECONSTRUCTION,
        "PAGES2k",
        "tas_anomaly",
        "degC_anomaly",
        "1961-1990",
        years,
        median,
        {"p2_5": low, "p97_5": high},
        notes=(
            "proxy reconstruction ensemble median, not a thermometer",
            "reference is 1961-1990 and the year is an April-March window in the source note",
            "the 2.5 and 97.5 percent range describes the reconstruction ensemble",
        ),
    )
    instrument = AnnualRecord(
        source.dataset_id + "-instrumental-target",
        Semantics.OBSERVATIONAL_ANALYSIS,
        "CowtanWay",
        "tas_anomaly",
        "degC_anomaly",
        "1961-1990",
        years,
        instrumental,
        notes=("Cowtan and Way instrumental target embedded in the PAGES2k file. It is not a proxy value.",),
    )
    return {"proxy": proxy, "instrumental_target": instrument}


def read_erf_components(text: str, source: PinnedSource) -> AnnualRecord:
    if source.unit != "W/m^2":
        raise UnitError("ERF components must be W/m^2")
    table = parse_delimited(text)
    years = table.column("time")
    if np.any(~np.isfinite(years)):
        raise ContractError("ERF year column has a missing value")
    component_names = [name for name in table.columns if name != "time"]
    components = {name: table.column(name) for name in component_names}
    total = np.zeros(years.shape, dtype=np.float64)
    for name, values in components.items():
        if np.any(~np.isfinite(values)):
            raise ContractError(f"ERF component {name} contains a missing value. Refusing to treat it as zero.")
        total = total + values
    groups = _group_erf(components)
    return AnnualRecord(
        source.dataset_id,
        Semantics.FORCING_ASSESSMENT,
        source.source_id,
        "erf_total",
        "W/m^2",
        "preindustrial_1750",
        years.astype(np.int64),
        total,
        {**groups, "components": np.column_stack([components[name] for name in component_names])},
        notes=(
            "total is the sum of the published component columns",
            "positive forcing warms",
            "this sum is not a license to add a CO2 concentration on top",
        ),
    )


def _group_erf(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def grab(*names: str) -> np.ndarray:
        missing = [name for name in names if name not in components]
        if missing:
            raise ContractError(f"ERF file is missing {missing}")
        stacked = components[names[0]].copy()
        for name in names[1:]:
            stacked = stacked + components[name]
        return stacked

    aerosol = grab("aerosol-radiation_interactions", "aerosol-cloud_interactions")
    co2 = grab("CO2")
    solar = grab("solar")
    volcanic = grab("volcanic")
    known = {"aerosol-radiation_interactions", "aerosol-cloud_interactions", "CO2", "solar", "volcanic"}
    other = np.zeros_like(co2)
    for name, values in components.items():
        if name not in known:
            other = other + values
    return {
        "erf_co2": co2,
        "erf_aerosol": aerosol,
        "erf_solar": solar,
        "erf_volcanic": volcanic,
        "erf_other": other,
    }


def read_erf_aggregates(text: str, source: PinnedSource) -> AnnualRecord:
    table = parse_delimited(text)
    time = table.column("time")
    # Mid-year stamps. The calendar year is the floor of the stamp.
    if np.any(~np.isfinite(time)):
        raise ContractError("aggregate ERF time is missing")
    years = np.floor(time).astype(np.int64)
    total = table.column("total")
    return AnnualRecord(
        source.dataset_id,
        Semantics.FORCING_ASSESSMENT,
        source.source_id,
        "erf_total",
        "W/m^2",
        "preindustrial_1750",
        years,
        total,
        {
            "midpoint": time,
            "erf_co2": table.column("CO2"),
            "erf_aerosol": table.column("aerosol"),
            "erf_solar": table.column("solar"),
            "erf_volcanic": table.column("volcanic"),
        },
        notes=("time stamps are mid-year. The stored year is the floor, so 1750.5 belongs to 1750.",),
    )


def compare_component_and_aggregate_totals(
    components: AnnualRecord,
    aggregates: AnnualRecord,
    *,
    tolerance_w_m2: float = 1.0e-6,
) -> dict[str, float]:
    """Compare the component sum with the aggregate total on shared years.

    A failure here means the two IGCC files are not being read as the same
    partition. It does not license averaging them.
    """

    left = {int(year): float(value) for year, value in zip(components.years, components.values, strict=True)}
    shared_left = []
    shared_right = []
    for year, value in zip(aggregates.years, aggregates.values, strict=True):
        if int(year) in left:
            shared_left.append(left[int(year)])
            shared_right.append(float(value))
    if not shared_left:
        raise ContractError("component and aggregate ERF files share no years")
    delta = np.asarray(shared_left) - np.asarray(shared_right)
    max_abs = float(np.max(np.abs(delta)))
    return {
        "shared_years": float(len(shared_left)),
        "max_abs_diff_w_m2": max_abs,
        "within_tolerance": float(max_abs <= tolerance_w_m2),
    }


def read_ghg_concentrations(text: str, source: PinnedSource) -> dict[str, AnnualRecord]:
    table = parse_delimited(text)
    years = table.column("YYYY").astype(np.int64)
    records = {
        "co2": AnnualRecord(
            source.dataset_id + "-co2",
            Semantics.FORCING_ASSESSMENT,
            source.source_id,
            "co2_concentration",
            "ppm_CO2",
            None,
            years,
            table.column("CO2"),
            notes=("CO2 dry-air mole fraction in ppm. Not a forcing.",),
        ),
        "ch4": AnnualRecord(
            source.dataset_id + "-ch4",
            Semantics.FORCING_ASSESSMENT,
            source.source_id,
            "ch4_concentration",
            "ppb_CH4",
            None,
            years,
            table.column("CH4"),
            notes=("CH4 in ppb.",),
        ),
        "n2o": AnnualRecord(
            source.dataset_id + "-n2o",
            Semantics.FORCING_ASSESSMENT,
            source.source_id,
            "n2o_concentration",
            "ppb_N2O",
            None,
            years,
            table.column("N2O"),
            notes=("N2O in ppb.",),
        ),
    }
    for record in records.values():
        finite = record.values[np.isfinite(record.values)]
        if finite.size and np.any(finite <= 0.0):
            raise ContractError(f"{record.dataset_id} has a nonpositive concentration")
    return records


def read_gwp_emissions(text: str, source: PinnedSource) -> AnnualRecord:
    if source.variable != "co2e_gwp100_emissions":
        raise ContractError("GWP reader received a non-GWP source")
    table = parse_delimited(text)
    time = table.column("time")
    years = np.floor(time).astype(np.int64)
    ffi = table.column("CO2-FFI")
    return AnnualRecord(
        source.dataset_id,
        Semantics.EMISSION_INVENTORY,
        source.source_id,
        "co2e_gwp100_emissions",
        "GtCO2e/yr",
        None,
        years,
        ffi,
        {
            "co2_lulucf": table.column("CO2-LULUCF"),
            "ch4": table.column("CH4"),
            "n2o": table.column("N2O"),
            "f_gases": table.column("F-gases"),
            "lower_bound": table.column("timebound_lower"),
            "upper_bound": table.column("timebound_upper"),
        },
        notes=(
            "values are the CO2-FFI column interpreted as GtCO2e per year under GWP100",
            "this record is rejected by the carbon-cycle mass balance",
            "missing early years stay missing",
        ),
    )


def read_energy_inventory(text: str, source: PinnedSource) -> AnnualRecord:
    if source.variable != "heat_content" or source.unit != "ZJ":
        raise ContractError("energy inventory source must be heat content in ZJ")
    table = parse_delimited(text)
    lower = table.column("timebound_lower")
    upper = table.column("timebound_upper")
    if np.any(upper <= lower):
        raise ContractError("energy inventory bounds are not increasing")
    total = table.column("Total")
    ocean = table.column("ocean_full-depth")
    return AnnualRecord(
        source.dataset_id,
        Semantics.ENERGY_INVENTORY,
        source.source_id,
        "heat_content",
        "ZJ",
        None,
        lower.astype(np.int64),
        total,
        {
            "bound_upper": upper,
            "ocean_full_depth_zj": ocean,
            "ocean_0_700_error_zj": table.column("ocean_0-700m_error"),
            "atmosphere_zj": table.column("atmosphere"),
            "land_zj": table.column("land"),
        },
        notes=(
            "filename says imbalance; unit is ZJ, so the variable is heat content",
            "error columns are one standard deviation",
            "year is the lower bound of the file interval",
        ),
    )


def energy_uptake_from_inventory(record: AnnualRecord) -> AnnualRecord:
    """First difference of the inventory, converted to W m-2.

    The first sample has no predecessor and stays missing. The result is an
    energy uptake rate, not a copy of the stock.
    """

    if record.variable != "heat_content" or record.unit != "ZJ":
        raise ContractError("uptake is derived from a ZJ heat-content record")
    years = record.years.astype(np.int64)
    stock = record.values
    upper = record.extras["bound_upper"]
    rate = np.full(stock.shape, np.nan, dtype=np.float64)
    for index in range(1, stock.size):
        if not np.isfinite(stock[index]) or not np.isfinite(stock[index - 1]):
            continue
        delta_years = float(upper[index] - upper[index - 1])
        if delta_years <= 0.0:
            raise ContractError("energy inventory upper bounds do not advance")
        zj_per_year = (stock[index] - stock[index - 1]) / delta_years
        rate[index] = float(zj_per_year_to_w_m2(np.array([zj_per_year]))[0])
    return AnnualRecord(
        record.dataset_id + "-uptake",
        Semantics.ENERGY_INVENTORY,
        record.source_id,
        "energy_uptake_rate",
        "W/m^2",
        None,
        years,
        rate,
        notes=("derived by differencing the ZJ inventory. The stock itself was not relabeled as a flux.",),
    )
