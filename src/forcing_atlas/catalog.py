"""Pinned source catalog.

Checksums that are known before download are stored here. Checksums that are
computed after a transfer are written into the local manifest by the acquirer
and are not invented in this file. A missing checksum is not a passing check.
"""

from __future__ import annotations

from dataclasses import dataclass

from forcing_atlas.identity import MemberRole, Semantics, TimeLabel


@dataclass(frozen=True, slots=True)
class PinnedSource:
    dataset_id: str
    semantics: Semantics
    source_id: str
    title: str
    version: str
    variable: str
    unit: str
    license_id: str
    url: str
    citation: str
    file_name: str
    baseline_id: str | None = None
    calendar: str | None = None
    experiment: str | None = None
    member_id: str | None = None
    member_role: MemberRole = MemberRole.SINGLE
    grid_label: str | None = None
    time_label: TimeLabel = TimeLabel.YEAR_START
    byte_size: int | None = None
    checksum_md5: str | None = None
    license_note: str = ""
    unit_note: str = ""
    group: str = "core"


HADCRUT_ROOT = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0"
HADCRUT_VERSION = "HadCRUT.5.1.0.0"
HADCRUT_CITATION = (
    "Morice, C. P., et al. An updated assessment of near-surface temperature change from 1850. "
    "Journal of Geophysical Research: Atmospheres, 2021. HadCRUT.5.1.0.0, Met Office Hadley Centre."
)
BERKELEY_CITATION = (
    "Rohde, R. A. and Hausfather, Z. The Berkeley Earth Land/Ocean Temperature Record. "
    "Earth System Science Data, 2020. https://doi.org/10.5194/essd-12-3469-2020"
)
PAGES_CITATION = (
    "PAGES 2k Consortium. Consistent multidecadal variability in global temperature reconstructions "
    "and simulations over the Common Era. Nature Geoscience, 2019. "
    "https://doi.org/10.1038/s41561-019-0400-0"
)
IGCC_CITATION = (
    "Indicators of Global Climate Change, v2024.05.29b. "
    "https://doi.org/10.5281/zenodo.11388387"
)
CLIMATEBENCH_CITATION = (
    "Watson-Parris, D., et al. ClimateBench v1.0: A Benchmark for Data-Driven Climate Projections. "
    "Journal of Advances in Modeling Earth Systems, 2022. https://doi.org/10.1029/2021MS002954 "
    "Dataset https://doi.org/10.5281/zenodo.7064308"
)

IGCC_RAW = "https://raw.githubusercontent.com/ClimateIndicator/data/v2024.05.29b/data"


def _hadcrut_csv(region: str, kind: str) -> PinnedSource:
    folder = "analysis/diagnostics" if kind == "analysis" else "non-infilled/diagnostics"
    prefix = "analysis" if kind == "analysis" else "noninfilled"
    name = f"{HADCRUT_VERSION}.{prefix}.summary_series.{region}.annual.csv"
    return PinnedSource(
        dataset_id=f"hadcrut5-{prefix}-{region}-annual",
        semantics=Semantics.OBSERVATIONAL_ANALYSIS if kind == "analysis" else Semantics.OBSERVATION,
        source_id="HadCRUT5",
        title=f"HadCRUT5 {prefix} {region} annual anomaly",
        version=HADCRUT_VERSION,
        variable="tas_anomaly",
        unit="degC_anomaly",
        license_id="crown-copyright-met-office",
        url=f"{HADCRUT_ROOT}/{folder}/{name}",
        citation=HADCRUT_CITATION,
        file_name=name,
        baseline_id="1961-1990",
        calendar="proleptic_gregorian",
        time_label=TimeLabel.YEAR_START,
        license_note=(
            "Crown Copyright, Met Office Hadley Centre. The 5.1.0.0 download page states Crown Copyright. "
            "The file is acquired locally and is not redistributed in this repository."
        ),
        unit_note="Anomaly in deg C relative to 1961-1990. Summary limits are the 2.5 and 97.5 percent levels.",
        group="instrumental",
    )


PINNED: tuple[PinnedSource, ...] = (
    _hadcrut_csv("global", "analysis"),
    _hadcrut_csv("northern_hemisphere", "analysis"),
    _hadcrut_csv("southern_hemisphere", "analysis"),
    _hadcrut_csv("global", "noninfilled"),
    PinnedSource(
        dataset_id="hadcrut5-analysis-global-ensemble-annual",
        semantics=Semantics.OBSERVATIONAL_ANALYSIS,
        source_id="HadCRUT5",
        title="HadCRUT5 analysis global annual ensemble",
        version=HADCRUT_VERSION,
        variable="tas_anomaly",
        unit="degC_anomaly",
        license_id="crown-copyright-met-office",
        url=f"{HADCRUT_ROOT}/analysis/diagnostics/{HADCRUT_VERSION}.analysis.ensemble_series.global.annual.csv",
        citation=HADCRUT_CITATION,
        file_name=f"{HADCRUT_VERSION}.analysis.ensemble_series.global.annual.csv",
        baseline_id="1961-1990",
        calendar="proleptic_gregorian",
        member_role=MemberRole.UNCERTAINTY_ENSEMBLE,
        time_label=TimeLabel.YEAR_START,
        license_note="Crown Copyright. Uncertainty members are one analysis, not independent models.",
        unit_note=(
            "Realizations are the 200-member analysis ensemble. "
            "The coverage uncertainty column is one standard deviation, not the 2.5 to 97.5 percent summary interval."
        ),
        group="instrumental",
    ),
    PinnedSource(
        dataset_id="hadcrut5-analysis-ensemble-mean-grid",
        semantics=Semantics.OBSERVATIONAL_ANALYSIS,
        source_id="HadCRUT5",
        title="HadCRUT5 analysis ensemble-mean anomaly grid",
        version=HADCRUT_VERSION,
        variable="tas_anomaly",
        unit="degC_anomaly",
        license_id="crown-copyright-met-office",
        url=f"{HADCRUT_ROOT}/analysis/{HADCRUT_VERSION}.analysis.anomalies.ensemble_mean.nc",
        citation=HADCRUT_CITATION,
        file_name=f"{HADCRUT_VERSION}.analysis.anomalies.ensemble_mean.nc",
        baseline_id="1961-1990",
        calendar="proleptic_gregorian",
        grid_label="5x5",
        byte_size=32_829_594,
        time_label=TimeLabel.YEAR_MIDPOINT,
        license_note="Crown Copyright. The analysis grid infills unobserved cells. It is not the non-infilled field.",
        group="instrumental",
    ),
    PinnedSource(
        dataset_id="hadcrut5-noninfilled-ensemble-mean-grid",
        semantics=Semantics.OBSERVATION,
        source_id="HadCRUT5",
        title="HadCRUT5 non-infilled ensemble-mean anomaly grid",
        version=HADCRUT_VERSION,
        variable="tas_anomaly",
        unit="degC_anomaly",
        license_id="crown-copyright-met-office",
        url=f"{HADCRUT_ROOT}/non-infilled/{HADCRUT_VERSION}.noninfilled.anomalies.ensemble_mean.nc",
        citation=HADCRUT_CITATION,
        file_name=f"{HADCRUT_VERSION}.noninfilled.anomalies.ensemble_mean.nc",
        baseline_id="1961-1990",
        calendar="proleptic_gregorian",
        grid_label="5x5",
        byte_size=24_107_871,
        time_label=TimeLabel.YEAR_MIDPOINT,
        license_note="Crown Copyright. Empty cells are unobserved. They are not zeros.",
        group="instrumental",
    ),
    PinnedSource(
        dataset_id="berkeley-land-ocean-complete",
        semantics=Semantics.OBSERVATIONAL_ANALYSIS,
        source_id="BerkeleyEarth",
        title="Berkeley Earth land-ocean monthly summary",
        version="land-ocean-file-2025-01",
        variable="tas_anomaly",
        unit="degC_anomaly",
        license_id="cc-by-nc-4.0",
        url="https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt",
        citation=BERKELEY_CITATION,
        file_name="Land_and_Ocean_complete.txt",
        baseline_id="1951-1980",
        calendar="proleptic_gregorian",
        time_label=TimeLabel.YEAR_MIDPOINT,
        license_note="CC BY-NC 4.0. Non-commercial use. The ocean component reinterpolates HadSST.",
        unit_note="Anomalies versus Jan 1951-Dec 1980. The uncertainty column is the stated 95 percent interval, not one standard deviation.",
        group="instrumental",
    ),
    PinnedSource(
        dataset_id="pages2k-2019-ensemble-summary",
        semantics=Semantics.PROXY_RECONSTRUCTION,
        source_id="PAGES2k",
        title="PAGES2k 2019 full-ensemble median and 95 percent range",
        version="neukom2019-median-2021-10-21",
        variable="tas_anomaly",
        unit="degC_anomaly",
        license_id="noaa-paleoclimatology-citation",
        url="https://www.ncei.noaa.gov/pub/data/paleo/pages2k/neukom2019temp/recons/Full_ensemble_median_and_95pct_range.txt",
        citation=PAGES_CITATION,
        file_name="Full_ensemble_median_and_95pct_range.txt",
        baseline_id="1961-1990",
        time_label=TimeLabel.YEAR_START,
        license_note="NOAA Paleoclimatology distribution. Cite Neukom et al. 2019 and the study page.",
        unit_note=(
            "Degrees Celsius versus 1961-1990, April-March annual window. "
            "The embedded Cowtan and Way column is an instrumental target, not a proxy."
        ),
        group="proxy",
    ),
    PinnedSource(
        dataset_id="igcc-erf-components-1750-2023",
        semantics=Semantics.FORCING_ASSESSMENT,
        source_id="IGCC-2023",
        title="IGCC effective radiative forcing components, 1750-2023",
        version="v2024.05.29b",
        variable="erf_total",
        unit="W/m^2",
        license_id="cc-by-4.0",
        url=f"{IGCC_RAW}/effective_radiative_forcing/ERF_best_1750-2023.csv",
        citation=IGCC_CITATION,
        file_name="ERF_best_1750-2023.csv",
        baseline_id="preindustrial_1750",
        time_label=TimeLabel.YEAR_START,
        byte_size=185_951,
        checksum_md5="7a4531ec03a705896c4291759bbe0daa",
        unit_note="Annual-mean best-estimate ERF in W m-2. Positive warms. Volcanic values are negative during eruptions.",
        group="forcing",
    ),
    PinnedSource(
        dataset_id="igcc-erf-aggregates-1750-2023",
        semantics=Semantics.FORCING_ASSESSMENT,
        source_id="IGCC-2023",
        title="IGCC effective radiative forcing aggregates",
        version="v2024.05.29b",
        variable="erf_total",
        unit="W/m^2",
        license_id="cc-by-4.0",
        url=f"{IGCC_RAW}/effective_radiative_forcing/ERF_best_aggregates_1750-2023.csv",
        citation=IGCC_CITATION,
        file_name="ERF_best_aggregates_1750-2023.csv",
        baseline_id="preindustrial_1750",
        time_label=TimeLabel.YEAR_MIDPOINT,
        byte_size=95_619,
        unit_note="Mid-year stamps. Do not treat the year label as identical to the component file without using the year identity.",
        group="forcing",
    ),
    PinnedSource(
        dataset_id="igcc-erf-aggregates-p05",
        semantics=Semantics.FORCING_ASSESSMENT,
        source_id="IGCC-2023",
        title="IGCC ERF aggregate 5th percentile",
        version="v2024.05.29b",
        variable="erf_total",
        unit="W/m^2",
        license_id="cc-by-4.0",
        url=f"{IGCC_RAW}/effective_radiative_forcing/ERF_p05_aggregates_1750-2023.csv",
        citation=IGCC_CITATION,
        file_name="ERF_p05_aggregates_1750-2023.csv",
        baseline_id="preindustrial_1750",
        time_label=TimeLabel.YEAR_MIDPOINT,
        byte_size=101_809,
        group="forcing",
    ),
    PinnedSource(
        dataset_id="igcc-erf-aggregates-p95",
        semantics=Semantics.FORCING_ASSESSMENT,
        source_id="IGCC-2023",
        title="IGCC ERF aggregate 95th percentile",
        version="v2024.05.29b",
        variable="erf_total",
        unit="W/m^2",
        license_id="cc-by-4.0",
        url=f"{IGCC_RAW}/effective_radiative_forcing/ERF_p95_aggregates_1750-2023.csv",
        citation=IGCC_CITATION,
        file_name="ERF_p95_aggregates_1750-2023.csv",
        baseline_id="preindustrial_1750",
        time_label=TimeLabel.YEAR_MIDPOINT,
        byte_size=99_610,
        group="forcing",
    ),
    PinnedSource(
        dataset_id="igcc-ghg-concentrations",
        semantics=Semantics.FORCING_ASSESSMENT,
        source_id="IGCC-2023",
        title="IGCC greenhouse gas concentrations",
        version="v2024.05.29b",
        variable="co2_concentration",
        unit="ppm_CO2",
        license_id="cc-by-4.0",
        url=f"{IGCC_RAW}/greenhouse_gas_concentrations/ghg_concentrations.csv",
        citation=IGCC_CITATION,
        file_name="ghg_concentrations.csv",
        time_label=TimeLabel.YEAR_START,
        checksum_md5="10a4d4d1a27d07bc16007c368dfa6741",
        unit_note="CO2 in ppm. CH4 and N2O in ppb. Other gases in ppt. Not a radiative forcing.",
        group="forcing",
    ),
    PinnedSource(
        dataset_id="igcc-ghg-emissions-gwp100",
        semantics=Semantics.EMISSION_INVENTORY,
        source_id="IGCC-2023",
        title="IGCC greenhouse gas emissions as GWP100",
        version="v2024.05.29b",
        variable="co2e_gwp100_emissions",
        unit="GtCO2e/yr",
        license_id="cc-by-4.0",
        url=f"{IGCC_RAW}/greenhouse_gas_emissions/greenhouse_gas_emissions_co2eq.csv",
        citation=IGCC_CITATION,
        file_name="greenhouse_gas_emissions_co2eq.csv",
        time_label=TimeLabel.YEAR_MIDPOINT,
        checksum_md5="23d68bf4efc8776e3f80113bcd705273",
        unit_note=(
            "Metadata says GWP100 (AR6) and does not print the mass unit. "
            "Recent CO2-FFI values near 36 match GtCO2 per year, so the reader tags GtCO2e/yr as an inference. "
            "The series is not a carbon-cycle mass flux."
        ),
        group="forcing",
    ),
    PinnedSource(
        dataset_id="igcc-energy-inventory",
        semantics=Semantics.ENERGY_INVENTORY,
        source_id="IGCC-2023",
        title="IGCC Earth energy inventory by component",
        version="v2024.05.29b",
        variable="heat_content",
        unit="ZJ",
        license_id="cc-by-4.0",
        url=f"{IGCC_RAW}/earth_energy_imbalance/earth_energy_imbalance.csv",
        citation=IGCC_CITATION,
        file_name="earth_energy_imbalance.csv",
        time_label=TimeLabel.BOUNDED_INTERVAL,
        byte_size=9_553,
        checksum_md5="4035613b42d63bf84a5d2b049ef70505",
        unit_note=(
            "The filename says imbalance. The metadata says values are in ZJ, so this is a heat inventory. "
            "Error columns are one standard deviation. A flux requires an explicit time difference."
        ),
        group="forcing",
    ),
    PinnedSource(
        dataset_id="climatebench-cmip6-zip",
        semantics=Semantics.SOURCE_SIMULATION,
        source_id="ClimateBench",
        title="ClimateBench CMIP6 annual archive",
        version="1.0.0",
        variable="tas_absolute",
        unit="K",
        license_id="cc-by-4.0",
        url="https://zenodo.org/api/records/7064308/files/CMIP6.zip/content",
        citation=CLIMATEBENCH_CITATION,
        file_name="CMIP6.zip",
        experiment="ssp245",
        byte_size=1_473_338_296,
        checksum_md5="de91683206e167fc1d81d555217a7925",
        license_note="CC BY 4.0. Members are extracted by range request. The zip itself is not stored.",
        unit_note="Archive of annual tas and pr. The time axis inside each file is authoritative, not the years in the filename.",
        group="simulation-archive",
    ),
    PinnedSource(
        dataset_id="climatebench-train-val",
        semantics=Semantics.SOURCE_SIMULATION,
        source_id="NorESM2-LM",
        title="ClimateBench NorESM2 training archive",
        version="1.0.0",
        variable="tas_absolute",
        unit="K",
        license_id="cc-by-4.0",
        url="https://zenodo.org/api/records/7064308/files/train_val.tar.gz/content",
        citation=CLIMATEBENCH_CITATION,
        file_name="train_val.tar.gz",
        experiment="multiple",
        member_id="r1i1p1f1",
        member_role=MemberRole.INITIAL_CONDITION,
        byte_size=839_129_430,
        checksum_md5="9bf0ebd9662667cc1ed78fd3eb5cad4e",
        license_note="CC BY 4.0. One model family with several experiments. Not a multi-model ensemble.",
        group="simulation-archive",
    ),
    PinnedSource(
        dataset_id="climatebench-test",
        semantics=Semantics.SOURCE_SIMULATION,
        source_id="ClimateBench",
        title="ClimateBench held-out model archive",
        version="1.0.0",
        variable="tas_absolute",
        unit="K",
        license_id="cc-by-4.0",
        url="https://zenodo.org/api/records/7064308/files/test.tar.gz/content",
        citation=CLIMATEBENCH_CITATION,
        file_name="test.tar.gz",
        experiment="ssp245",
        byte_size=74_442_376,
        checksum_md5="a09a3ddc8c24e1d6d4b1ee4be5936510",
        group="simulation-archive",
    ),
)


# Annual tas members pulled out of CMIP6.zip. The zip listing showed ssp245 and
# piControl only. Historical, ssp126, ssp370, ssp585, hist-GHG, and hist-aer
# are not in that zip. They are expected inside the NorESM2 train_val archive
# and are not claimed here.
CMIP6_TAS_MEMBERS: tuple[dict[str, str], ...] = (
    {"source_id": "CanESM5", "experiment": "ssp245", "member": "r1i1p1f1", "grid": "gn", "years": "2015-2100"},
    {"source_id": "CanESM5", "experiment": "piControl", "member": "r1i1p1f1", "grid": "gn", "years": "5366-5451"},
    {"source_id": "MIROC6", "experiment": "ssp245", "member": "r1i1p1f1", "grid": "gn", "years": "2015-2100"},
    {"source_id": "MIROC6", "experiment": "piControl", "member": "r1i1p1f1", "grid": "gn", "years": "3365-3450"},
    {"source_id": "MPI-ESM1-2-LR", "experiment": "ssp245", "member": "r1i1p1f1", "grid": "gn", "years": "2015-2100"},
    {"source_id": "MPI-ESM1-2-LR", "experiment": "piControl", "member": "r1i1p1f1", "grid": "gn", "years": "2015-2100"},
    {"source_id": "HadGEM3-GC31-LL", "experiment": "ssp245", "member": "r1i1p1f3", "grid": "gn", "years": "2015-2100"},
    {"source_id": "HadGEM3-GC31-LL", "experiment": "piControl", "member": "r1i1p1f1", "grid": "gn", "years": "2015-2100"},
    {"source_id": "NorESM2-LM", "experiment": "ssp245", "member": "r1i1p1f1", "grid": "gn", "years": "2015-2100"},
    {"source_id": "NorESM2-LM", "experiment": "piControl", "member": "r1i1p1f1", "grid": "gn", "years": "1765-1850"},
)


def cmip6_member_name(entry: dict[str, str]) -> str:
    return (
        f"tas_Ayr_{entry['source_id']}_{entry['experiment']}_{entry['member']}_"
        f"{entry['grid']}_{entry['years']}.nc"
    )


def sources_by_id() -> dict[str, PinnedSource]:
    return {source.dataset_id: source for source in PINNED}


def sources_in_group(group: str) -> tuple[PinnedSource, ...]:
    return tuple(source for source in PINNED if source.group == group)
