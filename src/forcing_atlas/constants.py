"""Physical constants and declared conversion factors.

Every factor that is a convention rather than a derived identity is named.
Callers that need a different convention pass it explicitly. Nothing here
silently replaces one climate variable with another.
"""

from __future__ import annotations

__version__ = "0.1.0"

# IUGG mean Earth radius. Area and energy conversions in this project use this
# sphere. An ellipsoid is a different geometry and is not implied.
EARTH_RADIUS_M = 6_371_000.0

DAY_SECONDS = 86_400.0
# Julian year used by the energy-balance time coordinate. Calendar annualization
# still uses the dataset calendar. These are different clocks.
JULIAN_YEAR_DAYS = 365.25
JULIAN_YEAR_SECONDS = JULIAN_YEAR_DAYS * DAY_SECONDS
NOLEAP_YEAR_SECONDS = 365.0 * DAY_SECONDS
DAY360_YEAR_SECONDS = 360.0 * DAY_SECONDS

# Ice-point offset on the absolute temperature scale. Anomalies do not use it.
ABSOLUTE_ZERO_OFFSET_C = 273.15

# Mass. 1 gigatonne = 1e9 tonnes = 1e12 kg. 1 megatonne = 1e9 kg.
KG_PER_GT = 1.0e12
KG_PER_MT = 1.0e9

# Stoichiometry. These convert chemical mass, not warming potentials.
KG_CO2_PER_KG_C = 44.0 / 12.0
KG_C_PER_KG_CO2 = 12.0 / 44.0
KG_CH4_PER_KG_C = 16.0 / 12.0

# Declared carbon-mass conversion. 1 ppm of CO2 corresponds to this many
# gigatonnes of carbon in a well-mixed atmosphere under the AR5 conventional
# factor used by Joos et al. (2013). It is not a unit identity.
GTC_PER_PPM_CO2 = 2.124

# Assessed effective radiative forcing for a doubling of CO2, AR6 central value.
# Myhre et al. (1998) used F = 5.35 ln(C/C0), which implies about 3.71 W m-2.
# Both are available. The default pathway adapter uses the AR6 central ERF.
CO2_ERF_DOUBLING_AR6_W_M2 = 3.93
MYHRE_1998_ALPHA_W_M2 = 5.35

# Local disk budget for this project, in bytes. The accountant refuses writes
# that would pass the cap. It does not scan unrelated user directories.
PROJECT_DISK_CAP_BYTES = 5_000_000_000

# Chunk size for streaming acquisition.
DEFAULT_CHUNK_BYTES = 1 << 20

# Missing-data policy name. Numeric code paths test this string in evidence.
MISSING_POLICY = "nan_not_zero"
