# Forcing Atlas: observed change and conditional futures

This note records what the local calculations actually did. A pathway below is conditional on a stated forcing. It is not a prediction of policy, and it is not an observation of the future.

## Scope

The instrumental layer is HadCRUT5.1.0.0 and Berkeley Earth land-ocean. The proxy layer is the PAGES 2k 2019 ensemble median. The forcing layer is the IGCC effective radiative forcing table for 1750-2023. The emulation layer is the two-box model and the spatial emulator in this repository. Four ssp245 annual temperature files from ClimateBench are a source-simulation sample. They are not mixed into the observational target.

## Methods in brief

Area-weighted means use cell area on a sphere of radius 6,371 km. Monthly HadCRUT fields are annualized with Gregorian month lengths. A year with a missing month stays missing in that cell. The two-box model is

    Cs dTs/dt = F - lambda Ts - gamma (Ts - Td)
    Cd dTd/dt = gamma (Ts - Td)

with a zero anomaly in 1750 and a shift onto 1961-1990 computed from the training window. Parameters are searched only on 1850-1994, inside a hard prior. 1995-2023 is reported and not used to move the fit.

The spatial emulator has a global head and a residual whose area mean is removed. Seeds 11, 29, and 47 are separate initial weights. The loss does not see 1995-2023.

Fingerprint scaling uses three single-forcing replays of the same two-box parameters. It is not a DAMIP attribution. The negative control shuffles the HadCRUT series in time.

## Studies

The machine-readable record is `results/studies.json`. Each entry has a claim, a computed result, a limit, and the source files.

S01. After rebasing Berkeley Earth from 1951-1980 onto 1961-1990, the root-mean-square difference from HadCRUT over 1880-2020 is 0.050 K. The ocean parts share HadSST, so this is not two independent ocean records.

S02. The 1995-2023 HadCRUT mean is 0.620 K above 1961-1990. The 1850-1900 mean of the same published series is -0.357 K, so that recent window is 0.977 K above 1850-1900. The early window is a reference, not a control run.

S03. From 1850 to 2000 the PAGES 2k median differs from the Cowtan and Way target embedded in the same file by 0.075 K root-mean-square. The reconstruction is not used as a thermometer.

S04. Over 1970-2023 the HadCRUT northern-hemisphere slope is 0.276 K per decade and the southern-hemisphere slope is 0.119 K per decade.

S05. In 2010 the non-infilled grid has mean cell coverage 0.725. The analysis grid, which infills, has 0.993. The non-infilled Arctic overlap mean is 1.756 K with coverage 0.356, against a global area mean of 0.648 K. The Arctic number is not a complete Arctic average.

S06. Over 1991-1993 the HadCRUT mean is 0.205 K and the historical two-box emulation is 0.208 K. The solar-plus-volcanic emulation is -0.118 K. The eruption is in the volcanic term. That experiment is not hist-nat.

S07. Scaling the two-box CO2, aerosol, and natural fingerprints to HadCRUT on the training window gives R2 0.713. The same fit to a time-shuffled HadCRUT series gives R2 0.009. The Gram matrix condition number is 702.8, so the fingerprints are collinear. They are one emulator, not hist-GHG, hist-aer, and hist-nat.

S08. The training search stops at lambda 0.501 W m-2 K-1, the low edge of the prior. The particle 5 to 95 percent range is 0.503 to 0.567. Holdout root-mean-square error is 0.322 K. Persistence on the same years is 0.124 K. Dividing 3.93 W m-2 by this lambda is not an assessed equilibrium climate sensitivity.

S09. The three spatial seeds score 0.779 K, 0.740 K, and 0.705 K on the 1995-2023 grid. A latitude-uniform trend scores 0.801 K. Inverse-error weights are 0.317, 0.333, and 0.350. On the held-out years that remain inside the training forcing radius, the reloaded ensemble has Arctic mean absolute error 1.52 K and tropical mean absolute error 0.41 K. The 95th percentile of absolute cell error is 1.75 K. The year 2023 is an abstention. This is not a withheld model family.

S10. With that same boundary-value feedback, holding the last assessed forcing reaches 2.66 K by 2100 and declining toward the 1990 forcing reaches 1.65 K, both versus 1961-1990. These are not SSPs. The map is not extended.

S11. By 2020, anomalies versus each file's own 2015-2019 mean are 0.33 K for CanESM5 (365-day calendar), 0.10 K for HadGEM3-GC31-LL (360-day), -0.03 K for MIROC6 (Gregorian), and about 0.00 K for MPI-ESM1-2-LR (proleptic Gregorian). One member and one pathway. Not comparable to HadCRUT's 1961-1990 anomaly.

S12. The IGCC energy file rises from 1.54 ZJ to 473.02 ZJ and then has missing years. It is a heat-content stock. It is not the radiative forcing.

## What this does not settle

The two-box holdout is worse than persistence, and the feedback sits on a prior bound. The spatial model beats a uniform trend only by a small margin, and its Arctic errors are several times the tropical errors. Collinear emulator fingerprints are not a detection and attribution of the historical record in the DAMIP sense. Four ssp245 members do not identify the effect of a mitigation policy.
