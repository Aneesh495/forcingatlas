"""Fingerprint scaling recovers a known mixture and loses a shuffled target."""

from __future__ import annotations

import numpy as np

from forcing_atlas.attribution import fit_scaling, shuffle_observation


def test_known_mixture_is_recovered() -> None:
    years = np.arange(1900, 1980)
    generator = np.random.default_rng(2)
    co2 = np.linspace(0.0, 1.0, years.size)
    aerosol = -0.4 * co2 + 0.05 * generator.normal(size=years.size)
    natural = 0.1 * np.sin(np.linspace(0, 8, years.size))
    observed = 1.2 * co2 + 0.8 * aerosol + 0.5 * natural
    fit = fit_scaling(
        years,
        observed,
        {"co2": co2, "aerosol": aerosol, "natural": natural},
        ("co2", "aerosol", "natural"),
        train_end=1969,
        holdout_start=1970,
    )
    np.testing.assert_allclose(fit.beta, [1.2, 0.8, 0.5], atol=1e-8)
    assert fit.train_r2 == 1.0
    assert fit.holdout_rmse < 1e-8


def test_shuffled_target_fits_worse() -> None:
    years = np.arange(1900, 1980)
    co2 = np.linspace(0.0, 1.0, years.size)
    aerosol = -0.3 * co2
    natural = np.sin(np.linspace(0, 6, years.size))
    observed = 1.0 * co2 + 0.5 * aerosol + 0.2 * natural
    real = fit_scaling(
        years,
        observed,
        {"co2": co2, "aerosol": aerosol, "natural": natural},
        ("co2", "aerosol", "natural"),
        train_end=1979,
        holdout_start=1979,
    )
    shuffled = fit_scaling(
        years,
        shuffle_observation(observed, seed=4),
        {"co2": co2, "aerosol": aerosol, "natural": natural},
        ("co2", "aerosol", "natural"),
        train_end=1979,
        holdout_start=1979,
    )
    assert real.train_r2 > shuffled.train_r2 + 0.5
