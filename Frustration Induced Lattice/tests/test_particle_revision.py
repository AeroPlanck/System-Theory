"""Regression tests for the boundary-correlation peak interpolation."""

import math

import numpy as np
import pytest

import boundary_arc_correlation_analysis as analysis


@pytest.mark.parametrize(
    "samples",
    [
        (0.9, 1.0, 0.8),
        (0.8, 1.0, 0.9),
        (0.7, 1.0, 0.7),
        (0.72, 0.8, 0.64),
    ],
)
def test_refined_peak_matches_independent_quadratic(samples, monkeypatch):
    """The reported height and location must lie at the fitted parabola vertex.

    Supply a known correlation to isolate interpolation from density estimation,
    while retaining the production normalization, peak search, and arc scaling.
    """
    peak_bin = 100
    arc_step = 0.01
    effective_radius = analysis.BINS * arc_step / (2.0 * math.pi)
    correlation = np.zeros(analysis.BINS)
    correlation[0] = 2.0
    correlation[peak_bin - 1 : peak_bin + 2] = 2.0 * np.asarray(samples)
    monkeypatch.setattr(
        analysis.np.fft,
        "irfft",
        lambda spectrum, n: correlation.copy(),
    )
    monkeypatch.setattr(
        analysis,
        "gaussian_filter1d",
        lambda values, sigma, mode: values.copy(),
    )

    spacing, height = analysis.autocorrelation_spacing(
        np.array([0.0, 0.1]), effective_radius
    )

    # Fit independently instead of copying the production interpolation formula.
    polynomial = np.polyfit(np.array([-1.0, 0.0, 1.0]), samples, deg=2)
    vertex = -polynomial[1] / (2.0 * polynomial[0])
    expected_height = np.polyval(polynomial, vertex)
    assert spacing == pytest.approx((peak_bin + vertex) * arc_step, abs=1e-12)
    assert height == pytest.approx(expected_height, abs=1e-12)

