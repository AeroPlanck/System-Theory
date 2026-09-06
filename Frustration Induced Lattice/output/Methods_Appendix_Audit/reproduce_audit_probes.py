"""Read-only mathematical/code probes for the 2026-09-03 Methods audit.

Run with --strip to also recompute the published finite-strip crossing count.
No manuscript, production code, simulation data, or plot is written.
Once revisions exist, this historical audit uses their pre-revision code
snapshot so that its original counterexamples remain reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
original_code_snapshot = (
    Path(__file__).resolve().parents[1]
    / "Methods_Appendix_Revision_2026-09-03" / "before"
)
sys.path.insert(0, str(original_code_snapshot) if original_code_snapshot.is_dir()
                else r"D:\PrivatePythonProject\Math\Lattice")

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.optimize import brentq

from Dispersion import M_matrix_vectorized, eigs_at_k
from ChernNumberCompute import compute_topology
import SpectralFlow as strip


def parameters(k_eff=20.75, alpha=0.99 * np.pi, omega=0.0, d0=1.0):
    rho0 = 0.0204
    return (3.0, omega, k_eff / (rho0 * np.pi * d0**2), alpha, rho0, d0)


def complex_rows(values):
    return [[float(z.real), float(z.imag)] for z in np.asarray(values).ravel()]


def topology_result(params, **kwargs):
    value, integer, diagnostic = compute_topology(
        params, [0], Q=60.0, N_theta=71, N_phi=91, delta=1.0e-3, **kwargs
    )
    return {
        "C_value": value,
        "C_integer": integer,
        "min_sigma": diagnostic["min_sigma"],
        "min_det": diagnostic["min_det"],
        "unresolved_bad_plaquettes": diagnostic["unresolved_bad_plaquette_count"],
        "gap_valid_flag_present": any("gap" in key for key in diagnostic),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strip", action="store_true")
    args = parser.parse_args()
    result = {}

    # No gradients, rho=4, Ghat(0)=1, lambda=1, omega=0, alpha=pi/2, z=1.
    alpha = 0.5 * np.pi
    denominator = -8.0
    forcing = np.exp(1j * alpha)
    q_text = (np.sin(alpha) - 1j * np.cos(alpha)) / denominator
    q_correct = 1j * forcing / denominator
    result["nonlinear_Q_sign_probe"] = {
        "Q_text": complex_rows([q_text])[0],
        "Q_algebraic_solution": complex_rows([q_correct])[0],
        "text_residual_abs": float(abs(1j * denominator * q_text + forcing)),
        "correct_residual_abs": float(abs(1j * denominator * q_correct + forcing)),
    }
    assert abs(1j * denominator * q_text + forcing) > 1.9
    assert abs(1j * denominator * q_correct + forcing) < 1e-14

    # These are the actual defaults in SpectralFlow.py, not particle parameters.
    params = strip.DEFAULT_PARAMS
    k_line = brentq(lambda k: np.max(eigs_at_k(0.0, k, params).imag) - 10.0, 0.0, 2.0)
    bulk_at_line = eigs_at_k(0.0, k_line, params)
    result["claimed_horizontal_line_gap"] = {
        "actual_strip_parameters": list(params),
        "kx": 0.0,
        "ky": k_line,
        "bulk_eigenvalues_real_imag": complex_rows(bulk_at_line),
        "max_bulk_matrix_implementation_difference": float(
            np.max(np.abs(strip.M_matrix_standalone(0.0, k_line, params)
                          - M_matrix_vectorized(0.0, k_line, *params)))
        ),
    }
    assert np.min(abs(bulk_at_line.imag - 10.0)) < 1e-10
    assert np.min(abs(bulk_at_line.imag + 10.0)) < 1e-10

    # Characteristic polynomial: sigma^3 - 2a sigma^2
    # + (a^2+b^2+v^2 k^2/2) sigma - a v^2 k^2/2.
    params = parameters()

    def coefficients(k):
        matrix = M_matrix_vectorized(k, 0.0, *params)
        a, b = float(matrix[1, 1].real), float(matrix[1, 2].real)
        c2 = params[0] ** 2 * k**2 / 2.0
        return a, b, c2

    def discriminant(k):
        a, b, c2 = coefficients(k)
        p, q, r = -2 * a, a*a + b*b + c2, -a*c2
        return p*p*q*q - 4*q**3 - 4*p**3*r - 27*r*r + 18*p*q*r

    radial_grid = np.linspace(0.0, 1.0, 10001)
    disc = np.array([discriminant(k) for k in radial_grid])
    brackets = np.flatnonzero(disc[:-1] * disc[1:] < 0)
    ep_roots = [brentq(discriminant, radial_grid[i], radial_grid[i+1]) for i in brackets]
    k_b0 = brentq(lambda k: coefficients(k)[1], 0.0, 1.0)
    a_b0, _, c2_b0 = coefficients(k_b0)
    result["finite_k_EP_counterexample"] = {
        "K_effective": 20.75,
        "alpha_over_pi": 0.99,
        "b_zero_k": k_b0,
        "a_at_b_zero": a_b0,
        "two_by_two_discriminant": a_b0*a_b0 - 4*c2_b0,
        "eigenvalues_at_origin": complex_rows(eigs_at_k(0.0, 0.0, params)),
        "eigenvalues_at_b_zero": complex_rows(eigs_at_k(k_b0, 0.0, params)),
        "cubic_discriminant_zeroes": ep_roots,
        "first_nonzero_scan_k": float(60 * np.tan((np.pi - 1e-3) / 70 / 2)),
        "unvalidated_topology_output": topology_result(params),
    }
    assert len(ep_roots) == 2
    assert np.max(abs(eigs_at_k(k_b0, 0.0, params).imag)) < 1e-9

    params = parameters(k_eff=20.0, alpha=np.pi)
    v, omega, lam, alpha, rho0, d0 = params
    result["singular_pi_endpoint_is_evaluated"] = {
        "floating_D0": float(2*omega - 2*lam*rho0*np.pi*d0*d0*np.sin(alpha)),
        "largest_matrix_entry_at_k1": float(np.max(abs(M_matrix_vectorized(1.0, 0.0, *params)))),
        "topology_output": topology_result(params),
    }

    params = parameters(k_eff=20.0, alpha=0.5*np.pi, omega=30.0)
    result["hardcoded_infinity_cap_counterexample"] = {
        "omega": 30.0,
        "default_cap": topology_result(params),
        "correct_u_minus_cap": topology_result(
            params, infty_basis=np.array([[0.0], [1.0], [-1j]])
        ),
    }

    synthetic = strip.FlowData(
        ky=np.array([0.0, 1.0]),
        eigvals=np.array([[-1j], [1j]]),
        left_weight=np.array([[0.90], [0.01]]),
        right_weight=np.array([[0.01], [0.01]]),
    )
    sf_left, sf_right, _, _ = strip.count_horizontal_crossings(synthetic, 0.0, 0.45)
    result["two_sided_localization_counterexample"] = {
        "left_weights": [0.90, 0.01],
        "threshold": 0.45,
        "returned_count": [sf_left, sf_right],
    }
    assert sf_left == 1

    if args.strip:
        data = strip.compute_strip_data(
            strip.DEFAULT_PARAMS, ky_max=50.0, n_ky=101, n_cells=36,
            kx_cut=40.0, n_kx=384, hop_cut=35, edge_width=6,
        )
        result["reproduced_finite_strip_counts"] = {}
        for c in (10.0, -10.0):
            left, right, _, _ = strip.count_horizontal_crossings(data, c, 0.45)
            result["reproduced_finite_strip_counts"][str(c)] = [left, right]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
