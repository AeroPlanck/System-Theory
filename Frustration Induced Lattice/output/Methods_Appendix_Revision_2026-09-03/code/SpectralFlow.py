"""UV-regularized finite-strip spectra and empirical reference-line counts.

Defaults match the circular particle experiment: N=2000, L=7 (diameter),
R=L/2, K=20.75, v=3, omega=0, d0=1.  The closure coefficient is
lambda=K/(rho0*pi*d0**2), with rho0=N/(pi*R**2).

The horizontal levels +/-10 are retained empirical reference levels. A
crossing count is not automatically a line-gap spectral-flow invariant.
This implementation tracks individual eigenvectors, flags near degeneracy
or ambiguous assignments, and excludes flagged crossings. It does NOT
implement exceptional-cluster/Riesz-subspace continuation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from scipy.special import j1
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
})


def Ghat(q: np.ndarray | float, d0: float) -> np.ndarray | float:
    """Fourier/Hankel transform used in the uploaded notes.

    Ghat(q) = 2*pi*d0/q * J1(q*d0), with limit Ghat(0)=pi*d0^2.
    """
    q_arr = np.asarray(q, dtype=float)
    out = np.empty_like(q_arr, dtype=float)
    small = np.abs(q_arr) < 1e-10
    out[small] = np.pi * d0**2
    out[~small] = 2.0 * np.pi * d0 / q_arr[~small] * j1(q_arr[~small] * d0)
    if np.isscalar(q):
        return float(out)
    return out


def M_matrix_standalone(qx: float, qy: float, params: tuple) -> np.ndarray:
    """Standalone bulk symbol M(qx,qy).

    params = (v, omega, lam, alpha, rho0, d0)
    D0 is computed as 2*omega - 2*lambda*rho0*G0*sin(alpha),
    with G0 = pi*d0^2.
    """
    v, omega, lam, alpha, rho0, d0 = params
    q = float(np.hypot(qx, qy))
    Gq = Ghat(q, d0)
    G0 = np.pi * d0**2
    D0 = 2.0 * omega - 2.0 * lam * rho0 * G0 * np.sin(alpha)
    denominator_scale = max(1.0, abs(2.0 * omega), abs(2.0 * lam * rho0 * G0))
    if not np.all(np.isfinite(params)) or abs(D0) <= 1e-12 * denominator_scale:
        raise ValueError("The closed bulk matrix requires a nonsingular D0; phase-lag endpoints are excluded.")

    a = 0.5 * lam * rho0 * Gq * np.cos(alpha)
    b = (
        -omega
        + lam * rho0 * G0 * np.sin(alpha)
        - 0.5 * lam * rho0 * Gq * np.sin(alpha)
        + (v**2 / (4.0 * D0)) * q**2
    )

    return np.array(
        [
            [0.0, -1j * v * qx, -1j * v * qy],
            [-0.5j * v * qx, a, b],
            [-0.5j * v * qy, -b, a],
        ],
        dtype=np.complex128,
    )


def fourier_blocks_for_ky(
    ky: float,
    params: tuple,
    kx_cut: float,
    n_kx: int,
    max_R: int,
) -> dict[int, np.ndarray]:
    """T_R(ky) = (1/N) sum_j M(kx_j,ky) exp(i*kx_j*a*R)."""
    a = np.pi / kx_cut
    kx_grid = np.linspace(-kx_cut, kx_cut, n_kx, endpoint=False)
    M_samples = np.empty((n_kx, 3, 3), dtype=np.complex128)
    for j, kx in enumerate(kx_grid):
        M_samples[j] = M_matrix_standalone(kx, ky, params)

    blocks: dict[int, np.ndarray] = {}
    for R in range(-max_R, max_R + 1):
        phase = np.exp(1j * kx_grid * a * R)
        blocks[R] = np.einsum("j,jab->ab", phase, M_samples) / n_kx
    return blocks


def build_strip_operator(
    ky: float,
    params: tuple,
    n_cells: int,
    kx_cut: float,
    n_kx: int,
    hop_cut: int | None = None,
) -> np.ndarray:
    if hop_cut is None:
        hop_cut = n_cells - 1
    hop_cut = min(hop_cut, n_cells - 1)

    blocks = fourier_blocks_for_ky(ky, params, kx_cut, n_kx, hop_cut)
    L = np.zeros((3 * n_cells, 3 * n_cells), dtype=np.complex128)
    for i in range(n_cells):
        for j in range(n_cells):
            R = i - j
            if abs(R) <= hop_cut:
                L[3*i:3*i+3, 3*j:3*j+3] = blocks[R]
    return L


def eig_left_right(A: np.ndarray):
    # scipy.linalg.eig returns already paired left/right vectors. There is no
    # need to solve a second eigenproblem or rematch the current left frame.
    vals_r, L, R = eig(A, left=True, right=True)
    for n in range(R.shape[1]):
        s = L[:, n].conj().T @ R[:, n]
        if abs(s) > 1e-14:
            L[:, n] /= np.conj(s)
    return vals_r, R, L


def assignment_to_previous(prev_L, prev_vals, vals, R, eig_weight=0.25):
    overlap = np.abs(prev_L.conj().T @ R)
    overlap_scale = max(float(np.max(overlap)), 1e-15)
    overlap_cost = 1.0 - np.clip(overlap / overlap_scale, 0.0, 1.0)

    eig_diff = np.abs(prev_vals[:, None] - vals[None, :])
    eig_scale = max(float(np.max(eig_diff)), 1e-15)
    eig_cost = eig_diff / eig_scale

    cost = (1.0 - eig_weight) * overlap_cost + eig_weight * eig_cost
    row_ind, col_ind = linear_sum_assignment(cost)

    order = np.zeros(R.shape[1], dtype=int)
    for r, c in zip(row_ind, col_ind):
        order[r] = c
    # A small/negative row alternative margin is conservative evidence of an
    # ambiguous individual-branch assignment, not a subspace diagnostic.
    alternatives = cost.copy()
    alternatives[np.arange(len(order)), order] = np.inf
    margin = np.min(alternatives, axis=1) - cost[np.arange(len(order)), order]
    return order, margin


def match_to_previous(prev_L, prev_vals, vals, R, eig_weight=0.25):
    """Backward-compatible two-value wrapper for individual branch matching."""
    order, _ = assignment_to_previous(prev_L, prev_vals, vals, R, eig_weight)
    return vals[order], R[:, order]


def edge_weights(R: np.ndarray, n_cells: int, edge_width: int):
    n_modes = R.shape[1]
    left = np.zeros(n_modes)
    right = np.zeros(n_modes)
    for n in range(n_modes):
        psi = R[:, n].reshape(n_cells, 3)
        dens = np.sum(np.abs(psi)**2, axis=1)
        s = float(np.sum(dens))
        if s > 1e-30:
            left[n] = float(np.sum(dens[:edge_width])) / s
            right[n] = float(np.sum(dens[-edge_width:])) / s
    return left, right


@dataclass
class FlowData:
    ky: np.ndarray
    eigvals: np.ndarray
    left_weight: np.ndarray
    right_weight: np.ndarray
    params: tuple | None = None
    ambiguous: np.ndarray | None = None
    assignment_ambiguous: np.ndarray | None = None
    diagnostics: dict | None = None


def compute_strip_data(
    params: tuple,
    ky_max: float = 40.0,
    n_ky: int = 101,
    n_cells: int = 36,
    kx_cut: float = 40.0,
    n_kx: int = 384,
    hop_cut: int | None = None,
    edge_width: int = 6,
    eig_weight: float = 0.25,
    relative_gap_tol: float = 1e-7,
    self_overlap_tol: float = 1e-8,
    assignment_margin_tol: float = 1e-6,
):
    if not (1 <= edge_width < n_cells / 2):
        raise ValueError("edge_width must satisfy 1 <= edge_width < n_cells/2")
    if n_ky < 3 or ky_max <= 0 or n_kx < 2 or kx_cut <= 0:
        raise ValueError("Positive cutoffs and nontrivial momentum grids are required")
    ky_grid = np.linspace(-ky_max, ky_max, n_ky)

    vals_list = []
    lw_list = []
    rw_list = []
    ambiguous_list = []
    assignment_list = []
    relative_separation_min = np.inf
    normalized_self_overlap_min = np.inf
    assignment_margin_min = np.inf

    prev_L = None
    prev_vals = None

    for idx, ky in enumerate(ky_grid):
        A = build_strip_operator(ky, params, n_cells, kx_cut, n_kx, hop_cut)
        vals, R, L = eig_left_right(A)

        if idx == 0:
            order = np.lexsort((vals.real, vals.imag))
            vals = vals[order]
            R = R[:, order]
            L = L[:, order]
        else:
            order, margin = assignment_to_previous(prev_L, prev_vals, vals, R, eig_weight)
            vals, R, L = vals[order], R[:, order], L[:, order]
            assignment_list.append(margin <= assignment_margin_tol)
            assignment_margin_min = min(assignment_margin_min, float(np.min(margin)))

        separation = np.abs(vals[:, None] - vals[None, :])
        scale = np.maximum(1.0, np.maximum(np.abs(vals[:, None]), np.abs(vals[None, :])))
        relative_separation = separation / scale
        np.fill_diagonal(relative_separation, np.inf)
        per_mode_separation = np.min(relative_separation, axis=1)
        self_overlap = np.abs(np.sum(L.conj() * R, axis=0)) / (
            np.linalg.norm(L, axis=0) * np.linalg.norm(R, axis=0)
        )
        ambiguous_list.append((per_mode_separation <= relative_gap_tol) | (self_overlap <= self_overlap_tol))
        relative_separation_min = min(relative_separation_min, float(np.min(per_mode_separation)))
        normalized_self_overlap_min = min(normalized_self_overlap_min, float(np.min(self_overlap)))

        lw, rw = edge_weights(R, n_cells, edge_width)
        vals_list.append(vals)
        lw_list.append(lw)
        rw_list.append(rw)
        prev_L = L
        prev_vals = vals

    return FlowData(
        ky=ky_grid,
        eigvals=np.asarray(vals_list),
        left_weight=np.asarray(lw_list),
        right_weight=np.asarray(rw_list),
        params=tuple(float(x) for x in params),
        ambiguous=np.asarray(ambiguous_list),
        assignment_ambiguous=np.asarray(assignment_list),
        diagnostics={
            "tracking_method": "individual biorthogonal eigenvector Hungarian assignment; no subspace continuation",
            "relative_gap_tol": relative_gap_tol,
            "self_overlap_tol": self_overlap_tol,
            "assignment_margin_tol": assignment_margin_tol,
            "relative_eigenvalue_separation_min": relative_separation_min,
            "normalized_self_overlap_min": normalized_self_overlap_min,
            "assignment_margin_min": assignment_margin_min,
            "near_degenerate_mode_sample_count": int(np.sum(ambiguous_list)),
            "ambiguous_assignment_count": int(np.sum(assignment_list)),
            "n_cells": n_cells, "kx_cut": kx_cut, "n_kx": n_kx,
            "hop_cut": n_cells - 1 if hop_cut is None else min(hop_cut, n_cells - 1),
            "edge_width": edge_width, "ky_max": ky_max, "n_ky": n_ky,
            "delta_ky": float(ky_grid[1] - ky_grid[0]), "eig_weight": eig_weight,
        },
    )


def count_horizontal_crossings(
    data: FlowData,
    c: float,
    edge_threshold: float = 0.45,
    return_diagnostics: bool = False,
    zero_tol: float = 1e-10,
):
    """Count reference crossings with persistent labels and resolved tracking.

    Both bracketing samples must carry the SAME edge label. Flagged samples
    and assignments are excluded and explicitly reported. Exact grid-vertex
    crossings are bracketed once by their nearest nonzero neighbors; endpoint
    contacts are not counted as closed-interval transverse crossings.
    """
    g = data.eigvals.imag - c
    sf_left = 0
    sf_right = 0
    crossings_left = []
    crossings_right = []
    rejected = []
    endpoint_contact_count = int(np.sum(np.abs(g[[0, -1]]) <= zero_tol))

    n_ky, n_modes = g.shape
    for mode in range(n_modes):
        nonzero = np.flatnonzero(np.abs(g[:, mode]) > zero_tol)
        for i, j in zip(nonzero[:-1], nonzero[1:]):
            g0, g1 = g[i, mode], g[j, mode]
            if g0 * g1 < 0:
                sign = +1 if (g1 - g0) > 0 else -1
                left_two = data.left_weight[[i, j], mode]
                right_two = data.right_weight[[i, j], mode]
                wl, wr = float(np.mean(left_two)), float(np.mean(right_two))
                fraction = abs(g0) / (abs(g0) + abs(g1))
                ky_cross = data.ky[i] + (data.ky[j] - data.ky[i]) * fraction
                z_cross = (1 - fraction) * data.eigvals[i, mode] + fraction * data.eigvals[j, mode]
                record = (mode, float(ky_cross), complex(z_cross), int(sign), float(wl), float(wr))
                near_degenerate = data.ambiguous is not None and np.any(data.ambiguous[i:j+1, mode])
                ambiguous_match = data.assignment_ambiguous is not None and np.any(data.assignment_ambiguous[i:j, mode])
                left_persistent = np.all(left_two >= edge_threshold) and np.all(left_two > right_two)
                right_persistent = np.all(right_two >= edge_threshold) and np.all(right_two > left_two)
                if near_degenerate or ambiguous_match or not (left_persistent or right_persistent):
                    rejected.append({
                        "mode": int(mode), "ky": float(ky_cross), "sign": int(sign),
                        "near_degenerate": bool(near_degenerate), "ambiguous_assignment": bool(ambiguous_match),
                        "persistent_edge_label": bool(left_persistent or right_persistent),
                        "left_endpoint_weights": left_two.tolist(), "right_endpoint_weights": right_two.tolist(),
                    })
                elif left_persistent:
                    sf_left += sign
                    crossings_left.append(record)
                elif right_persistent:
                    sf_right += sign
                    crossings_right.append(record)
    if return_diagnostics:
        return sf_left, sf_right, crossings_left, crossings_right, {
            "level": float(c), "edge_threshold": float(edge_threshold),
            "two_sided_persistent_label_required": True, "zero_tol": zero_tol,
            "excluded_crossings": rejected, "endpoint_contact_count": endpoint_contact_count,
            "interpretation": "finite-cutoff resolved reference-line crossing count, not a certified bulk line-gap invariant",
        }
    return sf_left, sf_right, crossings_left, crossings_right


def plot_verified_style(
    data: FlowData,
    filename: str = "spectral_flow_verified_repro.png",
    edge_threshold: float = 0.45,
    params: tuple | None = None,
):
    ky = data.ky
    fig, ax = plt.subplots(figsize=(12, 7))

    vals = data.eigvals
    edge_max = np.maximum(data.left_weight, data.right_weight)
    weak = edge_max < edge_threshold
    left = (data.left_weight >= edge_threshold) & (data.left_weight > data.right_weight)
    right = (data.right_weight >= edge_threshold) & (data.right_weight > data.left_weight)

    ax.plot(np.repeat(ky[:, None], vals.shape[1], axis=1)[weak], vals.imag[weak], ".", color="lightgray", ms=2, alpha=0.7)
    ax.plot(np.repeat(ky[:, None], vals.shape[1], axis=1)[left], vals.imag[left], ".", color="tab:blue", ms=3, label="left edge")
    ax.plot(np.repeat(ky[:, None], vals.shape[1], axis=1)[right], vals.imag[right], ".", color="tab:orange", ms=3, label="right edge")

    ax.axhline(10.0, color="black", ls="--", lw=1.2, label=r"Im $\sigma=10$")
    ax.axhline(-10.0, color="black", ls=":", lw=1.2, label=r"Im $\sigma=-10$")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"Im $\sigma$")

    actual_params = params if params is not None else data.params
    if actual_params is None:
        raise ValueError("Plotting requires actual params in FlowData or the params argument")
    v, omega, lam, alpha, rho0, d0 = actual_params
    G0 = np.pi * d0**2
    D0 = 2.0 * omega - 2.0 * lam * rho0 * G0 * np.sin(alpha)
    alpha_over_pi = alpha / np.pi
    ax.set_title(
        rf"Finite-strip spectrum, $\alpha={alpha_over_pi:.3g}\pi$, "
        rf"$v={v:g},\ \omega={omega:g},\ K={lam*rho0*G0:g},\ d_0={d0:g}$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    return fig, ax


# main.py:1409-1411 sets circleRadius=boundaryLength/2, hence R=3.5.
PARTICLE_PARAMETERS = {"N": 2000, "L": 7.0, "R": 3.5, "K": 20.75, "v": 3.0, "omega": 0.0, "d0": 1.0}
DEFAULT_ALPHA_OVER_PI = (0.2, 0.4, 0.5, 0.6, 0.8)


def matched_particle_params(alpha):
    p = PARTICLE_PARAMETERS
    rho0 = p["N"] / (np.pi * p["R"]**2)
    lam = p["K"] / (rho0 * np.pi * p["d0"]**2)
    return (p["v"], p["omega"], lam, float(alpha), rho0, p["d0"])


DEFAULT_PARAMS = matched_particle_params(0.5 * np.pi)


if __name__ == "__main__":
    # Publish run_strip_matched_sweep.py alongside this module. Direct execution
    # runs the same five nonsingular alphas and validation grids as that runner.
    from run_strip_matched_sweep import main
    main()
