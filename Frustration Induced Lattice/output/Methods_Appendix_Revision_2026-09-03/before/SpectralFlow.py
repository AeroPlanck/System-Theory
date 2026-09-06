"""
spectral_flow_verified_repro.py

This script is meant to reproduce the style/setting of the earlier
`spectral_flow_verified.png`, not the default plot of `spectral_flow_strip.py`.

Main differences from `spectral_flow_strip.py`:
1. The bulk matrix is defined standalone from ChernNumber.md, rather than
   importing `Dispersion.M_matrix_vectorized`.
2. The plot shows Im(sigma) versus k_y, not the rotated line-gap coordinate
   g(z)=Re(exp(-i theta)(z-z0)).
3. It counts crossings of horizontal line gaps Im(sigma)=c, with c=+10 and -10.
4. It uses D0 = 2*omega - 2*lambda*rho0*G0*sin(alpha), where
   G0 = pi*d0^2, matching the title of the earlier verified figure.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
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
    vals_r, R = np.linalg.eig(A)
    vals_l, L_raw = np.linalg.eig(A.conj().T)

    cost = np.abs(vals_l.conj()[:, None] - vals_r[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    left_order = np.zeros_like(col_ind)
    for r, c in zip(row_ind, col_ind):
        left_order[c] = r
    L = L_raw[:, left_order]

    for n in range(R.shape[1]):
        s = L[:, n].conj().T @ R[:, n]
        if abs(s) > 1e-14:
            L[:, n] /= np.conj(s)
    return vals_r, R, L


def match_to_previous(prev_L, prev_vals, vals, R, eig_weight=0.25):
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


def compute_strip_data(
    params: tuple,
    ky_max: float = 40.0,
    n_ky: int = 101,
    n_cells: int = 36,
    kx_cut: float = 40.0,
    n_kx: int = 384,
    hop_cut: int | None = None,
    edge_width: int = 6,
):
    ky_grid = np.linspace(-ky_max, ky_max, n_ky)

    vals_list = []
    lw_list = []
    rw_list = []

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
            vals, R = match_to_previous(prev_L, prev_vals, vals, R)
            # Rebuild left eigenvectors for next matching in the current order.
            _, R0, L0 = eig_left_right(A)
            overlap = np.abs(L0.conj().T @ R)
            row_ind, col_ind = linear_sum_assignment(1.0 - overlap / max(float(np.max(overlap)), 1e-15))
            L_ordered = np.zeros_like(L0)
            for r, c in zip(row_ind, col_ind):
                L_ordered[:, c] = L0[:, r]
            L = L_ordered

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
    )


def count_horizontal_crossings(
    data: FlowData,
    c: float,
    edge_threshold: float = 0.45,
):
    """Count crossings of Im(sigma)=c separated by left/right edge weight."""
    g = data.eigvals.imag - c
    sf_left = 0
    sf_right = 0
    crossings_left = []
    crossings_right = []

    n_ky, n_modes = g.shape
    for mode in range(n_modes):
        for i in range(n_ky - 1):
            g0, g1 = g[i, mode], g[i + 1, mode]
            if g0 * g1 < 0:
                sign = +1 if (g1 - g0) > 0 else -1
                wl = 0.5 * (data.left_weight[i, mode] + data.left_weight[i + 1, mode])
                wr = 0.5 * (data.right_weight[i, mode] + data.right_weight[i + 1, mode])
                ky_cross = data.ky[i] + (data.ky[i+1] - data.ky[i]) * abs(g0) / (abs(g0) + abs(g1))
                z_cross = 0.5 * (data.eigvals[i, mode] + data.eigvals[i + 1, mode])
                record = (mode, float(ky_cross), complex(z_cross), int(sign), float(wl), float(wr))
                if wl >= edge_threshold and wl > wr:
                    sf_left += sign
                    crossings_left.append(record)
                elif wr >= edge_threshold and wr > wl:
                    sf_right += sign
                    crossings_right.append(record)
    return sf_left, sf_right, crossings_left, crossings_right


def plot_verified_style(
    data: FlowData,
    filename: str = "spectral_flow_verified_repro.png",
    edge_threshold: float = 0.45,
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

    v, omega, lam, alpha, rho0, d0 = DEFAULT_PARAMS
    G0 = np.pi * d0**2
    D0 = 2.0 * omega - 2.0 * lam * rho0 * G0 * np.sin(alpha)
    alpha_over_pi = alpha / np.pi
    ax.set_title(
        rf"Strip spectral flow, $\alpha={alpha_over_pi:.3g}\pi$, "
        rf"$D_0=2\omega-2\lambda\rho_0 G_0\sin\alpha$  "
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    return fig, ax


# Default parameters used for the earlier verified-style figure.
v = 3.0
omega = 1.5
rho0 = 0.0204
d0 = 2.0
lam = 20.0 / (rho0 * np.pi * d0**2)
alpha = 0.5 * np.pi
DEFAULT_PARAMS = (v, omega, lam, alpha, rho0, d0)


if __name__ == "__main__":
    data = compute_strip_data(
        params=DEFAULT_PARAMS,
        ky_max=50.0,
        n_ky=101,
        n_cells=36,
        kx_cut=40.0,
        n_kx=384,
        hop_cut=35,
        edge_width=6,
    )

    for c in (+10.0, -10.0):
        sfL, sfR, left_cross, right_cross = count_horizontal_crossings(data, c=c, edge_threshold=0.45)
        print(f"Line Im(sigma)={c:g}: SF_left={sfL}, SF_right={sfR}")
        print("  left crossings:")
        for item in left_cross:
            print("   ", item)
        print("  right crossings:")
        for item in right_cross:
            print("   ", item)

    plot_verified_style(data, filename="spectral_flow_verified_repro.png")
    plt.show()
