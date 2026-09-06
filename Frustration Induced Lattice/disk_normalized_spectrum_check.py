"""Convergence check for the normalized top-hat continuum operator on a disk.

This is an independent numerical experiment.  It imports no project model and
does not write to the reference ``Dispersion.py`` or LaTeX files.

The angular block for total azimuthal index m is written in the circular
polarization variables

    rho = R(r) exp(i m phi),
    p+  = P+(r) exp(i (m+1) phi),
    p-  = P-(r) exp(i (m-1) phi).

The radial differential operators are Chebyshev--Lobatto collocation
matrices.  Boundary and regularity conditions are imposed by a tau
generalized eigenproblem.  The normalized disk convolution is evaluated with
an independent Gauss--Legendre quadrature and barycentric interpolation.

The main purpose is diagnostic: at alpha=pi/2 the bulk spectrum is neutral,
so any disk growth-rate selection has to converge under both radial and
integral-quadrature refinement before it is physically interpretable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eig


OUTPUT_DIR = Path("output") / "Normalized_Disk_Spectrum_Check"


@dataclass(frozen=True)
class DiskSpec:
    diameter: float
    coupling: float
    m: int
    n_radial: int
    quadrature_factor: int = 8
    speed: float = 3.0
    interaction_radius: float = 1.0
    omega: float = 0.0
    alpha: float = math.pi / 2.0

    @property
    def radius(self) -> float:
        return self.diameter / 2.0


def chebyshev_lobatto(n: int, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return descending disk radii and first/second radial derivatives."""
    if n < 8:
        raise ValueError("n must be at least 8")
    order = n - 1
    j = np.arange(n)
    x = np.cos(np.pi * j / order)
    c = np.ones(n)
    c[[0, -1]] = 2.0
    c *= (-1.0) ** j
    dx = x[:, None] - x[None, :]
    d = (c[:, None] / c[None, :]) / (dx + np.eye(n))
    d -= np.diag(np.sum(d, axis=1))
    r = radius * (x + 1.0) / 2.0
    dr = (2.0 / radius) * d
    return r, dr, dr @ dr


def chebyshev_barycentric_matrix(x_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Interpolation matrix from Chebyshev--Lobatto nodes to x_eval."""
    n = len(x_nodes)
    weights = (-1.0) ** np.arange(n)
    weights[[0, -1]] *= 0.5
    difference = x_eval[:, None] - x_nodes[None, :]
    close = np.isclose(difference, 0.0, atol=5e-15, rtol=0.0)
    matrix = np.empty_like(difference)
    ordinary = ~np.any(close, axis=1)
    terms = weights[None, :] / difference[ordinary]
    matrix[ordinary] = terms / np.sum(terms, axis=1, keepdims=True)
    for row in np.flatnonzero(~ordinary):
        matrix[row] = 0.0
        matrix[row, np.flatnonzero(close[row])[0]] = 1.0
    return matrix


def top_hat_angular_kernel(
    radii: np.ndarray,
    source_radii: np.ndarray,
    angular_index: int,
    interaction_radius: float,
) -> np.ndarray:
    """Exact angular Fourier coefficient of a disk top-hat kernel."""
    r = radii[:, None]
    rp = source_radii[None, :]
    result = np.zeros((len(radii), len(source_radii)), dtype=float)
    full = r + rp <= interaction_radius
    none = np.abs(r - rp) >= interaction_radius
    partial = ~(full | none)
    ell = abs(int(angular_index))
    if ell == 0:
        result[full] = 2.0 * np.pi
    # All nonzero angular harmonics vanish when the complete source circle is
    # inside the interaction disk.
    if np.any(partial):
        ii, jj = np.nonzero(partial)
        cosine = (
            radii[ii] ** 2
            + source_radii[jj] ** 2
            - interaction_radius**2
        ) / (2.0 * radii[ii] * source_radii[jj])
        chi = np.arccos(np.clip(cosine, -1.0, 1.0))
        if ell == 0:
            result[ii, jj] = 2.0 * chi
        else:
            result[ii, jj] = 2.0 * np.sin(ell * chi) / ell
    return result


@lru_cache(maxsize=256)
def disk_convolution_cached(
    radius: float,
    interaction_radius: float,
    angular_index: int,
    n_radial: int,
    quadrature_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return row-normalized angular convolution and collocation metadata."""
    r, dr, d2r = chebyshev_lobatto(n_radial, radius)
    x_nodes = 2.0 * r / radius - 1.0
    # The angular kernel has square-root endpoint behavior at
    # r'=|r-d0| and r'=r+d0.  A single global quadrature produces a small,
    # non-monotone real part at alpha=pi/2.  Split every target row at those
    # geometric endpoints so that quadrature refinement is meaningful.
    order = max(quadrature_factor * n_radial, 96)
    base_x, base_w = leggauss(order)
    raw_operator = np.zeros((n_radial, n_radial), dtype=float)
    accessible_area = np.zeros(n_radial, dtype=float)

    for row, target_radius in enumerate(r):
        full_end = max(0.0, interaction_radius - target_radius)
        full_end = min(radius, full_end)
        partial_start = abs(target_radius - interaction_radius)
        if target_radius < interaction_radius:
            partial_start = interaction_radius - target_radius
        partial_start = max(0.0, partial_start)
        partial_end = min(radius, target_radius + interaction_radius)

        intervals: list[tuple[float, float]] = []
        if full_end > 1e-14:
            intervals.append((0.0, full_end))
        if partial_end - partial_start > 1e-14:
            intervals.append((partial_start, partial_end))

        for lower, upper in intervals:
            rq = 0.5 * (upper - lower) * base_x + 0.5 * (upper + lower)
            wr = 0.5 * (upper - lower) * base_w
            xq = 2.0 * rq / radius - 1.0
            interpolation = chebyshev_barycentric_matrix(x_nodes, xq)
            kernel_zero = top_hat_angular_kernel(
                np.array([target_radius]), rq, 0, interaction_radius
            )[0]
            kernel_ell = top_hat_angular_kernel(
                np.array([target_radius]), rq, angular_index, interaction_radius
            )[0]
            radial_weight = wr * rq
            accessible_area[row] += np.dot(kernel_zero, radial_weight)
            raw_operator[row] += (kernel_ell * radial_weight) @ interpolation

    convolution = raw_operator / accessible_area[:, None]
    return convolution, accessible_area, r, dr, d2r


def angular_convolution(spec: DiskSpec, ell: int):
    return disk_convolution_cached(
        spec.radius,
        spec.interaction_radius,
        abs(int(ell)),
        spec.n_radial,
        spec.quadrature_factor,
    )


def laplacian_block(r: np.ndarray, dr: np.ndarray, d2r: np.ndarray, ell: int) -> np.ndarray:
    inverse_r = np.zeros_like(r)
    inverse_r[:-1] = 1.0 / r[:-1]
    # The origin row is removed by a regularity tau condition.
    return d2r + np.diag(inverse_r) @ dr - np.diag((ell * inverse_r) ** 2)


def ladder_block(r: np.ndarray, dr: np.ndarray, ell: int, direction: str) -> np.ndarray:
    inverse_r = np.zeros_like(r)
    inverse_r[:-1] = 1.0 / r[:-1]
    if direction == "up":
        return dr - np.diag(ell * inverse_r)
    if direction == "down":
        return dr + np.diag(ell * inverse_r)
    raise ValueError(direction)


def build_tau_problem(spec: DiskSpec) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build A u = sigma B u for one azimuthal block."""
    n = spec.n_radial
    plus_ell = spec.m + 1
    minus_ell = spec.m - 1
    c_plus, area, r, dr, d2r = angular_convolution(spec, plus_ell)
    c_minus, _, _, _, _ = angular_convolution(spec, minus_ell)

    d0_value = 2.0 * spec.omega - 2.0 * spec.coupling * math.sin(spec.alpha)
    if abs(d0_value) < 1e-12:
        raise ValueError("second-harmonic closure is singular")

    identity = np.eye(n)
    a_prefactor = 0.5 * spec.coupling * math.cos(spec.alpha)
    a_plus = a_prefactor * c_plus
    a_minus = a_prefactor * c_minus

    lap_plus = laplacian_block(r, dr, d2r, plus_ell)
    lap_minus = laplacian_block(r, dr, d2r, minus_ell)
    local_rotation = -spec.omega + spec.coupling * math.sin(spec.alpha)
    convolution_rotation = -0.5 * spec.coupling * math.sin(spec.alpha)
    b_plus = (
        local_rotation * identity
        + convolution_rotation * c_plus
        - (spec.speed**2 / (4.0 * d0_value)) * lap_plus
    )
    b_minus = (
        local_rotation * identity
        + convolution_rotation * c_minus
        - (spec.speed**2 / (4.0 * d0_value)) * lap_minus
    )

    matrix = np.zeros((3 * n, 3 * n), dtype=np.complex128)
    mass = np.eye(3 * n, dtype=np.complex128)
    rho_slice = slice(0, n)
    plus_slice = slice(n, 2 * n)
    minus_slice = slice(2 * n, 3 * n)

    # Continuity: div p = (partial_- p+ + partial_+ p-)/2.
    matrix[rho_slice, plus_slice] = -0.5 * spec.speed * ladder_block(
        r, dr, plus_ell, "down"
    )
    matrix[rho_slice, minus_slice] = -0.5 * spec.speed * ladder_block(
        r, dr, minus_ell, "up"
    )
    # Circular-polarization form of the two momentum equations.
    matrix[plus_slice, rho_slice] = -0.5 * spec.speed * ladder_block(
        r, dr, spec.m, "up"
    )
    matrix[minus_slice, rho_slice] = -0.5 * spec.speed * ladder_block(
        r, dr, spec.m, "down"
    )
    matrix[plus_slice, plus_slice] = a_plus - 1j * b_plus
    matrix[minus_slice, minus_slice] = a_minus + 1j * b_minus

    outer = 0
    origin = n - 1

    def replace_with_constraint(row: int, entries: list[tuple[slice, np.ndarray]]) -> None:
        matrix[row, :] = 0.0
        mass[row, :] = 0.0
        for block, values in entries:
            matrix[row, block] = values

    # Analytic regularity h_l(r)=O(r^|l|).  For l=0 use h'(0)=0;
    # otherwise exclude the singular solution with h(0)=0.
    for block_number, ell in enumerate((spec.m, plus_ell, minus_ell)):
        row = block_number * n + origin
        block = slice(block_number * n, (block_number + 1) * n)
        values = dr[origin].copy() if ell == 0 else np.eye(n)[origin]
        replace_with_constraint(row, [(block, values)])

    # Specular moment conditions at r=R:
    # P_- = -P_+ and P_+' + P_-' - 2m P_+/R = 0.
    wall_value = np.zeros(n)
    wall_value[outer] = 1.0
    replace_with_constraint(
        n + outer,
        [(plus_slice, wall_value), (minus_slice, wall_value)],
    )
    derivative_plus = dr[outer].copy()
    derivative_plus[outer] -= 2.0 * spec.m / spec.radius
    replace_with_constraint(
        2 * n + outer,
        [(plus_slice, derivative_plus), (minus_slice, dr[outer].copy())],
    )

    diagnostics = {
        "normalization_error": float(
            np.max(np.abs(angular_convolution(spec, 0)[0] @ np.ones(n) - 1.0))
        ),
        "accessible_area_center": float(area[-1]),
        "accessible_area_wall": float(area[0]),
    }
    return matrix, mass, diagnostics


def solve_block(spec: DiskSpec, spectral_cutoff: float = 300.0) -> pd.DataFrame:
    matrix, mass, diagnostics = build_tau_problem(spec)
    values, vectors = eig(matrix, mass, right=True, check_finite=False)
    finite = np.isfinite(values) & (np.abs(values) <= spectral_cutoff)
    values = values[finite]
    vectors = vectors[:, finite]
    matrix_norm = np.linalg.norm(matrix, ord=np.inf)
    mass_norm = np.linalg.norm(mass, ord=np.inf)
    records: list[dict[str, float | int]] = []
    for index, value in enumerate(values):
        vector = vectors[:, index]
        vector_norm = np.linalg.norm(vector)
        residual = np.linalg.norm(matrix @ vector - value * (mass @ vector))
        residual /= max(
            (matrix_norm + abs(value) * mass_norm) * vector_norm,
            np.finfo(float).tiny,
        )
        records.append(
            {
                "diameter": spec.diameter,
                "coupling": spec.coupling,
                "m": spec.m,
                "n_radial": spec.n_radial,
                "quadrature_factor": spec.quadrature_factor,
                "real": float(value.real),
                "imag": float(value.imag),
                "abs_sigma": float(abs(value)),
                "residual": float(residual),
                **diagnostics,
            }
        )
    return pd.DataFrame.from_records(records)


def nearest_convergence(low: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    """Match each low-resolution eigenvalue to its nearest high-resolution value."""
    if low.empty or high.empty:
        return pd.DataFrame()
    low_values = low["real"].to_numpy() + 1j * low["imag"].to_numpy()
    high_values = high["real"].to_numpy() + 1j * high["imag"].to_numpy()
    distance = np.abs(low_values[:, None] - high_values[None, :])
    nearest = np.argmin(distance, axis=1)
    result = low.copy()
    result["matched_real_high"] = high_values[nearest].real
    result["matched_imag_high"] = high_values[nearest].imag
    result["resolution_shift"] = distance[np.arange(len(low_values)), nearest]
    return result


def run_scan(
    diameters: tuple[float, ...],
    couplings: tuple[float, ...],
    m_values: tuple[int, ...],
    resolutions: tuple[int, ...],
    quadrature_factor: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_spectra: list[pd.DataFrame] = []
    total = len(diameters) * len(couplings) * len(m_values) * len(resolutions)
    completed = 0
    for diameter in diameters:
        for coupling in couplings:
            for m in m_values:
                for n_radial in resolutions:
                    spec = DiskSpec(
                        diameter=diameter,
                        coupling=coupling,
                        m=m,
                        n_radial=n_radial,
                        quadrature_factor=quadrature_factor,
                    )
                    spectrum = solve_block(spec)
                    all_spectra.append(spectrum)
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"solved {completed}/{total}", flush=True)
    full = pd.concat(all_spectra, ignore_index=True)
    full.to_csv(OUTPUT_DIR / "Disk_Block_Eigenvalues.csv", index=False)

    summary_records: list[dict[str, float | int]] = []
    convergence_records: list[pd.DataFrame] = []
    top_n = max(resolutions)
    previous_n = sorted(resolutions)[-2] if len(resolutions) > 1 else top_n
    group_columns = ["diameter", "coupling", "m"]
    for keys, group in full.groupby(group_columns, sort=True):
        high = group[group.n_radial == top_n].copy()
        low = group[group.n_radial == previous_n].copy()
        matched = nearest_convergence(low, high)
        if not matched.empty:
            matched["n_radial_high"] = top_n
            convergence_records.append(matched)

        # A conservative converged-mode filter.  A physical growing mode must
        # survive refinement; merely taking max Re at one resolution is unsafe.
        stable = matched[
            (matched.resolution_shift < 2.5e-2)
            & (matched.residual < 1e-8)
            & (matched.abs_sigma < 100.0)
        ]
        if stable.empty:
            max_real = np.nan
            selected_imag = np.nan
            shift = np.nan
            stable_count = 0
        else:
            selected = stable.loc[stable.real.idxmax()]
            max_real = float(selected.matched_real_high)
            selected_imag = float(selected.matched_imag_high)
            shift = float(selected.resolution_shift)
            stable_count = len(stable)
        diameter, coupling, m = keys
        summary_records.append(
            {
                "diameter": diameter,
                "coupling": coupling,
                "m": int(m),
                "n_radial_low": previous_n,
                "n_radial_high": top_n,
                "stable_mode_count": stable_count,
                "max_converged_real": max_real,
                "selected_imag": selected_imag,
                "selected_resolution_shift": shift,
                "max_raw_real_high": float(high.real.max()),
                "normalization_error": float(high.normalization_error.max()),
                "accessible_area_center": float(high.accessible_area_center.iloc[0]),
                "accessible_area_wall": float(high.accessible_area_wall.iloc[0]),
            }
        )

    summary = pd.DataFrame.from_records(summary_records)
    summary.to_csv(OUTPUT_DIR / "Disk_Block_Convergence_Summary.csv", index=False)
    if convergence_records:
        pd.concat(convergence_records, ignore_index=True).to_csv(
            OUTPUT_DIR / "Disk_Block_Resolution_Matches.csv", index=False
        )

    winners = (
        summary.sort_values(
            ["diameter", "coupling", "max_converged_real"],
            ascending=[True, True, False],
        )
        .groupby(["diameter", "coupling"], as_index=False)
        .first()
    )
    winners.to_csv(OUTPUT_DIR / "Disk_Block_Selected_Modes.csv", index=False)
    configuration = {
        "diameters": diameters,
        "couplings": couplings,
        "m_values": m_values,
        "resolutions": resolutions,
        "quadrature_factor": quadrature_factor,
        "speed": 3.0,
        "interaction_radius": 1.0,
        "omega": 0.0,
        "alpha_over_pi": 0.5,
        "spectral_cutoff": 300.0,
        "converged_match_tolerance": 0.025,
    }
    (OUTPUT_DIR / "Disk_Block_Configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(winners.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small diagnostic scan")
    parser.add_argument("--quadrature-factor", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        run_scan(
            diameters=(3.3,),
            couplings=(20.75,),
            m_values=(8, 9, 10, 11),
            resolutions=(24, 32),
            quadrature_factor=args.quadrature_factor,
        )
    else:
        run_scan(
            diameters=(3.3, 4.58),
            couplings=(8.0, 12.0, 20.75, 40.0),
            m_values=tuple(range(5, 19)),
            resolutions=(32, 44, 56),
            quadrature_factor=args.quadrature_factor,
        )


if __name__ == "__main__":
    main()
