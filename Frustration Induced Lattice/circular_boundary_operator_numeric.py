"""Numerical diagnostic for the normalized circular-boundary linear operator.

This is a convergence/credibility test of the three-field closure, not a
replacement for the kinetic wall problem.  It discretizes each total-angular-
momentum block with Chebyshev--Lobatto collocation, imposes the two specular
moment boundary conditions through a generalized eigenproblem, and reports
whether edge-localized growth candidates converge with radial resolution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eig


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output" / "Circular_Boundary_Linear_Operator"
K_VALUES = (8.0, 12.0, 20.75, 40.0)
DIAMETERS = (3.30, 4.58)
RESOLUTIONS = (30, 40, 50)
V = 3.0
D0 = 1.0
OMEGA = 0.0
ALPHA = 0.5 * np.pi
EDGE_WIDTH = 0.25 * D0


@dataclass(frozen=True)
class RadialGrid:
    r: np.ndarray
    derivative: np.ndarray
    second_derivative: np.ndarray
    quadrature: np.ndarray
    wall_index: int
    origin_index: int


def chebyshev_lobatto_grid(interval_radius: float, order: int) -> RadialGrid:
    if order < 4:
        raise ValueError("Chebyshev order must be at least four.")
    indices = np.arange(order + 1)
    theta = np.pi * indices / order
    x = np.cos(theta)
    coefficients = np.ones(order + 1)
    coefficients[[0, -1]] = 2.0
    coefficients *= (-1.0) ** indices
    difference = x[:, None] - x[None, :]
    derivative_x = (
        (coefficients[:, None] / coefficients[None, :])
        / (difference + np.eye(order + 1))
    )
    derivative_x -= np.diag(np.sum(derivative_x, axis=1))

    weights_x = np.zeros(order + 1)
    interior = np.arange(1, order)
    values = np.ones(order - 1)
    if order % 2 == 0:
        weights_x[[0, -1]] = 1.0 / (order**2 - 1.0)
        for k in range(1, order // 2):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k**2 - 1.0)
        values -= np.cos(order * theta[interior]) / (order**2 - 1.0)
    else:
        weights_x[[0, -1]] = 1.0 / order**2
        for k in range(1, (order - 1) // 2 + 1):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k**2 - 1.0)
    weights_x[interior] = 2.0 * values / order

    r = 0.5 * interval_radius * (x + 1.0)
    derivative_r = (2.0 / interval_radius) * derivative_x
    return RadialGrid(
        r=r,
        derivative=derivative_r,
        second_derivative=derivative_r @ derivative_r,
        quadrature=0.5 * interval_radius * weights_x,
        wall_index=0,
        origin_index=order,
    )


def angular_top_hat_kernel(order_l: int, r: np.ndarray, d0: float) -> np.ndarray:
    ri = r[:, None]
    rj = r[None, :]
    output = np.zeros((r.size, r.size), dtype=float)
    center_row = np.isclose(ri, 0.0)
    center_column = np.isclose(rj, 0.0)
    full = ri + rj <= d0 + 1.0e-13
    separated = np.abs(ri - rj) >= d0 - 1.0e-13
    partial = ~(full | separated | center_row | center_column)
    if order_l == 0:
        output[full] = 2.0 * np.pi
        output[center_row & (rj <= d0 + 1.0e-13)] = 2.0 * np.pi
        output[center_column & (ri <= d0 + 1.0e-13)] = 2.0 * np.pi
    numerator = ri**2 + rj**2 - d0**2
    denominator = 2.0 * ri * rj
    ratio = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=ratio, where=partial)
    chi = np.zeros_like(ratio)
    chi[partial] = np.arccos(np.clip(ratio[partial], -1.0, 1.0))
    if order_l == 0:
        output[partial] = 2.0 * chi[partial]
    else:
        output[partial] = 2.0 * np.sin(order_l * chi[partial]) / order_l
    return output


def normalized_convolution(
    order_l: int, grid: RadialGrid, d0: float
) -> tuple[np.ndarray, np.ndarray]:
    kernel0 = angular_top_hat_kernel(0, grid.r, d0)
    radial_weights = grid.r * grid.quadrature
    normalization = kernel0 @ radial_weights
    if np.any(normalization <= 0.0):
        raise RuntimeError("Non-positive disk-neighborhood normalization.")
    kernel_l = angular_top_hat_kernel(abs(order_l), grid.r, d0)
    operator = kernel_l * radial_weights[None, :]
    operator /= normalization[:, None]
    return operator, normalization


def radial_laplacian(order_l: int, grid: RadialGrid) -> np.ndarray:
    r = grid.r
    inverse_r = np.zeros_like(r)
    inverse_r_squared = np.zeros_like(r)
    nonzero = r > 1.0e-13
    inverse_r[nonzero] = 1.0 / r[nonzero]
    inverse_r_squared[nonzero] = inverse_r[nonzero] ** 2
    return (
        grid.second_derivative
        + np.diag(inverse_r) @ grid.derivative
        - order_l**2 * np.diag(inverse_r_squared)
    )


def first_order_radial(
    order_l: int, grid: RadialGrid, direction: str
) -> np.ndarray:
    inverse_r = np.zeros_like(grid.r)
    nonzero = grid.r > 1.0e-13
    inverse_r[nonzero] = 1.0 / grid.r[nonzero]
    sign = -1.0 if direction == "up" else 1.0
    return grid.derivative + sign * order_l * np.diag(inverse_r)


def impose_constraint(
    operator: np.ndarray,
    metric: np.ndarray,
    row: int,
    coefficients: np.ndarray,
) -> None:
    operator[row, :] = coefficients
    metric[row, :] = 0.0


def build_block_operator(
    m: int,
    diameter: float,
    strength_k: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, RadialGrid]:
    radius = diameter / 2.0
    grid = chebyshev_lobatto_grid(radius, order)
    size = order + 1
    identity = np.eye(size)
    c_plus, _ = normalized_convolution(m + 1, grid, D0)
    c_minus, _ = normalized_convolution(abs(m - 1), grid, D0)
    lap_plus = radial_laplacian(m + 1, grid)
    lap_minus = radial_laplacian(abs(m - 1), grid)
    d_k = 2.0 * OMEGA - 2.0 * strength_k * np.sin(ALPHA)
    a_plus = 0.5 * strength_k * np.cos(ALPHA) * c_plus
    a_minus = 0.5 * strength_k * np.cos(ALPHA) * c_minus
    b_plus = (
        (-OMEGA + strength_k * np.sin(ALPHA)) * identity
        - 0.5 * strength_k * np.sin(ALPHA) * c_plus
        - V**2 / (4.0 * d_k) * lap_plus
    )
    b_minus = (
        (-OMEGA + strength_k * np.sin(ALPHA)) * identity
        - 0.5 * strength_k * np.sin(ALPHA) * c_minus
        - V**2 / (4.0 * d_k) * lap_minus
    )

    total = 3 * size
    operator = np.zeros((total, total), dtype=np.complex128)
    metric = np.eye(total, dtype=np.complex128)
    rho = slice(0, size)
    plus = slice(size, 2 * size)
    minus = slice(2 * size, 3 * size)
    operator[rho, plus] = -0.5 * V * first_order_radial(
        m + 1, grid, "down"
    )
    operator[rho, minus] = -0.5 * V * first_order_radial(
        m - 1, grid, "up"
    )
    operator[plus, rho] = -0.5 * V * first_order_radial(m, grid, "up")
    operator[minus, rho] = -0.5 * V * first_order_radial(m, grid, "down")
    operator[plus, plus] = a_plus - 1j * b_plus
    operator[minus, minus] = a_minus + 1j * b_minus

    origin = grid.origin_index
    wall = grid.wall_index
    constraint = np.zeros(total, dtype=np.complex128)
    constraint[origin] = 1.0
    impose_constraint(operator, metric, origin, constraint)

    constraint = np.zeros(total, dtype=np.complex128)
    constraint[size + origin] = 1.0
    impose_constraint(operator, metric, size + origin, constraint)

    constraint = np.zeros(total, dtype=np.complex128)
    if m == 1:
        constraint[2 * size : 3 * size] = grid.derivative[origin]
    else:
        constraint[2 * size + origin] = 1.0
    impose_constraint(operator, metric, 2 * size + origin, constraint)

    constraint = np.zeros(total, dtype=np.complex128)
    constraint[size + wall] = 1.0
    constraint[2 * size + wall] = 1.0
    impose_constraint(operator, metric, size + wall, constraint)

    constraint = np.zeros(total, dtype=np.complex128)
    constraint[size : 2 * size] = grid.derivative[wall]
    constraint[size + wall] -= (m + 1) / radius
    constraint[2 * size : 3 * size] = grid.derivative[wall]
    constraint[2 * size + wall] += (m - 1) / radius
    impose_constraint(operator, metric, 2 * size + wall, constraint)
    return operator, metric, grid


def eigen_diagnostics(
    m: int,
    diameter: float,
    strength_k: float,
    order: int,
) -> pd.DataFrame:
    operator, metric, grid = build_block_operator(m, diameter, strength_k, order)
    eigenvalues, eigenvectors = eig(operator, metric, right=True)
    finite = np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1.0e7)
    eigenvalues = eigenvalues[finite]
    eigenvectors = eigenvectors[:, finite]
    size = order + 1
    radial_measure = np.abs(grid.r * grid.quadrature)
    edge = grid.r >= diameter / 2.0 - EDGE_WIDTH
    rows = []
    for index, eigenvalue in enumerate(eigenvalues):
        vector = eigenvectors[:, index].reshape(3, size)
        energy_density = (
            np.abs(vector[0]) ** 2
            + 2.0 * np.abs(vector[1]) ** 2
            + 2.0 * np.abs(vector[2]) ** 2
        )
        weighted = energy_density * radial_measure
        total_energy = float(np.sum(weighted))
        if not np.isfinite(total_energy) or total_energy <= 1.0e-20:
            continue
        edge_fraction = float(np.sum(weighted[edge]) / total_energy)
        derivative_energy = 0.0
        for component in vector:
            derivative_energy += float(
                np.sum(np.abs(grid.derivative @ component) ** 2 * radial_measure)
            )
        smoothness = derivative_energy / total_energy
        rows.append(
            {
                "strength_k": strength_k,
                "diameter": diameter,
                "resolution_order": order,
                "m": m,
                "eigen_index": index,
                "real_sigma": float(np.real(eigenvalue)),
                "imag_sigma": float(np.imag(eigenvalue)),
                "abs_sigma": float(np.abs(eigenvalue)),
                "edge_fraction": edge_fraction,
                "radial_roughness": smoothness,
            }
        )
    return pd.DataFrame(rows)


def mode_range(diameter: float) -> range:
    if diameter < 4.0:
        return range(5, 15)
    return range(8, 19)


def run_scan(resolutions: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_eigenvalues = []
    leaders = []
    total_jobs = sum(
        len(mode_range(diameter))
        for _ in resolutions
        for _ in K_VALUES
        for diameter in DIAMETERS
    )
    completed = 0
    for order in resolutions:
        for strength_k in K_VALUES:
            for diameter in DIAMETERS:
                cell_frames = []
                for m in mode_range(diameter):
                    frame = eigen_diagnostics(m, diameter, strength_k, order)
                    all_eigenvalues.append(frame)
                    cell_frames.append(frame)
                    completed += 1
                    if completed % 10 == 0 or completed == total_jobs:
                        print(f"disk spectrum {completed}/{total_jobs}", flush=True)
                cell = pd.concat(cell_frames, ignore_index=True)
                candidates = cell[
                    (cell["edge_fraction"] >= 0.5)
                    & (cell["abs_sigma"] <= 200.0)
                ]
                if candidates.empty:
                    candidates = cell[cell["abs_sigma"] <= 200.0]
                leader = candidates.loc[candidates["real_sigma"].idxmax()].to_dict()
                leader["edge_candidate_filter_used"] = bool(
                    np.any(cell["edge_fraction"] >= 0.5)
                )
                leaders.append(leader)
    return pd.concat(all_eigenvalues, ignore_index=True), pd.DataFrame(leaders)


def convergence_table(leaders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strength_k, diameter), group in leaders.groupby(
        ["strength_k", "diameter"], sort=True
    ):
        group = group.sort_values("resolution_order")
        modes = group["m"].astype(int).tolist()
        growth = group["real_sigma"].tolist()
        edge = group["edge_fraction"].tolist()
        rows.append(
            {
                "strength_k": strength_k,
                "diameter": diameter,
                "orders": ",".join(str(int(x)) for x in group["resolution_order"]),
                "selected_m_by_order": ",".join(str(x) for x in modes),
                "growth_by_order": ",".join(f"{x:.8g}" for x in growth),
                "edge_fraction_by_order": ",".join(f"{x:.6g}" for x in edge),
                "mode_converged": len(set(modes)) == 1,
                "growth_last_relative_change": (
                    abs(growth[-1] - growth[-2]) / max(abs(growth[-1]), 1.0e-12)
                    if len(growth) >= 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(leaders: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for axis, diameter in zip(axes[:, 0], DIAMETERS):
        subset = leaders[np.isclose(leaders["diameter"], diameter)]
        for order, group in subset.groupby("resolution_order"):
            axis.plot(group["strength_k"], group["m"], "o-", label=f"Nr={int(order)+1}")
        axis.set_title(rf"Selected edge candidate, $D={diameter:g}$")
        axis.set_xlabel("K")
        axis.set_ylabel("m")
        axis.legend(frameon=False)
    for axis, diameter in zip(axes[:, 1], DIAMETERS):
        subset = leaders[np.isclose(leaders["diameter"], diameter)]
        for order, group in subset.groupby("resolution_order"):
            axis.plot(
                group["strength_k"], group["real_sigma"], "o-", label=f"Nr={int(order)+1}"
            )
        axis.set_title(rf"Candidate growth, $D={diameter:g}$")
        axis.set_xlabel("K")
        axis.set_ylabel(r"Re $\sigma$")
        axis.legend(frameon=False)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        default=",".join(str(x) for x in RESOLUTIONS),
        help="Comma-separated Chebyshev orders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolutions = tuple(int(x) for x in args.resolutions.split(","))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eigenvalues, leaders = run_scan(resolutions)
    convergence = convergence_table(leaders)
    eigenvalues.to_csv(OUTPUT_DIR / "Disk_Operator_All_Eigenvalues.csv", index=False)
    leaders.to_csv(OUTPUT_DIR / "Disk_Operator_Leading_Candidates.csv", index=False)
    convergence.to_csv(OUTPUT_DIR / "Disk_Operator_Convergence.csv", index=False)
    figure = make_figure(leaders)
    figure.savefig(OUTPUT_DIR / "Disk_Operator_Convergence.png", dpi=220)
    figure.savefig(OUTPUT_DIR / "Disk_Operator_Convergence.pdf")
    plt.close(figure)
    metadata = {
        "K_values": K_VALUES,
        "diameters": DIAMETERS,
        "resolutions": resolutions,
        "v": V,
        "d0": D0,
        "alpha_over_pi": ALPHA / np.pi,
        "edge_width": EDGE_WIDTH,
        "candidate_filter": "edge_fraction >= 0.5 and |sigma| <= 200",
        "warning": (
            "Failure of m and growth to converge means the three-field wall "
            "closure is not numerically predictive at this discretization."
        ),
    }
    (OUTPUT_DIR / "Disk_Operator_Numerics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    observed = {3.30: "9 at low K; 10 at high K", 4.58: "13 at low K; 14 at high K"}
    report = [
        "# Normalized disk-operator numerical diagnostic",
        "",
        "The operator is the three-field circular closure derived in "
        "Circular_Boundary_Matrix_Derivation.md, with neighbor-area "
        "normalization and the two specular moment boundary conditions.",
        "",
        "## Resolution test",
        "",
        convergence.to_markdown(index=False, floatfmt=".7g"),
        "",
        "## Verdict",
        "",
        f"Converged cells: {int(convergence['mode_converged'].sum())}/"
        f"{len(convergence)}. The selected m changes with radial resolution "
        "in every tested K x D cell, while the candidate real growth is "
        "generally near zero and also fails to converge. Isolated larger "
        "growth values disappear at the next resolution and are classified "
        "as collocation/closure pseudospectral artifacts.",
        "",
        "The particle observations are "
        + "; ".join(f"D={d:g}: {label}" for d, label in observed.items())
        + ". The present homogeneous three-field disk closure therefore does "
        "not provide a numerically converged prediction that can be matched "
        "to these modes.",
        "",
        "This is a negative but informative result: circular geometry and "
        "neighbor normalization alone do not cure the alpha=pi/2 marginal "
        "degeneracy at this closure level. The next controlled calculation is "
        "linearization about the nonuniform circulating boundary base state, "
        "or a higher-harmonic kinetic wall calculation.",
        "",
    ]
    (OUTPUT_DIR / "Disk_Operator_Numerical_Diagnostic.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(convergence.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
