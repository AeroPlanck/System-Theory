"""Prototype spectrum of the normalized circular-boundary hydrodynamic operator.

This file does not modify or import-write any reference source.  It constructs
the angular-momentum blocks derived from the bulk density--polarization closure,
but keeps the particle-model neighbourhood normalization near a circular wall.

The radial discretization is Chebyshev--Lobatto collocation.  The nonlocal
top-hat kernel is integrated by a separate Gauss--Legendre rule.  Regularity at
r=0 and the n=1,n=2 specular-reflection moment conditions at r=R are imposed by
exact linear-constraint elimination (not penalty rows).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.linalg import eig
from scipy.special import roots_legendre


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "Normalized_Disk_Operator_Prototype"


@dataclass(frozen=True)
class Case:
    diameter: float
    coupling: float
    alpha_over_pi: float = 0.5
    speed: float = 3.0
    omega: float = 0.0
    interaction_radius: float = 1.0


def chebyshev_lobatto(n: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Ascending Chebyshev--Lobatto nodes on [0,R] and first derivative."""
    if n < 8:
        raise ValueError("n must be at least 8")
    j = np.arange(n)
    x = -np.cos(np.pi * j / (n - 1))
    weights = (-1.0) ** j
    weights[[0, -1]] *= 0.5
    dx = x[:, None] - x[None, :]
    derivative = np.empty((n, n), dtype=float)
    mask = ~np.eye(n, dtype=bool)
    ratio = weights[None, :] / weights[:, None]
    off_diagonal = np.zeros_like(dx)
    np.divide(ratio, dx, out=off_diagonal, where=mask)
    derivative[mask] = off_diagonal[mask]
    derivative[np.diag_indices(n)] = 0.0
    derivative[np.diag_indices(n)] = -np.sum(derivative, axis=1)
    r = 0.5 * radius * (x + 1.0)
    derivative *= 2.0 / radius
    # A direct polynomial check catches transposition/sign mistakes.
    if not np.allclose(derivative @ r, 1.0, atol=2e-11, rtol=2e-11):
        raise RuntimeError("Chebyshev differentiation check failed")
    return r, derivative


def barycentric_interpolation(nodes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolation from Chebyshev--Lobatto nodes to arbitrary targets."""
    n = nodes.size
    j = np.arange(n)
    weights = (-1.0) ** j
    weights[[0, -1]] *= 0.5
    delta = targets[:, None] - nodes[None, :]
    matrix = np.empty_like(delta)
    for row in range(targets.size):
        exact = np.flatnonzero(np.abs(delta[row]) < 5e-15)
        if exact.size:
            matrix[row] = 0.0
            matrix[row, exact[0]] = 1.0
        else:
            values = weights / delta[row]
            matrix[row] = values / np.sum(values)
    return matrix


def angular_top_hat_kernel(
    r: np.ndarray, rp: np.ndarray, order: int, interaction_radius: float
) -> np.ndarray:
    """Angular Fourier coefficient of 1_{|x-x'| <= d0}."""
    rr = np.asarray(r, dtype=float)[:, None]
    ss = np.asarray(rp, dtype=float)[None, :]
    d0 = float(interaction_radius)
    result = np.zeros((rr.shape[0], ss.shape[1]), dtype=float)

    full = d0 >= rr + ss - 2e-14
    separated = d0 <= np.abs(rr - ss) + 2e-14
    partial = ~(full | separated)
    if order == 0:
        result[full] = 2.0 * np.pi
    # Every nonzero Fourier coefficient integrates to zero on a full circle.
    if np.any(partial):
        denominator = 2.0 * rr * ss
        cosine = np.empty_like(denominator)
        cosine[partial] = (
            (rr * rr + ss * ss - d0 * d0)[partial] / denominator[partial]
        )
        chi = np.arccos(np.clip(cosine[partial], -1.0, 1.0))
        if order == 0:
            result[partial] = 2.0 * chi
        else:
            result[partial] = 2.0 * np.sin(abs(order) * chi) / abs(order)
    return result


@dataclass
class RadialData:
    r: np.ndarray
    derivative: np.ndarray
    laplacians: dict[int, np.ndarray]
    convolutions: dict[int, np.ndarray]
    neighbor_area: np.ndarray


def radial_data(
    radius: float,
    n: int,
    orders: Iterable[int],
    interaction_radius: float,
    quadrature_factor: int = 6,
) -> RadialData:
    r, derivative = chebyshev_lobatto(n, radius)
    nq = max(180, quadrature_factor * n)
    xq, wq = roots_legendre(nq)
    rq = 0.5 * radius * (xq + 1.0)
    wq = 0.5 * radius * wq
    interpolation = barycentric_interpolation(r, rq)
    radial_weights = wq * rq

    kernel_zero = angular_top_hat_kernel(r, rq, 0, interaction_radius)
    neighbor_area = kernel_zero @ radial_weights
    if np.min(neighbor_area) <= 0.0:
        raise RuntimeError("non-positive local interaction area")

    convolutions: dict[int, np.ndarray] = {}
    laplacians: dict[int, np.ndarray] = {}
    identity = np.eye(n)
    safe_inverse_r = np.zeros_like(r)
    safe_inverse_r[1:] = 1.0 / r[1:]
    for raw_order in sorted(set(int(value) for value in orders)):
        order = abs(raw_order)
        kernel = angular_top_hat_kernel(r, rq, order, interaction_radius)
        convolution = (kernel * radial_weights[None, :]) @ interpolation
        convolutions[raw_order] = convolution / neighbor_area[:, None]
        laplacians[raw_order] = (
            derivative @ derivative
            + np.diag(safe_inverse_r) @ derivative
            - (raw_order * raw_order) * np.diag(safe_inverse_r**2) @ identity
        )
    return RadialData(r, derivative, laplacians, convolutions, neighbor_area)


def component_slice(component: int, n: int) -> slice:
    return slice(component * n, (component + 1) * n)


@dataclass
class Block:
    matrix: np.ndarray
    lift: np.ndarray
    free_indices: np.ndarray
    full_operator: np.ndarray
    constraints: np.ndarray
    r: np.ndarray


def build_block(case: Case, mode: int, n: int, quadrature_factor: int = 6) -> Block:
    radius = case.diameter / 2.0
    lp = mode + 1
    lm = mode - 1
    data = radial_data(
        radius,
        n,
        (lp, lm),
        case.interaction_radius,
        quadrature_factor=quadrature_factor,
    )
    r = data.r
    derivative = data.derivative
    identity = np.eye(n)
    alpha = case.alpha_over_pi * np.pi
    denominator = 2.0 * case.omega - 2.0 * case.coupling * np.sin(alpha)
    if abs(denominator) < 1e-12:
        raise ValueError("second-harmonic closure denominator is singular")
    beta = case.speed * case.speed / (4.0 * denominator)

    def operators(order: int) -> tuple[np.ndarray, np.ndarray]:
        convolution = data.convolutions[order]
        laplacian = data.laplacians[order]
        aa = 0.5 * case.coupling * np.cos(alpha) * convolution
        bb = (
            (-case.omega + case.coupling * np.sin(alpha)) * identity
            - 0.5 * case.coupling * np.sin(alpha) * convolution
            - beta * laplacian
        )
        return aa, bb

    ap, bp = operators(lp)
    am, bm = operators(lm)
    safe_inverse_r = np.zeros_like(r)
    safe_inverse_r[1:] = 1.0 / r[1:]

    def up(order: int) -> np.ndarray:
        return derivative - order * np.diag(safe_inverse_r)

    def down(order: int) -> np.ndarray:
        return derivative + order * np.diag(safe_inverse_r)

    total = 3 * n
    operator = np.zeros((total, total), dtype=np.complex128)
    rho = component_slice(0, n)
    plus = component_slice(1, n)
    minus = component_slice(2, n)
    operator[rho, plus] = -0.5 * case.speed * down(lp)
    operator[rho, minus] = -0.5 * case.speed * up(lm)
    operator[plus, rho] = -0.5 * case.speed * up(mode)
    operator[minus, rho] = -0.5 * case.speed * down(mode)
    operator[plus, plus] = ap - 1j * bp
    operator[minus, minus] = am + 1j * bm

    constraints = np.zeros((5, total), dtype=np.complex128)
    # Analytic regularity at the disk center.
    angular_orders = (mode, lp, lm)
    for component, order in enumerate(angular_orders):
        block = component_slice(component, n)
        if order == 0:
            constraints[component, block] = derivative[0]
        else:
            constraints[component, component * n] = 1.0

    # n=1 specular moment: P_-(R) = -P_+(R).
    constraints[3, 2 * n - 1] = 1.0
    constraints[3, 3 * n - 1] = 1.0
    # n=2 specular moment:
    # [d_r+(m-1)/R]P_- = -[d_r-(m+1)/R]P_+.
    constraints[4, plus] = derivative[-1]
    constraints[4, 2 * n - 1] += -lp / radius
    constraints[4, minus] = derivative[-1]
    constraints[4, 3 * n - 1] += lm / radius

    pivots = np.array([0, n, 2 * n, 2 * n - 1, 3 * n - 1], dtype=int)
    free = np.setdiff1d(np.arange(total), pivots, assume_unique=True)
    cp = constraints[:, pivots]
    condition = np.linalg.cond(cp)
    if not np.isfinite(condition) or condition > 1e12:
        raise RuntimeError(f"ill-conditioned boundary elimination: {condition:g}")
    lift = np.zeros((total, free.size), dtype=np.complex128)
    lift[free, np.arange(free.size)] = 1.0
    lift[pivots] = -np.linalg.solve(cp, constraints[:, free])
    if np.linalg.norm(constraints @ lift, ord=np.inf) > 2e-9:
        raise RuntimeError("boundary constraint elimination failed")
    reduced = operator[free] @ lift
    return Block(reduced, lift, free, operator, constraints, r)


@dataclass
class EigenRecord:
    diameter: float
    coupling: float
    alpha_over_pi: float
    radial_points: int
    mode: int
    eigen_index: int
    real_part: float
    imaginary_part: float
    edge_weight: float
    radial_centroid_over_R: float
    constraint_residual: float
    equation_residual: float
    eigenvector_condition: float


def mode_spectrum(
    case: Case,
    mode: int,
    n: int,
    edge_width: float = 0.25,
    quadrature_factor: int = 6,
) -> list[EigenRecord]:
    block = build_block(case, mode, n, quadrature_factor=quadrature_factor)
    values, left, right = eig(block.matrix, left=True, right=True)
    radius = case.diameter / 2.0
    # Positive quadrature only for diagnostics; interpolate modal density to GL nodes.
    nq = max(240, 6 * n)
    xq, wq = roots_legendre(nq)
    rq = 0.5 * radius * (xq + 1.0)
    wq = 0.5 * radius * wq
    interp = barycentric_interpolation(block.r, rq)
    records: list[EigenRecord] = []
    scale = max(1.0, np.linalg.norm(block.matrix, ord=np.inf))
    for index, value in enumerate(values):
        reduced_vector = right[:, index]
        full = block.lift @ reduced_vector
        rho = interp @ full[0:n]
        plus = interp @ full[n : 2 * n]
        minus = interp @ full[2 * n : 3 * n]
        density = np.abs(rho) ** 2 + 0.5 * (
            np.abs(plus) ** 2 + np.abs(minus) ** 2
        )
        measure = wq * rq
        norm = float(np.sum(measure * density))
        if norm <= 1e-30:
            edge_weight = math.nan
            centroid = math.nan
        else:
            edge_weight = float(
                np.sum(measure[rq >= radius - edge_width] * density[rq >= radius - edge_width])
                / norm
            )
            centroid = float(np.sum(measure * rq * density) / norm / radius)
        constraint_residual = float(
            np.linalg.norm(block.constraints @ full) / max(np.linalg.norm(full), 1e-30)
        )
        equation_residual = float(
            np.linalg.norm(
                block.full_operator[block.free_indices] @ full
                - value * full[block.free_indices]
            )
            / (scale * max(np.linalg.norm(full[block.free_indices]), 1e-30))
        )
        overlap = np.vdot(left[:, index], reduced_vector)
        eigenvector_condition = float(
            np.linalg.norm(left[:, index])
            * np.linalg.norm(reduced_vector)
            / max(abs(overlap), 1e-300)
        )
        records.append(
            EigenRecord(
                diameter=case.diameter,
                coupling=case.coupling,
                alpha_over_pi=case.alpha_over_pi,
                radial_points=n,
                mode=mode,
                eigen_index=index,
                real_part=float(value.real),
                imaginary_part=float(value.imag),
                edge_weight=edge_weight,
                radial_centroid_over_R=centroid,
                constraint_residual=constraint_residual,
                equation_residual=equation_residual,
                eigenvector_condition=eigenvector_condition,
            )
        )
    return records


def select_candidate(
    records: list[EigenRecord], edge_threshold: float, frequency_cutoff: float
) -> EigenRecord | None:
    eligible = [
        row
        for row in records
        if np.isfinite(row.edge_weight)
        and row.edge_weight >= edge_threshold
        and abs(row.imaginary_part) <= frequency_cutoff
        and row.constraint_residual <= 1e-7
        and row.equation_residual <= 1e-7
    ]
    return max(eligible, key=lambda row: row.real_part) if eligible else None


def run(
    cases: list[Case],
    resolutions: list[int],
    modes: list[int],
    edge_threshold: float,
    quadrature_factor: int,
    frequency_cutoff: float = 100.0,
) -> tuple[list[EigenRecord], list[dict[str, object]]]:
    all_records: list[EigenRecord] = []
    summary: list[dict[str, object]] = []
    for case in cases:
        for n in resolutions:
            candidates: list[EigenRecord] = []
            for mode in modes:
                records = mode_spectrum(
                    case,
                    mode,
                    n,
                    quadrature_factor=quadrature_factor,
                )
                all_records.extend(records)
                candidate = select_candidate(records, edge_threshold, frequency_cutoff)
                if candidate is not None:
                    candidates.append(candidate)
            best = max(candidates, key=lambda row: row.real_part) if candidates else None
            summary.append(
                {
                    **asdict(case),
                    "radial_points": n,
                    "mode_min": min(modes),
                    "mode_max": max(modes),
                    "edge_threshold": edge_threshold,
                    "frequency_cutoff": frequency_cutoff,
                    "selected_mode": None if best is None else best.mode,
                    "max_edge_real_part": None if best is None else best.real_part,
                    "selected_imaginary_part": None if best is None else best.imaginary_part,
                    "selected_edge_weight": None if best is None else best.edge_weight,
                    "selected_centroid_over_R": None
                    if best is None
                    else best.radial_centroid_over_R,
                    "selected_eigenvector_condition": None
                    if best is None
                    else best.eigenvector_condition,
                    "maximum_real_part_any_mode": max(
                        (row.real_part for row in all_records if row.diameter == case.diameter
                         and row.coupling == case.coupling and row.radial_points == n),
                        default=math.nan,
                    ),
                }
            )
            print(
                f"D={case.diameter:g} K={case.coupling:g} Nr={n}: "
                + (
                    "no edge candidate"
                    if best is None
                    else f"m={best.mode} Re={best.real_part:+.6e} "
                    f"Im={best.imaginary_part:+.6e} W={best.edge_weight:.3f} "
                    f"cond={best.eigenvector_condition:.2e}"
                ),
                flush=True,
            )
    return all_records, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", nargs="+", type=int, default=[36, 48, 64, 80])
    parser.add_argument("--m-min", type=int, default=0)
    parser.add_argument("--m-max", type=int, default=22)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--quadrature-factor", type=int, default=6)
    parser.add_argument("--frequency-cutoff", type=float, default=100.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        cases = [Case(3.3, 20.75)]
        resolutions = [28, 36]
        modes = list(range(6, 14))
    else:
        cases = [
            Case(diameter, coupling)
            for diameter in (3.3, 4.58)
            for coupling in (8.0, 12.0, 20.75, 40.0)
        ]
        resolutions = args.resolutions
        modes = list(range(args.m_min, args.m_max + 1))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, summary = run(
        cases,
        resolutions,
        modes,
        args.edge_threshold,
        args.quadrature_factor,
        args.frequency_cutoff,
    )
    write_csv(OUTPUT / "Disk_Operator_Eigenvalues.csv", [asdict(row) for row in records])
    write_csv(OUTPUT / "Disk_Operator_Resolution_Summary.csv", summary)
    configuration = {
        "cases": [asdict(case) for case in cases],
        "resolutions": resolutions,
        "modes": modes,
        "edge_threshold": args.edge_threshold,
        "quadrature_factor": args.quadrature_factor,
        "frequency_cutoff": args.frequency_cutoff,
        "normalization": "local top-hat interaction area",
        "boundary_conditions": [
            "P_minus(R)=-P_plus(R)",
            "[d+(m-1)/R]P_minus=-[d-(m+1)/R]P_plus",
        ],
    }
    (OUTPUT / "Disk_Operator_Configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
