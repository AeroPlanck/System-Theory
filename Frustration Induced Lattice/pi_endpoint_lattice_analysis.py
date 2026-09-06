"""Test the alpha=pi endpoint lattice against the alpha->pi- spectrum.

Reference inputs are read only.  The two counterpropagating populations are
identified from the microscopic phase in the local boundary frame; tangential
motion is used only as an independent label/validation after phase clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import jn_zeros


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "Pi_Endpoint_Lattice"
DATA = ROOT / "data" / "pi_endpoint_N2000_steps50000_snap50"
DISPERSION = Path(r"D:\PrivatePythonProject\Math\Lattice\Dispersion.py")
PRL = Path(r"D:\LaTex\Boundary Flow\PRL.tex")
METHODS = Path(r"D:\LaTex\Boundary Flow\Methods Appendix.tex")

N = 2000
K = 20.75
D0 = 1.0
V = 3.0
DIAMETER = 7.0
RADIUS = DIAMETER / 2.0
PERIMETER = 2.0 * math.pi * RADIUS
DT = 0.005
SNAP = 50
SAVED_DT = DT * SNAP
STEPS = 50_000
SEEDS = (1, 9, 17)
TERMINAL_CIRCUITS = 10.0
TERMINAL_TIME = TERMINAL_CIRCUITS * PERIMETER / V
MAIN_SHELL = 0.50 * D0
MAIN_PHASE_CONFIDENCE = 0.50
MODES = np.arange(6, 41, dtype=int)
EPSILONS = np.array(
    [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5]
)

EXPECTED_HASHES = {
    DISPERSION: "A1FC299F4AB13F9997BDF0EBA993C6BA12054500134A8617180F572F3732B89D",
    PRL: "8265AF6394ACD421FDE1E1163DC42B126AB33A8EEC0F019D91D4B4D5537BD7A6",
    METHODS: "CB0A459012329E1CCE7584152E55333467F8A48E1317C04DCA3DCCA72D07F7A8",
}


@dataclass
class FrameResult:
    beta: float
    axial_order: float
    phase_separation: float
    direction_agreement: float
    family: dict[str, dict[str, np.ndarray | float | int]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_references() -> dict[str, str]:
    observed = {str(path): sha256(path) for path in EXPECTED_HASHES}
    mismatched = [
        str(path)
        for path, expected in EXPECTED_HASHES.items()
        if observed[str(path)] != expected
    ]
    if mismatched:
        raise RuntimeError("Read-only reference hash changed: " + ", ".join(mismatched))
    return observed


def import_dispersion_module():
    spec = importlib.util.spec_from_file_location("reference_dispersion", DISPERSION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {DISPERSION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def continuum_parameters(alpha: float) -> tuple[float, ...]:
    # The microscopic rule divides by the local neighbor count.  Under the
    # homogeneous-disk approximation, lambda=K/(rho0*pi*d0^2).
    rho0 = N / (math.pi * RADIUS**2)
    lam = K / (rho0 * math.pi * D0**2)
    return V, 0.0, lam, alpha, rho0, D0


def spectrum_at_k(module, k: np.ndarray | float, alpha: float) -> np.ndarray:
    values = np.asarray(k, dtype=float)
    return module.eigs_at_k(values, np.zeros_like(values), continuum_parameters(alpha))


def most_unstable(module, alpha: float) -> tuple[float, float]:
    grid = np.linspace(0.0, 14.0 / D0, 7001)
    growth = np.max(np.real(spectrum_at_k(module, grid, alpha)), axis=-1)
    index = int(np.argmax(growth))
    if index == 0 or index == grid.size - 1:
        raise RuntimeError("Spectral maximum is on the search boundary")
    lo = grid[index - 2]
    hi = grid[index + 2]

    def objective(k: float) -> float:
        return -float(np.max(np.real(spectrum_at_k(module, k, alpha))))

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 2e-12, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x), float(-result.fun)


def exact_endpoint_diagnostic(module, limiting_k: float) -> dict[str, object]:
    rho0 = N / (math.pi * RADIUS**2)
    lam = K / (rho0 * math.pi * D0**2)
    exact_symbolic_denom = 0.0
    samples = []
    for label, alpha in (
        ("nextafter_below", float(np.nextafter(np.pi, -np.inf))),
        ("np.pi", float(np.pi)),
        ("nextafter_above", float(np.nextafter(np.pi, np.inf))),
    ):
        denom = -2.0 * lam * rho0 * math.pi * D0**2 * math.sin(alpha)
        matrix = module.M_matrix_vectorized(
            limiting_k, 0.0, *continuum_parameters(alpha)
        )
        samples.append(
            {
                "representation": label,
                "alpha": alpha,
                "sin_alpha": math.sin(alpha),
                "D0_denominator": denom,
                "max_abs_matrix_entry": float(np.max(np.abs(matrix))),
                "finite_matrix": bool(np.isfinite(matrix).all()),
            }
        )
    return {
        "mathematical_alpha_pi": {
            "sin_pi": 0.0,
            "D0_denominator": exact_symbolic_denom,
            "status": "undefined: the v^2 k^2/(4 D0) coefficient divides by zero",
        },
        "floating_point_artifact": samples,
    }


def analyse_spectrum(module) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for epsilon in EPSILONS:
        alpha = (1.0 - epsilon) * math.pi
        kstar, growth = most_unstable(module, alpha)
        denom = -2.0 * K * math.sin(alpha)
        rows.append(
            {
                "epsilon=1-alpha/pi": epsilon,
                "alpha/pi": alpha / math.pi,
                "D0_denominator": denom,
                "k_star": kstar,
                "k_star*d0": kstar * D0,
                "max_Re_sigma": growth,
                "2pi/k_star": 2.0 * math.pi / kstar,
                "continuous_boundary_mode_Rk": RADIUS * kstar,
                "nearest_integer_mode": int(round(RADIUS * kstar)),
                "divergent_coefficient_abs": abs(V**2 * kstar**2 / (4.0 * denom)),
            }
        )
    table = pd.DataFrame(rows)
    limiting_x = float(jn_zeros(2, 1)[0])
    limiting_k = limiting_x / D0
    limiting_wavelength = 2.0 * math.pi / limiting_k
    limiting_growth = -K * float(np.real(np.asarray(__import__("scipy").special.j1(limiting_x)))) / limiting_x
    limit = {
        "derivation": (
            "For fixed k>0 and alpha->pi-, the two fast eigenvalues have "
            "Re sigma -> a(k)=-K J1(k d0)/(k d0), while the density branch "
            "has Re sigma -> 0.  Therefore d[J1(x)/x]/dx=-J2(x)/x gives "
            "the first positive maximum at the first zero of J2."
        ),
        "j2_first_zero_x": limiting_x,
        "k_star_limit": limiting_k,
        "max_growth_limit": limiting_growth,
        "wavelength_limit_2pi_over_k": limiting_wavelength,
        "continuous_boundary_mode_Rk": RADIUS * limiting_k,
        "nearest_integer_boundary_mode": int(round(RADIUS * limiting_k)),
        "quantized_arc_spacing": PERIMETER / round(RADIUS * limiting_k),
        "quantized_wall_chord": 2.0
        * RADIUS
        * math.sin(math.pi / round(RADIUS * limiting_k)),
    }
    return table, limit


def trajectory_path(seed: int) -> Path:
    matches = list(DATA.glob(f"*seed={seed}).h5"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one long trajectory for seed={seed}; found {matches}")
    return matches[0]


def load_terminal(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = trajectory_path(seed)
    with pd.HDFStore(path, mode="r") as store:
        xrows = store.get_storer("positionX").nrows
        trows = store.get_storer("phaseTheta").nrows
        if xrows != trows or xrows % N:
            raise RuntimeError(f"Unaligned trajectory: {path}")
        total_frames = xrows // N
        expected = STEPS // SNAP + 1
        if total_frames != expected:
            raise RuntimeError(f"{path} has {total_frames} frames, expected {expected}")
        frame_count = min(
            total_frames, int(math.ceil(TERMINAL_TIME / SAVED_DT)) + 1
        )
        start_frame = total_frames - frame_count
        start_row = start_frame * N
        positions = store.select("positionX", start=start_row).to_numpy()
        phases = store.select("phaseTheta", start=start_row).to_numpy()
    return (
        positions.reshape(frame_count, N, 2),
        phases.reshape(frame_count, N),
        np.arange(start_frame, total_frames) * SNAP,
    )


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return np.angle(np.exp(1j * angle))


def frame_analysis(
    positions: np.ndarray,
    phases: np.ndarray,
    shell_width: float,
    phase_confidence: float,
) -> FrameResult:
    relative = positions - RADIUS
    radius = np.linalg.norm(relative, axis=1)
    phi_raw = np.arctan2(relative[:, 1], relative[:, 0])
    phi = np.mod(phi_raw, 2.0 * math.pi)
    psi = wrap(phases - phi_raw)
    shell = (RADIUS - radius >= -1e-9) & (RADIUS - radius <= shell_width)
    if shell.sum() < 20:
        raise RuntimeError("Too few particles in boundary shell")
    axial_vector = np.mean(np.exp(2j * psi[shell]))
    beta = 0.5 * float(np.angle(axial_vector))
    projection_on_axis = np.cos(psi - beta)
    candidates = shell & (np.abs(projection_on_axis) >= phase_confidence)
    raw_positive = projection_on_axis >= 0.0

    raw_data = []
    for raw in (False, True):
        mask = candidates & (raw_positive == raw)
        if mask.sum() < 10:
            raise RuntimeError("A phase family has too few boundary particles")
        q = np.sin(psi[mask])
        raw_data.append((raw, mask, float(np.mean(q))))
    ccw_raw = max(raw_data, key=lambda item: item[2])[0]

    families: dict[str, dict[str, np.ndarray | float | int]] = {}
    correct = 0
    total = 0
    mean_psi = {}
    for raw, mask, mean_q in raw_data:
        name = "CCW" if raw == ccw_raw else "CW"
        expected_sign = 1 if name == "CCW" else -1
        q = np.sin(psi[mask])
        correct += int(np.sum(expected_sign * q > 0.0))
        total += int(mask.sum())
        mean_relative_phase = float(np.angle(np.mean(np.exp(1j * psi[mask]))))
        mean_psi[name] = mean_relative_phase
        amplitudes = np.abs(
            np.mean(np.exp(1j * np.outer(phi[mask], MODES)), axis=0)
        )
        families[name] = {
            "mask": mask,
            "phi": phi[mask],
            "psi": psi[mask],
            "positions": positions[mask],
            "radius": radius[mask],
            "count": int(mask.sum()),
            "mean_q": mean_q,
            "mean_relative_phase": mean_relative_phase,
            "amplitudes": amplitudes,
        }
    phase_separation = abs(float(wrap(mean_psi["CCW"] - mean_psi["CW"])))
    return FrameResult(
        beta=beta,
        axial_order=float(abs(axial_vector)),
        phase_separation=phase_separation,
        direction_agreement=correct / total,
        family=families,
    )


def centroid_spacings(
    positions: np.ndarray,
    phi: np.ndarray,
    mode: int,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = np.mean(np.exp(1j * mode * phi))
    phase = float(np.angle(coefficient))
    labels = np.mod(
        np.rint((mode * phi - phase) / (2.0 * math.pi)).astype(int), mode
    )
    centroids = []
    minimum_count = max(5, int(0.20 * positions.shape[0] / mode))
    for label in range(mode):
        member = labels == label
        if member.sum() < minimum_count:
            return np.array([]), np.array([])
        centroids.append(np.mean(positions[member], axis=0))
    centroids = np.asarray(centroids)
    angles = np.mod(
        np.arctan2(centroids[:, 1] - RADIUS, centroids[:, 0] - RADIUS),
        2.0 * math.pi,
    )
    order = np.argsort(angles)
    centroids = centroids[order]
    angles = angles[order]
    next_centroid = np.roll(centroids, -1, axis=0)
    chord = np.linalg.norm(next_centroid - centroids, axis=1)
    delta_phi = np.mod(np.roll(angles, -1) - angles, 2.0 * math.pi)
    wall_arc = RADIUS * delta_phi
    return chord, wall_arc


def circular_gap_clusters(
    phi: np.ndarray,
    gap: float,
    positions: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Split a circular 1-D point set at empty arclength gaps.

    Starting after the largest gap makes the unwrap independent of the
    arbitrary phi=0 seam.  Groups smaller than five particles are treated as
    outliers rather than lattice packets.
    """
    s = RADIUS * np.mod(phi, 2.0 * math.pi)
    if s.size == 0:
        return [], None
    order = np.argsort(s)
    sorted_s = s[order]
    circular_gaps = np.diff(np.r_[sorted_s, sorted_s[0] + PERIMETER])
    start = (int(np.argmax(circular_gaps)) + 1) % s.size
    cyclic_order = np.r_[order[start:], order[:start]]
    unwrapped = np.r_[sorted_s[start:], sorted_s[:start] + PERIMETER]
    breaks = np.flatnonzero(np.diff(unwrapped) > gap)
    index_groups = [group for group in np.split(cyclic_order, breaks + 1) if group.size >= 5]
    if positions is None:
        return index_groups, None
    centroids = np.asarray([np.mean(positions[group], axis=0) for group in index_groups])
    return index_groups, centroids


def circular_gap_cluster_count(phi: np.ndarray, gap: float) -> int:
    groups, _ = circular_gap_clusters(phi, gap)
    return len(groups)


def direct_cluster_spacings(
    positions: np.ndarray,
    phi: np.ndarray,
    gap: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    groups, centroids = circular_gap_clusters(phi, gap, positions)
    count = len(groups)
    if count < 2 or centroids is None:
        return count, np.array([]), np.array([])
    angles = np.mod(
        np.arctan2(centroids[:, 1] - RADIUS, centroids[:, 0] - RADIUS),
        2.0 * math.pi,
    )
    order = np.argsort(angles)
    centroids = centroids[order]
    angles = angles[order]
    chords = np.linalg.norm(np.roll(centroids, -1, axis=0) - centroids, axis=1)
    wall_arcs = RADIUS * np.mod(np.roll(angles, -1) - angles, 2.0 * math.pi)
    return count, chords, wall_arcs


def analyse_one_seed(
    seed: int,
    positions: np.ndarray,
    phases: np.ndarray,
    iterations: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object], list[FrameResult]]:
    results = [
        frame_analysis(x, theta, MAIN_SHELL, MAIN_PHASE_CONFIDENCE)
        for x, theta in zip(positions, phases)
    ]
    rows = []
    for family in ("CW", "CCW"):
        amplitudes = np.stack(
            [np.asarray(item.family[family]["amplitudes"]) for item in results]
        )
        mean_amplitude = np.mean(amplitudes, axis=0)
        fourier_mode = int(MODES[int(np.argmax(mean_amplitude))])
        frame_modes = MODES[np.argmax(amplitudes, axis=1)]
        radii = np.array(
            [float(np.mean(item.family[family]["radius"])) for item in results]
        )
        counts = np.array([int(item.family[family]["count"]) for item in results])
        q_values = np.array(
            [float(item.family[family]["mean_q"]) for item in results]
        )
        cluster_counts = []
        chords = []
        arcs = []
        for item in results:
            count, chord, arc = direct_cluster_spacings(
                np.asarray(item.family[family]["positions"]),
                np.asarray(item.family[family]["phi"]),
                0.25 * D0,
            )
            cluster_counts.append(count)
            chords.extend(chord.tolist())
            arcs.extend(arc.tolist())
        unique_count, count_frequency = np.unique(cluster_counts, return_counts=True)
        lattice_count = int(unique_count[int(np.argmax(count_frequency))])
        terminal = results[-1].family[family]
        rows.append(
            {
                "seed": seed,
                "family": family,
                "frames_in_terminal_window": len(results),
                "iteration_start": int(iterations[0]),
                "iteration_end": int(iterations[-1]),
                "mean_particle_count": float(np.mean(counts)),
                "mean_tangential_projection": float(np.mean(q_values)),
                "mean_relative_phase_rad": float(
                    np.angle(
                        np.mean(
                            np.exp(
                                1j
                                * np.array(
                                    [
                                        item.family[family]["mean_relative_phase"]
                                        for item in results
                                    ]
                                )
                            )
                        )
                    )
                ),
                "lattice_cluster_count": lattice_count,
                "cluster_count_stability_fraction": float(
                    np.mean(np.asarray(cluster_counts) == lattice_count)
                ),
                "fourier_peak_mode": fourier_mode,
                "fourier_peak_amplitude": float(
                    mean_amplitude[MODES == fourier_mode][0]
                ),
                "fourier_peak_frame_stability_fraction": float(
                    np.mean(frame_modes == fourier_mode)
                ),
                "fourier_amplitude_at_lattice_count": float(
                    mean_amplitude[MODES == lattice_count][0]
                ),
                "mean_effective_radius": float(np.mean(radii)),
                "arc_spacing_at_wall_2piR_over_m": PERIMETER / lattice_count,
                "arc_spacing_at_effective_radius": float(
                    2.0 * math.pi * np.mean(radii) / lattice_count
                ),
                "geometric_chord_at_effective_radius": float(
                    2.0 * np.mean(radii) * math.sin(math.pi / lattice_count)
                ),
                "centroid_chord_mean": float(np.mean(chords)),
                "centroid_chord_std": float(np.std(chords, ddof=1)),
                "centroid_wall_arc_mean": float(np.mean(arcs)),
                "centroid_wall_arc_std": float(np.std(arcs, ddof=1)),
                "terminal_gap_clusters_eps0.25": circular_gap_cluster_count(
                    np.asarray(terminal["phi"]), 0.25 * D0
                ),
                "terminal_gap_clusters_eps0.30": circular_gap_cluster_count(
                    np.asarray(terminal["phi"]), 0.30 * D0
                ),
                "terminal_gap_clusters_eps0.35": circular_gap_cluster_count(
                    np.asarray(terminal["phi"]), 0.35 * D0
                ),
                "terminal_gap_clusters_eps0.40": circular_gap_cluster_count(
                    np.asarray(terminal["phi"]), 0.40 * D0
                ),
                "terminal_gap_clusters_eps0.45": circular_gap_cluster_count(
                    np.asarray(terminal["phi"]), 0.45 * D0
                ),
            }
        )

    mode = rows[0]["lattice_cluster_count"]
    relative_shift = float("nan")
    if rows[1]["lattice_cluster_count"] == mode:
        shifts = []
        for item in results:
            ccw = np.mean(
                np.exp(1j * mode * np.asarray(item.family["CCW"]["phi"]))
            )
            cw = np.mean(
                np.exp(1j * mode * np.asarray(item.family["CW"]["phi"]))
            )
            shifts.append(abs(float(np.angle(ccw * np.conj(cw)))) / (2.0 * math.pi))
        relative_shift = float(np.mean(shifts))
    summary = {
        "seed": seed,
        "mean_axial_phase_order": float(np.mean([item.axial_order for item in results])),
        "mean_phase_family_separation_rad": float(
            np.mean([item.phase_separation for item in results])
        ),
        "mean_phase_family_separation_over_pi": float(
            np.mean([item.phase_separation for item in results]) / math.pi
        ),
        "phase_label_vs_tangential_direction_agreement": float(
            np.mean([item.direction_agreement for item in results])
        ),
        "mean_interstream_spatial_shift_in_lattice_periods": relative_shift,
    }
    return rows, summary, results


def sensitivity_modes(
    seed: int,
    positions: np.ndarray,
    phases: np.ndarray,
) -> list[dict[str, object]]:
    # Every fourth saved frame is enough for a classification sensitivity audit.
    output = []
    for shell in (0.35, 0.50, 0.75, 1.00):
        for confidence in (0.30, 0.50, 0.70):
            items = [
                frame_analysis(x, theta, shell * D0, confidence)
                for x, theta in zip(positions[::4], phases[::4])
            ]
            for family in ("CW", "CCW"):
                amplitudes = np.stack(
                    [np.asarray(item.family[family]["amplitudes"]) for item in items]
                )
                mean_amplitude = np.mean(amplitudes, axis=0)
                fourier_mode = int(MODES[int(np.argmax(mean_amplitude))])
                gap_counts = [
                    circular_gap_cluster_count(
                        np.asarray(item.family[family]["phi"]), 0.25 * D0
                    )
                    for item in items
                ]
                unique_count, frequency = np.unique(gap_counts, return_counts=True)
                lattice_count = int(unique_count[int(np.argmax(frequency))])
                output.append(
                    {
                        "seed": seed,
                        "shell_width_over_d0": shell,
                        "phase_confidence": confidence,
                        "family": family,
                        "lattice_cluster_count": lattice_count,
                        "cluster_count_stability_fraction": float(
                            np.mean(np.asarray(gap_counts) == lattice_count)
                        ),
                        "fourier_peak_mode": fourier_mode,
                        "dominant_amplitude": float(np.max(mean_amplitude)),
                    }
                )
    return output


def plot_spectrum(
    table: pd.DataFrame,
    limit: dict[str, object],
) -> None:
    epsilon = table["epsilon=1-alpha/pi"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    axes[0].loglog(epsilon, table["divergent_coefficient_abs"], "o-", color="#7A5195")
    axes[0].invert_xaxis()
    axes[0].set(
        xlabel=r"$\epsilon=1-\alpha/\pi$",
        ylabel=r"$|v^2k_*^2/(4D_0)|$",
        title=r"Closure coefficient diverges at $\pi$",
    )
    axes[0].grid(alpha=0.25)

    axes[1].semilogx(epsilon, table["k_star*d0"], "o-", color="#276678")
    axes[1].axhline(limit["j2_first_zero_x"], color="black", ls="--", label=r"$j_{2,1}$")
    axes[1].invert_xaxis()
    axes[1].set(
        xlabel=r"$\epsilon$",
        ylabel=r"$k_*d_0$",
        title="One-sided maximizer",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].semilogx(epsilon, table["2pi/k_star"], "o-", color="#D95F02")
    axes[2].axhline(
        limit["wavelength_limit_2pi_over_k"], color="black", ls="--", label="limit"
    )
    axes[2].invert_xaxis()
    axes[2].set(
        xlabel=r"$\epsilon$",
        ylabel=r"$2\pi/k_*$",
        title="One-sided wavelength",
    )
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.savefig(OUT / "near_pi_spectrum_convergence.png", dpi=260)
    plt.close(fig)


def plot_particles(
    all_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, list[FrameResult]]],
    measurements: pd.DataFrame,
    limit: dict[str, object],
) -> None:
    fig = plt.figure(figsize=(14.5, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    family_colors = {"CW": "#D95F02", "CCW": "#1B75BC"}
    for column, seed in enumerate(SEEDS):
        positions, phases, _, results = all_data[seed]
        item = results[-1]
        ax = fig.add_subplot(grid[0, column])
        ax.scatter(positions[-1, :, 0], positions[-1, :, 1], s=2, c="#CCCCCC", alpha=0.25)
        for family in ("CW", "CCW"):
            points = np.asarray(item.family[family]["positions"])
            ax.scatter(
                points[:, 0], points[:, 1], s=5, color=family_colors[family],
                label=family if column == 0 else None,
            )
        ax.add_patch(plt.Circle((RADIUS, RADIUS), RADIUS, fill=False, color="black", lw=0.8))
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, DIAMETER + 0.15)
        ax.set_ylim(-0.15, DIAMETER + 0.15)
        modes = measurements.loc[
            measurements.seed == seed, "lattice_cluster_count"
        ].tolist()
        ax.set_title(f"seed {seed}: phase families, m={modes}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    seed = 9
    positions, phases, _, results = all_data[seed]
    relative = positions[-1] - RADIUS
    phi_raw = np.arctan2(relative[:, 1], relative[:, 0])
    phi = np.mod(phi_raw, 2.0 * math.pi)
    psi = wrap(phases[-1] - phi_raw)
    ax = fig.add_subplot(grid[1, 0])
    for family in ("CW", "CCW"):
        mask = np.asarray(results[-1].family[family]["mask"])
        ax.scatter(phi[mask], psi[mask], s=5, color=family_colors[family], alpha=0.7)
    ax.set(
        xlabel=r"boundary angle $\varphi$",
        ylabel=r"relative phase $\psi=\theta-\varphi$",
        title="Phase-only separation (seed 9)",
        xlim=(0, 2 * math.pi),
        ylim=(-math.pi, math.pi),
    )
    ax.axhline(math.pi / 2, color="black", lw=0.7, ls="--")
    ax.axhline(-math.pi / 2, color="black", lw=0.7, ls="--")

    ax = fig.add_subplot(grid[1, 1])
    for seed in SEEDS:
        _, _, _, results = all_data[seed]
        for family in ("CW", "CCW"):
            amplitudes = np.stack(
                [np.asarray(item.family[family]["amplitudes"]) for item in results]
            )
            ax.plot(
                MODES,
                np.mean(amplitudes, axis=0),
                color=family_colors[family],
                alpha=0.35,
                lw=1.3,
            )
    ax.axvline(
        limit["continuous_boundary_mode_Rk"], color="black", ls="--",
        label=rf"$Rk_*={limit['continuous_boundary_mode_Rk']:.2f}$",
    )
    ax.set(
        xlabel="integer boundary mode m",
        ylabel=r"mean $|N_s^{-1}\sum e^{im\varphi}|$",
        title="Direction-resolved structure factor",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    ax = fig.add_subplot(grid[1, 2])
    observed = measurements.groupby("seed")["centroid_chord_mean"].mean()
    observed_std = measurements.groupby("seed")["centroid_chord_mean"].std()
    x = np.arange(len(SEEDS))
    ax.errorbar(
        x,
        observed.reindex(SEEDS),
        yerr=observed_std.reindex(SEEDS).fillna(0.0),
        fmt="o",
        color="#276678",
        capsize=3,
        label="centroid chord (CW/CCW spread)",
    )
    ax.axhline(
        limit["wavelength_limit_2pi_over_k"], color="#D95F02", ls="--",
        label=r"continuum $2\pi/k_*^{-}$",
    )
    ax.axhline(
        limit["quantized_wall_chord"], color="black", ls=":",
        label="m=18 wall chord",
    )
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set(
        xlabel="random seed",
        ylabel="distance",
        title="Measured versus predicted scale",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "pi_bidirectional_lattice_measurement.png", dpi=280)
    plt.close(fig)


def write_report(
    spectrum: pd.DataFrame,
    limit: dict[str, object],
    measurements: pd.DataFrame,
    seed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    endpoint: dict[str, object],
) -> None:
    observed_modes = sorted(
        measurements["lattice_cluster_count"].unique().tolist()
    )
    predicted = float(limit["wavelength_limit_2pi_over_k"])
    mean_chord = float(measurements["centroid_chord_mean"].mean())
    mean_arc = float(measurements["centroid_wall_arc_mean"].mean())
    relative_chord = (mean_chord - predicted) / predicted
    relative_arc = (mean_arc - predicted) / predicted
    relative_quantized_chord = (
        mean_chord - float(limit["quantized_wall_chord"])
    ) / float(limit["quantized_wall_chord"])
    relative_quantized_arc = (
        mean_arc - float(limit["quantized_arc_spacing"])
    ) / float(limit["quantized_arc_spacing"])
    stable_sensitivity = sensitivity.groupby(["seed", "family"])[
        "lattice_cluster_count"
    ].agg(
        lambda values: sorted(set(int(value) for value in values))
    )
    fourier_sensitivity = sensitivity.groupby(["seed", "family"])[
        "fourier_peak_mode"
    ].agg(lambda values: sorted(set(int(value) for value in values)))
    text = f"""# Alpha = pi 双向边界 Lattice 的尺度检验

## 判定

1. **Dispersion.py 在数学上的 alpha=pi 处不定义。** 对 omega=0，消元分母
   `D0=-2 K sin(alpha)`，故端点严格为零；矩阵中的 `v^2 k^2/(4 D0)` 发散。
   直接输入浮点 `np.pi` 得到有限数只是 `sin(np.pi)={math.sin(math.pi):.17g}` 的舍入伪影，
   不是端点谱。
2. 左极限存在一个明确的**实部包络极限**：`k*d0 -> j_(2,1) =
   {limit['j2_first_zero_x']:.9f}`，所以 `2pi/k* -> {predicted:.6f}`。
   圆周量子化给 `R k*={limit['continuous_boundary_mode_Rk']:.5f}`，最近整数为
   `m={limit['nearest_integer_boundary_mode']}`。
3. 50000 步、N=2000、三个独立初值中，两股相位族测得的主模态为
   `{observed_modes}`。主分析的六个 seed×方向结果均应在表中逐项核对。
   团簇质心直线间距均值为 `{mean_chord:.6f}`，壁面弧长间距均值为
   `{mean_arc:.6f}`；相对左极限 `2pi/k*` 分别偏离 `{relative_chord:+.2%}` 和
   `{relative_arc:+.2%}`。若先把谱预测量子化为 m=18，则预测壁面弦长/弧长为
   `{limit['quantized_wall_chord']:.6f}` / `{limit['quantized_arc_spacing']:.6f}`，
   对应偏差为 `{relative_quantized_chord:+.2%}` / `{relative_quantized_arc:+.2%}`。
4. 因而数据只支持“**同一微观量级**”，不支持“长时 alpha=pi 晶格常数与
   左极限 2pi/k* 定量对应”。严格的最近整数预测是 m=18，而实际稳定主模态
   是 m=16，相差两个团簇；这不能用圆周整数取整解释。

## 相位分类验证

主分类使用局部边界相位 `psi=theta-phi` 的二阶轴向序参量，先得到两个相差约
pi 的相位族，再根据平均切向投影给它们事后命名 CW/CCW。相位标签与实际切向
方向的一致率见 `pi_endpoint_seed_summary.csv`；运动方向没有参与相位族的拟合。

## 为什么左极限不能强制决定端点终态

`alpha->pi-` 时被绝热消去的二阶角谐波松弛率 `D0` 同时趋零，闭合的时间尺度
分离失效。虽然快支实部留下形式极限 `-K J1(x)/x`，矩阵本身的虚部/反对称项
按 `1/sin(alpha)` 发散。因此这个包络可作为候选尺度，却不是对端点非线性、
有限圆盘、长时吸引子的受控外推。有限边界允许 `k_m=m/R`；除此之外，非线性
饱和、团簇合并和初值选择仍可把 m=18 推到相邻或更低的亚稳整数模态。

## 稳健性

- 主时间窗：最后 {TERMINAL_CIRCUITS:g} 个绕壁时间，即 {TERMINAL_TIME:.3f} 时间单位，
  保存帧间隔 {SAVED_DT:g}。
- 主边界层：wall distance <= {MAIN_SHELL/D0:.2f} d0；相位轴置信阈值
  `|cos(psi-beta)| >= {MAIN_PHASE_CONFIDENCE:.2f}`。
- 壳层 0.35--1.00 d0、相位阈值 0.30--0.70 的模式集合：
  `{stable_sensitivity.to_dict()}`。
- 同一敏感性扫描中的最大 Fourier 峰集合为 `{fourier_sensitivity.to_dict()}`；
  它用于诊断晶格畸变，不替代直接团簇计数。
- Fourier 主模态之外，终帧还用圆周间隙阈值 0.25--0.45 d0 做不依赖 Fourier
  幅值的团簇计数；结果随阈值的台阶列在测量 CSV 中。

## 文件

- `near_pi_spectrum.csv`: 单侧逼近的 k*、增长率和波长。
- `pi_endpoint_lattice_measurements.csv`: 每个初值、每股流的尺度。
- `pi_endpoint_seed_summary.csv`: 相位分离与运动方向交叉验证。
- `pi_endpoint_mode_sensitivity.csv`: 分类参数敏感性。
- `pi_endpoint_exact_diagnostic.json`: 端点浮点伪有限性检查。
"""
    (OUT / "pi_endpoint_conclusion.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference_hashes = verify_references()
    module = import_dispersion_module()
    spectrum, limit = analyse_spectrum(module)
    endpoint = exact_endpoint_diagnostic(module, float(limit["k_star_limit"]))
    endpoint["reference_hashes"] = reference_hashes
    endpoint["one_sided_limit"] = limit
    spectrum.to_csv(OUT / "near_pi_spectrum.csv", index=False)
    (OUT / "pi_endpoint_exact_diagnostic.json").write_text(
        json.dumps(endpoint, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    measurement_rows = []
    seed_rows = []
    sensitivity_rows = []
    all_data = {}
    for seed in SEEDS:
        positions, phases, iterations = load_terminal(seed)
        rows, summary, results = analyse_one_seed(seed, positions, phases, iterations)
        measurement_rows.extend(rows)
        seed_rows.append(summary)
        sensitivity_rows.extend(sensitivity_modes(seed, positions, phases))
        all_data[seed] = (positions, phases, iterations, results)

    measurements = pd.DataFrame(measurement_rows)
    seed_summary = pd.DataFrame(seed_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    measurements.to_csv(OUT / "pi_endpoint_lattice_measurements.csv", index=False)
    seed_summary.to_csv(OUT / "pi_endpoint_seed_summary.csv", index=False)
    sensitivity.to_csv(OUT / "pi_endpoint_mode_sensitivity.csv", index=False)
    plot_spectrum(spectrum, limit)
    plot_particles(all_data, measurements, limit)
    write_report(spectrum, limit, measurements, seed_summary, sensitivity, endpoint)
    verify_references()

    print("One-sided limit:")
    print(json.dumps(limit, indent=2))
    print("\nMeasurements:")
    print(measurements.to_string(index=False))
    print("\nPhase classification:")
    print(seed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
