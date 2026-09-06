"""Controlled coupling-strength sweep for the critical circular boundary lattice.

The reference continuum and manuscript files are opened read-only.  New
particle trajectories and all derived artifacts are written below this
repository.  The default screen fixes N=2000, alpha=pi/2 and 50,000 steps and
uses two diameters deliberately chosen to discriminate integer mode-selection
rules.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.cluster import DBSCAN

import critical_boundary_lattice_analysis as critical
from CircularFigure import expected_data_path
from main import phaseCmap, phaseNorm
from small_circular_alpha_sweep import ExperimentConfig, build_model


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "boundary_lattice_k_sweep"
OUTPUT_DIR = PROJECT_DIR / "output" / "Boundary_Lattice_K_Sweep"
REFERENCE_DISPERSION = Path(r"D:\PrivatePythonProject\Math\Lattice\Dispersion.py")
REFERENCE_PRL = Path(r"D:\LaTex\Boundary Flow\PRL.tex")
REFERENCE_METHODS = Path(r"D:\LaTex\Boundary Flow\Methods Appendix.tex")

K_VALUES = (8.0, 12.0, 20.75, 40.0)
DIAMETERS = (3.30, 4.58)
SEEDS = (9, 10, 11)
ALPHA_OVER_PI = 0.5
N_AGENTS = 2000
DISTANCE_D0 = 1.0
SPEED_V = 3.0
DT = 0.005
ITERATIONS = 50000
SNAPSHOT_INTERVAL = 500
TERMINAL_WINDOW_FRAMES = 20
SHELL_WIDTH = 0.25 * DISTANCE_D0


@dataclass(frozen=True, order=True)
class KCondition:
    strength_k: float
    diameter: float
    seed: int


def conditions() -> list[KCondition]:
    return [
        KCondition(k, diameter, seed)
        for k in K_VALUES
        for diameter in DIAMETERS
        for seed in SEEDS
    ]


def config_for(condition: KCondition) -> ExperimentConfig:
    return ExperimentConfig(
        strengthK=condition.strength_k,
        distanceD0=DISTANCE_D0,
        speedV=SPEED_V,
        freqDist="uniform",
        omegaMin=0.0,
        deltaOmega=0.0,
        agentsNum=N_AGENTS,
        dt=DT,
        shotsnaps=SNAPSHOT_INTERVAL,
        randomSeed=condition.seed,
        iterations=ITERATIONS,
    )


def model_for(condition: KCondition):
    return build_model(
        condition.diameter,
        ALPHA_OVER_PI,
        config_for(condition),
        DATA_DIR,
    )


def _simulate_one(condition: KCondition) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = config_for(condition)
    model = model_for(condition)
    path = expected_data_path(model)
    if path.exists() and not critical.hdf_is_complete(path, config):
        model.overWrite = True
    model.run(ITERATIONS)
    if not critical.hdf_is_complete(path, config):
        raise RuntimeError(f"Incomplete trajectory: {path}")
    return str(path)


def ensure_simulations(items: Sequence[KCondition], workers: int) -> None:
    missing = []
    for condition in items:
        path = expected_data_path(model_for(condition))
        if not critical.hdf_is_complete(path, config_for(condition)):
            missing.append(condition)
    if not missing:
        print("All exact K-sweep trajectories already exist.", flush=True)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    worker_count = min(max(1, workers), 4, len(missing))
    print(
        f"Generating {len(missing)} trajectories with {worker_count} workers; "
        f"N={N_AGENTS}, steps={ITERATIONS}...",
        flush=True,
    )
    if worker_count == 1:
        for index, condition in enumerate(missing, start=1):
            _simulate_one(condition)
            print(
                f"[{index:02d}/{len(missing):02d}] K={condition.strength_k:g}, "
                f"D={condition.diameter:g}, seed={condition.seed}",
                flush=True,
            )
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = {
            executor.submit(_simulate_one, condition): condition
            for condition in missing
        }
        for index, future in enumerate(as_completed(futures), start=1):
            condition = futures[future]
            future.result()
            print(
                f"[{index:02d}/{len(missing):02d}] K={condition.strength_k:g}, "
                f"D={condition.diameter:g}, seed={condition.seed}",
                flush=True,
            )


def load_terminal_window(
    condition: KCondition,
    *,
    config: ExperimentConfig | None = None,
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, Path]:
    active_config = config if config is not None else config_for(condition)
    model = build_model(
        condition.diameter,
        ALPHA_OVER_PI,
        active_config,
        data_dir,
    )
    path = expected_data_path(model)
    if not critical.hdf_is_complete(path, active_config):
        raise RuntimeError(f"Missing or incomplete trajectory: {path}")
    with pd.HDFStore(path, mode="r") as store:
        rows = store.get_storer("phaseTheta").nrows
        frame_count = rows // N_AGENTS
        keep = min(TERMINAL_WINDOW_FRAMES, frame_count)
        start = (frame_count - keep) * N_AGENTS
        positions = store.select("positionX", start=start).to_numpy().reshape(
            keep, N_AGENTS, 2
        )
        phases = store.select("phaseTheta", start=start).to_numpy().reshape(
            keep, N_AGENTS
        )
    return positions, phases, path


def cluster_geometry(
    positions: np.ndarray,
    shell: np.ndarray,
    center: np.ndarray,
    expected_mode: int,
) -> dict[str, float]:
    shell_positions = positions[shell]
    labels = DBSCAN(eps=0.10 * DISTANCE_D0, min_samples=10).fit_predict(
        shell_positions
    )
    centers = []
    for label in sorted(set(labels) - {-1}):
        members = shell_positions[labels == label]
        if members.shape[0] >= 20:
            centers.append(np.mean(members, axis=0))
    if len(centers) != expected_mode or expected_mode < 2:
        return {
            "cluster_radius_mean": np.nan,
            "cluster_radius_std": np.nan,
            "actual_chord_mean": np.nan,
            "actual_chord_std": np.nan,
        }
    centers_array = np.asarray(centers)
    relative = centers_array - center
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    ordered = centers_array[np.argsort(angles)]
    chords = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    radii = np.linalg.norm(relative, axis=1)
    return {
        "cluster_radius_mean": float(np.mean(radii)),
        "cluster_radius_std": float(np.std(radii, ddof=1)),
        "actual_chord_mean": float(np.mean(chords)),
        "actual_chord_std": float(np.std(chords, ddof=1)),
    }


def estimate_fourier_locking_time(
    path: Path,
    condition: KCondition,
    target_mode: int,
    *,
    consecutive_frames: int = 10,
) -> tuple[float, bool]:
    """Find the first sustained high-amplitude lock to the terminal mode."""

    with pd.HDFStore(path, mode="r") as store:
        positions = store["positionX"].to_numpy().reshape(-1, N_AGENTS, 2)
    center = np.array([condition.diameter / 2.0, condition.diameter / 2.0])
    radius = condition.diameter / 2.0
    qualified = []
    for frame_positions in positions:
        relative = frame_positions - center
        radial_distance = np.linalg.norm(relative, axis=1)
        shell = radius - radial_distance <= SHELL_WIDTH
        shell_fraction = float(np.mean(shell))
        if np.count_nonzero(shell) < 20:
            qualified.append(False)
            continue
        polar_angles = np.mod(
            np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi
        )
        mode, amplitude, _ = critical.fourier_fundamental(polar_angles[shell])
        qualified.append(
            bool(
                mode == target_mode
                and amplitude >= 0.90
                and shell_fraction >= 0.70
            )
        )
    qualified_array = np.asarray(qualified, dtype=bool)
    locking_index = None
    for start in range(0, qualified_array.size - consecutive_frames + 1):
        if np.all(qualified_array[start : start + consecutive_frames]):
            locking_index = start
            break
    if locking_index is None:
        return np.nan, False
    persistent = bool(np.all(qualified_array[locking_index:]))
    time = locking_index * SNAPSHOT_INTERVAL * DT
    return float(time), persistent


def measure_condition(
    condition: KCondition,
    *,
    config: ExperimentConfig | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[str, object]:
    positions, phases, path = load_terminal_window(
        condition, config=config, data_dir=data_dir
    )
    center = np.array([condition.diameter / 2.0, condition.diameter / 2.0])
    radius = condition.diameter / 2.0
    temporal_modes: list[int] = []
    temporal_amplitudes: list[float] = []
    terminal = None

    for frame_positions, frame_phases in zip(positions, phases):
        relative = frame_positions - center
        radial_distance = np.linalg.norm(relative, axis=1)
        wall_distance = radius - radial_distance
        polar_angles = np.mod(
            np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi
        )
        physical_shell = wall_distance <= SHELL_WIDTH
        physical_shell_fraction = float(np.mean(physical_shell))
        shell_has_enough_particles = bool(np.count_nonzero(physical_shell) >= 20)
        shell = (
            physical_shell
            if shell_has_enough_particles
            else np.ones(N_AGENTS, dtype=bool)
        )
        mode, amplitude, amplitudes = critical.fourier_fundamental(
            polar_angles[shell]
        )
        temporal_modes.append(mode)
        temporal_amplitudes.append(amplitude)
        terminal = (
            frame_positions,
            frame_phases,
            polar_angles,
            wall_distance,
            shell,
            mode,
            amplitude,
            amplitudes,
        )

    assert terminal is not None
    (
        frame_positions,
        frame_phases,
        polar_angles,
        wall_distance,
        shell,
        fourier_mode,
        fourier_amplitude,
        _,
    ) = terminal
    peak_count = critical.periodic_peak_count(polar_angles[shell])
    dbscan_count = critical.dbscan_cluster_count(frame_positions[shell])
    observed_mode = critical.consensus_mode(
        fourier_mode, peak_count, dbscan_count
    )
    temporal_mode = int(np.rint(np.median(temporal_modes)))
    mode_stability = float(np.mean(np.asarray(temporal_modes) == temporal_mode))
    median_amplitude = float(np.median(temporal_amplitudes))
    shell_fraction = physical_shell_fraction
    lattice_formed = bool(
        shell_has_enough_particles
        and fourier_amplitude >= 0.90
        and median_amplitude >= 0.90
        and mode_stability >= 0.90
        and shell_fraction >= 0.70
        and fourier_mode == temporal_mode
        and peak_count == fourier_mode
        and dbscan_count == fourier_mode
    )

    geometry = cluster_geometry(
        frame_positions, shell, center, observed_mode
    )
    effective_radius = geometry["cluster_radius_mean"]
    if not np.isfinite(effective_radius):
        effective_radius = float(np.median(radius - wall_distance[shell]))
    effective_q = observed_mode / effective_radius
    effective_arc = 2.0 * np.pi * effective_radius / observed_mode
    geometric_chord = 2.0 * effective_radius * np.sin(
        np.pi / observed_mode
    )

    tangential = np.sin(frame_phases[shell] - polar_angles[shell])
    directional = np.abs(tangential) >= 0.2
    if np.any(directional):
        direction_sign = float(np.sign(np.mean(tangential[directional])))
        handedness = float(np.abs(np.mean(np.sign(tangential[directional]))))
    else:
        direction_sign = np.nan
        handedness = np.nan
    locking_time, locking_persistent = estimate_fourier_locking_time(
        path, condition, observed_mode
    )

    return {
        "strength_k": condition.strength_k,
        "diameter": condition.diameter,
        "seed": condition.seed,
        "fourier_mode_terminal": fourier_mode,
        "fourier_amplitude_terminal": fourier_amplitude,
        "peak_count_terminal": peak_count,
        "dbscan_count_terminal": dbscan_count,
        "observed_mode": observed_mode,
        "temporal_mode_median": temporal_mode,
        "temporal_mode_stability": mode_stability,
        "temporal_amplitude_median": median_amplitude,
        "shell_particle_fraction": shell_fraction,
        "lattice_formed": lattice_formed,
        "heading_handedness_terminal": handedness,
        "heading_direction_sign_terminal": direction_sign,
        "wall_radius": radius,
        "cluster_radius_mean": geometry["cluster_radius_mean"],
        "cluster_radius_std": geometry["cluster_radius_std"],
        "wall_distance_of_clusters": radius - effective_radius,
        "fourier_locking_time_10_frames": locking_time,
        "fourier_lock_persistent_to_end": locking_persistent,
        "effective_wavenumber": effective_q,
        "effective_arc_spacing": effective_arc,
        "geometric_chord_spacing": geometric_chord,
        "actual_chord_mean": geometry["actual_chord_mean"],
        "actual_chord_std": geometry["actual_chord_std"],
        "bare_turning_length_v_over_k": SPEED_V / condition.strength_k,
        "source_file": str(path),
    }


def load_dispersion_module():
    spec = importlib.util.spec_from_file_location(
        "reference_dispersion_k_sweep", REFERENCE_DISPERSION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {REFERENCE_DISPERSION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def continuum_params(strength_k: float, alpha_over_pi: float):
    rho0 = 1.0
    continuum_lambda = strength_k / (rho0 * np.pi * DISTANCE_D0**2)
    return (
        SPEED_V,
        0.0,
        continuum_lambda,
        alpha_over_pi * np.pi,
        rho0,
        DISTANCE_D0,
    )


def radial_growth(module, k: np.ndarray, strength_k: float, alpha: float):
    values = module.eigs_at_k(
        np.asarray(k), np.zeros_like(k), continuum_params(strength_k, alpha)
    )
    return np.max(np.real(values), axis=-1)


def optimize_one_sided_peak(module, strength_k: float) -> tuple[float, float]:
    alpha = 0.500001
    epsilon = np.pi * (alpha - 0.5)
    grid = np.linspace(1.0e-5, 12.0, 24001)
    normalized = radial_growth(module, grid, strength_k, alpha) / epsilon
    index = int(np.argmax(normalized))
    left = grid[max(0, index - 3)]
    right = grid[min(grid.size - 1, index + 3)]
    result = minimize_scalar(
        lambda x: -float(
            radial_growth(module, np.array([x]), strength_k, alpha)[0]
            / epsilon
        ),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1.0e-11},
    )
    return float(result.x), float(-result.fun)


def bulk_discrete_mode(module, strength_k: float, diameter: float) -> int:
    modes = np.arange(1, 41)
    k = modes / (diameter / 2.0)
    alpha = 0.500001
    growth = radial_growth(module, k, strength_k, alpha)
    return int(modes[int(np.argmax(growth))])


def spectral_table(module) -> pd.DataFrame:
    rows = []
    for strength_k in K_VALUES:
        k_star, coefficient = optimize_one_sided_peak(module, strength_k)
        row = {
            "strength_k": strength_k,
            "one_sided_k_star": k_star,
            "one_sided_lambda_star": 2.0 * np.pi / k_star,
            "growth_coefficient": coefficient,
            "bare_turning_length_v_over_k": SPEED_V / strength_k,
        }
        for diameter in DIAMETERS:
            row[f"bulk_mode_D{diameter:.2f}"] = bulk_discrete_mode(
                module, strength_k, diameter
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(measurements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strength_k, diameter), group in measurements.groupby(
        ["strength_k", "diameter"], sort=True
    ):
        formed = group[group["lattice_formed"]]
        row: dict[str, object] = {
            "strength_k": strength_k,
            "diameter": diameter,
            "realizations": len(group),
            "lattice_formed_count": int(group["lattice_formed"].sum()),
            "lattice_formed_rate": float(group["lattice_formed"].mean()),
            "formed_modes": ",".join(str(int(x)) for x in sorted(formed["observed_mode"])),
        }
        for column in (
            "observed_mode",
            "effective_wavenumber",
            "effective_arc_spacing",
            "geometric_chord_spacing",
            "actual_chord_mean",
            "wall_distance_of_clusters",
            "fourier_locking_time_10_frames",
            "fourier_amplitude_terminal",
        ):
            row[f"{column}_median"] = (
                float(formed[column].median()) if not formed.empty else np.nan
            )
            row[f"{column}_min"] = (
                float(formed[column].min()) if not formed.empty else np.nan
            )
            row[f"{column}_max"] = (
                float(formed[column].max()) if not formed.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_k_dependence(formed: pd.DataFrame) -> dict[str, object]:
    if formed.empty or formed["strength_k"].nunique() < 2:
        return {
            "available": False,
            "inference_valid": False,
            "reason": "Fewer than two K values have strictly formed lattices.",
        }
    k_values = np.sort(formed["strength_k"].unique())
    reference_diameter = float(np.min(formed["diameter"]))
    inverse_k = 1.0 / formed["strength_k"].to_numpy(float)
    diameter_indicator = (
        formed["diameter"].to_numpy(float) > reference_diameter + 1.0e-9
    ).astype(float)
    y = formed["effective_arc_spacing"].to_numpy(float)
    reduced = np.column_stack([np.ones_like(y), diameter_indicator])
    full = np.column_stack([np.ones_like(y), inverse_k, diameter_indicator])
    reduced_coefficients = np.linalg.lstsq(reduced, y, rcond=None)[0]
    full_coefficients = np.linalg.lstsq(full, y, rcond=None)[0]
    reduced_sse = float(np.sum((y - reduced @ reduced_coefficients) ** 2))
    full_sse = float(np.sum((y - full @ full_coefficients) ** 2))
    per_k = {}
    for strength_k, group in formed.groupby("strength_k", sort=True):
        per_k[f"{strength_k:g}"] = {
            "n_formed": int(len(group)),
            "arc_spacing_median": float(group["effective_arc_spacing"].median()),
            "arc_spacing_min": float(group["effective_arc_spacing"].min()),
            "arc_spacing_max": float(group["effective_arc_spacing"].max()),
            "actual_chord_median": float(group["actual_chord_mean"].median()),
            "effective_q_median": float(group["effective_wavenumber"].median()),
        }
    return {
        "available": True,
        "inference_valid": False,
        "reason": (
            "Only three common seeds per K x D cell were run. Coefficients are "
            "descriptive; no p-values or bootstrap confidence intervals are reported."
        ),
        "descriptive_model": "a_eff = intercept + slope / K + diameter fixed effect",
        "intercept": float(full_coefficients[0]),
        "slope": float(full_coefficients[1]),
        "diameter_fixed_effect_offset": float(full_coefficients[2]),
        "reference_diameter": reference_diameter,
        "constant_K_model_SSE": reduced_sse,
        "inverse_K_model_SSE": full_sse,
        "SSE_reduction": reduced_sse - full_sse,
        "k_values": k_values.tolist(),
        "formed_sample_size": int(len(formed)),
        "per_K_descriptive_ranges": per_k,
    }


def create_terminal_figure(measurements: pd.DataFrame) -> plt.Figure:
    with plt.rc_context(
        {"font.family": "STIXGeneral", "mathtext.fontset": "stix"}
    ):
        fig, axes = plt.subplots(
            len(K_VALUES), len(DIAMETERS), figsize=(7.4, 13.2), constrained_layout=True
        )
        for row_index, strength_k in enumerate(K_VALUES):
            for column_index, diameter in enumerate(DIAMETERS):
                axis = axes[row_index, column_index]
                condition = KCondition(strength_k, diameter, SEEDS[0])
                positions, phases, _ = load_terminal_window(condition)
                terminal_positions = positions[-1]
                terminal_phases = phases[-1]
                radius = diameter / 2.0
                axis.quiver(
                    terminal_positions[:, 0],
                    terminal_positions[:, 1],
                    np.cos(terminal_phases),
                    np.sin(terminal_phases),
                    terminal_phases,
                    cmap=phaseCmap,
                    norm=phaseNorm,
                    angles="xy",
                    scale_units="xy",
                    scale=14,
                    width=0.004,
                    headwidth=2.8,
                    headlength=3.2,
                    alpha=0.88,
                )
                circle = plt.Circle(
                    (radius, radius), radius, fill=False, color="black", lw=1.0
                )
                axis.add_patch(circle)
                row = measurements[
                    (np.isclose(measurements["strength_k"], strength_k))
                    & (np.isclose(measurements["diameter"], diameter))
                    & (measurements["seed"] == SEEDS[0])
                ].iloc[0]
                state = "formed" if row["lattice_formed"] else "not formed"
                axis.set_title(
                    rf"$K={strength_k:g},\ D={diameter:g}$; "
                    rf"$m={int(row['observed_mode'])}$ ({state})",
                    fontsize=10,
                )
                axis.set_aspect("equal")
                axis.set_xlim(-0.08, diameter + 0.08)
                axis.set_ylim(-0.08, diameter + 0.08)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
        fig.suptitle(
            r"Critical circular lattice: $N=2000$, $\alpha=\pi/2$, "
            r"$t=250$ (seed 9)",
            fontsize=14,
        )
    return fig


def create_summary_figure(
    measurements: pd.DataFrame, summary: pd.DataFrame, spectral: pd.DataFrame
) -> plt.Figure:
    with plt.rc_context(
        {"font.family": "STIXGeneral", "mathtext.fontset": "stix"}
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
        formed = measurements[measurements["lattice_formed"]]

        axis = axes[0, 0]
        colors = {DIAMETERS[0]: "#4C78A8", DIAMETERS[1]: "#E45756"}
        for diameter, group in formed.groupby("diameter"):
            color = colors[diameter]
            axis.scatter(
                group["strength_k"],
                group["observed_mode"],
                s=32,
                color=color,
                alpha=0.68,
            )
            medians = group.groupby("strength_k", as_index=False)["observed_mode"].median()
            axis.plot(
                medians["strength_k"],
                medians["observed_mode"],
                "o-",
                color=color,
                lw=1.7,
                label=rf"formed, $D={diameter:g}$",
            )
        axis.set_xlabel(r"Coupling $K$")
        axis.set_ylabel(r"Integer mode $m$")
        axis.set_title("(a) Integer mode among strictly formed states")
        axis.legend(frameon=False)

        axis = axes[0, 1]
        for diameter, group in formed.groupby("diameter"):
            axis.scatter(
                group["strength_k"],
                group["effective_arc_spacing"],
                s=33,
                alpha=0.75,
                label=rf"arc, $D={diameter:g}$",
            )
            axis.scatter(
                group["strength_k"],
                group["actual_chord_mean"],
                marker="x",
                s=38,
                alpha=0.8,
                label=rf"chord, $D={diameter:g}$",
            )
        axis.plot(
            spectral["strength_k"],
            spectral["one_sided_lambda_star"],
            "k--",
            lw=1.4,
            label=r"bulk $2\pi/k_*^+$",
        )
        axis.set_xlabel(r"Coupling $K$")
        axis.set_ylabel("Effective spacing")
        axis.set_title("(b) Nonlinear boundary spacing vs bulk wavelength")
        axis.legend(frameon=False, fontsize=8, ncol=2)

        axis = axes[1, 0]
        for diameter, group in summary.groupby("diameter"):
            axis.plot(
                group["strength_k"],
                group["lattice_formed_rate"],
                "o-",
                label=rf"$D={diameter:g}$",
            )
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel(r"Coupling $K$")
        axis.set_ylabel("Strict crystallization fraction")
        axis.set_title("(c) Formation robustness at t=250")
        axis.legend(frameon=False)

        axis = axes[1, 1]
        axis.plot(
            spectral["strength_k"],
            spectral["one_sided_k_star"],
            "o-",
            color="#4C78A8",
            label=r"bulk $k_*^+$",
        )
        for diameter, group in formed.groupby("diameter"):
            axis.scatter(
                group["strength_k"],
                group["effective_wavenumber"],
                s=36,
                alpha=0.72,
                label=rf"boundary $q=m/R_{{eff}}$, $D={diameter:g}$",
            )
        axis.set_xlabel(r"Coupling $K$")
        axis.set_ylabel("Wavenumber")
        axis.set_title("(d) Bulk and boundary selections")
        axis.legend(frameon=False, fontsize=8)
    return fig


def create_kinetics_figure(measurements: pd.DataFrame) -> plt.Figure:
    formed = measurements[measurements["lattice_formed"]]
    with plt.rc_context(
        {"font.family": "STIXGeneral", "mathtext.fontset": "stix"}
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        colors = {DIAMETERS[0]: "#4C78A8", DIAMETERS[1]: "#E45756"}
        axis = axes[0]
        for diameter, group in formed.groupby("diameter"):
            color = colors[diameter]
            axis.scatter(
                group["strength_k"],
                group["wall_distance_of_clusters"],
                color=color,
                alpha=0.70,
                s=34,
            )
            median = group.groupby("strength_k", as_index=False)[
                "wall_distance_of_clusters"
            ].median()
            axis.plot(
                median["strength_k"],
                median["wall_distance_of_clusters"],
                "o-",
                color=color,
                label=rf"$D={diameter:g}$",
            )
        axis.set_xlabel(r"Coupling $K$")
        axis.set_ylabel(r"Cluster-wall distance $R-R_{eff}$")
        axis.set_title("(a) Stronger coupling localizes clusters closer to the wall")
        axis.legend(frameon=False)

        axis = axes[1]
        axis.scatter(
            formed["strength_k"],
            formed["fourier_locking_time_10_frames"],
            color="0.55",
            alpha=0.72,
            s=34,
            label="formed realizations",
        )
        median = formed.groupby("strength_k", as_index=False)[
            "fourier_locking_time_10_frames"
        ].median()
        axis.plot(
            median["strength_k"],
            median["fourier_locking_time_10_frames"],
            "o-",
            color="#54A24B",
            lw=1.8,
            label="median",
        )
        axis.set_xlabel(r"Coupling $K$")
        axis.set_ylabel("Sustained Fourier-locking time")
        axis.set_title("(b) Stronger coupling locks the boundary mode earlier")
        axis.legend(frameon=False)
    return figure


def reference_hashes() -> dict[str, str]:
    hashes = {}
    for path in (REFERENCE_DISPERSION, REFERENCE_PRL, REFERENCE_METHODS):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def write_report(
    measurements: pd.DataFrame,
    summary: pd.DataFrame,
    spectral: pd.DataFrame,
    statistics: dict[str, object],
) -> None:
    formed = measurements[measurements["lattice_formed"]]
    lines = [
        "# Critical boundary lattice: controlled K sweep",
        "",
        f"Fixed parameters: N={N_AGENTS}, alpha/pi={ALPHA_OVER_PI}, "
        f"d0={DISTANCE_D0}, v={SPEED_V}, dt={DT}, steps={ITERATIONS} "
        f"(t={DT * ITERATIONS:g}); seeds={list(SEEDS)}.",
        "",
        "The strict crystallization rule is identical to the earlier "
        "quantization analysis: terminal and last-20-frame median Fourier "
        "amplitudes >=0.90, mode stability >=0.90, boundary-shell fraction "
        ">=0.70, and Fourier/peak/DBSCAN counts all agree.",
        "",
        "## Cell summary",
        "",
        summary.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## One-sided bulk spectrum",
        "",
        spectral.to_markdown(index=False, floatfmt=".7g"),
        "",
        "## Statistical comparison",
        "",
        "```json",
        json.dumps(statistics, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    if formed.empty:
        lines.extend(
            [
                "No condition passed the strict lattice criterion by t=250; "
                "therefore no terminal wavelength claim is made.",
                "",
            ]
        )
    else:
        medians = formed.groupby("strength_k")[
            ["effective_arc_spacing", "actual_chord_mean", "effective_wavenumber"]
        ].median()
        lines.extend(
            [
                "## Formed-state medians by K",
                "",
                medians.reset_index().to_markdown(index=False, floatfmt=".6g"),
                "",
                "Interpretation must condition on lattice formation: failed "
                "runs are kinetics/attractor outcomes, not wavelength samples.",
                "",
            ]
        )
    lines.extend(
        [
            "Reference files were read-only and were not modified. Their SHA-256 "
            "values are stored in K_Sweep_Configuration.json.",
            "",
        ]
    )
    (OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def analyze() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, condition in enumerate(conditions(), start=1):
        rows.append(measure_condition(condition))
        print(
            f"Measured [{index:02d}/{len(conditions()):02d}] "
            f"K={condition.strength_k:g}, D={condition.diameter:g}, "
            f"seed={condition.seed}",
            flush=True,
        )
    measurements = pd.DataFrame(rows).sort_values(
        ["strength_k", "diameter", "seed"]
    )
    summary = summarize(measurements)
    module = load_dispersion_module()
    spectral = spectral_table(module)
    statistics = descriptive_k_dependence(
        measurements[measurements["lattice_formed"]]
    )

    measurements.to_csv(
        OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Measurements.csv", index=False
    )
    summary.to_csv(OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Summary.csv", index=False)
    spectral.to_csv(OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Bulk_Spectrum.csv", index=False)
    (OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    terminal_figure = create_terminal_figure(measurements)
    terminal_figure.savefig(
        OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Terminal_States.png", dpi=240
    )
    terminal_figure.savefig(
        OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Terminal_States.pdf"
    )
    plt.close(terminal_figure)
    summary_figure = create_summary_figure(measurements, summary, spectral)
    summary_figure.savefig(
        OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Analysis.png", dpi=240
    )
    summary_figure.savefig(OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Analysis.pdf")
    plt.close(summary_figure)
    kinetics_figure = create_kinetics_figure(measurements)
    kinetics_figure.savefig(
        OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Kinetics.png", dpi=240
    )
    kinetics_figure.savefig(
        OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Kinetics.pdf"
    )
    plt.close(kinetics_figure)

    configuration = {
        "experiment": {
            "K_values": K_VALUES,
            "diameters": DIAMETERS,
            "seeds": SEEDS,
            "alpha_over_pi": ALPHA_OVER_PI,
            "N": N_AGENTS,
            "d0": DISTANCE_D0,
            "v": SPEED_V,
            "dt": DT,
            "iterations": ITERATIONS,
            "physical_time": DT * ITERATIONS,
            "shotsnaps": SNAPSHOT_INTERVAL,
        },
        "strict_lattice_criterion": {
            "terminal_fourier_amplitude_min": 0.90,
            "last_20_median_amplitude_min": 0.90,
            "last_20_mode_stability_min": 0.90,
            "shell_fraction_min": 0.70,
            "count_agreement": "Fourier = angular peaks = DBSCAN",
        },
        "references_sha256": reference_hashes(),
    }
    (OUTPUT_DIR / "K_Sweep_Configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(measurements, summary, spectral, statistics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulate_only and args.analyze_only:
        raise ValueError("Choose at most one of --simulate-only/--analyze-only")
    if not args.analyze_only:
        ensure_simulations(conditions(), args.workers)
    if not args.simulate_only:
        analyze()


if __name__ == "__main__":
    mp.freeze_support()
    main()
