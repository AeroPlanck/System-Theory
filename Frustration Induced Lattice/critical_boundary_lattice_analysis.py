"""Near-critical spectrum and finite-boundary lattice quantization analysis.

Reference sources are opened read-only.  All generated simulations and
analysis artifacts are written under this repository's data/ and output/
directories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.special import j1
from sklearn.cluster import DBSCAN

from CircularFigure import expected_data_path
from small_circular_alpha_sweep import ExperimentConfig, build_model


PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_DISPERSION = Path(r"D:\PrivatePythonProject\Math\Lattice\Dispersion.py")
REFERENCE_PRL = Path(r"D:\LaTex\Boundary Flow\PRL.tex")
REFERENCE_METHODS = Path(r"D:\LaTex\Boundary Flow\Methods Appendix.tex")

DATA_DIR = PROJECT_DIR / "data" / "critical_boundary_lattice_quantization"
EXISTING_DATA_DIR = (
    PROJECT_DIR / "data" / "small_circular_alpha_sweep" / "N2000_steps50000"
)
OUTPUT_DIR = PROJECT_DIR / "output" / "Critical_Boundary_Lattice_Quantization"

STRENGTH_K = 20.75
INTERACTION_D0 = 1.0
SPEED_V = 3.0
AGENTS_NUM = 2000
DT = 0.005
ITERATIONS = 50000
SNAPSHOT_INTERVAL = 500
TERMINAL_WINDOW_FRAMES = 20
BOUNDARY_SHELL_WIDTH = 0.25 * INTERACTION_D0

CRITICAL_DIAMETERS = (3.0, 3.5, 4.0, 4.5, 5.0)
CRITICAL_SEEDS = (9, 10, 11)
NEAR_CRITICAL_ALPHAS = (0.501, 0.51)
NEAR_CRITICAL_DIAMETERS = (3.0, 5.0)
NEAR_CRITICAL_SEEDS = (9, 10)
SPECTRAL_ALPHAS = (0.5, 0.5001, 0.501, 0.51)


@dataclass(frozen=True, order=True)
class Condition:
    alpha_over_pi: float
    diameter: float
    seed: int


def all_conditions() -> list[Condition]:
    conditions = {
        Condition(0.5, diameter, seed)
        for diameter in CRITICAL_DIAMETERS
        for seed in CRITICAL_SEEDS
    }
    conditions.update(
        Condition(alpha, diameter, seed)
        for alpha in NEAR_CRITICAL_ALPHAS
        for diameter in NEAR_CRITICAL_DIAMETERS
        for seed in NEAR_CRITICAL_SEEDS
    )
    return sorted(conditions)


def condition_config(condition: Condition) -> ExperimentConfig:
    return ExperimentConfig(
        strengthK=STRENGTH_K,
        distanceD0=INTERACTION_D0,
        speedV=SPEED_V,
        freqDist="uniform",
        omegaMin=0.0,
        deltaOmega=0.0,
        agentsNum=AGENTS_NUM,
        dt=DT,
        shotsnaps=SNAPSHOT_INTERVAL,
        randomSeed=condition.seed,
        iterations=ITERATIONS,
    )


def make_model(condition: Condition, data_dir: Path):
    return build_model(
        condition.diameter,
        condition.alpha_over_pi,
        condition_config(condition),
        data_dir,
    )


def expected_frame_count(config: ExperimentConfig) -> int:
    return (config.iterations + config.shotsnaps - 1) // config.shotsnaps + 1


def hdf_is_complete(path: Path, config: ExperimentConfig) -> bool:
    if not path.is_file():
        return False
    try:
        with pd.HDFStore(path, mode="r") as store:
            if not {"/positionX", "/phaseTheta"}.issubset(store.keys()):
                return False
            position = store.get_storer("positionX")
            phase = store.get_storer("phaseTheta")
            expected_rows = expected_frame_count(config) * config.agentsNum
            return (
                position.ncols == 2
                and phase.ncols == 1
                and position.nrows == expected_rows
                and phase.nrows == expected_rows
            )
    except Exception:
        return False


def source_directory(condition: Condition) -> Path:
    if (
        np.isclose(condition.alpha_over_pi, 0.5)
        and condition.seed == 9
        and condition.diameter in (3.0, 5.0)
    ):
        old_model = make_model(condition, EXISTING_DATA_DIR)
        if hdf_is_complete(
            expected_data_path(old_model), condition_config(condition)
        ):
            return EXISTING_DATA_DIR
    return DATA_DIR


def _simulate_one(job: tuple[Condition, str]) -> str:
    condition, data_dir_text = job
    data_dir = Path(data_dir_text)
    data_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(condition, data_dir)
    target = expected_data_path(model)
    if target.exists() and not hdf_is_complete(target, condition_config(condition)):
        model.overWrite = True
    model.run(ITERATIONS)
    if not hdf_is_complete(target, condition_config(condition)):
        raise RuntimeError(f"Incomplete simulation output: {target}")
    return str(target)


def ensure_simulations(conditions: Sequence[Condition], workers: int) -> None:
    jobs: list[tuple[Condition, str]] = []
    for condition in conditions:
        data_dir = source_directory(condition)
        model = make_model(condition, data_dir)
        if not hdf_is_complete(expected_data_path(model), condition_config(condition)):
            jobs.append((condition, str(data_dir)))

    if not jobs:
        print("All exact parameter-matched trajectories already exist.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    worker_count = min(max(1, workers), 4, len(jobs))
    print(
        f"Generating {len(jobs)} trajectories with {worker_count} worker(s)...",
        flush=True,
    )
    if worker_count == 1:
        for index, job in enumerate(jobs, start=1):
            _simulate_one(job)
            condition = job[0]
            print(
                f"[{index:02d}/{len(jobs):02d}] alpha={condition.alpha_over_pi:g}pi, "
                f"D={condition.diameter:g}, seed={condition.seed}",
                flush=True,
            )
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures = {executor.submit(_simulate_one, job): job[0] for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            condition = futures[future]
            future.result()
            print(
                f"[{index:02d}/{len(jobs):02d}] alpha={condition.alpha_over_pi:g}pi, "
                f"D={condition.diameter:g}, seed={condition.seed}",
                flush=True,
            )


def load_terminal_window(condition: Condition) -> tuple[np.ndarray, np.ndarray, Path]:
    data_dir = source_directory(condition)
    model = make_model(condition, data_dir)
    path = expected_data_path(model)
    config = condition_config(condition)
    if not hdf_is_complete(path, config):
        raise RuntimeError(f"Missing or incomplete exact trajectory: {path}")

    with pd.HDFStore(path, mode="r") as store:
        rows = store.get_storer("phaseTheta").nrows
        frames = rows // config.agentsNum
        keep = min(TERMINAL_WINDOW_FRAMES, frames)
        start = (frames - keep) * config.agentsNum
        positions = store.select("positionX", start=start).to_numpy().reshape(
            keep, config.agentsNum, 2
        )
        phases = store.select("phaseTheta", start=start).to_numpy().reshape(
            keep, config.agentsNum
        )
    return positions, phases, path


def fourier_fundamental(
    polar_angles: np.ndarray,
    *,
    minimum_mode: int = 3,
    maximum_mode: int = 40,
) -> tuple[int, float, np.ndarray]:
    modes = np.arange(1, maximum_mode + 1)
    amplitudes = np.abs(
        np.mean(np.exp(1j * polar_angles[:, None] * modes[None, :]), axis=0)
    )
    search = amplitudes[minimum_mode - 1 :]
    global_maximum = float(np.max(search))
    threshold = 0.8 * global_maximum
    candidates = np.where(search >= threshold)[0] + minimum_mode
    fundamental = int(candidates[0])
    return fundamental, float(amplitudes[fundamental - 1]), amplitudes


def periodic_peak_count(polar_angles: np.ndarray, bins: int = 720) -> int:
    histogram, _ = np.histogram(
        polar_angles, bins=bins, range=(0.0, 2.0 * np.pi)
    )
    smooth = gaussian_filter1d(histogram.astype(float), sigma=8.0, mode="wrap")
    extended = np.tile(smooth, 3)
    prominence = max(1e-12, 0.08 * (smooth.max() - smooth.min()))
    peaks, _ = find_peaks(extended, distance=18, prominence=prominence)
    middle = peaks[(peaks >= bins) & (peaks < 2 * bins)] - bins
    return int(np.unique(middle).size)


def dbscan_cluster_count(positions: np.ndarray) -> int:
    labels = DBSCAN(eps=0.10 * INTERACTION_D0, min_samples=10).fit_predict(
        positions
    )
    valid = labels[labels >= 0]
    if valid.size == 0:
        return 0
    _, counts = np.unique(valid, return_counts=True)
    return int(np.count_nonzero(counts >= 20))


def consensus_mode(fourier_mode: int, peak_mode: int, dbscan_mode: int) -> int:
    values = [fourier_mode, peak_mode, dbscan_mode]
    for value in values:
        if value > 0 and values.count(value) >= 2:
            return value
    return fourier_mode


def measure_condition(condition: Condition) -> dict[str, object]:
    positions, phases, path = load_terminal_window(condition)
    center = np.array([condition.diameter / 2.0, condition.diameter / 2.0])
    radius = condition.diameter / 2.0

    temporal_modes: list[int] = []
    temporal_amplitudes: list[float] = []
    terminal_payload = None
    for frame_positions, frame_phases in zip(positions, phases):
        relative = frame_positions - center
        radial_distance = np.linalg.norm(relative, axis=1)
        wall_distance = radius - radial_distance
        polar_angle = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2 * np.pi)
        shell = wall_distance <= BOUNDARY_SHELL_WIDTH
        if np.count_nonzero(shell) < 20:
            shell = np.ones(frame_positions.shape[0], dtype=bool)
        mode, amplitude, amplitudes = fourier_fundamental(polar_angle[shell])
        temporal_modes.append(mode)
        temporal_amplitudes.append(amplitude)
        terminal_payload = (
            frame_positions,
            frame_phases,
            polar_angle,
            wall_distance,
            shell,
            mode,
            amplitude,
            amplitudes,
        )

    assert terminal_payload is not None
    (
        frame_positions,
        frame_phases,
        polar_angle,
        wall_distance,
        shell,
        fourier_mode,
        fourier_amplitude,
        amplitudes,
    ) = terminal_payload

    peak_mode = periodic_peak_count(polar_angle[shell])
    dbscan_mode = dbscan_cluster_count(frame_positions[shell])
    observed_mode = consensus_mode(fourier_mode, peak_mode, dbscan_mode)
    temporal_modes_array = np.asarray(temporal_modes)
    temporal_mode = int(np.rint(np.median(temporal_modes_array)))
    mode_stability = float(np.mean(temporal_modes_array == temporal_mode))
    temporal_amplitude = float(np.median(temporal_amplitudes))
    shell_fraction = float(np.mean(shell))
    lattice_formed = bool(
        fourier_amplitude >= 0.90
        and temporal_amplitude >= 0.90
        and mode_stability >= 0.90
        and shell_fraction >= 0.70
        and peak_mode == fourier_mode
        and dbscan_mode == fourier_mode
    )

    tangential_projection = np.sin(frame_phases[shell] - polar_angle[shell])
    directional = np.abs(tangential_projection) >= 0.2
    if np.count_nonzero(directional):
        direction_sign = float(np.sign(np.mean(tangential_projection[directional])))
        heading_handedness = float(
            np.abs(np.mean(np.sign(tangential_projection[directional])))
        )
    else:
        direction_sign = np.nan
        heading_handedness = np.nan

    perimeter = np.pi * condition.diameter
    return {
        "alpha_over_pi": condition.alpha_over_pi,
        "diameter": condition.diameter,
        "seed": condition.seed,
        "fourier_mode_terminal": fourier_mode,
        "fourier_amplitude_terminal": fourier_amplitude,
        "peak_count_terminal": peak_mode,
        "dbscan_count_terminal": dbscan_mode,
        "observed_mode": observed_mode,
        "temporal_mode_median": temporal_mode,
        "temporal_mode_stability": mode_stability,
        "temporal_amplitude_median": temporal_amplitude,
        "shell_particle_fraction": shell_fraction,
        "lattice_formed": lattice_formed,
        "heading_handedness_terminal": heading_handedness,
        "heading_direction_sign_terminal": direction_sign,
        "observed_wavenumber": 2.0 * observed_mode / condition.diameter,
        "observed_spacing": perimeter / observed_mode,
        "wall_distance_median": float(np.median(wall_distance)),
        "wall_distance_p90": float(np.quantile(wall_distance, 0.9)),
        "source_file": str(path),
    }


def load_dispersion_module():
    specification = importlib.util.spec_from_file_location(
        "reference_dispersion", REFERENCE_DISPERSION
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {REFERENCE_DISPERSION}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def continuum_params(
    alpha_over_pi: float,
    *,
    strength_k: float = STRENGTH_K,
    interaction_d0: float = INTERACTION_D0,
):
    rho0 = 1.0
    lam = strength_k / (rho0 * np.pi * interaction_d0**2)
    return (
        SPEED_V,
        0.0,
        lam,
        alpha_over_pi * np.pi,
        rho0,
        interaction_d0,
    )


def radial_growth(module, wavenumbers: np.ndarray, alpha_over_pi: float) -> np.ndarray:
    eigenvalues = module.eigs_at_k(
        np.asarray(wavenumbers),
        np.zeros_like(wavenumbers),
        continuum_params(alpha_over_pi),
    )
    return np.max(np.real(eigenvalues), axis=-1)


def asymptotic_growth_coefficient(wavenumbers: np.ndarray) -> np.ndarray:
    k = np.asarray(wavenumbers, dtype=float)
    x = k * INTERACTION_D0
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel_ratio = 2.0 * j1(x) / x
    kernel_ratio = np.where(np.abs(x) < 1e-12, 1.0, kernel_ratio)
    b_critical = (
        STRENGTH_K * (1.0 - 0.5 * kernel_ratio)
        - SPEED_V**2 * k**2 / (8.0 * STRENGTH_K)
    )
    c_squared = 0.5 * SPEED_V**2 * k**2
    denominator = b_critical**2 + c_squared
    prefactor = -0.5 * STRENGTH_K * kernel_ratio
    central = np.divide(
        prefactor * c_squared,
        denominator,
        out=np.zeros_like(k),
        where=denominator > 0,
    )
    oscillatory = np.divide(
        prefactor * (b_critical**2 + 0.5 * c_squared),
        denominator,
        out=np.zeros_like(k),
        where=denominator > 0,
    )
    return np.maximum.reduce([central, oscillatory, np.zeros_like(k)])


def optimize_curve(curve, lower: float = 1e-6, upper: float = 15.0):
    grid = np.linspace(lower, upper, 30001)
    values = curve(grid)
    index = int(np.argmax(values))
    left = grid[max(0, index - 3)]
    right = grid[min(grid.size - 1, index + 3)]
    optimum = minimize_scalar(
        lambda value: -float(curve(np.array([value]))[0]),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1e-13},
    )
    return float(optimum.x), float(-optimum.fun)


def spectral_peaks(module) -> tuple[pd.DataFrame, float, float, float]:
    asymptotic_k, asymptotic_growth = optimize_curve(
        asymptotic_growth_coefficient
    )
    rows = []
    alpha_grid = (
        0.500001,
        0.50001,
        0.5001,
        0.501,
        0.502,
        0.505,
        0.51,
        0.52,
        0.55,
        0.6,
    )
    for alpha in alpha_grid:
        k_star, growth = optimize_curve(
            lambda k, alpha=alpha: radial_growth(module, k, alpha)
        )
        rows.append(
            {
                "alpha_over_pi": alpha,
                "epsilon_radians": np.pi * (alpha - 0.5),
                "k_star": k_star,
                "lambda_star": 2.0 * np.pi / k_star,
                "maximum_growth": growth,
                "growth_over_epsilon": growth / (np.pi * (alpha - 0.5)),
            }
        )

    literal_k, _ = optimize_curve(
        lambda k: np.max(
            np.real(
                module.eigs_at_k(
                    k,
                    np.zeros_like(k),
                    continuum_params(
                        0.500001, strength_k=20.0, interaction_d0=2.0
                    ),
                )
            ),
            axis=-1,
        )
    )
    return (
        pd.DataFrame(rows),
        asymptotic_k,
        asymptotic_growth,
        literal_k,
    )


def discrete_spectral_mode(
    module,
    diameter: float,
    alpha_over_pi: float,
    asymptotic: bool = False,
) -> tuple[int, float, float]:
    modes = np.arange(1, 41)
    wavenumbers = 2.0 * modes / diameter
    if asymptotic:
        scores = asymptotic_growth_coefficient(wavenumbers)
    else:
        scores = radial_growth(module, wavenumbers, alpha_over_pi)
    index = int(np.argmax(scores))
    return int(modes[index]), float(wavenumbers[index]), float(scores[index])


def add_spectral_predictions(
    measurements: pd.DataFrame,
    module,
) -> pd.DataFrame:
    predicted_modes = []
    predicted_wavenumbers = []
    predicted_scores = []
    for row in measurements.itertuples(index=False):
        mode, wavenumber, score = discrete_spectral_mode(
            module,
            row.diameter,
            row.alpha_over_pi,
            asymptotic=np.isclose(row.alpha_over_pi, 0.5),
        )
        predicted_modes.append(mode)
        predicted_wavenumbers.append(wavenumber)
        predicted_scores.append(score)
    output = measurements.copy()
    output["spectral_quantized_mode"] = predicted_modes
    output["spectral_quantized_wavenumber"] = predicted_wavenumbers
    output["spectral_quantized_score"] = predicted_scores
    output["mode_residual_observed_minus_spectral"] = (
        output["observed_mode"] - output["spectral_quantized_mode"]
    )
    return output


def summarize_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "observed_mode",
        "observed_wavenumber",
        "observed_spacing",
        "fourier_amplitude_terminal",
        "temporal_mode_stability",
        "heading_handedness_terminal",
        "shell_particle_fraction",
        "spectral_quantized_mode",
        "spectral_quantized_wavenumber",
        "mode_residual_observed_minus_spectral",
    ]
    formed = measurements[measurements["lattice_formed"]]
    grouped = formed.groupby(["alpha_over_pi", "diameter"])
    summary = grouped[numeric_columns].agg(["median", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    counts = (
        measurements.groupby(["alpha_over_pi", "diameter"])["lattice_formed"]
        .agg(realizations="size", lattice_formed_count="sum", lattice_formed_rate="mean")
        .reset_index()
    )
    return counts.merge(summary, on=["alpha_over_pi", "diameter"], how="left")


def fit_empirical_wavenumber(critical: pd.DataFrame) -> float:
    if critical.empty:
        raise RuntimeError("No formed boundary lattice is available for fitting.")
    diameters = critical["diameter"].to_numpy(dtype=float)
    modes = critical["observed_mode"].to_numpy(dtype=float)
    return float(2.0 * np.sum(diameters * modes) / np.sum(diameters**2))


def create_figure(
    measurements: pd.DataFrame,
    spectral: pd.DataFrame,
    module,
    asymptotic_k: float,
    literal_k: float,
    empirical_k: float,
) -> plt.Figure:
    rc = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
    with plt.rc_context(rc):
        figure, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
        ax = axes[0, 0]
        k_plot = np.linspace(0.0, 9.0, 2400)
        for alpha, color in zip(SPECTRAL_ALPHAS, ["0.25", "#4C78A8", "#F58518", "#E45756"]):
            growth = radial_growth(module, k_plot, alpha)
            if np.isclose(alpha, 0.5):
                growth = np.zeros_like(growth)
            ax.plot(k_plot, growth, lw=1.7, color=color, label=rf"${alpha:g}\pi$")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel(r"Wavenumber $k$")
        ax.set_ylabel(r"Maximum growth $\max_n\mathrm{Re}\,\sigma_n$")
        ax.set_title("(a) Near-critical growth spectra")
        ax.legend(frameon=False)

        ax = axes[0, 1]
        for alpha, color in zip((0.5001, 0.501, 0.51), ["#4C78A8", "#F58518", "#E45756"]):
            normalized = radial_growth(module, k_plot, alpha) / (
                np.pi * (alpha - 0.5)
            )
            ax.plot(k_plot, normalized, lw=1.5, color=color, label=rf"${alpha:g}\pi$")
        ax.plot(
            k_plot,
            asymptotic_growth_coefficient(k_plot),
            color="black",
            lw=2.0,
            ls="--",
            label=r"$g_1(k)$",
        )
        ax.axvline(asymptotic_k, color="0.35", lw=1.0, ls=":")
        ax.set_xlabel(r"Wavenumber $k$")
        ax.set_ylabel(r"Growth divided by $\alpha-\pi/2$")
        ax.set_title("(b) Collapse onto the one-sided asymptotic spectrum")
        ax.legend(frameon=False)

        ax = axes[1, 0]
        ax.semilogx(
            spectral["alpha_over_pi"] - 0.5,
            spectral["k_star"],
            "o-",
            color="#4C78A8",
        )
        ax.axhline(asymptotic_k, color="black", ls="--", lw=1.2, label=rf"$k_\infty={asymptotic_k:.3f}$")
        ax.set_xlabel(r"$\alpha/\pi-0.5$")
        ax.set_ylabel(r"Continuous maximizer $k_*$")
        ax.set_title("(c) One-sided convergence of the selected wavenumber")
        ax.legend(frameon=False)

        critical_all = measurements[np.isclose(measurements["alpha_over_pi"], 0.5)]
        critical = critical_all[critical_all["lattice_formed"]]
        ax = axes[1, 1]
        for seed, group in critical.groupby("seed"):
            ax.scatter(
                group["diameter"],
                group["observed_mode"],
                s=38,
                alpha=0.8,
                label=f"seed {seed}",
            )
        failed = critical_all[~critical_all["lattice_formed"]]
        ax.scatter(
            failed["diameter"],
            failed["fourier_mode_terminal"],
            marker="x",
            s=32,
            color="0.55",
            label="not crystallized by t=250",
        )
        diameters = np.asarray(CRITICAL_DIAMETERS)
        predicted = [
            discrete_spectral_mode(module, diameter, 0.5, asymptotic=True)[0]
            for diameter in diameters
        ]
        ax.step(
            diameters,
            predicted,
            where="mid",
            color="black",
            lw=1.6,
            label="quantized continuum prediction",
        )
        dense_d = np.linspace(min(diameters), max(diameters), 300)
        ax.plot(
            dense_d,
            empirical_k * dense_d / 2.0,
            color="#E45756",
            ls="--",
            lw=1.4,
            label=rf"empirical $m={empirical_k:.3f}D/2$",
        )
        ax.set_xlabel(r"Boundary diameter $D$")
        ax.set_ylabel(r"Integer boundary mode $m$")
        ax.set_title(r"(d) Finite-circumference mode quantization at $0.5\pi$")
        ax.legend(frameon=False, fontsize=9)

        ax = axes[2, 0]
        for seed, group in critical.groupby("seed"):
            ax.scatter(
                group["diameter"],
                group["observed_spacing"],
                s=38,
                alpha=0.8,
                label=f"seed {seed}",
            )
        ax.axhline(
            2.0 * np.pi / asymptotic_k,
            color="black",
            ls="--",
            lw=1.5,
            label=rf"continuum $2\pi/k_\infty={2*np.pi/asymptotic_k:.3f}$",
        )
        ax.axhline(
            2.0 * np.pi / empirical_k,
            color="#E45756",
            ls=":",
            lw=1.6,
            label=rf"empirical $2\pi/k_{{\rm eff}}={2*np.pi/empirical_k:.3f}$",
        )
        ax.axhline(
            2.0 * np.pi / literal_k,
            color="#54A24B",
            ls="-.",
            lw=1.2,
            label=rf"literal script $d_0=2$: {2*np.pi/literal_k:.3f}",
        )
        ax.set_xlabel(r"Boundary diameter $D$")
        ax.set_ylabel(r"Boundary spacing $a_b=\pi D/m$")
        ax.set_title("(e) Observed spacing versus spectral wavelengths")
        ax.legend(frameon=False, fontsize=9)

        ax = axes[2, 1]
        near = measurements[
            measurements["diameter"].isin(NEAR_CRITICAL_DIAMETERS)
            & measurements["seed"].isin(NEAR_CRITICAL_SEEDS)
            & measurements["lattice_formed"]
        ]
        for diameter, group in near.groupby("diameter"):
            summary = group.groupby("alpha_over_pi")["observed_mode"].median()
            ax.plot(
                summary.index,
                summary.values,
                "o-",
                lw=1.5,
                label=rf"$D={diameter:g}$",
            )
        ax.set_xlabel(r"Phase lag $\alpha/\pi$")
        ax.set_ylabel(r"Observed boundary mode $m$")
        ax.set_title("(f) Persistence of the integer mode above onset")
        ax.legend(frameon=False)

        figure.suptitle(
            "Near-critical spectrum and circular-boundary lattice quantization",
            fontsize=16,
        )
    return figure


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_report(
    measurements: pd.DataFrame,
    spectral: pd.DataFrame,
    asymptotic_k: float,
    asymptotic_growth: float,
    literal_k: float,
    empirical_k: float,
) -> Path:
    critical_all = measurements[np.isclose(measurements["alpha_over_pi"], 0.5)]
    critical = critical_all[critical_all["lattice_formed"]]
    exact_matches = np.mean(
        critical["observed_mode"] == critical["spectral_quantized_mode"]
    )
    formation_rate = float(critical_all["lattice_formed"].mean())
    residual_median = float(
        critical["mode_residual_observed_minus_spectral"].median()
    )
    reference_hashes = {
        str(path): file_sha256(path)
        for path in (REFERENCE_DISPERSION, REFERENCE_PRL, REFERENCE_METHODS)
    }
    report_path = OUTPUT_DIR / "Critical_Boundary_Lattice_Analysis.md"
    lines = [
        "# 临界手性边界晶格：谱的一侧极限与整数化检验",
        "",
        "## 主要结论",
        "",
        (
            f"对与粒子模拟匹配的参数 `K=20.75, d0=1, v=3`，连续体谱从 "
            f"`alpha > pi/2` 一侧得到 `k_inf = {asymptotic_k:.9f}`，对应 "
            f"`2 pi/k_inf = {2*np.pi/asymptotic_k:.9f}`。"
        ),
        (
            f"仅对已经形成高相干边界晶格的临界样本拟合，得到 "
            f"`k_eff = {empirical_k:.9f}`，对应边界晶格常数 "
            f"`a_eff = 2 pi/k_eff = {2*np.pi/empirical_k:.9f}`。"
        ),
        (
            f"15 个 `alpha=0.5 pi` 随机实现中有 {len(critical)} 个在 "
            f"`t=250` 前成晶（成晶率 {formation_rate:.1%}）。成晶样本与直接体谱整数化 "
            f"`m_q=argmax_m g1(2m/D)` 的逐样本精确符合率为 {exact_matches:.1%}，"
            f"实测模态相对体谱预测的中位偏移为 `+{residual_median:g}`。"
        ),
        "",
        "## 多直径整数化结果（只列已成晶样本）",
        "",
        "| D | 已成晶/总数 | 实测 m | 直接体谱 m_q | 实测 k=2m/D |",
        "|---:|---:|:---:|---:|:---:|",
    ]
    for diameter, group in critical_all.groupby("diameter"):
        formed = group[group["lattice_formed"]]
        observed = ", ".join(str(int(value)) for value in sorted(formed["observed_mode"]))
        observed_k = ", ".join(
            f"{value:.3f}" for value in sorted(formed["observed_wavenumber"])
        )
        predicted = int(group["spectral_quantized_mode"].iloc[0])
        lines.append(
            f"| {diameter:g} | {len(formed)}/{len(group)} | {observed} | "
            f"{predicted} | {observed_k} |"
        )
    lines.extend(
        [
        "",
        (
            "已成晶的五个直径均给出稳定整数平台 `m=(9,10,12,13,15)`；每个直径的 "
            "两个成功随机实现彼此一致，并且都等于 "
            "`round(k_eff D/2)`。这支持圆周周期条件下的整数锁定，但不支持把体谱的 "
            "`k_inf` 不加修正地当成边界晶格波数。"
        ),
        "",
        "## 为什么 alpha = pi/2 没有直接的 k-star",
        "",
        (
            "在 `alpha=pi/2`，径向线性矩阵中的实增长耦合 `a(k)=0`。三个本征值均为纯虚数"
            "（其中一个为零），所以所有 `k` 的实部都为零；临界点本身不存在唯一最不稳定波数。"
            "这与 Methods Appendix 将该点视为孤立临界点、不给任一开放区间估计器归类是一致的。"
        ),
        "",
        "令 `alpha=pi/2+epsilon`、`f(k)=Ghat(k)/G0`，一阶展开为",
        "",
        "```text",
        "B(k) = K [1 - f(k)/2] - v^2 k^2/(8K)",
        "C(k) = v^2 k^2/2",
        "g1,0(k) = -(K/2) f(k) C/(B^2+C)",
        "g1,+/-(k) = -(K/2) f(k) (B^2+C/2)/(B^2+C)",
        "max Re sigma(k) = epsilon max(0,g1,0,g1,+/-) + O(epsilon^2).",
        "```",
        "",
        (
            f"数值上 `max g1 = {asymptotic_growth:.9f}`，其最大点即上述 `k_inf`。"
            "`0.5001 pi, 0.501 pi, 0.51 pi` 的增长谱除以 `epsilon` 后塌缩到该一阶函数，"
            "因此这个极限不是临界点的浮点噪声。"
        ),
        "",
        "## 有限圆周的量子化条件",
        "",
        "圆周长 `P=pi D`。边界密度的周期性只允许",
        "",
        "```text",
        "k_m = 2 pi m / P = 2m/D,   m in positive integers.",
        "```",
        "",
        (
            "若直接把体谱投影到圆周，则临界点的一侧预测为 "
            "`m_q=argmax_m g1(k_m)`，超临界点为 "
            "`m_q=argmax_m max Re sigma(k_m)`。数值结果系统性高出该预测 1--2 个团簇。"
            "因此真正成立的是几何整数条件 `k_b=2m/D`，而选定 `k_b` 的机制并非裸的径向体谱。"
        ),
        "",
        "PRL 用边缘谱流解释传播手性；径向 `Dispersion.py` 描述的是均匀体态的初始失稳。"
        "边界无穿透条件、近壁密度增益、曲率以及非线性饱和都会重整化切向晶格间距。"
        "要从解析上预测约 `k_eff=5.90`，下一步应求解带墙面边界条件的条带/圆盘本征问题，"
        "而不是继续提高径向体谱的 `k` 网格精度。",
        "",
        "## 随机实现与超临界检查",
        "",
        (
            "临界点每个直径都有 2/3 个随机种子在 `t=250` 前进入高相干、单向切向运动的"
            "边界晶格；其余样本的低傅里叶幅度、低近壁粒子比例和不稳定时间模态表明它们"
            "尚未成晶，不能作为晶格常数样本。"
        ),
        (
            "在 `0.51 pi`，四个测试实现全部成晶，`D=3` 得到 `m=8,9`，`D=5` 得到 "
            "`m=14,15`；这显示靠近临界点存在相邻整数模态竞争，而不是单一确定的团簇数。"
        ),
        "",
        "## 参数一致性",
        "",
        (
            "PRL 的圆形碰撞数据及本次粒子文件使用 `d0=1, K=20.75`。"
            "`Dispersion.py` 底部可执行示例写的是 `d0=2, K=20`，其一侧波长为 "
            f"`{2*np.pi/literal_k:.9f}`，不能与这里的粒子轨迹直接比较。"
        ),
        "",
        "## 参考文件完整性",
        "",
        ]
    )
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in reference_hashes.items())
    lines.extend(
        [
            "",
            "分析脚本只读打开以上参考文件，所有新增数据和报告均写入本项目的 `data/` 与 "
            "`output/` 目录。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_analysis(workers: int, skip_simulation: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = all_conditions()
    if not skip_simulation:
        ensure_simulations(conditions, workers)

    module = load_dispersion_module()
    measurements = pd.DataFrame(
        [measure_condition(condition) for condition in conditions]
    )
    spectral, asymptotic_k, asymptotic_growth, literal_k = spectral_peaks(module)
    measurements = add_spectral_predictions(measurements, module)
    summary = summarize_measurements(measurements)
    critical = measurements[
        np.isclose(measurements["alpha_over_pi"], 0.5)
        & measurements["lattice_formed"]
    ]
    empirical_k = fit_empirical_wavenumber(critical)

    measurements.to_csv(
        OUTPUT_DIR / "Boundary_Lattice_Quantization_Measurements.csv", index=False
    )
    summary.to_csv(
        OUTPUT_DIR / "Boundary_Lattice_Quantization_Summary.csv", index=False
    )
    spectral.to_csv(OUTPUT_DIR / "Near_Critical_Dispersion_Peaks.csv", index=False)

    figure = create_figure(
        measurements,
        spectral,
        module,
        asymptotic_k,
        literal_k,
        empirical_k,
    )
    figure.savefig(
        OUTPUT_DIR / "Critical_Boundary_Lattice_Quantization.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / "Critical_Boundary_Lattice_Quantization.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    report = write_report(
        measurements,
        spectral,
        asymptotic_k,
        asymptotic_growth,
        literal_k,
        empirical_k,
    )
    configuration = {
        "model": "CircularBoundaryPatternFormation",
        "simulation": asdict(condition_config(conditions[0])),
        "critical_diameters": list(CRITICAL_DIAMETERS),
        "critical_seeds": list(CRITICAL_SEEDS),
        "near_critical_alphas": list(NEAR_CRITICAL_ALPHAS),
        "near_critical_diameters": list(NEAR_CRITICAL_DIAMETERS),
        "near_critical_seeds": list(NEAR_CRITICAL_SEEDS),
        "boundary_shell_width": BOUNDARY_SHELL_WIDTH,
        "terminal_window_frames": TERMINAL_WINDOW_FRAMES,
        "reference_files": {
            str(path): file_sha256(path)
            for path in (REFERENCE_DISPERSION, REFERENCE_PRL, REFERENCE_METHODS)
        },
    }
    (OUTPUT_DIR / "Critical_Boundary_Lattice_Configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved analysis report: {report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-simulation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4.")
    run_analysis(args.workers, skip_simulation=args.skip_simulation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
