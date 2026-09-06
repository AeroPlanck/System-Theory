"""Test whether the critical boundary-lattice spacing scales with d0."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
DATA_DIR = PROJECT_DIR / "data" / "boundary_lattice_d0_sweep"
K_SWEEP_DATA_DIR = PROJECT_DIR / "data" / "boundary_lattice_k_sweep"
OUTPUT_DIR = PROJECT_DIR / "output" / "Boundary_Lattice_D0_Sweep"
REFERENCE_DISPERSION = Path(r"D:\PrivatePythonProject\Math\Lattice\Dispersion.py")

STRENGTH_K = 40.0
D0_VALUES = (0.75, 1.0, 1.25)
SIMILARITY_DIAMETER_RATIOS = (3.30, 4.58)
FIXED_DIAMETER = 4.58
SEEDS = (9, 10, 11)
ALPHA_OVER_PI = 0.5
SPEED_V = 3.0
N_AGENTS = 2000
DT = 0.005
ITERATIONS = 50000
SNAPSHOT_INTERVAL = 500
TERMINAL_FRAMES = 20


@dataclass(frozen=True, order=True)
class D0Condition:
    d0: float
    diameter: float
    seed: int


def all_conditions() -> list[D0Condition]:
    items = {
        D0Condition(d0, round(d0 * ratio, 6), seed)
        for d0 in D0_VALUES
        for ratio in SIMILARITY_DIAMETER_RATIOS
        for seed in SEEDS
    }
    items.update(
        D0Condition(d0, FIXED_DIAMETER, seed)
        for d0 in D0_VALUES
        for seed in SEEDS
    )
    return sorted(items)


def config_for(condition: D0Condition) -> ExperimentConfig:
    return ExperimentConfig(
        strengthK=STRENGTH_K,
        distanceD0=condition.d0,
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


def source_directory(condition: D0Condition) -> Path:
    if np.isclose(condition.d0, 1.0) and any(
        np.isclose(condition.diameter, value)
        for value in SIMILARITY_DIAMETER_RATIOS
    ):
        model = build_model(
            condition.diameter,
            ALPHA_OVER_PI,
            config_for(condition),
            K_SWEEP_DATA_DIR,
        )
        if critical.hdf_is_complete(expected_data_path(model), config_for(condition)):
            return K_SWEEP_DATA_DIR
    return DATA_DIR


def model_for(condition: D0Condition, data_dir: Path | None = None):
    return build_model(
        condition.diameter,
        ALPHA_OVER_PI,
        config_for(condition),
        data_dir if data_dir is not None else source_directory(condition),
    )


def simulate_one(condition: D0Condition) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = model_for(condition, DATA_DIR)
    config = config_for(condition)
    path = expected_data_path(model)
    if path.exists() and not critical.hdf_is_complete(path, config):
        model.overWrite = True
    model.run(ITERATIONS)
    if not critical.hdf_is_complete(path, config):
        raise RuntimeError(f"Incomplete trajectory: {path}")
    return str(path)


def ensure_simulations(items: Sequence[D0Condition], workers: int) -> None:
    missing = []
    for condition in items:
        path = expected_data_path(model_for(condition))
        if not critical.hdf_is_complete(path, config_for(condition)):
            missing.append(condition)
    if not missing:
        print("All exact d0-sweep trajectories already exist.", flush=True)
        return
    worker_count = min(max(1, workers), 4, len(missing))
    print(
        f"Generating {len(missing)} trajectories with {worker_count} workers; "
        f"N={N_AGENTS}, steps={ITERATIONS}...",
        flush=True,
    )
    if worker_count == 1:
        for index, condition in enumerate(missing, start=1):
            simulate_one(condition)
            print(
                f"[{index:02d}/{len(missing):02d}] d0={condition.d0:g}, "
                f"D={condition.diameter:g}, seed={condition.seed}",
                flush=True,
            )
        return
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {executor.submit(simulate_one, item): item for item in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            condition = futures[future]
            future.result()
            print(
                f"[{index:02d}/{len(missing):02d}] d0={condition.d0:g}, "
                f"D={condition.diameter:g}, seed={condition.seed}",
                flush=True,
            )


def load_frames(
    condition: D0Condition, *, terminal_only: bool
) -> tuple[np.ndarray, np.ndarray, Path]:
    config = config_for(condition)
    path = expected_data_path(model_for(condition))
    if not critical.hdf_is_complete(path, config):
        raise RuntimeError(f"Missing trajectory: {path}")
    with pd.HDFStore(path, mode="r") as store:
        rows = store.get_storer("phaseTheta").nrows
        frames = rows // N_AGENTS
        keep = min(TERMINAL_FRAMES, frames) if terminal_only else frames
        start = (frames - keep) * N_AGENTS
        positions = store.select("positionX", start=start).to_numpy().reshape(
            keep, N_AGENTS, 2
        )
        phases = store.select("phaseTheta", start=start).to_numpy().reshape(
            keep, N_AGENTS
        )
    return positions, phases, path


def dbscan_count(positions: np.ndarray, d0: float) -> int:
    labels = DBSCAN(eps=0.10 * d0, min_samples=10).fit_predict(positions)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return 0
    _, counts = np.unique(valid, return_counts=True)
    return int(np.count_nonzero(counts >= 20))


def cluster_geometry(
    positions: np.ndarray,
    shell: np.ndarray,
    center: np.ndarray,
    expected_mode: int,
    d0: float,
) -> dict[str, float]:
    shell_positions = positions[shell]
    labels = DBSCAN(eps=0.10 * d0, min_samples=10).fit_predict(shell_positions)
    centers = []
    for label in sorted(set(labels) - {-1}):
        members = shell_positions[labels == label]
        if members.shape[0] >= 20:
            centers.append(np.mean(members, axis=0))
    if len(centers) != expected_mode or expected_mode < 2:
        return {
            "cluster_radius": np.nan,
            "actual_chord": np.nan,
            "actual_chord_std": np.nan,
        }
    centers = np.asarray(centers)
    relative = centers - center
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    ordered = centers[np.argsort(angles)]
    chords = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    return {
        "cluster_radius": float(np.mean(np.linalg.norm(relative, axis=1))),
        "actual_chord": float(np.mean(chords)),
        "actual_chord_std": float(np.std(chords, ddof=1)),
    }


def locking_time(
    condition: D0Condition, target_mode: int
) -> tuple[float, bool]:
    positions, _, _ = load_frames(condition, terminal_only=False)
    center = np.array([condition.diameter / 2.0] * 2)
    radius = condition.diameter / 2.0
    qualified = []
    for frame in positions:
        relative = frame - center
        radial = np.linalg.norm(relative, axis=1)
        shell = radius - radial <= 0.25 * condition.d0
        if np.count_nonzero(shell) < 20:
            qualified.append(False)
            continue
        angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2 * np.pi)
        mode, amplitude, _ = critical.fourier_fundamental(angles[shell])
        qualified.append(
            bool(
                mode == target_mode
                and amplitude >= 0.90
                and np.mean(shell) >= 0.70
            )
        )
    qualified = np.asarray(qualified)
    for start in range(qualified.size - 9):
        if np.all(qualified[start : start + 10]):
            return float(start * SNAPSHOT_INTERVAL * DT), bool(
                np.all(qualified[start:])
            )
    return np.nan, False


def measure(condition: D0Condition) -> dict[str, object]:
    positions, phases, path = load_frames(condition, terminal_only=True)
    center = np.array([condition.diameter / 2.0] * 2)
    radius = condition.diameter / 2.0
    temporal_modes = []
    temporal_amplitudes = []
    terminal = None
    for frame_positions, frame_phases in zip(positions, phases):
        relative = frame_positions - center
        radial = np.linalg.norm(relative, axis=1)
        wall_distance = radius - radial
        angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2 * np.pi)
        physical_shell = wall_distance <= 0.25 * condition.d0
        enough = np.count_nonzero(physical_shell) >= 20
        shell = physical_shell if enough else np.ones(N_AGENTS, dtype=bool)
        mode, amplitude, _ = critical.fourier_fundamental(angles[shell])
        temporal_modes.append(mode)
        temporal_amplitudes.append(amplitude)
        terminal = (
            frame_positions,
            frame_phases,
            angles,
            wall_distance,
            physical_shell,
            shell,
            enough,
            mode,
            amplitude,
        )
    assert terminal is not None
    (
        frame_positions,
        frame_phases,
        angles,
        wall_distance,
        physical_shell,
        shell,
        enough,
        fourier_mode,
        fourier_amplitude,
    ) = terminal
    peak_mode = critical.periodic_peak_count(angles[shell])
    dbscan_mode = dbscan_count(frame_positions[shell], condition.d0)
    observed_mode = critical.consensus_mode(fourier_mode, peak_mode, dbscan_mode)
    temporal_mode = int(np.rint(np.median(temporal_modes)))
    stability = float(np.mean(np.asarray(temporal_modes) == temporal_mode))
    median_amplitude = float(np.median(temporal_amplitudes))
    shell_fraction = float(np.mean(physical_shell))
    formed = bool(
        enough
        and fourier_amplitude >= 0.90
        and median_amplitude >= 0.90
        and stability >= 0.90
        and shell_fraction >= 0.70
        and fourier_mode == temporal_mode
        and peak_mode == fourier_mode
        and dbscan_mode == fourier_mode
    )
    geometry = cluster_geometry(
        frame_positions, shell, center, observed_mode, condition.d0
    )
    effective_radius = geometry["cluster_radius"]
    if not np.isfinite(effective_radius):
        effective_radius = float(np.median(radius - wall_distance[shell]))
    arc = 2.0 * np.pi * effective_radius / observed_mode
    chord_geometric = 2.0 * effective_radius * np.sin(np.pi / observed_mode)
    tangential = np.sin(frame_phases[shell] - angles[shell])
    directional = np.abs(tangential) >= 0.2
    handedness = (
        float(np.abs(np.mean(np.sign(tangential[directional]))))
        if np.any(directional)
        else np.nan
    )
    lock_time, persistent = locking_time(condition, observed_mode)
    return {
        "d0": condition.d0,
        "diameter": condition.diameter,
        "diameter_over_d0": condition.diameter / condition.d0,
        "seed": condition.seed,
        "fourier_mode_terminal": fourier_mode,
        "peak_count_terminal": peak_mode,
        "dbscan_count_terminal": dbscan_mode,
        "observed_mode": observed_mode,
        "temporal_mode_median": temporal_mode,
        "fourier_amplitude_terminal": fourier_amplitude,
        "temporal_amplitude_median": median_amplitude,
        "temporal_mode_stability": stability,
        "shell_particle_fraction": shell_fraction,
        "lattice_formed": formed,
        "heading_handedness_terminal": handedness,
        "effective_radius": effective_radius,
        "wall_distance_of_clusters": radius - effective_radius,
        "effective_wavenumber": observed_mode / effective_radius,
        "effective_arc_spacing": arc,
        "geometric_chord_spacing": chord_geometric,
        "actual_chord_mean": geometry["actual_chord"],
        "actual_chord_std": geometry["actual_chord_std"],
        "arc_over_d0": arc / condition.d0,
        "geometric_chord_over_d0": chord_geometric / condition.d0,
        "actual_chord_over_d0": geometry["actual_chord"] / condition.d0,
        "wall_distance_over_d0": (radius - effective_radius) / condition.d0,
        "fourier_locking_time_10_frames": lock_time,
        "fourier_lock_persistent_to_end": persistent,
        "source_file": str(path),
    }


def load_dispersion_module():
    spec = importlib.util.spec_from_file_location(
        "reference_dispersion_d0_sweep", REFERENCE_DISPERSION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {REFERENCE_DISPERSION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def continuum_params(d0: float, alpha_over_pi: float):
    rho0 = 1.0
    lam = STRENGTH_K / (rho0 * np.pi * d0**2)
    return SPEED_V, 0.0, lam, alpha_over_pi * np.pi, rho0, d0


def radial_growth(module, k: np.ndarray, d0: float) -> np.ndarray:
    alpha = 0.500001
    values = module.eigs_at_k(
        np.asarray(k), np.zeros_like(k), continuum_params(d0, alpha)
    )
    return np.max(np.real(values), axis=-1) / (np.pi * (alpha - 0.5))


def spectral_table(module) -> pd.DataFrame:
    rows = []
    for d0 in D0_VALUES:
        x_grid = np.linspace(1.0e-5, 12.0, 24001)
        k_grid = x_grid / d0
        growth = radial_growth(module, k_grid, d0)
        index = int(np.argmax(growth))
        left = k_grid[max(0, index - 3)]
        right = k_grid[min(k_grid.size - 1, index + 3)]
        result = minimize_scalar(
            lambda k: -float(radial_growth(module, np.array([k]), d0)[0]),
            bounds=(left, right),
            method="bounded",
        )
        k_star = float(result.x)
        rows.append(
            {
                "d0": d0,
                "bulk_k_star": k_star,
                "bulk_k_star_times_d0": k_star * d0,
                "bulk_lambda_star": 2.0 * np.pi / k_star,
                "bulk_lambda_over_d0": 2.0 * np.pi / (k_star * d0),
                "growth_coefficient": float(-result.fun),
            }
        )
    return pd.DataFrame(rows)


def add_bulk_modes(measurements: pd.DataFrame, module) -> pd.DataFrame:
    modes_out = []
    for row in measurements.itertuples(index=False):
        modes = np.arange(1, 51)
        k = modes / (row.diameter / 2.0)
        scores = radial_growth(module, k, row.d0)
        modes_out.append(int(modes[int(np.argmax(scores))]))
    output = measurements.copy()
    output["bulk_quantized_mode"] = modes_out
    output["observed_minus_bulk_mode"] = (
        output["observed_mode"] - output["bulk_quantized_mode"]
    )
    return output


def protocol_rows(measurements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in measurements.to_dict(orient="records"):
        for ratio in SIMILARITY_DIAMETER_RATIOS:
            if np.isclose(row["diameter_over_d0"], ratio, atol=1.0e-5):
                payload = dict(row)
                payload["protocol"] = "similarity_scaled"
                payload["nominal_diameter_over_d0"] = ratio
                rows.append(payload)
        if np.isclose(row["diameter"], FIXED_DIAMETER):
            payload = dict(row)
            payload["protocol"] = "fixed_diameter"
            payload["nominal_diameter_over_d0"] = np.nan
            rows.append(payload)
    return pd.DataFrame(rows)


def summarize(protocol: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["protocol", "d0", "diameter", "diameter_over_d0"]
    for keys, group in protocol.groupby(group_columns, sort=True):
        formed = group[group["lattice_formed"]]
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "realizations": len(group),
                "formed_count": int(group["lattice_formed"].sum()),
                "formed_rate": float(group["lattice_formed"].mean()),
                "formed_modes": ",".join(
                    str(int(x)) for x in sorted(formed["observed_mode"])
                ),
            }
        )
        for column in (
            "observed_mode",
            "effective_arc_spacing",
            "actual_chord_mean",
            "arc_over_d0",
            "actual_chord_over_d0",
            "wall_distance_over_d0",
            "fourier_locking_time_10_frames",
            "bulk_quantized_mode",
            "observed_minus_bulk_mode",
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


def scaling_diagnostics(protocol: pd.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {
        "inference_valid": False,
        "reason": "Three common seeds per cell; results are descriptive.",
    }
    for name in ("similarity_scaled", "fixed_diameter"):
        formed = protocol[
            (protocol["protocol"] == name) & protocol["lattice_formed"]
        ]
        if formed.empty:
            output[name] = {"formed_samples": 0}
            continue
        d0 = formed["d0"].to_numpy(float)
        chord = formed["actual_chord_mean"].to_numpy(float)
        arc = formed["effective_arc_spacing"].to_numpy(float)
        chord_slope = float(np.sum(d0 * chord) / np.sum(d0**2))
        arc_slope = float(np.sum(d0 * arc) / np.sum(d0**2))
        output[name] = {
            "formed_samples": int(len(formed)),
            "actual_chord_over_d0_median": float(
                formed["actual_chord_over_d0"].median()
            ),
            "actual_chord_over_d0_range": [
                float(formed["actual_chord_over_d0"].min()),
                float(formed["actual_chord_over_d0"].max()),
            ],
            "arc_over_d0_median": float(formed["arc_over_d0"].median()),
            "arc_over_d0_range": [
                float(formed["arc_over_d0"].min()),
                float(formed["arc_over_d0"].max()),
            ],
            "through_origin_fit_actual_chord_equals_c_d0": chord_slope,
            "through_origin_fit_arc_equals_c_d0": arc_slope,
            "actual_chord_relative_RMSE": float(
                np.sqrt(np.mean((chord - chord_slope * d0) ** 2))
                / np.mean(chord)
            ),
            "arc_relative_RMSE": float(
                np.sqrt(np.mean((arc - arc_slope * d0) ** 2)) / np.mean(arc)
            ),
        }
    fixed = protocol[
        (protocol["protocol"] == "fixed_diameter")
        & protocol["lattice_formed"]
    ]
    if not fixed.empty:
        medians = fixed.groupby("d0", as_index=False)["observed_mode"].median()
        design = np.column_stack(
            [np.ones(len(medians)), 1.0 / medians["d0"].to_numpy(float)]
        )
        coefficients = np.linalg.lstsq(
            design, medians["observed_mode"].to_numpy(float), rcond=None
        )[0]
        output["fixed_diameter_mode_fit"] = {
            "model": "m = intercept + slope/d0",
            "intercept": float(coefficients[0]),
            "slope": float(coefficients[1]),
        }
    return output


def terminal_figure(measurements: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.3), constrained_layout=True)
    display_conditions = []
    for d0 in D0_VALUES:
        display_conditions.append(D0Condition(d0, round(4.58 * d0, 6), 9))
    for d0 in D0_VALUES:
        display_conditions.append(D0Condition(d0, FIXED_DIAMETER, 9))
    for axis, condition in zip(axes.flat, display_conditions):
        positions, phases, _ = load_frames(condition, terminal_only=True)
        pos = positions[-1]
        theta = phases[-1]
        radius = condition.diameter / 2.0
        axis.quiver(
            pos[:, 0], pos[:, 1], np.cos(theta), np.sin(theta), theta,
            cmap=phaseCmap, norm=phaseNorm, angles="xy", scale_units="xy",
            scale=14, width=0.004, headwidth=2.8, headlength=3.2, alpha=0.88,
        )
        axis.add_patch(
            plt.Circle((radius, radius), radius, fill=False, color="black", lw=1)
        )
        row = measurements[
            np.isclose(measurements["d0"], condition.d0)
            & np.isclose(measurements["diameter"], condition.diameter)
            & (measurements["seed"] == condition.seed)
        ].iloc[0]
        state = "formed" if row["lattice_formed"] else "not formed"
        axis.set_title(
            rf"$d_0={condition.d0:g},D={condition.diameter:g},"
            rf"m={int(row['observed_mode'])}$ ({state})",
            fontsize=10,
        )
        axis.set_aspect("equal")
        axis.set_xlim(-0.05, condition.diameter + 0.05)
        axis.set_ylim(-0.05, condition.diameter + 0.05)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel(r"Similarity: $D/d_0=4.58$", fontsize=11)
    axes[1, 0].set_ylabel(r"Fixed container: $D=4.58$", fontsize=11)
    figure.suptitle(r"Boundary-lattice $d_0$ sweep; $K=40$, seed 9, $t=250$")
    return figure


def analysis_figure(
    protocol: pd.DataFrame, summary: pd.DataFrame, spectral: pd.DataFrame
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)
    formed = protocol[protocol["lattice_formed"]]
    similarity = formed[formed["protocol"] == "similarity_scaled"]
    fixed = formed[formed["protocol"] == "fixed_diameter"]
    axis = axes[0, 0]
    for ratio, group in similarity.groupby("diameter_over_d0"):
        median = group.groupby("d0", as_index=False)["observed_mode"].median()
        axis.scatter(group["d0"], group["observed_mode"], alpha=0.65)
        axis.plot(median["d0"], median["observed_mode"], "o-", label=rf"$D/d_0={ratio:g}$")
    axis.set_xlabel(r"Interaction range $d_0$")
    axis.set_ylabel(r"Mode $m$")
    axis.set_title(r"(a) Similarity protocol: mode invariance")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for ratio, group in similarity.groupby("diameter_over_d0"):
        axis.scatter(
            group["d0"], group["actual_chord_over_d0"], alpha=0.7,
            label=rf"chord, $D/d_0={ratio:g}$",
        )
    axis.axhline(1.0, color="black", ls="--", lw=1.2, label=r"$a=d_0$")
    axis.set_xlabel(r"Interaction range $d_0$")
    axis.set_ylabel(r"Actual centroid chord $a_{chord}/d_0$")
    axis.set_title(r"(b) Direct test of $a_{chord}=d_0$")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    axis.scatter(1.0 / fixed["d0"], fixed["observed_mode"], alpha=0.68)
    medians = fixed.groupby("d0", as_index=False)["observed_mode"].median()
    axis.plot(1.0 / medians["d0"], medians["observed_mode"], "o-")
    axis.set_xlabel(r"$1/d_0$ at fixed $D=4.58$")
    axis.set_ylabel(r"Mode $m$")
    axis.set_title(r"(c) Fixed container: expected $m\sim D/d_0$")

    axis = axes[1, 1]
    for protocol_name, marker in (("similarity_scaled", "o"), ("fixed_diameter", "s")):
        group = formed[formed["protocol"] == protocol_name]
        axis.scatter(
            group["d0"], group["actual_chord_over_d0"], marker=marker,
            alpha=0.66, label=protocol_name,
        )
    axis.plot(
        spectral["d0"], spectral["bulk_lambda_over_d0"], "k--",
        label=r"bulk $(2\pi/k_*^+)/d_0$",
    )
    axis.axhline(1.0, color="0.45", ls=":")
    axis.set_xlabel(r"Interaction range $d_0$")
    axis.set_ylabel(r"Spacing divided by $d_0$")
    axis.set_title("(d) Boundary spacing vs bulk wavelength")
    axis.legend(frameon=False, fontsize=8)
    return figure


def write_report(
    summary: pd.DataFrame,
    spectral: pd.DataFrame,
    diagnostics: dict[str, object],
) -> None:
    report = [
        "# Boundary-lattice interaction-range sweep",
        "",
        f"Fixed K={STRENGTH_K}, N={N_AGENTS}, alpha/pi={ALPHA_OVER_PI}, "
        f"v={SPEED_V}, dt={DT}, steps={ITERATIONS}; seeds={list(SEEDS)}.",
        "",
        "## Cell summary (formed-state wavelength statistics exclude failed runs)",
        "",
        summary.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Bulk one-sided spectrum",
        "",
        spectral.to_markdown(index=False, floatfmt=".7g"),
        "",
        "## Descriptive scaling diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        "```",
        "",
        "Three seeds per cell are sufficient for a controlled scaling screen, "
        "not for a high-precision phase-boundary or formation-probability estimate.",
        "",
    ]
    (OUTPUT_DIR / "Boundary_Lattice_D0_Sweep_Report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def analyze() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    items = all_conditions()
    for index, condition in enumerate(items, start=1):
        rows.append(measure(condition))
        print(
            f"Measured [{index:02d}/{len(items):02d}] d0={condition.d0:g}, "
            f"D={condition.diameter:g}, seed={condition.seed}",
            flush=True,
        )
    measurements = pd.DataFrame(rows).sort_values(["d0", "diameter", "seed"])
    module = load_dispersion_module()
    measurements = add_bulk_modes(measurements, module)
    protocol = protocol_rows(measurements)
    summary = summarize(protocol)
    spectral = spectral_table(module)
    diagnostics = scaling_diagnostics(protocol)
    measurements.to_csv(OUTPUT_DIR / "D0_Sweep_Measurements.csv", index=False)
    protocol.to_csv(OUTPUT_DIR / "D0_Sweep_Protocol_Measurements.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "D0_Sweep_Summary.csv", index=False)
    spectral.to_csv(OUTPUT_DIR / "D0_Sweep_Bulk_Spectrum.csv", index=False)
    (OUTPUT_DIR / "D0_Sweep_Scaling_Diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figure = terminal_figure(measurements)
    figure.savefig(OUTPUT_DIR / "D0_Sweep_Terminal_States.png", dpi=230)
    figure.savefig(OUTPUT_DIR / "D0_Sweep_Terminal_States.pdf")
    plt.close(figure)
    figure = analysis_figure(protocol, summary, spectral)
    figure.savefig(OUTPUT_DIR / "D0_Sweep_Analysis.png", dpi=230)
    figure.savefig(OUTPUT_DIR / "D0_Sweep_Analysis.pdf")
    plt.close(figure)
    configuration = {
        "K": STRENGTH_K,
        "d0_values": D0_VALUES,
        "similarity_D_over_d0": SIMILARITY_DIAMETER_RATIOS,
        "fixed_diameter": FIXED_DIAMETER,
        "seeds": SEEDS,
        "N": N_AGENTS,
        "alpha_over_pi": ALPHA_OVER_PI,
        "v": SPEED_V,
        "dt": DT,
        "steps": ITERATIONS,
        "physical_time": DT * ITERATIONS,
        "strict_criterion": (
            "terminal and median A>=.9; stability>=.9; shell>=.7; "
            "terminal Fourier=temporal mode=peaks=DBSCAN"
        ),
    }
    (OUTPUT_DIR / "D0_Sweep_Configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    write_report(summary, spectral, diagnostics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulate_only and args.analyze_only:
        raise ValueError("Choose at most one action flag.")
    if not args.analyze_only:
        ensure_simulations(all_conditions(), args.workers)
    if not args.simulate_only:
        analyze()


if __name__ == "__main__":
    mp.freeze_support()
    main()
