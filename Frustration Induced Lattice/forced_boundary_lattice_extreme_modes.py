"""Test strongly underfilled and overfilled planted boundary lattices.

All cases use the same physical parameters and failed random seed.  Only the
number of deliberately planted boundary clusters changes.  The outputs are
intervention trajectories and must not be represented as spontaneous states.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forced_boundary_lattice_experiment as base
from CircularFigure import expected_data_path
from main import phaseCmap, phaseNorm
from small_circular_alpha_sweep import _calc_dot_phase_collision_fast


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "forced_boundary_lattice_extreme_modes"
OUTPUT_DIR = PROJECT_DIR / "output" / "Forced_Boundary_Lattice_Extreme_Modes"

N_AGENTS = 2000
DT = 0.005
ITERATIONS = 50_000
SNAPSHOT_INTERVAL = 100
TERMINAL_FRAMES = 20

STRENGTH_K = 20.75
D0 = 1.0
ALPHA_OVER_PI = 0.5
DIAMETER = 4.5
SEED = 11
GEOMETRIC_MODE = int(np.floor(np.pi * DIAMETER / D0))
TARGET_MODES = (5, 7, 21, 28)
KEY_TIMES = (0.0, 0.5, 2.5, 10.0, 50.0, 125.0, 250.0)


@dataclass(frozen=True, order=True)
class ExtremeCondition:
    target_mode: int
    source_file: str
    family: str = "extreme_mode_intervention"
    strength_k: float = STRENGTH_K
    d0: float = D0
    alpha_over_pi: float = ALPHA_OVER_PI
    diameter: float = DIAMETER
    seed: int = SEED

    @property
    def label(self) -> str:
        relation = "underfilled" if self.target_mode < GEOMETRIC_MODE else "overfilled"
        return f"{relation}_m{self.target_mode}_vs_mgeom{GEOMETRIC_MODE}"


def source_failure_file() -> str:
    candidates = [
        condition
        for condition in base.selected_conditions()
        if np.isclose(condition.strength_k, STRENGTH_K)
        and np.isclose(condition.d0, D0)
        and np.isclose(condition.diameter, DIAMETER)
        and condition.seed == SEED
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one D=4.5 failed source, found {len(candidates)}.")
    return candidates[0].source_file


def conditions() -> list[ExtremeCondition]:
    source = source_failure_file()
    return [ExtremeCondition(mode, source) for mode in TARGET_MODES]


class ExtremeForcedCircularBoundaryLattice(base.ForcedCircularBoundaryLattice):
    """Planted lattice with deliberately stronger finite perturbations."""

    def _plant_boundary_lattice(self) -> None:
        rng = np.random.default_rng(self.randomSeed + 7919 * self.targetMode)
        mode = self.targetMode
        spacing_angle = 2.0 * np.pi / mode
        global_angle = rng.uniform(0.0, 2.0 * np.pi)
        center_jitter = rng.normal(0.0, 0.08 * spacing_angle, mode)
        center_jitter -= np.mean(center_jitter)
        cluster_angles = np.mod(
            global_angle + spacing_angle * np.arange(mode) + center_jitter,
            2.0 * np.pi,
        )

        counts = np.full(mode, self.agentsNum // mode, dtype=int)
        counts[: self.agentsNum % mode] += 1
        cluster_ids = np.repeat(np.arange(mode), counts)
        rng.shuffle(cluster_ids)

        target_radius = self.circleRadius - 0.035 * self.distanceD0
        tangent_offset = rng.normal(0.0, 0.050 * self.distanceD0, self.agentsNum)
        inward_offset = np.abs(
            rng.normal(0.0, 0.035 * self.distanceD0, self.agentsNum)
        )
        particle_angles = cluster_angles[cluster_ids] + tangent_offset / target_radius
        particle_radius = np.maximum(0.0, target_radius - inward_offset)
        self.positionX = self.circleCenter + np.column_stack(
            [
                particle_radius * np.cos(particle_angles),
                particle_radius * np.sin(particle_angles),
            ]
        )
        self.phaseTheta = np.mod(
            particle_angles
            + 0.5 * np.pi
            + rng.normal(0.0, 0.080, self.agentsNum),
            2.0 * np.pi,
        )

    def __str__(self) -> str:
        return (
            f"ExtremeForcedCircularBoundaryLattice("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.3f},"
            f"m={self.targetMode},v={self.speedV:.1f},N={self.agentsNum},"
            f"dt={self.dt:.3f},snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


def build_model(condition: ExtremeCondition, *, overwrite: bool = False):
    model = ExtremeForcedCircularBoundaryLattice(
        target_mode=condition.target_mode,
        source_family=condition.family,
        source_file=condition.source_file,
        strengthK=condition.strength_k,
        distanceD0=condition.d0,
        phaseLagA0=condition.alpha_over_pi * np.pi,
        boundaryLength=condition.diameter,
        speedV=3.0,
        agentsNum=N_AGENTS,
        dt=DT,
        shotsnaps=SNAPSHOT_INTERVAL,
        randomSeed=condition.seed,
        savePath=str(DATA_DIR),
        overWrite=overwrite,
    )
    model._calc_dot_phase_collision = _calc_dot_phase_collision_fast
    return model


def expected_frames() -> int:
    return ITERATIONS // SNAPSHOT_INTERVAL + 1


def hdf_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with pd.HDFStore(path, mode="r") as store:
            if not {"/positionX", "/phaseTheta"}.issubset(store.keys()):
                return False
            pos = store.get_storer("positionX")
            phase = store.get_storer("phaseTheta")
            rows = expected_frames() * N_AGENTS
            return (
                pos.ncols == 2
                and phase.ncols == 1
                and pos.nrows == rows
                and phase.nrows == rows
            )
    except Exception:
        return False


def attach_metadata(path: Path, condition: ExtremeCondition) -> None:
    metadata = {
        "data_kind": "extreme_mode_forced_initial_condition_then_original_dynamics",
        "scientific_warning": "Intervention trajectory; not spontaneous crystallization.",
        "target_mode": condition.target_mode,
        "geometric_mode_floor_pi_D_over_d0": GEOMETRIC_MODE,
        "mode_ratio_to_geometric": condition.target_mode / GEOMETRIC_MODE,
        "center_angular_jitter_fraction": 0.08,
        "phase_noise_std_radian": 0.08,
        "source_failed_hdf5": condition.source_file,
        "strength_k": STRENGTH_K,
        "d0": D0,
        "alpha_over_pi": ALPHA_OVER_PI,
        "diameter": DIAMETER,
        "seed": SEED,
        "agents_num": N_AGENTS,
        "dt": DT,
        "iterations": ITERATIONS,
        "snapshot_interval": SNAPSHOT_INTERVAL,
    }
    with pd.HDFStore(path, mode="a") as store:
        store.put("metadata", pd.DataFrame([metadata]), format="fixed")
        attrs = store.root._v_attrs
        attrs.data_kind = metadata["data_kind"]
        attrs.scientific_warning = metadata["scientific_warning"]
        attrs.target_mode = condition.target_mode
        attrs.geometric_mode = GEOMETRIC_MODE
        attrs.source_failed_hdf5 = condition.source_file


def simulate_one(condition: ExtremeCondition) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    probe = build_model(condition)
    path = expected_data_path(probe)
    if not hdf_is_complete(path):
        model = build_model(condition, overwrite=path.exists())
        model.run(ITERATIONS)
    if not hdf_is_complete(path):
        raise RuntimeError(f"Incomplete extreme-mode trajectory: {path}")
    attach_metadata(path, condition)
    return str(path)


def ensure_simulations(items: Sequence[ExtremeCondition], workers: int) -> None:
    missing = [item for item in items if not hdf_is_complete(expected_data_path(build_model(item)))]
    if not missing:
        for item in items:
            attach_metadata(expected_data_path(build_model(item)), item)
        print("All extreme-mode trajectories already exist.", flush=True)
        return
    worker_count = min(max(1, workers), 4, len(missing))
    print(
        f"Generating {len(missing)} extreme-mode trajectories with {worker_count} workers; "
        f"m={list(TARGET_MODES)}, N={N_AGENTS}, steps={ITERATIONS}...",
        flush=True,
    )
    if worker_count == 1:
        for index, item in enumerate(missing, start=1):
            simulate_one(item)
            print(f"[{index:02d}/{len(missing):02d}] {item.label}", flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {executor.submit(simulate_one, item): item for item in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            future.result()
            print(f"[{index:02d}/{len(missing):02d}] {item.label}", flush=True)


def load_frames(condition: ExtremeCondition) -> tuple[np.ndarray, np.ndarray, Path]:
    path = expected_data_path(build_model(condition))
    if not hdf_is_complete(path):
        raise RuntimeError(f"Missing extreme-mode HDF5: {path}")
    with pd.HDFStore(path, mode="r") as store:
        positions = store.select("positionX").to_numpy().reshape(-1, N_AGENTS, 2)
        phases = store.select("phaseTheta").to_numpy().reshape(-1, N_AGENTS)
    return positions, phases, path


def frame_metrics(
    positions: np.ndarray, phases: np.ndarray, condition: ExtremeCondition
) -> dict[str, float | int]:
    center = np.array([condition.diameter / 2.0] * 2)
    radius = condition.diameter / 2.0
    relative = positions - center
    radial = np.linalg.norm(relative, axis=1)
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    shell = radius - radial <= 0.25 * condition.d0
    use = shell if np.count_nonzero(shell) >= 20 else np.ones(N_AGENTS, dtype=bool)
    mode, amplitude, _ = base.critical.fourier_fundamental(angles[use])
    target_amplitude = float(
        np.abs(np.mean(np.exp(1j * condition.target_mode * angles[use])))
    )
    tangential = np.sin(phases[use] - angles[use])
    directional = np.abs(tangential) >= 0.2
    handedness = (
        float(np.abs(np.mean(np.sign(tangential[directional]))))
        if np.any(directional)
        else np.nan
    )
    return {
        "dominant_mode": mode,
        "dominant_amplitude": amplitude,
        "target_amplitude": target_amplitude,
        "shell_fraction": float(np.mean(shell)),
        "handedness": handedness,
    }


def analyze(condition: ExtremeCondition) -> tuple[dict[str, object], pd.DataFrame]:
    positions, phases, path = load_frames(condition)
    timeline = pd.DataFrame(
        [frame_metrics(x, theta, condition) for x, theta in zip(positions, phases)]
    )
    timeline.insert(0, "frame", np.arange(len(timeline)))
    timeline.insert(1, "time", timeline["frame"] * SNAPSHOT_INTERVAL * DT)
    tail = timeline.tail(TERMINAL_FRAMES)
    terminal_positions = positions[-1]
    terminal_phases = phases[-1]
    center = np.array([condition.diameter / 2.0] * 2)
    radius = condition.diameter / 2.0
    relative = terminal_positions - center
    radial = np.linalg.norm(relative, axis=1)
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    shell = radius - radial <= 0.25 * condition.d0
    use = shell if np.count_nonzero(shell) >= 20 else np.ones(N_AGENTS, dtype=bool)
    peak_mode = base.critical.periodic_peak_count(angles[use])
    dbscan_mode = base.dbscan_count(terminal_positions[use], condition.d0)
    fourier_mode = int(timeline.iloc[-1]["dominant_mode"])
    observed_mode = base.critical.consensus_mode(fourier_mode, peak_mode, dbscan_mode)
    temporal_mode = int(np.rint(np.median(tail["dominant_mode"])))
    temporal_stability = float(np.mean(tail["dominant_mode"] == temporal_mode))
    formed = bool(
        timeline.iloc[-1]["dominant_amplitude"] >= 0.90
        and np.median(tail["dominant_amplitude"]) >= 0.90
        and temporal_stability >= 0.90
        and timeline.iloc[-1]["shell_fraction"] >= 0.70
        and fourier_mode == temporal_mode == peak_mode == dbscan_mode
    )
    chord, chord_std = base.cluster_chords(
        terminal_positions, use, center, observed_mode, condition.d0
    )
    target_mask = timeline["dominant_mode"].to_numpy() == condition.target_mode
    departures = np.flatnonzero(~target_mask)
    first_departure = (
        float(timeline.iloc[departures[0]]["time"]) if departures.size else np.nan
    )
    result = {
        **asdict(condition),
        "label": condition.label,
        "geometric_mode": GEOMETRIC_MODE,
        "mode_ratio_to_geometric": condition.target_mode / GEOMETRIC_MODE,
        "initial_chord_over_d0": (
            2.0
            * (condition.diameter / 2.0 - 0.035 * condition.d0)
            * np.sin(np.pi / condition.target_mode)
            / condition.d0
        ),
        "target_mode_fraction_all_frames": float(np.mean(target_mask)),
        "first_target_departure_time": first_departure,
        "fourier_mode_terminal": fourier_mode,
        "peak_count_terminal": peak_mode,
        "dbscan_count_terminal": dbscan_mode,
        "observed_mode_terminal": observed_mode,
        "temporal_mode_tail": temporal_mode,
        "temporal_mode_stability_tail": temporal_stability,
        "target_amplitude_min": float(timeline["target_amplitude"].min()),
        "target_amplitude_terminal": float(timeline.iloc[-1]["target_amplitude"]),
        "dominant_amplitude_terminal": float(timeline.iloc[-1]["dominant_amplitude"]),
        "shell_fraction_min": float(timeline["shell_fraction"].min()),
        "shell_fraction_terminal": float(timeline.iloc[-1]["shell_fraction"]),
        "handedness_terminal": float(timeline.iloc[-1]["handedness"]),
        "lattice_formed_terminal": formed,
        "target_mode_retained_terminal": bool(formed and observed_mode == condition.target_mode),
        "actual_chord_over_d0_terminal": chord / condition.d0,
        "actual_chord_std_over_d0_terminal": chord_std / condition.d0,
        "hdf5": str(path),
    }
    return result, timeline


def _draw_frame(
    axis: plt.Axes,
    positions: np.ndarray,
    phases: np.ndarray,
    condition: ExtremeCondition,
    time: float,
    metrics: pd.Series,
) -> None:
    center = np.array([condition.diameter / 2.0] * 2)
    radius = condition.diameter / 2.0
    relative = positions - center
    radial = np.linalg.norm(relative, axis=1)
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    shell = radius - radial <= 0.25 * condition.d0
    use = shell if np.count_nonzero(shell) >= 20 else np.ones(N_AGENTS, dtype=bool)
    peak_count = base.critical.periodic_peak_count(angles[use])
    dbscan_count = base.dbscan_count(positions[use], condition.d0)
    axis.quiver(
        positions[:, 0],
        positions[:, 1],
        np.cos(phases),
        np.sin(phases),
        phases,
        cmap=phaseCmap,
        norm=phaseNorm,
        scale_units="inches",
        scale=20.0,
        width=0.0019,
    )
    axis.add_patch(
        plt.Circle(
            (condition.diameter / 2.0, condition.diameter / 2.0),
            condition.diameter / 2.0,
            fill=False,
            color="black",
            linewidth=0.85,
        )
    )
    pad = 0.025 * condition.diameter
    axis.set_xlim(-pad, condition.diameter + pad)
    axis.set_ylim(-pad, condition.diameter + pad)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        (
            f"t={time:g}\n"
            f"m_F/P/C={int(metrics['dominant_mode'])}/{peak_count}/{dbscan_count}, "
            f"A_target={metrics['target_amplitude']:.2f}, "
            f"f_shell={metrics['shell_fraction']:.2f}"
        ),
        fontsize=8.5,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_keyframe_figures(
    items: Sequence[ExtremeCondition],
    timelines: dict[int, pd.DataFrame],
    measurements: pd.DataFrame,
) -> None:
    key_frames = [int(round(time / (SNAPSHOT_INTERVAL * DT))) for time in KEY_TIMES]
    figure, axes = plt.subplots(
        len(items), len(key_frames), figsize=(3.0 * len(key_frames), 3.15 * len(items)), squeeze=False
    )
    for row, condition in enumerate(items):
        positions, phases, _ = load_frames(condition)
        timeline = timelines[condition.target_mode]
        for col, (frame, time) in enumerate(zip(key_frames, KEY_TIMES)):
            _draw_frame(
                axes[row, col], positions[frame], phases[frame], condition, time, timeline.iloc[frame]
            )
        axes[row, 0].set_ylabel(
            (
                f"target m={condition.target_mode}\n"
                f"m/m0={condition.target_mode / GEOMETRIC_MODE:.2f}"
            ),
            fontsize=10,
        )
    figure.suptitle(
        (
            f"Extreme planted modes at fixed K={STRENGTH_K}, D={DIAMETER}, d0={D0}, "
            f"geometric m0={GEOMETRIC_MODE}\n"
            "INTERVENTION INITIAL CONDITIONS; subsequent frames use original dynamics"
        ),
        fontsize=14,
        y=0.997,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.982))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "Extreme_Mode_Evolution_Keyframes.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    for condition in items:
        positions, phases, _ = load_frames(condition)
        timeline = timelines[condition.target_mode]
        fig, local_axes = plt.subplots(1, len(key_frames), figsize=(3.0 * len(key_frames), 3.25))
        for axis, frame, time in zip(local_axes, key_frames, KEY_TIMES):
            _draw_frame(axis, positions[frame], phases[frame], condition, time, timeline.iloc[frame])
        row = measurements.loc[measurements["target_mode"] == condition.target_mode].iloc[0]
        fig.suptitle(
            (
                f"{condition.label}: final m={int(row['observed_mode_terminal'])}, "
                f"retained={bool(row['target_mode_retained_terminal'])}"
            ),
            fontsize=13,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        fig.savefig(
            OUTPUT_DIR / f"Keyframes_{condition.label}.png", dpi=220, bbox_inches="tight"
        )
        plt.close(fig)


def make_timeline_figure(
    items: Sequence[ExtremeCondition], timelines: dict[int, pd.DataFrame]
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for condition in items:
        timeline = timelines[condition.target_mode]
        label = f"target m={condition.target_mode} ({condition.target_mode / GEOMETRIC_MODE:.2f} m0)"
        axes[0].plot(timeline["time"], timeline["dominant_mode"], label=label)
        axes[1].plot(timeline["time"], timeline["target_amplitude"], label=label)
        axes[2].plot(timeline["time"], timeline["shell_fraction"], label=label)
    axes[0].axhline(GEOMETRIC_MODE, color="black", linestyle="--", linewidth=1.0, label="geometric m0")
    axes[0].set_ylabel("Dominant mode")
    axes[1].set_ylabel("Target-mode amplitude")
    axes[2].set_ylabel("Shell fraction")
    axes[2].set_xlabel("Time")
    axes[1].axhline(0.9, color="gray", linestyle=":", linewidth=1.0)
    axes[2].axhline(0.7, color="gray", linestyle=":", linewidth=1.0)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle("Stability of strongly underfilled and overfilled planted modes")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "Extreme_Mode_Timelines.png", dpi=220, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "Extreme_Mode_Timelines.pdf", bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    items: Sequence[ExtremeCondition],
    measurements: pd.DataFrame,
    timelines: dict[int, pd.DataFrame],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(OUTPUT_DIR / "Extreme_Mode_Measurements.csv", index=False)
    for mode, timeline in timelines.items():
        timeline.to_csv(OUTPUT_DIR / f"Timeline_m{mode}.csv", index=False)
    configuration = {
        "warning": "Intervention trajectories; not spontaneous crystallization.",
        "strength_k": STRENGTH_K,
        "d0": D0,
        "alpha_over_pi": ALPHA_OVER_PI,
        "diameter": DIAMETER,
        "seed": SEED,
        "geometric_mode": GEOMETRIC_MODE,
        "target_modes": list(TARGET_MODES),
        "key_times": list(KEY_TIMES),
        "agents_num": N_AGENTS,
        "dt": DT,
        "iterations": ITERATIONS,
        "snapshot_interval": SNAPSHOT_INTERVAL,
    }
    (OUTPUT_DIR / "Extreme_Mode_Configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_keyframe_figures(items, timelines, measurements)
    make_timeline_figure(items, timelines)
    columns = [
        "target_mode",
        "geometric_mode",
        "mode_ratio_to_geometric",
        "initial_chord_over_d0",
        "target_mode_fraction_all_frames",
        "first_target_departure_time",
        "observed_mode_terminal",
        "temporal_mode_stability_tail",
        "target_amplitude_min",
        "target_amplitude_terminal",
        "shell_fraction_min",
        "lattice_formed_terminal",
        "target_mode_retained_terminal",
        "actual_chord_over_d0_terminal",
    ]
    report = [
        "# Extreme planted boundary-mode stability",
        "",
        "> These are intervention trajectories. They test nonlinear stability, not spontaneous accessibility.",
        "",
        f"Fixed K={STRENGTH_K}, alpha/pi={ALPHA_OVER_PI}, D={DIAMETER}, d0={D0}, "
        f"N={N_AGENTS}, steps={ITERATIONS}; geometric reference m0={GEOMETRIC_MODE}.",
        "",
        measurements[columns].to_markdown(index=False),
        "",
        "The planted states include 8% cluster-center angular jitter and 0.08 rad heading noise.",
    ]
    (OUTPUT_DIR / "Extreme_Mode_Report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test extreme planted cluster counts.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--simulate-only", action="store_true")
    group.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = conditions()
    if not args.analyze_only:
        ensure_simulations(items, args.workers)
    if args.simulate_only:
        return
    rows = []
    timelines: dict[int, pd.DataFrame] = {}
    for index, item in enumerate(items, start=1):
        row, timeline = analyze(item)
        rows.append(row)
        timelines[item.target_mode] = timeline
        print(f"Analyzed [{index:02d}/{len(items):02d}] {item.label}", flush=True)
    write_outputs(items, pd.DataFrame(rows), timelines)


if __name__ == "__main__":
    main()
