"""Plant boundary lattices in failed runs, evolve them, and render videos.

The generated trajectories are intervention experiments, not spontaneous
crystallization data.  Every output filename and HDF5 metadata record carries
that distinction.  Original trajectories and manuscript/reference files are
read-only inputs and are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import critical_boundary_lattice_analysis as critical
from CircularFigure import expected_data_path
from main import CircularBoundaryPatternFormation, phaseCmap, phaseNorm
from small_circular_alpha_sweep import _calc_dot_phase_collision_fast


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "forced_boundary_lattice"
OUTPUT_DIR = PROJECT_DIR / "output" / "Forced_Boundary_Lattice"
VIDEO_DIR = OUTPUT_DIR / "Videos"
CRITICAL_MEASUREMENTS = (
    PROJECT_DIR
    / "output"
    / "Critical_Boundary_Lattice_Quantization"
    / "Boundary_Lattice_Quantization_Measurements.csv"
)
D0_MEASUREMENTS = (
    PROJECT_DIR
    / "output"
    / "Boundary_Lattice_D0_Sweep"
    / "D0_Sweep_Measurements.csv"
)

N_AGENTS = 2000
DT = 0.005
ITERATIONS = 50_000
SNAPSHOT_INTERVAL = 100
TERMINAL_FRAMES = 20
VIDEO_MAX_FRAMES = 251
VIDEO_FPS = 25
VIDEO_DPI = 120


@dataclass(frozen=True, order=True)
class ForcedCondition:
    family: str
    strength_k: float
    d0: float
    alpha_over_pi: float
    diameter: float
    seed: int
    source_file: str

    @property
    def target_mode(self) -> int:
        # Choosing the lower compatible integer keeps the nominal arc spacing
        # at least d0.  The observed stable modes in this project differ by at
        # most one from this rule over the selected diameter range.
        return max(3, int(np.floor(np.pi * self.diameter / self.d0)))

    @property
    def label(self) -> str:
        return (
            f"K{self.strength_k:g}_d0{self.d0:g}_D{self.diameter:g}_"
            f"a{self.alpha_over_pi:g}pi_seed{self.seed}_m{self.target_mode}"
        ).replace(".", "p")


def _as_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def selected_conditions() -> list[ForcedCondition]:
    """Resolve six documented failed runs from existing measurement tables."""

    if not CRITICAL_MEASUREMENTS.is_file() or not D0_MEASUREMENTS.is_file():
        raise FileNotFoundError(
            "Required failure inventories are missing. Run the critical and d0 "
            "analyses before this intervention experiment."
        )

    critical_rows = pd.read_csv(CRITICAL_MEASUREMENTS)
    critical_failed = critical_rows[
        np.isclose(critical_rows["alpha_over_pi"], 0.5)
        & ~_as_boolean(critical_rows["lattice_formed"])
    ].copy()
    conditions = [
        ForcedCondition(
            family="critical_failure",
            strength_k=20.75,
            d0=1.0,
            alpha_over_pi=float(row.alpha_over_pi),
            diameter=float(row.diameter),
            seed=int(row.seed),
            source_file=str(row.source_file),
        )
        for row in critical_failed.itertuples(index=False)
    ]

    d0_rows = pd.read_csv(D0_MEASUREMENTS)
    d0_failed = d0_rows[
        np.isclose(d0_rows["d0"], 1.25)
        & np.isclose(d0_rows["diameter"], 4.58)
        & (d0_rows["seed"] == 9)
        & ~_as_boolean(d0_rows["lattice_formed"])
    ]
    conditions.extend(
        ForcedCondition(
            family="d0_failure",
            strength_k=40.0,
            d0=float(row.d0),
            alpha_over_pi=0.5,
            diameter=float(row.diameter),
            seed=int(row.seed),
            source_file=str(row.source_file),
        )
        for row in d0_failed.itertuples(index=False)
    )
    conditions = sorted(conditions, key=lambda item: (item.diameter / item.d0, item.family))
    if len(conditions) != 6:
        raise RuntimeError(
            f"Expected six selected failed states, resolved {len(conditions)}."
        )
    missing_sources = [item.source_file for item in conditions if not Path(item.source_file).is_file()]
    if missing_sources:
        listing = "\n".join(f"  - {path}" for path in missing_sources)
        raise FileNotFoundError(f"Original failed HDF5 source(s) missing:\n{listing}")
    return conditions


class ForcedCircularBoundaryLattice(CircularBoundaryPatternFormation):
    """The original circular model with an explicitly planted lattice at t=0."""

    def __init__(
        self,
        *,
        target_mode: int,
        source_family: str,
        source_file: str,
        strengthK: float,
        distanceD0: float,
        phaseLagA0: float,
        boundaryLength: float,
        speedV: float = 3.0,
        agentsNum: int = N_AGENTS,
        dt: float = DT,
        shotsnaps: int = SNAPSHOT_INTERVAL,
        randomSeed: int = 9,
        savePath: str | None = None,
        overWrite: bool = False,
    ) -> None:
        self.targetMode = int(target_mode)
        self.sourceFamily = str(source_family)
        self.sourceFile = str(source_file)
        super().__init__(
            strengthK=strengthK,
            distanceD0=distanceD0,
            phaseLagA0=phaseLagA0,
            boundaryLength=boundaryLength,
            speedV=speedV,
            freqDist="uniform",
            omegaMin=0.0,
            deltaOmega=0.0,
            agentsNum=agentsNum,
            dt=dt,
            tqdm=False,
            savePath=savePath,
            shotsnaps=shotsnaps,
            randomSeed=randomSeed,
            overWrite=overWrite,
        )
        self._plant_boundary_lattice()

    def _plant_boundary_lattice(self) -> None:
        rng = np.random.default_rng(self.randomSeed + 4919 * self.targetMode)
        mode = self.targetMode
        spacing_angle = 2.0 * np.pi / mode
        global_angle = rng.uniform(0.0, 2.0 * np.pi)
        center_jitter = rng.normal(0.0, 0.015 * spacing_angle, mode)
        center_jitter -= np.mean(center_jitter)
        cluster_angles = np.mod(
            global_angle + spacing_angle * np.arange(mode) + center_jitter,
            2.0 * np.pi,
        )

        counts = np.full(mode, self.agentsNum // mode, dtype=int)
        counts[: self.agentsNum % mode] += 1
        cluster_ids = np.repeat(np.arange(mode), counts)
        rng.shuffle(cluster_ids)

        # The centroid shell offset and cluster widths are taken from the
        # naturally formed K sweeps in this project, expressed in d0 units.
        target_radius = self.circleRadius - 0.025 * self.distanceD0
        tangent_offset = rng.normal(0.0, 0.035 * self.distanceD0, self.agentsNum)
        inward_offset = np.abs(
            rng.normal(0.0, 0.020 * self.distanceD0, self.agentsNum)
        )
        particle_angles = (
            cluster_angles[cluster_ids] + tangent_offset / target_radius
        )
        particle_radius = np.maximum(
            0.0, target_radius - inward_offset
        )
        self.positionX = self.circleCenter + np.column_stack(
            [
                particle_radius * np.cos(particle_angles),
                particle_radius * np.sin(particle_angles),
            ]
        )

        # Positive tangential headings reproduce the handedness of all
        # naturally formed alpha=pi/2 samples in the comparison dataset.
        phase_noise = rng.normal(0.0, 0.010, self.agentsNum)
        self.phaseTheta = np.mod(
            particle_angles + 0.5 * np.pi + phase_noise,
            2.0 * np.pi,
        )

    def __str__(self) -> str:
        return (
            f"ForcedCircularBoundaryLattice("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.3f},"
            f"m={self.targetMode},v={self.speedV:.1f},N={self.agentsNum},"
            f"dt={self.dt:.3f},snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


def build_model(condition: ForcedCondition, *, overwrite: bool = False):
    model = ForcedCircularBoundaryLattice(
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


def expected_frame_count() -> int:
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
            rows = expected_frame_count() * N_AGENTS
            return (
                pos.ncols == 2
                and phase.ncols == 1
                and pos.nrows == rows
                and phase.nrows == rows
            )
    except Exception:
        return False


def attach_metadata(path: Path, condition: ForcedCondition) -> None:
    metadata = {
        "data_kind": "forced_initial_condition_then_original_model_dynamics",
        "scientific_warning": (
            "Intervention trajectory; not evidence of spontaneous crystallization."
        ),
        "initialization": "approximately equally spaced coherent boundary clusters",
        "target_mode_formula": "floor(pi * diameter / d0)",
        "target_mode": condition.target_mode,
        "source_failed_hdf5": condition.source_file,
        "source_family": condition.family,
        "strength_k": condition.strength_k,
        "d0": condition.d0,
        "alpha_over_pi": condition.alpha_over_pi,
        "diameter": condition.diameter,
        "seed": condition.seed,
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
        attrs.target_mode_formula = metadata["target_mode_formula"]
        attrs.source_failed_hdf5 = condition.source_file


def simulate_one(condition: ForcedCondition) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    probe = build_model(condition)
    path = expected_data_path(probe)
    if hdf_is_complete(path):
        attach_metadata(path, condition)
        return str(path)
    model = build_model(condition, overwrite=path.exists())
    model.run(ITERATIONS)
    if not hdf_is_complete(path):
        raise RuntimeError(f"Incomplete forced trajectory: {path}")
    attach_metadata(path, condition)
    return str(path)


def ensure_simulations(conditions: Sequence[ForcedCondition], workers: int) -> None:
    missing = [
        condition
        for condition in conditions
        if not hdf_is_complete(expected_data_path(build_model(condition)))
    ]
    if not missing:
        for condition in conditions:
            attach_metadata(expected_data_path(build_model(condition)), condition)
        print("All forced-lattice HDF5 trajectories already exist.", flush=True)
        return
    worker_count = min(max(1, workers), 4, len(missing))
    print(
        f"Generating {len(missing)} forced trajectories with {worker_count} workers; "
        f"N={N_AGENTS}, steps={ITERATIONS}, snap={SNAPSHOT_INTERVAL}...",
        flush=True,
    )
    if worker_count == 1:
        for index, condition in enumerate(missing, start=1):
            simulate_one(condition)
            print(
                f"[{index:02d}/{len(missing):02d}] {condition.label}", flush=True
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
                f"[{index:02d}/{len(missing):02d}] {condition.label}", flush=True
            )


def load_forced_frames(
    condition: ForcedCondition, *, terminal_only: bool = False
) -> tuple[np.ndarray, np.ndarray, Path]:
    path = expected_data_path(build_model(condition))
    if not hdf_is_complete(path):
        raise RuntimeError(f"Missing forced trajectory: {path}")
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


def load_source_terminal(condition: ForcedCondition) -> tuple[np.ndarray, np.ndarray]:
    path = Path(condition.source_file)
    with pd.HDFStore(path, mode="r") as store:
        pos_rows = store.get_storer("positionX").nrows
        phase_rows = store.get_storer("phaseTheta").nrows
        positions = store.select(
            "positionX", start=max(0, pos_rows - N_AGENTS)
        ).to_numpy()
        phases = store.select(
            "phaseTheta", start=max(0, phase_rows - N_AGENTS)
        ).to_numpy().reshape(-1)
    if positions.shape != (N_AGENTS, 2) or phases.shape != (N_AGENTS,):
        raise RuntimeError(f"Unexpected source HDF5 terminal shape: {path}")
    return positions, phases


def dbscan_count(positions: np.ndarray, d0: float) -> int:
    labels = DBSCAN(eps=0.10 * d0, min_samples=10).fit_predict(positions)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return 0
    _, counts = np.unique(valid, return_counts=True)
    return int(np.count_nonzero(counts >= 20))


def cluster_chords(
    positions: np.ndarray,
    shell: np.ndarray,
    center: np.ndarray,
    expected_mode: int,
    d0: float,
) -> tuple[float, float]:
    shell_positions = positions[shell]
    labels = DBSCAN(eps=0.10 * d0, min_samples=10).fit_predict(shell_positions)
    centers = []
    for label in sorted(set(labels) - {-1}):
        members = shell_positions[labels == label]
        if members.shape[0] >= 20:
            centers.append(np.mean(members, axis=0))
    if len(centers) != expected_mode or expected_mode < 2:
        return np.nan, np.nan
    centers_array = np.asarray(centers)
    relative = centers_array - center
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    ordered = centers_array[np.argsort(angles)]
    chords = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    return float(np.mean(chords)), float(np.std(chords, ddof=1))


def measure_condition(condition: ForcedCondition) -> dict[str, object]:
    positions, phases, path = load_forced_frames(condition, terminal_only=True)
    center = np.array([condition.diameter / 2.0] * 2)
    radius = condition.diameter / 2.0
    modes: list[int] = []
    amplitudes: list[float] = []
    terminal = None
    for frame_positions, frame_phases in zip(positions, phases):
        relative = frame_positions - center
        radial = np.linalg.norm(relative, axis=1)
        wall_distance = radius - radial
        angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
        physical_shell = wall_distance <= 0.25 * condition.d0
        shell = physical_shell if np.count_nonzero(physical_shell) >= 20 else np.ones(N_AGENTS, bool)
        mode, amplitude, _ = critical.fourier_fundamental(angles[shell])
        modes.append(mode)
        amplitudes.append(amplitude)
        terminal = (
            frame_positions,
            frame_phases,
            angles,
            wall_distance,
            physical_shell,
            shell,
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
        fourier_mode,
        fourier_amplitude,
    ) = terminal
    peak_mode = critical.periodic_peak_count(angles[shell])
    dbscan_mode = dbscan_count(frame_positions[shell], condition.d0)
    observed_mode = critical.consensus_mode(fourier_mode, peak_mode, dbscan_mode)
    temporal_mode = int(np.rint(np.median(modes)))
    stability = float(np.mean(np.asarray(modes) == temporal_mode))
    median_amplitude = float(np.median(amplitudes))
    shell_fraction = float(np.mean(physical_shell))
    formed = bool(
        fourier_amplitude >= 0.90
        and median_amplitude >= 0.90
        and stability >= 0.90
        and shell_fraction >= 0.70
        and fourier_mode == temporal_mode
        and peak_mode == fourier_mode
        and dbscan_mode == fourier_mode
    )
    chord, chord_std = cluster_chords(
        frame_positions, shell, center, observed_mode, condition.d0
    )
    tangential = np.sin(frame_phases[shell] - angles[shell])
    directional = np.abs(tangential) >= 0.2
    handedness = (
        float(np.abs(np.mean(np.sign(tangential[directional]))))
        if np.any(directional)
        else np.nan
    )
    return {
        **asdict(condition),
        "diameter_over_d0": condition.diameter / condition.d0,
        "target_mode": condition.target_mode,
        "target_arc_over_d0": (
            np.pi * condition.diameter / condition.target_mode / condition.d0
        ),
        "fourier_mode_terminal": fourier_mode,
        "peak_count_terminal": peak_mode,
        "dbscan_count_terminal": dbscan_mode,
        "observed_mode_terminal": observed_mode,
        "temporal_mode_median": temporal_mode,
        "target_mode_fraction_tail": float(
            np.mean(np.asarray(modes) == condition.target_mode)
        ),
        "fourier_amplitude_terminal": fourier_amplitude,
        "temporal_amplitude_median": median_amplitude,
        "temporal_mode_stability": stability,
        "shell_particle_fraction": shell_fraction,
        "lattice_formed_terminal": formed,
        "target_mode_retained": bool(formed and observed_mode == condition.target_mode),
        "heading_handedness_terminal": handedness,
        "actual_chord_mean": chord,
        "actual_chord_std": chord_std,
        "actual_chord_over_d0": chord / condition.d0,
        "wall_distance_median_over_d0": float(np.median(wall_distance)) / condition.d0,
        "forced_hdf5": str(path),
    }


def _draw_state(
    axis: plt.Axes,
    positions: np.ndarray,
    phases: np.ndarray,
    condition: ForcedCondition,
    title: str,
) -> None:
    axis.quiver(
        positions[:, 0],
        positions[:, 1],
        np.cos(phases),
        np.sin(phases),
        phases,
        cmap=phaseCmap,
        norm=phaseNorm,
        scale_units="inches",
        scale=18.0,
        width=0.0021,
    )
    circle = plt.Circle(
        (condition.diameter / 2.0, condition.diameter / 2.0),
        condition.diameter / 2.0,
        fill=False,
        color="black",
        linewidth=1.0,
    )
    axis.add_patch(circle)
    pad = 0.03 * condition.diameter
    axis.set_xlim(-pad, condition.diameter + pad)
    axis.set_ylim(-pad, condition.diameter + pad)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_comparison_figure(
    conditions: Sequence[ForcedCondition], measurements: pd.DataFrame
) -> Path:
    rows = len(conditions)
    figure, axes = plt.subplots(rows, 3, figsize=(10.5, 3.25 * rows), squeeze=False)
    by_label = measurements.set_index("label") if "label" in measurements.columns else None
    for row_index, condition in enumerate(conditions):
        source_positions, source_phases = load_source_terminal(condition)
        forced_positions, forced_phases, _ = load_forced_frames(condition)
        measurement = measurements.iloc[row_index] if by_label is None else by_label.loc[condition.label]
        _draw_state(
            axes[row_index, 0],
            source_positions,
            source_phases,
            condition,
            f"Original failed terminal\nD={condition.diameter:g}, d0={condition.d0:g}",
        )
        _draw_state(
            axes[row_index, 1],
            forced_positions[0],
            forced_phases[0],
            condition,
            f"Forced t=0, target m={condition.target_mode}",
        )
        status = "formed" if bool(measurement["lattice_formed_terminal"]) else "not formed"
        _draw_state(
            axes[row_index, 2],
            forced_positions[-1],
            forced_phases[-1],
            condition,
            (
                f"Dynamics t={ITERATIONS * DT:g}, "
                f"m={int(measurement['observed_mode_terminal'])} ({status})"
            ),
        )
    figure.suptitle(
        "Forced boundary lattice intervention: original failure -> planted state -> dynamics",
        fontsize=14,
        y=0.998,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.994))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "Forced_Boundary_Lattice_Comparison.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def render_video(condition: ForcedCondition) -> Path:
    positions, phases, _ = load_forced_frames(condition)
    frame_count = positions.shape[0]
    keep = min(frame_count, VIDEO_MAX_FRAMES)
    frame_indices = np.unique(
        np.rint(np.linspace(0, frame_count - 1, keep)).astype(int)
    )
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output = VIDEO_DIR / f"ForcedBoundaryLattice_{condition.label}.mp4"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; cannot render MP4.")
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg

    figure, axis = plt.subplots(figsize=(6.4, 6.4))
    first = frame_indices[0]
    quiver = axis.quiver(
        positions[first, :, 0],
        positions[first, :, 1],
        np.cos(phases[first]),
        np.sin(phases[first]),
        phases[first],
        cmap=phaseCmap,
        norm=phaseNorm,
        scale_units="inches",
        scale=16.0,
        width=0.0021,
    )
    circle = plt.Circle(
        (condition.diameter / 2.0, condition.diameter / 2.0),
        condition.diameter / 2.0,
        fill=False,
        color="black",
        linewidth=1.2,
    )
    axis.add_patch(circle)
    pad = 0.025 * condition.diameter
    axis.set_xlim(-pad, condition.diameter + pad)
    axis.set_ylim(-pad, condition.diameter + pad)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        (
            f"Forced boundary lattice -> original dynamics\n"
            f"K={condition.strength_k:g}, d0={condition.d0:g}, "
            f"D={condition.diameter:g}, alpha={condition.alpha_over_pi:g}pi, "
            f"target m={condition.target_mode}"
        ),
        fontsize=12,
    )
    time_text = axis.text(
        0.02,
        0.025,
        "",
        transform=axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    axis.text(
        0.98,
        0.025,
        "INTERVENTION / NOT SPONTANEOUS",
        transform=axis.transAxes,
        ha="right",
        fontsize=8,
        color="#8B1A1A",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    writer = animation.FFMpegWriter(
        fps=VIDEO_FPS,
        codec="libx264",
        bitrate=-1,
        metadata={
            "title": f"Forced boundary lattice {condition.label}",
            "comment": "Planted initial condition followed by original model dynamics.",
        },
        extra_args=[
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    with writer.saving(figure, str(output), VIDEO_DPI):
        for frame_index in frame_indices:
            quiver.set_offsets(positions[frame_index])
            quiver.set_UVC(
                np.cos(phases[frame_index]),
                np.sin(phases[frame_index]),
                phases[frame_index],
            )
            iteration = frame_index * SNAPSHOT_INTERVAL
            time_text.set_text(
                f"t={iteration * DT:6.1f}  (iteration {iteration:5d})"
            )
            writer.grab_frame()
    plt.close(figure)
    return output


def write_outputs(
    conditions: Sequence[ForcedCondition], measurements: pd.DataFrame
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements = measurements.copy()
    measurements.insert(0, "label", [item.label for item in conditions])
    measurements.to_csv(
        OUTPUT_DIR / "Forced_Boundary_Lattice_Measurements.csv", index=False
    )
    configuration = {
        "data_kind": "forced_initial_condition_then_original_model_dynamics",
        "warning": "Not evidence of spontaneous crystallization.",
        "agents_num": N_AGENTS,
        "dt": DT,
        "iterations": ITERATIONS,
        "snapshot_interval": SNAPSHOT_INTERVAL,
        "target_mode_formula": "floor(pi * diameter / d0)",
        "conditions": [
            {**asdict(item), "target_mode": item.target_mode, "label": item.label}
            for item in conditions
        ],
    }
    (OUTPUT_DIR / "Forced_Boundary_Lattice_Configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_comparison_figure(conditions, measurements)

    columns = [
        "family",
        "strength_k",
        "d0",
        "diameter",
        "seed",
        "target_mode",
        "observed_mode_terminal",
        "target_mode_fraction_tail",
        "lattice_formed_terminal",
        "target_mode_retained",
        "fourier_amplitude_terminal",
        "temporal_mode_stability",
        "shell_particle_fraction",
        "actual_chord_over_d0",
    ]
    report_lines = [
        "# Forced boundary-lattice intervention",
        "",
        "> **Scientific status:** These are planted/intervention trajectories, not "
        "evidence of spontaneous crystallization. The first frame is deliberately "
        "rearranged; all later frames follow the unmodified model dynamics.",
        "",
        f"N={N_AGENTS}, dt={DT}, steps={ITERATIONS}, snap={SNAPSHOT_INTERVAL}; "
        "target mode `floor(pi * D / d0)`.",
        "",
        "## Terminal diagnostics",
        "",
        measurements[columns].to_markdown(index=False),
        "",
        "## Provenance",
        "",
        "Each HDF5 file contains `/positionX`, `/phaseTheta`, and `/metadata`. "
        "The metadata records the original failed HDF5 path and labels the output "
        "as an intervention trajectory. Original HDF5 files are untouched.",
    ]
    (OUTPUT_DIR / "Forced_Boundary_Lattice_Report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plant failed circular states into boundary lattices and render MP4s."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--simulate-only", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    modes.add_argument("--videos-only", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = selected_conditions()
    if not args.analyze_only and not args.videos_only:
        ensure_simulations(conditions, args.workers)
    if args.simulate_only:
        return
    if not args.videos_only:
        rows = []
        for index, condition in enumerate(conditions, start=1):
            rows.append(measure_condition(condition))
            print(f"Measured [{index:02d}/{len(conditions):02d}] {condition.label}")
        measurements = pd.DataFrame(rows)
        write_outputs(conditions, measurements)
    if not args.skip_videos:
        for index, condition in enumerate(conditions, start=1):
            output = render_video(condition)
            print(
                f"Rendered [{index:02d}/{len(conditions):02d}] {output.name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
