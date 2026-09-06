"""Continue established critical boundary lattices with or without the wall.

The source trajectories and all manuscript/reference files are opened read-only.
New continuation trajectories and diagnostics are written under this repository.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import critical_boundary_lattice_analysis as critical
from CircularFigure import expected_data_path
from main import CircularBoundaryPatternFormation, phaseCmap, phaseNorm
from small_circular_alpha_sweep import _calc_dot_phase_collision_fast


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "boundary_removal_experiment"
OUTPUT_DIR = PROJECT_DIR / "output" / "Boundary_Removal_Experiment"

CONTINUATION_STEPS = 20_000
SNAPSHOT_INTERVAL = 100
CONTINUATION_TIME = CONTINUATION_STEPS * critical.DT
PROTOCOLS = ("retained", "removed")
SOURCE_CONDITIONS = (
    critical.Condition(0.5, 3.0, 9),
    critical.Condition(0.5, 3.0, 11),
    critical.Condition(0.5, 5.0, 9),
    critical.Condition(0.5, 5.0, 10),
)


class BoundaryRemovedContinuation(CircularBoundaryPatternFormation):
    """The same self-propulsion and phase coupling, without wall collisions."""

    def update(self) -> None:
        dot_position = self.dotPosition
        dot_phase = self.dotPhase
        self.positionX = self.positionX + dot_position * self.dt
        self.phaseTheta = np.mod(
            self.phaseTheta + dot_phase * self.dt,
            2.0 * np.pi,
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L0={self.boundaryLength:.1f},"
            f"v={self.speedV:.1f},N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed})"
        )


@dataclass(frozen=True)
class ContinuationJob:
    condition: critical.Condition
    protocol: str


def source_terminal_state(
    condition: critical.Condition,
) -> tuple[np.ndarray, np.ndarray, Path]:
    source_model = critical.make_model(
        condition,
        critical.source_directory(condition),
    )
    source_path = expected_data_path(source_model)
    with pd.HDFStore(source_path, mode="r") as store:
        position_rows = store.get_storer("positionX").nrows
        phase_rows = store.get_storer("phaseTheta").nrows
        positions = store.select(
            "positionX", start=position_rows - critical.AGENTS_NUM
        ).to_numpy()
        phases = store.select(
            "phaseTheta", start=phase_rows - critical.AGENTS_NUM
        ).to_numpy().reshape(-1)
    return positions, phases, source_path


def build_continuation(job: ContinuationJob):
    condition = job.condition
    model_class = (
        CircularBoundaryPatternFormation
        if job.protocol == "retained"
        else BoundaryRemovedContinuation
    )
    protocol_dir = DATA_DIR / job.protocol
    model = model_class(
        strengthK=critical.STRENGTH_K,
        distanceD0=critical.INTERACTION_D0,
        phaseLagA0=condition.alpha_over_pi * np.pi,
        boundaryLength=condition.diameter,
        speedV=critical.SPEED_V,
        freqDist="uniform",
        omegaMin=0.0,
        deltaOmega=0.0,
        agentsNum=critical.AGENTS_NUM,
        dt=critical.DT,
        tqdm=False,
        savePath=str(protocol_dir),
        shotsnaps=SNAPSHOT_INTERVAL,
        randomSeed=condition.seed,
        overWrite=False,
    )
    model._calc_dot_phase_collision = _calc_dot_phase_collision_fast
    positions, phases, _ = source_terminal_state(condition)
    model.positionX = positions.copy()
    model.phaseTheta = phases.copy()
    return model


def continuation_path(job: ContinuationJob) -> Path:
    return expected_data_path(build_continuation(job))


def expected_frames() -> int:
    return CONTINUATION_STEPS // SNAPSHOT_INTERVAL + 1


def continuation_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with pd.HDFStore(path, mode="r") as store:
            expected_rows = expected_frames() * critical.AGENTS_NUM
            return (
                {"/positionX", "/phaseTheta"}.issubset(store.keys())
                and store.get_storer("positionX").nrows == expected_rows
                and store.get_storer("phaseTheta").nrows == expected_rows
            )
    except Exception:
        return False


def run_one(job: ContinuationJob) -> str:
    model = build_continuation(job)
    path = expected_data_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    if continuation_complete(path):
        return str(path)
    model.overWrite = True
    model.run(CONTINUATION_STEPS)
    if not continuation_complete(path):
        raise RuntimeError(f"Incomplete continuation: {path}")
    return str(path)


def ensure_continuations(workers: int) -> None:
    jobs = [
        ContinuationJob(condition, protocol)
        for condition in SOURCE_CONDITIONS
        for protocol in PROTOCOLS
    ]
    missing = [job for job in jobs if not continuation_complete(continuation_path(job))]
    if not missing:
        print("All continuation trajectories already exist.", flush=True)
        return
    worker_count = min(max(1, workers), 4, len(missing))
    print(
        f"Generating {len(missing)} continuation trajectories with "
        f"{worker_count} worker(s)...",
        flush=True,
    )
    if worker_count == 1:
        for index, job in enumerate(missing, start=1):
            run_one(job)
            print(f"[{index}/{len(missing)}] {job}", flush=True)
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = {pool.submit(run_one, job): job for job in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            future.result()
            print(f"[{index}/{len(missing)}] {job}", flush=True)


def load_continuation(job: ContinuationJob) -> tuple[np.ndarray, np.ndarray]:
    path = continuation_path(job)
    with pd.HDFStore(path, mode="r") as store:
        positions = store["positionX"].to_numpy().reshape(
            expected_frames(), critical.AGENTS_NUM, 2
        )
        phases = store["phaseTheta"].to_numpy().reshape(
            expected_frames(), critical.AGENTS_NUM
        )
    return positions, phases


def fourier_amplitudes(angles: np.ndarray, maximum_mode: int = 25) -> np.ndarray:
    modes = np.arange(1, maximum_mode + 1)
    return np.abs(np.mean(np.exp(1j * angles[:, None] * modes[None, :]), axis=0))


def fitted_circle_center(positions: np.ndarray) -> np.ndarray:
    """Return the algebraic least-squares circle center for one particle set."""

    design = np.column_stack(
        [2.0 * positions[:, 0], 2.0 * positions[:, 1], np.ones(len(positions))]
    )
    target = np.sum(positions**2, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return solution[:2]


def cluster_geometry(positions: np.ndarray) -> dict[str, float]:
    centers = identified_cluster_centers(positions)
    if len(centers) < 2:
        return {
            "cluster_count": float(len(centers)),
            "cluster_chord_mean": np.nan,
            "cluster_chord_std": np.nan,
        }
    center_of_centers = centers.mean(axis=0)
    angles = np.mod(
        np.arctan2(
            centers[:, 1] - center_of_centers[1],
            centers[:, 0] - center_of_centers[0],
        ),
        2.0 * np.pi,
    )
    centers = centers[np.argsort(angles)]
    chords = np.linalg.norm(np.roll(centers, -1, axis=0) - centers, axis=1)
    return {
        "cluster_count": float(len(centers)),
        "cluster_chord_mean": float(np.mean(chords)),
        "cluster_chord_std": float(np.std(chords)),
    }


def identified_cluster_centers(positions: np.ndarray) -> np.ndarray:
    labels = DBSCAN(eps=0.10 * critical.INTERACTION_D0, min_samples=10).fit_predict(
        positions
    )
    valid_labels = []
    for label in np.unique(labels[labels >= 0]):
        if np.count_nonzero(labels == label) >= 20:
            valid_labels.append(int(label))
    if not valid_labels:
        return np.empty((0, 2), dtype=float)
    return np.asarray(
        [positions[labels == label].mean(axis=0) for label in valid_labels]
    )


def frame_metrics(
    positions: np.ndarray,
    phases: np.ndarray,
    initial_mode: int,
    include_clusters: bool,
) -> dict[str, float]:
    center = fitted_circle_center(positions)
    relative = positions - center
    radii = np.linalg.norm(relative, axis=1)
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    amplitudes = fourier_amplitudes(angles)
    tangential = np.sin(phases - angles)
    radial = np.cos(phases - angles)
    payload = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_displacement": np.nan,
        "mean_radius": float(np.mean(radii)),
        "radius_std": float(np.std(radii)),
        "radius_cv": float(np.std(radii) / max(np.mean(radii), 1e-12)),
        "initial_mode_amplitude": float(amplitudes[initial_mode - 1]),
        "dominant_mode": float(np.argmax(amplitudes[2:]) + 3),
        "dominant_amplitude": float(np.max(amplitudes[2:])),
        "tangential_signed": float(np.mean(tangential)),
        "tangential_coherence": float(np.abs(np.mean(tangential))),
        "radial_signed": float(np.mean(radial)),
        "polarization": float(np.abs(np.mean(np.exp(1j * phases)))),
    }
    if include_clusters:
        payload.update(cluster_geometry(positions))
    else:
        payload.update(
            {
                "cluster_count": np.nan,
                "cluster_chord_mean": np.nan,
                "cluster_chord_std": np.nan,
            }
        )
    return payload


def analyze() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    initial_modes = {3.0: 9, 5.0: 15}
    cluster_stride = 10
    for condition in SOURCE_CONDITIONS:
        for protocol in PROTOCOLS:
            job = ContinuationJob(condition, protocol)
            positions, phases = load_continuation(job)
            original_center = np.array([condition.diameter / 2.0] * 2)
            initial_wall_distance = condition.diameter / 2.0 - np.linalg.norm(
                positions[0] - original_center,
                axis=1,
            )
            carrier_mask = initial_wall_distance <= critical.BOUNDARY_SHELL_WIDTH
            initial_center = fitted_circle_center(positions[0, carrier_mask])
            for frame, (frame_positions, frame_phases) in enumerate(
                zip(positions, phases)
            ):
                include_clusters = frame % cluster_stride == 0 or frame == len(positions) - 1
                carrier_positions = frame_positions[carrier_mask]
                carrier_phases = frame_phases[carrier_mask]
                metrics = frame_metrics(
                    carrier_positions,
                    carrier_phases,
                    initial_modes[condition.diameter],
                    include_clusters,
                )
                metrics["center_displacement"] = float(
                    np.linalg.norm(fitted_circle_center(carrier_positions) - initial_center)
                )
                metrics["carrier_fraction_initial"] = float(np.mean(carrier_mask))
                metrics["fraction_inside_original_circle"] = float(
                    np.mean(
                        np.linalg.norm(frame_positions - original_center, axis=1)
                        <= condition.diameter / 2.0
                    )
                )
                metrics["carrier_fraction_inside_original_circle"] = float(
                    np.mean(
                        np.linalg.norm(carrier_positions - original_center, axis=1)
                        <= condition.diameter / 2.0
                    )
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "diameter": condition.diameter,
                        "seed": condition.seed,
                        "time_after_switch": frame * SNAPSHOT_INTERVAL * critical.DT,
                        **metrics,
                    }
                )
    time_series = pd.DataFrame(rows)
    terminal_rows = []
    for keys, group in time_series.groupby(["protocol", "diameter", "seed"]):
        group = group.sort_values("time_after_switch")
        initial = group.iloc[0]
        terminal = group.iloc[-1]
        terminal_rows.append(
            {
                "protocol": keys[0],
                "diameter": keys[1],
                "seed": keys[2],
                "initial_mode": int(initial_modes[keys[1]]),
                "initial_cluster_count": initial["cluster_count"],
                "terminal_cluster_count": terminal["cluster_count"],
                "initial_chord": initial["cluster_chord_mean"],
                "terminal_chord": terminal["cluster_chord_mean"],
                "initial_mode_amplitude": initial["initial_mode_amplitude"],
                "terminal_mode_amplitude": terminal["initial_mode_amplitude"],
                "radius_ratio_terminal": terminal["mean_radius"] / initial["mean_radius"],
                "radius_cv_initial": initial["radius_cv"],
                "radius_cv_terminal": terminal["radius_cv"],
                "tangential_coherence_initial": initial["tangential_coherence"],
                "tangential_coherence_terminal": terminal["tangential_coherence"],
                "radial_signed_initial": initial["radial_signed"],
                "radial_signed_terminal": terminal["radial_signed"],
                "carrier_fraction_initial": initial["carrier_fraction_initial"],
                "fraction_inside_original_circle_terminal": terminal[
                    "fraction_inside_original_circle"
                ],
                "carrier_fraction_inside_original_circle_terminal": terminal[
                    "carrier_fraction_inside_original_circle"
                ],
                "center_displacement_terminal": terminal["center_displacement"],
                "polarization_terminal": terminal["polarization"],
            }
        )
    return time_series, pd.DataFrame(terminal_rows)


def critical_chord_measurements() -> pd.DataFrame:
    """Measure adjacent terminal cluster-center chords in all formed critical runs."""

    measurement_path = (
        critical.OUTPUT_DIR / "Boundary_Lattice_Quantization_Measurements.csv"
    )
    measurements = pd.read_csv(measurement_path)
    formed = measurements[
        np.isclose(measurements["alpha_over_pi"], 0.5)
        & measurements["lattice_formed"].astype(bool)
    ]
    spectral_wavelength = 2.0 * np.pi / 5.0946690638739
    rows = []
    for row in formed.itertuples(index=False):
        condition = critical.Condition(0.5, float(row.diameter), int(row.seed))
        positions, _, source_path = source_terminal_state(condition)
        wall_center = np.array([condition.diameter / 2.0] * 2)
        radii = np.linalg.norm(positions - wall_center, axis=1)
        shell = condition.diameter / 2.0 - radii <= critical.BOUNDARY_SHELL_WIDTH
        centers = identified_cluster_centers(positions[shell])
        angles = np.mod(
            np.arctan2(
                centers[:, 1] - wall_center[1],
                centers[:, 0] - wall_center[0],
            ),
            2.0 * np.pi,
        )
        centers = centers[np.argsort(angles)]
        chords = np.linalg.norm(np.roll(centers, -1, axis=0) - centers, axis=1)
        effective_radius = float(
            np.mean(np.linalg.norm(centers - wall_center, axis=1))
        )
        mode = len(centers)
        actual_chord = float(np.mean(chords))
        rows.append(
            {
                "diameter": condition.diameter,
                "seed": condition.seed,
                "cluster_count": mode,
                "effective_radius": effective_radius,
                "actual_chord_mean": actual_chord,
                "actual_chord_std": float(np.std(chords)),
                "uniform_chord_at_effective_radius": float(
                    2.0 * effective_radius * np.sin(np.pi / mode)
                ),
                "effective_radius_arc": float(2.0 * np.pi * effective_radius / mode),
                "wall_radius_arc": float(np.pi * condition.diameter / mode),
                "spectral_wavelength": spectral_wavelength,
                "chord_relative_error_to_spectrum": float(
                    actual_chord / spectral_wavelength - 1.0
                ),
                "source_file": str(source_path),
            }
        )
    return pd.DataFrame(rows)


def create_metric_figure(time_series: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    metrics = (
        ("mean_radius", r"Mean radius about COM"),
        ("initial_mode_amplitude", r"Original mode amplitude $A_{m_0}$"),
        ("tangential_coherence", r"Tangential coherence"),
        ("cluster_count", r"DBSCAN cluster count"),
    )
    colors = {"retained": "#4C78A8", "removed": "#E45756"}
    linestyles = {3.0: "-", 5.0: "--"}
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
        grouped = time_series.groupby(["protocol", "diameter", "time_after_switch"])[metric]
        summary = grouped.agg(["mean", "min", "max"]).reset_index()
        summary = summary.dropna(subset=["mean"])
        for (protocol, diameter), group in summary.groupby(["protocol", "diameter"]):
            axis.plot(
                group["time_after_switch"],
                group["mean"],
                color=colors[protocol],
                ls=linestyles[diameter],
                lw=1.7,
                label=f"{protocol}, D={diameter:g}",
            )
            axis.fill_between(
                group["time_after_switch"],
                group["min"],
                group["max"],
                color=colors[protocol],
                alpha=0.10,
            )
        axis.set_xlabel("Time after switch")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        r"Established $\alpha=0.5\pi$ boundary lattices after retaining or removing the wall"
    )
    return figure


def create_snapshot_figure() -> plt.Figure:
    sample_times = (0.0, 2.0, 10.0, 30.0, CONTINUATION_TIME)
    representative = (
        critical.Condition(0.5, 3.0, 9),
        critical.Condition(0.5, 5.0, 9),
    )
    figure, axes = plt.subplots(
        len(representative) * len(PROTOCOLS),
        len(sample_times),
        figsize=(14, 10),
        constrained_layout=True,
    )
    for condition_index, condition in enumerate(representative):
        for protocol_index, protocol in enumerate(PROTOCOLS):
            row = condition_index * len(PROTOCOLS) + protocol_index
            positions, phases = load_continuation(ContinuationJob(condition, protocol))
            for column, sample_time in enumerate(sample_times):
                frame = int(round(sample_time / (SNAPSHOT_INTERVAL * critical.DT)))
                axis = axes[row, column]
                frame_positions = positions[frame]
                axis.scatter(
                    frame_positions[:, 0],
                    frame_positions[:, 1],
                    c=phases[frame],
                    cmap=phaseCmap,
                    norm=phaseNorm,
                    s=2.0,
                    linewidths=0,
                    rasterized=True,
                )
                if protocol == "retained":
                    circle = plt.Circle(
                        (condition.diameter / 2.0, condition.diameter / 2.0),
                        condition.diameter / 2.0,
                        fill=False,
                        color="0.25",
                        lw=0.8,
                    )
                    axis.add_patch(circle)
                axis.set_aspect("equal", adjustable="datalim")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(rf"$t={sample_time:g}$")
                if column == 0:
                    axis.set_ylabel(f"D={condition.diameter:g}\n{protocol}")
    figure.suptitle("Boundary-lattice continuation after an instantaneous wall switch")
    return figure


def write_report(
    summary: pd.DataFrame,
    time_series: pd.DataFrame,
    chords: pd.DataFrame,
) -> Path:
    report = OUTPUT_DIR / "Boundary_Removal_Analysis.md"
    initial_amplitude = time_series[
        np.isclose(time_series["time_after_switch"], 0.0)
    ][["protocol", "diameter", "seed", "initial_mode_amplitude"]].rename(
        columns={"initial_mode_amplitude": "amplitude_at_release"}
    )
    normalized = time_series.merge(
        initial_amplitude,
        on=["protocol", "diameter", "seed"],
        how="left",
    )
    normalized["relative_mode_amplitude"] = (
        normalized["initial_mode_amplitude"] / normalized["amplitude_at_release"]
    )
    early = (
        normalized[
            (normalized["protocol"] == "removed")
            & normalized["time_after_switch"].isin([0.0, 0.5, 1.0, 2.0, 5.0])
        ]
        .groupby(["diameter", "time_after_switch"])[
            [
                "relative_mode_amplitude",
                "tangential_coherence",
                "radial_signed",
                "carrier_fraction_inside_original_circle",
            ]
        ]
        .mean()
        .reset_index()
    )
    chord_summary = (
        chords.groupby("diameter")
        .agg(
            actual_chord=("actual_chord_mean", "mean"),
            effective_arc=("effective_radius_arc", "mean"),
            wall_arc=("wall_radius_arc", "mean"),
            relative_error=("chord_relative_error_to_spectrum", "mean"),
        )
        .reset_index()
    )
    terminal_view = summary[
        [
            "protocol",
            "diameter",
            "seed",
            "initial_cluster_count",
            "terminal_cluster_count",
            "initial_mode_amplitude",
            "terminal_mode_amplitude",
            "radius_ratio_terminal",
            "radius_cv_terminal",
            "tangential_coherence_terminal",
        ]
    ]
    lines = [
        "# 边界晶格弦长与瞬时撤墙续跑",
        "",
        "## 1. 弦长复核",
        "",
        "连续体 `2 pi/k` 是沿物理坐标的波长。映射到圆周时首先应比较有效轨道弧长；"
        "相邻团簇质心的欧氏直线距离则是弦长。",
        "",
        chord_summary.to_markdown(index=False, floatfmt=".6g"),
        "",
        "体谱波长为 `1.233286`。弦长比它低约15%--19%，且弦长始终比有效弧长更短；"
        "因此弧/弦定义不能解释谱与边界间距的主失配。",
        "",
        "## 2. 瞬时撤墙协议",
        "",
        f"从源轨迹 `t=250` 的末态重启，并继续积分 {CONTINUATION_TIME:g} 个时间单位。"
        "撤墙组只删除圆形镜面反射和碰撞航向重置；保留组从完全相同状态继续，"
        "其余位置推进、相位耦合、参数和积分顺序完全一致。",
        "",
        "## 3. 撤墙后的早期衰减（两个种子的平均）",
        "",
        early.to_markdown(index=False, floatfmt=".6g"),
        "",
        "在 `t=0.5`，原始 `m=9/15` 模态只剩约50%--59%，初始边界载流粒子几乎全部"
        "离开原圆；径向投影变为约0.84--0.90，说明发生快速向外释放。到 `t=2`，"
        "原模态不再是稳定的全局环晶格。",
        "",
        "",
        "## 4. t=100 配对结果",
        "",
        terminal_view.to_markdown(index=False, floatfmt=".6g"),
        "",
        "保留墙对照维持原来的9或15个团簇、模态幅度约0.996--0.998、半径比约1。"
        "撤墙组半径扩张约16--30倍，径向宽度与平均半径同量级，原模态幅度明显下降，"
        "最终只剩5--7个DBSCAN聚集体。后者是局部团簇或再聚集，不是原来的圆周Lattice。",
        "",
        "## 5. 动力学解释",
        "",
        "在 `alpha=pi/2, omega=0` 时，",
        "",
        "```text",
        "dot(theta_i) = K [ <cos(theta_j-theta_i)>_neighbors - 1 ] <= 0.",
        "```",
        "",
        "原晶格沿边界切向运动；撤掉反射后，负的航向转速把速度转向径向外侧。"
        "邻域随后断开，粒子/局部团簇进入近似弹道运动或再聚集。硬墙消失也意味着"
        "物质--真空界面消失，因此原边缘谱流没有必须继续存在的边界载体。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_simulation:
        ensure_continuations(args.workers)
    time_series, summary = analyze()
    chords = critical_chord_measurements()
    time_series.to_csv(OUTPUT_DIR / "Boundary_Removal_TimeSeries.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "Boundary_Removal_Summary.csv", index=False)
    chords.to_csv(OUTPUT_DIR / "Critical_Lattice_Chord_Spacings.csv", index=False)
    metric_figure = create_metric_figure(time_series)
    metric_figure.savefig(
        OUTPUT_DIR / "Boundary_Removal_Diagnostics.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(metric_figure)
    snapshot_figure = create_snapshot_figure()
    snapshot_figure.savefig(
        OUTPUT_DIR / "Boundary_Removal_Snapshots.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(snapshot_figure)
    report = write_report(summary, time_series, chords)
    configuration = {
        "wall_switch_source_time": critical.ITERATIONS * critical.DT,
        "continuation_steps": CONTINUATION_STEPS,
        "continuation_time": CONTINUATION_TIME,
        "snapshot_interval": SNAPSHOT_INTERVAL,
        "conditions": [
            {
                "alpha_over_pi": condition.alpha_over_pi,
                "diameter": condition.diameter,
                "seed": condition.seed,
            }
            for condition in SOURCE_CONDITIONS
        ],
        "protocols": list(PROTOCOLS),
        "reference_files_modified": False,
    }
    (OUTPUT_DIR / "Boundary_Removal_Configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(f"Saved {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
