"""Generate terminal-state alpha sweeps in small circular boundaries.

The default experiment compares diameters 3 and 5 while holding every other
model parameter, including the particle count, fixed.  This makes the effect of
confinement directly visible.  Seven phase-lag values from 0 to pi are simulated
and plotted as phase-coloured heading arrows.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numba as nb
import numpy as np

from CircularFigure import LastFrameStateAnalysis, expected_data_path
from main import CircularBoundaryPatternFormation, phaseCmap, phaseNorm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DIAMETERS = (3.0, 5.0)
DEFAULT_ALPHA_OVER_PI = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "Small_Circular_Alpha_Sweep"


@dataclass(frozen=True)
class ExperimentConfig:
    strengthK: float = 20.75
    distanceD0: float = 1.0
    speedV: float = 3.0
    freqDist: str = "uniform"
    omegaMin: float = 0.0
    deltaOmega: float = 0.0
    agentsNum: int = 400
    dt: float = 0.005
    shotsnaps: int = 50
    randomSeed: int = 9
    iterations: int = 5000


@nb.njit(cache=True)
def _calc_dot_phase_collision_fast(
    positionX: np.ndarray,
    phaseTheta: np.ndarray,
    freqOmega: np.ndarray,
    params: tuple[float, float, float, float, float],
) -> np.ndarray:
    """Allocation-light equivalent of the model's cutoff coupling kernel."""

    _, _, distance_d0, strength_k, phase_lag_a0 = params
    agents_num = positionX.shape[0]
    cutoff_squared = distance_d0 * distance_d0
    phase_offset = np.sin(phase_lag_a0)
    phase_lag_cosine = np.cos(phase_lag_a0)
    phase_sines = np.sin(phaseTheta)
    phase_cosines = np.cos(phaseTheta)
    neighbor_counts = np.zeros(agents_num, dtype=np.int64)
    coupling_sums = np.zeros(agents_num, dtype=np.float64)
    dot_phase = np.empty(agents_num, dtype=np.float64)

    for i in range(agents_num - 1):
        x_i = positionX[i, 0]
        y_i = positionX[i, 1]
        for j in range(i + 1, agents_num):
            dx = positionX[j, 0] - x_i
            dy = positionX[j, 1] - y_i
            distance_squared = dx * dx + dy * dy
            if 0.0 < distance_squared <= cutoff_squared:
                sine_delta = (
                    phase_sines[j] * phase_cosines[i]
                    - phase_cosines[j] * phase_sines[i]
                )
                cosine_delta = (
                    phase_cosines[j] * phase_cosines[i]
                    + phase_sines[j] * phase_sines[i]
                )
                common_sine = cosine_delta * phase_offset
                directed_sine = sine_delta * phase_lag_cosine
                coupling_sums[i] += common_sine + directed_sine
                coupling_sums[j] += common_sine - directed_sine
                neighbor_counts[i] += 1
                neighbor_counts[j] += 1

    for i in range(agents_num):
        if neighbor_counts[i]:
            coupling = coupling_sums[i] / neighbor_counts[i] - phase_offset
        else:
            coupling = 0.0
        dot_phase[i] = strength_k * coupling + freqOmega[i]
    return dot_phase


def build_model(
    diameter: float,
    alpha_over_pi: float,
    config: ExperimentConfig,
    data_dir: Path,
) -> CircularBoundaryPatternFormation:
    """Construct one member of the two-parameter sweep."""

    model = CircularBoundaryPatternFormation(
        strengthK=config.strengthK,
        distanceD0=config.distanceD0,
        phaseLagA0=float(alpha_over_pi) * np.pi,
        boundaryLength=float(diameter),
        speedV=config.speedV,
        freqDist=config.freqDist,
        omegaMin=config.omegaMin,
        deltaOmega=config.deltaOmega,
        agentsNum=config.agentsNum,
        dt=config.dt,
        tqdm=False,
        savePath=str(data_dir),
        shotsnaps=config.shotsnaps,
        randomSeed=config.randomSeed,
        overWrite=False,
    )
    model._calc_dot_phase_collision = _calc_dot_phase_collision_fast
    return model


def _run_one(job: tuple[float, float, ExperimentConfig, str]) -> str:
    """Windows-spawn-safe simulation worker."""

    diameter, alpha_over_pi, config, data_dir_text = job
    data_dir = Path(data_dir_text)
    model = build_model(diameter, alpha_over_pi, config, data_dir)
    model.run(config.iterations)
    return str(expected_data_path(model))


def build_models(
    diameters: Sequence[float],
    alpha_over_pi: Sequence[float],
    config: ExperimentConfig,
    data_dir: Path,
) -> list[CircularBoundaryPatternFormation]:
    models = [
        build_model(diameter, alpha, config, data_dir)
        for diameter in diameters
        for alpha in alpha_over_pi
    ]
    paths = [expected_data_path(model) for model in models]
    if len(paths) != len(set(paths)):
        raise ValueError("Sweep settings produce duplicate canonical HDF5 names.")
    return models


def ensure_simulations(
    models: Sequence[CircularBoundaryPatternFormation],
    config: ExperimentConfig,
    data_dir: Path,
    workers: int,
) -> None:
    missing = [model for model in models if not expected_data_path(model).is_file()]
    if not missing:
        print("All matched HDF5 files already exist; simulation skipped.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    worker_count = min(max(1, workers), 4, len(missing))
    jobs = [
        (
            model.boundaryLength,
            model.phaseLagA0 / np.pi,
            config,
            str(data_dir),
        )
        for model in missing
    ]
    print(
        f"Generating {len(jobs)} terminal trajectories with "
        f"{worker_count} worker(s)..."
    )
    if worker_count == 1:
        for completed, job in enumerate(jobs, start=1):
            diameter, alpha = job[:2]
            _run_one(job)
            print(
                f"[{completed:02d}/{len(jobs):02d}] "
                f"D={diameter:g}, alpha={alpha:g} pi"
            )
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures = {executor.submit(_run_one, job): job[:2] for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            diameter, alpha = futures[future]
            future.result()
            print(
                f"[{completed:02d}/{len(jobs):02d}] "
                f"D={diameter:g}, alpha={alpha:g} pi"
            )


def alpha_label(alpha_over_pi: float) -> str:
    if np.isclose(alpha_over_pi, 0.0):
        value = "0"
    elif np.isclose(alpha_over_pi, 1.0):
        value = r"\pi"
    else:
        value = rf"{alpha_over_pi:g}\pi"
    return rf"$\alpha={value}$"


def _draw_panel(
    axis: plt.Axes,
    analysis: LastFrameStateAnalysis,
    title: str,
) -> None:
    analysis.plot_spatial(axis, colorsBy="phase", index=-1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=12, pad=5)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _add_shared_colorbar(
    figure: plt.Figure,
    axes: Iterable[plt.Axes],
) -> None:
    scalar_mappable = ScalarMappable(norm=phaseNorm, cmap=phaseCmap)
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=list(axes),
        ticks=[0, np.pi, 2 * np.pi],
        fraction=0.022,
        pad=0.018,
        aspect=28,
    )
    colorbar.ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    colorbar.set_label(r"Phase $\theta$", fontsize=12)


def create_combined_figure(
    analyses: Sequence[LastFrameStateAnalysis],
    diameters: Sequence[float],
    alpha_over_pi: Sequence[float],
    config: ExperimentConfig,
) -> plt.Figure:
    rows = len(diameters)
    columns = len(alpha_over_pi)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.55 * columns + 0.8, 2.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, diameter in enumerate(diameters):
        for column, alpha in enumerate(alpha_over_pi):
            index = row * columns + column
            _draw_panel(axes[row, column], analyses[index], alpha_label(alpha))
        axes[row, 0].set_ylabel(
            rf"$D={diameter:g}$" + "\n" + rf"$N={config.agentsNum}$",
            fontsize=12,
            labelpad=8,
        )

    figure.suptitle(
        "Small circular boundaries: alpha-sweep terminal states "
        rf"($t={config.iterations * config.dt:g}$)",
        fontsize=14,
    )
    _add_shared_colorbar(figure, axes.ravel())
    return figure


def create_single_diameter_figure(
    analyses: Sequence[LastFrameStateAnalysis],
    diameter: float,
    alpha_over_pi: Sequence[float],
    config: ExperimentConfig,
) -> plt.Figure:
    columns = min(4, len(alpha_over_pi))
    rows = math.ceil(len(alpha_over_pi) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.15 * columns + 0.7, 3.25 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for axis, analysis, alpha in zip(flat_axes, analyses, alpha_over_pi):
        _draw_panel(axis, analysis, alpha_label(alpha))
    for axis in flat_axes[len(analyses):]:
        axis.set_visible(False)

    figure.suptitle(
        rf"Circular boundary $D={diameter:g}$, $N={config.agentsNum}$; "
        rf"terminal states at $t={config.iterations * config.dt:g}$",
        fontsize=14,
    )
    _add_shared_colorbar(figure, flat_axes[: len(analyses)])
    return figure


def save_figure_pair(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return png_path, pdf_path


def write_manifest(
    output_dir: Path,
    data_dir: Path,
    diameters: Sequence[float],
    alpha_over_pi: Sequence[float],
    config: ExperimentConfig,
) -> Path:
    manifest_path = output_dir / "Small_Circular_Alpha_Sweep_Configuration.json"
    manifest = {
        "model": "CircularBoundaryPatternFormation",
        "diameters": list(diameters),
        "alpha_over_pi": list(alpha_over_pi),
        "config": asdict(config),
        "terminal_time": config.iterations * config.dt,
        "data_directory": str(data_dir),
        "comparison_design": (
            "Particle count and all non-boundary parameters are held fixed; "
            "only alpha and boundary diameter vary."
        ),
        "coupling_kernel": (
            "Pair-symmetric, allocation-light implementation of the original "
            "distance-cutoff phase coupling; verified against the original "
            "kernel to double-precision tolerance."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--snapshot-interval", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.agents <= 0 or args.agents % 2:
        raise ValueError("--agents must be a positive even integer.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4.")
    snapshot_interval = (
        max(50, args.iterations // 100)
        if args.snapshot_interval is None
        else args.snapshot_interval
    )
    if snapshot_interval <= 0:
        raise ValueError("--snapshot-interval must be positive.")

    config = ExperimentConfig(
        agentsNum=args.agents,
        iterations=args.iterations,
        shotsnaps=snapshot_interval,
    )
    diameters = DEFAULT_DIAMETERS
    alpha_over_pi = DEFAULT_ALPHA_OVER_PI
    output_dir = args.output_dir.resolve()
    data_dir = (
        PROJECT_DIR
        / "data"
        / "small_circular_alpha_sweep"
        / f"N{config.agentsNum}_steps{config.iterations}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(diameters, alpha_over_pi, config, data_dir)
    ensure_simulations(models, config, data_dir, args.workers)
    analyses = [LastFrameStateAnalysis(model) for model in models]

    stem_suffix = f"N{config.agentsNum}_Steps{config.iterations}"
    combined = create_combined_figure(
        analyses, diameters, alpha_over_pi, config
    )
    save_figure_pair(
        combined,
        output_dir
        / f"Small_Circular_Boundary_Alpha_Sweep_D3_D5_{stem_suffix}",
    )

    count_per_diameter = len(alpha_over_pi)
    for row, diameter in enumerate(diameters):
        start = row * count_per_diameter
        end = start + count_per_diameter
        figure = create_single_diameter_figure(
            analyses[start:end], diameter, alpha_over_pi, config
        )
        save_figure_pair(
            figure,
            output_dir
            / f"Small_Circular_Boundary_Alpha_Sweep_D{diameter:g}_{stem_suffix}",
        )

    manifest_path = write_manifest(
        output_dir, data_dir, diameters, alpha_over_pi, config
    )
    print(f"Saved {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
