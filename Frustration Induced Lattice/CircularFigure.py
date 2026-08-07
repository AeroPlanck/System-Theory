"""Plot terminal states of a circular-boundary phase-lag sweep.

Workflow
--------
1. Keep every model parameter fixed except ``phaseLagA0``.
2. Resolve each HDF5 path with the model's canonical ``str(model)`` name.
3. Optionally generate missing data with a deliberately small process pool.
4. Abort before plotting if any requested file is still missing or invalid.
5. Read only the last complete frame and reuse ``StateAnalysis.plot_spatial``.
6. Save one labelled multi-panel figure with one shared phase colorbar.

The current HDF5 schema does not store iteration numbers.  Consequently,
``SIMULATION_ITERATIONS`` controls newly generated files, while existing files
are plotted at their unambiguous last saved frame.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from main import (
    CircularBoundaryPatternFormation,
    StateAnalysis,
    phaseCmap,
    phaseNorm,
)


PROJECT_DIR = Path(__file__).resolve().parent


# =============================================================================
# USER CONFIGURATION: edit this block; the workflow below normally needs no edit
# =============================================================================

# Fixed model parameters. phaseLagA0 is intentionally absent because it is the
# only traversed model parameter.
MODEL_PARAMETERS = {
    "strengthK": 20.75,
    "distanceD0": 1.0,
    "boundaryLength": 7.0,
    "speedV": 3.0,
    "freqDist": "uniform",   # "uniform", "cauchy", or "identical"
    "initPhaseTheta": None,
    "omegaMin": 0.0,
    "deltaOmega": 0.0,
    "agentsNum": 2000,
    "dt": 0.005,
    "shotsnaps": 10,
    "randomSeed": 9,
}

# Sweep values are written in units of pi for convenient editing.
# Irregular/custom sweep:
PHASE_LAG_A0_OVER_PI = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
# Regular-range example (replace the line above when needed):
# PHASE_LAG_A0_OVER_PI = np.linspace(0.0, 1.0, 11)
# Fixed-step example:
# PHASE_LAG_A0_OVER_PI = np.arange(0.0, 1.01, 0.1)
PHASE_LAG_A0_VALUES = PHASE_LAG_A0_OVER_PI * np.pi

# Data generation and output.
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = PROJECT_DIR / "figs" / "circular_boundary_phase_lag_terminal_states.pdf"
SIMULATION_ITERATIONS = 10000
GENERATE_MISSING = True
MAX_WORKERS = 2
HARD_WORKER_LIMIT = 4

# Integrated-figure layout. Particle arrows themselves remain controlled by
# StateAnalysis in main.py.
PLOT_COLUMNS = 3
PANEL_WIDTH = 3.25
PANEL_HEIGHT = 3.45
LABEL_FONTSIZE = 15.0
TITLE_FONTSIZE = 15.0
COLORBAR_LABEL_FONTSIZE = 14.0
FIGURE_DPI = 300

# Monochrome boundary-rotation annotation. The percentages are calculated from
# terminal-frame particles in the outer shell and normalized over particles
# with a sufficiently strong tangential heading.
BOUNDARY_SHELL_FRACTION = 0.01
TANGENTIAL_THRESHOLD = 0.20
ROTATION_LABEL_FONTSIZE = 9.0
ROTATION_LABEL_POSITION = (0.975, 0.025)
ROTATION_LABEL_BACKGROUND_ALPHA = 0.2

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


class DataPreflightError(RuntimeError):
    """Raised when the all-or-nothing data preflight cannot be satisfied."""


def build_model(
    phase_lag_a0: float,
    data_dir: Path = DATA_DIR,
    show_iteration_progress: bool = False,
) -> CircularBoundaryPatternFormation:
    """Construct one model; only ``phaseLagA0`` varies between calls."""

    return CircularBoundaryPatternFormation(
        phaseLagA0=phase_lag_a0,
        tqdm=show_iteration_progress,
        savePath=str(data_dir),
        overWrite=False,
        **MODEL_PARAMETERS,
    )


def expected_data_path(model: CircularBoundaryPatternFormation) -> Path:
    """Use the exact naming contract implemented by ``main.py``."""

    return Path(model.savePath) / f"{model}.h5"


def build_sweep_models(
    phase_lag_values: Sequence[float] = PHASE_LAG_A0_VALUES,
    data_dir: Path = DATA_DIR,
) -> list[CircularBoundaryPatternFormation]:
    phase_lag_values = tuple(float(value) for value in phase_lag_values)
    if not phase_lag_values:
        raise ValueError("PHASE_LAG_A0_VALUES must contain at least one value.")
    if not all(np.isfinite(value) for value in phase_lag_values):
        raise ValueError("Every phaseLagA0 value must be finite.")

    models = [
        build_model(phase_lag, data_dir)
        for phase_lag in phase_lag_values
    ]
    paths = [expected_data_path(model) for model in models]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "The phaseLagA0 sweep produces duplicate canonical filenames. "
            "Increase the spacing between values; main.py formats A0 to 0.001."
        )
    return models


def _run_generation_job(job: tuple[float, Path, int, bool]) -> tuple[float, str]:
    """Top-level worker function so Windows ``spawn`` can pickle it."""

    phase_lag_a0, data_dir, simulation_iterations, show_iteration_progress = job
    model = build_model(
        phase_lag_a0,
        data_dir,
        show_iteration_progress=show_iteration_progress,
    )
    model.run(simulation_iterations)
    return phase_lag_a0, str(expected_data_path(model))


def _missing_models(
    models: Sequence[CircularBoundaryPatternFormation],
) -> list[CircularBoundaryPatternFormation]:
    return [model for model in models if not expected_data_path(model).is_file()]


def ensure_data_files(
    models: Sequence[CircularBoundaryPatternFormation],
    *,
    data_dir: Path = DATA_DIR,
    generate_missing: bool = GENERATE_MISSING,
    simulation_iterations: int = SIMULATION_ITERATIONS,
    max_workers: int = MAX_WORKERS,
) -> None:
    """Generate only missing files, then enforce an all-or-nothing recheck."""

    missing = _missing_models(models)
    if not missing:
        return

    missing_names = "\n".join(f"  - {expected_data_path(model)}" for model in missing)
    if not generate_missing:
        raise DataPreflightError(
            "Missing exact parameter-matched HDF5 file(s); plotting was stopped:\n"
            f"{missing_names}"
        )

    if simulation_iterations <= 0:
        raise ValueError("SIMULATION_ITERATIONS must be positive.")
    if not 1 <= max_workers <= HARD_WORKER_LIMIT:
        raise ValueError(
            f"MAX_WORKERS must be between 1 and {HARD_WORKER_LIMIT}; "
            "the hard limit protects this device from oversized pools."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            model.phaseLagA0,
            data_dir,
            simulation_iterations,
            len(missing) == 1,
        )
        for model in missing
    ]
    worker_count = min(max_workers, len(jobs))
    print(
        f"Generating {len(jobs)} missing HDF5 file(s) with "
        f"{worker_count} worker process(es)."
    )

    spawn_context = mp.get_context("spawn")
    try:
        with spawn_context.Pool(processes=worker_count) as pool:
            if len(jobs) == 1:
                phase_lag_a0, _ = pool.apply(_run_generation_job, (jobs[0],))
                print(f"Generated HDF5 for alpha={phase_lag_a0 / np.pi:.3g} pi")
            else:
                completed_jobs = pool.imap_unordered(_run_generation_job, jobs)
                with tqdm(
                    completed_jobs,
                    total=len(jobs),
                    desc="Generating HDF5",
                    unit="file",
                    dynamic_ncols=True,
                ) as progress:
                    for phase_lag_a0, _ in progress:
                        progress.set_postfix_str(
                            f"alpha={phase_lag_a0 / np.pi:.3g} pi",
                            refresh=False,
                        )
    except Exception as exc:
        raise DataPreflightError(
            "Missing-data generation failed; all subsequent work was stopped: "
            f"{exc}"
        ) from exc

    still_missing = _missing_models(models)
    if still_missing:
        failed_names = "\n".join(
            f"  - {expected_data_path(model)}" for model in still_missing
        )
        raise DataPreflightError(
            "Generation finished but exact matched file(s) are still missing; "
            "all subsequent work was stopped:\n"
            f"{failed_names}"
        )


class LastFrameStateAnalysis(StateAnalysis):
    """Memory-efficient StateAnalysis that contains one complete terminal frame."""

    def __init__(self, model: CircularBoundaryPatternFormation):
        self.model = model
        target_path = expected_data_path(model)

        try:
            with pd.HDFStore(target_path, mode="r") as store:
                required_keys = {"/positionX", "/phaseTheta"}
                missing_keys = required_keys.difference(store.keys())
                if missing_keys:
                    names = ", ".join(sorted(missing_keys))
                    raise DataPreflightError(
                        f"{target_path} is missing required HDF5 key(s): {names}"
                    )

                position_storer = store.get_storer("positionX")
                phase_storer = store.get_storer("phaseTheta")
                position_rows = position_storer.nrows
                phase_rows = phase_storer.nrows
                agents_num = model.agentsNum

                if position_storer.ncols != 2 or phase_storer.ncols != 1:
                    raise DataPreflightError(
                        f"{target_path} has incompatible column counts: "
                        f"positionX={position_storer.ncols}, "
                        f"phaseTheta={phase_storer.ncols}."
                    )
                if position_rows < agents_num or phase_rows < agents_num:
                    raise DataPreflightError(
                        f"{target_path} does not contain one complete N={agents_num} frame."
                    )
                if position_rows % agents_num or phase_rows % agents_num:
                    raise DataPreflightError(
                        f"{target_path} contains partial frames; row counts must be "
                        f"multiples of N={agents_num}."
                    )
                if position_rows != phase_rows:
                    raise DataPreflightError(
                        f"{target_path} has unaligned histories: "
                        f"positionX={position_rows} rows, phaseTheta={phase_rows} rows."
                    )

                position_x = store.select(
                    "positionX", start=position_rows - agents_num
                ).to_numpy()
                phase_theta = store.select(
                    "phaseTheta", start=phase_rows - agents_num
                ).to_numpy()
        except DataPreflightError:
            raise
        except Exception as exc:
            raise DataPreflightError(f"Cannot read {target_path}: {exc}") from exc

        if position_x.shape != (model.agentsNum, 2):
            raise DataPreflightError(
                f"{target_path} terminal positionX shape is {position_x.shape}, "
                f"expected {(model.agentsNum, 2)}."
            )
        if phase_theta.shape != (model.agentsNum, 1):
            raise DataPreflightError(
                f"{target_path} terminal phaseTheta shape is {phase_theta.shape}, "
                f"expected {(model.agentsNum, 1)}."
            )
        if not np.isfinite(position_x).all() or not np.isfinite(phase_theta).all():
            raise DataPreflightError(
                f"{target_path} terminal frame contains NaN or infinite values."
            )

        self.source_snapshot_count = phase_rows // model.agentsNum
        self.source_path = target_path
        self.TNum = 1
        self.totalPositionX = position_x.reshape(1, model.agentsNum, 2)
        self.totalPhaseTheta = phase_theta.reshape(1, model.agentsNum)


def subplot_label(index: int) -> str:
    """Return a, b, ..., z, aa, ab, ... for any number of panels."""

    if index < 0:
        raise ValueError("Subplot index cannot be negative.")
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("a") + remainder) + label
        if value == 0:
            return label
        value -= 1


def alpha_math_label(phase_lag_a0: float) -> str:
    """Format ``phaseLagA0`` as a compact mathematical multiple of pi."""

    multiple = phase_lag_a0 / np.pi
    if np.isclose(multiple, 0.0, atol=1e-12):
        value = "0"
    elif np.isclose(multiple, 1.0, atol=1e-12):
        value = r"\pi"
    elif np.isclose(multiple, -1.0, atol=1e-12):
        value = r"-\pi"
    else:
        value = rf"{multiple:.3g}\pi"
    return rf"$\alpha={value}$"


def boundary_rotation_statistics(
    analysis: LastFrameStateAnalysis,
) -> tuple[float, float, int, int]:
    """Return terminal CW/CCW percentages for tangential boundary particles.

    Positive tangential projection is counterclockwise and negative projection
    is clockwise. Percentages are normalized over the classified tangential
    particles, so they sum to 100 when at least one particle is classified.
    """

    if not 0 < BOUNDARY_SHELL_FRACTION <= 1:
        raise ValueError("BOUNDARY_SHELL_FRACTION must be in (0, 1].")
    if not 0 <= TANGENTIAL_THRESHOLD < 1:
        raise ValueError("TANGENTIAL_THRESHOLD must be in [0, 1).")

    position_x, phase_theta = analysis.get_state(-1)
    relative_position = position_x - analysis.model.circleCenter
    radial_distance = np.linalg.norm(relative_position, axis=1)
    shell_width = analysis.model.circleRadius * BOUNDARY_SHELL_FRACTION
    boundary_mask = radial_distance >= analysis.model.circleRadius - shell_width

    polar_angle = np.arctan2(relative_position[:, 1], relative_position[:, 0])
    # dot((cos(theta), sin(theta)), (-sin(phi), cos(phi)))
    tangential_projection = np.sin(phase_theta - polar_angle)
    tangential_mask = boundary_mask & (
        np.abs(tangential_projection) >= TANGENTIAL_THRESHOLD
    )

    boundary_count = int(np.count_nonzero(boundary_mask))
    tangential_count = int(np.count_nonzero(tangential_mask))
    if tangential_count == 0:
        return np.nan, np.nan, boundary_count, tangential_count

    classified = tangential_projection[tangential_mask]
    clockwise_percent = float(np.mean(classified < 0) * 100)
    counterclockwise_percent = float(np.mean(classified > 0) * 100)
    return (
        clockwise_percent,
        counterclockwise_percent,
        boundary_count,
        tangential_count,
    )


def add_rotation_annotation(
    axis: plt.Axes,
    analysis: LastFrameStateAnalysis,
) -> None:
    """Add a neutral corner label without competing with the phase colormap."""

    clockwise, counterclockwise, _, tangential_count = (
        boundary_rotation_statistics(analysis)
    )
    if tangential_count == 0:
        text = "Boundary rotation\nCW   --\nCCW  --"
    else:
        text = (
            f"Boundary rotation  n={tangential_count}\n"
            f"CW    {clockwise:5.1f}%\n"
            f"CCW  {counterclockwise:5.1f}%"
        )

    axis.text(
        *ROTATION_LABEL_POSITION,
        text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=ROTATION_LABEL_FONTSIZE,
        color="#202020",
        linespacing=1.05,
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#707070",
            "linewidth": 0.6,
            "alpha": ROTATION_LABEL_BACKGROUND_ALPHA,
        },
    )


def load_all_terminal_states(
    models: Sequence[CircularBoundaryPatternFormation],
) -> list[LastFrameStateAnalysis]:
    """Load every panel before creating a figure, preventing partial output."""

    analyses = [LastFrameStateAnalysis(model) for model in models]
    for analysis in analyses:
        print(
            f"Validated alpha={analysis.model.phaseLagA0 / np.pi:.3g} pi: "
            f"{analysis.source_snapshot_count} saved frame(s), "
            f"using terminal frame from {analysis.source_path.name}"
        )
    snapshot_counts = {analysis.source_snapshot_count for analysis in analyses}
    if len(snapshot_counts) > 1:
        print(
            "WARNING: Existing HDF5 files have different saved-frame counts. "
            "The schema has no iteration-number metadata, so the figure uses "
            "each file's actual terminal frame without claiming a shared end step."
        )
    return analyses


def create_integrated_figure(
    analyses: Sequence[LastFrameStateAnalysis],
) -> plt.Figure:
    """Create labelled panels and one phase colorbar matching StateAnalysis."""

    panel_count = len(analyses)
    columns = min(PLOT_COLUMNS, panel_count)
    if columns <= 0:
        raise ValueError("PLOT_COLUMNS must be positive.")
    rows = math.ceil(panel_count / columns)

    rc_params = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
    with plt.rc_context(rc_params):
        figure, axes_grid = plt.subplots(
            rows,
            columns,
            figsize=(
                PANEL_WIDTH * columns + 0.65,
                PANEL_HEIGHT * rows,
            ),
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes_grid.ravel()

        for index, (axis, analysis) in enumerate(zip(axes, analyses)):
            analysis.plot_spatial(axis, colorsBy="phase", index=-1)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                alpha_math_label(analysis.model.phaseLagA0),
                fontsize=TITLE_FONTSIZE,
                pad=7,
            )
            axis.text(
                0.025,
                0.975,
                f"({subplot_label(index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=LABEL_FONTSIZE,
                fontweight="bold",
                color="black",
                zorder=5,
            )
            add_rotation_annotation(axis, analysis)

        for unused_axis in axes[panel_count:]:
            unused_axis.set_visible(False)

        scalar_mappable = ScalarMappable(norm=phaseNorm, cmap=phaseCmap)
        scalar_mappable.set_array([])
        colorbar = figure.colorbar(
            scalar_mappable,
            ax=list(axes[:panel_count]),
            ticks=[0, np.pi, 2 * np.pi],
            fraction=0.035,
            pad=0.025,
            aspect=max(18, 6 * rows),
        )
        colorbar.ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        colorbar.set_label(
            r"Phase $\theta$",
            fontsize=COLORBAR_LABEL_FONTSIZE,
        )

    return figure


def run_workflow(
    *,
    output_path: Path = OUTPUT_PATH,
    generate_missing: bool = GENERATE_MISSING,
    simulation_iterations: int = SIMULATION_ITERATIONS,
    max_workers: int = MAX_WORKERS,
    check_only: bool = False,
) -> Path | None:
    models = build_sweep_models()
    ensure_data_files(
        models,
        data_dir=DATA_DIR,
        generate_missing=generate_missing,
        simulation_iterations=simulation_iterations,
        max_workers=max_workers,
    )
    analyses = load_all_terminal_states(models)

    if check_only:
        print("Preflight completed; check-only mode did not create a figure.")
        return None

    figure = create_integrated_figure(analyses)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Saved integrated figure: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot last-frame circular-boundary states while traversing phaseLagA0."
        )
    )
    parser.add_argument(
        "--no-generate-missing",
        action="store_true",
        help="Abort immediately instead of generating absent matched HDF5 files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Generation worker count (hard-capped at {HARD_WORKER_LIMIT}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Iteration count used only when generating missing data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Integrated figure path; the suffix selects the output format.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate/load every terminal frame without saving a figure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_missing = GENERATE_MISSING and not args.no_generate_missing
    max_workers = MAX_WORKERS if args.workers is None else args.workers
    simulation_iterations = (
        SIMULATION_ITERATIONS if args.iterations is None else args.iterations
    )
    output_path = OUTPUT_PATH if args.output is None else args.output.resolve()

    try:
        run_workflow(
            output_path=output_path,
            generate_missing=generate_missing,
            simulation_iterations=simulation_iterations,
            max_workers=max_workers,
            check_only=args.check_only,
        )
    except (DataPreflightError, ValueError) as exc:
        print(f"STOPPED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
