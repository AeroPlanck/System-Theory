"""Generate only absent HDF5 trajectories for the refined alpha sweep.

Safety contract
---------------
* Existing ``data/*.h5`` files are never opened for writing, continued, moved,
  or overwritten.
* Every new trajectory is written to a private staging directory, validated,
  and then published atomically without replacing an existing destination.
* At most two simulations can run concurrently.

The user-adjustable defaults are collected immediately below.  Command-line
arguments may lower the worker count or change the iteration count, but the
two-worker device limit cannot be exceeded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback

import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import boundary_defect_analysis as bda
from phase_informed_boundary_flow_analysis import CASE_SPECS, build_model


ROOT = Path(__file__).resolve().parent


# =============================================================================
# USER-ADJUSTABLE GENERATION PARAMETERS
# =============================================================================

ALPHA_OVER_PI: tuple[float, ...] = tuple(index / 10 for index in range(11))
ITERATIONS: int = 80_000
MAXIMUM_WORKERS: int = 2
DEFAULT_WORKERS: int = 2
PROGRESS_INTERVAL: int = 1_000
HDF5_SNAPSHOTS_PER_BATCH: int = 100
NUMBA_THREADS_PER_WORKER: int = 4
KERNEL_VALIDATION_TOLERANCE: float = 5.0e-12
DATA_DIRECTORY: Path = ROOT / "data"
STAGING_DIRECTORY: Path = ROOT / "output" / "H5"
MAXIMUM_WINDOWS_HDF5_PATH_LENGTH: int = 259


# =============================================================================
# GENERATION IMPLEMENTATION
# =============================================================================


@dataclass(frozen=True)
class Job:
    job_id: int
    case_index: int
    condition: str
    alpha_over_pi: float
    target_path: Path


@dataclass(frozen=True)
class Result:
    job_id: int
    condition: str
    alpha_over_pi: float
    target_path: Path
    frames: int
    size_bytes: int
    status: str


@nb.njit(cache=True, parallel=True)
def _calc_dot_phase_cell_list(
    position_x: np.ndarray,
    phase_theta: np.ndarray,
    freq_omega: np.ndarray,
    params: tuple[float, ...],
) -> np.ndarray:
    """Exact cutoff interaction using a cell list and a trigonometric identity."""
    agents_num = position_x.shape[0]
    distance_d0 = params[2]
    cell_size = np.nextafter(distance_d0, np.inf)
    strength_k = params[3]
    phase_lag_a0 = params[4]
    min_x = np.min(position_x[:, 0])
    max_x = np.max(position_x[:, 0])
    min_y = np.min(position_x[:, 1])
    max_y = np.max(position_x[:, 1])
    cells_x = max(1, int(np.floor((max_x - min_x) / cell_size)) + 1)
    cells_y = max(1, int(np.floor((max_y - min_y) / cell_size)) + 1)
    heads = np.full(cells_x * cells_y, -1, dtype=np.int64)
    links = np.full(agents_num, -1, dtype=np.int64)

    for particle in range(agents_num - 1, -1, -1):
        cell_x = int(np.floor((position_x[particle, 0] - min_x) / cell_size))
        cell_y = int(np.floor((position_x[particle, 1] - min_y) / cell_size))
        cell_x = min(max(cell_x, 0), cells_x - 1)
        cell_y = min(max(cell_y, 0), cells_y - 1)
        flat_cell = cell_y * cells_x + cell_x
        links[particle] = heads[flat_cell]
        heads[flat_cell] = particle

    output = np.empty(agents_num, dtype=np.float64)
    cos_phase = np.cos(phase_theta)
    sin_phase = np.sin(phase_theta)
    sin_lag = np.sin(phase_lag_a0)
    for particle in nb.prange(agents_num):
        center_x = int(
            np.floor((position_x[particle, 0] - min_x) / cell_size)
        )
        center_y = int(
            np.floor((position_x[particle, 1] - min_y) / cell_size)
        )
        center_x = min(max(center_x, 0), cells_x - 1)
        center_y = min(max(center_y, 0), cells_y - 1)
        neighbor_count = 0
        neighbor_cos_sum = 0.0
        neighbor_sin_sum = 0.0

        # A cell is one ulp wider than d0, so every point within the exact d0
        # cutoff lies in the same or an immediately adjacent cell.
        for cell_y in range(max(0, center_y - 1), min(cells_y, center_y + 2)):
            for cell_x in range(max(0, center_x - 1), min(cells_x, center_x + 2)):
                other = heads[cell_y * cells_x + cell_x]
                while other >= 0:
                    dx = position_x[other, 0] - position_x[particle, 0]
                    dy = position_x[other, 1] - position_x[particle, 1]
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance <= distance_d0 and distance > 0.0:
                        neighbor_count += 1
                        neighbor_cos_sum += cos_phase[other]
                        neighbor_sin_sum += sin_phase[other]
                    other = links[other]

        if neighbor_count == 0:
            output[particle] = freq_omega[particle]
            continue
        shifted_lag = phase_lag_a0 - phase_theta[particle]
        phase_sum = (
            np.cos(shifted_lag) * neighbor_sin_sum
            + np.sin(shifted_lag) * neighbor_cos_sum
        )
        coupling = phase_sum / neighbor_count - sin_lag
        output[particle] = strength_k * coupling + freq_omega[particle]
    return output


def _cell_list_dispatch(positionX, phaseTheta, freqOmega, params):
    """Preserve the keyword interface used by ``main.py`` properties."""
    return _calc_dot_phase_cell_list(positionX, phaseTheta, freqOmega, params)


def _install_optimized_kernel(model) -> None:
    model._calc_dot_phase_collision = _cell_list_dispatch


def validate_optimized_kernel() -> float:
    """Cross-check optimized and reference updates for all four geometries."""
    maximum_error = 0.0
    for case_index, spec in enumerate(CASE_SPECS):
        alpha = 0.3 if case_index % 2 == 0 else 0.9
        reference = build_model(spec, alpha)
        optimized = build_model(spec, alpha)
        reference_phase_rate = reference.dotPhase.copy()
        optimized_phase_rate = _calc_dot_phase_cell_list(
            optimized.positionX,
            optimized.phaseTheta,
            optimized.freqOmega,
            optimized.dotThetaParams,
        )
        error = float(np.max(np.abs(reference_phase_rate - optimized_phase_rate)))
        maximum_error = max(maximum_error, error)
        if not np.allclose(
            reference_phase_rate,
            optimized_phase_rate,
            rtol=0.0,
            atol=KERNEL_VALIDATION_TOLERANCE,
        ):
            raise RuntimeError(
                f"Optimized interaction kernel failed for {spec.condition}: "
                f"max error={error:.3e}"
            )

        # One complete step also audits the class-specific collision boundary.
        reference.update()
        _install_optimized_kernel(optimized)
        optimized.update()
        position_error = float(
            np.max(np.abs(reference.positionX - optimized.positionX))
        )
        phase_error = float(
            np.max(np.abs(reference.phaseTheta - optimized.phaseTheta))
        )
        maximum_error = max(maximum_error, position_error, phase_error)
        if position_error > KERNEL_VALIDATION_TOLERANCE or phase_error > KERNEL_VALIDATION_TOLERANCE:
            raise RuntimeError(
                f"Optimized full update failed for {spec.condition}: "
                f"position error={position_error:.3e}, phase error={phase_error:.3e}"
            )
    return maximum_error


def _inspect_hdf(path: Path, agents_num: int, expected_frames: int) -> tuple[int, int]:
    """Return ``(frames, bytes)`` after enforcing the trajectory schema."""
    with pd.HDFStore(path, mode="r") as store:
        required = {"/positionX", "/phaseTheta"}
        if not required.issubset(store.keys()):
            raise RuntimeError(f"{path} lacks positionX or phaseTheta")
        position = store.get_storer("positionX")
        phase = store.get_storer("phaseTheta")
        if position.ncols != 2 or phase.ncols != 1:
            raise RuntimeError(f"{path} has an unexpected column schema")
        if position.nrows != phase.nrows or position.nrows % agents_num:
            raise RuntimeError(f"{path} contains incomplete or unaligned frames")
        frames = position.nrows // agents_num
    if frames != expected_frames:
        raise RuntimeError(
            f"{path} has {frames} frames; expected exactly {expected_frames}"
        )
    return frames, path.stat().st_size


def _publish_exclusive(staged_path: Path, final_path: Path) -> None:
    """Publish atomically without overwriting an existing destination."""
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing HDF5: {final_path}")
    if os.name == "nt":
        # Windows rename is atomic within the volume and fails when the target
        # exists.  It is more broadly permitted than creating a hard link.
        os.rename(staged_path, final_path)
    else:
        # POSIX rename may replace a destination, whereas link is exclusive.
        os.link(staged_path, final_path)
        staged_path.unlink()


def _write_progress(progress_path: Path, completed_steps: int) -> None:
    """Publish one immutable progress marker; never replace a file being read."""
    marker = progress_path.parent / f"{progress_path.name}_{completed_steps:06d}"
    marker.touch(exist_ok=True)


def _flush_snapshots(
    store: pd.HDFStore,
    position_buffer: list[np.ndarray],
    phase_buffer: list[np.ndarray],
    agents_num: int,
) -> None:
    """Append buffered frames while preserving the per-frame particle index."""
    if not position_buffer:
        return
    frame_count = len(position_buffer)
    repeated_index = np.tile(np.arange(agents_num), frame_count)
    positions = np.stack(position_buffer, axis=0).reshape(-1, 2)
    phases = np.stack(phase_buffer, axis=0).reshape(-1, 1)
    store.append(
        key="positionX",
        value=pd.DataFrame(positions, index=repeated_index),
    )
    store.append(
        key="phaseTheta",
        value=pd.DataFrame(phases, index=repeated_index),
    )
    position_buffer.clear()
    phase_buffer.clear()


def _run_model_with_progress(
    model,
    iterations: int,
    progress_path: Path,
    staged_path: Path,
    cancel_path: Path,
) -> None:
    """Run standard dynamics and save identical frames in larger HDF batches."""
    if staged_path.exists():
        raise RuntimeError("A supposedly private staging HDF5 already exists")
    position_buffer = [model.positionX.copy()]
    phase_buffer = [model.phaseTheta.copy()]
    with pd.HDFStore(staged_path, mode="w") as store:
        for index in range(model.counts, iterations):
            model.update()
            model.counts = index + 1
            if model.counts % model.shotsnaps == 0:
                position_buffer.append(model.positionX.copy())
                phase_buffer.append(model.phaseTheta.copy())
            if len(position_buffer) >= HDF5_SNAPSHOTS_PER_BATCH:
                _flush_snapshots(
                    store,
                    position_buffer,
                    phase_buffer,
                    model.agentsNum,
                )
            if model.counts % PROGRESS_INTERVAL == 0:
                _write_progress(progress_path, model.counts)
                if cancel_path.exists():
                    raise RuntimeError("Cancelled after a peer generation failure")
        if model.counts % model.shotsnaps != 0:
            position_buffer.append(model.positionX.copy())
            phase_buffer.append(model.phaseTheta.copy())
        _flush_snapshots(
            store,
            position_buffer,
            phase_buffer,
            model.agentsNum,
        )
    _write_progress(progress_path, iterations)


def _generate_one(
    job: Job,
    iterations: int,
    progress_path: Path,
    cancel_path: Path,
    error_path: Path,
) -> Result:
    """Worker entry point; publish only a complete, validated new file."""
    try:
        nb.set_num_threads(NUMBA_THREADS_PER_WORKER)
        spec = CASE_SPECS[job.case_index]
        model = build_model(spec, job.alpha_over_pi)
        _install_optimized_kernel(model)
        final_path = bda.data_path(model)
        if final_path != job.target_path:
            raise RuntimeError(
                f"Target-name drift for {job.condition}, "
                f"alpha/pi={job.alpha_over_pi:.1f}: "
                f"{final_path} != {job.target_path}"
            )
        if final_path.exists():
            return Result(
                job.job_id,
                job.condition,
                job.alpha_over_pi,
                final_path,
                0,
                final_path.stat().st_size,
                "skipped-existing",
            )

        temporary = progress_path.parent / f"J{job.job_id:02d}"
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            model.savePath = str(temporary)
            model.overWrite = False
            model.tqdm = False
            staged_path = temporary / f"{model}.h5"
            if (
                os.name == "nt"
                and len(str(staged_path.resolve()))
                > MAXIMUM_WINDOWS_HDF5_PATH_LENGTH
            ):
                raise RuntimeError(
                    f"Staging HDF5 path is too long ({len(str(staged_path.resolve()))}): "
                    f"{staged_path}"
                )
            _run_model_with_progress(
                model,
                iterations,
                progress_path,
                staged_path,
                cancel_path,
            )
            expected_frames = math.ceil(iterations / model.shotsnaps) + 1
            frames, size_bytes = _inspect_hdf(
                staged_path,
                agents_num=model.agentsNum,
                expected_frames=expected_frames,
            )

            _publish_exclusive(staged_path, final_path)
            return Result(
                job.job_id,
                job.condition,
                job.alpha_over_pi,
                final_path,
                frames,
                size_bytes,
                "generated",
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _existing_metadata() -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in DATA_DIRECTORY.glob("*.h5")
    }


def _assert_existing_unchanged(before: dict[Path, tuple[int, int]]) -> None:
    changed = []
    for path, metadata in before.items():
        if not path.is_file():
            changed.append(f"missing: {path}")
        elif (path.stat().st_size, path.stat().st_mtime_ns) != metadata:
            changed.append(f"changed: {path}")
    if changed:
        raise RuntimeError(
            "Pre-existing HDF5 integrity check failed:\n  " + "\n  ".join(changed)
        )


def find_missing_jobs(
    only_condition: str | None = None,
    only_alpha: float | None = None,
) -> tuple[list[Job], int]:
    """Build exact refined-grid targets and retain only absent files."""
    selected_specs = [
        (case_index, spec)
        for case_index, spec in enumerate(CASE_SPECS)
        if only_condition is None or spec.condition == only_condition
    ]
    if not selected_specs:
        raise ValueError(f"Unknown condition: {only_condition}")
    if only_alpha is None:
        selected_alphas = ALPHA_OVER_PI
    else:
        matches = [alpha for alpha in ALPHA_OVER_PI if math.isclose(alpha, only_alpha)]
        if len(matches) != 1:
            raise ValueError(
                f"alpha/pi={only_alpha} is outside the configured refined grid"
            )
        selected_alphas = (matches[0],)
    jobs: list[Job] = []
    exact_target_count = 0
    for case_index, spec in selected_specs:
        for alpha in selected_alphas:
            model = build_model(spec, alpha)
            target = bda.data_path(model)
            exact_target_count += 1
            if not target.is_file():
                jobs.append(
                    Job(
                        job_id=len(jobs),
                        case_index=case_index,
                        condition=spec.condition,
                        alpha_over_pi=alpha,
                        target_path=target,
                    )
                )
    return jobs, exact_target_count


def _print_plan(jobs: list[Job], exact_target_count: int, iterations: int, workers: int) -> None:
    print(
        f"Exact refined-grid targets: {exact_target_count}; "
        f"existing: {exact_target_count - len(jobs)}; missing: {len(jobs)}"
    )
    print(f"Iterations per missing file: {iterations:,}; workers: {workers}")
    for job in jobs:
        print(
            f"  [{job.job_id + 1:02d}/{len(jobs):02d}] "
            f"{job.condition}, alpha/pi={job.alpha_over_pi:.1f}\n"
            f"      {job.target_path.name}"
        )


def generate_missing(
    iterations: int,
    workers: int,
    dry_run: bool = False,
    only_condition: str | None = None,
    only_alpha: float | None = None,
) -> list[Result]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 1 <= workers <= MAXIMUM_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAXIMUM_WORKERS}; "
            "the device limit cannot be overridden"
        )
    jobs, exact_target_count = find_missing_jobs(only_condition, only_alpha)
    _print_plan(jobs, exact_target_count, iterations, workers)
    if dry_run or not jobs:
        return []

    nb.set_num_threads(NUMBA_THREADS_PER_WORKER)
    validation_error = validate_optimized_kernel()
    print(
        "Validated cell-list interaction against main.py for all geometries; "
        f"maximum absolute error={validation_error:.3e}"
    )
    before = _existing_metadata()
    context = mp.get_context("spawn")
    results: list[Result] = []
    failures: list[str] = []
    latest_steps = {job.job_id: 0 for job in jobs}
    progress_directory = STAGING_DIRECTORY / f"R{os.getpid()}"
    progress_directory.mkdir(parents=True, exist_ok=False)
    cancel_path = progress_directory / "C.flag"

    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            pending: dict[Future, Job] = {}
            next_job_index = 0
            while next_job_index < min(workers, len(jobs)):
                job = jobs[next_job_index]
                future = executor.submit(
                    _generate_one,
                    job,
                    iterations,
                    progress_directory / f"P{job.job_id:02d}",
                    cancel_path,
                    progress_directory / f"E{job.job_id:02d}",
                )
                pending[future] = job
                next_job_index += 1
            with (
                tqdm(
                    total=len(jobs),
                    desc="HDF5 files",
                    unit="file",
                    position=0,
                    dynamic_ncols=True,
                ) as file_bar,
                tqdm(
                    total=len(jobs) * iterations,
                    desc="Simulation steps",
                    unit="step",
                    position=1,
                    dynamic_ncols=True,
                ) as step_bar,
            ):
                while pending:
                    for job in pending.values():
                        progress_path = (
                            progress_directory / f"P{job.job_id:02d}"
                        )
                        try:
                            completed_steps = max(
                                int(marker.name.rsplit("_", 1)[1])
                                for marker in progress_directory.glob(
                                    f"{progress_path.name}_*"
                                )
                            )
                        except (OSError, ValueError, StopIteration):
                            continue
                        delta = max(
                            0, completed_steps - latest_steps[job.job_id]
                        )
                        latest_steps[job.job_id] = max(
                            latest_steps[job.job_id], completed_steps
                        )
                        step_bar.update(delta)

                    completed = [future for future in pending if future.done()]
                    for future in completed:
                        job = pending.pop(future)
                        try:
                            result = future.result()
                            results.append(result)
                            remaining = iterations - latest_steps[job.job_id]
                            if remaining > 0:
                                step_bar.update(remaining)
                                latest_steps[job.job_id] = iterations
                            tqdm.write(
                                f"{result.status}: {result.condition}, "
                                f"alpha/pi={result.alpha_over_pi:.1f}, "
                                f"frames={result.frames:,}, "
                                f"size={result.size_bytes / (1024 ** 3):.3f} GiB"
                            )
                        except Exception:
                            error_path = (
                                progress_directory / f"E{job.job_id:02d}"
                            )
                            try:
                                remote_error = error_path.read_text(encoding="utf-8")
                            except (FileNotFoundError, OSError):
                                remote_error = traceback.format_exc()
                            failure = (
                                f"{job.condition}, alpha/pi={job.alpha_over_pi:.1f}\n"
                                f"{remote_error}"
                            )
                            if not failures:
                                failures.append(failure)
                                cancel_path.write_text("cancel", encoding="ascii")
                                tqdm.write(
                                    f"FAILED; stopping new submissions: "
                                    f"{job.condition}, "
                                    f"alpha/pi={job.alpha_over_pi:.1f}\n{failure}"
                                )
                            else:
                                tqdm.write(
                                    f"cancelled after peer failure: {job.condition}, "
                                    f"alpha/pi={job.alpha_over_pi:.1f}"
                                )
                        file_bar.update(1)
                    while (
                        not failures
                        and next_job_index < len(jobs)
                        and len(pending) < workers
                    ):
                        job = jobs[next_job_index]
                        future = executor.submit(
                            _generate_one,
                            job,
                            iterations,
                            progress_directory / f"P{job.job_id:02d}",
                            cancel_path,
                            progress_directory / f"E{job.job_id:02d}",
                        )
                        pending[future] = job
                        next_job_index += 1
                    if pending:
                        time.sleep(0.5)
    finally:
        shutil.rmtree(progress_directory, ignore_errors=True)

    _assert_existing_unchanged(before)
    if failures:
        raise RuntimeError(
            f"{len(failures)} generation job(s) failed:\n\n" + "\n\n".join(failures)
        )

    # Re-read every newly published file from disk after all worker processes
    # have exited.  This is the final completeness check.
    jobs_by_id = {job.job_id: job for job in jobs}
    for result in results:
        if result.status != "generated":
            continue
        job = jobs_by_id[result.job_id]
        model = build_model(CASE_SPECS[job.case_index], result.alpha_over_pi)
        expected_frames = math.ceil(iterations / model.shotsnaps) + 1
        _inspect_hdf(result.target_path, model.agentsNum, expected_frames)
    try:
        STAGING_DIRECTORY.rmdir()
    except OSError:
        # Retain the directory only if it contains staging material from a
        # different or interrupted invocation.
        pass
    return sorted(results, key=lambda item: item.job_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help=f"Iterations for each missing trajectory (default: {ITERATIONS:,}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAXIMUM_WORKERS + 1),
        default=DEFAULT_WORKERS,
        help=f"Concurrent simulations; hard maximum {MAXIMUM_WORKERS}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List exact missing targets without writing data.",
    )
    parser.add_argument(
        "--only-condition",
        choices=tuple(spec.condition for spec in CASE_SPECS),
        help="Restrict generation to one configured geometry.",
    )
    parser.add_argument(
        "--only-alpha",
        type=float,
        help="Restrict generation to one configured alpha/pi value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = generate_missing(
        args.iterations,
        args.workers,
        args.dry_run,
        args.only_condition,
        args.only_alpha,
    )
    if args.dry_run:
        print("Dry run complete; no HDF5 file was written.")
        return 0
    generated = sum(result.status == "generated" for result in results)
    skipped = sum(result.status == "skipped-existing" for result in results)
    print(f"Complete: generated={generated}, skipped-existing={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
