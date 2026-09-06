"""Generate missing seed 1--10 trajectories for the lattice-scale study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from generate_missing_hdf5 import _calc_dot_phase_cell_list, _cell_list_dispatch
from main import CircularBoundaryPatternFormation


ROOT = Path(__file__).resolve().parent
N = 2000
STEPS = 50_000
SNAP = 50
DT = 0.005
SEEDS = tuple(range(1, 11))


@dataclass(frozen=True)
class Condition:
    label: str
    alpha_over_pi: float
    directory: Path
    seed: int


FAMILIES = (
    ("halfpi", 0.5, ROOT / "data" / "halfpi_boundary_N2000_steps50000_snap50"),
    ("alpha06", 0.6, ROOT / "data" / "alpha06_bulk_N2000_steps50000_snap50"),
    ("pi", 1.0, ROOT / "data" / "pi_endpoint_N2000_steps50000_snap50"),
)


def build(condition: Condition) -> CircularBoundaryPatternFormation:
    return CircularBoundaryPatternFormation(
        strengthK=20.75,
        distanceD0=1.0,
        phaseLagA0=condition.alpha_over_pi * math.pi,
        boundaryLength=7.0,
        speedV=3.0,
        freqDist="uniform",
        omegaMin=0.0,
        deltaOmega=0.0,
        agentsNum=N,
        dt=DT,
        tqdm=False,
        savePath=str(condition.directory),
        shotsnaps=SNAP,
        randomSeed=condition.seed,
        overWrite=False,
    )


def target_path(condition: Condition) -> Path:
    return condition.directory / f"{build(condition)}.h5"


def inspect(path: Path) -> tuple[int, int]:
    with pd.HDFStore(path, mode="r") as store:
        if not {"/positionX", "/phaseTheta"}.issubset(store.keys()):
            raise RuntimeError(f"Incomplete HDF5 schema: {path}")
        nx = store.get_storer("positionX").nrows
        nt = store.get_storer("phaseTheta").nrows
        if nx != nt or nx % N:
            raise RuntimeError(f"Unaligned HDF5 rows: {path}")
    return nx // N, path.stat().st_size


def flush(
    store: pd.HDFStore,
    positions: list[np.ndarray],
    phases: list[np.ndarray],
) -> None:
    if not positions:
        return
    frames = len(positions)
    index = np.tile(np.arange(N), frames)
    store.append(
        "positionX",
        pd.DataFrame(np.stack(positions).reshape(-1, 2), index=index),
    )
    store.append(
        "phaseTheta",
        pd.DataFrame(np.stack(phases).reshape(-1, 1), index=index),
    )
    positions.clear()
    phases.clear()


def validate_kernel(alpha_over_pi: float) -> float:
    condition = Condition("validation", alpha_over_pi, ROOT / "data", 1)
    model = build(condition)
    reference = model._calc_dot_phase_collision(
        model.positionX, model.phaseTheta, model.freqOmega, model.dotThetaParams
    )
    optimized = _calc_dot_phase_cell_list(
        model.positionX, model.phaseTheta, model.freqOmega, model.dotThetaParams
    )
    return float(np.max(np.abs(reference - optimized)))


def run_one(condition: Condition) -> tuple[str, int, str, int, int]:
    nb.set_num_threads(2)
    condition.directory.mkdir(parents=True, exist_ok=True)
    destination = target_path(condition)
    expected_frames = STEPS // SNAP + 1
    if destination.exists():
        frames, size = inspect(destination)
        if frames != expected_frames:
            raise RuntimeError(
                f"Refusing partial existing file ({frames} frames): {destination}"
            )
        return condition.label, condition.seed, "reused", frames, size

    model = build(condition)
    model._calc_dot_phase_collision = _cell_list_dispatch
    _calc_dot_phase_cell_list(
        model.positionX, model.phaseTheta, model.freqOmega, model.dotThetaParams
    )
    temporary = condition.directory / (
        f".{condition.label}_seed_{condition.seed}_{os.getpid()}.h5"
    )
    positions = [model.positionX.copy()]
    phases = [model.phaseTheta.copy()]
    try:
        with pd.HDFStore(temporary, mode="w") as store:
            for step in range(1, STEPS + 1):
                model.update()
                if step % SNAP == 0:
                    positions.append(model.positionX.copy())
                    phases.append(model.phaseTheta.copy())
                if len(positions) >= 25:
                    flush(store, positions, phases)
                if step % 10_000 == 0:
                    print(
                        f"{condition.label} seed={condition.seed}: {step}/{STEPS}",
                        flush=True,
                    )
            flush(store, positions, phases)
        frames, size = inspect(temporary)
        if frames != expected_frames:
            raise RuntimeError(f"Generated {frames} frames; expected {expected_frames}")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        os.rename(temporary, destination)
        return condition.label, condition.seed, "generated", frames, size
    finally:
        if temporary.exists():
            temporary.unlink()


def all_conditions() -> list[Condition]:
    return [
        Condition(label, alpha, directory, seed)
        for label, alpha, directory in FAMILIES
        for seed in SEEDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for _, alpha, _ in FAMILIES:
        error = validate_kernel(alpha)
        if error > 5e-12:
            raise RuntimeError(
                f"Optimized kernel validation failed at alpha/pi={alpha}: {error:.3e}"
            )
        print(
            f"Validated alpha/pi={alpha:g}: max abs RHS error={error:.3e}",
            flush=True,
        )

    conditions = all_conditions()
    jobs: list[Condition] = []
    for condition in conditions:
        path = target_path(condition)
        if path.exists():
            frames, _ = inspect(path)
            if frames != STEPS // SNAP + 1:
                raise RuntimeError(f"Partial existing trajectory: {path}")
            print(f"{condition.label} seed={condition.seed}: reused", flush=True)
        else:
            jobs.append(condition)
    print(f"Missing trajectories: {len(jobs)}", flush=True)
    if not jobs:
        return

    workers = min(max(1, args.workers), 4, len(jobs))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(run_one, condition): condition for condition in jobs}
        for future in as_completed(futures):
            label, seed, status, frames, size = future.result()
            print(
                f"{label} seed={seed}: {status}, frames={frames}, "
                f"size={size / 2**20:.1f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
