"""Generate independent long trajectories for the singular alpha=pi endpoint.

The microscopic model is regular at alpha=pi even though the eliminated-mode
continuum matrix is not.  Existing HDF5 files are never overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DATA_DIR = ROOT / "data" / "pi_endpoint_N2000_steps50000_snap50"
N = 2000
ITERATIONS = 50_000
SNAPSHOTS = 50
DT = 0.005
SEEDS = (1, 9, 17)


def build(seed: int) -> CircularBoundaryPatternFormation:
    return CircularBoundaryPatternFormation(
        strengthK=20.75,
        distanceD0=1.0,
        phaseLagA0=math.pi,
        boundaryLength=7.0,
        speedV=3.0,
        freqDist="uniform",
        omegaMin=0.0,
        deltaOmega=0.0,
        agentsNum=N,
        dt=DT,
        tqdm=False,
        savePath=str(DATA_DIR),
        shotsnaps=SNAPSHOTS,
        randomSeed=seed,
        overWrite=False,
    )


def target_path(seed: int) -> Path:
    model = build(seed)
    return DATA_DIR / f"{model}.h5"


def inspect(path: Path) -> tuple[int, int]:
    with pd.HDFStore(path, mode="r") as store:
        if not {"/positionX", "/phaseTheta"}.issubset(store.keys()):
            raise RuntimeError(f"Incomplete schema: {path}")
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


def run_one(seed: int) -> tuple[int, str, int, int]:
    nb.set_num_threads(2)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = target_path(seed)
    expected_frames = ITERATIONS // SNAPSHOTS + 1
    if destination.exists():
        frames, size = inspect(destination)
        if frames != expected_frames:
            raise RuntimeError(
                f"Refusing partial existing output ({frames} frames): {destination}"
            )
        return seed, "reused", frames, size

    model = build(seed)
    model._calc_dot_phase_collision = _cell_list_dispatch
    # Compile before opening the output file so a compile failure leaves no HDF5.
    _calc_dot_phase_cell_list(
        model.positionX, model.phaseTheta, model.freqOmega, model.dotThetaParams
    )
    temporary = DATA_DIR / f".seed_{seed}_{os.getpid()}.h5"
    positions = [model.positionX.copy()]
    phases = [model.phaseTheta.copy()]
    try:
        with pd.HDFStore(temporary, mode="w") as store:
            for step in range(1, ITERATIONS + 1):
                model.update()
                if step % SNAPSHOTS == 0:
                    positions.append(model.positionX.copy())
                    phases.append(model.phaseTheta.copy())
                if len(positions) >= 25:
                    flush(store, positions, phases)
                if step % 5000 == 0:
                    print(f"seed={seed}: {step}/{ITERATIONS}", flush=True)
            flush(store, positions, phases)
        frames, size = inspect(temporary)
        if frames != expected_frames:
            raise RuntimeError(
                f"Generated {frames} frames; expected {expected_frames}"
            )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        os.rename(temporary, destination)
        return seed, "generated", frames, size
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_kernel() -> float:
    model = build(SEEDS[0])
    reference = model._calc_dot_phase_collision(
        model.positionX, model.phaseTheta, model.freqOmega, model.dotThetaParams
    )
    optimized = _calc_dot_phase_cell_list(
        model.positionX, model.phaseTheta, model.freqOmega, model.dotThetaParams
    )
    return float(np.max(np.abs(reference - optimized)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    workers = min(max(args.workers, 1), len(SEEDS))
    error = validate_kernel()
    if error > 5e-12:
        raise RuntimeError(f"Cell-list validation failed: {error:.3e}")
    print(f"Validated optimized phase RHS: max abs error={error:.3e}", flush=True)
    if workers == 1:
        for requested_seed in SEEDS:
            seed, status, frames, size = run_one(requested_seed)
            print(
                f"seed={seed}: {status}, frames={frames}, size={size/2**20:.1f} MiB",
                flush=True,
            )
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(run_one, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed, status, frames, size = future.result()
            print(
                f"seed={seed}: {status}, frames={frames}, size={size/2**20:.1f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
