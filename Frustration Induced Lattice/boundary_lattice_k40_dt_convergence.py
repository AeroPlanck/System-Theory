"""Half-step convergence check for the strongest-coupling K-sweep cells."""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import boundary_lattice_k_sweep as sweep
import critical_boundary_lattice_analysis as critical
from CircularFigure import expected_data_path
from small_circular_alpha_sweep import ExperimentConfig, build_model


OUTPUT_DIR = sweep.OUTPUT_DIR
FINE_DT = 0.0025
FINE_ITERATIONS = 100000
FINE_SNAPSHOT_INTERVAL = 1000
CHECK_SEED = 9


def fine_config(diameter: float) -> ExperimentConfig:
    del diameter
    return ExperimentConfig(
        strengthK=40.0,
        distanceD0=sweep.DISTANCE_D0,
        speedV=sweep.SPEED_V,
        freqDist="uniform",
        omegaMin=0.0,
        deltaOmega=0.0,
        agentsNum=sweep.N_AGENTS,
        dt=FINE_DT,
        shotsnaps=FINE_SNAPSHOT_INTERVAL,
        randomSeed=CHECK_SEED,
        iterations=FINE_ITERATIONS,
    )


def fine_model(diameter: float):
    return build_model(
        diameter,
        sweep.ALPHA_OVER_PI,
        fine_config(diameter),
        sweep.DATA_DIR,
    )


def simulate_one(diameter: float) -> str:
    sweep.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = fine_config(diameter)
    model = fine_model(diameter)
    path = expected_data_path(model)
    if path.exists() and not critical.hdf_is_complete(path, config):
        model.overWrite = True
    model.run(FINE_ITERATIONS)
    if not critical.hdf_is_complete(path, config):
        raise RuntimeError(f"Incomplete convergence trajectory: {path}")
    return str(path)


def ensure_simulations() -> None:
    missing = []
    for diameter in sweep.DIAMETERS:
        model = fine_model(diameter)
        path = expected_data_path(model)
        if not critical.hdf_is_complete(path, fine_config(diameter)):
            missing.append(diameter)
    if not missing:
        print("Both half-step trajectories already exist.", flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=len(missing), mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {executor.submit(simulate_one, d): d for d in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            diameter = futures[future]
            future.result()
            print(
                f"[{index}/{len(missing)}] K=40, D={diameter:g}, "
                f"dt={FINE_DT:g}, steps={FINE_ITERATIONS}",
                flush=True,
            )


def analyze() -> pd.DataFrame:
    coarse_path = OUTPUT_DIR / "Boundary_Lattice_K_Sweep_Measurements.csv"
    coarse = pd.read_csv(coarse_path)
    rows = []
    fields = (
        "lattice_formed",
        "fourier_mode_terminal",
        "temporal_mode_median",
        "fourier_amplitude_terminal",
        "temporal_amplitude_median",
        "temporal_mode_stability",
        "shell_particle_fraction",
        "effective_wavenumber",
        "effective_arc_spacing",
        "actual_chord_mean",
        "wall_distance_of_clusters",
    )
    for diameter in sweep.DIAMETERS:
        condition = sweep.KCondition(40.0, diameter, CHECK_SEED)
        fine = sweep.measure_condition(
            condition,
            config=fine_config(diameter),
            data_dir=sweep.DATA_DIR,
        )
        coarse_row = coarse[
            np.isclose(coarse["strength_k"], 40.0)
            & np.isclose(coarse["diameter"], diameter)
            & (coarse["seed"] == CHECK_SEED)
        ].iloc[0]
        row: dict[str, object] = {
            "strength_k": 40.0,
            "diameter": diameter,
            "seed": CHECK_SEED,
            "coarse_dt": sweep.DT,
            "coarse_steps": sweep.ITERATIONS,
            "fine_dt": FINE_DT,
            "fine_steps": FINE_ITERATIONS,
            "physical_time": FINE_DT * FINE_ITERATIONS,
        }
        for field in fields:
            row[f"coarse_{field}"] = coarse_row[field]
            row[f"fine_{field}"] = fine[field]
        row["mode_match"] = bool(
            int(coarse_row["fourier_mode_terminal"])
            == int(fine["fourier_mode_terminal"])
        )
        for field in (
            "effective_wavenumber",
            "effective_arc_spacing",
            "actual_chord_mean",
            "wall_distance_of_clusters",
        ):
            row[f"relative_change_{field}"] = float(
                (fine[field] - coarse_row[field]) / coarse_row[field]
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_DIR / "K40_DT_Convergence.csv", index=False)
    report = [
        "# K=40 time-step convergence",
        "",
        "The main sweep used dt=0.005 and 50,000 steps (t=250). This check "
        "halves dt to 0.0025 and doubles the steps to 100,000 at the same "
        "physical time, using the same seed 9 initial condition.",
        "",
        output.to_markdown(index=False, floatfmt=".7g"),
        "",
        "A matching integer mode together with small spacing changes supports "
        "the claim that the K=40 result is not a time-step artifact. With only "
        "one seed per diameter this is a numerical convergence check, not a "
        "statistical test.",
        "",
    ]
    (OUTPUT_DIR / "K40_DT_Convergence.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    metadata = {
        "K": 40.0,
        "diameters": sweep.DIAMETERS,
        "seed": CHECK_SEED,
        "coarse": {"dt": sweep.DT, "steps": sweep.ITERATIONS},
        "fine": {"dt": FINE_DT, "steps": FINE_ITERATIONS},
    }
    (OUTPUT_DIR / "K40_DT_Convergence_Config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    ensure_simulations()
    print(analyze().to_string(index=False), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
