"""Direct first-shell hexagonal spacing for alpha=0.6*pi, seeds 1--10.

Only distance observables are exported.  No vortex/cluster-count statistic is
written to the result table.
"""

from __future__ import annotations

from pathlib import Path
import os

import numba as nb
import pandas as pd

import alpha06_bulk_hex_lattice_analysis as core


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "Lattice_Scale_Comparison"
DATA = Path(os.environ.get("FIL_DATA_DIR", ROOT / "data"))
SEEDS = tuple(range(1, 11))


def standardized_trajectory(seed: int) -> tuple[Path, int]:
    path = DATA / (
        "CircularBoundaryPatternFormation(K=20.750,D0=1.000,A0=1.885,L=7.0,"
        "v=3.0,dist=uniform,wMin=0.000,dw=0.000,N=2000,dt=0.005,"
        f"snap=50,seed={seed}).h5"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing exact trajectory: {path}")
    return path, 50


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core.trajectory = standardized_trajectory
    nb.set_num_threads(4)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        _, summary, _, _ = core.analyse_seed(seed)
        rows.append(
            {
                "seed": seed,
                "trajectory": summary["trajectory"],
                "iteration_start": summary["iteration_start"],
                "iteration_end": summary["iteration_end"],
                "sampled_terminal_frames": summary["sampled_terminal_frames"],
                "first_shell_spacing_mean": summary["first_shell_bond_mean"],
                "first_shell_spacing_time_std": summary[
                    "first_shell_bond_time_std"
                ],
                "first_shell_spacing_pooled_median": summary[
                    "first_shell_bond_median"
                ],
            }
        )
        print(
            f"seed={seed}: first-shell spacing="
            f"{summary['first_shell_bond_mean']:.6f}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "Alpha06_Hex_FirstShell_Spacing_10Seeds.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
