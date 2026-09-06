"""Direct first-shell hexagonal spacing for alpha=0.6*pi, seeds 1--10.

Only distance observables are exported.  No vortex/cluster-count statistic is
written to the result table.
"""

from __future__ import annotations

from pathlib import Path

import numba as nb
import pandas as pd

import alpha06_bulk_hex_lattice_analysis as core


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "Lattice_Scale_Comparison"
DATA = ROOT / "data" / "alpha06_bulk_N2000_steps50000_snap50"
SEEDS = tuple(range(1, 11))


def standardized_trajectory(seed: int) -> tuple[Path, int]:
    matches = list(DATA.glob(f"*seed={seed}).h5"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one standardized trajectory: seed={seed}, {matches}")
    return matches[0], 50


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
