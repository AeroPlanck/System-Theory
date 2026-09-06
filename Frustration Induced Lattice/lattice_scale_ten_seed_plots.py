"""Three independent seed-resolved lattice-spacing figures (seeds 1--10)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "Lattice_Scale_Comparison"
HALF_CSV = OUT / "HalfPi_Boundary_Arc_Correlation_10Seeds.csv"
ALPHA06_CSV = OUT / "Alpha06_Hex_FirstShell_Spacing_10Seeds.csv"
PI_CSV = OUT / "Pi_Boundary_Arc_Correlation_10Seeds.csv"
SPECTRUM_CSV = (
    ROOT
    / "output"
    / "Critical_Boundary_Lattice_Quantization"
    / "Near_Critical_Dispersion_Peaks.csv"
)
PI_SPECTRUM_CSV = ROOT / "output" / "Pi_Endpoint_Lattice" / "near_pi_spectrum.csv"
PEAK_HEIGHT_MIN = 0.30
TEMPORAL_CV_MAX = 0.10
VERTICAL_SPAN = 0.50
HALFPI_YMIN = 0.90
ALPHA06_YMIN = 1.10
PI_YMIN = 1.10


def resolved_autocorrelation_peak(table: pd.DataFrame) -> pd.Series:
    """Quality gate for a reproducible local-period peak, not a cluster count."""
    coefficient_of_variation = (
        table["boundary_spacing_time_std"] / table["boundary_spacing_mean"]
    )
    return (
        (table["autocorrelation_peak_height_mean"] >= PEAK_HEIGHT_MIN)
        & (coefficient_of_variation <= TEMPORAL_CV_MAX)
    )


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Random seed")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, 11))
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def fixed_vertical_scale(ax: plt.Axes, lower: float) -> None:
    """Use the same 0.5 d0 ordinate span in every comparison panel."""
    ax.set_ylim(lower, lower + VERTICAL_SPAN)


def plot_half(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 3.25), constrained_layout=True)
    resolved = resolved_autocorrelation_peak(table)
    stable = table.loc[resolved]
    unresolved = table.loc[~resolved]
    ax.errorbar(
        stable["seed"],
        stable["boundary_spacing_mean"],
        yerr=stable["boundary_spacing_time_std"],
        fmt="o",
        color="#1769aa",
        capsize=2.5,
        markersize=5.5,
        label="Density-correlation spacing",
    )
    ax.axhline(
        1.0,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=r"$d_0$ reference",
    )
    y_unresolved = HALFPI_YMIN + VERTICAL_SPAN - 0.035
    if not unresolved.empty:
        ax.scatter(
            unresolved["seed"],
            np.full(len(unresolved), y_unresolved),
            marker="x",
            s=38,
            linewidths=1.3,
            color="#8b5a2b",
            label="No stable correlation peak",
            zorder=4,
        )
    fixed_vertical_scale(ax, HALFPI_YMIN)
    style_axis(ax, r"Boundary lattice spacing ($d_0$)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(OUT / "HalfPi_Boundary_Lattice_Seed_Robustness.pdf")
    fig.savefig(OUT / "HalfPi_Boundary_Lattice_Seed_Robustness.png", dpi=300)
    plt.close(fig)


def plot_alpha06(table: pd.DataFrame, prediction: float) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 3.25), constrained_layout=True)
    ax.errorbar(
        table["seed"],
        table["first_shell_spacing_mean"],
        yerr=table["first_shell_spacing_time_std"],
        fmt="o",
        color="#1769aa",
        capsize=2.5,
        markersize=5.5,
        label="Particle simulation",
    )
    ax.axhline(
        prediction,
        color="#b23a48",
        linestyle="--",
        linewidth=1.4,
        label=r"$a_{\mathrm{hex}}^{\mathrm{lin}}$",
    )
    fixed_vertical_scale(ax, ALPHA06_YMIN)
    style_axis(ax, r"Hexagonal lattice constant ($d_0$)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(OUT / "Alpha06_Hex_Lattice_Seed_Robustness.pdf")
    fig.savefig(OUT / "Alpha06_Hex_Lattice_Seed_Robustness.png", dpi=300)
    plt.close(fig)


def plot_pi(table: pd.DataFrame, left_limit: float) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 3.25), constrained_layout=True)
    table = table.copy()
    table["resolved"] = resolved_autocorrelation_peak(table)
    geometry_corrected = 2.0 * left_limit / np.sqrt(3.0)
    colors = {"CW": "#1769aa", "CCW": "#e07a1f"}
    offsets = {"CW": -0.10, "CCW": 0.10}
    for family in ("CW", "CCW"):
        part = table[(table["family"] == family) & table["resolved"]].sort_values(
            "seed"
        )
        ax.errorbar(
            part["seed"].to_numpy(float) + offsets[family],
            part["boundary_spacing_mean"],
            yerr=part["boundary_spacing_time_std"],
            fmt="o",
            color=colors[family],
            capsize=2.0,
            markersize=5,
            label=family,
        )
    ax.axhline(
        left_limit,
        color="0.45",
        linestyle=":",
        linewidth=1.2,
        label=r"$\ell_{\pi}^{-}=2\pi/k_{\ast}^{-}$",
    )
    ax.axhline(
        geometry_corrected,
        color="#b23a48",
        linestyle="--",
        linewidth=1.4,
        label=r"$a_{\pi}^{-,\triangle}=2\ell_{\pi}^{-}/\sqrt{3}$",
    )
    stable = table.loc[table["resolved"]]
    y_top = PI_YMIN + VERTICAL_SPAN
    unresolved = table.loc[~table["resolved"]]
    if not unresolved.empty:
        x = unresolved["seed"].to_numpy(float) + np.array(
            [offsets[family] for family in unresolved["family"]]
        )
        ax.scatter(
            x,
            np.full(len(unresolved), y_top - 0.012),
            marker="x",
            s=38,
            linewidths=1.3,
            color="#8b5a2b",
            label="No stable correlation peak",
            zorder=4,
        )
    fixed_vertical_scale(ax, PI_YMIN)
    style_axis(ax, r"Per-stream lattice spacing ($d_0$)")
    ax.legend(frameon=False, fontsize=7.4, ncols=3, loc="upper center")
    fig.savefig(OUT / "Pi_Boundary_Lattice_Seed_Robustness.pdf")
    fig.savefig(OUT / "Pi_Boundary_Lattice_Seed_Robustness.png", dpi=300)
    plt.close(fig)


def main() -> None:
    half = pd.read_csv(HALF_CSV).sort_values("seed")
    alpha06 = pd.read_csv(ALPHA06_CSV).sort_values("seed")
    pi = pd.read_csv(PI_CSV).sort_values(["seed", "family"])
    if tuple(half["seed"]) != tuple(range(1, 11)):
        raise RuntimeError("alpha=pi/2 table does not contain seeds 1--10")
    if tuple(alpha06["seed"]) != tuple(range(1, 11)):
        raise RuntimeError("alpha=0.6pi table does not contain seeds 1--10")
    if sorted(pi["seed"].unique()) != list(range(1, 11)):
        raise RuntimeError("alpha=pi table does not contain seeds 1--10")

    spectrum = pd.read_csv(SPECTRUM_CSV)
    k_star = float(
        spectrum.loc[np.isclose(spectrum["alpha_over_pi"], 0.6), "k_star"].iloc[0]
    )
    alpha06_prediction = 4.0 * np.pi / (np.sqrt(3.0) * k_star)
    pi_left_limit = float(pd.read_csv(PI_SPECTRUM_CSV).iloc[-1]["2pi/k_star"])
    pi_geometry_corrected = 2.0 * pi_left_limit / np.sqrt(3.0)

    plot_half(half)
    plot_alpha06(alpha06, alpha06_prediction)
    plot_pi(pi, pi_left_limit)
    half_resolved = resolved_autocorrelation_peak(half)
    pi_resolved = resolved_autocorrelation_peak(pi)
    half_stable = half.loc[half_resolved]
    pi_stable = pi.loc[pi_resolved]
    summary = {
        "seeds": list(range(1, 11)),
        "autocorrelation_peak_quality_gate": {
            "minimum_mean_peak_height": PEAK_HEIGHT_MIN,
            "maximum_temporal_coefficient_of_variation": TEMPORAL_CV_MAX,
        },
        "common_vertical_axis_span_d0": VERTICAL_SPAN,
        "halfpi_resolved_seeds": half.loc[half_resolved, "seed"].astype(int).tolist(),
        "halfpi_unresolved_seeds": half.loc[~half_resolved, "seed"].astype(int).tolist(),
        "halfpi_mean": float(half_stable["boundary_spacing_mean"].mean()),
        "halfpi_seed_min": float(half_stable["boundary_spacing_mean"].min()),
        "halfpi_seed_max": float(half_stable["boundary_spacing_mean"].max()),
        "alpha06_mean": float(alpha06["first_shell_spacing_mean"].mean()),
        "alpha06_seed_min": float(alpha06["first_shell_spacing_mean"].min()),
        "alpha06_seed_max": float(alpha06["first_shell_spacing_mean"].max()),
        "alpha06_prediction": alpha06_prediction,
        "alpha06_relative_difference": float(
            alpha06["first_shell_spacing_mean"].mean() / alpha06_prediction - 1.0
        ),
        "pi_unresolved_seed_streams": [
            f"seed {int(row.seed)} {row.family}"
            for row in pi.loc[~pi_resolved].itertuples()
        ],
        "pi_mean": float(pi_stable["boundary_spacing_mean"].mean()),
        "pi_seed_stream_min": float(pi_stable["boundary_spacing_mean"].min()),
        "pi_seed_stream_max": float(pi_stable["boundary_spacing_mean"].max()),
        "pi_left_limit": pi_left_limit,
        "pi_relative_difference": float(
            pi_stable["boundary_spacing_mean"].mean() / pi_left_limit - 1.0
        ),
        "pi_geometry_conversion_factor": float(2.0 / np.sqrt(3.0)),
        "pi_geometry_corrected_left_limit": pi_geometry_corrected,
        "pi_geometry_corrected_relative_difference": float(
            pi_stable["boundary_spacing_mean"].mean()
            / pi_geometry_corrected
            - 1.0
        ),
        "boundary_spacing_method": "first nonzero circular density-autocorrelation peak",
        "cluster_count_statistics_used": False,
    }
    (OUT / "Lattice_Scale_10Seed_Summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
