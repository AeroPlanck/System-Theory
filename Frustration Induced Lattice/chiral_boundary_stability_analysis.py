"""Measure handedness-gated chiral boundary-flow stability from HDF5 data.

The complete discrete definition is

    S_chi = D_chi * (M_chi * U_t * C_s) ** (1 / 3),

where D_chi is a hard handedness gate, M_chi measures block-current strength,
U_t measures continuity of that strength, and C_s measures cumulative
perimeter coverage.  Every plotted symbol is derived explicitly from the
stored positionX and phaseTheta arrays.  This script only reads HDF5 files and
only writes new analysis products below output/Handedness_Gated_Chiral_Stability.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import dedent

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boundary_defect_analysis as bda
import main as model_library


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "Handedness_Gated_Chiral_Stability"
BLOCK_COUNT = 20
ARC_BIN_COUNT = 64
ALPHAS = tuple(float(value) for value in bda.ALPHA_OVER_PI)
MIN_TRANSPORT_EPISODE_TIME = 0.5
MIN_DIRECTIONAL_PERSISTENCE = 0.75
MIN_TANGENTIALITY = 0.35
EPS = np.finfo(float).eps

NAVY = "#18344F"
TEAL = "#157A74"
GOLD = "#C28B2C"
RED = "#A64545"
MUTED = "#5D6873"
GRID = "#DCE2E7"
PALE_GOLD = "#FBF3E3"


CASE_SPECS = (
    ("Square", "square", 0.0, "Square - no defect"),
    ("Square", "square_defect", 1.0, "Square - four defects H=1"),
    ("Square", "square_defect", 1.5, "Square - four defects H=1.5"),
    ("Square", "square_defect", 3.0, "Square - four defects H=3"),
    ("Circular", "circle", 0.0, "Circular - no defect"),
    ("Circular", "circle_defect", 3.0, "Circular - single defect H=3"),
)


def build_model(label: str, alpha_over_pi: float, height: float):
    """Build one exact existing-data model; alpha is the only sweep variable."""
    if label == "circle":
        return bda.build_model(model_library.CircularBoundaryPatternFormation, alpha_over_pi)
    if label == "circle_defect":
        return bda.build_model(
            model_library.CollisionBoundaryMidpointSpikePatternFormation,
            alpha_over_pi,
            protrusionHeight=height,
            protrusionHalfWidth=bda.SPIKE_HALF_WIDTH,
        )
    if label == "square":
        return bda.build_model(model_library.CollisionBoundaryPatternFormation, alpha_over_pi)
    if label == "square_defect":
        return bda.build_model(
            model_library.CollisionBoundaryFourSpikePatternFormation,
            alpha_over_pi,
            protrusionHeight=height,
            protrusionHalfWidth=bda.SPIKE_HALF_WIDTH,
        )
    raise ValueError(f"Unsupported model label: {label}")


def transport_episode_mask(
    edge: np.ndarray, tangential_velocity: np.ndarray, saved_dt: float,
) -> np.ndarray:
    """Return z[n,i]: frames belonging to an accepted boundary episode."""
    frames, particles = edge.shape
    minimum_frames = max(2, int(math.ceil(MIN_TRANSPORT_EPISODE_TIME / saved_dt)))
    accepted = np.zeros((frames, particles), dtype=bool)
    for particle in range(particles):
        padded = np.concatenate(([False], edge[:, particle], [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            if stop - start < minimum_frames:
                continue
            segment = tangential_velocity[start:stop, particle]
            signed_sum = float(segment.sum())
            absolute_sum = float(np.abs(segment).sum())
            persistence = abs(signed_sum) / absolute_sum if absolute_sum > EPS else 0.0
            tangentiality = float(np.mean(np.abs(segment)))
            if (
                persistence >= MIN_DIRECTIONAL_PERSISTENCE
                and tangentiality >= MIN_TANGENTIALITY
            ):
                accepted[start:stop, particle] = True
    return accepted


def versioned_path(base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = base.with_name(f"{base.stem}_V{index}{base.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = versioned_path(OUTPUT / f"{stem}.png")
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=420, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
        }
    )


def _block_arc_flux(model) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    window = bda.load_window(model)
    geometry = bda.geometry_for(model)
    distance, arc, tangent = bda.project_boundary(window.positions, geometry)
    heading = np.stack([np.cos(window.phases), np.sin(window.phases)], axis=-1)
    q = np.einsum("...j,...j->...", heading, tangent)
    width, _, _, _, _ = bda.adaptive_edge_width(distance, q, geometry.length_scale)
    edge = distance <= width
    saved_dt = model.dt * model.shotsnaps
    valid = transport_episode_mask(edge, q, saved_dt)

    arc_index = np.floor(np.mod(arc, geometry.perimeter) / geometry.perimeter * ARC_BIN_COUNT).astype(int)
    arc_index = np.clip(arc_index, 0, ARC_BIN_COUNT - 1)
    blocks = np.array_split(np.arange(window.positions.shape[0]), BLOCK_COUNT)
    signed_flux = np.zeros((BLOCK_COUNT, ARC_BIN_COUNT), dtype=float)
    absolute_flux = np.zeros_like(signed_flux)
    block_current = np.zeros(BLOCK_COUNT, dtype=float)

    for block_index, block in enumerate(blocks):
        selected = valid[block]
        bins = arc_index[block][selected]
        values = q[block][selected]
        signed_flux[block_index] = np.bincount(
            bins, weights=values, minlength=ARC_BIN_COUNT
        )
        absolute_flux[block_index] = np.bincount(
            bins, weights=np.abs(values), minlength=ARC_BIN_COUNT
        )
        denominator = float(absolute_flux[block_index].sum())
        block_current[block_index] = (
            float(signed_flux[block_index].sum()) / denominator if denominator > EPS else 0.0
        )

    direction_sum = float(block_current.sum())
    direction = 0.0 if abs(direction_sum) <= EPS else float(np.sign(direction_sum))
    direction_fixedness = float(
        abs(direction_sum) / (np.abs(block_current).sum() + EPS)
    )
    magnitude = np.abs(block_current)
    temporal_uniformity = float(
        magnitude.sum() ** 2 / (BLOCK_COUNT * np.square(magnitude).sum() + EPS)
    )

    coherent_flux = np.maximum(direction * signed_flux, 0.0)
    cumulative_coherent_flux = coherent_flux.sum(axis=0)
    coherent_total = float(cumulative_coherent_flux.sum())
    spatial_coverage = float(
        coherent_total ** 2
        / (ARC_BIN_COUNT * np.square(cumulative_coherent_flux).sum() + EPS)
    )
    spatial_profile = np.divide(
        cumulative_coherent_flux,
        coherent_total,
        out=np.zeros_like(cumulative_coherent_flux),
        where=coherent_total > EPS,
    )
    current_strength = float(magnitude.mean())
    signed_mean_current = float(block_current.mean())
    directed_current = float(abs(signed_mean_current))
    stability_core = float(
        np.cbrt(max(0.0, current_strength * temporal_uniformity * spatial_coverage))
    )
    stability = float(direction_fixedness * stability_core)
    normalized_arc_current = np.divide(
        signed_flux,
        absolute_flux,
        out=np.zeros_like(signed_flux),
        where=absolute_flux > EPS,
    )

    summary = {
        "alpha_over_pi": float(model.phaseLagA0 / np.pi),
        "defect_height": float(getattr(model, "protrusionHeight", 0.0)),
        "majority_direction": "CCW" if direction > 0 else ("CW" if direction < 0 else "None"),
        "direction_fixedness": direction_fixedness,
        "current_strength": current_strength,
        "temporal_uniformity": temporal_uniformity,
        "spatial_coverage": spatial_coverage,
        "signed_mean_block_current": signed_mean_current,
        "directed_current": directed_current,
        "stability_core": stability_core,
        "handedness_gated_stability": stability,
        "active_blocks": int(np.count_nonzero(magnitude > EPS)),
        "accepted_particle_frames": int(np.count_nonzero(valid)),
        "adaptive_edge_width_over_L": width / geometry.length_scale,
        "terminal_saved_frame": int(window.total_frames - 1),
        "hdf5_file": bda.data_path(model).name,
    }
    detail = {
        "block_current": block_current,
        "normalized_arc_current": normalized_arc_current,
        "spatial_profile": spatial_profile,
        "block_time": np.array([
            float(block.mean() * saved_dt) for block in blocks
        ]),
        "arc_fraction": (np.arange(ARC_BIN_COUNT) + 0.5) / ARC_BIN_COUNT,
    }
    return summary, detail


def compute_all() -> tuple[pd.DataFrame, dict[tuple[str, float], dict[str, np.ndarray]]]:
    specifications = []
    models = []
    for group, label, height, condition in CASE_SPECS:
        for alpha in ALPHAS:
            model = build_model(label, alpha, height)
            models.append(model)
            specifications.append((group, label, height, condition, alpha, model))
    bda.validate_exact_files(models)

    rows = []
    details: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for index, (group, label, height, condition, alpha, model) in enumerate(specifications, start=1):
        print(f"Stability {index}/{len(specifications)}: {bda.data_path(model).name}")
        summary, detail = _block_arc_flux(model)
        summary.update(
            {
                "geometry_group": group,
                "model_label": label,
                "condition": condition,
            }
        )
        rows.append(summary)
        details[(condition, float(alpha))] = detail

    table = pd.DataFrame(rows).sort_values(
        ["geometry_group", "defect_height", "alpha_over_pi"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "Handedness_Gated_Chiral_Stability_Values.csv", index=False)
    return table, details


def _condition_series(group: str):
    if group == "Square":
        return [
            (0.0, "No defect", NAVY),
            (1.0, "H=1", "#2E6F89"),
            (1.5, "H=1.5", TEAL),
            (3.0, "H=3", "#8FBF26"),
        ]
    return [(0.0, "No defect", NAVY), (3.0, "H=3", GOLD)]


def plot_sweep_and_phase(table: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplot_mosaic(
        [["square", "circle"], ["phase", "phase"]],
        figsize=(10.8, 8.0), constrained_layout=True,
        height_ratios=[1.0, 1.15],
    )
    for key, group, title in (
        ("square", "Square", "Square boundary and four symmetric defects"),
        ("circle", "Circular", "Circular boundary and one defect"),
    ):
        axis = axes[key]
        subset = table[table.geometry_group == group]
        for height, label, color in _condition_series(group):
            part = subset[np.isclose(subset.defect_height, height)].sort_values("alpha_over_pi")
            axis.plot(
                part.alpha_over_pi,
                part.handedness_gated_stability,
                marker="o", ms=5, lw=2, color=color, label=label,
            )
        axis.set(
            title=title,
            xlabel=r"Phase lag $\alpha$",
            ylabel=r"Handedness-gated stability $\widetilde{\mathcal{S}}_{\chi}$",
            xlim=(-0.02, 1.02), ylim=(-0.03, 1.03),
        )
        axis.set_xticks(
            np.arange(0, 1.01, 0.2),
            ["0", r"$0.2\pi$", r"$0.4\pi$", r"$0.6\pi$", r"$0.8\pi$", r"$\pi$"],
        )
        axis.grid(True, color=GRID, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes["phase"]
    markers = {
        "circle": "o", "circle_defect": "D", "square": "s", "square_defect": "^",
    }
    labels_used = set()
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.colormaps["viridis"]
    for _, item in table.iterrows():
        label = item.condition if item.condition not in labels_used else None
        labels_used.add(item.condition)
        axis.scatter(
            item.direction_fixedness, item.handedness_gated_stability,
            s=54, marker=markers[item.model_label],
            color=cmap(norm(item.alpha_over_pi)), edgecolor="white", linewidth=0.55,
            label=label, zorder=3,
        )
    axis.set(
        xlabel=r"Handedness fixedness $D_\chi$",
        ylabel=r"Handedness-gated stability $\widetilde{\mathcal{S}}_{\chi}$",
        xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
        title="Fixed-handedness gate versus final stability",
    )
    axis.grid(True, color=GRID, lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower right")
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.015)
    colorbar.set_label(r"Phase lag $\alpha/\pi$")
    fig.suptitle("Handedness-Gated Chiral Boundary-Flow Stability", fontsize=14, fontweight="bold")
    return save_figure(fig, "Handedness_Gated_Stability_Alpha_Sweep")


def plot_component_heatmaps(table: pd.DataFrame) -> tuple[Path, Path]:
    order = [
        "Circular - no defect", "Circular - single defect H=3",
        "Square - no defect", "Square - four defects H=1",
        "Square - four defects H=1.5", "Square - four defects H=3",
    ]
    metrics = [
        ("direction_fixedness", r"Handedness fixedness $D_\chi$"),
        ("current_strength", r"Current strength $M_\chi$"),
        ("temporal_uniformity", r"Magnitude continuity $U_t$"),
        ("spatial_coverage", r"Perimeter coverage $C_s$"),
        ("handedness_gated_stability", r"Final stability $\widetilde{\mathcal{S}}_\chi$"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15.2, 4.8), constrained_layout=True, sharey=True)
    image = None
    for axis, (metric, title) in zip(axes, metrics):
        matrix = table.pivot(index="condition", columns="alpha_over_pi", values=metric).reindex(order)
        image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(len(matrix.columns)), ["0", ".2", ".4", ".6", ".8", "1"])
        axis.set_xlabel(r"$\alpha/\pi$")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iat[row, column]
                color = "white" if value < 0.48 else "#17212B"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
        axis.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_yticks(range(len(order)), order)
    fig.colorbar(image, ax=list(axes), fraction=0.018, pad=0.015, label="Metric value: 0 to 1")
    fig.suptitle(
        r"Complete decomposition: $\widetilde{\mathcal{S}}_\chi"
        r"=D_\chi(M_\chi U_t C_s)^{1/3}$",
        fontsize=14, fontweight="bold",
    )
    return save_figure(fig, "Handedness_Gated_Stability_Component_Heatmaps")


def _choose_representatives(table: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    eligible = table[
        table.current_strength.ge(0.10)
        & table.active_blocks.ge(max(3, BLOCK_COUNT // 4))
    ].copy()
    chosen: list[tuple[str, pd.Series]] = []
    used = set()

    def take(label: str, metric: str, largest: bool) -> None:
        candidates = eligible[~eligible.index.isin(used)].sort_values(metric, ascending=not largest)
        if candidates.empty:
            return
        item = candidates.iloc[0]
        used.add(item.name)
        chosen.append((label, item))

    take("Stable reference", "handedness_gated_stability", True)

    switching = eligible[
        eligible.model_label.eq("circle_defect")
        & np.isclose(eligible.defect_height, 3.0)
        & np.isclose(eligible.alpha_over_pi, 0.0)
    ]
    if not switching.empty:
        item = switching.iloc[0]
        used.add(item.name)
        chosen.append(("Handedness switching", item))

    take("Magnitude intermittency", "temporal_uniformity", False)
    take("Spatial localization", "spatial_coverage", False)
    return chosen


def plot_representative_spacetime(
    table: pd.DataFrame, details: dict[tuple[str, float], dict[str, np.ndarray]],
) -> tuple[Path, Path]:
    representatives = _choose_representatives(table)
    fig, axes = plt.subplots(
        2, len(representatives), figsize=(13.2, 5.8), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 0.75]},
    )
    if len(representatives) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    image = None
    for column, (category, item) in enumerate(representatives):
        detail = details[(item.condition, float(item.alpha_over_pi))]
        axis = axes[0, column]
        image = axis.imshow(
            detail["normalized_arc_current"], origin="lower", aspect="auto",
            extent=(0, 1, detail["block_time"][0], detail["block_time"][-1]),
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        axis.set_title(
            f"{category}\n{item.condition},  "
            rf"$\alpha={item.alpha_over_pi:g}\pi$" + "\n" +
            rf"$\widetilde{{\mathcal{{S}}}}_\chi={item.handedness_gated_stability:.2f}$",
            fontsize=9.5,
        )
        axis.set_xlabel(r"Normalized boundary arclength $s/P$")
        if column == 0:
            axis.set_ylabel(r"Terminal-window time $t$")
        axis = axes[1, column]
        axis.plot(detail["block_time"], detail["block_current"], color=NAVY, marker="o", ms=3, lw=1.5)
        axis.axhline(0, color="#9AA2A9", lw=0.8)
        axis.set_ylim(-1.05, 1.05)
        axis.grid(True, color=GRID, lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel(r"Terminal-window time $t$")
        if column == 0:
            axis.set_ylabel(r"Block current $J_b$")
        axis.text(
            0.03, 0.04,
            rf"$D_\chi={item.direction_fixedness:.2f}$  "
            rf"$M_\chi={item.current_strength:.2f}$  "
            rf"$U_t={item.temporal_uniformity:.2f}$  "
            rf"$C_s={item.spatial_coverage:.2f}$",
            transform=axis.transAxes, fontsize=7.6, va="bottom",
        )
    fig.colorbar(image, ax=list(axes[0]), fraction=0.018, pad=0.012,
                 label=r"Normalized local current $j_{bk}$ (CW $<0<$ CCW)")
    fig.suptitle(
        "Block-Resolved Boundary Current and Stability Failure Modes",
        fontsize=14, fontweight="bold",
    )
    return save_figure(fig, "Representative_Handedness_Gated_Stability_Spacetime")


def write_metric_dictionary() -> tuple[Path, Path]:
    """Write machine-readable constants and a complete plain-text derivation."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configuration = {
        "raw_hdf5_fields": {
            "positionX": "Flattened saved positions; reshaped to r[n,i]=(x[n,i],y[n,i]).",
            "phaseTheta": "Flattened saved headings theta[n,i] in radians.",
        },
        "saved_time_step": "Delta_t_s = model.dt * model.shotsnaps",
        "terminal_window_time": bda.ANALYSIS_WINDOW_TIME,
        "block_count_B": BLOCK_COUNT,
        "arc_bin_count_K": ARC_BIN_COUNT,
        "minimum_episode_time": MIN_TRANSPORT_EPISODE_TIME,
        "minimum_episode_directional_persistence": MIN_DIRECTIONAL_PERSISTENCE,
        "minimum_episode_tangentiality": MIN_TANGENTIALITY,
        "radial_bin_count_R": bda.RADIAL_BIN_COUNT,
        "maximum_radial_depth_over_L": bda.MAX_ANALYSIS_DEPTH_FRACTION,
        "direction_probe_depth_over_L": bda.DIRECTION_PROBE_FRACTION,
        "peak_search_depth_over_L": bda.PEAK_SEARCH_DEPTH_FRACTION,
        "edge_floor_fraction": bda.EDGE_FLOOR_FRACTION,
        "edge_floor_consecutive_bins": bda.EDGE_FLOOR_CONSECUTIVE_BINS,
        "alpha_over_pi": list(ALPHAS),
    }
    json_path = OUTPUT / "Metric_Configuration.json"
    json_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    text = dedent(
        f"""
        HANDEDNESS-GATED CHIRAL BOUNDARY-FLOW STABILITY: DISCRETE DEFINITIONS
        ======================================================================

        1. Stored data and indices
        --------------------------
        n = 0,...,T-1 indexes saved frames in the final {bda.ANALYSIS_WINDOW_TIME:g}
        model-time-unit window, so T is the number of selected saved frames;
        i = 1,...,N indexes particles, so N is the particle count.  The HDF5 field
        positionX gives r_i^n = (x_i^n,y_i^n), and phaseTheta gives theta_i^n.
        The saved-frame interval is Delta_t_s = dt * shotsnaps, where dt is the
        model integrator step and shotsnaps is the number of integrator steps
        between two stored frames.

        2. Boundary projection and signed tangential velocity
        -----------------------------------------------------
        p_i^n = argmin_{{p on boundary}} ||r_i^n-p|| is the closest boundary
        point, d_i^n = ||r_i^n-p_i^n|| is wall distance, s_i^n is the CCW
        boundary arclength of p_i^n, and t_hat_i^n is the local CCW unit tangent.
        The boundary is reconstructed from the exact model class and parameters
        encoded by the matched HDF5 filename.
        The stored phase defines u_i^n=(cos(theta_i^n),sin(theta_i^n)), hence
            q_i^n = u_i^n dot t_hat_i^n in [-1,1].
        q_i^n>0 is CCW, q_i^n<0 is CW, and |q_i^n| is tangential alignment.

        3. Adaptive wall-connected layer
        ---------------------------------
        Let L be model boundaryLength and R={bda.RADIAL_BIN_COUNT}.  The scalar
        r=0,...,R is a radial-bin-edge index (distinct from bold particle position).
        Radial edges are
        rho_r=0.45 L r/R.  The discrete radial signed-current profile is
            h_r=(1/(TN)) sum_(n,i) q_i^n 1[rho_r <= d_i^n < rho_(r+1)].
        It is smoothed with the normalized kernel (1,2,3,2,1)/9 using edge
        padding.  sigma_0 is the sign of sum q_i^n for d_i^n<=0.06L; if that
        sum vanishes, the first R/8 smoothed bins are used.  The first maximum
        of sigma_0*h_r within 0.16L is the wall-connected peak.  Moving outward,
        the layer stops immediately before the first {bda.EDGE_FLOOR_CONSECUTIVE_BINS}
        consecutive bins satisfying sigma_0*h_r<=0 or
        sigma_0*h_r<{bda.EDGE_FLOOR_FRACTION:.2f} times the peak.  Its outer edge is d_e.
        The instantaneous boundary mask is e_i^n=1[d_i^n<=d_e].

        4. Accepted transport episodes
        ------------------------------
        For each particle, split e_i^n=1 into maximal contiguous runs R_i,l,
        where l labels runs of that particle and |R_i,l| is a run's frame count.
        A run needs at least ell_min=max(2,ceil({MIN_TRANSPORT_EPISODE_TIME:g}/Delta_t_s))
        saved frames.  For each run define
            P_i,l = |sum_(n in R_i,l) q_i^n| / sum_(n in R_i,l) |q_i^n|,
            A_i,l = mean_(n in R_i,l) |q_i^n|.
        The run is accepted when P_i,l>={MIN_DIRECTIONAL_PERSISTENCE:.2f} and
        A_i,l>={MIN_TANGENTIALITY:.2f}.  z_i^n=1 only for frames in accepted runs;
        otherwise z_i^n=0.  No particle-population fraction multiplies any metric.

        5. Block and arc-bin fluxes
        ---------------------------
        The T saved frames are split in order into B={BLOCK_COUNT} blocks and the
        boundary perimeter P is split into K={ARC_BIN_COUNT} equal arclength bins.
        Here b=0,...,B-1 is a time-block index, k=0,...,K-1 is an arc-bin index,
        and kappa_i^n=floor(K*(s_i^n mod P)/P) is the bin containing s_i^n.
        For block b,
            F_bk = sum_(n in b,i) z_i^n 1[kappa_i^n=k] q_i^n,
            A_bk = sum_(n in b,i) z_i^n 1[kappa_i^n=k] |q_i^n|,
            J_b  = sum_k F_bk / sum_k A_bk,
            j_bk = F_bk/A_bk when A_bk>0, otherwise 0.
        Thus J_b and j_bk lie in [-1,1]; positive is CCW and negative is CW.

        6. Four components and final score
        -----------------------------------
        Let sigma=sign(sum_b J_b), with sigma=0 if the sum vanishes, and let
        epsilon be floating-point machine epsilon.  The positive-part operator
        is [x]_+=max(x,0).

        Handedness fixedness:
            D_chi = |sum_b J_b| / (sum_b |J_b| + epsilon).
        D_chi=1 means one fixed handedness; D_chi=0 means cancellation or no flow.

        Block-current strength:
            M_chi = (1/B) sum_b |J_b|.
        This is normalized current strength, not the fraction of carrier particles.

        Magnitude continuity:
            U_t = (sum_b |J_b|)^2 / (B sum_b J_b^2 + epsilon).
        U_t=1 for equal block magnitudes.  Direction reversals are not hidden here:
        they are penalized separately and linearly by D_chi.

        Cumulative perimeter coverage:
            g_k = sum_b max(sigma F_bk,0),
            C_s = (sum_k g_k)^2 / (K sum_k g_k^2 + epsilon).
        C_s=1 for uniform full-perimeter coherent flux and approaches 1/K when
        the cumulative coherent flux is confined to one bin.

        Handedness-gated stability:
            S_tilde_chi = D_chi * (M_chi U_t C_s)^(1/3).
        D_chi is a linear gate.  A direction-switching flow therefore cannot
        obtain a high score merely because |J_b| is strong and temporally uniform.

        7. Traceability chain
        ---------------------
        (positionX, phaseTheta, dt, shotsnaps, matched model boundary)
        -> (r_i^n,theta_i^n,Delta_t_s,p_i^n,d_i^n,s_i^n,t_hat_i^n)
        -> q_i^n -> d_e and z_i^n -> (F_bk,A_bk,J_b,j_bk)
        -> (D_chi,M_chi,U_t,C_s) -> S_tilde_chi.

        Operator conventions: 1[condition] is an indicator; ||.|| is Euclidean
        norm; |x| is absolute value; |R| is set cardinality; floor rounds down;
        mod is the periodic remainder; sign returns -1, 0, or +1; and all sums
        run only over the explicitly displayed discrete indices.
        """
    ).strip() + "\n"
    text_path = OUTPUT / "Metric_Definitions.txt"
    text_path.write_text(text, encoding="utf-8")
    return text_path, json_path


def main() -> int:
    configure_plotting()
    table, details = compute_all()
    definition_text, configuration_json = write_metric_dictionary()
    outputs = [
        *plot_sweep_and_phase(table),
        *plot_component_heatmaps(table),
        *plot_representative_spacetime(table, details),
        definition_text,
        configuration_json,
    ]
    print(table[
         ["condition", "alpha_over_pi", "direction_fixedness", "current_strength",
          "temporal_uniformity", "spatial_coverage", "handedness_gated_stability"]
    ].to_string(index=False))
    for path in outputs:
        print(f"FIGURE={path}")
    print(f"CSV={OUTPUT / 'Handedness_Gated_Chiral_Stability_Values.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
