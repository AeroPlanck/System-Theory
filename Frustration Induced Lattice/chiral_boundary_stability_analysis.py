"""Analyze the stability of identity-independent chiral boundary transport.

The script only reads exact parameter-matched HDF5 trajectories.  It computes
one corrected stability index,

    S_chi = (G_chi * U_t * C_s) ** (1 / 3),

where C_s is the perimeter participation ratio of the cumulative coherent
flux, so a translating packet or a remote same-chirality relay is not punished
merely for changing location.  G_chi = abs(mean(J_b)) = D_chi * M_chi is the
directed net-current strength; M_chi is the mean magnitude of the normalized
block current, not a carrier-population fraction.  The script writes CSV data
and publication-ready PNG/PDF figures and never edits PRL.tex or HDF5.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boundary_defect_analysis as bda
import two_metric_boundary_transport_report as report


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "Chiral_Boundary_Stability_Corrected"
BLOCK_COUNT = 20
ARC_BIN_COUNT = 64
EPS = np.finfo(float).eps

NAVY = "#18344F"
TEAL = "#157A74"
GOLD = "#C28B2C"
RED = "#A64545"
MUTED = "#5D6873"
GRID = "#DCE2E7"
PALE_GOLD = "#FBF3E3"


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
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
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
    episode_sign = report._transport_episode_field(edge, q, saved_dt)
    valid = episode_sign != 0

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
    direction = 1.0 if direction_sum >= 0 else -1.0
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
    directed_current = float(abs(block_current.mean()))
    stability = float(
        np.cbrt(max(0.0, directed_current * temporal_uniformity * spatial_coverage))
    )
    normalized_arc_current = np.divide(
        signed_flux,
        absolute_flux,
        out=np.zeros_like(signed_flux),
        where=absolute_flux > EPS,
    )

    summary = {
        "alpha_over_pi": float(model.phaseLagA0 / np.pi),
        "defect_height": float(getattr(model, "protrusionHeight", 0.0)),
        "direction": "CCW" if direction > 0 else "CW",
        "direction_fixedness": direction_fixedness,
        "temporal_uniformity": temporal_uniformity,
        "spatial_coverage": spatial_coverage,
        "current_strength": current_strength,
        "directed_current": directed_current,
        "chiral_flow_stability": stability,
        "mean_abs_block_current": current_strength,
        "active_blocks": int(np.count_nonzero(magnitude > EPS)),
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
    for group, label, height, condition in report.non_pi_specs():
        for alpha in report.ALPHAS:
            model = report.build_model(label, alpha, height)
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
    transport_path = report.DATA_OUT / "two_metric_values.csv"
    if transport_path.is_file():
        transport = pd.read_csv(transport_path)[
            ["geometry_group", "condition", "defect_height", "alpha_over_pi", "long_time_chiral_transport"]
        ]
        table = table.merge(
            transport,
            on=["geometry_group", "condition", "defect_height", "alpha_over_pi"],
            how="left",
            validate="one_to_one",
        )
    else:
        table["long_time_chiral_transport"] = np.nan
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "Chiral_Boundary_Stability_Values.csv", index=False)
    return table, details


def _condition_series(group: str):
    if group == "symmetric_square":
        return [
            (0.0, "无缺陷", NAVY),
            (1.0, "H=1", "#2E6F89"),
            (1.5, "H=1.5", TEAL),
            (3.0, "H=3", "#8FBF26"),
        ]
    return [(0.0, "无缺陷", NAVY), (3.0, "H=3", GOLD)]


def plot_sweep_and_phase(table: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplot_mosaic(
        [["square", "circle"], ["phase", "phase"]],
        figsize=(10.8, 8.0), constrained_layout=True,
        height_ratios=[1.0, 1.15],
    )
    for key, group, title in (
        ("square", "symmetric_square", "方形与四个对称缺陷"),
        ("circle", "asymmetric_circle", "圆形与单个非对称缺陷"),
    ):
        axis = axes[key]
        subset = table[table.geometry_group == group]
        for height, label, color in _condition_series(group):
            part = subset[np.isclose(subset.defect_height, height)].sort_values("alpha_over_pi")
            axis.plot(
                part.alpha_over_pi,
                part.chiral_flow_stability,
                marker="o", ms=5, lw=2, color=color, label=label,
            )
        axis.axvspan(0.5, 0.9, color=PALE_GOLD, alpha=0.65, zorder=-5)
        axis.axhline(0.8, color=RED, lw=0.9, ls=":")
        axis.set(
            title=title,
            xlabel=r"相位滞后 $\alpha$",
            ylabel=r"手性边界流稳定度 $\mathcal{S}_{\chi}$",
            xlim=(-0.02, 0.82), ylim=(-0.03, 1.03),
        )
        axis.set_xticks(np.arange(0, 0.81, 0.2), ["0", r"$0.2\pi$", r"$0.4\pi$", r"$0.6\pi$", r"$0.8\pi$"])
        axis.grid(True, color=GRID, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes["phase"]
    markers = {
        "circle": "o", "circle_defect": "D", "square": "s", "square_defect": "^",
    }
    labels_used = set()
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.8)
    cmap = mpl.colormaps["viridis"]
    for _, item in table.iterrows():
        label = item.condition if item.condition not in labels_used else None
        labels_used.add(item.condition)
        axis.scatter(
            abs(item.long_time_chiral_transport), item.chiral_flow_stability,
            s=54, marker=markers[item.model_label],
            color=cmap(norm(item.alpha_over_pi)), edgecolor="white", linewidth=0.55,
            label=label, zorder=3,
        )
    axis.axvline(0.64, color=RED, lw=0.9, ls=":")
    axis.axhline(0.8, color=RED, lw=0.9, ls=":")
    axis.set(
        xlabel=r"长时单手性输运 $|\mathcal{T}_{\rm LT}|$",
        ylabel=r"手性边界流稳定度 $\mathcal{S}_{\chi}$",
        xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
        title="输运存在性—稳定性相图",
    )
    axis.grid(True, color=GRID, lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower right")
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.015)
    colorbar.set_label(r"$\alpha/\pi$")
    fig.suptitle("手性边界流稳定性随相位滞后与边界缺陷的变化", fontsize=14, fontweight="bold")
    return save_figure(fig, "Stability_Sweep_And_Phase")


def plot_component_heatmaps(table: pd.DataFrame) -> tuple[Path, Path]:
    order = [
        "圆形无缺陷", "圆形单缺陷 H=3", "方形无缺陷",
        "方形四缺陷 H=1", "方形四缺陷 H=1.5", "方形四缺陷 H=3",
    ]
    metrics = [
        ("directed_current", r"定向净流强度 $G_\chi$"),
        ("temporal_uniformity", r"时间连续性 $U_t$"),
        ("spatial_coverage", r"全边界覆盖 $C_s$"),
        ("chiral_flow_stability", r"综合稳定度 $\mathcal{S}_\chi$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.6), constrained_layout=True, sharey=True)
    image = None
    for axis, (metric, title) in zip(axes, metrics):
        matrix = table.pivot(index="condition", columns="alpha_over_pi", values=metric).reindex(order)
        image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(len(matrix.columns)), ["0", ".2", ".4", ".6", ".8"])
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
    fig.colorbar(image, ax=list(axes), fraction=0.018, pad=0.015, label="0（不稳定）—1（稳定）")
    fig.suptitle("修正稳定度的三因素分解：定向净流、时间与全边界覆盖", fontsize=14, fontweight="bold")
    return save_figure(fig, "Stability_Component_Heatmaps")


def _choose_representatives(table: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    eligible = table[
        table.long_time_chiral_transport.abs().ge(0.30)
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

    take("稳定参照", "chiral_flow_stability", True)
    take("净流衰减型", "directed_current", False)
    take("时间间歇型", "temporal_uniformity", False)
    take("空间局域型", "spatial_coverage", False)
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
            f"{category}\n{item.condition},  α={item.alpha_over_pi:g}π\n"
            rf"$\mathcal{{S}}_\chi={item.chiral_flow_stability:.2f}$",
            fontsize=9.5,
        )
        axis.set_xlabel(r"归一化边界弧长 $s/P$")
        if column == 0:
            axis.set_ylabel("末端窗口时间")
        axis = axes[1, column]
        axis.plot(detail["block_time"], detail["block_current"], color=NAVY, marker="o", ms=3, lw=1.5)
        axis.axhline(0, color="#9AA2A9", lw=0.8)
        axis.set_ylim(-1.05, 1.05)
        axis.grid(True, color=GRID, lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("末端窗口时间")
        if column == 0:
            axis.set_ylabel(r"分块手性流 $J_b$")
        axis.text(
            0.03, 0.04,
            rf"$G_\chi={item.directed_current:.2f}$  $U_t={item.temporal_uniformity:.2f}$  "
            rf"$C_s={item.spatial_coverage:.2f}$",
            transform=axis.transAxes, fontsize=7.6, va="bottom",
        )
    fig.colorbar(image, ax=list(axes[0]), fraction=0.018, pad=0.012,
                 label="局部归一化切向流（CW ← 0 → CCW）")
    fig.suptitle("代表性状态的边界流时空结构与稳定性失效方式", fontsize=14, fontweight="bold")
    return save_figure(fig, "Representative_Stability_Spacetime")


def main() -> int:
    configure_plotting()
    table, details = compute_all()
    outputs = [
        *plot_sweep_and_phase(table),
        *plot_component_heatmaps(table),
        *plot_representative_spacetime(table, details),
    ]
    print(table[
         ["condition", "alpha_over_pi", "direction_fixedness", "temporal_uniformity",
         "spatial_coverage", "current_strength", "directed_current",
         "chiral_flow_stability",
         "long_time_chiral_transport"]
    ].to_string(index=False))
    for path in outputs:
        print(f"FIGURE={path}")
    print(f"CSV={OUTPUT / 'Chiral_Boundary_Stability_Values.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
