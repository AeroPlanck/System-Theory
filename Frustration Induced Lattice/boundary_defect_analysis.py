"""Quantify and visualize boundary transport from existing HDF5 trajectories.

This file is deliberately self-contained.  It never runs/continues a model,
never writes HDF5, and never touches PRL.tex.  All user-facing parameters live
in the configuration block immediately below.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

import main as model_library
from main import phaseCmap, phaseNorm


PROJECT_DIR = Path(__file__).resolve().parent


# =============================================================================
# USER CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output" / "Boundary_Defect_Analysis"
LIGHT_RERENDER_DIR = (
    PROJECT_DIR / "output" / "Pdf" / "Light_Boundary_Rerendered"
)

# Existing-data contract.  phaseLagA0 is the only swept parameter.
ALPHA_OVER_PI = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# Terminal-state comparisons additionally show the independently classified
# pattern-formation threshold alpha = pi/2.  Keep this grid separate so that a
# morphology-only panel cannot silently change any metric sweep.
TERMINAL_COMPARISON_ALPHA_OVER_PI = np.array(
    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
)
COMMON_PARAMETERS = {
    "strengthK": 20.75,
    "distanceD0": 1.0,
    "boundaryLength": 7.0,
    "speedV": 3.0,
    "freqDist": "uniform",
    "initPhaseTheta": None,
    "omegaMin": 0.0,
    "deltaOmega": 0.0,
    "agentsNum": 2000,
    "dt": 0.005,
    "shotsnaps": 10,
    "randomSeed": 9,
}
FOUR_SPIKE_HEIGHTS = (1.0, 1.5, 3.0)
SPIKE_HALF_WIDTH = 0.25
ASYMMETRIC_SPIKE_HEIGHT = 3.0

# A terminal time window long enough for a free particle to traverse roughly
# one square perimeter (P/v = 28/3 ~= 9.33).  Only the final part is read.
ANALYSIS_WINDOW_TIME = 10.0

# Adaptive wall-connected layer.  The signed radial current is smoothed, then
# followed outwards from its first near-wall maximum.  The layer stops at the
# first sign reversal or sustained fall below EDGE_FLOOR_FRACTION of that peak.
RADIAL_BIN_COUNT = 48
MAX_ANALYSIS_DEPTH_FRACTION = 0.45
DIRECTION_PROBE_FRACTION = 0.06
PEAK_SEARCH_DEPTH_FRACTION = 0.16
EDGE_FLOOR_FRACTION = 0.12
EDGE_FLOOR_CONSECUTIVE_BINS = 3
PENETRATION_CURRENT_QUANTILE = 0.90

# Persistent carrier definition.  All three conditions must hold over the
# terminal time window.  These thresholds are exported with the results.
MIN_BOUNDARY_RESIDENCE = 0.60
MIN_DIRECTIONAL_PERSISTENCE = 0.75
MIN_MEAN_TANGENTIALITY = 0.35
ARC_BIN_COUNT = 72
TEMPORAL_BLOCK_COUNT = 10

# Publication-style rendering.  A neutral light boundary does not compete with
# the cyclic phase colormap.  Existing PDFs are preserved; rerenders are new.
BOUNDARY_COLOR = "#B9BEC7"
BOUNDARY_ALPHA = 0.72
BOUNDARY_LINEWIDTH = 0.72
QUIVER_SCALE = 15.0
QUIVER_WIDTH = 0.0042
FIGURE_DPI = 400
PLOT_COLUMNS = 3

# Historical frame choices that were explicitly selected earlier.  All other
# panels use their file's true terminal frame.  Iterations map to exact saved
# frames because shotsnaps=10 for this data family.
MIDPOINT_SELECTED_ITERATIONS = {0.4: 48_500, 0.6: 78_500}
SQUARE_RERENDER_ITERATIONS = (48_000, 49_000, 49_500, 50_000)
FOUR_SPIKE_RERENDER_SPECS = (
    (1.0, 50_000),
    (1.0, 80_000),
    (1.5, 50_000),
    (3.0, 50_000),
)

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


EPS = np.finfo(float).eps


class DataContractError(RuntimeError):
    """Raised before plotting when an exact HDF5 match is absent or invalid."""


@dataclass(frozen=True)
class BoundaryGeometry:
    kind: str
    perimeter: float
    length_scale: float
    vertices: np.ndarray | None = None
    center: np.ndarray | None = None
    radius: float | None = None
    midpoint_half_angle: float | None = None
    spike_height: float | None = None


@dataclass
class WindowData:
    positions: np.ndarray
    phases: np.ndarray
    total_frames: int
    frame_indices: np.ndarray


def _model_parameters(**extra: float) -> dict:
    return {**COMMON_PARAMETERS, **extra, "savePath": str(DATA_DIR), "overWrite": False}


def build_model(model_class: type, alpha_over_pi: float, **extra: float):
    return model_class(
        phaseLagA0=float(alpha_over_pi * np.pi),
        **_model_parameters(**extra),
    )


def data_path(model) -> Path:
    return Path(model.savePath) / f"{model}.h5"


def validate_exact_files(models: Sequence) -> None:
    missing = [data_path(model) for model in models if not data_path(model).is_file()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise DataContractError(
            "Exact parameter-matched HDF5 file(s) are absent.  No output was "
            f"created for this group:\n{listing}"
        )


def _hdf_layout(model) -> tuple[int, int]:
    path = data_path(model)
    try:
        with pd.HDFStore(path, mode="r") as store:
            required = {"/positionX", "/phaseTheta"}
            if not required.issubset(store.keys()):
                raise DataContractError(f"{path} lacks positionX or phaseTheta.")
            position = store.get_storer("positionX")
            phase = store.get_storer("phaseTheta")
            n = model.agentsNum
            if position.ncols != 2 or phase.ncols != 1:
                raise DataContractError(f"Unexpected column schema in {path}.")
            if position.nrows != phase.nrows or position.nrows % n:
                raise DataContractError(f"Incomplete or unaligned frames in {path}.")
            return position.nrows // n, n
    except DataContractError:
        raise
    except Exception as exc:
        raise DataContractError(f"Cannot inspect {path}: {exc}") from exc


def load_window(model, duration: float = ANALYSIS_WINDOW_TIME) -> WindowData:
    total_frames, n = _hdf_layout(model)
    saved_dt = model.dt * model.shotsnaps
    requested = max(2, int(math.ceil(duration / saved_dt)) + 1)
    count = min(total_frames, requested)
    start_frame = total_frames - count
    start_row = start_frame * n
    path = data_path(model)
    with pd.HDFStore(path, mode="r") as store:
        positions = store.select("positionX", start=start_row).to_numpy()
        phases = store.select("phaseTheta", start=start_row).to_numpy()
    positions = positions.reshape(count, n, 2)
    phases = phases.reshape(count, n)
    if not np.isfinite(positions).all() or not np.isfinite(phases).all():
        raise DataContractError(f"Non-finite values in terminal window of {path}.")
    return WindowData(
        positions=positions,
        phases=phases,
        total_frames=total_frames,
        frame_indices=np.arange(start_frame, total_frames),
    )


def load_frame(model, iteration: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    total_frames, n = _hdf_layout(model)
    if iteration is None:
        frame = total_frames - 1
    else:
        if iteration < 0 or iteration % model.shotsnaps:
            raise DataContractError(
                f"Iteration {iteration} is not an exact saved frame for snap={model.shotsnaps}."
            )
        frame = iteration // model.shotsnaps
        if frame >= total_frames:
            raise DataContractError(
                f"{data_path(model).name} has no saved frame at iteration {iteration}."
            )
    with pd.HDFStore(data_path(model), mode="r") as store:
        positions = store.select("positionX", start=frame * n, stop=(frame + 1) * n).to_numpy()
        phases = store.select("phaseTheta", start=frame * n, stop=(frame + 1) * n).to_numpy()
    return positions, phases.reshape(-1), frame


def geometry_for(model) -> BoundaryGeometry:
    if isinstance(model, model_library.CollisionBoundaryMidpointSpikePatternFormation):
        radius = float(model.circleRadius)
        half_angle = float(np.arcsin(model.protrusionHalfWidth / radius))
        arc_length = radius * (2 * np.pi - 2 * half_angle)
        side = float(np.linalg.norm(model.spikeTip - model.spikeBaseLeft))
        return BoundaryGeometry(
            kind="midpoint_circle",
            perimeter=arc_length + 2 * side,
            length_scale=float(model.boundaryLength),
            center=np.asarray(model.circleCenter),
            radius=radius,
            midpoint_half_angle=half_angle,
            spike_height=float(model.protrusionHeight),
        )
    if isinstance(model, model_library.CircularBoundaryPatternFormation):
        radius = float(model.circleRadius)
        return BoundaryGeometry(
            kind="circle",
            perimeter=2 * np.pi * radius,
            length_scale=float(model.boundaryLength),
            center=np.asarray(model.circleCenter),
            radius=radius,
        )
    if isinstance(model, model_library.CollisionBoundaryFourSpikePatternFormation):
        vertices = np.asarray(model.boundaryVertices, dtype=float)
    elif isinstance(model, model_library.CollisionBoundaryPatternFormation):
        length = float(model.boundaryLength)
        vertices = np.array([[0, 0], [length, 0], [length, length], [0, length]], dtype=float)
    else:
        raise TypeError(f"Unsupported boundary geometry: {model.__class__.__name__}")
    edges = np.roll(vertices, -1, axis=0) - vertices
    return BoundaryGeometry(
        kind="polygon",
        perimeter=float(np.linalg.norm(edges, axis=1).sum()),
        length_scale=float(model.boundaryLength),
        vertices=vertices,
    )


def _project_segments(points: np.ndarray, vertices: np.ndarray):
    flat = points.reshape(-1, 2)
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
    best_d2 = np.full(flat.shape[0], np.inf)
    best_s = np.zeros(flat.shape[0])
    best_tangent = np.zeros((flat.shape[0], 2))
    for start, edge, length, s0 in zip(vertices, edges, lengths, cumulative):
        tangent = edge / length
        fraction = np.clip(((flat - start) @ edge) / (length * length), 0.0, 1.0)
        projection = start + fraction[:, None] * edge
        d2 = np.einsum("ij,ij->i", flat - projection, flat - projection)
        take = d2 < best_d2
        best_d2[take] = d2[take]
        best_s[take] = s0 + fraction[take] * length
        best_tangent[take] = tangent
    shape = points.shape[:-1]
    return (
        np.sqrt(best_d2).reshape(shape),
        best_s.reshape(shape),
        best_tangent.reshape(*shape, 2),
    )


def _project_circle(points: np.ndarray, geometry: BoundaryGeometry):
    relative = points - geometry.center
    radius_now = np.linalg.norm(relative, axis=-1)
    angle = np.mod(np.arctan2(relative[..., 1], relative[..., 0]), 2 * np.pi)
    tangent = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)
    return np.abs(geometry.radius - radius_now), geometry.radius * angle, tangent


def _segment_candidate(flat: np.ndarray, start: np.ndarray, end: np.ndarray, s0: float):
    edge = end - start
    length = float(np.linalg.norm(edge))
    tangent = edge / length
    fraction = np.clip(((flat - start) @ edge) / (length * length), 0.0, 1.0)
    projection = start + fraction[:, None] * edge
    d2 = np.einsum("ij,ij->i", flat - projection, flat - projection)
    return d2, s0 + fraction * length, np.broadcast_to(tangent, flat.shape)


def _project_midpoint_circle(points: np.ndarray, geometry: BoundaryGeometry):
    flat = points.reshape(-1, 2)
    center = geometry.center
    radius = geometry.radius
    half_angle = geometry.midpoint_half_angle
    start_angle = -0.5 * np.pi + half_angle
    end_angle = 1.5 * np.pi - half_angle
    relative = flat - center
    raw_angle = np.arctan2(relative[:, 1], relative[:, 0])
    raw_angle = np.where(raw_angle < start_angle, raw_angle + 2 * np.pi, raw_angle)
    arc_angle = np.clip(raw_angle, start_angle, end_angle)
    arc_projection = center + radius * np.column_stack([np.cos(arc_angle), np.sin(arc_angle)])
    best_d2 = np.einsum("ij,ij->i", flat - arc_projection, flat - arc_projection)
    best_s = radius * (arc_angle - start_angle)
    best_tangent = np.column_stack([-np.sin(arc_angle), np.cos(arc_angle)])

    base_right = center + radius * np.array([np.cos(start_angle), np.sin(start_angle)])
    base_left = center + radius * np.array([np.cos(end_angle), np.sin(end_angle)])
    tip = np.array([center[0], center[1] - radius + geometry.spike_height])
    arc_length = radius * (end_angle - start_angle)
    left_length = float(np.linalg.norm(tip - base_left))
    candidates = (
        _segment_candidate(flat, base_left, tip, arc_length),
        _segment_candidate(flat, tip, base_right, arc_length + left_length),
    )
    for d2, s, tangent in candidates:
        take = d2 < best_d2
        best_d2[take] = d2[take]
        best_s[take] = s[take]
        best_tangent[take] = tangent[take]
    shape = points.shape[:-1]
    return (
        np.sqrt(best_d2).reshape(shape),
        best_s.reshape(shape),
        best_tangent.reshape(*shape, 2),
    )


def project_boundary(points: np.ndarray, geometry: BoundaryGeometry):
    if geometry.kind == "circle":
        return _project_circle(points, geometry)
    if geometry.kind == "midpoint_circle":
        return _project_midpoint_circle(points, geometry)
    return _project_segments(points, geometry.vertices)


def _smooth_profile(values: np.ndarray) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    padded = np.pad(values, (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def adaptive_edge_width(distance: np.ndarray, tangential: np.ndarray, length_scale: float):
    max_depth = MAX_ANALYSIS_DEPTH_FRACTION * length_scale
    edges = np.linspace(0.0, max_depth, RADIAL_BIN_COUNT + 1)
    bin_width = edges[1] - edges[0]
    in_range = distance < max_depth
    current, _ = np.histogram(distance[in_range], bins=edges, weights=tangential[in_range])
    current = current / distance.size
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth_profile(current)

    probe = distance <= DIRECTION_PROBE_FRACTION * length_scale
    signed_probe = float(tangential[probe].sum())
    if abs(signed_probe) <= EPS:
        signed_probe = float(smoothed[: max(2, RADIAL_BIN_COUNT // 8)].sum())
    direction = 1.0 if signed_probe >= 0 else -1.0
    coherent = direction * smoothed
    search_bins = max(2, int(np.ceil(PEAK_SEARCH_DEPTH_FRACTION * length_scale / bin_width)))
    peak_idx = int(np.argmax(coherent[:search_bins]))
    peak_value = max(float(coherent[peak_idx]), EPS)

    cutoff_idx = RADIAL_BIN_COUNT - 1
    low_run = 0
    for idx in range(peak_idx + 1, RADIAL_BIN_COUNT):
        is_low = coherent[idx] <= 0 or coherent[idx] < EDGE_FLOOR_FRACTION * peak_value
        low_run = low_run + 1 if is_low else 0
        if low_run >= EDGE_FLOOR_CONSECUTIVE_BINS:
            # Exclude the complete sustained low/reversed run: the retained
            # layer is precisely the first current lobe connected to the wall.
            cutoff_idx = max(peak_idx, idx - low_run)
            break
    width = float(edges[min(cutoff_idx + 1, RADIAL_BIN_COUNT)])
    return width, direction, centers, current, smoothed


def _participation_coverage(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    if positive.size == 0:
        return 0.0
    return float(positive.sum() ** 2 / (weights.size * np.square(positive).sum()))


def compute_metrics(model) -> tuple[dict, dict]:
    window = load_window(model)
    geometry = geometry_for(model)
    distance, arc, tangent = project_boundary(window.positions, geometry)
    heading = np.stack([np.cos(window.phases), np.sin(window.phases)], axis=-1)
    tangential = np.einsum("...j,...j->...", heading, tangent)
    width, direction, radial_centers, radial_current, radial_smooth = adaptive_edge_width(
        distance, tangential, geometry.length_scale
    )
    edge = distance <= width

    preliminary_signed_sum = float(np.sum(tangential[edge]))
    preliminary_absolute_sum = float(np.sum(np.abs(tangential[edge])))
    preliminary_current = float(np.mean(tangential * edge))
    preliminary_purity = (
        preliminary_signed_sum / preliminary_absolute_sum
        if preliminary_absolute_sum > EPS else 0.0
    )

    residence = edge.mean(axis=0)
    signed_particle = np.sum(tangential * edge, axis=0)
    absolute_particle = np.sum(np.abs(tangential) * edge, axis=0)
    persistence = np.divide(
        np.abs(signed_particle), absolute_particle,
        out=np.zeros_like(signed_particle), where=absolute_particle > EPS,
    )
    tangentiality = np.divide(
        absolute_particle, edge.sum(axis=0),
        out=np.zeros_like(absolute_particle), where=edge.sum(axis=0) > 0,
    )
    carriers = (
        (residence >= MIN_BOUNDARY_RESIDENCE)
        & (persistence >= MIN_DIRECTIONAL_PERSISTENCE)
        & (tangentiality >= MIN_MEAN_TANGENTIALITY)
    )

    carrier_edge = edge & carriers[None, :]
    signed_sum = float(np.sum(tangential[carrier_edge]))
    absolute_sum = float(np.sum(np.abs(tangential[carrier_edge])))
    current = float(np.mean(tangential * carrier_edge))
    purity = signed_sum / absolute_sum if absolute_sum > EPS else 0.0
    carrier_direction = 0.0 if abs(signed_sum) <= EPS else float(np.sign(signed_sum))

    max_depth = MAX_ANALYSIS_DEPTH_FRACTION * geometry.length_scale
    radial_edges = np.linspace(0.0, max_depth, RADIAL_BIN_COUNT + 1)
    carrier_radial_current, _ = np.histogram(
        distance[carrier_edge],
        bins=radial_edges,
        weights=tangential[carrier_edge],
    )
    carrier_radial_current = carrier_radial_current / distance.size
    coherent_bin = np.maximum(carrier_direction * carrier_radial_current, 0.0)
    active = radial_edges[1:] <= width + 1e-12
    coherent_active = coherent_bin * active
    if coherent_active.sum() > EPS:
        cumulative = np.cumsum(coherent_active) / coherent_active.sum()
        q_idx = int(np.searchsorted(cumulative, PENETRATION_CURRENT_QUANTILE))
        penetration = float(radial_edges[min(q_idx + 1, RADIAL_BIN_COUNT)])
    else:
        penetration = 0.0

    arc_idx = np.floor(np.mod(arc, geometry.perimeter) / geometry.perimeter * ARC_BIN_COUNT).astype(int)
    arc_idx = np.clip(arc_idx, 0, ARC_BIN_COUNT - 1)
    arc_weights = np.bincount(
        arc_idx[carrier_edge],
        weights=carrier_direction * tangential[carrier_edge],
        minlength=ARC_BIN_COUNT,
    )
    arc_weights = np.maximum(arc_weights, 0.0)
    coverage = _participation_coverage(arc_weights)

    frame_groups = np.array_split(np.arange(window.positions.shape[0]), TEMPORAL_BLOCK_COUNT)
    block_current = np.array([
        float(np.mean(tangential[group] * carrier_edge[group]))
        for group in frame_groups if group.size
    ])
    mean_block = float(np.mean(block_current))
    std_block = float(np.std(block_current, ddof=1)) if block_current.size > 1 else 0.0
    stability = abs(mean_block) / (abs(mean_block) + std_block + EPS)

    alpha_over_pi = float(model.phaseLagA0 / np.pi)
    defect_height = float(getattr(model, "protrusionHeight", 0.0))
    row = {
        "geometry_group": (
            "asymmetric_circle" if geometry.kind in {"circle", "midpoint_circle"}
            else "symmetric_square"
        ),
        "model_class": model.__class__.__name__,
        "alpha_over_pi": alpha_over_pi,
        "defect_height": defect_height,
        "defect_height_over_L": defect_height / geometry.length_scale,
        "terminal_saved_frame": int(window.total_frames - 1),
        "window_first_saved_frame": int(window.frame_indices[0]),
        "window_frames": int(window.positions.shape[0]),
        "edge_direction": (
            "none" if carrier_direction == 0 else ("CCW" if carrier_direction > 0 else "CW")
        ),
        "signed_edge_current": current,
        "edge_current_magnitude": abs(current),
        "signed_chirality_purity": purity,
        "chirality_purity_magnitude": abs(purity),
        "preliminary_layer_signed_current": preliminary_current,
        "preliminary_layer_signed_purity": preliminary_purity,
        "persistent_carrier_fraction": float(carriers.mean()),
        "persistent_carrier_count": int(carriers.sum()),
        "adaptive_edge_width": width,
        "penetration_width_90": penetration,
        "penetration_width_90_over_L": penetration / geometry.length_scale,
        "effective_boundary_coverage": coverage,
        "temporal_stability": stability,
        "block_current_std": std_block,
        "hdf5_file": data_path(model).name,
    }
    detail = {
        "radial_distance": radial_centers,
        "radial_current": radial_current,
        "radial_current_smoothed": radial_smooth,
        "direction": direction,
        "edge_width": width,
        "residence": residence,
        "persistence": persistence,
        "tangentiality": tangentiality,
        "carriers": carriers,
        "arc_weights": arc_weights,
        "block_current": block_current,
    }
    return row, detail


def _alpha_label(value: float) -> str:
    if np.isclose(value, 0):
        return r"$0$"
    if np.isclose(value, 1):
        return r"$\pi$"
    return rf"${value:g}\pi$"


def _subplot_label(index: int) -> str:
    return chr(ord("a") + index)


def draw_boundary(ax: plt.Axes, model) -> None:
    geometry = geometry_for(model)
    style = dict(
        color=BOUNDARY_COLOR,
        alpha=BOUNDARY_ALPHA,
        linewidth=BOUNDARY_LINEWIDTH,
        zorder=4,
    )
    if geometry.kind == "circle":
        ax.add_patch(plt.Circle(geometry.center, geometry.radius, fill=False, **style))
    elif geometry.kind == "midpoint_circle":
        boundary = np.vstack([model.boundaryVertices, model.boundaryVertices[0]])
        ax.plot(
            boundary[:, 0], boundary[:, 1],
            solid_capstyle="round", solid_joinstyle="round", **style,
        )
    else:
        boundary = np.vstack([geometry.vertices, geometry.vertices[0]])
        ax.plot(
            boundary[:, 0], boundary[:, 1],
            solid_capstyle="round", solid_joinstyle="round", **style,
        )


def draw_state(ax: plt.Axes, model, iteration: int | None = None) -> int:
    positions, phases, frame = load_frame(model, iteration)
    ax.quiver(
        positions[:, 0], positions[:, 1],
        np.cos(phases), np.sin(phases), phases,
        cmap=phaseCmap, norm=phaseNorm,
        scale_units="inches", scale=QUIVER_SCALE, width=QUIVER_WIDTH,
        pivot="middle", zorder=2,
    )
    draw_boundary(ax, model)
    pad = 0.018 * model.boundaryLength
    ax.set_xlim(-pad, model.boundaryLength + pad)
    ax.set_ylim(-pad, model.boundaryLength + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return frame


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def state_sweep_figure(
    models: Sequence,
    output: Path,
    iterations: Sequence[int | None] | None = None,
    row_labels: Sequence[str] | None = None,
    save_png: bool = False,
    figure_title: str | None = None,
) -> None:
    validate_exact_files(models)
    if iterations is None:
        iterations = [None] * len(models)
    if row_labels is None:
        columns = min(PLOT_COLUMNS, len(models))
    else:
        if not row_labels or len(models) % len(row_labels):
            raise ValueError(
                "A labeled state sweep requires an equal number of panels in every row."
            )
        columns = len(models) // len(row_labels)
    rows = math.ceil(len(models) / columns)
    with mpl.rc_context(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            rows, columns,
            figsize=(3.0 * columns + 0.55, 3.05 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        flat = axes.ravel()
        for idx, (ax, model, iteration) in enumerate(zip(flat, models, iterations)):
            draw_state(ax, model, iteration)
            alpha = model.phaseLagA0 / np.pi
            ax.set_title(rf"$\alpha={_alpha_label(alpha)[1:-1]}$", fontsize=14, pad=3)
            ax.text(
                0.018, 0.982, f"({_subplot_label(idx)})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=13, fontweight="bold", color="#262626", zorder=6,
            )
        for ax in flat[len(models):]:
            ax.set_visible(False)
        if row_labels:
            for row, label in enumerate(row_labels):
                axes[row, 0].text(
                    -0.08, 0.5, label, transform=axes[row, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=12,
                )
        mappable = ScalarMappable(norm=phaseNorm, cmap=phaseCmap)
        cbar = fig.colorbar(
            mappable, ax=list(flat[:len(models)]),
            ticks=[0, np.pi, 2 * np.pi], fraction=0.022, pad=0.012,
        )
        cbar.ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        cbar.set_label(r"Phase $\theta$", fontsize=13)
        if figure_title:
            fig.suptitle(figure_title, fontsize=16, fontweight="bold")
        if save_png:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output.with_suffix(".png"),
                dpi=FIGURE_DPI,
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"Saved {output.with_suffix('.png')}")
        save_figure(fig, output)


METRIC_SPECS = (
    ("signed_edge_current", r"Signed edge current $J_{\rm e}$", (-1.03, 1.03)),
    ("signed_chirality_purity", r"Directional purity $\chi_{\rm e}$", (-1.03, 1.03)),
    ("persistent_carrier_fraction", r"Carrier fraction $f_{\rm c}$", (0, 1.03)),
    ("penetration_width_90_over_L", r"Penetration $\xi_{90}/L$", (0, None)),
    ("effective_boundary_coverage", r"Effective coverage $F_{\rm cov}$", (0, 1.03)),
    ("temporal_stability", r"Temporal stability $S_J$", (0, 1.03)),
)

HEATMAP_METRIC_SPECS = (
    ("edge_current_magnitude", r"Edge current $|J_{\rm e}|$"),
    ("chirality_purity_magnitude", r"Directional purity $|\chi_{\rm e}|$"),
    ("persistent_carrier_fraction", r"Carrier fraction $f_{\rm c}$"),
    ("penetration_width_90_over_L", r"Penetration $\xi_{90}/L$"),
    ("effective_boundary_coverage", r"Effective coverage $F_{\rm cov}$"),
    ("temporal_stability", r"Temporal stability $S_J$"),
)


def plot_metric_curves(table: pd.DataFrame, group: str, output: Path) -> None:
    subset = table[table.geometry_group == group].copy()
    heights = sorted(subset.defect_height.unique())
    colors = mpl.colormaps["viridis"](np.linspace(0.12, 0.86, len(heights)))
    labels = {
        0.0: "No defect",
        1.0: r"$H=1.0$",
        1.5: r"$H=1.5$",
        3.0: r"$H=3.0$",
    }
    with mpl.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.35), constrained_layout=True)
        for panel, (ax, (column, ylabel, ylim)) in enumerate(zip(axes.ravel(), METRIC_SPECS)):
            for color, height in zip(colors, heights):
                data = subset[np.isclose(subset.defect_height, height)].sort_values("alpha_over_pi")
                ax.plot(
                    data.alpha_over_pi, data[column], "o-",
                    color=color, lw=1.7, ms=4.6,
                    markeredgecolor="white", markeredgewidth=0.55,
                    label=labels.get(height, rf"$H={height:g}$"),
                )
            ax.axvline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
            if ylim[0] < 0:
                ax.axhline(0, color="#8D8D8D", lw=0.75, zorder=0)
            ax.set_xlim(-0.02, 1.02)
            if ylim[0] < 0:
                values = subset[column].to_numpy(dtype=float)
                bound = min(1.03, max(0.08, 1.12 * float(np.nanmax(np.abs(values)))))
                ax.set_ylim(-bound, bound)
            elif ylim[1] is None:
                ax.set_ylim(bottom=ylim[0])
            else:
                ax.set_ylim(*ylim)
            ax.set_xticks(ALPHA_OVER_PI, [_alpha_label(v) for v in ALPHA_OVER_PI])
            ax.set_xlabel(r"Phase lag $\alpha$")
            ax.set_ylabel(ylabel)
            ax.text(0.02, 0.97, f"({_subplot_label(panel)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=12, fontweight="bold")
            ax.grid(True, color="#E3E5E8", lw=0.55)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False, ncol=2, fontsize=10, loc="best")
        save_figure(fig, output)


def plot_defect_response_heatmaps(table: pd.DataFrame, output: Path) -> None:
    subset = table[(table.geometry_group == "symmetric_square") & (table.defect_height > 0)].copy()
    baseline = (
        table[(table.geometry_group == "symmetric_square") & np.isclose(table.defect_height, 0)]
        .set_index("alpha_over_pi")
    )
    for column, _ in HEATMAP_METRIC_SPECS:
        subset[f"delta_{column}"] = subset.apply(
            lambda row: row[column] - baseline.loc[row.alpha_over_pi, column], axis=1
        )
    with mpl.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.65), constrained_layout=True)
        for panel, (ax, (column, ylabel)) in enumerate(zip(axes.ravel(), HEATMAP_METRIC_SPECS)):
            matrix = subset.pivot(index="defect_height", columns="alpha_over_pi", values=f"delta_{column}")
            matrix = matrix.sort_index().reindex(columns=ALPHA_OVER_PI)
            vmax = max(float(np.nanmax(np.abs(matrix.to_numpy()))), 1e-6)
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(ALPHA_OVER_PI)), [_alpha_label(v) for v in ALPHA_OVER_PI])
            ax.set_yticks(range(len(matrix.index)), [rf"${h / COMMON_PARAMETERS['boundaryLength']:.3f}$" for h in matrix.index])
            ax.set_xlabel(r"Phase lag $\alpha$")
            ax.set_ylabel(r"Defect depth $H/L$")
            ax.set_title(rf"$\Delta$ {ylabel}", fontsize=11)
            ax.text(-0.12, 1.06, f"({_subplot_label(panel)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=12, fontweight="bold")
            threshold = 0.46 * vmax
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix.iloc[row, col]
                    ax.text(col, row, f"{value:+.2f}", ha="center", va="center",
                            fontsize=8.3, color="white" if abs(value) > threshold else "#262626")
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
        save_figure(fig, output)


def plot_radial_profile_examples(details: dict, output: Path) -> None:
    keys = [
        ("square_H0_a0.4", "No defect", 0.4),
        ("square_H3_a0.4", r"$H=3$", 0.4),
        ("square_H0_a0.6", "No defect", 0.6),
        ("square_H3_a0.6", r"$H=3$", 0.6),
    ]
    with mpl.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.35), constrained_layout=True)
        for panel, (ax, alpha) in enumerate(zip(axes, (0.4, 0.6))):
            for name, label, key_alpha in keys:
                if not np.isclose(alpha, key_alpha):
                    continue
                detail = details[name]
                direction = detail["direction"]
                ax.plot(
                    detail["radial_distance"] / COMMON_PARAMETERS["boundaryLength"],
                    direction * detail["radial_current_smoothed"],
                    lw=1.8, label=label,
                )
                ax.axvline(
                    detail["edge_width"] / COMMON_PARAMETERS["boundaryLength"],
                    color=ax.lines[-1].get_color(), lw=1.0, ls=":",
                )
            ax.axhline(0, color="#8E8E8E", lw=0.7)
            ax.set_xlabel(r"Wall distance $d/L$")
            ax.set_ylabel(r"Oriented radial current density $\sigma j_s(d)$")
            ax.set_title(rf"$\alpha={alpha:g}\pi$")
            ax.text(0.02, 0.96, f"({_subplot_label(panel)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=12, fontweight="bold")
            ax.grid(True, color="#E3E5E8", lw=0.55)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        save_figure(fig, output)


def metric_models() -> tuple[list, list]:
    square = [
        build_model(model_library.CollisionBoundaryPatternFormation, alpha)
        for alpha in ALPHA_OVER_PI
    ]
    for height in FOUR_SPIKE_HEIGHTS:
        square.extend(
            build_model(
                model_library.CollisionBoundaryFourSpikePatternFormation,
                alpha,
                protrusionHeight=height,
                protrusionHalfWidth=SPIKE_HALF_WIDTH,
            )
            for alpha in ALPHA_OVER_PI
        )
    circle = [
        build_model(model_library.CircularBoundaryPatternFormation, alpha)
        for alpha in ALPHA_OVER_PI
    ]
    circle.extend(
        build_model(
            model_library.CollisionBoundaryMidpointSpikePatternFormation,
            alpha,
            protrusionHeight=ASYMMETRIC_SPIKE_HEIGHT,
            protrusionHalfWidth=SPIKE_HALF_WIDTH,
        )
        for alpha in ALPHA_OVER_PI
    )
    return square, circle


def run_metrics() -> tuple[pd.DataFrame, dict]:
    square, circle = metric_models()
    models = square + circle
    validate_exact_files(models)
    rows: list[dict] = []
    details: dict = {}
    for index, model in enumerate(models, start=1):
        print(f"Metrics {index}/{len(models)}: {data_path(model).name}")
        row, detail = compute_metrics(model)
        rows.append(row)
        group = "square" if row["geometry_group"] == "symmetric_square" else "circle"
        details[
            f"{group}_H{row['defect_height']:g}_a{row['alpha_over_pi']:g}"
        ] = detail
    table = pd.DataFrame(rows).sort_values(
        ["geometry_group", "defect_height", "alpha_over_pi"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "Boundary_Defect_Metrics.csv", index=False)
    configuration = {
        "analysis_window_time": ANALYSIS_WINDOW_TIME,
        "radial_bin_count": RADIAL_BIN_COUNT,
        "max_analysis_depth_fraction": MAX_ANALYSIS_DEPTH_FRACTION,
        "direction_probe_fraction": DIRECTION_PROBE_FRACTION,
        "peak_search_depth_fraction": PEAK_SEARCH_DEPTH_FRACTION,
        "edge_floor_fraction": EDGE_FLOOR_FRACTION,
        "edge_floor_consecutive_bins": EDGE_FLOOR_CONSECUTIVE_BINS,
        "penetration_current_quantile": PENETRATION_CURRENT_QUANTILE,
        "min_boundary_residence": MIN_BOUNDARY_RESIDENCE,
        "min_directional_persistence": MIN_DIRECTIONAL_PERSISTENCE,
        "min_mean_tangentiality": MIN_MEAN_TANGENTIALITY,
        "arc_bin_count": ARC_BIN_COUNT,
        "temporal_block_count": TEMPORAL_BLOCK_COUNT,
    }
    (OUTPUT_DIR / "Analysis_Parameters.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    write_metric_definitions(OUTPUT_DIR / "Metric_Definitions.txt")
    return table, details


def write_metric_definitions(path: Path) -> None:
    text = rf"""Boundary-transport metric definitions
=====================================

Boundary coordinates
--------------------
For each saved particle state, d_i(t) is the shortest Euclidean distance to
the true boundary; s_i(t) is the corresponding counterclockwise arc coordinate;
t_hat(s_i) is the local counterclockwise unit tangent; and
q_i(t) = v_i(t) . t_hat(s_i) / v = cos(theta_i - theta_tangent) in [-1, 1].

Adaptive wall-connected layer
-----------------------------
The time-averaged signed current is radially binned as
    j_s(d_k) = (1 / N T) sum_{{t,i in bin k}} q_i(t).
Its direction sigma is the sign of the current within
d <= {DIRECTION_PROBE_FRACTION:.3f} L.  Starting from the first near-wall
maximum of sigma*j_s, the edge layer ends at the first sign reversal or after
{EDGE_FLOOR_CONSECUTIVE_BINS} consecutive bins below
{EDGE_FLOOR_FRACTION:.2f} of that maximum.  This is d_e.  It selects the first
wall-connected current lobe and excludes a separated counterflowing bulk vortex.

Reported metrics
----------------
1. Signed edge current:
       J_e = < q_i(t) 1[d_i(t) <= d_e] 1[i is a carrier] >_{{i,t}}.
   |J_e| includes both the number of participating particles and their
   tangential alignment.  Positive is CCW and negative is CW.

2. Directional purity:
       chi_e = sum q_i 1_edge 1_carrier / sum |q_i| 1_edge 1_carrier,
       -1 <= chi_e <= 1.
   |chi_e|=1 is purely unidirectional; 0 is balanced counterpropagation.

3. Persistent carrier fraction:
   For particle i, residence R_i=<1_edge>_t, persistence
   P_i=|sum_t q_i 1_edge|/sum_t |q_i|1_edge, and mean tangentiality
   A_i=sum_t |q_i|1_edge/sum_t 1_edge.  A persistent carrier obeys
       R_i >= {MIN_BOUNDARY_RESIDENCE:.2f},
       P_i >= {MIN_DIRECTIONAL_PERSISTENCE:.2f},
       A_i >= {MIN_MEAN_TANGENTIALITY:.2f}.
   f_c is the fraction of all particles satisfying all three conditions.

4. Penetration width:
   xi_90 is the smallest d <= d_e containing
   {100*PENETRATION_CURRENT_QUANTILE:.0f}% of the direction-consistent radial
   current of persistent carriers.  The table also reports xi_90/L.

5. Effective boundary coverage:
   Divide the boundary into M={ARC_BIN_COUNT} equal arc bins.  Let g_m be the
   nonnegative, direction-consistent flux of persistent carriers in bin m.
       F_cov = (sum_m g_m)^2 / (M sum_m g_m^2).
   F_cov approaches 1 for uniform full-boundary transport and becomes small
   when transport is confined to one side.  This participation-ratio definition
   avoids an arbitrary occupied-bin threshold.

6. Temporal stability:
   Split the terminal window into B={TEMPORAL_BLOCK_COUNT} blocks and calculate
   carrier current J_b in each block with the same d_e and carrier set.  Then
       S_J = |mean_b J_b| / (|mean_b J_b| + std_b J_b).
   S_J approaches 1 for a steady signed current and 0 for a weak/fluctuating one.

Defect response
---------------
For every alpha and metric X, the plotted defect response is the bounded,
non-ratio difference Delta X(H,alpha)=X(H,alpha)-X(0,alpha).  This avoids
spurious divergences when the no-defect baseline current is close to zero.

Data window
-----------
Only the final {ANALYSIS_WINDOW_TIME:g} model-time units of every existing HDF5
trajectory are used.  Files are never continued, regenerated, or overwritten.
"""
    path.write_text(text, encoding="utf-8")


def create_comparison_states() -> None:
    square_models = [
        build_model(model_library.CollisionBoundaryPatternFormation, alpha)
        for alpha in TERMINAL_COMPARISON_ALPHA_OVER_PI
    ]
    square_models.extend(
        build_model(
            model_library.CollisionBoundaryFourSpikePatternFormation,
            alpha,
            protrusionHeight=1.0,
            protrusionHalfWidth=SPIKE_HALF_WIDTH,
        )
        for alpha in TERMINAL_COMPARISON_ALPHA_OVER_PI
    )
    circle_models = [
        build_model(model_library.CircularBoundaryPatternFormation, alpha)
        for alpha in TERMINAL_COMPARISON_ALPHA_OVER_PI
    ]
    circle_models.extend(
        build_model(
            model_library.CollisionBoundaryMidpointSpikePatternFormation,
            alpha,
            protrusionHeight=ASYMMETRIC_SPIKE_HEIGHT,
            protrusionHalfWidth=SPIKE_HALF_WIDTH,
        )
        for alpha in TERMINAL_COMPARISON_ALPHA_OVER_PI
    )
    state_sweep_figure(
        square_models,
        OUTPUT_DIR / "Square_Boundary_And_Four_Symmetric_Defects_Alpha_Sweep_Terminal_States_V2.pdf",
        row_labels=("No defect", r"Four defects: $H=1.0$"),
        save_png=True,
        figure_title="Square Boundary: No Defect vs Four Symmetric Defects",
    )
    state_sweep_figure(
        circle_models,
        OUTPUT_DIR / "Circular_Boundary_And_Single_Defect_Alpha_Sweep_Terminal_States_V2.pdf",
        row_labels=("No defect", r"Single defect: $H=3.0$"),
        save_png=True,
        figure_title="Circular Boundary: No Defect vs Single Defect",
    )


def create_light_rerenders() -> None:
    circular = [
        build_model(model_library.CircularBoundaryPatternFormation, alpha)
        for alpha in ALPHA_OVER_PI
    ]
    state_sweep_figure(
        circular,
        LIGHT_RERENDER_DIR / "Circular_Boundary_Alpha_Sweep_Terminal_Light_Boundary.pdf",
    )

    midpoint = [
        build_model(
            model_library.CollisionBoundaryMidpointSpikePatternFormation,
            alpha,
            protrusionHeight=3.0,
            protrusionHalfWidth=SPIKE_HALF_WIDTH,
        )
        for alpha in ALPHA_OVER_PI
    ]
    midpoint_iterations = [
        MIDPOINT_SELECTED_ITERATIONS.get(float(alpha)) for alpha in ALPHA_OVER_PI
    ]
    state_sweep_figure(
        midpoint,
        LIGHT_RERENDER_DIR / "Circular_Midpoint_Spike_H3_Alpha_Sweep_Selected_Terminal_Light_Boundary.pdf",
        midpoint_iterations,
    )

    square = [
        build_model(model_library.CollisionBoundaryPatternFormation, alpha)
        for alpha in ALPHA_OVER_PI
    ]
    for iteration in SQUARE_RERENDER_ITERATIONS:
        state_sweep_figure(
            square,
            LIGHT_RERENDER_DIR / f"Square_No_Spike_Alpha_Sweep_T{iteration}_Light_Boundary.pdf",
            [iteration] * len(square),
        )

    for height, iteration in FOUR_SPIKE_RERENDER_SPECS:
        models = [
            build_model(
                model_library.CollisionBoundaryFourSpikePatternFormation,
                alpha,
                protrusionHeight=height,
                protrusionHalfWidth=SPIKE_HALF_WIDTH,
            )
            for alpha in ALPHA_OVER_PI
        ]
        state_sweep_figure(
            models,
            LIGHT_RERENDER_DIR
            / f"Square_Four_Spike_H{height:.1f}_W{SPIKE_HALF_WIDTH:.2f}_Alpha_Sweep_T{iteration}_Light_Boundary.pdf",
            [iteration] * len(models),
        )


def run_all() -> None:
    table, details = run_metrics()
    plot_metric_curves(
        table, "symmetric_square", OUTPUT_DIR / "Metrics_Symmetric_Defects.pdf"
    )
    plot_metric_curves(
        table, "asymmetric_circle", OUTPUT_DIR / "Metrics_Asymmetric_Defect.pdf"
    )
    plot_defect_response_heatmaps(
        table, OUTPUT_DIR / "Defect_Response_Heatmaps.pdf"
    )
    plot_radial_profile_examples(
        details, OUTPUT_DIR / "Adaptive_Edge_Layer_Examples.pdf"
    )
    create_comparison_states()
    create_light_rerenders()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "metrics", "states", "rerender"), default="all"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "all":
            run_all()
        elif args.mode == "metrics":
            table, details = run_metrics()
            plot_metric_curves(table, "symmetric_square", OUTPUT_DIR / "Metrics_Symmetric_Defects.pdf")
            plot_metric_curves(table, "asymmetric_circle", OUTPUT_DIR / "Metrics_Asymmetric_Defect.pdf")
            plot_defect_response_heatmaps(table, OUTPUT_DIR / "Defect_Response_Heatmaps.pdf")
            plot_radial_profile_examples(details, OUTPUT_DIR / "Adaptive_Edge_Layer_Examples.pdf")
        elif args.mode == "states":
            create_comparison_states()
        else:
            create_light_rerenders()
    except (DataContractError, ValueError, TypeError) as exc:
        print(f"STOPPED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
