"""Analyse relay-form boundary transport from exact matched HDF5 trajectories.

The two headline observables are deliberately Eulerian and intensity-free:

``Xi_relay``
    Maximum product of time-block fraction and gate fraction over one
    same-sign, forward-kinematic spacetime connected component.

``Xi_fix``
    Absolute mean of reliable block-handedness labels in {-1,0,+1}.  It is
    undefined when fewer than half of the terminal-window blocks have enough
    spatial and independent-particle evidence.

Particle identity, crossing count, speed, and carrier population never weight
either score.  Identity is retained only for the relay diagnostic that asks
whether one particle spans the whole supported interval.  The alpha=pi state
is treated separately through a dynamic boundary-density structure factor;
that diagnostic is not a third sweep metric.

This module reads existing HDF5 files and writes only below
``output/Relay_Boundary_Flow_Analysis``.  It never runs a simulation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, NamedTuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boundary_defect_analysis as bda
import main as model_library


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AnalysisConfig:
    """All user-adjustable numerical and rendering parameters."""

    output_dir: Path = ROOT / "output" / "Relay_Boundary_Flow_Analysis"
    analysis_window_time: float = 10.0
    time_block_count: int = 20
    block_origin_fraction: float = 0.00
    boundary_gate_count: int = 64
    edge_layer_over_d0: float = 0.50
    contact_width_over_d0: float = 0.15
    projection_jump_step_factor: float = 2.00
    minimum_interval_recurrence: int = 2
    minimum_interval_recurrence_fraction: float = 0.20
    minimum_directional_purity: float = 0.75
    minimum_episode_time: float = 0.50
    minimum_episode_directional_persistence: float = 0.75
    minimum_episode_tangentiality: float = 0.35
    minimum_particles_per_gate: int = 2
    minimum_particles_per_block: int = 4
    minimum_active_gates: int = 3
    minimum_active_gate_fraction: float = 0.00
    minimum_supported_block_fraction: float = 0.50
    support_sensitivity: tuple[int, ...] = (3, 5, 7, 10)
    null_repetitions: int = 199
    null_random_seed: int = 20260824
    structure_spatial_modes: int = 31
    structure_frequency_bins: int = 80
    figure_dpi: int = 420


CONFIG = AnalysisConfig()
ALPHAS = tuple(float(value) for value in bda.ALPHA_OVER_PI)
EPS = np.finfo(float).eps

# Output words use underscore-separated Title_Case, matching the project rule.
CASE_SPECS = (
    ("Square", "Square", "square", 0.0),
    ("Square", "Square_Four_Defects_H1", "square_defect", 1.0),
    ("Square", "Square_Four_Defects_H1_5", "square_defect", 1.5),
    ("Square", "Square_Four_Defects_H3", "square_defect", 3.0),
    ("Circular", "Circular", "circle", 0.0),
    ("Circular", "Circular_Single_Defect_H3", "circle_defect", 3.0),
)

CASE_LABELS = {
    "Square": "Square",
    "Square_Four_Defects_H1": "Square, four defects (H=1)",
    "Square_Four_Defects_H1_5": "Square, four defects (H=1.5)",
    "Square_Four_Defects_H3": "Square, four defects (H=3)",
    "Circular": "Circular",
    "Circular_Single_Defect_H3": "Circular, single defect (H=3)",
}

COLORS = {
    "Square": "#38598C",
    "Square_Four_Defects_H1": "#6C86B5",
    "Square_Four_Defects_H1_5": "#8D6B9F",
    "Square_Four_Defects_H3": "#704264",
    "Circular": "#278C7B",
    "Circular_Single_Defect_H3": "#93722E",
}

MARKERS = {
    "Square": "s",
    "Square_Four_Defects_H1": "D",
    "Square_Four_Defects_H1_5": "P",
    "Square_Four_Defects_H3": "X",
    "Circular": "o",
    "Circular_Single_Defect_H3": "^",
}


class CrossingData(NamedTuple):
    counts_ccw: np.ndarray
    counts_cw: np.ndarray
    interval_presence_ccw: np.ndarray
    interval_presence_cw: np.ndarray
    interval_gate_ccw: np.ndarray
    interval_gate_cw: np.ndarray
    particle_ids_ccw: list[list[set[int]]]
    particle_ids_cw: list[list[set[int]]]
    normalized_handedness: np.ndarray
    active: np.ndarray
    node_sign: np.ndarray
    supported: np.ndarray
    block_handedness: np.ndarray
    block_particle_count: np.ndarray
    recurrence_threshold: np.ndarray
    maximum_gate_advance: int
    accepted_episode_count: int
    accepted_interval_count: int
    rejected_projection_jump_count: int
    edge_width: float
    contact_width: float
    distance: np.ndarray
    arc: np.ndarray
    positions: np.ndarray
    total_frames: int
    frame_indices: np.ndarray


def build_model(label: str, alpha_over_pi: float, height: float):
    """Build the exact model signature used by the existing-data contract."""
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


def all_specifications() -> list[tuple[str, str, str, float, float, object]]:
    specifications = []
    for group, condition, label, height in CASE_SPECS:
        for alpha in ALPHAS:
            model = build_model(label, alpha, height)
            specifications.append((group, condition, label, height, alpha, model))
    return specifications


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.0,
        }
    )


def versioned_path(base: Path) -> Path:
    """Return a non-overwriting output path."""
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        return base
    version = 2
    while True:
        candidate = base.with_name(f"{base.stem}_V{version}{base.suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def save_figure(fig: plt.Figure, stem: str, config: AnalysisConfig) -> tuple[Path, Path]:
    png = versioned_path(config.output_dir / f"{stem}.png")
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=config.figure_dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def _wrap_arclength(delta: np.ndarray, perimeter: float) -> np.ndarray:
    """Map arclength increments to [-P/2, P/2)."""
    return np.mod(delta + 0.5 * perimeter, perimeter) - 0.5 * perimeter


def _runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    """Yield half-open maximal True intervals of a one-dimensional mask."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    yield from zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def _crossed_gates(
    start: float, delta: float, perimeter: float, gate_count: int,
) -> list[int]:
    """Enumerate gates crossed by one wrapped, unwrapped-in-time step."""
    if delta == 0.0:
        return []
    spacing = perimeter / gate_count
    end = start + delta
    tolerance = 64 * EPS * max(1.0, perimeter)
    if delta > 0:
        first = math.floor((start + tolerance) / spacing) + 1
        last = math.floor((end + tolerance) / spacing)
        return [integer % gate_count for integer in range(first, last + 1)]
    first = math.ceil((end - tolerance) / spacing)
    last = math.ceil((start - tolerance) / spacing) - 1
    return [integer % gate_count for integer in range(first, last + 1)]


def crossing_field(model, config: AnalysisConfig, minimum_active_gates: int | None = None) -> CrossingData:
    """Construct the discrete wall-anchored gate-crossing field."""
    minimum_active_gates = (
        config.minimum_active_gates
        if minimum_active_gates is None
        else int(minimum_active_gates)
    )
    window = bda.load_window(model, duration=config.analysis_window_time)
    geometry = bda.geometry_for(model)
    distance, arc, _ = bda.project_boundary(window.positions, geometry)
    saved_dt = float(model.dt * model.shotsnaps)
    physical_step = float(model.speedV * saved_dt)
    edge_width = float(config.edge_layer_over_d0 * model.distanceD0)
    contact_width = float(min(edge_width, config.contact_width_over_d0 * model.distanceD0))
    intervals = window.positions.shape[0] - 1
    block_width_intervals = intervals / config.time_block_count
    block_origin = int(round(config.block_origin_fraction * block_width_intervals))
    interval_blocks = np.floor(
        (np.arange(intervals) - block_origin) / block_width_intervals
    ).astype(int)
    interval_in_blocks = (
        (interval_blocks >= 0) & (interval_blocks < config.time_block_count)
    )

    counts_ccw = np.zeros(
        (config.time_block_count, config.boundary_gate_count), dtype=np.int64
    )
    counts_cw = np.zeros_like(counts_ccw)
    interval_gate_ccw = np.zeros((intervals, config.boundary_gate_count), dtype=bool)
    interval_gate_cw = np.zeros_like(interval_gate_ccw)
    particle_ids_ccw = [
        [set() for _ in range(config.boundary_gate_count)]
        for _ in range(config.time_block_count)
    ]
    particle_ids_cw = [
        [set() for _ in range(config.boundary_gate_count)]
        for _ in range(config.time_block_count)
    ]
    accepted_episode_count = 0
    accepted_interval_count = 0
    rejected_projection_jump_count = 0
    minimum_episode_intervals = max(
        1, int(math.ceil(config.minimum_episode_time / saved_dt))
    )

    for particle in range(model.agentsNum):
        in_layer = distance[:, particle] <= edge_width
        for start_frame, stop_frame in _runs(in_layer):
            if stop_frame - start_frame - 1 < minimum_episode_intervals:
                continue
            if float(np.min(distance[start_frame:stop_frame, particle])) > contact_width:
                continue
            s0 = arc[start_frame : stop_frame - 1, particle]
            s1 = arc[start_frame + 1 : stop_frame, particle]
            delta = _wrap_arclength(s1 - s0, geometry.perimeter)
            kinematic = np.abs(delta) <= config.projection_jump_step_factor * physical_step
            rejected_projection_jump_count += int(np.count_nonzero(~kinematic))
            accepted_delta = delta[kinematic]
            absolute_sum = float(np.abs(accepted_delta).sum())
            if absolute_sum <= EPS:
                continue
            directional_persistence = abs(float(accepted_delta.sum())) / absolute_sum
            tangentiality = float(np.mean(np.abs(accepted_delta) / physical_step))
            if (
                directional_persistence < config.minimum_episode_directional_persistence
                or tangentiality < config.minimum_episode_tangentiality
            ):
                continue
            accepted_episode_count += 1
            frame_numbers = np.arange(start_frame, stop_frame - 1)
            for local, frame in enumerate(frame_numbers):
                if not bool(kinematic[local]):
                    continue
                step = float(delta[local])
                gates = _crossed_gates(
                    float(s0[local]), step, geometry.perimeter, config.boundary_gate_count
                )
                if not gates:
                    continue
                accepted_interval_count += 1
                if not interval_in_blocks[frame]:
                    continue
                block = int(interval_blocks[frame])
                target = counts_ccw if step > 0 else counts_cw
                target_particles = particle_ids_ccw if step > 0 else particle_ids_cw
                target_intervals = interval_gate_ccw if step > 0 else interval_gate_cw
                for gate in gates:
                    target[block, gate] += 1
                    target_particles[block][gate].add(particle)
                    target_intervals[frame, gate] = True

    interval_presence_ccw = np.zeros_like(counts_ccw)
    interval_presence_cw = np.zeros_like(counts_cw)
    recurrence_threshold = np.zeros(config.time_block_count, dtype=int)
    for block in range(config.time_block_count):
        selected = (interval_blocks == block) & interval_in_blocks
        interval_presence_ccw[block] = interval_gate_ccw[selected].sum(axis=0)
        interval_presence_cw[block] = interval_gate_cw[selected].sum(axis=0)
        recurrence_threshold[block] = max(
            config.minimum_interval_recurrence,
            int(math.ceil(config.minimum_interval_recurrence_fraction * selected.sum())),
        )
    block_duration = intervals * saved_dt / config.time_block_count
    gate_spacing = geometry.perimeter / config.boundary_gate_count
    maximum_gate_advance = int(math.ceil(model.speedV * block_duration / gate_spacing)) + 1

    total = interval_presence_ccw + interval_presence_cw
    normalized_handedness = np.divide(
        interval_presence_ccw - interval_presence_cw,
        total,
        out=np.zeros_like(total, dtype=float),
        where=total > 0,
    )
    particle_count_ccw = np.array(
        [[len(ids) for ids in row] for row in particle_ids_ccw], dtype=int
    )
    particle_count_cw = np.array(
        [[len(ids) for ids in row] for row in particle_ids_cw], dtype=int
    )
    share_ccw = np.divide(
        interval_presence_ccw,
        total,
        out=np.zeros_like(total, dtype=float),
        where=total > 0,
    )
    share_cw = np.divide(
        interval_presence_cw,
        total,
        out=np.zeros_like(total, dtype=float),
        where=total > 0,
    )
    node_ccw = (
        (interval_presence_ccw >= recurrence_threshold[:, None])
        & (share_ccw >= config.minimum_directional_purity)
        & (particle_count_ccw >= config.minimum_particles_per_gate)
    )
    node_cw = (
        (interval_presence_cw >= recurrence_threshold[:, None])
        & (share_cw >= config.minimum_directional_purity)
        & (particle_count_cw >= config.minimum_particles_per_gate)
    )
    node_sign = node_ccw.astype(np.int8) - node_cw.astype(np.int8)
    active = node_sign != 0
    required_gates = max(
        minimum_active_gates,
        int(math.ceil(config.minimum_active_gate_fraction * config.boundary_gate_count)),
    )
    block_particle_count = np.zeros(config.time_block_count, dtype=int)
    for block in range(config.time_block_count):
        particles: set[int] = set()
        for gate in np.flatnonzero(node_ccw[block]):
            particles.update(particle_ids_ccw[block][gate])
        for gate in np.flatnonzero(node_cw[block]):
            particles.update(particle_ids_cw[block][gate])
        block_particle_count[block] = len(particles)
    supported = (
        (active.sum(axis=1) >= required_gates)
        & (block_particle_count >= config.minimum_particles_per_block)
    )
    block_handedness = np.full(config.time_block_count, np.nan)
    for block in np.flatnonzero(active.any(axis=1)):
        positive_fraction = float(np.mean(node_sign[block, active[block]] > 0))
        if positive_fraction >= config.minimum_directional_purity:
            block_handedness[block] = 1.0
        elif positive_fraction <= 1.0 - config.minimum_directional_purity:
            block_handedness[block] = -1.0
        else:
            block_handedness[block] = 0.0

    return CrossingData(
        counts_ccw=counts_ccw,
        counts_cw=counts_cw,
        interval_presence_ccw=interval_presence_ccw,
        interval_presence_cw=interval_presence_cw,
        interval_gate_ccw=interval_gate_ccw,
        interval_gate_cw=interval_gate_cw,
        particle_ids_ccw=particle_ids_ccw,
        particle_ids_cw=particle_ids_cw,
        normalized_handedness=normalized_handedness,
        active=active,
        node_sign=node_sign,
        supported=supported,
        block_handedness=block_handedness,
        block_particle_count=block_particle_count,
        recurrence_threshold=recurrence_threshold,
        maximum_gate_advance=maximum_gate_advance,
        accepted_episode_count=accepted_episode_count,
        accepted_interval_count=accepted_interval_count,
        rejected_projection_jump_count=rejected_projection_jump_count,
        edge_width=edge_width,
        contact_width=contact_width,
        distance=distance,
        arc=arc,
        positions=window.positions,
        total_frames=window.total_frames,
        frame_indices=window.frame_indices,
    )


def rethreshold_field(
    field: CrossingData, minimum_active_gates: int, minimum_particles_per_block: int,
) -> CrossingData:
    """Change only the binary support gate; raw crossings remain untouched."""
    supported = (
        (field.active.sum(axis=1) >= int(minimum_active_gates))
        & (field.block_particle_count >= int(minimum_particles_per_block))
    )
    return field._replace(supported=supported)


def _component_nodes(field: CrossingData, sign: int) -> list[set[tuple[int, int]]]:
    """Return same-sign, forward-kinematic spacetime components."""
    nodes = {
        (int(block), int(gate))
        for block, gate in np.argwhere(field.node_sign == sign)
        if field.supported[int(block)]
    }
    if not nodes:
        return []
    parent = {node: node for node in nodes}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    gate_count = field.node_sign.shape[1]
    for block in range(field.node_sign.shape[0] - 1):
        current = np.flatnonzero(
            (field.node_sign[block] == sign) & field.supported[block]
        )
        following = set(np.flatnonzero(
            (field.node_sign[block + 1] == sign) & field.supported[block + 1]
        ).tolist())
        for gate in current:
            for advance in range(field.maximum_gate_advance + 1):
                next_gate = int((int(gate) + sign * advance) % gate_count)
                if next_gate in following:
                    union((block, int(gate)), (block + 1, next_gate))

    components: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for node in nodes:
        components.setdefault(find(node), set()).add(node)
    return list(components.values())


def _component_score(component: set[tuple[int, int]], block_count: int, gate_count: int) -> float:
    blocks = {block for block, _ in component}
    gates = {gate for _, gate in component}
    return float((len(blocks) / block_count) * (len(gates) / gate_count))


def relay_metrics(
    field: CrossingData, gate_count: int, minimum_supported_fraction: float = 0.50,
) -> dict[str, float | int | str | bool]:
    """Evaluate graph-connected relay coverage and handedness fixation."""
    block_count = field.node_sign.shape[0]
    signed_components = {
        sign: _component_nodes(field, sign) for sign in (-1, 1)
    }
    best_by_sign: dict[int, tuple[float, set[tuple[int, int]]]] = {}
    for sign in (-1, 1):
        components = signed_components[sign]
        best_by_sign[sign] = max(
            ((_component_score(component, block_count, gate_count), component)
             for component in components),
            default=(0.0, set()),
            key=lambda item: (item[0], len(item[1])),
        )
    best_sign = max((-1, 1), key=lambda sign: best_by_sign[sign][0])
    xi_relay, best_component = best_by_sign[best_sign]
    component_blocks = {block for block, _ in best_component}
    component_gates = {gate for _, gate in best_component}

    directional_particles = (
        field.particle_ids_ccw if best_sign > 0 else field.particle_ids_cw
    )
    particle_blocks: dict[int, set[int]] = {}
    for block, gate in best_component:
        for particle in directional_particles[block][gate]:
            particle_blocks.setdefault(particle, set()).add(block)
    longest_particle_span = max((len(blocks) for blocks in particle_blocks.values()), default=0)
    component_block_count = len(component_blocks)
    max_particle_block_fraction = (
        longest_particle_span / component_block_count if component_block_count else 0.0
    )
    one_particle_every_block = (
        component_block_count > 0 and longest_particle_span == component_block_count
    )

    supported_values = field.block_handedness[field.supported]
    supported_fraction = float(field.supported.mean())
    if supported_values.size == 0 or supported_fraction < minimum_supported_fraction:
        xi_fix = float("nan")
        mean_handedness = float("nan")
    else:
        mean_handedness = float(np.mean(supported_values))
        xi_fix = float(abs(mean_handedness))
    direction = "NA"
    if np.isfinite(mean_handedness):
        direction = "CCW" if mean_handedness > 0 else ("CW" if mean_handedness < 0 else "Balanced")

    return {
        "xi_relay": float(xi_relay),
        "xi_relay_ccw": float(best_by_sign[1][0]),
        "xi_relay_cw": float(best_by_sign[-1][0]),
        "xi_fix": xi_fix,
        "mean_eulerian_handedness": mean_handedness,
        "mean_handedness_direction": direction,
        "supported_block_count": int(field.supported.sum()),
        "supported_block_fraction": supported_fraction,
        "relay_component_block_count": component_block_count,
        "relay_component_gate_count": len(component_gates),
        "relay_component_direction": "CCW" if best_sign > 0 else "CW",
        "one_particle_crosses_in_every_component_block": bool(one_particle_every_block),
        "maximum_single_particle_component_block_fraction": float(max_particle_block_fraction),
    }


def permutation_null_pvalue(
    field: CrossingData, observed: float, repetitions: int, random_seed: int,
) -> tuple[float, float, float]:
    """Break causal alignment by independently rotating each block on the ring."""
    if repetitions <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(random_seed)
    null_scores = np.zeros(repetitions, dtype=float)
    for repetition in range(repetitions):
        permuted = np.stack(
            [
                np.roll(row, int(rng.integers(field.node_sign.shape[1])))
                for row in field.node_sign
            ]
        )
        null_field = field._replace(node_sign=permuted)
        score = 0.0
        for sign in (-1, 1):
            for component in _component_nodes(null_field, sign):
                score = max(
                    score,
                    _component_score(
                        component, field.node_sign.shape[0], field.node_sign.shape[1]
                    ),
                )
        null_scores[repetition] = score
    pvalue = float((1 + np.count_nonzero(null_scores >= observed - 1e-12)) / (repetitions + 1))
    return pvalue, float(np.quantile(null_scores, 0.95)), float(np.median(null_scores))


def boundary_density(field: CrossingData, geometry, gate_count: int) -> np.ndarray:
    """Return fixed-layer boundary occupancy rho[n,k] from stored positions."""
    inside = field.distance <= field.edge_width
    indices = np.floor(np.mod(field.arc, geometry.perimeter) / geometry.perimeter * gate_count)
    indices = np.clip(indices.astype(int), 0, gate_count - 1)
    density = np.zeros((field.arc.shape[0], gate_count), dtype=float)
    for frame in range(field.arc.shape[0]):
        density[frame] = np.bincount(
            indices[frame, inside[frame]], minlength=gate_count
        )
    return density


def dynamic_structure_factor(
    density: np.ndarray, saved_dt: float, max_mode: int, max_frequency_bins: int,
) -> dict[str, np.ndarray | float]:
    """Compute the double-demeaned dynamic boundary structure factor."""
    fluctuation = (
        density
        - density.mean(axis=0, keepdims=True)
        - density.mean(axis=1, keepdims=True)
        + density.mean()
    )
    temporal_taper = np.hanning(fluctuation.shape[0])[:, None]
    spectrum = np.fft.fftshift(np.fft.fft2(fluctuation * temporal_taper), axes=(0, 1))
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.fftshift(np.fft.fftfreq(fluctuation.shape[0], d=saved_dt))
    modes = np.fft.fftshift(np.fft.fftfreq(fluctuation.shape[1]) * fluctuation.shape[1])
    mode_mask = np.abs(modes) <= max_mode
    if max_frequency_bins < frequencies.size:
        positive_limit = min(max_frequency_bins // 2, frequencies.size // 2)
        center = frequencies.size // 2
        frequency_slice = slice(center - positive_limit, center + positive_limit + 1)
    else:
        frequency_slice = slice(None)
    selected = power[frequency_slice][:, mode_mask]
    selected_frequencies = frequencies[frequency_slice]
    selected_modes = modes[mode_mask]

    # Exclude the zero-frequency and zero-mode axes when measuring propagation
    # peak concentration.  This scalar is an alpha=pi validation diagnostic,
    # never a headline sweep metric.
    propagating = selected.copy()
    propagating[np.isclose(selected_frequencies, 0.0), :] = 0.0
    propagating[:, np.isclose(selected_modes, 0.0)] = 0.0
    nonzero_total = float(propagating.sum())
    mode_frequency_product = np.outer(selected_frequencies, selected_modes)

    def sector_summary(sector_sign: int) -> tuple[float, float, float]:
        sector = np.where(sector_sign * mode_frequency_product > 0, propagating, 0.0)
        flat = np.sort(sector.ravel())[::-1]
        fraction = float(flat[:2].sum() / nonzero_total) if nonzero_total > EPS else 0.0
        peak_index = np.unravel_index(int(np.argmax(sector)), sector.shape)
        return (
            fraction,
            float(selected_frequencies[peak_index[0]]),
            float(selected_modes[peak_index[1]]),
        )

    positive_fraction, positive_frequency, positive_mode = sector_summary(1)
    negative_fraction, negative_frequency, negative_mode = sector_summary(-1)
    return {
        "power": selected,
        "frequencies": selected_frequencies,
        "modes": selected_modes,
        "positive_sector_peak_pair_fraction": positive_fraction,
        "negative_sector_peak_pair_fraction": negative_fraction,
        "positive_sector_peak_frequency": positive_frequency,
        "negative_sector_peak_frequency": negative_frequency,
        "positive_sector_peak_mode": positive_mode,
        "negative_sector_peak_mode": negative_mode,
    }


def analyse_one(model, config: AnalysisConfig) -> tuple[dict, dict]:
    field = crossing_field(model, config)
    metrics = relay_metrics(
        field, config.boundary_gate_count, config.minimum_supported_block_fraction
    )
    null_pvalue, null_q95, null_median = permutation_null_pvalue(
        field,
        float(metrics["xi_relay"]),
        config.null_repetitions,
        config.null_random_seed + int(round(1000 * model.phaseLagA0)),
    )
    geometry = bda.geometry_for(model)
    density = boundary_density(field, geometry, config.boundary_gate_count)
    structure = None
    if np.isclose(model.phaseLagA0, np.pi):
        structure = dynamic_structure_factor(
            density,
            model.dt * model.shotsnaps,
            config.structure_spatial_modes,
            config.structure_frequency_bins,
        )
    summary = {
        **metrics,
        "relay_block_shift_null_pvalue": null_pvalue,
        "relay_block_shift_null_q95": null_q95,
        "relay_block_shift_null_median": null_median,
        "alpha_over_pi": float(model.phaseLagA0 / np.pi),
        "defect_height": float(getattr(model, "protrusionHeight", 0.0)),
        "accepted_edge_episode_count": field.accepted_episode_count,
        "accepted_gate_crossing_interval_count": field.accepted_interval_count,
        "rejected_projection_jump_count": field.rejected_projection_jump_count,
        "edge_width": field.edge_width,
        "contact_width": field.contact_width,
        "median_active_gate_count": float(np.median(field.active.sum(axis=1))),
        "terminal_saved_frame": int(field.total_frames - 1),
        "window_first_saved_frame": int(field.frame_indices[0]),
        "hdf5_file": bda.data_path(model).name,
        "pi_structure_positive_sector_peak_pair_fraction": (
            float(structure["positive_sector_peak_pair_fraction"])
            if structure is not None else float("nan")
        ),
        "pi_structure_negative_sector_peak_pair_fraction": (
            float(structure["negative_sector_peak_pair_fraction"])
            if structure is not None else float("nan")
        ),
        "pi_structure_bidirectional_peak_floor": (
            min(
                float(structure["positive_sector_peak_pair_fraction"]),
                float(structure["negative_sector_peak_pair_fraction"]),
            ) if structure is not None else float("nan")
        ),
    }
    detail = {"field": field, "density": density, "structure": structure}
    return summary, detail


def compute_all(config: AnalysisConfig) -> tuple[pd.DataFrame, dict[tuple[str, float], dict]]:
    specifications = all_specifications()
    bda.validate_exact_files([item[-1] for item in specifications])
    rows: list[dict] = []
    details: dict[tuple[str, float], dict] = {}
    for index, (group, condition, label, height, alpha, model) in enumerate(specifications, start=1):
        print(f"Relay analysis {index}/{len(specifications)}: {bda.data_path(model).name}")
        summary, detail = analyse_one(model, config)
        summary.update(
            {
                "geometry_group": group,
                "condition": condition,
                "model_label": label,
            }
        )
        rows.append(summary)
        details[(condition, alpha)] = detail
    return pd.DataFrame(rows), details


def compute_support_sensitivity(config: AnalysisConfig) -> pd.DataFrame:
    """Re-evaluate the reviewed G_min grid without changing any raw data."""
    specifications = all_specifications()
    bda.validate_exact_files([item[-1] for item in specifications])
    rows: list[dict] = []
    for index, (group, condition, label, height, alpha, model) in enumerate(specifications, start=1):
        print(f"Sensitivity {index}/{len(specifications)}: {condition}, alpha/pi={alpha:g}")
        raw_field = crossing_field(model, config)
        for threshold in config.support_sensitivity:
            field = rethreshold_field(
                raw_field, threshold, config.minimum_particles_per_block
            )
            metrics = relay_metrics(
                field, config.boundary_gate_count, config.minimum_supported_block_fraction
            )
            rows.append(
                {
                    "geometry_group": group,
                    "condition": condition,
                    "alpha_over_pi": alpha,
                    "minimum_active_gates": threshold,
                    "xi_relay": metrics["xi_relay"],
                    "xi_fix": metrics["xi_fix"],
                    "supported_block_count": metrics["supported_block_count"],
                }
            )
    return pd.DataFrame(rows)


def compute_parameter_sensitivity(config: AnalysisConfig) -> pd.DataFrame:
    """One-factor-at-a-time audit on representative transport regimes."""
    cases = (
        ("Square", "square", 0.0, 0.4),
        ("Square", "square", 0.0, 0.6),
        ("Square_Four_Defects_H3", "square_defect", 3.0, 0.0),
        ("Square_Four_Defects_H3", "square_defect", 3.0, 0.8),
        ("Circular", "circle", 0.0, 0.4),
        ("Circular", "circle", 0.0, 0.8),
        ("Circular_Single_Defect_H3", "circle_defect", 3.0, 0.4),
        ("Circular_Single_Defect_H3", "circle_defect", 3.0, 0.6),
    )
    variations: list[tuple[str, float, AnalysisConfig]] = [("Baseline", 0.0, config)]
    grids = {
        "Edge_Width_Over_Interaction_Radius": ("edge_layer_over_d0", (0.35, 0.75)),
        "Contact_Width_Over_Interaction_Radius": ("contact_width_over_d0", (0.10, 0.20)),
        "Projection_Jump_Step_Factor": ("projection_jump_step_factor", (1.50, 2.50)),
        "Time_Block_Count": ("time_block_count", (16, 24)),
        "Boundary_Gate_Count": ("boundary_gate_count", (48, 80)),
        "Interval_Recurrence_Fraction": ("minimum_interval_recurrence_fraction", (0.15, 0.25)),
        "Directional_Purity": ("minimum_directional_purity", (0.67, 0.80)),
        "Particles_Per_Gate": ("minimum_particles_per_gate", (1, 3)),
        "Episode_Time": ("minimum_episode_time", (0.25, 0.75)),
        "Episode_Persistence": ("minimum_episode_directional_persistence", (0.67, 0.80)),
        "Episode_Tangentiality": ("minimum_episode_tangentiality", (0.25, 0.45)),
        "Block_Origin_Fraction": ("block_origin_fraction", (0.50,)),
    }
    for factor, (attribute, values) in grids.items():
        for value in values:
            variations.append((factor, float(value), replace(config, **{attribute: value})))

    models = [build_model(label, alpha, height) for _, label, height, alpha in cases]
    bda.validate_exact_files(models)
    rows: list[dict] = []
    total = len(cases) * len(variations)
    counter = 0
    for condition, label, height, alpha in cases:
        model = build_model(label, alpha, height)
        for factor, value, varied in variations:
            counter += 1
            print(f"Parameter sensitivity {counter}/{total}: {condition}, alpha/pi={alpha:g}, {factor}")
            field = crossing_field(model, varied)
            metrics = relay_metrics(
                field,
                varied.boundary_gate_count,
                varied.minimum_supported_block_fraction,
            )
            rows.append(
                {
                    "condition": condition,
                    "alpha_over_pi": alpha,
                    "factor": factor,
                    "factor_value": value,
                    "xi_relay": metrics["xi_relay"],
                    "xi_fix": metrics["xi_fix"],
                    "supported_block_fraction": metrics["supported_block_fraction"],
                }
            )
    return pd.DataFrame(rows)


def compute_pi_structure_sensitivity(config: AnalysisConfig) -> pd.DataFrame:
    """Audit the alpha=pi spectrum against layer and arclength resolution."""
    cases = (
        ("Circular", "circle", 0.0),
        ("Circular_Single_Defect_H3", "circle_defect", 3.0),
        ("Square", "square", 0.0),
        ("Square_Four_Defects_H3", "square_defect", 3.0),
    )
    variations = (
        ("Baseline", 0.0, config),
        ("Edge_Width_Over_Interaction_Radius", 0.35, replace(config, edge_layer_over_d0=0.35)),
        ("Edge_Width_Over_Interaction_Radius", 0.75, replace(config, edge_layer_over_d0=0.75)),
        ("Boundary_Gate_Count", 48.0, replace(config, boundary_gate_count=48, structure_spatial_modes=23)),
        ("Boundary_Gate_Count", 80.0, replace(config, boundary_gate_count=80, structure_spatial_modes=39)),
    )
    models = [build_model(label, 1.0, height) for _, label, height in cases]
    bda.validate_exact_files(models)
    rows: list[dict] = []
    for condition, label, height in cases:
        model = build_model(label, 1.0, height)
        for factor, value, varied in variations:
            field = crossing_field(model, varied)
            density = boundary_density(field, bda.geometry_for(model), varied.boundary_gate_count)
            structure = dynamic_structure_factor(
                density,
                model.dt * model.shotsnaps,
                varied.structure_spatial_modes,
                varied.structure_frequency_bins,
            )
            rows.append(
                {
                    "condition": condition,
                    "factor": factor,
                    "factor_value": value,
                    "positive_sector_peak_pair_fraction": structure[
                        "positive_sector_peak_pair_fraction"
                    ],
                    "negative_sector_peak_pair_fraction": structure[
                        "negative_sector_peak_pair_fraction"
                    ],
                    "bidirectional_peak_floor": min(
                        structure["positive_sector_peak_pair_fraction"],
                        structure["negative_sector_peak_pair_fraction"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_tables(
    table: pd.DataFrame,
    support_sensitivity: pd.DataFrame,
    parameter_sensitivity: pd.DataFrame,
    structure_sensitivity: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    values_path = versioned_path(config.output_dir / "Relay_Boundary_Flow_Values.csv")
    support_path = versioned_path(
        config.output_dir / "Relay_Boundary_Flow_Support_Sensitivity.csv"
    )
    parameter_path = versioned_path(
        config.output_dir / "Relay_Boundary_Flow_Parameter_Sensitivity.csv"
    )
    structure_path = versioned_path(
        config.output_dir / "Pi_State_Structure_Sensitivity.csv"
    )
    configuration_path = versioned_path(
        config.output_dir / "Relay_Boundary_Flow_Configuration.json"
    )
    table.to_csv(values_path, index=False)
    support_sensitivity.to_csv(support_path, index=False)
    parameter_sensitivity.to_csv(parameter_path, index=False)
    structure_sensitivity.to_csv(structure_path, index=False)
    serializable = asdict(config)
    serializable["output_dir"] = str(serializable["output_dir"])
    configuration_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return values_path, support_path, parameter_path, structure_path, configuration_path


def plot_alpha_sweep(table: pd.DataFrame, config: AnalysisConfig) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for condition in CASE_LABELS:
        subset = table[table.condition.eq(condition)].sort_values("alpha_over_pi")
        style = {
            "color": COLORS[condition],
            "marker": MARKERS[condition],
            "ms": 5.0,
            "lw": 1.45,
            "label": CASE_LABELS[condition],
        }
        axes[0].plot(subset.alpha_over_pi, subset.xi_relay, **style)
        axes[1].plot(subset.alpha_over_pi, subset.xi_fix, **style)
    for axis, ylabel, title in (
        (axes[0], r"Relay coverage $\Xi_{\mathrm{relay}}$", "Persistent perimeter coverage"),
        (axes[1], r"Handedness fixation $\Xi_{\mathrm{fix}}$", "Mean Eulerian handedness"),
    ):
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(ALPHAS)
        axis.set_xticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1$"])
        axis.set_xlabel(r"Phase lag $\alpha/\pi$")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, color="#D9DEE4", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    for index, axis in enumerate(axes):
        axis.text(-0.14, 1.03, f"({chr(97 + index)})", transform=axis.transAxes,
                  fontweight="bold", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Relay Boundary Flow: Coverage and Handedness Are Separate")
    return save_figure(fig, "Relay_Boundary_Flow_Alpha_Sweep", config)


def plot_support_sensitivity(sensitivity: pd.DataFrame, config: AnalysisConfig) -> tuple[Path, Path]:
    conditions = list(CASE_LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for axis, condition in zip(axes.flat, conditions):
        subset = sensitivity[sensitivity.condition.eq(condition)]
        matrix = subset.pivot(
            index="minimum_active_gates", columns="alpha_over_pi", values="xi_relay"
        ).reindex(index=config.support_sensitivity, columns=ALPHAS)
        image = axis.imshow(matrix.to_numpy(), origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_title(CASE_LABELS[condition], fontsize=9.3)
        axis.set_xticks(range(len(ALPHAS)))
        axis.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"])
        axis.set_yticks(range(len(config.support_sensitivity)), config.support_sensitivity)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = float(matrix.iloc[row, column])
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.0,
                          color="white" if value < 0.45 else "black")
    for axis in axes[-1]:
        axis.set_xlabel(r"Phase lag $\alpha/\pi$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Support threshold $G_{\min}$")
    fig.colorbar(image, ax=list(axes.flat), fraction=0.018, pad=0.014,
                 label=r"Relay coverage $\Xi_{\mathrm{relay}}$")
    fig.suptitle("Support-Threshold Sensitivity")
    return save_figure(fig, "Relay_Boundary_Flow_Support_Sensitivity", config)


def plot_parameter_sensitivity(
    sensitivity: pd.DataFrame, config: AnalysisConfig,
) -> tuple[Path, Path]:
    factors = [factor for factor in sensitivity.factor.unique() if factor != "Baseline"]
    cases = sensitivity[["condition", "alpha_over_pi"]].drop_duplicates()
    case_keys = list(cases.itertuples(index=False, name=None))
    relay_delta = np.full((len(factors), len(case_keys)), np.nan)
    fix_delta = np.full_like(relay_delta, np.nan)
    for column, (condition, alpha) in enumerate(case_keys):
        subset = sensitivity[
            sensitivity.condition.eq(condition)
            & np.isclose(sensitivity.alpha_over_pi, alpha)
        ]
        baseline = subset[subset.factor.eq("Baseline")].iloc[0]
        for row, factor in enumerate(factors):
            varied = subset[subset.factor.eq(factor)]
            relay_delta[row, column] = float(
                np.max(np.abs(varied.xi_relay.to_numpy() - baseline.xi_relay))
            )
            finite = varied.xi_fix.notna() & np.isfinite(baseline.xi_fix)
            if finite.any():
                fix_delta[row, column] = float(
                    np.max(np.abs(varied.loc[finite, "xi_fix"].to_numpy() - baseline.xi_fix))
                )

    labels = [
        f"{CASE_LABELS[condition].replace(', four defects (H=3)', ', H=3').replace(', single defect (H=3)', ', defect')}\n"
        + rf"$\alpha={alpha:g}\pi$"
        for condition, alpha in case_keys
    ]
    display_factors = [factor.replace("_", " ") for factor in factors]
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 8.3), constrained_layout=True, sharex=True)
    images = []
    for index, (axis, matrix, title) in enumerate(
        zip(axes, (relay_delta, fix_delta), (r"Maximum $|\Delta\Xi_{\rm relay}|$", r"Maximum $|\Delta\Xi_{\rm fix}|$"))
    ):
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, origin="upper", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        images.append(image)
        axis.set_yticks(range(len(factors)), display_factors)
        axis.set_title(title)
        axis.text(-0.08, 1.02, f"({chr(97 + index)})", transform=axis.transAxes,
                  fontweight="bold", va="bottom")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.8,
                              color="white" if value < 0.45 else "black")
    axes[-1].set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    fig.colorbar(images[0], ax=list(axes), fraction=0.018, pad=0.014,
                 label="Maximum absolute change from baseline")
    fig.suptitle("One-Factor-at-a-Time Parameter Sensitivity")
    return save_figure(fig, "Relay_Boundary_Flow_Parameter_Sensitivity", config)


def plot_pi_structure_sensitivity(
    sensitivity: pd.DataFrame, config: AnalysisConfig,
) -> tuple[Path, Path]:
    conditions = (
        "Circular", "Circular_Single_Defect_H3", "Square", "Square_Four_Defects_H3"
    )
    fig, axis = plt.subplots(figsize=(7.3, 3.9), constrained_layout=True)
    for index, condition in enumerate(conditions):
        subset = sensitivity[sensitivity.condition.eq(condition)]
        baseline = float(subset[subset.factor.eq("Baseline")].bidirectional_peak_floor.iloc[0])
        low = float(subset.bidirectional_peak_floor.min())
        high = float(subset.bidirectional_peak_floor.max())
        axis.errorbar(
            index, baseline, yerr=[[baseline - low], [high - baseline]],
            color=COLORS[condition], marker=MARKERS[condition], ms=6, capsize=4, lw=1.4,
        )
    axis.set_xticks(range(len(conditions)), [CASE_LABELS[item] for item in conditions], rotation=17, ha="right")
    axis.set_ylabel("Weaker propagation-sector peak fraction")
    axis.set_ylim(bottom=0)
    axis.grid(True, axis="y", color="#D9DEE4", lw=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(r"$\alpha=\pi$: Structure-Spectrum Resolution Sensitivity")
    return save_figure(fig, "Pi_State_Structure_Sensitivity", config)


def plot_gate_fields(details: dict[tuple[str, float], dict], config: AnalysisConfig) -> tuple[Path, Path]:
    selections = (
        ("Circular_Single_Defect_H3", 0.4, "Relay across a defect"),
        ("Circular_Single_Defect_H3", 0.6, "Pattern-supported boundary flow"),
        ("Square_Four_Defects_H3", 0.0, "Sparse defect-side motion"),
        ("Square", 0.4, "Square boundary reference"),
    )
    fig, axes = plt.subplots(1, len(selections), figsize=(13.0, 3.8), constrained_layout=True, sharey=True)
    image = None
    for panel, (axis, (condition, alpha, title)) in enumerate(zip(axes, selections)):
        field: CrossingData = details[(condition, alpha)]["field"]
        masked = np.ma.masked_where(~field.active, field.normalized_handedness)
        image = axis.imshow(masked, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1,
                            extent=(0, 1, 0, config.time_block_count))
        axis.set_title(title, fontsize=9.5)
        axis.set_xlabel(r"Boundary arclength $\xi/L_{\partial}$")
        axis.set_xlim(0, 1)
        axis.text(-0.10, 1.02, f"({chr(97 + panel)})",
                  transform=axis.transAxes, fontweight="bold", va="bottom")
    axes[0].set_ylabel("Terminal-window time block")
    fig.colorbar(image, ax=list(axes), fraction=0.018, pad=0.014,
                 label=r"Reliable node handedness (CW $<0<$ CCW)")
    fig.suptitle("Same-Sign Reliable Gate-Crossing Nodes")
    return save_figure(fig, "Relay_Boundary_Flow_Gate_Fields", config)


def plot_pi_directional_crossings(
    details: dict[tuple[str, float], dict], config: AnalysisConfig,
) -> tuple[Path, Path]:
    conditions = (
        "Circular", "Circular_Single_Defect_H3", "Square", "Square_Four_Defects_H3"
    )
    fig, axes = plt.subplots(2, len(conditions), figsize=(13.0, 5.6), constrained_layout=True,
                             sharex=True, sharey=True)
    for column, condition in enumerate(conditions):
        field: CrossingData = details[(condition, 1.0)]["field"]
        for row, (values, direction) in enumerate(
            ((field.interval_gate_ccw, "CCW"), (field.interval_gate_cw, "CW"))
        ):
            axes[row, column].imshow(
                values,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="Greys",
                vmin=0,
                vmax=1,
                extent=(0, 1, 0, config.analysis_window_time),
            )
            axes[row, column].set_xlabel(r"Boundary arclength $\xi/L_{\partial}$")
            if column == 0:
                axes[row, column].set_ylabel(f"Time ({direction})")
        axes[0, column].set_title(CASE_LABELS[condition], fontsize=9.2)
        axes[0, column].text(
            -0.10, 1.02, f"({chr(97 + column)})", transform=axes[0, column].transAxes,
            fontweight="bold", va="bottom",
        )
        axes[1, column].text(
            -0.10, 1.02, f"({chr(101 + column)})", transform=axes[1, column].transAxes,
            fontweight="bold", va="bottom",
        )
    fig.suptitle(r"$\alpha=\pi$: Actual Direction-Resolved Gate Crossings")
    return save_figure(fig, "Pi_State_Direction_Resolved_Gate_Crossings", config)


def plot_pi_structure(
    table: pd.DataFrame, details: dict[tuple[str, float], dict], config: AnalysisConfig,
) -> tuple[Path, Path]:
    conditions = (
        "Circular",
        "Circular_Single_Defect_H3",
        "Square",
        "Square_Four_Defects_H3",
    )
    fig, axes = plt.subplots(2, len(conditions), figsize=(13.2, 6.4), constrained_layout=True)
    image = None
    for column, condition in enumerate(conditions):
        detail = details[(condition, 1.0)]
        density = detail["density"]
        structure = detail["structure"]
        fluctuation = density - density.mean(axis=0, keepdims=True)
        scale = max(float(np.percentile(np.abs(fluctuation), 98)), EPS)
        axes[0, column].imshow(
            fluctuation, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale,
            extent=(0, 1, 0, config.analysis_window_time),
        )
        axes[0, column].set_title(CASE_LABELS[condition], fontsize=9.3)
        axes[0, column].set_xlabel(r"Boundary arclength $\xi/L_{\partial}$")
        axes[0, column].text(-0.10, 1.02, f"({chr(97 + column)})",
                             transform=axes[0, column].transAxes, fontweight="bold", va="bottom")
        power = np.asarray(structure["power"], dtype=float)
        log_power = np.log10(power / max(float(power.max()), EPS) + 1e-8)
        image = axes[1, column].imshow(
            log_power,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-5,
            vmax=0,
            extent=(
                float(structure["modes"][0]),
                float(structure["modes"][-1]),
                float(structure["frequencies"][0]),
                float(structure["frequencies"][-1]),
            ),
        )
        axes[1, column].set_xlabel("Boundary mode m")
        axes[1, column].axhline(0, color="white", lw=0.45, alpha=0.65)
        axes[1, column].axvline(0, color="white", lw=0.45, alpha=0.65)
        item = table[
            table.condition.eq(condition) & np.isclose(table.alpha_over_pi, 1.0)
        ].iloc[0]
        axes[1, column].text(
            0.03, 0.04,
            rf"$Q_+={item.pi_structure_positive_sector_peak_pair_fraction:.3f}$" + "\n"
            + rf"$Q_-={item.pi_structure_negative_sector_peak_pair_fraction:.3f}$",
            transform=axes[1, column].transAxes, color="white", fontsize=7.5,
            ha="left", va="bottom",
        )
        axes[1, column].text(-0.10, 1.02, f"({chr(101 + column)})",
                             transform=axes[1, column].transAxes, fontweight="bold", va="bottom")
    axes[0, 0].set_ylabel("Terminal-window time")
    axes[1, 0].set_ylabel("Frequency (cycles/time)")
    fig.colorbar(image, ax=list(axes[1]), fraction=0.018, pad=0.014,
                 label=r"$\log_{10}[S(m,f)/S_{\max}]$")
    fig.suptitle(r"$\alpha=\pi$: Boundary Density and Dynamic Structure")
    return save_figure(fig, "Pi_State_Dynamic_Boundary_Structure", config)


def write_method_definition(config: AnalysisConfig) -> Path:
    """Write an English, symbol-complete audit trail; this is not a PDF."""
    path = versioned_path(config.output_dir / "Relay_Boundary_Flow_Definitions.txt")
    text = f"""RELAY BOUNDARY FLOW: DISCRETE DEFINITIONS
=========================================

Stored quantities and non-conflicting symbols
----------------------------------------------
X_i^n=(x_i^n,y_i^n) is particle i in HDF5 positionX at saved frame n, and
Theta_i^n is the corresponding phaseTheta value.  Delta t_s=dt*shotsnaps.
The interaction radius is r_int (model distanceD0), the reconstructed boundary
perimeter is L_partial, the number of gates is N_g={config.boundary_gate_count},
and the number of time blocks is N_b={config.time_block_count}.

Boundary projection and accepted episodes
-----------------------------------------
p_i^n=argmin_(p on boundary)||X_i^n-p||, d_i^n=||X_i^n-p_i^n||, and xi_i^n in
[0,L_partial) is counterclockwise arclength of p_i^n.  The fixed geometric edge
layer is w_e={config.edge_layer_over_d0:g}r_int and the wall-contact anchor is
w_c={config.contact_width_over_d0:g}r_int.  A maximal episode E_iq in w_e is
accepted only if its duration is at least {config.minimum_episode_time:g},
min_(n in E_iq)d_i^n<=w_c, and the actual wrapped arclength steps

delta ell_i^n=wrap_[-L_partial/2,L_partial/2)(xi_i^(n+1)-xi_i^n)

pass |delta ell_i^n|<={config.projection_jump_step_factor:g}v Delta t_s.  On the
remaining episode intervals require

|sum_n delta ell_i^n|/sum_n|delta ell_i^n|
    >= {config.minimum_episode_directional_persistence:g},
mean_n |delta ell_i^n|/(v Delta t_s)
    >= {config.minimum_episode_tangentiality:g}.

Binary interval evidence and reliable spacetime nodes
-----------------------------------------------------
Split L_partial into N_g gates and the terminal window into N_b ordered blocks.
E_bk^+ (E_bk^-) is the number of distinct saved intervals in block b containing
at least one accepted CCW (CW) crossing of gate k.  Each interval contributes
only presence/absence, regardless of how many particles cross.  Let

m_b=max({config.minimum_interval_recurrence},
        ceil({config.minimum_interval_recurrence_fraction:g}|I_b|)),

where |I_b| is the number of saved intervals in block b.  A sign-s node exists,
Z_bk^s=1, only if E_bk^s>=m_b,
E_bk^s/(E_bk^++E_bk^-) >= {config.minimum_directional_purity:g}, and at least
{config.minimum_particles_per_gate} distinct particles support that sign at the
gate.  A block is supported only if it has at least {config.minimum_active_gates}
reliable signed gates and at least {config.minimum_particles_per_block} distinct
supporting particles.  These are Boolean evidence gates; their counts never
multiply either headline score.

Headline metric 1: same-sign connected relay coverage
-----------------------------------------------------
Connect (b,k,s) to (b+1,k',s) only when both reliable nodes have the same sign
and [s(k'-k)] mod N_g <= a_max, where

a_max=ceil(v Delta T_b/(L_partial/N_g))+1.

For each resulting spacetime component C, define

tau_C=|{{b:(b,k) in C}}|/N_b,
c_C=|{{k:(b,k) in C}}|/N_g.

The first headline metric is

Xi_relay=max_(s,C) tau_C c_C.

It is invariant to carrier identity and contains neither crossing number nor
carrier-population fraction.  Particle labels are inspected only after scoring:
"one particle crosses in every component block" does not imply continuous wall
residence or a complete circuit and is only an exclusionary relay diagnostic.

Headline metric 2: handedness fixation
---------------------------------------
In each supported block, set chi_b=+1 if at least
{config.minimum_directional_purity:g} of reliable nodes are CCW, chi_b=-1 if at
least that fraction are CW, and chi_b=0 otherwise.  Let S be the supported-block
set.  If |S|/N_b<{config.minimum_supported_block_fraction:g}, Xi_fix is NA.
Otherwise

Xi_fix=|(1/|S|)sum_(b in S)chi_b|.

Thus ambiguous bidirectional blocks and temporal handedness reversals both lower
Xi_fix, while insufficient time support is never reported as an artificially
high fixedness.  High Xi_relay and high Xi_fix jointly identify persistent,
perimeter-covering, single-handed relay transport.

Alpha=pi structural validation
------------------------------
rho_nk=sum_i 1[d_i^n<=w_e]1[xi_i^n lies in gate k].  Double demean it as
delta rho_nk=rho_nk-mean_n rho_nk-mean_k rho_nk+mean_(n,k)rho_nk and compute

S(m,f)=|sum_(n,k) delta rho_nk w_n
        exp[-2pi i(mk/N_g+f n Delta t_s)]|^2,

with Hann taper w_n.  The two non-conjugate propagation sectors mf>0 and mf<0
are reported separately: each sector score is the sum of its two largest
conjugate-peak powers divided by total nonzero-mode, nonzero-frequency power.
Both sector peaks plus simultaneous signed CCW/CW crossing maps support a
bidirectional traveling boundary lattice; density S(m,f) alone cannot exclude a
standing wave.  This is an alpha=pi structure check, not a third sweep metric.

Null and interpretation limits
------------------------------
The causal-connectivity null independently circularly shifts every block's gate
field, preserving each block's signed morphology and support while breaking
inter-block alignment.  All supplied trajectories use seed=9.  The common final
10-time-unit window is only about one free perimeter transit, and terminal saved
frames differ across files.  Results are therefore single-realization,
terminal-window descriptions rather than ensemble-level long-time statistics.
"""
    path.write_text(text, encoding="utf-8")
    return path


def synthetic_control_failures() -> list[str]:
    """Exercise causal connectivity against deterministic negative controls."""
    block_count, gate_count = 20, 64

    def score(node_sign: np.ndarray, maximum_advance: int) -> float:
        synthetic = SimpleNamespace(
            node_sign=node_sign,
            supported=np.ones(block_count, dtype=bool),
            maximum_gate_advance=maximum_advance,
        )
        return max(
            (
                _component_score(component, block_count, gate_count)
                for sign in (-1, 1)
                for component in _component_nodes(synthetic, sign)
            ),
            default=0.0,
        )

    failures: list[str] = []
    coherent = np.ones((block_count, gate_count), dtype=np.int8)
    if not np.isclose(score(coherent, 2), 1.0):
        failures.append("A fully coherent synthetic flow did not score one.")

    disconnected = np.zeros((block_count, gate_count), dtype=np.int8)
    for block in range(block_count):
        disconnected[block, (20 * block + np.array([0, 21, 42])) % gate_count] = 1
    if score(disconnected, 2) > 0.10:
        failures.append("Spatially disconnected synthetic crossings formed a false relay.")

    alternating = np.ones((block_count, gate_count), dtype=np.int8)
    alternating[1::2] = -1
    if score(alternating, 2) > 0.10:
        failures.append("Blockwise handedness reversals formed a false long relay.")

    gate_tests = (
        (0.9, 0.2, 8.0, 8, [1]),
        (1.1, -0.2, 8.0, 8, [1]),
        (7.9, 0.2, 8.0, 8, [0]),
        (0.1, -0.2, 8.0, 8, [0]),
        (1.0, 0.2, 8.0, 8, []),
        (1.0, -0.2, 8.0, 8, []),
    )
    if any(_crossed_gates(*test[:4]) != test[4] for test in gate_tests):
        failures.append("The half-open discrete gate-crossing convention failed.")
    return failures


def validate_results(
    table: pd.DataFrame,
    support_sensitivity: pd.DataFrame,
    parameter_sensitivity: pd.DataFrame | None = None,
    structure_sensitivity: pd.DataFrame | None = None,
) -> list[str]:
    """Return only data-contract, range, and synthetic-control failures."""
    failures: list[str] = []
    if len(table) != len(CASE_SPECS) * len(ALPHAS):
        failures.append("The exact-data sweep is incomplete.")
    if not table.xi_relay.between(0, 1).all():
        failures.append("Xi_relay left its [0,1] range.")
    finite_fix = table.xi_fix.dropna()
    if not finite_fix.between(0, 1).all():
        failures.append("Xi_fix left its [0,1] range.")
    for name, audit in (
        ("support sensitivity", support_sensitivity),
        ("parameter sensitivity", parameter_sensitivity),
    ):
        if audit is not None and not audit.empty:
            if not audit.xi_relay.between(0, 1).all():
                failures.append(f"Xi_relay left [0,1] in the {name} table.")
            finite = audit.xi_fix.dropna()
            if not finite.between(0, 1).all():
                failures.append(f"Xi_fix left [0,1] in the {name} table.")
    if structure_sensitivity is not None and not structure_sensitivity.empty:
        structural = structure_sensitivity[
            ["positive_sector_peak_pair_fraction", "negative_sector_peak_pair_fraction"]
        ]
        if not np.isfinite(structural.to_numpy()).all() or (structural.to_numpy() < 0).any():
            failures.append("The alpha=pi structure sensitivity table is invalid.")
    pi_rows = table[
        table.condition.isin(["Circular", "Square"])
        & np.isclose(table.alpha_over_pi, 1.0)
    ]
    if (
        len(pi_rows) != 2
        or not np.isfinite(
            pi_rows.pi_structure_positive_sector_peak_pair_fraction
        ).all()
        or not np.isfinite(
            pi_rows.pi_structure_negative_sector_peak_pair_fraction
        ).all()
    ):
        failures.append("The alpha=pi dynamic-structure diagnostic is missing.")
    failures.extend(synthetic_control_failures())
    return failures


def scientific_warnings(table: pd.DataFrame) -> list[str]:
    """Report observations without converting a preferred story into validation."""
    warnings = [
        "All trajectories use randomSeed=9; these are single-realization descriptive results.",
        "The common 10-time-unit terminal window is only about one free perimeter transit.",
        "Terminal saved-frame indices differ across HDF5 files; only each file's actual terminal window is compared.",
    ]
    circular_pi = table[
        table.condition.isin(["Circular", "Circular_Single_Defect_H3"])
        & np.isclose(table.alpha_over_pi, 1.0)
    ].pi_structure_bidirectional_peak_floor
    square_pi = table[
        table.condition.isin(["Square", "Square_Four_Defects_H3"])
        & np.isclose(table.alpha_over_pi, 1.0)
    ].pi_structure_bidirectional_peak_floor
    if len(circular_pi) == 2 and len(square_pi) == 2:
        if float(circular_pi.min()) <= float(square_pi.max()):
            warnings.append(
                "The alpha=pi direction-resolved structure spectrum does not separate circular and square states."
            )
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Compute and print validation results without writing output files.",
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="Skip all sensitivity grids during exploratory runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_plotting()
    table, details = compute_all(CONFIG)
    if args.skip_sensitivity:
        support_sensitivity = pd.DataFrame()
        parameter_sensitivity = pd.DataFrame()
        structure_sensitivity = pd.DataFrame()
    else:
        support_sensitivity = compute_support_sensitivity(CONFIG)
        parameter_sensitivity = compute_parameter_sensitivity(CONFIG)
        structure_sensitivity = compute_pi_structure_sensitivity(CONFIG)
    failures = validate_results(
        table, support_sensitivity, parameter_sensitivity, structure_sensitivity
    )
    columns = [
        "condition", "alpha_over_pi", "xi_relay", "xi_fix",
        "supported_block_count", "relay_component_block_count",
        "relay_component_gate_count", "one_particle_crosses_in_every_component_block",
        "maximum_single_particle_component_block_fraction",
        "pi_structure_positive_sector_peak_pair_fraction",
        "pi_structure_negative_sector_peak_pair_fraction",
    ]
    print(table[columns].to_string(index=False))
    if failures:
        print("\nSTOP: validation failed; no figures or Methods text were written.")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    warnings = scientific_warnings(table)
    print("\nScientific limitations / warnings:")
    for warning in warnings:
        print(f"  - {warning}")
    if args.validate_only:
        print("\nValidation passed.  No output files were written.")
        return 0

    table_paths = write_tables(
        table,
        support_sensitivity,
        parameter_sensitivity,
        structure_sensitivity,
        CONFIG,
    )
    outputs = [
        *plot_alpha_sweep(table, CONFIG),
        *plot_support_sensitivity(support_sensitivity, CONFIG),
        *plot_parameter_sensitivity(parameter_sensitivity, CONFIG),
        *plot_pi_structure_sensitivity(structure_sensitivity, CONFIG),
        *plot_gate_fields(details, CONFIG),
        *plot_pi_directional_crossings(details, CONFIG),
        *plot_pi_structure(table, details, CONFIG),
        write_method_definition(CONFIG),
        *table_paths,
    ]
    for output in outputs:
        print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
