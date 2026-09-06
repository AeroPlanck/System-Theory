"""Phase-informed, relay-compatible boundary-flow analysis.

Only existing HDF5 trajectories are read.  The script never runs, continues,
or overwrites a simulation.  Exactly two headline observables are reported:

``Xi_Persist``
    Recurrence duty cycle: the fraction of terminal-window blocks containing
    a calibrated directional boundary signal, after requiring a connected
    active boundary length of at least 2*d0 over the observation window.

``Xi_Sign``
    Long-window net chirality: the absolute mean of geometrically normalized
    block chirality over active blocks.

The singular alpha=0 endpoint and the separate bidirectional alpha=pi boundary-
lattice-flow problem are excluded from these single-chirality measures.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import boundary_defect_analysis as bda
import main as model_library


ROOT = Path(__file__).resolve().parent


# =============================================================================
# USER-ADJUSTABLE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AnalysisConfig:
    """Pre-registered numerical and rendering controls."""

    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "output" / "Phase_Informed_Boundary_Flow_Refined"

    # Existing exact-file sweep and requested refined grid.  The latter is
    # validation-only until every parameter-matched HDF5 file exists.
    alpha_over_pi: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    refined_alpha_over_pi: tuple[float, ...] = tuple(
        round(value, 1) for value in np.linspace(0.0, 1.0, 11)
    )
    terminal_circuit_times: float = 10.0

    # Boundary discretization and short physical time block.
    gate_length_over_d0: float = 0.50
    block_time_over_d0_over_v: float = 1.00
    macro_extent_over_d0: float = 2.00

    # Phase/kinematic-calibration quality gates.  q0 is a pre-registered
    # directional-resolution dead band, not a tangential-dominance threshold.
    minimum_phase_projection: float = 0.05
    minimum_displacement_over_v_dt: float = 0.02
    maximum_projection_step_factor: float = 1.05
    boundary_layer_over_d0: float = 1.00
    join_exclusion_over_d0: float = 0.05
    projection_tie_over_d0: float = 1.0e-7

    # Displacement is a direction-neutral, full-window kinematic calibration.
    # Its sample count never weights a block or a headline observable.
    minimum_calibration_trials: int = 3
    minimum_calibration_agreement: float = 2.0 / 3.0

    # Conditional phase-direction coherence within one short block.
    minimum_directional_purity: float = 2.0 / 3.0
    minimum_active_blocks_for_sign: int = 3
    minimum_support_pairs: int = 1
    support_pair_sensitivity: tuple[int, ...] = (1, 2, 3)

    # High-alpha wall-requirement filter W > epsilon_W.
    w_dead_zone_over_v_over_d0: float = 0.02
    validate_optimized_phase_rhs: bool = True
    phase_rhs_validation_tolerance: float = 5.0e-10

    # Pre-registered robustness check; it never changes the main-table q0.
    q0_sensitivity: tuple[float, ...] = (0.02, 0.05, 0.10)

    figure_dpi: int = 450


CONFIG = AnalysisConfig()


@dataclass(frozen=True)
class CaseSpec:
    group: str
    condition: str
    label: str
    defect_height: float = 0.0


# Only the four geometries used in the boundary/defect comparison are retained.
CASE_SPECS = (
    CaseSpec("Circular", "Circular", "circle"),
    CaseSpec("Circular", "Circular_Single_Defect_H3", "circle_defect", 3.0),
    CaseSpec("Square", "Square", "square"),
    CaseSpec("Square", "Square_Four_Symmetric_Defects_H1", "square_defect", 1.0),
)


CASE_LABELS = {
    "Circular": "Circular",
    "Circular_Single_Defect_H3": "Circular, single defect (H = 3)",
    "Square": "Square",
    "Square_Four_Symmetric_Defects_H1": "Square, four symmetric defects (H = 1)",
}


COLORS = {
    "Circular": "#276678",
    "Circular_Single_Defect_H3": "#7A5195",
    "Square": "#3A7D44",
    "Square_Four_Symmetric_Defects_H1": "#9A6B28",
}


MARKERS = {
    "Circular": "o",
    "Circular_Single_Defect_H3": "^",
    "Square": "s",
    "Square_Four_Symmetric_Defects_H1": "D",
}


# =============================================================================
# GEOMETRY AND BASIC DATA
# =============================================================================


class Projection(NamedTuple):
    distance: np.ndarray
    arc: np.ndarray
    tangent: np.ndarray
    curvature: np.ndarray
    component: np.ndarray
    valid: np.ndarray


class FrameState(NamedTuple):
    projection: Projection
    q: np.ndarray
    preselection: np.ndarray
    wall_required: np.ndarray


class GateScheme(NamedTuple):
    component_starts: np.ndarray
    component_lengths: np.ndarray
    component_regular_offsets: np.ndarray
    component_regular_lengths: np.ndarray
    component_gate_counts: np.ndarray
    component_gate_lengths: np.ndarray
    component_gate_offsets: np.ndarray
    gate_lengths: np.ndarray
    regular_perimeter: float
    omitted_perimeter_fraction: float


class AnalysisDetail(NamedTuple):
    block_gate_sign: np.ndarray
    block_chirality: np.ndarray
    block_active: np.ndarray
    gate_lengths: np.ndarray
    maximum_contiguous_extent: float
    extent_valid: bool
    calibrated_gate_count: int
    calibrated_perimeter_fraction: float
    block_active_arclength_fraction: np.ndarray
    mean_active_arclength_fraction: float
    active_block_mean_arclength_fraction: float
    preselection_grid_coverage: float
    selected_grid_coverage: float
    w_selection_retention: float
    representative_wall_distance_quantiles_over_d0: np.ndarray
    maximum_contiguous_extent_no_cross_join: float
    extent_valid_no_cross_join: bool
    block_activity_lag1_correlation: float
    block_chirality_lag1_correlation: float
    omitted_perimeter_fraction: float
    support_pair_thresholds: np.ndarray
    support_pair_xi_persist: np.ndarray
    support_pair_xi_sign: np.ndarray
    calibration_trial_count: int
    calibration_success_count: int
    calibration_agreement: float
    phase_rhs_validation_error: float


def build_gate_scheme(
    geometry: bda.BoundaryGeometry,
    distance_d0: float,
    config: AnalysisConfig,
) -> GateScheme:
    """Split every retained regular boundary component independently at joins."""
    if geometry.kind == "circle":
        component_lengths = np.array((geometry.perimeter,), dtype=float)
    elif geometry.kind == "polygon":
        edges = np.roll(geometry.vertices, -1, axis=0) - geometry.vertices
        component_lengths = np.linalg.norm(edges, axis=1)
    elif geometry.kind == "midpoint_circle":
        arc_length = geometry.radius * (
            2.0 * np.pi - 2.0 * geometry.midpoint_half_angle
        )
        side_length = 0.5 * (geometry.perimeter - arc_length)
        component_lengths = np.array(
            (arc_length, side_length, side_length), dtype=float
        )
    else:
        raise TypeError(f"Unsupported geometry kind: {geometry.kind}")
    component_starts = np.concatenate(
        (np.array((0.0,)), np.cumsum(component_lengths[:-1]))
    )
    if geometry.kind == "circle":
        component_regular_offsets = np.zeros_like(component_lengths)
    else:
        requested_exclusion = config.join_exclusion_over_d0 * distance_d0
        component_regular_offsets = np.minimum(
            requested_exclusion,
            np.nextafter(0.5 * component_lengths, 0.0),
        )
    component_regular_lengths = component_lengths - 2.0 * component_regular_offsets
    if np.any(component_regular_lengths <= 0.0):
        raise RuntimeError("Join exclusions remove a complete boundary component.")
    target = config.gate_length_over_d0 * distance_d0
    component_gate_counts = np.maximum(
        1, np.ceil(component_regular_lengths / target)
    ).astype(int)
    component_gate_lengths = component_regular_lengths / component_gate_counts
    component_gate_offsets = np.concatenate(
        (np.array((0,)), np.cumsum(component_gate_counts[:-1]))
    ).astype(int)
    gate_lengths = np.concatenate(
        [
            np.full(count, length, dtype=float)
            for count, length in zip(component_gate_counts, component_gate_lengths)
        ]
    )
    regular_perimeter = float(np.sum(component_regular_lengths))
    if not np.isclose(gate_lengths.sum(), regular_perimeter):
        raise RuntimeError("Grid lengths do not reproduce the retained perimeter.")
    return GateScheme(
        component_starts,
        component_lengths,
        component_regular_offsets,
        component_regular_lengths,
        component_gate_counts,
        component_gate_lengths,
        component_gate_offsets,
        gate_lengths,
        regular_perimeter,
        float(1.0 - regular_perimeter / geometry.perimeter),
    )


def build_model(spec: CaseSpec, alpha_over_pi: float):
    """Instantiate the exact model signature used to name existing HDF5 data."""
    if spec.label == "circle":
        cls = model_library.CircularBoundaryPatternFormation
        extra = {}
    elif spec.label == "circle_defect":
        cls = model_library.CollisionBoundaryMidpointSpikePatternFormation
        extra = {
            "protrusionHeight": spec.defect_height,
            "protrusionHalfWidth": bda.SPIKE_HALF_WIDTH,
        }
    elif spec.label == "square":
        cls = model_library.CollisionBoundaryPatternFormation
        extra = {}
    elif spec.label == "square_defect":
        cls = model_library.CollisionBoundaryFourSpikePatternFormation
        extra = {
            "protrusionHeight": spec.defect_height,
            "protrusionHalfWidth": bda.SPIKE_HALF_WIDTH,
        }
    else:
        raise ValueError(f"Unsupported case label: {spec.label}")
    return bda.build_model(cls, alpha_over_pi, **extra)


def all_jobs(config: AnalysisConfig, alpha_grid: tuple[float, ...] | None = None):
    alpha_values = config.alpha_over_pi if alpha_grid is None else alpha_grid
    return [
        (spec, alpha, build_model(spec, alpha))
        for spec in CASE_SPECS
        for alpha in alpha_values
    ]


def _segment_candidates(points: np.ndarray, vertices: np.ndarray):
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    relative = points[:, None, :] - vertices[None, :, :]
    fractions = np.einsum("nkj,kj->nk", relative, edges) / np.square(lengths)
    fractions = np.clip(fractions, 0.0, 1.0)
    projected = vertices[None, :, :] + fractions[..., None] * edges[None, :, :]
    residual = points[:, None, :] - projected
    distances = np.linalg.norm(residual, axis=2)
    arcs = cumulative[None, :] + fractions * lengths[None, :]
    tangents = edges / lengths[:, None]
    return distances, arcs, tangents, fractions, lengths


def _choose_component(
    distances: np.ndarray,
    arcs: np.ndarray,
    tangents: np.ndarray,
    curvatures: np.ndarray,
    local_valid: np.ndarray,
    tie_distance: float,
) -> Projection:
    count = distances.shape[0]
    best = np.argmin(distances, axis=1)
    rows = np.arange(count)
    sorted_distance = np.partition(distances, 1, axis=1)
    unique = (sorted_distance[:, 1] - sorted_distance[:, 0]) > tie_distance
    tangent = tangents[best] if tangents.ndim == 2 else tangents[rows, best]
    return Projection(
        distance=distances[rows, best],
        arc=arcs[rows, best],
        tangent=tangent,
        curvature=curvatures[best],
        component=best.astype(np.int16),
        valid=unique & local_valid[rows, best],
    )


def project_points(
    points: np.ndarray,
    geometry: bda.BoundaryGeometry,
    distance_d0: float,
    config: AnalysisConfig,
) -> Projection:
    """Return the unique regular closest-boundary coordinate of every point."""
    join_exclusion = config.join_exclusion_over_d0 * distance_d0
    tie_distance = config.projection_tie_over_d0 * distance_d0

    if geometry.kind == "circle":
        relative = points - geometry.center
        radius_now = np.linalg.norm(relative, axis=1)
        angle = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
        tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
        return Projection(
            distance=np.abs(geometry.radius - radius_now),
            arc=geometry.radius * angle,
            tangent=tangent,
            curvature=np.full(points.shape[0], 1.0 / geometry.radius),
            component=np.zeros(points.shape[0], dtype=np.int16),
            valid=radius_now > tie_distance,
        )

    if geometry.kind == "polygon":
        distances, arcs, tangents, fractions, lengths = _segment_candidates(
            points, geometry.vertices
        )
        endpoint_distance = np.minimum(fractions, 1.0 - fractions) * lengths[None, :]
        local_valid = endpoint_distance > join_exclusion
        return _choose_component(
            distances,
            arcs,
            tangents,
            np.zeros(lengths.size),
            local_valid,
            tie_distance,
        )

    if geometry.kind != "midpoint_circle":
        raise TypeError(f"Unsupported geometry kind: {geometry.kind}")

    center = geometry.center
    radius = float(geometry.radius)
    half_angle = float(geometry.midpoint_half_angle)
    start_angle = -0.5 * np.pi + half_angle
    end_angle = 1.5 * np.pi - half_angle
    relative = points - center
    raw = np.arctan2(relative[:, 1], relative[:, 0])
    raw = np.where(raw < start_angle, raw + 2.0 * np.pi, raw)
    angle = np.clip(raw, start_angle, end_angle)
    arc_projection = center + radius * np.column_stack((np.cos(angle), np.sin(angle)))
    arc_distance = np.linalg.norm(points - arc_projection, axis=1)
    arc_length = radius * (end_angle - start_angle)
    arc_s = radius * (angle - start_angle)
    arc_tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
    arc_local_valid = (
        (radius * (raw - start_angle) > join_exclusion)
        & (radius * (end_angle - raw) > join_exclusion)
    )

    base_right = center + radius * np.array((np.cos(start_angle), np.sin(start_angle)))
    base_left = center + radius * np.array((np.cos(end_angle), np.sin(end_angle)))
    tip = np.array((center[0], center[1] - radius + geometry.spike_height))
    vertices = np.vstack((base_left, tip, base_right))
    straight_distances, straight_arcs, straight_tangents, fractions, lengths = (
        _segment_candidates(points, vertices)
    )
    # _segment_candidates closes the three-point chain; discard its artificial
    # base-right -> base-left chord and retain only the two spike sides.
    straight_distances = straight_distances[:, :2]
    straight_arcs = straight_arcs[:, :2] + arc_length
    straight_tangents = straight_tangents[:2]
    fractions = fractions[:, :2]
    lengths = lengths[:2]
    straight_local_valid = (
        np.minimum(fractions, 1.0 - fractions) * lengths[None, :] > join_exclusion
    )

    distances = np.column_stack((arc_distance, straight_distances))
    arcs = np.column_stack((arc_s, straight_arcs))
    tangents = np.empty((points.shape[0], 3, 2), dtype=float)
    tangents[:, 0] = arc_tangent
    tangents[:, 1:] = straight_tangents[None, :, :]
    local_valid = np.column_stack((arc_local_valid, straight_local_valid))
    return _choose_component(
        distances,
        arcs,
        tangents,
        np.array((1.0 / radius, 0.0, 0.0)),
        local_valid,
        tie_distance,
    )


# =============================================================================
# PHASE-INFORMED EULERIAN CROSSING EVENTS
# =============================================================================


def wrap_arclength(delta: np.ndarray, perimeter: float) -> np.ndarray:
    return np.mod(delta + 0.5 * perimeter, perimeter) - 0.5 * perimeter


def free_phase_rhs_for_indices(
    positions: np.ndarray,
    phases: np.ndarray,
    model,
    indices: np.ndarray,
) -> np.ndarray:
    """Evaluate main.py's collision-boundary phase RHS for selected particles."""
    output = np.full(phases.size, np.nan)
    if indices.size == 0:
        return output
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=model.distanceD0, output_type="ndarray")
    radius2 = float(model.distanceD0 * model.distanceD0)
    phase_sum = np.zeros(phases.size, dtype=float)
    neighbor_count = np.zeros(phases.size, dtype=np.int64)
    if pairs.size:
        delta_x = positions[pairs[:, 1]] - positions[pairs[:, 0]]
        distance2 = np.einsum("ij,ij->i", delta_x, delta_x)
        pairs = pairs[(distance2 > 0.0) & (distance2 <= radius2)]
        if pairs.size:
            left = pairs[:, 0]
            right = pairs[:, 1]
            alpha = model.phaseLagA0
            phase_sum += np.bincount(
                left,
                weights=np.sin(phases[right] - phases[left] + alpha),
                minlength=phases.size,
            )
            phase_sum += np.bincount(
                right,
                weights=np.sin(phases[left] - phases[right] + alpha),
                minlength=phases.size,
            )
            neighbor_count += np.bincount(left, minlength=phases.size)
            neighbor_count += np.bincount(right, minlength=phases.size)

    rates = model.freqOmega.copy()
    populated = neighbor_count > 0
    rates[populated] += model.strengthK * (
        phase_sum[populated] / neighbor_count[populated]
        - math.sin(model.phaseLagA0)
    )
    output[indices] = rates[indices]
    return output


def prepare_frame(
    positions: np.ndarray,
    phases: np.ndarray,
    model,
    geometry: bda.BoundaryGeometry,
    high_alpha: bool,
    config: AnalysisConfig,
) -> FrameState:
    projection = project_points(positions, geometry, model.distanceD0, config)
    heading = np.column_stack((np.cos(phases), np.sin(phases)))
    q = np.einsum("ij,ij->i", heading, projection.tangent)
    base = projection.valid & (np.abs(q) >= config.minimum_phase_projection)
    if not high_alpha:
        return FrameState(projection, q, base, base)

    base &= projection.distance <= config.boundary_layer_over_d0 * model.distanceD0
    denominator = 1.0 - projection.curvature * projection.distance
    regular_parallel = denominator > 0.0
    preselection = base & regular_parallel
    indices = np.flatnonzero(preselection)
    omega = free_phase_rhs_for_indices(positions, phases, model, indices)
    kappa_d = np.zeros_like(q)
    kappa_d[regular_parallel] = (
        projection.curvature[regular_parallel] / denominator[regular_parallel]
    )
    wall_deficit = kappa_d * model.speedV * np.square(q) - omega * q
    tolerance = config.w_dead_zone_over_v_over_d0 * model.speedV / model.distanceD0
    wall_required = preselection & (wall_deficit > tolerance)
    return FrameState(projection, q, preselection, wall_required)


def validate_phase_rhs(
    positions: np.ndarray,
    phases: np.ndarray,
    model,
    config: AnalysisConfig,
) -> float:
    """Cross-check the optimized evaluator against main.py on one saved frame."""
    indices = np.arange(phases.size)
    optimized = free_phase_rhs_for_indices(positions, phases, model, indices)
    reference = model._calc_dot_phase_collision(
        positionX=positions,
        phaseTheta=phases,
        freqOmega=model.freqOmega,
        params=model.dotThetaParams,
    )
    error = float(np.max(np.abs(optimized - reference)))
    if error > config.phase_rhs_validation_tolerance:
        raise RuntimeError(
            "Optimized phase RHS does not reproduce main.py: "
            f"max error={error:.3e}.  No result was written."
        )
    return error


def frame_gate_representatives(
    state: FrameState,
    scheme: GateScheme,
    distance_d0: float,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one nearest-wall phase representative in each arclength cell."""
    gate_count = scheme.gate_lengths.size
    signs = np.zeros(gate_count, dtype=np.int8)
    particle_ids = np.full(gate_count, -1, dtype=np.int32)
    best_distance = np.full(gate_count, np.inf)
    tie = config.projection_tie_over_d0 * distance_d0
    for particle in np.flatnonzero(state.wall_required):
        component = int(state.projection.component[particle])
        local_arc = (
            state.projection.arc[particle] - scheme.component_starts[component]
            - scheme.component_regular_offsets[component]
        )
        if not (0.0 <= local_arc <= scheme.component_regular_lengths[component]):
            continue
        local_gate = min(
            int(math.floor(local_arc / scheme.component_gate_lengths[component])),
            int(scheme.component_gate_counts[component]) - 1,
        )
        gate = int(scheme.component_gate_offsets[component]) + local_gate
        wall_distance = state.projection.distance[particle]
        sign = 1 if state.q[particle] > 0.0 else -1
        if wall_distance < best_distance[gate] - tie:
            best_distance[gate] = wall_distance
            signs[gate] = sign
            particle_ids[gate] = particle
        elif wall_distance <= best_distance[gate] + tie and signs[gate] != sign:
            # An opposite-sign nearest-distance tie has no preferred direction.
            signs[gate] = 0
            particle_ids[gate] = -1
    representative_distance = np.where(particle_ids >= 0, best_distance, np.nan)
    return signs, particle_ids, representative_distance


def frame_grid_occupancy(
    state: FrameState,
    scheme: GateScheme,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """Return unweighted grid occupancy for an arbitrary candidate mask."""
    occupied = np.zeros(scheme.gate_lengths.size, dtype=bool)
    for particle in np.flatnonzero(candidate_mask):
        component = int(state.projection.component[particle])
        local_arc = (
            state.projection.arc[particle]
            - scheme.component_starts[component]
            - scheme.component_regular_offsets[component]
        )
        if not (0.0 <= local_arc <= scheme.component_regular_lengths[component]):
            continue
        local_grid = min(
            int(math.floor(local_arc / scheme.component_gate_lengths[component])),
            int(scheme.component_gate_counts[component]) - 1,
        )
        grid = int(scheme.component_gate_offsets[component]) + local_grid
        occupied[grid] = True
    return occupied


def calibration_increment(
    left: FrameState,
    right: FrameState,
    left_signs: np.ndarray,
    left_particle_ids: np.ndarray,
    model,
    geometry: bda.BoundaryGeometry,
    saved_dt: float,
    gate_count: int,
    high_alpha: bool,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Direction-neutral QC: does phase sign predict the representative's motion?"""
    trials = np.zeros(gate_count, dtype=np.int32)
    successes = np.zeros(gate_count, dtype=np.int32)
    ds = wrap_arclength(right.projection.arc - left.projection.arc, geometry.perimeter)
    direction = np.sign(ds)
    displacement_ratio = np.abs(ds) / (model.speedV * saved_dt)

    # Projection along an inner parallel circle is amplified by 1/(1-d/R).
    maximum_step = np.full(ds.size, model.speedV * saved_dt)
    curved = left.projection.curvature > 0.0
    if np.any(curved):
        if high_alpha:
            maximum_depth = np.full(
                ds.size, config.boundary_layer_over_d0 * model.distanceD0
            )
        else:
            # A particle can move radially by at most v*Delta t between saved
            # endpoints; include that amount in the worst reachable depth.
            maximum_depth = (
                np.maximum(left.projection.distance, right.projection.distance)
                + model.speedV * saved_dt
            )
        denominator = 1.0 - left.projection.curvature * maximum_depth
        safe = curved & (denominator > 0.0)
        maximum_step[safe] = model.speedV * saved_dt / denominator[safe]
        maximum_step[curved & ~safe] = np.inf
    maximum_step *= config.maximum_projection_step_factor
    increment_unique = np.isfinite(maximum_step) & (
        maximum_step < 0.5 * geometry.perimeter
    )

    for gate, particle in enumerate(left_particle_ids):
        if particle < 0 or left_signs[gate] == 0:
            continue
        trackable = (
            right.wall_required[particle]
            and left.projection.component[particle] == right.projection.component[particle]
            and np.sign(right.q[particle]) == left_signs[gate]
            and increment_unique[particle]
            and abs(ds[particle]) <= maximum_step[particle]
        )
        if not trackable:
            continue
        trials[gate] = 1
        if (
            displacement_ratio[particle] >= config.minimum_displacement_over_v_dt
            and direction[particle] == left_signs[gate]
        ):
            successes[gate] = 1
    return trials, successes


# =============================================================================
# TWO HEADLINE OBSERVABLES
# =============================================================================


def largest_circular_extent(mask: np.ndarray, lengths: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    if np.all(mask):
        return float(np.sum(lengths))
    doubled = np.concatenate((mask, mask))
    doubled_lengths = np.concatenate((lengths, lengths))
    best = current = 0.0
    for value, length in zip(doubled, doubled_lengths):
        current = current + length if value else 0.0
        best = max(best, current)
    return min(best, float(np.sum(lengths)))


def largest_linear_extent(mask: np.ndarray, lengths: np.ndarray) -> float:
    best = current = 0.0
    for value, length in zip(np.asarray(mask, dtype=bool), lengths):
        current = current + length if value else 0.0
        best = max(best, current)
    return float(best)


def largest_extent_without_cross_join(
    mask: np.ndarray,
    scheme: GateScheme,
) -> float:
    """Largest active arc when finite join masks break component adjacency."""
    if scheme.component_gate_counts.size == 1:
        return largest_circular_extent(mask, scheme.gate_lengths)
    best = 0.0
    for offset, count in zip(
        scheme.component_gate_offsets, scheme.component_gate_counts
    ):
        section = slice(int(offset), int(offset + count))
        best = max(
            best,
            largest_linear_extent(mask[section], scheme.gate_lengths[section]),
        )
    return float(best)


def lag1_correlation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    if values.size < 3:
        return float("nan")
    left, right = values[:-1], values[1:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def aggregate_votes(
    votes: np.ndarray,
    calibrated: np.ndarray,
    calibration_trials: np.ndarray,
    calibration_successes: np.ndarray,
    scheme: GateScheme,
    distance_d0: float,
    config: AnalysisConfig,
    compute_sensitivity: bool = True,
) -> tuple[float, float, AnalysisDetail]:
    positive = np.sum(votes > 0, axis=1)
    negative = np.sum(votes < 0, axis=1)
    evidence = positive + negative
    purity = np.divide(
        np.abs(positive - negative),
        evidence,
        out=np.zeros_like(evidence, dtype=float),
        where=evidence > 0,
    )
    active_gate = (
        (evidence >= config.minimum_support_pairs)
        & (purity >= config.minimum_directional_purity)
        & calibrated[None, :]
    )
    block_gate_sign = np.where(active_gate, np.sign(positive - negative), 0).astype(np.int8)

    global_active = np.any(block_gate_sign != 0, axis=0)
    maximum_extent = largest_circular_extent(global_active, scheme.gate_lengths)
    maximum_extent_no_cross_join = largest_extent_without_cross_join(
        global_active, scheme
    )
    extent_valid = maximum_extent >= config.macro_extent_over_d0 * distance_d0
    extent_valid_no_cross_join = (
        maximum_extent_no_cross_join >= config.macro_extent_over_d0 * distance_d0
    )
    block_active = np.any(block_gate_sign != 0, axis=1) & extent_valid

    active_lengths = np.sum(
        np.abs(block_gate_sign) * scheme.gate_lengths[None, :], axis=1
    )
    signed_lengths = np.sum(
        block_gate_sign * scheme.gate_lengths[None, :], axis=1
    )
    block_chirality = np.divide(
        signed_lengths,
        active_lengths,
        out=np.zeros_like(signed_lengths, dtype=float),
        where=active_lengths > 0,
    )

    xi_persist = (
        float(np.mean(block_active))
        if extent_valid and block_active.size
        else float("nan")
    )
    active_count = int(np.sum(block_active))
    if extent_valid and active_count >= config.minimum_active_blocks_for_sign:
        xi_sign = float(abs(np.mean(block_chirality[block_active])))
    else:
        xi_sign = float("nan")

    total_trials = int(np.sum(calibration_trials))
    total_successes = int(np.sum(calibration_successes))
    active_arclength_fraction = active_lengths / scheme.regular_perimeter
    calibrated_perimeter_fraction = float(
        np.clip(
            np.sum(scheme.gate_lengths[calibrated]) / scheme.regular_perimeter,
            0.0,
            1.0,
        )
    )
    detail = AnalysisDetail(
        block_gate_sign=block_gate_sign,
        block_chirality=block_chirality,
        block_active=block_active,
        gate_lengths=scheme.gate_lengths.copy(),
        maximum_contiguous_extent=maximum_extent,
        extent_valid=extent_valid,
        calibrated_gate_count=int(np.sum(calibrated)),
        calibrated_perimeter_fraction=calibrated_perimeter_fraction,
        block_active_arclength_fraction=active_arclength_fraction,
        mean_active_arclength_fraction=float(np.mean(active_arclength_fraction)),
        active_block_mean_arclength_fraction=(
            float(np.mean(active_arclength_fraction[block_active]))
            if np.any(block_active)
            else float("nan")
        ),
        preselection_grid_coverage=float("nan"),
        selected_grid_coverage=float("nan"),
        w_selection_retention=float("nan"),
        representative_wall_distance_quantiles_over_d0=np.full(4, np.nan),
        maximum_contiguous_extent_no_cross_join=maximum_extent_no_cross_join,
        extent_valid_no_cross_join=extent_valid_no_cross_join,
        block_activity_lag1_correlation=lag1_correlation(block_active.astype(float)),
        block_chirality_lag1_correlation=lag1_correlation(block_chirality),
        omitted_perimeter_fraction=scheme.omitted_perimeter_fraction,
        support_pair_thresholds=np.empty(0, dtype=int),
        support_pair_xi_persist=np.empty(0),
        support_pair_xi_sign=np.empty(0),
        calibration_trial_count=total_trials,
        calibration_success_count=total_successes,
        calibration_agreement=(
            total_successes / total_trials if total_trials else float("nan")
        ),
        phase_rhs_validation_error=float("nan"),
    )
    if compute_sensitivity:
        thresholds = np.asarray(config.support_pair_sensitivity, dtype=int)
        persist_values = np.empty(thresholds.size, dtype=float)
        sign_values = np.empty(thresholds.size, dtype=float)
        for index, threshold in enumerate(thresholds):
            persist_values[index], sign_values[index], _ = aggregate_votes(
                votes,
                calibrated,
                calibration_trials,
                calibration_successes,
                scheme,
                distance_d0,
                replace(config, minimum_support_pairs=int(threshold)),
                compute_sensitivity=False,
            )
        detail = detail._replace(
            support_pair_thresholds=thresholds,
            support_pair_xi_persist=persist_values,
            support_pair_xi_sign=sign_values,
        )
    return xi_persist, xi_sign, detail


def analyse_model(
    model,
    alpha_over_pi: float,
    config: AnalysisConfig,
    validate_rhs_once: bool,
) -> tuple[dict, AnalysisDetail]:
    geometry = bda.geometry_for(model)
    total_frames, _ = bda._hdf_layout(model)
    source = bda.data_path(model)
    requested_time = config.terminal_circuit_times * geometry.perimeter / model.speedV

    zero_endpoint = math.isclose(alpha_over_pi, 0.0, abs_tol=1.0e-12)
    pi_endpoint = math.isclose(alpha_over_pi, 1.0, abs_tol=1.0e-12)
    if zero_endpoint or pi_endpoint:
        detail = AnalysisDetail(
            block_gate_sign=np.empty((0, 0), dtype=np.int8),
            block_chirality=np.empty(0),
            block_active=np.empty(0, dtype=bool),
            gate_lengths=np.empty(0),
            maximum_contiguous_extent=float("nan"),
            extent_valid=False,
            calibrated_gate_count=0,
            calibrated_perimeter_fraction=float("nan"),
            block_active_arclength_fraction=np.empty(0),
            mean_active_arclength_fraction=float("nan"),
            active_block_mean_arclength_fraction=float("nan"),
            preselection_grid_coverage=float("nan"),
            selected_grid_coverage=float("nan"),
            w_selection_retention=float("nan"),
            representative_wall_distance_quantiles_over_d0=np.full(4, np.nan),
            maximum_contiguous_extent_no_cross_join=float("nan"),
            extent_valid_no_cross_join=False,
            block_activity_lag1_correlation=float("nan"),
            block_chirality_lag1_correlation=float("nan"),
            omitted_perimeter_fraction=float("nan"),
            support_pair_thresholds=np.empty(0, dtype=int),
            support_pair_xi_persist=np.empty(0),
            support_pair_xi_sign=np.empty(0),
            calibration_trial_count=0,
            calibration_success_count=0,
            calibration_agreement=float("nan"),
            phase_rhs_validation_error=float("nan"),
        )
        return {
            "alpha_over_pi": alpha_over_pi,
            "regime": (
                "Independent_Zero_Endpoint"
                if zero_endpoint
                else "Independent_Pi_Endpoint"
            ),
            "xi_persist": float("nan"),
            "xi_sign": float("nan"),
            "block_count": 0,
            "active_block_count": 0,
            "maximum_contiguous_extent": float("nan"),
            "extent_valid": False,
            "total_saved_frames": total_frames,
            "terminal_saved_frame": total_frames - 1,
            "terminal_iteration": (total_frames - 1) * model.shotsnaps,
            "analysis_window_time": 0.0,
            "requested_analysis_window_time": requested_time,
            "analysis_window_fraction": float("nan"),
            "n_blk": 0,
            "source_hdf5": str(source),
        }, detail

    saved_dt = float(model.dt * model.shotsnaps)
    block_time = config.block_time_over_d0_over_v * model.distanceD0 / model.speedV
    intervals_per_block = max(1, int(math.ceil(block_time / saved_dt)))
    window = bda.load_window(model, duration=requested_time)
    available_intervals = window.positions.shape[0] - 1
    block_count = available_intervals // intervals_per_block
    if block_count < config.minimum_active_blocks_for_sign:
        raise RuntimeError(f"Too few terminal blocks in {source.name}.")
    used_intervals = block_count * intervals_per_block
    start = window.positions.shape[0] - (used_intervals + 1)
    positions = window.positions[start:]
    phases = window.phases[start:]

    gate_scheme = build_gate_scheme(geometry, model.distanceD0, config)
    gate_count = int(gate_scheme.gate_lengths.size)
    phase_field = np.zeros((used_intervals + 1, gate_count), dtype=np.int8)
    calibration_trials = np.zeros(gate_count, dtype=np.int64)
    calibration_successes = np.zeros(gate_count, dtype=np.int64)
    high_alpha = 0.5 < alpha_over_pi < 1.0
    preselection_grid_slots = 0
    selected_grid_slots = 0
    representative_distances: list[np.ndarray] = []

    validation_error = float("nan")
    if high_alpha and validate_rhs_once and config.validate_optimized_phase_rhs:
        validation_error = validate_phase_rhs(positions[0], phases[0], model, config)

    left = prepare_frame(
        positions[0], phases[0], model, geometry, high_alpha, config
    )
    left_signs, left_particle_ids, left_distances = frame_gate_representatives(
        left, gate_scheme, model.distanceD0, config
    )
    preselection_grid_slots += int(
        np.sum(frame_grid_occupancy(left, gate_scheme, left.preselection))
    )
    selected_grid_slots += int(
        np.sum(frame_grid_occupancy(left, gate_scheme, left.wall_required))
    )
    representative_distances.append(left_distances[np.isfinite(left_distances)])
    phase_field[0] = left_signs
    for interval in range(used_intervals):
        right = prepare_frame(
            positions[interval + 1],
            phases[interval + 1],
            model,
            geometry,
            high_alpha,
            config,
        )
        right_signs, right_particle_ids, right_distances = frame_gate_representatives(
            right, gate_scheme, model.distanceD0, config
        )
        preselection_grid_slots += int(
            np.sum(frame_grid_occupancy(right, gate_scheme, right.preselection))
        )
        selected_grid_slots += int(
            np.sum(frame_grid_occupancy(right, gate_scheme, right.wall_required))
        )
        representative_distances.append(right_distances[np.isfinite(right_distances)])
        phase_field[interval + 1] = right_signs
        trial_increment, success_increment = calibration_increment(
            left,
            right,
            left_signs,
            left_particle_ids,
            model,
            geometry,
            saved_dt,
            gate_count,
            high_alpha,
            config,
        )
        calibration_trials += trial_increment
        calibration_successes += success_increment
        left = right
        left_signs = right_signs
        left_particle_ids = right_particle_ids

    calibration_ratio = np.divide(
        calibration_successes,
        calibration_trials,
        out=np.zeros(gate_count, dtype=float),
        where=calibration_trials > 0,
    )
    calibrated = (
        (calibration_trials >= config.minimum_calibration_trials)
        & (calibration_ratio >= config.minimum_calibration_agreement)
    )
    continuous_phase = np.where(
        (phase_field[:-1] == phase_field[1:]) & (phase_field[:-1] != 0),
        phase_field[:-1],
        0,
    ).astype(np.int8)
    votes = continuous_phase.reshape(block_count, intervals_per_block, gate_count)
    xi_persist, xi_sign, detail = aggregate_votes(
        votes,
        calibrated,
        calibration_trials,
        calibration_successes,
        gate_scheme,
        model.distanceD0,
        config,
    )
    total_grid_slots = (used_intervals + 1) * gate_count
    distance_samples = np.concatenate(representative_distances)
    distance_quantiles = (
        np.quantile(distance_samples / model.distanceD0, (0.50, 0.90, 0.99, 1.00))
        if distance_samples.size
        else np.full(4, np.nan)
    )
    detail = detail._replace(
        phase_rhs_validation_error=validation_error,
        preselection_grid_coverage=preselection_grid_slots / total_grid_slots,
        selected_grid_coverage=selected_grid_slots / total_grid_slots,
        w_selection_retention=(
            selected_grid_slots / preselection_grid_slots
            if high_alpha and preselection_grid_slots
            else float("nan")
        ),
        representative_wall_distance_quantiles_over_d0=distance_quantiles,
    )
    if alpha_over_pi < 0.5:
        regime = "Chiral_No_Pattern_Prior"
    elif alpha_over_pi > 0.5:
        regime = "Pattern_Supported_Prior"
    else:
        # The pattern-formation threshold is not assigned to either open
        # regime.  Its score uses the broad (no fixed radial cutoff) candidate
        # construction and is reported as an isolated critical-point estimate.
        regime = "Critical_Pattern_Threshold"
    return {
        "alpha_over_pi": alpha_over_pi,
        "regime": regime,
        "xi_persist": xi_persist,
        "xi_sign": xi_sign,
        "block_count": block_count,
        "active_block_count": int(np.sum(detail.block_active)),
        "maximum_contiguous_extent": detail.maximum_contiguous_extent,
        "maximum_contiguous_extent_no_cross_join": (
            detail.maximum_contiguous_extent_no_cross_join
        ),
        "extent_valid": detail.extent_valid,
        "extent_valid_no_cross_join": detail.extent_valid_no_cross_join,
        "mean_active_arclength_fraction": detail.mean_active_arclength_fraction,
        "active_block_mean_arclength_fraction": (
            detail.active_block_mean_arclength_fraction
        ),
        "calibrated_perimeter_fraction": detail.calibrated_perimeter_fraction,
        "preselection_grid_coverage": detail.preselection_grid_coverage,
        "selected_grid_coverage": detail.selected_grid_coverage,
        "w_selection_retention": detail.w_selection_retention,
        "representative_wall_distance_q50_over_d0": distance_quantiles[0],
        "representative_wall_distance_q90_over_d0": distance_quantiles[1],
        "representative_wall_distance_q99_over_d0": distance_quantiles[2],
        "representative_wall_distance_max_over_d0": distance_quantiles[3],
        "block_activity_lag1_correlation": (
            detail.block_activity_lag1_correlation
        ),
        "block_chirality_lag1_correlation": (
            detail.block_chirality_lag1_correlation
        ),
        "omitted_perimeter_fraction": detail.omitted_perimeter_fraction,
        "total_saved_frames": window.total_frames,
        "terminal_saved_frame": window.total_frames - 1,
        "terminal_iteration": (window.total_frames - 1) * model.shotsnaps,
        "analysis_window_time": used_intervals * saved_dt,
        "requested_analysis_window_time": requested_time,
        "analysis_window_fraction": used_intervals * saved_dt / requested_time,
        "n_blk": intervals_per_block,
        "source_hdf5": str(source),
    }, detail


# =============================================================================
# OUTPUT
# =============================================================================


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


def versioned_pair(output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        suffix = "" if version == 1 else f"_V{version}"
        png = output_dir / f"{stem}{suffix}.png"
        pdf = output_dir / f"{stem}{suffix}.pdf"
        if not png.exists() and not pdf.exists():
            return png, pdf
        version += 1


def plot_metrics(table: pd.DataFrame, config: AnalysisConfig) -> tuple[Path, Path]:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.05), sharex=True, sharey=True)
    quantities = (
        (
            "xi_persist",
            r"Active-block fraction $\Xi_{\mathrm{Persist}}$",
            "Long-time boundary-flow presence",
        ),
        (
            "xi_sign",
            r"Mean chirality $\Xi_{\mathrm{Sign}}$",
            "Mean boundary chirality",
        ),
    )
    for axis, (column, ylabel, title), panel_label in zip(
        axes, quantities, ("(a)", "(b)")
    ):
        axis.axvline(0.5, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
        axis.text(
            0.5,
            1.015,
            r"Pattern onset $\alpha=\pi/2$ (from particle trajectories)",
            ha="center",
            va="bottom",
            transform=axis.get_xaxis_transform(),
            fontsize=8.0,
            color="#555555",
        )
        for condition in CASE_LABELS:
            subset = table[table["condition"] == condition].sort_values("alpha_over_pi")
            subset = subset[subset["alpha_over_pi"] < 1.0]
            color = COLORS[condition]
            marker = MARKERS[condition]
            for part_index, regime_mask in enumerate((
                (subset["alpha_over_pi"] > 0.0) & (subset["alpha_over_pi"] < 0.5),
                (subset["alpha_over_pi"] > 0.5) & (subset["alpha_over_pi"] < 1.0),
            )):
                part = subset[regime_mask]
                axis.plot(
                    part["alpha_over_pi"],
                    part[column],
                    color=color,
                    linewidth=1.25,
                )
                full_window = part[part["analysis_window_fraction"] >= 0.995]
                short_window = part[part["analysis_window_fraction"] < 0.995]
                axis.scatter(
                    full_window["alpha_over_pi"],
                    full_window[column],
                    marker=marker,
                    s=30,
                    facecolors=color,
                    edgecolors=color,
                    linewidths=0.9,
                    zorder=3,
                )
                axis.scatter(
                    short_window["alpha_over_pi"],
                    short_window[column],
                    marker=marker,
                    s=30,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.1,
                    zorder=3,
                )
            critical = subset[np.isclose(subset["alpha_over_pi"], 0.5)]
            critical_full = critical[critical["analysis_window_fraction"] >= 0.995]
            critical_short = critical[critical["analysis_window_fraction"] < 0.995]
            axis.scatter(
                critical_full["alpha_over_pi"],
                critical_full[column],
                marker=marker,
                s=36,
                facecolors=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
            axis.scatter(
                critical_short["alpha_over_pi"],
                critical_short[column],
                marker=marker,
                s=36,
                facecolors="white",
                edgecolors=color,
                linewidths=1.1,
                zorder=4,
            )
            endpoint = subset[np.isclose(subset["alpha_over_pi"], 0.0)]
            axis.scatter(
                endpoint["alpha_over_pi"],
                endpoint[column],
                marker=marker,
                s=25,
                facecolors="none",
                edgecolors=color,
                linewidths=0.9,
                alpha=0.55,
                zorder=3,
            )
        axis.set_title(title, pad=18)
        axis.set_xlabel(r"Phase lag $\alpha/\pi$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.025, 1.025)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.65)
        axis.text(
            0.015,
            0.965,
            panel_label,
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
        )
    handles = [
        mpl.lines.Line2D(
            [], [], color=COLORS[condition], marker=MARKERS[condition],
            markerfacecolor=COLORS[condition], markersize=5.2, linewidth=1.2,
            label=CASE_LABELS[condition],
        )
        for condition in CASE_LABELS
    ]
    handles.extend((
        mpl.lines.Line2D(
            [], [], color="#555555", marker="o", markerfacecolor="#555555",
            linestyle="none", markersize=5.2, label="Full requested window",
        ),
        mpl.lines.Line2D(
            [], [], color="#555555", marker="o", markerfacecolor="white",
            linestyle="none", markersize=5.2, label="Shorter available window",
        ),
    ))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("Long-Time Boundary-Flow Stability", y=1.035, fontsize=12.2)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    png, pdf = versioned_pair(
        config.output_dir, "Phase_Informed_Boundary_Flow_Alpha_Sweep"
    )
    fig.savefig(png, dpi=config.figure_dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_diagnostics(table: pd.DataFrame, config: AnalysisConfig) -> tuple[Path, Path]:
    """Render boundary-coverage and particle-selection checks."""
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    panels = (
        (
            "mean_active_arclength_fraction",
            r"Mean active boundary length $\overline{a_b}$",
            "Active boundary length per block",
        ),
        (
            "calibrated_perimeter_fraction",
            r"Calibrated boundary length $f_{\mathrm{cal}}$",
            "Boundary passing the displacement check",
        ),
        (
            "w_selection_retention",
            r"Fraction retained by $W$ filter",
            r"High-$\alpha$ particle selection",
        ),
        (
            "representative_wall_distance_q50_over_d0",
            r"Distance quantiles $d/d_0$",
            "Selected-particle wall distance",
        ),
    )
    for axis, (column, ylabel, title), panel_label in zip(
        axes.flat, panels, ("(a)", "(b)", "(c)", "(d)")
    ):
        axis.axvline(0.5, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
        for condition in CASE_LABELS:
            subset = table[
                (table["condition"] == condition)
                & (table["alpha_over_pi"] > 0.0)
                & (table["alpha_over_pi"] < 1.0)
            ].sort_values("alpha_over_pi")
            if column == "w_selection_retention":
                subset = subset[subset["alpha_over_pi"] > 0.5]
            axis.plot(
                subset["alpha_over_pi"],
                subset[column],
                color=COLORS[condition],
                linewidth=1.15,
            )
            full_window = subset[subset["analysis_window_fraction"] >= 0.995]
            short_window = subset[subset["analysis_window_fraction"] < 0.995]
            axis.scatter(
                full_window["alpha_over_pi"],
                full_window[column],
                marker=MARKERS[condition],
                s=25,
                facecolors=COLORS[condition],
                edgecolors=COLORS[condition],
                linewidths=0.8,
                zorder=3,
            )
            axis.scatter(
                short_window["alpha_over_pi"],
                short_window[column],
                marker=MARKERS[condition],
                s=25,
                facecolors="white",
                edgecolors=COLORS[condition],
                linewidths=1.0,
                zorder=3,
            )
            if column == "representative_wall_distance_q50_over_d0":
                axis.fill_between(
                    subset["alpha_over_pi"].to_numpy(dtype=float),
                    subset["representative_wall_distance_q50_over_d0"].to_numpy(dtype=float),
                    subset["representative_wall_distance_q90_over_d0"].to_numpy(dtype=float),
                    color=COLORS[condition],
                    alpha=0.08,
                    linewidth=0.0,
                )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.025, 1.025)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.65)
        axis.text(
            0.015,
            0.965,
            panel_label,
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
        )
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel(r"Phase lag $\alpha/\pi$")
    axes[1, 1].set_xlabel(r"Phase lag $\alpha/\pi$")
    handles = [
        mpl.lines.Line2D(
            [], [], color=COLORS[condition], marker=MARKERS[condition],
            markerfacecolor=COLORS[condition], markersize=5.0, linewidth=1.15,
            label=CASE_LABELS[condition],
        )
        for condition in CASE_LABELS
    ]
    handles.extend((
        mpl.lines.Line2D(
            [], [], color="#555555", marker="o", markerfacecolor="#555555",
            linestyle="none", markersize=5.0, label="Full requested window",
        ),
        mpl.lines.Line2D(
            [], [], color="#555555", marker="o", markerfacecolor="white",
            linestyle="none", markersize=5.0, label="Shorter available window",
        ),
    ))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle("Boundary-Flow Coverage and Selection Diagnostics", y=1.01, fontsize=12.2)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.99))
    png, pdf = versioned_pair(
        config.output_dir, "Phase_Informed_Boundary_Flow_Diagnostics"
    )
    fig.savefig(png, dpi=config.figure_dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def write_definitions(config: AnalysisConfig) -> Path:
    path = config.output_dir / "Phase_Informed_Boundary_Flow_Definitions.txt"
    text = rf"""LONG-TIME BOUNDARY-FLOW STABILITY

Symbols and raw data
--------------------
X_i^n=(x_i^n,y_i^n) and theta_i^n are rows of positionX and phaseTheta at saved
frame n.  theta is the microscopic phase and the free-flight heading angle:
u_i^n=(cos(theta_i^n),sin(theta_i^n)).  Delta t_s=dt*shotsnaps.

The particle interaction distance is d0=model.distanceD0.  Capital D0 is
reserved for the continuum closure denominator and is never used as a particle
distance in this analysis.

Boundary coordinate and grid index g
------------------------------------
Pi(X_i^n)=(d_i^n,s_i^n,t_i^n,kappa_(b,i)^n,e_i^n) is the unique regular
closest-boundary projection; s and t are counterclockwise.  Polygonal and
arc-line components omit {config.join_exclusion_over_d0:g} d0 at each incident
vertex/join.  The independent closest-distance tie tolerance is
{config.projection_tie_over_d0:g} d0.  Let P_reg be the retained perimeter.

For retained component e of length L_e:
N_e=ceil[L_e/({config.gate_length_over_d0:g} d0)], ell_g=L_e/N_e.
No grid cell I_g crosses a join and sum_g ell_g=P_reg.

q_i^n=u_i^n dot t_i^n.  In each grid cell, Q_ng is the sign of q for the
nearest retained candidate with |q|>=q0={config.minimum_phase_projection:g};
Q_ng=0 for an empty candidate set or an opposite-sign nearest-distance tie.
Low-alpha candidates have no radial cutoff.  High-alpha candidates also require
d<=d0, 1-kappa_b*d>0, and W>epsilon_W.

The critical point alpha=pi/2 is assigned to neither open regime.  Its isolated
estimate uses the broad no-radial-cutoff construction and no W selection; it is
not an estimate obtained with either open-regime rule.

High-alpha W particle-selection rule
------------------------------------
N_i^n={{j!=i:||X_j^n-X_i^n||<=d0}} and
Omega_i^n=omega_i+K[mean_(j in N_i^n)sin(theta_j^n-theta_i^n+alpha)-sin(alpha)],
with Omega_i^n=omega_i when N_i^n is empty.  This heading angular velocity is
recomputed from the saved state using the main.py phase RHS.

kappa_d=kappa_b/(1-kappa_b*d),
W_i^n=kappa_d*v*(q_i^n)^2-Omega_i^n*q_i^n,
epsilon_W={config.w_dead_zone_over_v_over_d0:g} v/d0.

W is only a conditional selection score.  Its acceleration interpretation is
valid for approximately tangent motion along a constant-distance parallel
curve, not generally during oblique wall approach or reflection.  It is not a
force measurement, topology test, or headline observable.

Heading-displacement check
--------------------------
Track frame n's representative to n+1.  T_ng=1 requires valid same-component
endpoints, the regime filters at both endpoints, equal nonzero heading signs,
and a unique geometry-bounded wrapped increment.  E_ng=T_ng only if
|Delta s|/(v Delta t_s)>={config.minimum_displacement_over_v_dt:g} and
sign(Delta s)=Q_ng; otherwise E_ng=0.  Grid g is calibrated when
sum_n T_ng>={config.minimum_calibration_trials} and
C_g=sum_n E_ng/sum_n T_ng>={config.minimum_calibration_agreement:.8f}.
This is internal quality control, not independent validation.

Two-frame signal and block index n_blk
--------------------------------------
Y_ng=Q_ng when Q_ng=Q_(n+1)g!=0; otherwise Y_ng=0.  The representatives may be
different particles.  The target block time is d0/v and the saved-data block is
n_blk=ceil[(d0/v)/Delta t_s], Delta T_b=n_blk*Delta t_s.  Here n_blk=7.

For F saved frames, the requested terminal duration is T_req=10 P/v and
N_avail=min[F-1,ceil(T_req/Delta t_s)].  The complete-block count is
B=floor(N_avail/n_blk), and only frames n0=F-1-B*n_blk through F-1 are used.
The exported analysis_window_fraction is B*n_blk*Delta t_s/T_req; it identifies
records shorter than the requested ten-circuit observation window.

Within block b:
N_bg^+=#{{n in b:Y_ng=+1}}, N_bg^-=#{{n in b:Y_ng=-1}},
p_bg=|N_bg^+-N_bg^-|/(N_bg^++N_bg^-) if the denominator is positive, and
p_bg=0 otherwise.  Z_bg=sign(N_bg^+-N_bg^-) only if the support count is at
least {config.minimum_support_pairs}, p_bg>={config.minimum_directional_purity:.8f},
and g is calibrated; otherwise Z_bg=0.  Sensitivity uses minimum counts
{config.support_pair_sensitivity} without weighting either headline score.

The union A={{g:exists b,|Z_bg|=1}} must contain a connected boundary interval
of at least {config.macro_extent_over_d0:g} d0.  Different cells in this interval
may be active in different blocks, which allows relay transport.  A second
export disables cross-join adjacency.

Two stability measures
----------------------
M_b=1 if block b contains any Z_bg!=0 after the accumulated-extent check.

Xi_Persist=(1/B)sum_b M_b

is the fraction of time blocks containing a valid directional boundary signal.
It is not multiplied by particle number or carrier fraction.

chi_b=[sum_g ell_g Z_bg]/[sum_g ell_g |Z_bg|],
Xi_Sign=|sum_b M_b chi_b/sum_b M_b|.

Xi_Sign is the absolute mean block chirality over all active blocks.  It is one
only when every active block is entirely clockwise or every active block is
entirely counterclockwise.  Spatial cancellation and reversal reduce it.

Selection and coverage checks
-----------------------------
a_b=[sum_g ell_g|Z_bg|]/P_reg is active arclength fraction.
f_cal=[sum_g ell_g*1(g calibrated)]/P_reg is calibrated perimeter fraction.
The export also records pre-W and retained grid coverage, W-retention ratio,
omitted-perimeter fraction, the no-cross-join extent check, d/d0 representative
distance quantiles, and lag-one block correlations.

Limitations and endpoints
-------------------------
Results use one random seed, unequal terminal-window durations, temporally
correlated blocks, and same-window calibration.  Finite defect/join masks omit
strong-scattering neighborhoods.  The low-alpha construction has no radial
cutoff and may include remote nearest-boundary coordinates; distance quantiles
must therefore accompany it.  alpha=0 and alpha=pi are independent endpoints
and are reported as NA for both single-chirality measures.
"""
    path.write_text(text, encoding="utf-8")
    return path


def serializable_config(config: AnalysisConfig) -> dict:
    payload = asdict(config)
    payload["data_dir"] = str(config.data_dir)
    payload["output_dir"] = str(config.output_dir)
    return payload


def compute_q0_sensitivity(config: AnalysisConfig) -> pd.DataFrame:
    """Recompute two representative low-alpha states without changing HDF5."""
    rows = []
    for spec in (CASE_SPECS[2], CASE_SPECS[3]):
        for q0 in config.q0_sensitivity:
            sensitivity_config = replace(config, minimum_phase_projection=q0)
            model = build_model(spec, 0.2)
            row, detail = analyse_model(
                model,
                0.2,
                sensitivity_config,
                validate_rhs_once=False,
            )
            rows.append(
                {
                    "condition": spec.condition,
                    "condition_label": CASE_LABELS[spec.condition],
                    "alpha_over_pi": 0.2,
                    "minimum_phase_projection_q0": q0,
                    "xi_persist": row["xi_persist"],
                    "xi_sign": row["xi_sign"],
                    "calibrated_grid_count": detail.calibrated_gate_count,
                    "grid_count": detail.block_gate_sign.shape[1],
                    "maximum_contiguous_extent": detail.maximum_contiguous_extent,
                    "extent_valid": detail.extent_valid,
                }
            )
    return pd.DataFrame(rows)


def run(config: AnalysisConfig) -> list[Path]:
    jobs = all_jobs(config)
    # Exact-file validation happens before any output directory or figure exists.
    bda.validate_exact_files([model for _, _, model in jobs])

    rows: list[dict] = []
    support_rows: list[dict] = []
    phase_rhs_validated_conditions: set[str] = set()
    for index, (spec, alpha, model) in enumerate(jobs, start=1):
        print(
            f"[{index:02d}/{len(jobs):02d}] {CASE_LABELS[spec.condition]}, "
            f"alpha/pi={alpha:.1f}",
            flush=True,
        )
        row, detail = analyse_model(
            model,
            alpha,
            config,
            validate_rhs_once=(spec.condition not in phase_rhs_validated_conditions),
        )
        if np.isfinite(detail.phase_rhs_validation_error):
            phase_rhs_validated_conditions.add(spec.condition)
        row.update(
            {
                "group": spec.group,
                "condition": spec.condition,
                "condition_label": CASE_LABELS[spec.condition],
                "model_class": model.__class__.__name__,
                "defect_height": spec.defect_height,
                "defect_half_width": (
                    bda.SPIKE_HALF_WIDTH if spec.defect_height > 0.0 else 0.0
                ),
                "grid_count": (
                    detail.block_gate_sign.shape[1]
                    if detail.block_gate_sign.ndim == 2
                    else 0
                ),
                "calibrated_grid_count": detail.calibrated_gate_count,
                "calibration_trial_count": detail.calibration_trial_count,
                "calibration_success_count": detail.calibration_success_count,
                "calibration_agreement": detail.calibration_agreement,
                "phase_rhs_validation_error": detail.phase_rhs_validation_error,
            }
        )
        rows.append(row)
        for threshold, persist, sign in zip(
            detail.support_pair_thresholds,
            detail.support_pair_xi_persist,
            detail.support_pair_xi_sign,
        ):
            support_rows.append(
                {
                    "condition": spec.condition,
                    "condition_label": CASE_LABELS[spec.condition],
                    "alpha_over_pi": alpha,
                    "minimum_supported_frame_pairs": int(threshold),
                    "xi_persist": persist,
                    "xi_sign": sign,
                }
            )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(
        ["group", "condition", "alpha_over_pi"]
    )
    sensitivity = compute_q0_sensitivity(config)
    support_sensitivity = pd.DataFrame(support_rows)
    csv_path = config.output_dir / "Phase_Informed_Boundary_Flow_Metrics.csv"
    table.to_csv(csv_path, index=False)
    sensitivity_path = (
        config.output_dir / "Phase_Informed_Boundary_Flow_Q0_Sensitivity.csv"
    )
    sensitivity.to_csv(sensitivity_path, index=False)
    support_path = (
        config.output_dir
        / "Phase_Informed_Boundary_Flow_Support_Count_Sensitivity.csv"
    )
    support_sensitivity.to_csv(support_path, index=False)
    config_path = config.output_dir / "Phase_Informed_Boundary_Flow_Configuration.json"
    config_path.write_text(
        json.dumps(serializable_config(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    definitions_path = write_definitions(config)
    png_path, pdf_path = plot_metrics(table, config)
    diagnostics_png, diagnostics_pdf = plot_diagnostics(table, config)
    return [
        csv_path,
        sensitivity_path,
        support_path,
        config_path,
        definitions_path,
        png_path,
        pdf_path,
        diagnostics_png,
        diagnostics_pdf,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate exact HDF5 matches without analysing or writing output.",
    )
    parser.add_argument(
        "--alpha-grid",
        choices=("existing", "refined"),
        default="existing",
        help=(
            "Use the exact six-point existing grid or validate/run the requested "
            "0.1-pi refined grid. Missing exact files stop before any output."
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate definitions and figures from the existing metrics CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alpha_grid = (
        CONFIG.alpha_over_pi
        if args.alpha_grid == "existing"
        else CONFIG.refined_alpha_over_pi
    )
    effective_config = replace(CONFIG, alpha_over_pi=alpha_grid)
    if args.plots_only:
        csv_path = (
            effective_config.output_dir
            / "Phase_Informed_Boundary_Flow_Metrics.csv"
        )
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
        table = pd.read_csv(csv_path)
        outputs = [
            write_definitions(effective_config),
            *plot_metrics(table, effective_config),
            *plot_diagnostics(table, effective_config),
        ]
        print("Created:")
        for path in outputs:
            print(f"  {path}")
        return 0
    if args.check_only:
        jobs = all_jobs(effective_config)
        try:
            bda.validate_exact_files([model for _, _, model in jobs])
        except bda.DataContractError as exc:
            print(f"STOP: {exc}")
            return 2
        print(f"Validated {len(jobs)} exact existing HDF5 files; no output written.")
        return 0
    try:
        outputs = run(effective_config)
    except bda.DataContractError as exc:
        print(f"STOP: {exc}")
        return 2
    print("Created:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
