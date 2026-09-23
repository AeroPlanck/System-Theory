"""Cucker-Smale weighted particle models and terminal-state phase diagrams.

The dynamics follow the phase-lag model in ``main.py`` but replace the hard
indicator ``1[r <= D0]`` by the smooth communication weight

    psi_q,D0(r) = (1 + (r / D0) ** (2 q)) ** (-1 / 2).

For q=1 this is the scaled beta=1 Cucker-Smale weight used by Ha, Jeong and
Kang (Nonlinearity 23, 2010).  Increasing q keeps the transition centred at
D0 and makes it sharper.  In the q -> infinity limit, the kernel converges to
the hard indicator used in ``main.py`` (apart from the immaterial value exactly
at r=D0).  The weighted local average retains the zero self-coupling term with
weight one; this keeps the normalization smooth and makes an isolated
particle's interaction vanish in the hard-cutoff limit.

Three boundary classes are provided:

* ``SquarePeriodicBoundary``
* ``CircularCollisionBoundary``
* ``CircularSingleDefectCollisionBoundary`` (one inward triangular notch)

The command-line workflow sweeps high coupling strengths K>=15, equivalent
interaction radius D0, and at least three kernel sharpness values at
alpha=0.6*pi.  Every trajectory is retained as HDF5 data with the existing
``positionX`` and ``phaseTheta`` keys.  Runs shorter than 50,000 iterations are
rejected, so the default workflow cannot create short-lived production data.
Only terminal-state figures are produced; no video code is invoked.

Example (the default 3 x 2 x 3 high-coupling sweep)::

    python "粒子算法.py" --boundary circle-collision --generate-missing

"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numba as nb
import numpy as np
import pandas as pd

from main import (
    CollisionBoundaryMidpointSpikePatternFormation as _ReferenceSingleDefect,
    CircularBoundaryPatternFormation as _ReferenceCircle,
    phaseCmap,
    phaseNorm,
)
from swarmalatorlib.template import Swarmalators2D


PROJECT_DIR = Path(__file__).resolve().parent
PHASE_LAG_A0 = 0.6 * np.pi

# Production sweep: high-coupling sector only, with two exact D0 values
# starting at 1 as required for the reduced terminal phase diagram.
MINIMUM_COUPLING_STRENGTH = 15.0
MINIMUM_SAVED_ITERATIONS = 50_000
DEFAULT_AGENTS = 2_000
DEFAULT_ITERATIONS = 50_000
DEFAULT_SNAPSHOT_INTERVAL = 100
DEFAULT_WORKERS = 12
DEFAULT_STRENGTH_VALUES = (15.0, 17.5, 20.0)
DEFAULT_DISTANCE_VALUES = (1.0, 2.0)

# q=16, 4, 1 correspond to strong, medium and weak cutoff smoothing.
DEFAULT_KERNEL_SHARPNESS = (16.0, 4.0, 1.0)
SHARPNESS_LABELS = {16.0: "strong", 4.0: "medium", 1.0: "weak"}

DEFAULT_DATA_ROOT = PROJECT_DIR / "data" / "Cucker_Smale_Phase_Diagram"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output" / "Cucker_Smale_Phase_Diagram"
HARD_WORKER_LIMIT = 12


class DataContractError(RuntimeError):
    """Raised when a required trajectory is absent or has an invalid schema."""


def _saved_frame_count(iterations: int, snapshot_interval: int) -> int:
    """Return the frame count written by the template run loop."""

    regular_frames = iterations // snapshot_interval + 1
    return regular_frames + int(iterations % snapshot_interval != 0)


def cucker_smale_communication_weight(
    distance: np.ndarray | float,
    interaction_radius: float,
    kernel_sharpness: float,
) -> np.ndarray | float:
    """Return the generalized, cutoff-centred Cucker-Smale weight.

    ``kernel_sharpness=1`` recovers the scaled beta=1 Cucker-Smale weight
    ``(1 + (r/D0)^2)^(-1/2)``.  All q>0 values are positive, monotone and
    continuous, so no discrete neighbour mask is used.
    """

    if interaction_radius <= 0:
        raise ValueError("interaction_radius must be positive.")
    if kernel_sharpness <= 0:
        raise ValueError("kernel_sharpness must be positive.")
    values = np.asarray(distance, dtype=float)
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("distance must contain finite non-negative values.")
    result = (1.0 + (values / interaction_radius) ** (2.0 * kernel_sharpness)) ** -0.5
    if np.isscalar(distance):
        return float(result)
    return result


@nb.njit
def _cs_weight_from_scaled_squared(
    scaled_squared: float, kernel_sharpness: float
) -> float:
    """Fast exact paths for the three production q values."""

    if kernel_sharpness == 1.0:
        powered = scaled_squared
    elif kernel_sharpness == 4.0:
        squared = scaled_squared * scaled_squared
        powered = squared * squared
    elif kernel_sharpness == 16.0:
        squared = scaled_squared * scaled_squared
        fourth = squared * squared
        eighth = fourth * fourth
        powered = eighth * eighth
    else:
        powered = scaled_squared**kernel_sharpness
    return (1.0 + powered) ** -0.5


@nb.njit
def _calc_dot_phase_euclidean_cs(
    position_x: np.ndarray,
    phase_theta: np.ndarray,
    freq_omega: np.ndarray,
    interaction_radius: float,
    kernel_sharpness: float,
    strength_k: float,
    phase_lag_a0: float,
) -> np.ndarray:
    """Allocation-light all-pairs CS coupling for non-periodic boundaries."""

    agents_num = position_x.shape[0]
    radius_squared = interaction_radius * interaction_radius
    weighted_sum = np.zeros(agents_num, dtype=np.float64)
    # psi(0)=1 is retained in the denominator.  Its centred self-signal is
    # exactly zero, so it regularizes isolated particles without adding torque.
    weight_sum = np.ones(agents_num, dtype=np.float64)
    phase_sin = np.sin(phase_theta)
    phase_cos = np.cos(phase_theta)
    lag_sin = np.sin(phase_lag_a0)
    lag_cos = np.cos(phase_lag_a0)

    for i in range(agents_num - 1):
        for j in range(i + 1, agents_num):
            dx = position_x[j, 0] - position_x[i, 0]
            dy = position_x[j, 1] - position_x[i, 1]
            scaled_squared = (dx * dx + dy * dy) / radius_squared
            weight = _cs_weight_from_scaled_squared(
                scaled_squared, kernel_sharpness
            )
            sin_delta = phase_sin[j] * phase_cos[i] - phase_cos[j] * phase_sin[i]
            cos_delta_minus_one = (
                phase_cos[j] * phase_cos[i]
                + phase_sin[j] * phase_sin[i]
                - 1.0
            )
            symmetric_part = cos_delta_minus_one * lag_sin
            antisymmetric_part = sin_delta * lag_cos
            weighted_sum[i] += weight * (symmetric_part + antisymmetric_part)
            weighted_sum[j] += weight * (symmetric_part - antisymmetric_part)
            weight_sum[i] += weight
            weight_sum[j] += weight

    dot_phase = freq_omega.copy()
    for i in range(agents_num):
        if weight_sum[i] > 0.0:
            dot_phase[i] += strength_k * weighted_sum[i] / weight_sum[i]
    return dot_phase


@nb.njit
def _calc_dot_phase_periodic_cs(
    position_x: np.ndarray,
    phase_theta: np.ndarray,
    freq_omega: np.ndarray,
    boundary_length: float,
    interaction_radius: float,
    kernel_sharpness: float,
    strength_k: float,
    phase_lag_a0: float,
) -> np.ndarray:
    """All-pairs CS coupling using the square minimum-image convention."""

    agents_num = position_x.shape[0]
    radius_squared = interaction_radius * interaction_radius
    half_length = 0.5 * boundary_length
    weighted_sum = np.zeros(agents_num, dtype=np.float64)
    # Retain the zero self-coupling term in the smooth local normalization.
    weight_sum = np.ones(agents_num, dtype=np.float64)
    phase_sin = np.sin(phase_theta)
    phase_cos = np.cos(phase_theta)
    lag_sin = np.sin(phase_lag_a0)
    lag_cos = np.cos(phase_lag_a0)

    for i in range(agents_num - 1):
        for j in range(i + 1, agents_num):
            dx = position_x[j, 0] - position_x[i, 0]
            dy = position_x[j, 1] - position_x[i, 1]
            if dx > half_length:
                dx -= boundary_length
            elif dx < -half_length:
                dx += boundary_length
            if dy > half_length:
                dy -= boundary_length
            elif dy < -half_length:
                dy += boundary_length
            scaled_squared = (dx * dx + dy * dy) / radius_squared
            weight = _cs_weight_from_scaled_squared(
                scaled_squared, kernel_sharpness
            )
            sin_delta = phase_sin[j] * phase_cos[i] - phase_cos[j] * phase_sin[i]
            cos_delta_minus_one = (
                phase_cos[j] * phase_cos[i]
                + phase_sin[j] * phase_sin[i]
                - 1.0
            )
            symmetric_part = cos_delta_minus_one * lag_sin
            antisymmetric_part = sin_delta * lag_cos
            weighted_sum[i] += weight * (symmetric_part + antisymmetric_part)
            weighted_sum[j] += weight * (symmetric_part - antisymmetric_part)
            weight_sum[i] += weight
            weight_sum[j] += weight

    dot_phase = freq_omega.copy()
    for i in range(agents_num):
        if weight_sum[i] > 0.0:
            dot_phase[i] += strength_k * weighted_sum[i] / weight_sum[i]
    return dot_phase


class CuckerSmaleParticleModel(Swarmalators2D):
    """Common phase-lag dynamics with a continuous Cucker-Smale kernel."""

    periodic_distance = False
    file_tag = "CS"

    def __init__(
        self,
        strengthK: float,
        distanceD0: float,
        kernelSharpness: float,
        phaseLagA0: float = PHASE_LAG_A0,
        boundaryLength: float = 7.0,
        speedV: float = 3.0,
        freqDist: str = "uniform",
        initPhaseTheta: np.ndarray | None = None,
        omegaMin: float = 0.0,
        deltaOmega: float = 0.0,
        agentsNum: int = 400,
        dt: float = 0.005,
        tqdm: bool = False,
        savePath: str | None = None,
        shotsnaps: int = 50,
        randomSeed: int = 9,
        overWrite: bool = False,
    ) -> None:
        if strengthK < 0:
            raise ValueError("strengthK must be non-negative.")
        if distanceD0 <= 0:
            raise ValueError("distanceD0 must be positive.")
        if kernelSharpness <= 0:
            raise ValueError("kernelSharpness must be positive.")
        if boundaryLength <= 0 or speedV < 0 or dt <= 0:
            raise ValueError("boundaryLength/dt must be positive and speedV non-negative.")
        if agentsNum <= 0 or agentsNum % 2:
            raise ValueError("agentsNum must be a positive even integer.")
        if shotsnaps <= 0:
            raise ValueError("shotsnaps must be positive.")
        if freqDist not in {"uniform", "cauchy", "identical"}:
            raise ValueError("freqDist must be uniform, cauchy, or identical.")

        self.strengthK = float(strengthK)
        self.distanceD0 = float(distanceD0)
        self.kernelSharpness = float(kernelSharpness)
        self.phaseLagA0 = float(phaseLagA0)
        self.boundaryLength = float(boundaryLength)
        self.halfBoundaryLength = 0.5 * self.boundaryLength
        self.speedV = float(speedV)
        self.freqDist = freqDist
        self.initPhaseTheta = initPhaseTheta
        self.omegaMin = float(omegaMin)
        self.deltaOmega = float(deltaOmega)
        self.agentsNum = int(agentsNum)
        self.dt = float(dt)
        self.tqdm = bool(tqdm)
        self.savePath = savePath
        self.shotsnaps = int(shotsnaps)
        self.randomSeed = int(randomSeed)
        self.overWrite = bool(overWrite)
        self.counts = 0
        self.store = None
        self.temp = {}

        rng = np.random.default_rng(self.randomSeed)
        self.positionX = self._initial_positions(rng)
        self.phaseTheta = rng.random(self.agentsNum) * 2.0 * np.pi
        if initPhaseTheta is not None:
            phases = np.asarray(initPhaseTheta, dtype=float)
            if phases.shape != (self.agentsNum,) or not np.isfinite(phases).all():
                raise ValueError("initPhaseTheta must be finite with shape (agentsNum,).")
            self.phaseTheta = phases.copy()

        half_num = self.agentsNum // 2
        if self.freqDist == "uniform":
            positive = rng.uniform(
                self.omegaMin,
                self.omegaMin + self.deltaOmega,
                half_num,
            )
        elif self.freqDist == "identical":
            positive = np.full(half_num, self.omegaMin)
        else:
            positive = np.abs(rng.standard_cauchy(half_num))
        self.freqOmega = np.sort(np.concatenate([positive, -positive]))

    def _initial_positions(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @nb.njit
    def _direction(phase_theta: np.ndarray) -> np.ndarray:
        result = np.empty((phase_theta.shape[0], 2), dtype=np.float64)
        result[:, 0] = np.cos(phase_theta)
        result[:, 1] = np.sin(phase_theta)
        return result

    @property
    def dotPosition(self) -> np.ndarray:
        return self.speedV * self._direction(self.phaseTheta)

    @property
    def dotPhase(self) -> np.ndarray:
        if self.periodic_distance:
            return _calc_dot_phase_periodic_cs(
                self.positionX,
                self.phaseTheta,
                self.freqOmega,
                self.boundaryLength,
                self.distanceD0,
                self.kernelSharpness,
                self.strengthK,
                self.phaseLagA0,
            )
        return _calc_dot_phase_euclidean_cs(
            self.positionX,
            self.phaseTheta,
            self.freqOmega,
            self.distanceD0,
            self.kernelSharpness,
            self.strengthK,
            self.phaseLagA0,
        )

    @property
    def communicationWeight(self) -> np.ndarray:
        """Dense communication matrix for diagnostics, with a zero diagonal."""

        delta = self.positionX[:, None, :] - self.positionX[None, :, :]
        if self.periodic_distance:
            delta -= self.boundaryLength * np.round(delta / self.boundaryLength)
        distances = np.linalg.norm(delta, axis=-1)
        weights = cucker_smale_communication_weight(
            distances, self.distanceD0, self.kernelSharpness
        )
        np.fill_diagonal(weights, 0.0)
        return weights

    def _advance_positions(
        self, new_position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update(self) -> None:
        dot_position = self.dotPosition
        dot_phase = self.dotPhase
        new_position = self.positionX + dot_position * self.dt
        self.positionX, corrected_velocity = self._advance_positions(
            new_position, dot_position
        )
        collision = np.any(
            ~np.isclose(corrected_velocity, dot_position, rtol=0.0, atol=1e-12),
            axis=1,
        )
        if np.any(collision):
            reflected_phase = np.arctan2(
                corrected_velocity[:, 1], corrected_velocity[:, 0]
            )
            self.phaseTheta[collision] = reflected_phase[collision]
        self.phaseTheta = np.mod(
            self.phaseTheta + dot_phase * self.dt, 2.0 * np.pi
        )

    def append(self) -> None:
        if self.store is None or self.counts % self.shotsnaps:
            return
        self.store.append("positionX", pd.DataFrame(self.positionX))
        self.store.append("phaseTheta", pd.DataFrame(self.phaseTheta))

    def _existing_progress(self, path: Path) -> int:
        """Validate an existing trajectory and return its last saved step."""

        try:
            with pd.HDFStore(path, mode="r") as store:
                required = {"/positionX", "/phaseTheta"}
                if not required.issubset(store.keys()):
                    raise DataContractError(
                        f"{path} lacks positionX or phaseTheta and cannot resume."
                    )
                position_store = store.get_storer("positionX")
                phase_store = store.get_storer("phaseTheta")
                if position_store.ncols != 2 or phase_store.ncols != 1:
                    raise DataContractError(
                        f"Unexpected HDF5 column schema in {path}."
                    )
                position_rows = int(position_store.nrows)
                phase_rows = int(phase_store.nrows)
                if (
                    position_rows != phase_rows
                    or position_rows < self.agentsNum
                    or position_rows % self.agentsNum
                ):
                    raise DataContractError(
                        f"Incomplete or unaligned frames in {path}; refusing to append."
                    )
                frames = position_rows // self.agentsNum
                recorded = getattr(
                    position_store.attrs,
                    "completed_iterations",
                    getattr(position_store.attrs, "requested_iterations", None),
                )
                if recorded is not None:
                    recorded = int(recorded)
                if (
                    recorded is None
                    or _saved_frame_count(recorded, self.shotsnaps) != frames
                ):
                    # An interrupted run has no final metadata update.  Every
                    # durable intermediate frame is written on a snapshot
                    # boundary, so the last complete one is safe to resume.
                    recorded = (frames - 1) * self.shotsnaps
                positions = store.select(
                    "positionX", start=position_rows - self.agentsNum
                ).to_numpy()
                phases = store.select(
                    "phaseTheta", start=phase_rows - self.agentsNum
                ).to_numpy().reshape(-1)
        except DataContractError:
            raise
        except Exception as exc:
            raise DataContractError(f"Cannot inspect resumable HDF5 {path}: {exc}") from exc
        if not np.isfinite(positions).all() or not np.isfinite(phases).all():
            raise DataContractError(f"Non-finite terminal frame in {path}.")
        self.positionX = positions
        self.phaseTheta = phases
        return recorded

    def run(self, TNum: int) -> None:
        """Run or safely resume to the requested *total* iteration count."""

        target_iterations = int(TNum)
        if target_iterations <= 0:
            raise ValueError("TNum must be positive.")

        if self.savePath is None:
            self.store = None
            start_iteration = 0
            self.append()
        else:
            save_dir = Path(self.savePath)
            save_dir.mkdir(parents=True, exist_ok=True)
            target_path = save_dir / f"{self}.h5"
            if target_path.is_file():
                start_iteration = self._existing_progress(target_path)
                if start_iteration >= target_iterations:
                    print(
                        f"{target_path} already reaches step {start_iteration}; skipped."
                    )
                    return
                self.counts = start_iteration
                self.store = pd.HDFStore(target_path, mode="a")
                print(
                    f"Resuming {target_path.name}: step {start_iteration} -> "
                    f"{target_iterations}",
                    flush=True,
                )
            else:
                start_iteration = 0
                self.counts = 0
                self.store = pd.HDFStore(target_path, mode="w")
                self.append()

        try:
            for index in range(start_iteration, target_iterations):
                self.update()
                self.counts = index + 1
                self.append()
            self.append_final()
        finally:
            self.close()

    def draw_boundary(self, axis: plt.Axes) -> None:
        raise NotImplementedError

    def plot(
        self,
        ax: plt.Axes | None = None,
        colorsBy: str = "phase",
        showColorbar: bool = True,
    ) -> plt.Axes:
        if colorsBy not in {"phase", "freq"}:
            raise ValueError("colorsBy must be phase or freq.")
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        color_values = self.phaseTheta if colorsBy == "phase" else self.freqOmega
        color_map = phaseCmap if colorsBy == "phase" else "coolwarm"
        quiver = ax.quiver(
            self.positionX[:, 0],
            self.positionX[:, 1],
            np.cos(self.phaseTheta),
            np.sin(self.phaseTheta),
            color_values,
            cmap=color_map,
            norm=phaseNorm if colorsBy == "phase" else None,
            scale_units="inches",
            scale=15.0,
            width=0.003,
        )
        self.draw_boundary(ax)
        if showColorbar:
            ax.figure.colorbar(quiver, ax=ax)
        return ax

    def _geometry_string(self) -> str:
        return ""

    def __str__(self) -> str:
        return (
            f"{self.file_tag}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"q={self.kernelSharpness:.3f},A0={self.phaseLagA0:.3f},"
            f"L={self.boundaryLength:.1f},{self._geometry_string()}"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"wMin={self.omegaMin:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed})"
        )


class SquarePeriodicBoundary(CuckerSmaleParticleModel):
    """Square periodic boundary with minimum-image communication distance."""

    periodic_distance = True
    file_tag = "CSP"

    def _initial_positions(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random((self.agentsNum, 2)) * self.boundaryLength

    def _advance_positions(
        self, new_position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.mod(new_position, self.boundaryLength), velocity.copy()

    def draw_boundary(self, axis: plt.Axes) -> None:
        axis.set_xlim(0.0, self.boundaryLength)
        axis.set_ylim(0.0, self.boundaryLength)
        axis.set_aspect("equal", adjustable="box")


class CircularCollisionBoundary(CuckerSmaleParticleModel):
    """Circular specular-reflection boundary."""

    file_tag = "CSC"

    def __init__(self, *args, **kwargs) -> None:
        boundary_length = float(kwargs.get("boundaryLength", 7.0))
        self.circleCenter = np.array(
            [0.5 * boundary_length, 0.5 * boundary_length], dtype=float
        )
        self.circleRadius = 0.5 * boundary_length
        super().__init__(*args, **kwargs)

    def _initial_positions(self, rng: np.random.Generator) -> np.ndarray:
        angles = rng.random(self.agentsNum) * 2.0 * np.pi
        radii = np.sqrt(rng.random(self.agentsNum)) * self.circleRadius
        return self.circleCenter + np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )

    def _advance_positions(
        self, new_position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return _ReferenceCircle._handle_collision_circle(
            new_position, velocity, self.circleCenter, self.circleRadius
        )

    def draw_boundary(self, axis: plt.Axes) -> None:
        axis.add_patch(
            plt.Circle(
                self.circleCenter,
                self.circleRadius,
                fill=False,
                color="#333333",
                linewidth=0.8,
            )
        )
        pad = 0.02 * self.boundaryLength
        axis.set_xlim(-pad, self.boundaryLength + pad)
        axis.set_ylim(-pad, self.boundaryLength + pad)
        axis.set_aspect("equal", adjustable="box")


class CircularSingleDefectCollisionBoundary(CircularCollisionBoundary):
    """Circular collision wall with one inward triangular boundary defect."""

    file_tag = "CSD"

    def __init__(
        self,
        *args,
        defectHeight: float = 0.8,
        defectHalfWidth: float | None = None,
        **kwargs,
    ) -> None:
        boundary_length = float(kwargs.get("boundaryLength", 7.0))
        radius = 0.5 * boundary_length
        if defectHalfWidth is None:
            defectHalfWidth = max(1e-6, 0.35 * defectHeight)
        if not 0.0 <= defectHeight < radius:
            raise ValueError("defectHeight must be in [0, circle radius).")
        if not 0.0 < defectHalfWidth < radius:
            raise ValueError("defectHalfWidth must be in (0, circle radius).")
        self.defectHeight = float(defectHeight)
        self.defectHalfWidth = float(defectHalfWidth)
        center = np.array([radius, radius], dtype=float)
        self.spikeTip, self.spikeBaseLeft, self.spikeBaseRight = (
            _ReferenceSingleDefect._build_spike_geometry(
                center, radius, self.defectHeight, self.defectHalfWidth
            )
        )
        self.boundaryVertices = _ReferenceSingleDefect._build_spike_boundary_vertices(
            center, radius, self.defectHeight, self.defectHalfWidth
        )
        super().__init__(*args, **kwargs)

    def _initial_positions(self, rng: np.random.Generator) -> np.ndarray:
        positions = super()._initial_positions(rng)
        valid = _ReferenceSingleDefect._points_inside_spiked_circle(
            positions,
            self.circleCenter,
            self.circleRadius,
            self.spikeBaseLeft,
            self.spikeTip,
            self.spikeBaseRight,
        )
        while not np.all(valid):
            count = int(np.count_nonzero(~valid))
            angles = rng.random(count) * 2.0 * np.pi
            radii = np.sqrt(rng.random(count)) * self.circleRadius
            positions[~valid] = self.circleCenter + np.column_stack(
                [radii * np.cos(angles), radii * np.sin(angles)]
            )
            valid = _ReferenceSingleDefect._points_inside_spiked_circle(
                positions,
                self.circleCenter,
                self.circleRadius,
                self.spikeBaseLeft,
                self.spikeTip,
                self.spikeBaseRight,
            )
        return positions

    def _advance_positions(
        self, new_position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return _ReferenceSingleDefect._handle_collision_spiked_circle(
            new_position,
            velocity,
            self.circleCenter,
            self.circleRadius,
            self.spikeBaseLeft,
            self.spikeTip,
            self.spikeBaseRight,
        )

    def draw_boundary(self, axis: plt.Axes) -> None:
        boundary = np.vstack([self.boundaryVertices, self.boundaryVertices[0]])
        axis.plot(
            boundary[:, 0], boundary[:, 1], color="#333333", linewidth=0.8
        )
        pad = 0.02 * self.boundaryLength
        axis.set_xlim(-pad, self.boundaryLength + pad)
        axis.set_ylim(-pad, self.boundaryLength + pad)
        axis.set_aspect("equal", adjustable="box")

    def _geometry_string(self) -> str:
        return f"H={self.defectHeight:.3f},W={self.defectHalfWidth:.3f},"


BOUNDARY_CLASSES: dict[str, type[CuckerSmaleParticleModel]] = {
    "square-periodic": SquarePeriodicBoundary,
    "circle-collision": CircularCollisionBoundary,
    "circle-single-defect": CircularSingleDefectCollisionBoundary,
}


@dataclass(frozen=True)
class SweepConfig:
    boundary: str = "circle-collision"
    boundaryLength: float = 7.0
    speedV: float = 3.0
    freqDist: str = "uniform"
    omegaMin: float = 0.0
    deltaOmega: float = 0.0
    agentsNum: int = DEFAULT_AGENTS
    dt: float = 0.005
    shotsnaps: int = DEFAULT_SNAPSHOT_INTERVAL
    randomSeed: int = 9
    iterations: int = DEFAULT_ITERATIONS
    defectHeight: float = 0.8
    defectHalfWidth: float | None = None


def build_model(
    boundary: str,
    strength_k: float,
    distance_d0: float,
    kernel_sharpness: float,
    config: SweepConfig,
    data_dir: Path,
) -> CuckerSmaleParticleModel:
    """Construct one exact member of the K-D0-q sweep."""

    try:
        model_class = BOUNDARY_CLASSES[boundary]
    except KeyError as exc:
        raise ValueError(f"Unknown boundary {boundary!r}.") from exc
    kwargs = {
        "strengthK": strength_k,
        "distanceD0": distance_d0,
        "kernelSharpness": kernel_sharpness,
        "phaseLagA0": PHASE_LAG_A0,
        "boundaryLength": config.boundaryLength,
        "speedV": config.speedV,
        "freqDist": config.freqDist,
        "omegaMin": config.omegaMin,
        "deltaOmega": config.deltaOmega,
        "agentsNum": config.agentsNum,
        "dt": config.dt,
        "tqdm": False,
        "savePath": str(data_dir),
        "shotsnaps": config.shotsnaps,
        "randomSeed": config.randomSeed,
        "overWrite": False,
    }
    if model_class is CircularSingleDefectCollisionBoundary:
        kwargs.update(
            defectHeight=config.defectHeight,
            defectHalfWidth=config.defectHalfWidth,
        )
    return model_class(**kwargs)


def data_path(model: CuckerSmaleParticleModel) -> Path:
    return Path(model.savePath) / f"{model}.h5"


def trajectory_progress(model: CuckerSmaleParticleModel) -> int | None:
    """Return validated saved progress, or None when the file is absent."""

    path = data_path(model)
    if not path.is_file():
        return None
    return model._existing_progress(path)


def build_models(
    boundary: str,
    strengths: Sequence[float],
    distances: Sequence[float],
    sharpness_values: Sequence[float],
    config: SweepConfig,
    data_dir: Path,
) -> list[CuckerSmaleParticleModel]:
    models = [
        build_model(boundary, strength, distance, sharpness, config, data_dir)
        for sharpness in sharpness_values
        for distance in distances
        for strength in strengths
    ]
    paths = [data_path(model) for model in models]
    if len(paths) != len(set(paths)):
        raise ValueError(
            "Sweep values collapse to duplicate filenames at three-decimal precision."
        )
    return models


def _run_one(job: tuple[str, float, float, float, SweepConfig, str]) -> str:
    boundary, strength, distance, sharpness, config, data_dir_text = job
    data_dir = Path(data_dir_text)
    data_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        boundary, strength, distance, sharpness, config, data_dir
    )
    model.run(config.iterations)
    path = data_path(model)
    with pd.HDFStore(path, mode="a") as store:
        for key in ("positionX", "phaseTheta"):
            attributes = store.get_storer(key).attrs
            attributes.kernel = "(1 + (r/D0)^(2q))^(-1/2)"
            attributes.kernel_sharpness = sharpness
            attributes.phase_lag_a0 = PHASE_LAG_A0
            attributes.requested_iterations = config.iterations
            attributes.completed_iterations = config.iterations
            attributes.saved_frame_count = _saved_frame_count(
                config.iterations, config.shotsnaps
            )
            attributes.video_generated = False
    return str(path)


def ensure_trajectories(
    models: Sequence[CuckerSmaleParticleModel],
    config: SweepConfig,
    workers: int,
) -> None:
    progress = [(model, trajectory_progress(model)) for model in models]
    pending = [
        model
        for model, completed_iterations in progress
        if completed_iterations is None or completed_iterations < config.iterations
    ]
    if not pending:
        print(
            "All exact parameter-matched HDF5 trajectories already reach "
            f"step {config.iterations}."
        )
        return
    if not 1 <= workers <= HARD_WORKER_LIMIT:
        raise ValueError(f"workers must be between 1 and {HARD_WORKER_LIMIT}.")
    worker_count = min(workers, len(pending))
    jobs = [
        (
            config.boundary,
            model.strengthK,
            model.distanceD0,
            model.kernelSharpness,
            config,
            str(Path(model.savePath)),
        )
        for model in pending
    ]
    new_count = sum(completed is None for _, completed in progress)
    resume_count = len(pending) - new_count
    print(
        f"Processing {len(jobs)} HDF5 trajectories with {worker_count} worker(s): "
        f"{new_count} new, {resume_count} resumable.",
        flush=True,
    )
    if worker_count == 1:
        for index, job in enumerate(jobs, start=1):
            _run_one(job)
            print(f"[{index}/{len(jobs)}] completed", flush=True)
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        futures = {executor.submit(_run_one, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f"[{index}/{len(jobs)}] completed", flush=True)


def load_terminal_state(
    model: CuckerSmaleParticleModel,
) -> tuple[np.ndarray, np.ndarray, int]:
    path = data_path(model)
    if not path.is_file():
        raise DataContractError(f"Missing HDF5 trajectory: {path}")
    try:
        with pd.HDFStore(path, mode="r") as store:
            required = {"/positionX", "/phaseTheta"}
            if not required.issubset(store.keys()):
                raise DataContractError(f"{path} lacks positionX or phaseTheta.")
            position_store = store.get_storer("positionX")
            phase_store = store.get_storer("phaseTheta")
            n = model.agentsNum
            if position_store.ncols != 2 or phase_store.ncols != 1:
                raise DataContractError(f"Unexpected HDF5 column schema in {path}.")
            if (
                position_store.nrows != phase_store.nrows
                or position_store.nrows < n
                or position_store.nrows % n
            ):
                raise DataContractError(f"Incomplete or unaligned frames in {path}.")
            frames = position_store.nrows // n
            positions = store.select(
                "positionX", start=position_store.nrows - n
            ).to_numpy()
            phases = store.select(
                "phaseTheta", start=phase_store.nrows - n
            ).to_numpy().reshape(-1)
    except DataContractError:
        raise
    except Exception as exc:
        raise DataContractError(f"Cannot read {path}: {exc}") from exc
    if not np.isfinite(positions).all() or not np.isfinite(phases).all():
        raise DataContractError(f"Non-finite terminal values in {path}.")
    return positions, phases, frames


def _draw_terminal_panel(
    axis: plt.Axes,
    model: CuckerSmaleParticleModel,
    positions: np.ndarray,
    phases: np.ndarray,
) -> None:
    axis.quiver(
        positions[:, 0],
        positions[:, 1],
        np.cos(phases),
        np.sin(phases),
        phases,
        cmap=phaseCmap,
        norm=phaseNorm,
        scale_units="inches",
        scale=15.0,
        width=0.003,
    )
    model.draw_boundary(axis)
    axis.set_xticks([])
    axis.set_yticks([])


def _sharpness_label(value: float) -> str:
    for key, label in SHARPNESS_LABELS.items():
        if np.isclose(value, key):
            return label
    return "custom"


def create_terminal_phase_diagram(
    models: Sequence[CuckerSmaleParticleModel],
    states: Sequence[tuple[np.ndarray, np.ndarray, int]],
    strengths: Sequence[float],
    distances: Sequence[float],
    sharpness: float,
    config: SweepConfig,
) -> plt.Figure:
    rows, columns = len(distances), len(strengths)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns + 0.7, 2.55 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for row, distance in enumerate(distances):
        for column, strength in enumerate(strengths):
            index = row * columns + column
            positions, phases, _ = states[index]
            _draw_terminal_panel(axes[row, column], models[index], positions, phases)
            if row == 0:
                axes[row, column].set_title(rf"$K={strength:g}$", fontsize=11)
        axes[row, 0].set_ylabel(rf"$D_0={distance:g}$", fontsize=11)

    label = _sharpness_label(sharpness)
    figure.suptitle(
        rf"{config.boundary}: terminal states, $\alpha=0.6\pi$, "
        rf"$q={sharpness:g}$ ({label} cutoff), $N={config.agentsNum}$",
        fontsize=14,
    )
    scalar = ScalarMappable(norm=phaseNorm, cmap=phaseCmap)
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        ax=list(axes.ravel()),
        ticks=[0.0, np.pi, 2.0 * np.pi],
        fraction=0.018,
        pad=0.015,
        aspect=max(24, 6 * rows),
    )
    colorbar.ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    colorbar.set_label(r"Phase $\theta$")
    return figure


def create_kernel_figure(
    sharpness_values: Sequence[float],
) -> plt.Figure:
    normalized_distance = np.linspace(0.0, 2.5, 600)
    figure, axis = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    for sharpness in sharpness_values:
        weight = cucker_smale_communication_weight(
            normalized_distance, 1.0, sharpness
        )
        axis.plot(
            normalized_distance,
            weight,
            linewidth=2.0,
            label=rf"$q={sharpness:g}$ ({_sharpness_label(sharpness)})",
        )
    axis.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    axis.text(1.02, 0.04, r"$r=D_0$", transform=axis.get_xaxis_transform())
    axis.set(
        xlabel=r"Normalized distance $r/D_0$",
        ylabel=r"Communication weight $\psi_{q,D_0}(r)$",
        xlim=(0.0, 2.5),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    return figure


def save_figure_pair(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    # ``stem`` intentionally contains decimal parameter values such as
    # ``Alpha0.6pi``.  with_suffix() would treat the part after that decimal as
    # an existing extension and silently collapse distinct q values.
    png_path = Path(f"{stem}.png")
    pdf_path = Path(f"{stem}.pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return png_path, pdf_path


def write_manifest(
    output_dir: Path,
    data_dir: Path,
    strengths: Sequence[float],
    distances: Sequence[float],
    sharpness_values: Sequence[float],
    config: SweepConfig,
    source_frame_counts: dict[str, int],
) -> Path:
    manifest = {
        "model_family": "CuckerSmaleParticleModel",
        "boundary_class": BOUNDARY_CLASSES[config.boundary].__name__,
        "hdf5_filename_prefix": BOUNDARY_CLASSES[config.boundary].file_tag,
        "kernel": "(1 + (r/D0)^(2q))^(-1/2)",
        "kernel_reference": (
            "Scaled q=1 case matches the beta=1 Cucker-Smale communication "
            "weight in Ha, Jeong & Kang, Nonlinearity 23 (2010), section 2.1, "
            "following condition (2.2)."
        ),
        "hard_cutoff_limit": "q -> infinity",
        "coupling_normalization": (
            "sum of pair weights plus the unit self-weight; the centred "
            "self-coupling numerator is exactly zero"
        ),
        "sharpness_values": [float(value) for value in sharpness_values],
        "sharpness_labels": {
            str(value): _sharpness_label(value) for value in sharpness_values
        },
        "strength_values": [float(value) for value in strengths],
        "distance_values": [float(value) for value in distances],
        "production_policy": {
            "minimum_coupling_strength": MINIMUM_COUPLING_STRENGTH,
            "minimum_saved_iterations": MINIMUM_SAVED_ITERATIONS,
            "terminal_figures_only": True,
            "video_generated": False,
        },
        "phase_lag_a0": float(PHASE_LAG_A0),
        "alpha_over_pi": 0.6,
        "config": asdict(config),
        "data_directory": str(data_dir),
        "source_frame_counts": {
            name: int(count) for name, count in source_frame_counts.items()
        },
        "video_generated": False,
        "hdf5_keys": ["positionX", "phaseTheta"],
    }
    path = output_dir / (
        f"Cucker_Smale_Phase_Diagram_Configuration_{config.boundary}_"
        f"N{config.agentsNum}_Steps{config.iterations}_Seed{config.randomSeed}.json"
    )
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_workflow(
    strengths: Sequence[float],
    distances: Sequence[float],
    sharpness_values: Sequence[float],
    config: SweepConfig,
    data_dir: Path,
    output_dir: Path,
    generate_missing: bool,
    workers: int,
    check_only: bool,
) -> list[Path]:
    if len(sharpness_values) < 3:
        raise ValueError("At least three kernel sharpness values are required.")
    if not strengths or not distances:
        raise ValueError("strengths and distances must be non-empty.")
    if config.iterations < MINIMUM_SAVED_ITERATIONS:
        raise ValueError(
            f"Production runs must contain at least {MINIMUM_SAVED_ITERATIONS} "
            "iterations; shorter datasets are not generated or plotted."
        )
    if any(
        value < MINIMUM_COUPLING_STRENGTH or not np.isfinite(value)
        for value in strengths
    ):
        raise ValueError(
            "Every coupling strength must be finite and at least "
            f"{MINIMUM_COUPLING_STRENGTH:g}."
        )
    if any(value <= 0 or not np.isfinite(value) for value in distances):
        raise ValueError("Every interaction radius must be finite and positive.")
    if any(value <= 0 or not np.isfinite(value) for value in sharpness_values):
        raise ValueError("Every kernel sharpness must be finite and positive.")

    models = build_models(
        config.boundary,
        strengths,
        distances,
        sharpness_values,
        config,
        data_dir,
    )
    if generate_missing:
        data_dir.mkdir(parents=True, exist_ok=True)
        ensure_trajectories(models, config, workers)

    missing = [data_path(model) for model in models if not data_path(model).is_file()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing[:12])
        suffix = "\n  ..." if len(missing) > 12 else ""
        raise DataContractError(
            "Exact HDF5 trajectories are missing. Re-run with "
            f"--generate-missing:\n{listing}{suffix}"
        )

    states = [load_terminal_state(model) for model in models]
    if check_only:
        print(f"Validated {len(states)} HDF5 trajectories; no figure was written.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    kernel_figure = create_kernel_figure(sharpness_values)
    outputs.extend(
        save_figure_pair(kernel_figure, output_dir / "Cucker_Smale_Kernel_Comparison")
    )

    panels_per_sharpness = len(strengths) * len(distances)
    frame_counts: dict[str, int] = {}
    for sharpness_index, sharpness in enumerate(sharpness_values):
        start = sharpness_index * panels_per_sharpness
        stop = start + panels_per_sharpness
        subset_models = models[start:stop]
        subset_states = states[start:stop]
        for model, (_, _, frames) in zip(subset_models, subset_states):
            frame_counts[data_path(model).name] = frames
        figure = create_terminal_phase_diagram(
            subset_models,
            subset_states,
            strengths,
            distances,
            sharpness,
            config,
        )
        stem = output_dir / (
            f"Terminal_Phase_Diagram_{config.boundary}_"
            f"Alpha0.6pi_q{sharpness:g}_{_sharpness_label(sharpness)}_"
            f"N{config.agentsNum}_Steps{config.iterations}_Seed{config.randomSeed}"
        )
        outputs.extend(save_figure_pair(figure, stem))

    manifest_path = write_manifest(
        output_dir,
        data_dir,
        strengths,
        distances,
        sharpness_values,
        config,
        frame_counts,
    )
    outputs.append(manifest_path)
    print(f"Saved {manifest_path}")
    return outputs


def _parse_float_list(text: str, option_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a comma-separated list of numbers."
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option_name} cannot be empty.")
    return values


def _default_list(values: Iterable[float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary", choices=tuple(BOUNDARY_CLASSES), default="circle-collision"
    )
    parser.add_argument(
        "--strength-values", default=_default_list(DEFAULT_STRENGTH_VALUES)
    )
    parser.add_argument(
        "--distance-values", default=_default_list(DEFAULT_DISTANCE_VALUES)
    )
    parser.add_argument(
        "--kernel-sharpness", default=_default_list(DEFAULT_KERNEL_SHARPNESS)
    )
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--boundary-length", type=float, default=7.0)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--defect-height", type=float, default=0.8)
    parser.add_argument("--defect-half-width", type=float, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate absent HDF5 trajectories before plotting.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all exact HDF5 files without writing figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strengths = _parse_float_list(args.strength_values, "--strength-values")
    distances = _parse_float_list(args.distance_values, "--distance-values")
    sharpness_values = _parse_float_list(
        args.kernel_sharpness, "--kernel-sharpness"
    )
    config = SweepConfig(
        boundary=args.boundary,
        boundaryLength=args.boundary_length,
        speedV=args.speed,
        agentsNum=args.agents,
        dt=args.dt,
        shotsnaps=args.snapshot_interval,
        randomSeed=args.seed,
        iterations=args.iterations,
        defectHeight=args.defect_height,
        defectHalfWidth=args.defect_half_width,
    )
    experiment_tag = (
        f"N{config.agentsNum}_K{min(strengths):g}-{max(strengths):g}_"
        f"Steps{config.iterations}"
    )
    data_dir = (
        (DEFAULT_DATA_ROOT / args.boundary / experiment_tag)
        if args.data_dir is None
        else args.data_dir.resolve()
    )
    output_dir = (
        (DEFAULT_OUTPUT_ROOT / args.boundary / experiment_tag)
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    try:
        run_workflow(
            strengths,
            distances,
            sharpness_values,
            config,
            data_dir,
            output_dir,
            args.generate_missing,
            args.workers,
            args.check_only,
        )
    except (DataContractError, ValueError) as exc:
        print(f"STOPPED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
