"""Direct boundary-lattice spacing from circular density autocorrelation.

No cluster-number or circumference-divided-by-count observable is evaluated.
The spacing is the first real-space peak of the along-boundary density
autocorrelation after the zero-lag self peak.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "Lattice_Scale_Comparison"
HALF_DATA = ROOT / "data" / "halfpi_boundary_N2000_steps50000_snap50"
PI_DATA = ROOT / "data" / "pi_endpoint_N2000_steps50000_snap50"

N = 2000
SNAP = 50
RADIUS = 3.5
STEPS = 50_000
SEEDS = tuple(range(1, 11))
BINS = 2048
HALF_SHELL = 0.25
PI_SHELL = 0.50
PHASE_CONFIDENCE = 0.50


def trajectory(directory: Path, seed: int) -> Path:
    matches = list(directory.glob(f"*seed={seed}).h5"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one trajectory for seed={seed}: {matches}")
    return matches[0]


def load_window(
    directory: Path,
    seed: int,
    start_iteration: int,
    stop_iteration: int,
    sample_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = trajectory(directory, seed)
    first_frame = start_iteration // SNAP
    last_frame = stop_iteration // SNAP
    start_row = first_frame * N
    stop_row = (last_frame + 1) * N
    with pd.HDFStore(path, mode="r") as store:
        rows_x = store.get_storer("positionX").nrows
        rows_t = store.get_storer("phaseTheta").nrows
        expected_rows = (STEPS // SNAP + 1) * N
        if rows_x != expected_rows or rows_t != expected_rows:
            raise RuntimeError(f"Incomplete trajectory: {path}")
        positions = store.select(
            "positionX", start=start_row, stop=stop_row
        ).to_numpy()
        phases = store.select(
            "phaseTheta", start=start_row, stop=stop_row
        ).to_numpy()
    frames = last_frame - first_frame + 1
    positions = positions.reshape(frames, N, 2)
    phases = phases.reshape(frames, N)
    stride = sample_step // SNAP
    selected = np.arange(0, frames, stride)
    iterations = (first_frame + selected) * SNAP
    return positions[selected], phases[selected], iterations, path


def wrap(angle: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * angle))


def autocorrelation_spacing(
    polar_angles: np.ndarray,
    effective_radius: float,
) -> tuple[float, float]:
    """Return the local first-neighbor autocorrelation peak and its height."""
    histogram, _ = np.histogram(
        np.mod(polar_angles, 2.0 * math.pi),
        bins=BINS,
        range=(0.0, 2.0 * math.pi),
    )
    density = gaussian_filter1d(histogram.astype(float), sigma=2.5, mode="wrap")
    fluctuation = density - density.mean()
    spectrum = np.fft.rfft(fluctuation)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n=BINS)
    if correlation[0] <= 0:
        raise RuntimeError("Degenerate boundary-density autocorrelation")
    correlation /= correlation[0]
    correlation = gaussian_filter1d(correlation, sigma=2.0, mode="wrap")

    arc_step = 2.0 * math.pi * effective_radius / BINS
    arc = np.arange(BINS // 2 + 1) * arc_step
    values = correlation[: BINS // 2 + 1]
    local = (arc >= 0.60) & (arc <= 1.80)
    local_indices = np.flatnonzero(local)
    local_values = values[local]
    minimum_distance = max(1, int(round(0.35 / arc_step)))
    prominence = max(0.01, 0.05 * float(local_values.max() - local_values.min()))
    peaks, properties = find_peaks(
        local_values,
        prominence=prominence,
        distance=minimum_distance,
    )
    if peaks.size == 0:
        raise RuntimeError("No first-neighbor boundary autocorrelation peak")
    chosen_local = int(peaks[np.argmax(properties["prominences"])])
    chosen = int(local_indices[chosen_local])

    # Quadratic sub-bin refinement of the real-space peak.
    y_left, y_mid, y_right = values[chosen - 1 : chosen + 2]
    denominator = y_left - 2.0 * y_mid + y_right
    offset = 0.0 if abs(denominator) < 1e-14 else 0.5 * (y_left - y_right) / denominator
    offset = float(np.clip(offset, -0.5, 0.5))
    spacing = (chosen + offset) * arc_step
    peak_height = float(y_mid - 0.5 * (y_left - y_right) * offset)
    return float(spacing), peak_height


def halfpi_seed(seed: int) -> dict[str, float | int | str]:
    positions, _, iterations, path = load_window(
        HALF_DATA, seed, 40_000, 50_000, 100
    )
    center = np.array([RADIUS, RADIUS])
    spacings: list[float] = []
    heights: list[float] = []
    radii: list[float] = []
    for frame_positions in positions:
        relative = frame_positions - center
        radial = np.linalg.norm(relative, axis=1)
        wall = RADIUS - radial
        polar = np.arctan2(relative[:, 1], relative[:, 0])
        selected = (wall >= -1e-9) & (wall <= HALF_SHELL)
        effective_radius = float(np.mean(radial[selected]))
        spacing, height = autocorrelation_spacing(
            polar[selected], effective_radius
        )
        spacings.append(spacing)
        heights.append(height)
        radii.append(effective_radius)
    return {
        "seed": seed,
        "source_file": str(path),
        "iteration_start": int(iterations[0]),
        "iteration_end": int(iterations[-1]),
        "sampled_frames": int(len(spacings)),
        "effective_radius_mean": float(np.mean(radii)),
        "boundary_spacing_mean": float(np.mean(spacings)),
        "boundary_spacing_time_std": float(np.std(spacings, ddof=1)),
        "autocorrelation_peak_height_mean": float(np.mean(heights)),
    }


def phase_families(
    positions: np.ndarray,
    phases: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    center = np.array([RADIUS, RADIUS])
    relative = positions - center
    radial = np.linalg.norm(relative, axis=1)
    wall = RADIUS - radial
    polar = np.arctan2(relative[:, 1], relative[:, 0])
    relative_phase = wrap(phases - polar)
    shell = (wall >= -1e-9) & (wall <= PI_SHELL)
    axial = np.mean(np.exp(2j * relative_phase[shell]))
    beta = 0.5 * float(np.angle(axial))
    projection = np.cos(relative_phase - beta)
    candidates = shell & (np.abs(projection) >= PHASE_CONFIDENCE)
    positive_axis = projection >= 0.0

    raw: list[tuple[np.ndarray, float]] = []
    for sign in (False, True):
        selected = candidates & (positive_axis == sign)
        tangential = float(np.mean(np.sin(relative_phase[selected])))
        raw.append((selected, tangential))
    ccw_index = int(np.argmax([item[1] for item in raw]))
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, (selected, _) in enumerate(raw):
        family = "CCW" if index == ccw_index else "CW"
        result[family] = (polar[selected], radial[selected])
    return result


def pi_seed(seed: int) -> list[dict[str, float | int | str]]:
    terminal_steps = int(math.ceil(10.0 * 2.0 * math.pi * RADIUS / 3.0 / 0.005))
    start = STEPS - terminal_steps
    start -= start % SNAP
    positions, phases, iterations, path = load_window(
        PI_DATA, seed, start, STEPS, SNAP
    )
    by_family: dict[str, dict[str, list[float]]] = {
        "CW": {"spacing": [], "height": [], "radius": []},
        "CCW": {"spacing": [], "height": [], "radius": []},
    }
    for frame_positions, frame_phases in zip(positions, phases):
        families = phase_families(frame_positions, frame_phases)
        for family, (polar, radial) in families.items():
            effective_radius = float(np.mean(radial))
            spacing, height = autocorrelation_spacing(polar, effective_radius)
            by_family[family]["spacing"].append(spacing)
            by_family[family]["height"].append(height)
            by_family[family]["radius"].append(effective_radius)

    rows: list[dict[str, float | int | str]] = []
    for family in ("CW", "CCW"):
        spacing = np.asarray(by_family[family]["spacing"])
        rows.append(
            {
                "seed": seed,
                "family": family,
                "source_file": str(path),
                "iteration_start": int(iterations[0]),
                "iteration_end": int(iterations[-1]),
                "sampled_frames": int(spacing.size),
                "effective_radius_mean": float(
                    np.mean(by_family[family]["radius"])
                ),
                "boundary_spacing_mean": float(np.mean(spacing)),
                "boundary_spacing_time_std": float(np.std(spacing, ddof=1)),
                "autocorrelation_peak_height_mean": float(
                    np.mean(by_family[family]["height"])
                ),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    half = pd.DataFrame([halfpi_seed(seed) for seed in SEEDS])
    pi_rows = [row for seed in SEEDS for row in pi_seed(seed)]
    pi = pd.DataFrame(pi_rows)
    half.to_csv(OUT / "HalfPi_Boundary_Arc_Correlation_10Seeds.csv", index=False)
    pi.to_csv(OUT / "Pi_Boundary_Arc_Correlation_10Seeds.csv", index=False)
    print("alpha=pi/2 direct boundary spacings")
    print(half.to_string(index=False))
    print("\nalpha=pi phase-resolved direct boundary spacings")
    print(pi.to_string(index=False))


if __name__ == "__main__":
    main()
