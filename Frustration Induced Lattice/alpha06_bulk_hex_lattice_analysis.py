"""Compare the alpha=0.6*pi bulk hexagonal lattice with the linear spectrum.

The microscopic objects measured here are vortex rotation centers.  Particle
positions themselves circle those centers and therefore do not directly give
the lattice constant.  Reference TeX and Dispersion.py files are read only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.special import j0
from sklearn.cluster import DBSCAN

from generate_missing_hdf5 import _calc_dot_phase_cell_list
from main import CircularBoundaryPatternFormation
from pi_endpoint_lattice_analysis import import_dispersion_module, most_unstable


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "Alpha06_Bulk_Hex_Lattice"
EXTRA_DATA = ROOT / "data" / "alpha06_bulk_N2000_steps50000_snap50"
SEED9_PATH = ROOT / "data" / (
    "CircularBoundaryPatternFormation(K=20.750,D0=1.000,A0=1.885,L=7.0,"
    "v=3.0,dist=uniform,wMin=0.000,dw=0.000,N=2000,dt=0.005,snap=10,seed=9).h5"
)
DISPERSION = Path(r"D:\PrivatePythonProject\Math\Lattice\Dispersion.py")
PRL = Path(r"D:\LaTex\Boundary Flow\PRL.tex")
METHODS = Path(r"D:\LaTex\Boundary Flow\Methods Appendix.tex")

EXPECTED_HASHES = {
    DISPERSION: "A1FC299F4AB13F9997BDF0EBA993C6BA12054500134A8617180F572F3732B89D",
    PRL: "8265AF6394ACD421FDE1E1163DC42B126AB33A8EEC0F019D91D4B4D5537BD7A6",
    METHODS: "CB0A459012329E1CCE7584152E55333467F8A48E1317C04DCA3DCCA72D07F7A8",
}

N = 2000
K = 20.75
D0 = 1.0
V = 3.0
ALPHA = 0.6 * math.pi
DIAMETER = 7.0
RADIUS = DIAMETER / 2.0
DT = 0.005
SEEDS = (1, 9, 17)
TERMINAL_TIME = 50.0
TARGET_SAMPLE_DT = 0.5

# Same model-derived scale used by main.py's calc_lattice_constants:
# the single-vortex orbit radius v/(K |sin alpha|).
DBSCAN_EPS = V / (K * abs(math.sin(ALPHA)))
DBSCAN_MIN_SAMPLES = 10
MINIMUM_VORTEX_PARTICLES = 15
BULK_WALL_EXCLUSION = 0.50 * D0
RATE_FRACTION_MINIMUM = 0.20
FIRST_SHELL_FACTOR = 1.45
K_GRID = np.linspace(2.5, 8.0, 1101)


@dataclass
class FrameAnalysis:
    iteration: int
    positions: np.ndarray
    phases: np.ndarray
    rotation_centers: np.ndarray
    vortex_centers: np.ndarray
    vortex_sizes: np.ndarray
    first_shell_pairs: np.ndarray
    first_shell_distances: np.ndarray
    median_nearest_neighbor: float
    cutoff: float
    global_psi6: float
    core_coordination_mean: float
    radial_structure: np.ndarray
    bragg_peak_k: float


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_references() -> dict[str, str]:
    values = {str(path): file_hash(path) for path in EXPECTED_HASHES}
    changed = [
        str(path)
        for path, expected in EXPECTED_HASHES.items()
        if values[str(path)] != expected
    ]
    if changed:
        raise RuntimeError("Read-only reference hash changed: " + ", ".join(changed))
    return values


def build_model(seed: int, snap: int) -> CircularBoundaryPatternFormation:
    return CircularBoundaryPatternFormation(
        strengthK=K,
        distanceD0=D0,
        phaseLagA0=ALPHA,
        boundaryLength=DIAMETER,
        speedV=V,
        freqDist="uniform",
        omegaMin=0.0,
        deltaOmega=0.0,
        agentsNum=N,
        dt=DT,
        tqdm=False,
        savePath=None,
        shotsnaps=snap,
        randomSeed=seed,
        overWrite=False,
    )


def trajectory(seed: int) -> tuple[Path, int]:
    if seed == 9:
        return SEED9_PATH, 10
    matches = list(EXTRA_DATA.glob(f"*seed={seed}).h5"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one long trajectory for seed={seed}: {matches}")
    return matches[0], 50


def load_terminal(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, Path]:
    path, snap = trajectory(seed)
    saved_dt = DT * snap
    with pd.HDFStore(path, mode="r") as store:
        rows_x = store.get_storer("positionX").nrows
        rows_t = store.get_storer("phaseTheta").nrows
        if rows_x != rows_t or rows_x % N:
            raise RuntimeError(f"Unaligned HDF5 trajectory: {path}")
        total_frames = rows_x // N
        raw_count = min(total_frames, int(math.ceil(TERMINAL_TIME / saved_dt)) + 1)
        start_frame = total_frames - raw_count
        start_row = start_frame * N
        positions = store.select("positionX", start=start_row).to_numpy()
        phases = store.select("phaseTheta", start=start_row).to_numpy()
    positions = positions.reshape(raw_count, N, 2)
    phases = phases.reshape(raw_count, N)
    stride = max(1, int(round(TARGET_SAMPLE_DT / saved_dt)))
    selected = np.arange(0, raw_count, stride)
    if selected[-1] != raw_count - 1:
        selected = np.r_[selected, raw_count - 1]
    iterations = (start_frame + selected) * snap
    return positions[selected], phases[selected], iterations, total_frames, path


def phase_rates(
    positions: np.ndarray,
    phases: np.ndarray,
    model: CircularBoundaryPatternFormation,
) -> np.ndarray:
    return _calc_dot_phase_cell_list(
        positions, phases, model.freqOmega, model.dotThetaParams
    )


def instantaneous_rotation_centers(
    positions: np.ndarray,
    phases: np.ndarray,
    rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    minimum_rate = RATE_FRACTION_MINIMUM * K * abs(math.sin(ALPHA))
    valid = np.isfinite(rates) & (np.abs(rates) >= minimum_rate)
    centers = np.full_like(positions, np.nan)
    centers[valid, 0] = positions[valid, 0] - V / rates[valid] * np.sin(phases[valid])
    centers[valid, 1] = positions[valid, 1] + V / rates[valid] * np.cos(phases[valid])
    radius = np.linalg.norm(centers - RADIUS, axis=1)
    valid &= np.isfinite(centers).all(axis=1) & (radius <= RADIUS + 0.20 * D0)
    return centers, valid


def cluster_vortex_centers(
    rotation_centers: np.ndarray,
    valid: np.ndarray,
    eps: float = DBSCAN_EPS,
    wall_exclusion: float = BULK_WALL_EXCLUSION,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    selected = rotation_centers[valid]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(selected)
    centers = []
    sizes = []
    for label in sorted(set(labels) - {-1}):
        member = labels == label
        if member.sum() < MINIMUM_VORTEX_PARTICLES:
            continue
        centers.append(np.median(selected[member], axis=0))
        sizes.append(int(member.sum()))
    if not centers:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    centers = np.asarray(centers)
    sizes = np.asarray(sizes, dtype=int)
    bulk = np.linalg.norm(centers - RADIUS, axis=1) <= RADIUS - wall_exclusion
    return centers[bulk], sizes[bulk]


def radial_structure_factor(centers: np.ndarray) -> np.ndarray:
    count = centers.shape[0]
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    pair_distances = distances[np.triu_indices(count, 1)]
    return 1.0 + (2.0 / count) * np.sum(
        j0(K_GRID[:, None] * pair_distances[None, :]), axis=1
    )


def analyse_centers(
    iteration: int,
    positions: np.ndarray,
    phases: np.ndarray,
    rotation_centers: np.ndarray,
    centers: np.ndarray,
    sizes: np.ndarray,
) -> FrameAnalysis:
    if centers.shape[0] < 7:
        raise RuntimeError(f"Only {centers.shape[0]} bulk vortices at iteration {iteration}")
    tree = cKDTree(centers)
    nearest = tree.query(centers, k=2)[0][:, 1]
    median_nearest = float(np.median(nearest))
    cutoff = FIRST_SHELL_FACTOR * median_nearest
    pair_set = sorted(tree.query_pairs(cutoff))
    pairs = np.asarray(pair_set, dtype=int)
    if pairs.size == 0:
        raise RuntimeError("No first-shell vortex pairs")
    distances = np.linalg.norm(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
    vectors = centers[pairs[:, 1]] - centers[pairs[:, 0]]
    bond_angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    global_psi6 = float(abs(np.mean(np.exp(6j * bond_angles))))

    wall_distance = RADIUS - np.linalg.norm(centers - RADIUS, axis=1)
    coordination = []
    for index, point in enumerate(centers):
        neighbors = [item for item in tree.query_ball_point(point, cutoff) if item != index]
        if wall_distance[index] >= cutoff and len(neighbors) >= 3:
            coordination.append(len(neighbors))
    if not coordination:
        # This fallback is recorded implicitly by the finite disk size; it is
        # used only if the strict complete-shell core is empty.
        coordination = [
            len(tree.query_ball_point(point, cutoff)) - 1 for point in centers
        ]
    structure = radial_structure_factor(centers)
    search = (K_GRID >= 4.0) & (K_GRID <= 7.0)
    local = np.flatnonzero(search)
    peak_candidates, _ = find_peaks(structure[search], prominence=0.02)
    if peak_candidates.size:
        candidate_indices = local[peak_candidates]
        peak_index = int(candidate_indices[np.argmax(structure[candidate_indices])])
    else:
        peak_index = int(local[np.argmax(structure[search])])
    return FrameAnalysis(
        iteration=iteration,
        positions=positions,
        phases=phases,
        rotation_centers=rotation_centers,
        vortex_centers=centers,
        vortex_sizes=sizes,
        first_shell_pairs=pairs,
        first_shell_distances=distances,
        median_nearest_neighbor=median_nearest,
        cutoff=cutoff,
        global_psi6=global_psi6,
        core_coordination_mean=float(np.mean(coordination)),
        radial_structure=structure,
        bragg_peak_k=float(K_GRID[peak_index]),
    )


def analyse_seed(seed: int) -> tuple[list[FrameAnalysis], dict[str, object], Path, int]:
    positions, phases, iterations, total_frames, path = load_terminal(seed)
    model = build_model(seed, 10 if seed == 9 else 50)
    frames = []
    for x, theta, iteration in zip(positions, phases, iterations):
        rates = phase_rates(x, theta, model)
        rotation, valid = instantaneous_rotation_centers(x, theta, rates)
        centers, sizes = cluster_vortex_centers(rotation, valid)
        frames.append(
            analyse_centers(
                int(iteration), x, theta, rotation, centers, sizes
            )
        )
    frame_bond_means = np.array(
        [np.mean(frame.first_shell_distances) for frame in frames]
    )
    mean_structure = np.mean(
        np.stack([frame.radial_structure for frame in frames]), axis=0
    )
    search = (K_GRID >= 4.0) & (K_GRID <= 7.0)
    mean_peak = float(K_GRID[np.flatnonzero(search)[np.argmax(mean_structure[search])]])
    measured_a_from_peak = 4.0 * math.pi / (math.sqrt(3.0) * mean_peak)
    summary = {
        "seed": seed,
        "trajectory": str(path),
        "total_saved_frames": total_frames,
        "sampled_terminal_frames": len(frames),
        "iteration_start": int(iterations[0]),
        "iteration_end": int(iterations[-1]),
        "mean_bulk_vortex_count": float(
            np.mean([frame.vortex_centers.shape[0] for frame in frames])
        ),
        "bulk_vortex_count_min": int(
            min(frame.vortex_centers.shape[0] for frame in frames)
        ),
        "bulk_vortex_count_max": int(
            max(frame.vortex_centers.shape[0] for frame in frames)
        ),
        "first_shell_bond_mean": float(np.mean(frame_bond_means)),
        "first_shell_bond_time_std": float(np.std(frame_bond_means, ddof=1)),
        "first_shell_bond_median": float(
            np.median(
                np.concatenate([frame.first_shell_distances for frame in frames])
            )
        ),
        "mean_nearest_neighbor_minimum": float(
            np.mean([frame.median_nearest_neighbor for frame in frames])
        ),
        "mean_global_psi6": float(np.mean([frame.global_psi6 for frame in frames])),
        "mean_core_coordination": float(
            np.mean([frame.core_coordination_mean for frame in frames])
        ),
        "mean_structure_factor_peak_k": mean_peak,
        "hex_spacing_from_structure_peak": measured_a_from_peak,
        "plane_spacing_from_structure_peak": 2.0 * math.pi / mean_peak,
    }
    return frames, summary, path, total_frames


def sensitivity(
    seed: int,
    frames: list[FrameAnalysis],
) -> list[dict[str, object]]:
    rows = []
    selected_frames = frames[::25]
    if selected_frames[-1] is not frames[-1]:
        selected_frames.append(frames[-1])
    for eps in (0.10, 0.12, 0.14, DBSCAN_EPS / D0):
        for exclusion in (0.40, 0.50, 0.70):
            bond_means = []
            counts = []
            for frame in selected_frames:
                valid = np.isfinite(frame.rotation_centers).all(axis=1)
                centers, sizes = cluster_vortex_centers(
                    frame.rotation_centers,
                    valid,
                    eps=eps * D0,
                    wall_exclusion=exclusion * D0,
                )
                if centers.shape[0] < 7:
                    continue
                tree = cKDTree(centers)
                nearest = tree.query(centers, k=2)[0][:, 1]
                pairs = np.asarray(
                    sorted(tree.query_pairs(FIRST_SHELL_FACTOR * np.median(nearest))),
                    dtype=int,
                )
                if pairs.size == 0:
                    continue
                bonds = np.linalg.norm(
                    centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1
                )
                bond_means.append(float(np.mean(bonds)))
                counts.append(int(centers.shape[0]))
            rows.append(
                {
                    "seed": seed,
                    "dbscan_eps_over_d0": eps,
                    "bulk_wall_exclusion_over_d0": exclusion,
                    "analysed_frames": len(bond_means),
                    "mean_vortex_count": float(np.mean(counts)),
                    "mean_first_shell_bond": float(np.mean(bond_means)),
                }
            )
    return rows


def spectrum_data() -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    module = import_dispersion_module()
    kstar, growth = most_unstable(module, ALPHA)
    params = (
        V,
        0.0,
        K / ((N / (math.pi * RADIUS**2)) * math.pi * D0**2),
        ALPHA,
        N / (math.pi * RADIUS**2),
        D0,
    )
    ks = np.linspace(0.0, 9.0, 1801)
    eigs = module.eigs_at_k(ks, np.zeros_like(ks), params)
    growth_curve = np.max(np.real(eigs), axis=-1)
    plane_spacing = 2.0 * math.pi / kstar
    hex_spacing = 4.0 * math.pi / (math.sqrt(3.0) * kstar)
    return (
        {
            "alpha_over_pi": ALPHA / math.pi,
            "elimination_denominator_D0": -2.0 * K * math.sin(ALPHA),
            "k_star": kstar,
            "max_Re_sigma": growth,
            "plane_spacing_2pi_over_k_star": plane_spacing,
            "hex_nearest_neighbor_prediction": hex_spacing,
        },
        ks,
        growth_curve,
    )


def plot_results(
    all_frames: dict[int, list[FrameAnalysis]],
    summaries: pd.DataFrame,
    spectral: dict[str, float],
    ks: np.ndarray,
    growth: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(15, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    for column, seed in enumerate(SEEDS):
        frame = all_frames[seed][-1]
        ax = fig.add_subplot(grid[0, column])
        ax.scatter(
            frame.positions[:, 0], frame.positions[:, 1], c=frame.phases,
            cmap="hsv", vmin=0, vmax=2 * math.pi, s=3, alpha=0.25,
        )
        for pair in frame.first_shell_pairs:
            edge = frame.vortex_centers[pair]
            ax.plot(edge[:, 0], edge[:, 1], color="#505050", lw=0.8, alpha=0.65)
        ax.scatter(
            frame.vortex_centers[:, 0], frame.vortex_centers[:, 1],
            s=38, facecolor="white", edgecolor="black", linewidth=0.9,
        )
        ax.add_patch(
            plt.Circle((RADIUS, RADIUS), RADIUS, fill=False, color="black", lw=0.8)
        )
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, DIAMETER + 0.15)
        ax.set_ylim(-0.15, DIAMETER + 0.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"seed {seed}: {frame.vortex_centers.shape[0]} bulk vortices, "
            f"$\\Psi_6$={frame.global_psi6:.2f}"
        )

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(ks, growth, color="#276678", lw=1.8)
    ax.axvline(spectral["k_star"], color="#D95F02", ls="--", label=rf"$k_*={spectral['k_star']:.3f}$")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(
        xlabel=r"$k$",
        ylabel=r"$\max_j\operatorname{Re}\sigma_j$",
        title=r"Bulk dispersion at $\alpha=0.6\pi$",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    ax = fig.add_subplot(grid[1, 1])
    for seed in SEEDS:
        structure = np.mean(
            np.stack([frame.radial_structure for frame in all_frames[seed]]), axis=0
        )
        ax.plot(K_GRID, structure, label=f"seed {seed}", lw=1.5)
    ax.axvline(spectral["k_star"], color="black", ls="--", label=r"linear $k_*$")
    ax.set(
        xlabel=r"radial wave number $k$",
        ylabel=r"$S_{\rm radial}(k)$ of vortex centers",
        title="Measured reciprocal-lattice peak",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = fig.add_subplot(grid[1, 2])
    measured = summaries.set_index("seed")["first_shell_bond_mean"].reindex(SEEDS)
    errors = summaries.set_index("seed")["first_shell_bond_time_std"].reindex(SEEDS)
    x = np.arange(len(SEEDS))
    ax.errorbar(x, measured, yerr=errors, fmt="o", capsize=3, color="#276678", label="measured NN bond")
    ax.axhline(
        spectral["plane_spacing_2pi_over_k_star"], color="#D95F02", ls=":",
        label=r"raw $2\pi/k_*$",
    )
    ax.axhline(
        spectral["hex_nearest_neighbor_prediction"], color="black", ls="--",
        label=r"hex $4\pi/(\sqrt{3} k_*)$",
    )
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set(
        xlabel="random seed",
        ylabel="length",
        title="Nearest-neighbor lattice constant",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "alpha06_bulk_hex_lattice_test.png", dpi=280)
    plt.close(fig)


def write_report(
    spectral: dict[str, float],
    summaries: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
) -> None:
    measured = float(summaries["first_shell_bond_mean"].mean())
    measured_seed_min = float(summaries["first_shell_bond_mean"].min())
    measured_seed_max = float(summaries["first_shell_bond_mean"].max())
    bragg = float(summaries["mean_structure_factor_peak_k"].mean())
    raw = spectral["plane_spacing_2pi_over_k_star"]
    predicted_hex = spectral["hex_nearest_neighbor_prediction"]
    measured_plane = math.sqrt(3.0) * measured / 2.0
    direct_error = measured / raw - 1.0
    hex_error = measured / predicted_hex - 1.0
    plane_error = measured_plane / raw - 1.0
    bragg_error = bragg / spectral["k_star"] - 1.0
    sensitivity_min = float(sensitivity_table["mean_first_shell_bond"].min())
    sensitivity_max = float(sensitivity_table["mean_first_shell_bond"].max())
    report = f"""# alpha=0.6 pi 体内六角 Lattice 与谱尺度检验

## 结论

`Dispersion.py` 在 alpha=0.6 pi 处是有限的：消元分母为
`{spectral['elimination_denominator_D0']:.8f}`。最大实部给出

- `k* = {spectral['k_star']:.9f}`；
- `max Re sigma = {spectral['max_Re_sigma']:.9f}`；
- `2pi/k* = {raw:.9f}`。

但六角/三角点阵中，`2pi/k*` 是第一族晶格面的间距，不是团簇中心最近邻
晶格常数。若第一圈 Bragg 峰满足 `|G_1|=k*`，则

`a_hex = 4pi/(sqrt(3) k*) = {predicted_hex:.9f}`。

三个 50000 步初值的旋转中心第一壳层距离为 `{measured:.9f}`，seed 均值范围
`[{measured_seed_min:.9f}, {measured_seed_max:.9f}]`。于是：

- 把 `2pi/k*` 直接当最近邻距离：偏差 `{direct_error:+.2%}`；
- 使用六角倒格矢关系：偏差 `{hex_error:+.2%}`；
- 等价地，实测晶格面距 `sqrt(3)a/2={measured_plane:.9f}` 与 `2pi/k*`
  的偏差为 `{plane_error:+.2%}`。

旋转中心径向结构因子的第一 Bragg 峰平均为 `k_B={bragg:.6f}`，相对线性
`k*` 偏差 `{bragg_error:+.2%}`。因此线性谱对体内六角晶格给出了较好的近似
尺度选择，但不是精确等式；残差包含非线性饱和、圆盘边界应变和有限晶格缺陷。

## 测量定义

每个粒子的瞬时旋转中心按

`c_i = x_i + (v/dot(theta_i)) (-sin(theta_i), cos(theta_i))`

计算；在旋转中心空间用 DBSCAN 识别涡旋，排除距圆壁小于
`{BULK_WALL_EXCLUSION/D0:.2f} d0` 的中心。晶格常数取第一壳层无向键距离，
而不是每个中心的最短单一距离。分析最后 {TERMINAL_TIME:g} 时间单位，每个
初值抽取约 {TARGET_SAMPLE_DT:g} 时间单位一帧。

## 稳健性

DBSCAN 半径 0.10--{DBSCAN_EPS/D0:.3f} d0、体区壁面排除 0.40--0.70 d0 时，测量范围为
`[{sensitivity_min:.6f}, {sensitivity_max:.6f}]`。逐 seed 时间统计、六角序参量、
配位数和结构因子峰见 CSV。
"""
    (OUT / "alpha06_bulk_hex_lattice_conclusion.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference_hashes = verify_references()
    nb.set_num_threads(4)
    spectral, ks, growth = spectrum_data()
    all_frames = {}
    summaries = []
    sensitivity_rows = []
    for seed in SEEDS:
        frames, summary, path, total = analyse_seed(seed)
        all_frames[seed] = frames
        summaries.append(summary)
        sensitivity_rows.extend(sensitivity(seed, frames))
    summary_table = pd.DataFrame(summaries)
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    summary_table.to_csv(OUT / "alpha06_bulk_hex_lattice_measurements.csv", index=False)
    sensitivity_table.to_csv(OUT / "alpha06_bulk_hex_lattice_sensitivity.csv", index=False)
    details = {
        "spectral_prediction": spectral,
        "reference_hashes": reference_hashes,
        "measurement_configuration": {
            "N": N,
            "steps": 50_000,
            "K": K,
            "d0": D0,
            "v": V,
            "diameter": DIAMETER,
            "alpha_over_pi": ALPHA / math.pi,
            "terminal_time": TERMINAL_TIME,
            "sample_dt": TARGET_SAMPLE_DT,
            "dbscan_eps": DBSCAN_EPS,
            "bulk_wall_exclusion": BULK_WALL_EXCLUSION,
            "first_shell_factor": FIRST_SHELL_FACTOR,
        },
    }
    (OUT / "alpha06_bulk_hex_lattice_diagnostics.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(all_frames, summary_table, spectral, ks, growth)
    write_report(spectral, summary_table, sensitivity_table)
    verify_references()
    print("Spectral prediction:")
    print(json.dumps(spectral, indent=2))
    print("\nMeasurements:")
    print(summary_table.to_string(index=False))
    print("\nSensitivity range:")
    print(
        sensitivity_table.groupby("seed")["mean_first_shell_bond"]
        .agg(["min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()
