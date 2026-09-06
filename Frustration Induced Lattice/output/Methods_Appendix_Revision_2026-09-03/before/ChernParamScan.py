from itertools import combinations
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from ChernNumberCompute import compute_topology

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
})

# =========================
# 手动可调参数区（直接改这里）
# =========================
USER_CONFIG = {
    "fixed_params": {
        "rho0": 0.0204,
    },
    "alpha_scan": {
        "alpha_min": 0.00001 * np.pi,
        "alpha_max": 1.0 * np.pi,
        "n_alpha": 25
        
        ,
    },
    # 严格按 param_scan 遍历，不使用 base_params
    "param_scan": {
        "v": (3.0, 7.0, 3),
        "omega": (0.0, 0.0, 1),
        "d0": (1.0, 3.0, 3),
        # 若 lam 的上下限给数值，则直接使用；
        # 若为 None，则按当前 d0 动态生成：lam_auto_scale * (20/(rho0*pi*d0^2))
        "lam": (None, None, 3),
        "lam_auto_scale": (0.5, 1.5),
    },
    "compute": {
        "Q": 60.0,
        "N_theta": 71,
        "N_phi": 91,
        "delta": 1e-3,
        "det_tol": 1e-7,
        "svd_tol": 1e-6,
        "overlap_weight": 0.4,
        "eig_weight": 0.6,
        "phase_branch_tol": 0.1,
        "phase_jump_tol": 3.0,
        "refine_bad": True,
        "refine_max_level": 3,
    },
    "holes": [(0.0, 0.0, 0.0)],
}


def _build_combos(bands):
    combos = []
    for r in range(1, len(bands) + 1):
        for combo in combinations(bands, r):
            combos.append(list(combo))
    return combos


def _format_value(value):
    return f"{value:.4g}"


def _format_math_value(value):
    text = f"{value:.4g}"
    if "e" not in text and "E" not in text:
        return text

    mantissa, exponent = text.lower().split("e")
    exponent = int(exponent)
    if mantissa == "1":
        return rf"10^{{{exponent}}}"
    if mantissa == "-1":
        return rf"-10^{{{exponent}}}"
    return rf"{mantissa}\times 10^{{{exponent}}}"


def _format_filename_value(value):
    text = f"{value:.4g}"
    return text.replace("-", "m").replace(".", "p")


def _resolve_scan_arrays(cfg):
    ps = cfg["param_scan"]

    v_values = np.linspace(ps["v"][0], ps["v"][1], ps["v"][2])
    omega_values = np.linspace(ps["omega"][0], ps["omega"][1], ps["omega"][2])
    d0_values = np.linspace(ps["d0"][0], ps["d0"][1], ps["d0"][2])
    lam_n = ps["lam"][2]
    return v_values, omega_values, d0_values, lam_n


def _lam_values_for_d0(rho0, d0, param_scan):
    lam_min, lam_max, lam_n = param_scan["lam"]
    if lam_min is None or lam_max is None:
        base_lam = 20.0 / (rho0 * np.pi * d0**2)
        smin, smax = param_scan["lam_auto_scale"]
        lam_min = smin * base_lam
        lam_max = smax * base_lam
    return np.linspace(lam_min, lam_max, lam_n)


def _make_compute_kwargs(cfg):
    c = cfg["compute"]
    return {
        "Q": c["Q"],
        "N_theta": c["N_theta"],
        "N_phi": c["N_phi"],
        "delta": c["delta"],
        "holes": cfg["holes"],
        "det_tol": c["det_tol"],
        "svd_tol": c["svd_tol"],
        "overlap_weight": c["overlap_weight"],
        "eig_weight": c["eig_weight"],
        "phase_branch_tol": c["phase_branch_tol"],
        "phase_jump_tol": c["phase_jump_tol"],
        "refine_bad": c["refine_bad"],
        "refine_max_level": c["refine_max_level"],
    }


def _plot_for_group(output_path, title, alphas, d0_values, lam_values_by_d0, combos, group_results):
    x = alphas / np.pi
    n_rows = len(d0_values)
    n_cols = len(lam_values_by_d0[0]) if n_rows > 0 else 0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    colors = plt.get_cmap("tab10")
    for i_d0, d0 in enumerate(d0_values):
        for i_lam, lam in enumerate(lam_values_by_d0[i_d0]):
            ax = axes[i_d0, i_lam]
            for idx_combo, combo in enumerate(combos):
                key = (i_d0, i_lam, tuple(combo))
                ax.plot(
                    x,
                    group_results[key],
                    lw=1.15,
                    color=colors(idx_combo % 10),
                    label="bands=" + ",".join(str(k) for k in combo),
                )
            ax.grid(True, alpha=0.25)
            ax.set_title(
                rf"$d_0={_format_math_value(d0)},\ \lambda={_format_math_value(lam)}$"
            )
            if i_lam == 0:
                ax.set_ylabel(
                    rf"$d_0={_format_math_value(d0)}$" + "\n" + r"$\mathrm{Chern}$"
                )
            if i_d0 == n_rows - 1:
                ax.set_xlabel(r"$\alpha / \pi$")
            if i_d0 == 0 and i_lam == 0:
                ax.legend(fontsize=7, ncol=1, loc="best")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    cfg = USER_CONFIG
    output_dir = Path(__file__).resolve().parent / "scan_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    v_values, omega_values, d0_values, lam_n = _resolve_scan_arrays(cfg)
    alphas = np.linspace(cfg["alpha_scan"]["alpha_min"], cfg["alpha_scan"]["alpha_max"], cfg["alpha_scan"]["n_alpha"])
    rho0 = cfg["fixed_params"]["rho0"]
    compute_kwargs = _make_compute_kwargs(cfg)

    bands = [0, 1, 2]
    combos = _build_combos(bands)
    single_combos = [c for c in combos if len(c) == 1]
    mixed_combos = [c for c in combos if len(c) >= 2]

    total_alpha_steps = len(v_values) * len(omega_values) * len(d0_values) * lam_n * len(alphas)
    global_done = 0
    start_time = time.time()
    print("[info] 按 param_scan 做四维全遍历：v × omega × d0 × lam")
    print(
        f"[info] 网格规模 v={len(v_values)}, omega={len(omega_values)}, "
        f"d0={len(d0_values)}, lam={lam_n}, alpha={len(alphas)}"
    )

    for i_v, v in enumerate(v_values):
        for i_omega, omega in enumerate(omega_values):
            group_results = {}
            group_total = len(d0_values) * lam_n * len(alphas)
            group_done = 0
            lam_values_by_d0 = []
            print(
                f"[group] v={v:.6g} ({i_v + 1}/{len(v_values)}), "
                f"omega={omega:.6g} ({i_omega + 1}/{len(omega_values)})"
            )
            for i_d0, d0 in enumerate(d0_values):
                lam_values = _lam_values_for_d0(rho0, d0, cfg["param_scan"])
                lam_values_by_d0.append(lam_values)
                for i_lam, lam in enumerate(lam_values):
                    for combo in combos:
                        group_results[(i_d0, i_lam, tuple(combo))] = []
                    for alpha in alphas:
                        params = (float(v), float(omega), float(lam), float(alpha), float(rho0), float(d0))
                        for combo in combos:
                            c_val, _, _ = compute_topology(
                                params,
                                target_bands=combo,
                                **compute_kwargs,
                            )
                            group_results[(i_d0, i_lam, tuple(combo))].append(float(c_val))
                        group_done += 1
                        global_done += 1
                        group_ratio = group_done / max(group_total, 1)
                        total_ratio = global_done / max(total_alpha_steps, 1)
                        print(
                            f"\r  组内进度 {group_done}/{group_total} ({group_ratio * 100:5.1f}%)"
                            f" | 总进度 {global_done}/{total_alpha_steps} ({total_ratio * 100:5.1f}%)",
                            end="",
                            flush=True,
                        )
            print()

            tag_v = _format_filename_value(v)
            tag_omega = _format_filename_value(omega)
            single_path = output_dir / f"single_v_{tag_v}_omega_{tag_omega}.png"
            mixed_path = output_dir / f"mixed_v_{tag_v}_omega_{tag_omega}.png"
            _plot_for_group(
                single_path,
                "Single-band Chern | "
                + rf"$v={_format_math_value(v)},\ \omega={_format_math_value(omega)}$",
                alphas,
                d0_values,
                lam_values_by_d0,
                single_combos,
                group_results,
            )
            _plot_for_group(
                mixed_path,
                "Mixed-band Chern | "
                + rf"$v={_format_math_value(v)},\ \omega={_format_math_value(omega)}$",
                alphas,
                d0_values,
                lam_values_by_d0,
                mixed_combos,
                group_results,
            )
            print(f"[done] {single_path}")
            print(f"[done] {mixed_path}")

    print(f"[done] 全部完成，总耗时 {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
