"""Reproducible staged strip sweep with parameter, ambiguity and cutoff records.

No production source or manuscript is changed.  Run from any directory:
    python run_strip_matched_sweep.py
All outputs are placed under the sibling figures directory.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import hashlib
import json
from pathlib import Path
import time
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import brentq

import SpectralFlow as sf


OUT = Path(__file__).resolve().parent / "strip_outputs"
LEVELS = (10.0, -10.0)
GRID = {"ky_max": 50.0, "n_cells": 36, "kx_cut": 40.0, "n_kx": 384, "hop_cut": 35, "edge_width": 6}
THRESHOLD = 0.45


def serialize_crossing(item):
    mode, ky, value, sign, wl, wr = item
    return {"mode": int(mode), "ky": ky, "sigma_real": value.real, "sigma_imag": value.imag,
            "orientation": sign, "mean_left_weight": wl, "mean_right_weight": wr}


def crossing_report(data, level, threshold=THRESHOLD):
    left, right, cross_l, cross_r, diagnostic = sf.count_horizontal_crossings(
        data, level, edge_threshold=threshold, return_diagnostics=True,
    )
    return {"c": level, "count_left": left, "count_right": right,
            "crossings_left": [serialize_crossing(x) for x in cross_l],
            "crossings_right": [serialize_crossing(x) for x in cross_r], **diagnostic}


def radial_bulk_reference_diagnostic(params):
    """Find sampled/bracketed bulk intersections; do not certify an empty line.

    Rotational covariance permits radial sampling.  For each k, imaginary
    eigenvalue ordering is used only as a continuous real-valued level-search
    function, never as an assertion of isolated-band topology.
    """
    radial = np.unique(np.r_[np.linspace(0, 5, 10001), np.linspace(5, 80, 10001), np.linspace(80, 400, 4001)])
    imaginary = np.array([np.sort(np.linalg.eigvals(sf.M_matrix_standalone(k, 0.0, params)).imag) for k in radial])
    results = {}
    for level in LEVELS:
        records = []
        for band in range(3):
            values = imaginary[:, band] - level
            brackets = np.flatnonzero(values[:-1] * values[1:] < 0)
            roots = [float(radial[j]) for j in np.flatnonzero(np.abs(values) < 1e-9)]
            for index in brackets:
                root = brentq(lambda k: np.sort(np.linalg.eigvals(sf.M_matrix_standalone(k, 0.0, params)).imag)[band] - level,
                              radial[index], radial[index+1], xtol=1e-12)
                if not any(abs(root-old) < 1e-8 for old in roots):
                    roots.append(float(root))
            for root in sorted(roots):
                vals = np.linalg.eigvals(sf.M_matrix_standalone(root, 0.0, params))
                value = vals[np.argmin(np.abs(vals.imag-level))]
                records.append({"k": root, "example_kx": 0.0, "example_ky": root,
                                "sigma_real": float(value.real), "sigma_imag": float(value.imag),
                                "level_residual_abs": float(abs(value.imag-level))})
        results[str(level)] = {
            "reference_level": level, "intersections_detected": bool(records),
            "roots": records, "sampled_min_imaginary_distance": float(np.min(np.abs(imaginary-level))),
            "radial_max": 400.0, "sample_count": len(radial),
            "status": "bulk intersection detected" if records else "no intersection detected in the sampled radial interval; not a global certificate",
        }
        v, omega, lam, alpha, rho0, d0 = params
        coupling = lam*rho0*np.pi*d0**2
        threshold_case = omega == 0 and coupling > 0 and v > 0 and abs(alpha-np.pi/2) < 1e-14
        certified = threshold_case and 0 < abs(level) < coupling/2
        results[str(level)]["bulk_line_gap_certified"] = bool(certified)
        if certified:
            if records:
                raise RuntimeError("Numerical intersection contradicts the threshold gap bound")
            results[str(level)].update(
                status="bulk line gap certified at alpha=pi/2",
                minimum_oscillatory_frequency=coupling/2,
                certificate="For omega=0, alpha=pi/2, K>0: a=0, w^2=b^2+v^2*k^2/2. "
                "Ghat/G0<=1 gives b>=K/2-v^2*k^2/(8K). For k<=2K/v this implies w>=K/2; "
                "for k>=2K/v, w>=v*k/sqrt(2)>=sqrt(2)*K. Equality w=K/2 occurs at k=0.",
            )
    return results


def location_comparison(coarse, fine):
    result = {"signed_counts_equal": (coarse["count_left"], coarse["count_right"]) == (fine["count_left"], fine["count_right"])}
    for side in ("left", "right"):
        earlier = sorted(coarse["crossings_"+side], key=lambda x: x["ky"])
        later = sorted(fine["crossings_"+side], key=lambda x: x["ky"])
        paired = len(earlier) == len(later) and all(a["orientation"] == b["orientation"] for a, b in zip(earlier, later))
        result[side+"_crossing_lists_compatible"] = paired
        result[side+"_max_abs_ky_shift"] = max((abs(a["ky"]-b["ky"]) for a, b in zip(earlier, later)), default=0.0) if paired else None
    return result


def draw_figure(datasets, records):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.9), sharex=True, sharey=True)
    colors = ("#126E9B", "#CC5B31")
    for index, (fraction, data, record) in enumerate(zip(sf.DEFAULT_ALPHA_OVER_PI, datasets, records)):
        ax = axes.flat[index]
        vals = data.eigvals
        xs = np.broadcast_to(data.ky[:, None], vals.shape)
        left = (data.left_weight >= THRESHOLD) & (data.left_weight > data.right_weight)
        right = (data.right_weight >= THRESHOLD) & (data.right_weight > data.left_weight)
        resolved = ~data.ambiguous
        ordinary = ~(left | right) | ~resolved
        ax.scatter(xs[ordinary], vals.imag[ordinary], s=1.1, color="#B4BBC2", alpha=0.45, linewidths=0)
        ax.scatter(xs[left & resolved], vals.imag[left & resolved], s=4.1, color=colors[0], alpha=0.90, linewidths=0)
        ax.scatter(xs[right & resolved], vals.imag[right & resolved], s=4.1, color=colors[1], alpha=0.90, linewidths=0)
        for level in LEVELS:
            ax.axhline(level, color="#32363D", lw=0.75, ls=(0, (4, 3)), zorder=1)
        ax.set_title(rf"({chr(97+index)})  $\alpha={fraction:g}\pi$", loc="left", pad=7)
        text_lines = []
        for level in LEVELS:
            crossing = record["fine"]["reference_counts"][str(level)]
            text_lines.append(rf"$c={level:+g}:\ (n_L,n_R)=({crossing['count_left']},{crossing['count_right']})$")
        ax.text(0.03, 0.965, "\n".join(text_lines), transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.87, "pad": 2.5})
        ax.set_xlim(-50, 50)
        ax.set_ylim(-40, 40)
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.grid(True, color="#DCE0E4", linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#7A828A")
            spine.set_linewidth(0.65)
        if index >= 3:
            ax.set_xlabel(r"$k_y$")
        if index % 3 == 0:
            ax.set_ylabel(r"$\operatorname{Im}\sigma$")
    legend_ax = axes.flat[5]
    legend_ax.set_axis_off()
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=colors[0], label="Left-localized mode"),
               Line2D([], [], marker="o", ls="", ms=4, color=colors[1], label="Right-localized mode"),
               Line2D([], [], marker="o", ls="", ms=3, color="#B4BBC2", label="Other / unresolved mode"),
               Line2D([], [], color="#32363D", lw=0.9, ls="--", label=r"Empirical references: $c=\pm10$")]
    legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10, handlelength=2.0, borderaxespad=0)
    legend_ax.text(0.015, 0.52,
                   "Matched circular-particle parameters\n"
                   r"$N=2000,\ L=7,\ R=L/2=3.5$"+"\n"+
                   r"$v=3,\ K=20.75,\ \omega=0,\ d_0=1$"+"\n\n"+
                   "Finite-strip grid\n"+
                   r"$k_c=40,\ N_k=384,\ N_x=36,\ R_{\max}=35$"+"\n"+
                   r"$|k_y|\leq50,\ \Delta k_y=0.5,\ w=6,\ \eta=0.45$"+"\n\n"+
                   "Reference lines are bulk gaps at the threshold;\n"
                   "at the other four phase lags they cross bulk spectra.\n"
                   "Counts remain finite-cutoff diagnostics.",
                   transform=legend_ax.transAxes, va="top", fontsize=10, linespacing=1.38)
    fig.suptitle("Finite-strip spectra at matched circular-particle parameters", x=0.075, ha="left", y=0.99, fontsize=15)
    fig.text(0.075, 0.94, "Five nonsingular phase lags; the empirical reference levels are unchanged across the sweep.", fontsize=10, color="#49525C")
    fig.subplots_adjust(left=0.075, right=0.987, top=0.875, bottom=0.125, wspace=0.22, hspace=0.29)
    fig.text(0.075, 0.048,
             "Colored samples satisfy instantaneous edge localization. Counts additionally require the same edge label on both sides\n"
             "and resolved individual-branch tracking. Bulk-intersection and momentum-step diagnostics are reported separately.",
             fontsize=9, color="#49525C", linespacing=1.45)
    fig.savefig(OUT / "strip_circle_matched_alpha_sweep.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "strip_circle_matched_alpha_sweep.pdf", facecolor="white", metadata={"Title": "Matched circular-particle finite-strip alpha sweep", "Author": "Methods Appendix reproducible calculation"})
    plt.close(fig)


def draw_single_threshold(data, record):
    """New matched-parameter threshold panel; never overwrite the old figure."""
    path = OUT / "strip_circle_matched_alpha_050.png"
    fig, ax = sf.plot_verified_style(data, filename=str(path))
    fig.set_size_inches(7.6, 5.0)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-40, 40)
    ax.set_title("Matched circular-particle strip: " + r"$\alpha=\pi/2$" + "\n" +
                 r"$N=2000,\ L=7,\ v=3,\ \omega=0,\ K=20.75,\ d_0=1$", fontsize=12, pad=10)
    ax.set_xlabel(r"$k_y$", fontsize=12)
    ax.set_ylabel(r"$\operatorname{Im}\sigma$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.92)
    lines = []
    for level in LEVELS:
        crossing = record["fine"]["reference_counts"][str(level)]
        lines.append(rf"$c={level:+g}:\ (n_L,n_R)=({crossing['count_left']},{crossing['count_right']})$")
    ax.text(0.025, 0.025, "\n".join(lines), transform=ax.transAxes, va="bottom", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(OUT / "strip_circle_matched_alpha_050.pdf", facecolor="white")
    plt.close(fig)


def main(output_dir=None):
    global OUT
    if output_dir is not None:
        OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = OUT / "strip_data"
    data_dir.mkdir(exist_ok=True)
    records, fine_datasets, csv_rows = [], [], []
    started = time.perf_counter()
    for fraction in sf.DEFAULT_ALPHA_OVER_PI:
        params = sf.matched_particle_params(fraction*np.pi)
        record = {"alpha_over_pi": fraction, "params_v_omega_lambda_alpha_rho0_d0": list(params),
                  "lambda_rho0_G0": float(params[2]*params[4]*np.pi*params[5]**2),
                  "D0": float(2*params[1]-2*params[2]*params[4]*np.pi*params[5]**2*np.sin(params[3]))}
        record["bulk_reference_diagnostics"] = radial_bulk_reference_diagnostic(params)
        for name, count in (("coarse", 101), ("fine", 201)):
            tick = time.perf_counter()
            data = sf.compute_strip_data(params, n_ky=count, **GRID)
            level_reports = {str(level): crossing_report(data, level) for level in LEVELS}
            record[name] = {"diagnostics": data.diagnostics, "reference_counts": level_reports, "elapsed_seconds": time.perf_counter()-tick}
            np.savez_compressed(data_dir / f"alpha_{fraction:g}_{name}.npz", ky=data.ky, eigvals=data.eigvals,
                                left_weight=data.left_weight, right_weight=data.right_weight,
                                ambiguous=data.ambiguous, assignment_ambiguous=data.assignment_ambiguous,
                                params=np.asarray(params))
            for level in LEVELS:
                rep = level_reports[str(level)]
                csv_rows.append({"alpha_over_pi": fraction, "grid": name, "n_ky": count, "delta_ky": 100/(count-1),
                                 "c": level, "count_left": rep["count_left"], "count_right": rep["count_right"],
                                 "accepted_left_crossings": len(rep["crossings_left"]), "accepted_right_crossings": len(rep["crossings_right"]),
                                 "excluded_crossings": len(rep["excluded_crossings"]),
                                 "excluded_with_persistent_edge": sum(x["persistent_edge_label"] for x in rep["excluded_crossings"]),
                                 "bulk_intersections_detected": record["bulk_reference_diagnostics"][str(level)]["intersections_detected"],
                                 "bulk_line_gap_certified": record["bulk_reference_diagnostics"][str(level)]["bulk_line_gap_certified"],
                                 "v": params[0], "omega": params[1], "lambda": params[2], "rho0": params[4], "d0": params[5],
                                 "N": 2000, "L": 7, "R": 3.5, "K": 20.75, **GRID, "eta": THRESHOLD})
            print(f"alpha/pi={fraction:g}, {name}: " + "; ".join(f"c={level:+g}: ({level_reports[str(level)]['count_left']},{level_reports[str(level)]['count_right']})" for level in LEVELS)
                  + f" [{record[name]['elapsed_seconds']:.2f}s]", flush=True)
            if name == "fine":
                fine_datasets.append(data)
                record["edge_threshold_sensitivity"] = {str(eta): {str(level): {k: value for k, value in crossing_report(data, level, eta).items() if k in ("count_left", "count_right")}
                                                                        for level in LEVELS} for eta in (0.40, 0.45, 0.50)}
        record["ky_refinement_comparison"] = {str(level): location_comparison(record["coarse"]["reference_counts"][str(level)], record["fine"]["reference_counts"][str(level)]) for level in LEVELS}
        records.append(record)
    report = {"particle_parameters": sf.PARTICLE_PARAMETERS, "normalization": "rho0=N/[pi*(L/2)^2]; lambda=K/(rho0*pi*d0^2)",
              "geometry_source": "main.py:1409-1411: circleRadius=halfBoundaryLength=boundaryLength/2",
              "reference_level_selection": "Fixed empirical levels +/-10 retained unchanged; no selection to obtain any target integer",
              "excluded_phase_lag_endpoints": [0.0, 1.0], "records": records,
              "validation_scope": "ky step 1 to 0.5 and eta 0.40/0.45/0.50 only; not joint UV/width/cutoff convergence or subspace continuation",
              "elapsed_seconds": time.perf_counter()-started,
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ("SpectralFlow.py", "run_strip_matched_sweep.py")}}
    (OUT / "strip_circle_matched_diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (OUT / "strip_circle_matched_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    draw_figure(fine_datasets, records)
    threshold_index = sf.DEFAULT_ALPHA_OVER_PI.index(0.5)
    draw_single_threshold(fine_datasets[threshold_index], records[threshold_index])
    print(f"Completed: {time.perf_counter()-started:.2f}s; outputs under {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    main(args.output_dir)
