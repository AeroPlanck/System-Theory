"""Publication-size redraw of saved finite-strip data; no spectral solve.

All five curves, localization data, ambiguity masks and empirical reference
levels are reused without filtering or resampling. The saved crossing counts
are independently recomputed from the NPZ arrays using the revision's fixed
counting routine. This script writes only beside itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
REVISION = WORKSPACE / "output" / "Methods_Appendix_Revision_2026-09-03"
SOURCE = REVISION / "figures"
sys.path.insert(0, str(REVISION / "code"))
import SpectralFlow as sf

FRACTIONS = (0.2, 0.4, 0.5, 0.6, 0.8)
LEVELS = (10.0, -10.0)
THRESHOLD = 0.45
STEM = "strip_circle_matched_alpha_sweep_publication"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_report = SOURCE / "strip_circle_matched_diagnostics.json"
    report = json.loads(source_report.read_text(encoding="utf-8"))
    records = {float(row["alpha_over_pi"]): row for row in report["records"]}
    datasets, checks = [], []
    for fraction in FRACTIONS:
        path = SOURCE / "strip_data" / f"alpha_{fraction:g}_fine.npz"
        with np.load(path) as saved:
            data = sf.FlowData(ky=saved["ky"], eigvals=saved["eigvals"],
                               left_weight=saved["left_weight"], right_weight=saved["right_weight"],
                               ambiguous=saved["ambiguous"], assignment_ambiguous=saved["assignment_ambiguous"],
                               params=tuple(saved["params"]))
        record = records[fraction]
        assert data.eigvals.shape == (201, 108)
        assert np.allclose(np.diff(data.ky), 0.5)
        assert np.allclose(data.params, record["params_v_omega_lambda_alpha_rho0_d0"])
        crossing_checks = {}
        for level in LEVELS:
            left, right, cross_l, cross_r, diagnostic = sf.count_horizontal_crossings(
                data, level, THRESHOLD, return_diagnostics=True,
            )
            expected = record["fine"]["reference_counts"][str(level)]
            assert (left, right) == (expected["count_left"], expected["count_right"])
            assert len(diagnostic["excluded_crossings"]) == len(expected["excluded_crossings"])
            for current, original in ((cross_l, expected["crossings_left"]), (cross_r, expected["crossings_right"])):
                now = sorted(x[1] for x in current)
                before = sorted(x["ky"] for x in original)
                assert np.allclose(now, before, atol=1e-12, rtol=0)
            crossing_checks[str(level)] = {
                "count_left": left, "count_right": right,
                "accepted_crossings_left": len(cross_l), "accepted_crossings_right": len(cross_r),
                "excluded_crossings": len(diagnostic["excluded_crossings"]),
                "bulk_intersections_detected": record["bulk_reference_diagnostics"][str(level)]["intersections_detected"],
                "bulk_line_gap_certified": record["bulk_reference_diagnostics"][str(level)]["bulk_line_gap_certified"],
            }
        checks.append({"alpha_over_pi": fraction, "source_npz": str(path), "sha256": sha256(path),
                       "mode_samples": int(data.eigvals.size), "params": list(data.params),
                       "crossing_count_reverification": crossing_checks})
        datasets.append(data)

    # Dimensions are the final Methods text width, not a large canvas later
    # shrunk to unreadable size. Standard tick and legend text is >=7.5 pt.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8.0,
                         "axes.labelsize": 8.5, "axes.titlesize": 9.0,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 8.0, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": False})
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 7.0))
    palette = ("#126E9B", "#CC5B31")
    for index, (fraction, data) in enumerate(zip(FRACTIONS, datasets)):
        ax = axes.flat[index]
        xs = np.broadcast_to(data.ky[:, None], data.eigvals.shape)
        left = (data.left_weight >= THRESHOLD) & (data.left_weight > data.right_weight)
        right = (data.right_weight >= THRESHOLD) & (data.right_weight > data.left_weight)
        resolved = ~data.ambiguous
        other = ~(left | right) | ~resolved
        ax.scatter(xs[other], data.eigvals.imag[other], s=0.60, c="#ADB7C1", alpha=0.43,
                   linewidths=0, rasterized=False)
        ax.scatter(xs[left & resolved], data.eigvals.imag[left & resolved], s=2.0,
                   c=palette[0], alpha=0.92, linewidths=0, rasterized=False)
        ax.scatter(xs[right & resolved], data.eigvals.imag[right & resolved], s=2.0,
                   c=palette[1], alpha=0.92, linewidths=0, rasterized=False)
        for level in LEVELS:
            ax.axhline(level, c="#30363D", ls=(0, (4, 2.5)), lw=0.65, zorder=1)
        ax.set_title(rf"({chr(97+index)})  $\alpha={fraction:g}\pi$", loc="left", pad=5)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-40, 40)
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_xlabel(r"$k_y$", labelpad=2)
        ax.set_ylabel(r"$\operatorname{Im}\sigma$", labelpad=2)
        ax.tick_params(length=2.5, width=0.55, pad=2)
        ax.grid(True, color="#DCE0E4", lw=0.4, alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#7A828A")
            spine.set_linewidth(0.55)

    key = axes.flat[5]
    key.set_axis_off()
    handles = [Line2D([], [], marker="o", ls="", ms=3.5, color=palette[0], label="Left-localized mode"),
               Line2D([], [], marker="o", ls="", ms=3.5, color=palette[1], label="Right-localized mode"),
               Line2D([], [], marker="o", ls="", ms=3.0, color="#ADB7C1", label="Other / unresolved mode"),
               Line2D([], [], color="#30363D", lw=0.7, ls="--", label=r"Empirical levels $c=\pm10$")]
    key.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=0,
               handlelength=2.0, handletextpad=0.65, labelspacing=0.40)
    key.text(0.02, 0.54,
             "All five panels, at both levels:\n"+
             r"$(n_L,n_R)=(2,-2)$"+"\n\n"+
             r"Bulk line gaps only at $\alpha=\pi/2$."+"\n"+
             "Other panels: bulk intersections.\n\n"+
             "Counts require persistent edge labels\n"
             "and resolved tracking; they are\n"
             "finite-cutoff diagnostics.",
             transform=key.transAxes, va="top", fontsize=8.0, linespacing=1.15)
    fig.suptitle("Matched-parameter finite-strip spectra", x=0.085, y=0.986,
                 ha="left", fontsize=11)
    fig.text(0.085, 0.951, r"$v=3,\ \omega=0,\ K=20.75,\ d_0=1;\quad \Delta k_y=0.5$", fontsize=8.5)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.058, top=0.900,
                        hspace=0.34, wspace=0.28)
    png = OUT / f"{STEM}.png"
    pdf = OUT / f"{STEM}.pdf"
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Matched finite-strip phase-lag sweep", "Author": "Saved-data publication redraw"})
    fig.savefig(png, dpi=600, facecolor="white")
    plt.close(fig)
    verification = {
        "operation": "saved-data redraw; no eigenvalue computation or reference-level reselection",
        "source_diagnostics": str(source_report), "source_diagnostics_sha256": sha256(source_report),
        "source_counting_routine_sha256": sha256(REVISION / "code" / "SpectralFlow.py"),
        "figure_size_inches": [6.5, 7.0], "png_dpi": 600, "pdf_scatter_rasterized": False,
        "minimum_standard_tick_font_pt": 7.5, "legend_font_pt": 8.0,
        "plot_limits_unchanged": {"ky": [-50, 50], "imag_sigma": [-40, 40]},
        "source_checks": checks, "pdf_sha256": sha256(pdf), "png_sha256": sha256(png),
    }
    (OUT / "strip_sweep_redraw_verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"pdf": str(pdf), "png": str(png), "checked_saved_mode_samples": sum(row["mode_samples"] for row in checks)}, indent=2))


if __name__ == "__main__":
    main()
