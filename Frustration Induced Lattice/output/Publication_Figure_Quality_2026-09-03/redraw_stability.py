"""Print-sized redraw of saved stability statistics; no analysis is rerun."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
import phase_informed_boundary_flow_analysis as original

SOURCE = PROJECT/"output/Phase_Informed_Boundary_Flow_Refined/Phase_Informed_Boundary_Flow_Metrics.csv"
OUT = ROOT/"stability"
SERIES = []
LABELS = {"Circular": "Circle", "Circular_Single_Defect_H3": "Circle, one defect ($H=3$)",
          "Square": "Square", "Square_Four_Symmetric_Defects_H1": "Square, four defects ($H=1$)"}


def record(artist, rows, field):
    x = rows["alpha_over_pi"].to_numpy(dtype=float)
    y = rows[field].to_numpy(dtype=float)
    assert np.array_equal(artist.get_xdata(), x, equal_nan=True)
    assert np.array_equal(artist.get_ydata(), y, equal_nan=True)
    SERIES.append({"field": field, "condition": None if rows.empty else rows.condition.iloc[0],
                   "x": x.tolist(), "y": y.tolist()})


def draw_series(ax, rows, field, condition, critical=False):
    color, marker = original.COLORS[condition], original.MARKERS[condition]
    if not critical:
        line, = ax.plot(rows.alpha_over_pi, rows[field], color=color, lw=1.05, zorder=2)
        record(line, rows, field)
    for complete in (True, False):
        chosen = rows[(rows.analysis_window_fraction >= .995) == complete]
        scatter = ax.scatter(chosen.alpha_over_pi, chosen[field], marker=marker,
                             s=26 if critical else 20,
                             facecolors=color if complete else "white",
                             edgecolors="white" if complete and critical else color,
                             linewidths=.7, zorder=4 if critical else 3)
        assert np.allclose(scatter.get_offsets(), np.c_[chosen.alpha_over_pi, chosen[field]], equal_nan=True)


def decorate(ax, label, title, ylabel):
    ax.axvline(.5, color="#777777", lw=.75, ls="--", zorder=0)
    ax.set(xlim=(-.025, 1.025), xticks=np.linspace(0, 1, 6), ylabel=ylabel)
    ax.set_title(f"{label} {title}", loc="left", fontsize=9, pad=7)
    ax.grid(color="#d8d8d8", lw=.4, alpha=.7)
    ax.tick_params(length=3, width=.55, labelsize=8)


def legend(fig):
    handles = [Line2D([], [], color=original.COLORS[c], marker=original.MARKERS[c],
                      markerfacecolor=original.COLORS[c], markersize=4.3, lw=1., label=LABELS[c])
               for c in LABELS]
    handles += [Line2D([], [], color="#555555", marker="o", markerfacecolor=face,
                       ls="none", markersize=4.3, label=label)
                for face, label in (("#555555", "Full requested window"), ("white", "Shorter available window"))]
    fig.legend(handles=handles, ncol=2, frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(.51, .002), columnspacing=2,
               handlelength=1.8, labelspacing=.5)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/f"{name}.pdf", facecolor="white")
    fig.savefig(OUT/f"{name}.png", dpi=600, facecolor="white")
    plt.close(fig)


def headlines(table):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.982, bottom=.32, top=.84, wspace=.28)
    for ax, field, letter, title, ylabel in zip(
            axes, ("xi_persist", "xi_sign"), ("(a)", "(b)"),
            ("Boundary-flow recurrence", "Selected-signal chirality"),
            (r"Active-block fraction $\Xi_{\rm Persist}$", r"Mean chirality $\Xi_{\rm Sign}$")):
        for condition in LABELS:
            sub = table[(table.condition == condition) & (table.alpha_over_pi > 0) &
                        (table.alpha_over_pi < 1)].sort_values("alpha_over_pi")
            for mask in (sub.alpha_over_pi < .5, sub.alpha_over_pi > .5):
                draw_series(ax, sub[mask], field, condition)
            critical = sub[np.isclose(sub.alpha_over_pi, .5)]
            draw_series(ax, critical, field, condition, critical=True)
        decorate(ax, letter, title, ylabel)
        ax.set(ylim=(-.03, 1.03), yticks=np.linspace(0, 1, 6), xlabel=r"Phase lag $\alpha/\pi$")
    fig.text(.52, .958, r"Dashed line: pattern onset at $\alpha=\pi/2$; isolated markers use the critical-point rule",
             ha="center", va="top", fontsize=7.4)
    legend(fig)
    save(fig, "Boundary_Flow_Stability_Vector")


def diagnostics(table):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.8), sharex=True)
    fig.subplots_adjust(left=.09, right=.982, bottom=.215, top=.93, wspace=.29, hspace=.43)
    specifications = (
        ("mean_active_arclength_fraction", "(a)", "Active boundary length", r"Mean active fraction $\overline{a}$"),
        ("calibrated_perimeter_fraction", "(b)", "Displacement calibration", r"Calibrated fraction $f_{\rm cal}$"),
        ("w_selection_retention", "(c)", r"High-$\alpha$ membership gate", r"Retained-cell fraction $\eta_W$"),
        ("representative_wall_distance_q50_over_d0", "(d)", "Selected-particle wall distance", r"Distance $d/d_0$"))
    for ax, (field, letter, title, ylabel) in zip(axes.flat, specifications):
        for condition in LABELS:
            sub = table[(table.condition == condition) & (table.alpha_over_pi > 0) &
                        (table.alpha_over_pi < 1)].sort_values("alpha_over_pi")
            if field == "w_selection_retention":
                sub = sub[sub.alpha_over_pi > .5]
            draw_series(ax, sub, field, condition)
            if field == "representative_wall_distance_q50_over_d0":
                ax.fill_between(sub.alpha_over_pi.to_numpy(), sub[field].to_numpy(),
                                sub.representative_wall_distance_q90_over_d0.to_numpy(),
                                color=original.COLORS[condition], alpha=.08, lw=0)
        decorate(ax, letter, title, ylabel)
        if field != "representative_wall_distance_q50_over_d0":
            ax.set(ylim=(-.03, 1.03), yticks=np.linspace(0, 1, 6))
    axes[1, 0].set_xlabel(r"Phase lag $\alpha/\pi$")
    axes[1, 1].set_xlabel(r"Phase lag $\alpha/\pi$")
    axes[1, 1].text(.96, .94, r"Line: $d_{50}$; band: $d_{50}$-$d_{90}$", transform=axes[1, 1].transAxes,
                    ha="right", va="top", fontsize=7.4)
    legend(fig)
    save(fig, "Boundary_Flow_Diagnostics_Vector")


def main():
    table = pd.read_csv(SOURCE)
    assert len(table) == 44 and len(table[(table.alpha_over_pi>0)&(table.alpha_over_pi<1)]) == 36
    assert table.loc[table.alpha_over_pi.isin([0., 1.]), ["xi_persist", "xi_sign"]].isna().all().all()
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 8.5, "axes.labelsize": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .6}):
        headlines(table)
        diagnostics(table)
    report = {"source_csv": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "rows": len(table), "analysis_rerun": False, "all_plotted_values_equal_csv": True,
              "endpoints_unavailable": True, "critical_markers_disconnected_in_headlines": True,
              "retention_is_cell_occupancy_not_particle_fraction": True, "plotted_lines": SERIES}
    (OUT/"stability_redraw_provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("PASS: exact saved CSV values; endpoint NA preserved; vector PDF + 600 dpi PNG.")


if __name__ == "__main__":
    main()
