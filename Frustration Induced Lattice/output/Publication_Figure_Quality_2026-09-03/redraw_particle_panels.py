"""Redraw the exact published terminal frames as print-sized vector artwork.

No trajectory is simulated, continued, filtered, or overwritten. All 2000
particles are retained in each panel. Only rendering and typography change.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
import boundary_defect_analysis as source


def model_groups():
    phases = source.TERMINAL_COMPARISON_ALPHA_OVER_PI
    circle = [source.build_model(source.model_library.CircularBoundaryPatternFormation, a)
              for a in phases]
    circle += [source.build_model(source.model_library.CollisionBoundaryMidpointSpikePatternFormation,
                                 a, protrusionHeight=source.ASYMMETRIC_SPIKE_HEIGHT,
                                 protrusionHalfWidth=source.SPIKE_HALF_WIDTH) for a in phases]
    square = [source.build_model(source.model_library.CollisionBoundaryPatternFormation, a)
              for a in phases]
    square += [source.build_model(source.model_library.CollisionBoundaryFourSpikePatternFormation,
                                 a, protrusionHeight=1.0,
                                 protrusionHalfWidth=source.SPIKE_HALF_WIDTH) for a in phases]
    return [("Circular_Boundary_Terminal_States_Vector", circle, "Single defect: $H=3$"),
            ("Square_Boundary_Terminal_States_Vector", square, "Four defects: $H=1$")]


def redraw(name, models, defect_label):
    source.validate_exact_files(models)
    # Match the actual Appendix width (6.5 in), rather than shrinking a
    # 21.6-inch canvas and its 14-point labels by a factor of more than three.
    fig = plt.figure(figsize=(6.5, 2.40), facecolor="white")
    grid = fig.add_gridspec(2, 7, left=.047, right=.983, bottom=.16, top=.86,
                           hspace=.22, wspace=.07)
    records = []
    for index, model in enumerate(models):
        ax = fig.add_subplot(grid[index // 7, index % 7])
        positions, phases, frame = source.load_frame(model, iteration=None)
        assert positions.shape == (2000, 2) and phases.shape == (2000,)
        assert np.isfinite(positions).all() and np.isfinite(phases).all()
        artist = ax.quiver(positions[:, 0], positions[:, 1], np.cos(phases), np.sin(phases),
                          phases, cmap=source.phaseCmap, norm=source.phaseNorm,
                          scale_units="inches", scale=50., width=source.QUIVER_WIDTH,
                          pivot="middle", zorder=2, rasterized=False)
        # Quiver colour values and origins must remain exactly those read from
        # the saved frame; increasing resolution must not alter the physics.
        assert np.array_equal(artist.get_array(), phases)
        assert np.array_equal(artist.X, positions[:, 0])
        assert np.array_equal(artist.Y, positions[:, 1])
        source.draw_boundary(ax, model)
        pad = .018 * model.boundaryLength
        ax.set(xlim=(-pad, model.boundaryLength+pad), ylim=(-pad, model.boundaryLength+pad),
               xticks=[], yticks=[], aspect="equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        if index < 7:
            ax.set_title(rf"$\alpha={source._alpha_label(model.phaseLagA0 / np.pi)[1:-1]}$",
                         fontsize=8.3, pad=3)
        ax.text(.01, .99, f"({source._subplot_label(index)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.2, color="#202020", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.84, pad=.2))
        if index % 7 == 0:
            ax.text(-.15, .5, "No defect" if index == 0 else defect_label,
                    transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7.4)
        records.append({"panel": source._subplot_label(index),
                        "source_hdf5": str(source.data_path(model)),
                        "saved_frame": int(frame), "particle_count": int(len(phases)),
                        "alpha_over_pi": float(model.phaseLagA0 / np.pi),
                        "positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                        "phases_sha256": hashlib.sha256(phases.tobytes()).hexdigest(),
                        "all_particles_drawn": True})
    color_ax = fig.add_axes([.36, .078, .30, .025])
    bar = fig.colorbar(ScalarMappable(norm=source.phaseNorm, cmap=source.phaseCmap),
                       cax=color_ax, orientation="horizontal", ticks=[0, np.pi, 2*np.pi])
    # Vectorize the colorbar too, so the PDF contains no image-based labels or
    # gradients. A 256-cell vector colour ramp preserves the original cmap.
    bar.solids.set_rasterized(False)
    bar.ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    bar.ax.tick_params(labelsize=7.5, length=2, pad=1)
    fig.text(.32, .0905, r"$\theta$", fontsize=9, ha="center", va="center")
    out = ROOT / "particles"
    out.mkdir(exist_ok=True)
    fig.savefig(out/f"{name}.pdf", facecolor="white", metadata={"Creator": "Matplotlib; original HDF5 terminal frames"})
    fig.savefig(out/f"{name}.png", dpi=600, facecolor="white")
    plt.close(fig)
    return {"name": name, "size_inches": [6.5, 2.4], "png_dpi": 600,
            "source_figure_function": "boundary_defect_analysis.create_comparison_states",
            "same_terminal_frame_selection_as_V2": True, "panels": records}


def main():
    source.BOUNDARY_LINEWIDTH = .35
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                         "axes.linewidth": .45}):
        result = [redraw(*group) for group in model_groups()]
    (ROOT/"particles"/"particle_redraw_provenance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: 28 exact terminal frames; 56000 particles; vector PDF + 600 dpi PNG.")


if __name__ == "__main__":
    main()
