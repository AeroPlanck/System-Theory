"""Publication-size redraws: vector PDF and genuine 600 dpi PNG.

No particle simulation or Chern/strip scan is run. Chern and strip values
are read verbatim from the completed revision. The two inexpensive 1D
3x3 matrix cuts reuse the original Dispersion implementation and parameters.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
from PIL import Image


OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
REVISION = WORKSPACE / "output" / "Methods_Appendix_Revision_2026-09-03"
ORIGINAL_FIGURES = Path(r"D:\LaTex\Boundary Flow\Figures")
CHERN_CSV = REVISION / "figures" / "chern_platforms_gap_screened.csv"
STRIP_NPZ = REVISION / "figures" / "strip_data" / "alpha_0.5_fine.npz"
STRIP_DIAGNOSTICS = REVISION / "figures" / "strip_circle_matched_diagnostics.json"
BEFORE_DISPERSION = REVISION / "before" / "Dispersion.py"
CURRENT_DISPERSION = REVISION / "code" / "Dispersion.py"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 7.4,
    "axes.labelsize": 8.0, "axes.titlesize": 7.5,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2, "axes.linewidth": 0.65,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.width": 0.65, "ytick.major.width": 0.65,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "path.simplify": False, "savefig.dpi": 600,
    "axes.unicode_minus": True,
})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(fig, stem, title, source_names):
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    metadata = {"Title": title, "Author": "Methods Appendix figure redraw",
                "Subject": "; ".join(source_names),
                "Creator": "Matplotlib: vector paths and embedded TrueType fonts",
                "CreationDate": None, "ModDate": None}
    fig.savefig(pdf, format="pdf", facecolor="white", metadata=metadata)
    fig.savefig(png, format="png", dpi=600, facecolor="white")
    with Image.open(png) as image:
        pixels = list(image.size)
        dpi = list(image.info["dpi"])
    record = {"pdf": pdf.name, "png": png.name,
              "width_inches": float(fig.get_figwidth()),
              "height_inches": float(fig.get_figheight()),
              "png_pixels": pixels, "png_dpi": dpi,
              "pdf_sha256": sha256(pdf), "png_sha256": sha256(png)}
    plt.close(fig)
    return record


def dispersion_figures():
    before = load_module("dispersion_before_figure_redraw", BEFORE_DISPERSION)
    current = load_module("dispersion_current_figure_redraw", CURRENT_DISPERSION)
    records = []
    k = np.linspace(-10., 10., 1000)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for fraction in (.4, .6):
        params = (3., 0., 20./(.0204*np.pi*2**2), fraction*np.pi, .0204, 2.)
        matrices = before.M_matrix_vectorized(k, np.zeros_like(k), *params)
        updated = current.M_matrix_vectorized(k, np.zeros_like(k), *params)
        matrix_error = float(np.max(np.abs(matrices-updated)))
        assert matrix_error < 1e-12
        values = before.sort_eigs_continuous(np.linalg.eigvals(matrices))
        # The original PNG starts with +Im (blue), -Im (orange), real (green).
        assert values[0, 0].imag > 0 and values[0, 1].imag < 0
        assert abs(values[0, 2].imag) < 1e-12
        fig, axes = plt.subplots(2, 1, figsize=(1.70, 2.72), sharex=True)
        for branch in range(3):
            axes[0].plot(k, values[:, branch].real, color=colors[branch], lw=.9,
                         label=rf"${branch+1}$")
            axes[1].plot(k, values[:, branch].imag, color=colors[branch], lw=.9)
        for ax in axes:
            ax.axhline(0, color="black", lw=.55, zorder=4)
            ax.set_xlim(-11., 11.)
            ax.tick_params(direction="in", pad=2, labelsize=7.5)
            ax.xaxis.set_major_locator(FixedLocator([-10., 0., 10.]))
        if fraction == .4:
            axes[0].set_ylim(-.56, 3.27)
            axes[0].set_yticks([0., 1., 2., 3.])
        else:
            axes[0].set_ylim(-3.27, .56)
            axes[0].set_yticks([-3., -2., -1., 0.])
        axes[1].set_ylim(-27.5, 27.5)
        axes[1].set_yticks([-20., 0., 20.])
        axes[0].set_ylabel(r"$\mathrm{Re}\,\sigma(k)$", labelpad=3)
        axes[1].set_ylabel(r"$\mathrm{Im}\,\sigma(k)$", labelpad=3)
        axes[1].set_xlabel(r"$k$", labelpad=2)
        handles = [Line2D([], [], color=c, lw=1.1, label=rf"$n={i+1}$")
                   for i, c in enumerate(colors)]
        fig.legend(handles=handles, loc="upper center", ncol=3,
                   bbox_to_anchor=(.59, 1.0), frameon=False, handlelength=1.0,
                   handletextpad=.3, columnspacing=.7, borderaxespad=0, fontsize=7.5)
        fig.subplots_adjust(left=.265, right=.975, top=.918, bottom=.14, hspace=.16)
        stem = f"plot_1d_dispersion_alpha_{fraction:.2f}pi"
        record = export(fig, stem, f"Directional dispersion, alpha={fraction} pi",
                        [str(BEFORE_DISPERSION)])
        source_png = ORIGINAL_FIGURES / f"{stem}.png"
        np.savez_compressed(OUT/f"{stem}_data.npz", k=k, eigvals=values,
                            params=np.asarray(params))
        record.update(original_image=str(source_png), original_image_sha256=sha256(source_png),
                      params_order="v,omega,lambda,alpha,rho0,d0", params=list(params),
                      K_effective=20., k_count=len(k), k_range=[-10.,10.],
                      branch_colors=list(colors),
                      maximum_matrix_difference_from_current=matrix_error,
                      origin_eigenvalues=[[float(z.real), float(z.imag)]
                                         for z in before.eigs_at_k(0.,0.,params)],
                      numerical_source="Original Dispersion.py; same 1000-point cut and continuous branch ordering",
                      provenance_note="No original numeric sidecar was located. Parameters and original branch ordering were verified against source defaults and visible PNG landmarks.",
                      tex_recommended_width="0.48\\linewidth inside the existing single-column pair",
                      minimum_label_points=7.5)
        records.append(record)
    return records


def chern_figures():
    with CHERN_CSV.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2142
    assert all(float(row["v"]) == 3 and float(row["omega"]) == 0 for row in rows)
    fractions = np.array(sorted({float(row["alpha_over_pi"]) for row in rows}))
    lookup = {(float(row["d0"]), float(row["K_effective"]), row["bands"],
               float(row["alpha_over_pi"])): row for row in rows}
    records = []
    for kind, combos in (("mixed", ("0+1", "0+2", "1+2", "0+1+2")),
                         ("single", ("0", "1", "2"))):
        fig, axes = plt.subplots(3, 3, figsize=(3.43, 3.36), sharex=True, sharey=True)
        colors = ["#126E9B", "#B64938", "#458443", "#7956A6"]
        styles = ["-", "--", "-.", ":"]
        all_values, exact_mask, used_rows, handles = [], [], [], []
        for i, d0 in enumerate((1.,2.,3.)):
            for j, coupling in enumerate((10.,20.,30.)):
                ax = axes[i,j]
                for n, combo in enumerate(combos):
                    selected = [lookup[d0,coupling,combo,a] for a in fractions]
                    values = np.array([np.nan if row["C_pole"] == "NA" else float(row["C_pole"])
                                       for row in selected])
                    mask = np.array([row["C_pole"] == "NA" for row in selected])
                    assert np.array_equal(np.isnan(values),mask)
                    line, = ax.plot(fractions, values, styles[n], color=colors[n], lw=.95,
                                    marker=".", ms=2.0,
                                    label=r"$\{"+combo.replace("+",",")+r"\}$")
                    if i == j == 0:
                        handles.append(line)
                    all_values.append(values)
                    exact_mask.append(mask)
                    used_rows.extend(selected)
                ax.set_title(rf"$d_0={d0:g},\ K={coupling:g}$", fontsize=7.3, pad=3)
                ax.set_xlim(-.03,1.03)
                ax.set_ylim(-2.45,2.45)
                ax.set_xticks([0.,.5,1.])
                ax.set_yticks([-2.,0.,2.])
                ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
                ax.tick_params(pad=1.8, labelsize=7.2, length=2)
                ax.grid(True, linewidth=.35, color="#D9DDE1", zorder=0)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53,.995),
                   ncol=len(handles), frameon=False, handlelength=1.25,
                   handletextpad=.3, columnspacing=.7, fontsize=7.2)
        fig.text(.53,.904, r"$K=\lambda\rho_0\widehat G(0),\quad \mathcal{I}$ shown in legend",
                 ha="center", fontsize=7.2)
        fig.subplots_adjust(left=.112,right=.985,bottom=.16,top=.823,hspace=.61,wspace=.16)
        fig.text(.035,.52,r"$C$",rotation=90,ha="center",va="center",fontsize=8.5)
        fig.text(.54,.092,r"$\alpha/\pi$",ha="center",fontsize=8.5)
        fig.text(.53,.033,"Gaps: invalid or unresolved target projector.",
                 ha="center",fontsize=7.2)
        stem = f"{kind}_v_3_omega_0_gap_screened"
        record = export(fig,stem,f"{kind.capitalize()} Chern platforms: saved gap-screened data",
                        [str(CHERN_CSV)])
        np.savez_compressed(OUT/f"{stem}_plotted_data.npz", fractions=fractions,
                            values=np.asarray(all_values), invalid=np.asarray(exact_mask))
        record.update(original_image=str(ORIGINAL_FIGURES/f"{stem}.png"),
                      original_image_sha256=sha256(ORIGINAL_FIGURES/f"{stem}.png"),
                      source_csv=str(CHERN_CSV), source_csv_sha256=sha256(CHERN_CSV),
                      exact_csv_rows_used=len(used_rows), alpha_count=len(fractions),
                      valid_samples=sum(row["C_pole"]!="NA" for row in used_rows),
                      unavailable_samples=sum(row["C_pole"]=="NA" for row in used_rows),
                      no_interpolation_across_NA=True, numerical_difference_from_CSV=0.,
                      parameter_grid={"v":3.,"omega":0.,"rho0":.0204,
                                      "d0":[1.,2.,3.],"K_effective":[10.,20.,30.]},
                      tex_recommended_width="0.49\\linewidth inside the existing figure* pair",
                      minimum_label_points=7.2)
        records.append(record)
    return records


def strip_figure():
    with np.load(STRIP_NPZ) as saved:
        data = {name:saved[name].copy() for name in saved.files}
    ky, vals = data["ky"], data["eigvals"]
    params = data["params"]
    v,omega,lam,alpha,rho0,d0 = params
    assert vals.shape == (201,108)
    assert np.allclose([v,omega,alpha/np.pi,d0,lam*rho0*np.pi*d0*d0],
                       [3.,0.,.5,1.,20.75])
    diagnostics = json.loads(STRIP_DIAGNOSTICS.read_text(encoding="utf-8"))
    record = next(row for row in diagnostics["records"] if row["alpha_over_pi"]==.5)
    counts = record["fine"]["reference_counts"]
    assert all((counts[str(c)]["count_left"],counts[str(c)]["count_right"])==(2,-2)
               for c in (10.,-10.))
    wl,wr = data["left_weight"],data["right_weight"]
    # Exactly the masks in the original plot_verified_style: no new removal,
    # averaging, branch tracking or ambiguity-dependent recoloring.
    weak = np.maximum(wl,wr)<.45
    left = (wl>=.45)&(wl>wr)
    right = (wr>=.45)&(wr>wl)
    x = np.broadcast_to(ky[:,None],vals.shape)
    fig,ax = plt.subplots(figsize=(3.43,2.94))
    ax.plot(x[weak],vals.imag[weak],".",color="lightgray",ms=1.35,alpha=.7,
            rasterized=False,linestyle="none")
    ax.plot(x[left],vals.imag[left],".",color="#1f77b4",ms=2.05,
            rasterized=False,linestyle="none")
    ax.plot(x[right],vals.imag[right],".",color="#ff7f0e",ms=2.05,
            rasterized=False,linestyle="none")
    ax.axhline(10.,color="black",ls="--",lw=.85)
    ax.axhline(-10.,color="black",ls=":",lw=.85)
    ax.set_xlim(-50.,50.)
    ax.set_ylim(-40.,40.)
    ax.set_xticks([-50.,-25.,0.,25.,50.])
    ax.set_yticks([-40.,-20.,-10.,0.,10.,20.,40.])
    ax.tick_params(labelsize=7.5,pad=2)
    ax.set_xlabel(r"$k_y$",fontsize=8.5,labelpad=2)
    ax.set_ylabel(r"$\mathrm{Im}\,\sigma$",fontsize=8.5,labelpad=3)
    ax.grid(True,alpha=.7,color="#D9DDE1",lw=.4)
    ax.set_axisbelow(True)
    handles = [Line2D([],[],marker=".",ls="",ms=4,color="#1f77b4",label="Left edge"),
               Line2D([],[],marker=".",ls="",ms=4,color="#ff7f0e",label="Right edge"),
               Line2D([],[],marker=".",ls="",ms=4,color="lightgray",label="Other modes")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,.938),
               frameon=False,ncol=3,fontsize=7.5,columnspacing=.65,
               handletextpad=.15,handlelength=.9)
    fig.text(.55,.973,r"$\alpha=\pi/2,\ v=3,\ \omega=0,\ K=20.75,\ d_0=1$",
             ha="center",va="top",fontsize=8.)
    fig.subplots_adjust(left=.155,right=.98,bottom=.20,top=.837)
    fig.text(.55,.048,r"$c=\pm10:\quad(n_L,n_R)=(2,-2)$",ha="center",fontsize=8.)
    stem = "strip_circle_matched_alpha_050"
    result = export(fig,stem,"Finite-strip spectrum at the matched threshold parameters",
                    [str(STRIP_NPZ),str(STRIP_DIAGNOSTICS)])
    result.update(original_image=str(ORIGINAL_FIGURES/f"{stem}.png"),
                  original_image_sha256=sha256(ORIGINAL_FIGURES/f"{stem}.png"),
                  source_npz=str(STRIP_NPZ),source_npz_sha256=sha256(STRIP_NPZ),
                  source_diagnostics=str(STRIP_DIAGNOSTICS),
                  source_diagnostics_sha256=sha256(STRIP_DIAGNOSTICS),
                  params_order="v,omega,lambda,alpha,rho0,d0",params=params.tolist(),
                  stored_shape=list(vals.shape),stored_eigenvalues=int(vals.size),
                  ky_range=[float(ky.min()),float(ky.max())],
                  exact_original_mask_counts={"weak":int(weak.sum()),"left":int(left.sum()),
                                              "right":int(right.sum())},
                  coordinate_change=0.,reference_levels=[10.,-10.],
                  left_right_counts_at_both_levels=[2,-2],
                  source_arrays_reused_without_recomputation=True,
                  tex_recommended_width="\\linewidth in the existing single-column strip panel",
                  minimum_label_points=7.5)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    originals = [BEFORE_DISPERSION,CURRENT_DISPERSION,CHERN_CSV,STRIP_NPZ,STRIP_DIAGNOSTICS]
    hashes = {str(path):sha256(path) for path in originals}
    records = dispersion_figures()+chern_figures()+[strip_figure()]
    assert all(sha256(Path(path))==value for path,value in hashes.items())
    manifest = {"scope":"Five theory figures only; no particle simulations or long spectral scans",
                "scientific_data_changed":False,
                "source_sha256_before_and_after_equal":True,
                "source_hashes":hashes,"figures":records,
                "rasterization_requested":False,
                "source_PNGs_not_used_as_redraw_backgrounds":True}
    (OUT/"Theory_Figure_Verification.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps([{k:r[k] for k in ("pdf","png","width_inches","height_inches","png_pixels")}
                      for r in records],indent=2))


if __name__=="__main__":
    main()
