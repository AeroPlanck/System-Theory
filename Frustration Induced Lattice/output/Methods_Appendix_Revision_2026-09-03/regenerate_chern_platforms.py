"""Pole-formula platforms with independent radial gap screening.

Preserves the published v=3, omega=0, d0=1/2/3 and effective K=10/20/30
parameter grid. Extra endpoint-near samples expose undefined single bands.
NaN samples interrupt curves; they are never converted to zero.
"""

from itertools import combinations
from pathlib import Path
import csv
import json
import sys
import time
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from ChernNumberCompute import check_spectral_separation, compute_topology


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fractions = np.unique(np.r_[np.linspace(0.00001, 1.0, 25),
                               0., .01, .02, .025, .03, .97, .975, .98, .99])
    combos = [x for n in (1, 2, 3) for x in combinations(range(3), n)]
    rho0 = .0204
    rows, diagnostics, curves = [], [], {}
    start = time.perf_counter()
    for d0 in (1., 2., 3.):
        for coupling in (10., 20., 30.):
            lam = coupling / (rho0 * np.pi * d0**2)
            for combo in combos:
                curves[d0, coupling, combo] = []
            for fraction in fractions:
                params = (3., 0., lam, float(fraction*np.pi), rho0, d0)
                for combo in combos:
                    diag = check_spectral_separation(params, combo)
                    value = diag["pole_chern"] if diag["valid"] else np.nan
                    curves[d0, coupling, combo].append(value)
                    row = dict(v=3., omega=0., rho0=rho0, d0=d0, lam=lam,
                               K_effective=coupling, alpha_over_pi=float(fraction),
                               bands="+".join(map(str, combo)),
                               C_pole=int(value) if np.isfinite(value) else "NA",
                               status=diag["status"], reason=diag["reason"],
                               min_relative_gap=diag["min_relative_gap"],
                               finite_k_max=diag.get("finite_k_max", "NA"),
                               globally_proven=diag["globally_proven"])
                    rows.append(row)
                    diagnostics.append({**row, "diagnostic": diag})
            print(f"d0={d0:g}, K={coupling:g}: completed ({time.perf_counter()-start:.1f}s)", flush=True)
    with (out / "chern_platforms_gap_screened.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "chern_platforms_gap_screened_diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(json_safe(diagnostics), stream, indent=2, allow_nan=False)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 10,
                         "axes.titlesize": 10, "axes.labelsize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    for kind, selected in (("single", [c for c in combos if len(c)==1]),
                           ("mixed", [c for c in combos if len(c)>1])):
        fig, axes = plt.subplots(3, 3, figsize=(8.7, 6.7), sharex=True, sharey=True)
        handles = []
        colors = ["#126E9B", "#B64938", "#458443", "#7956A6"]
        styles = ["-", "--", "-.", ":"]
        for i, d0 in enumerate((1.,2.,3.)):
            for j, coupling in enumerate((10.,20.,30.)):
                ax = axes[i,j]
                for n, combo in enumerate(selected):
                    line, = ax.plot(fractions, curves[d0,coupling,combo], styles[n],
                                    color=colors[n], lw=1.6, marker=".", ms=2.4,
                                    label=r"$\mathcal{I}=\{"+",".join(map(str,combo))+r"\}$")
                    if i==j==0:
                        handles.append(line)
                ax.set_title(rf"$d_0={d0:g},\quad \lambda\rho_0G_0={coupling:g}$", pad=5)
                ax.set_xlim(-.03,1.03)
                ax.set_ylim(-2.45,2.45)
                ax.set_xticks([0,.5,1])
                ax.set_yticks([-2,0,2])
                ax.grid(True, alpha=.22)
                if i==2:
                    ax.set_xlabel(r"$\alpha/\pi$")
                if j==0:
                    ax.set_ylabel(r"$C$")
        fig.suptitle(f"{kind.capitalize()}-band Chern: radial gap screening + pole formula", fontsize=13, y=.995)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.958),
                   ncol=len(handles), frameon=False, fontsize=11)
        fig.subplots_adjust(left=.075,right=.99,bottom=.12,top=.85,hspace=.34,wspace=.17)
        fig.text(.075,.025, "Undefined / unresolved samples are gaps, not zeros. Lines only connect sampled valid values.", fontsize=9)
        fig.savefig(out/f"{kind}_v_3_omega_0_gap_screened.png", dpi=300)
        plt.close(fig)

    # Separate determinant-link evaluations test, but do not replace, the pole values.
    checks = []
    for fraction in (.2,.5,.8):
        params = (3.,0.,20.75/(rho0*np.pi),fraction*np.pi,rho0,1.)
        for combo in ((0,), (2,), (0,1)):
            raw, integer, diag = compute_topology(params,combo,N_phi=61)
            checks.append(dict(alpha_over_pi=fraction,bands=list(combo),raw=raw,
                               integer=integer,status=diag["status"],
                               pole=diag["pole_chern"],valid=diag["valid"]))
            if not diag["valid"] or integer!=diag["pole_chern"]:
                raise RuntimeError(f"Fukui cross-check failed: {checks[-1]}")
    with (out/"chern_platform_fukui_crosschecks.json").open("w",encoding="utf-8") as stream:
        json.dump(json_safe(checks),stream,indent=2,allow_nan=False)
    print(f"Completed {len(rows)} band-cluster samples and {len(checks)} Fukui checks.",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-existing-json", action="store_true",
                        help="Normalize unavailable diagnostic values to strict JSON null.")
    parser.add_argument("--reuse-screening", action="store_true",
                        help="Redraw saved gap-screening results, then repeat fresh Fukui checks.")
    args = parser.parse_args()
    if args.format_existing_json:
        for name in ("chern_platforms_gap_screened_diagnostics.json",
                     "chern_platform_fukui_crosschecks.json"):
            path = ROOT / "figures" / name
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                path.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False), encoding="utf-8")
    else:
        if args.reuse_screening:
            saved = json.loads((ROOT / "figures" / "chern_platforms_gap_screened_diagnostics.json").read_text(encoding="utf-8"))
            screening_cache = {
                (round(item["d0"], 10), round(item["K_effective"], 10),
                 round(item["alpha_over_pi"], 10), item["bands"]): item["diagnostic"]
                for item in saved
            }
            def check_spectral_separation(params, combo):
                v, omega, lam, alpha, rho0, d0 = params
                if v != 3 or omega != 0 or rho0 != .0204:
                    raise ValueError("Saved screening is only for the specified platform grid")
                key = (round(d0, 10), round(lam*rho0*np.pi*d0*d0, 10),
                       round(alpha/np.pi, 10), "+".join(map(str, combo)))
                return screening_cache[key]
        main()
