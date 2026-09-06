"""Print-size vector redraws of the three Methods definition schematics.

No trajectory, algorithm, legacy figure, or TeX source is modified.  All page
coordinates are in inches, so the declared font sizes remain the printed sizes
when these 6.5-inch-wide PDFs are included at the Methods text width.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle


OUT = Path(__file__).resolve().parent
WIDTH = 6.5
DPI = 600
INK = "#202124"
GRAY = "#596169"
LINE = "#D8DDE1"
BLUE = "#2A6F85"
PURPLE = "#76519A"
GREEN = "#3E7C4B"
RED = "#A33D3D"
GOLD = "#93621C"
LIGHT_BLUE = "#F0F6F8"
LIGHT_GREEN = "#F2F7F2"
LIGHT_PURPLE = "#F6F2F9"
LIGHT_GOLD = "#FBF7EF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

QA = []


def page(height, title):
    fig = plt.figure(figsize=(WIDTH, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, height), aspect="equal")
    ax.axis("off")
    text(ax, 0.16, height - 0.15, title, size=12.0, weight="bold", va="top")
    ax.plot([0.16, 6.34], [height - 0.43] * 2, color=LINE, lw=0.8)
    return fig, ax


def text(ax, x, y, value, *, size=9.0, color=INK, ha="left", va="center", weight="normal"):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, linespacing=1.25)


def panel(ax, x, top, width, height, label, title):
    ax.add_patch(Rectangle((x, top - height), width, height, facecolor="white",
                           edgecolor=LINE, linewidth=0.7))
    text(ax, x + 0.10, top - 0.16, f"({label})  {title}", size=10.0, weight="bold")


def box(ax, x, y, width, height, lines, *, edge=BLUE, face=LIGHT_BLUE, size=9.0):
    ax.add_patch(Rectangle((x, y), width, height, edgecolor=edge,
                           facecolor=face, linewidth=0.85))
    text(ax, x + width / 2, y + height / 2, lines, size=size, ha="center")


def arrow(ax, start, end, *, color=GRAY, lw=1.1, head=8.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))


def dot(ax, x, y, *, color=BLUE, radius=0.045):
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="white", lw=0.6, zorder=5))


def save(fig, ax, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for item in ax.texts:
        bounds = item.get_window_extent(renderer)
        if bounds.x0 < -0.5 or bounds.y0 < -0.5 or bounds.x1 > fig.bbox.width + 0.5 or bounds.y1 > fig.bbox.height + 0.5:
            clipped.append(item.get_text())
    if clipped:
        raise RuntimeError(f"Text outside {stem}: {clipped}")
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fig.savefig(pdf, metadata={"Title": stem.replace("_", " "), "Subject": "Vector schematic; illustrative, not simulation data"})
    fig.savefig(png, dpi=DPI)
    QA.append({"stem": stem, "width_inches": WIDTH, "height_inches": float(fig.get_figheight()),
               "png_dpi": DPI, "png_width_pixels": int(WIDTH * DPI),
               "png_height_pixels": int(round(fig.get_figheight() * DPI)),
               "minimum_declared_text_points": min(item.get_fontsize() for item in ax.texts),
               "text_outside_page": clipped, "pdf": str(pdf), "png": str(png)})
    plt.close(fig)


def projection_and_w():
    fig, ax = page(6.9, "Boundary projection and the W membership gate")
    panel(ax, .16, 6.34, 3.00, 2.65, "a", "Why distance is not enough")
    panel(ax, 3.34, 6.34, 3.00, 2.65, "b", "Local boundary coordinates")
    panel(ax, .16, 3.49, 3.00, 3.07, "c", "Near-tangential turning balance")
    panel(ax, 3.34, 3.49, 3.00, 3.07, "d", "Membership, not a weight")

    # (a) Both illustrative particles are near the same circular boundary.
    center = np.array([1.57, 5.17])
    radius = .70
    ax.add_patch(Circle(center, radius, fill=False, edgecolor=INK, lw=1.25))
    ax.add_patch(Circle(center, radius - .22, fill=False, edgecolor=GRAY, lw=.8, linestyle="--"))
    text(ax, 1.64, 6.00, "Physical boundary", size=8.7, ha="center")
    wall = center + .64 * np.array([np.cos(.42), np.sin(.42)])
    dot(ax, *wall, color=RED)
    arrow(ax, wall, wall + .26 * np.array([-np.sin(.42), np.cos(.42)]), color=RED)
    text(ax, 2.40, 5.17, "Boundary\ncandidate", size=8.5, color=RED)
    vortex = center + np.array([-.42, -.10])
    ax.add_patch(Arc(vortex, .54, .54, theta1=15, theta2=325, color=PURPLE, lw=1.15))
    arrow(ax, vortex + [.16, -.21], vortex + [.07, -.26], color=PURPLE, head=7)
    dot(ax, *(vortex + [-.27, 0]), color=PURPLE)
    dot(ax, *vortex, color=PURPLE, radius=.018)
    text(ax, 1.45, 4.34, "Nearby interior vortex", size=8.5, color=PURPLE, ha="center")
    box(ax, .29, 3.82, 2.74, .40, "Both can satisfy " + r"$d_i^n\leq d_0$." + "\nDistance alone does not separate them.",
        edge=GOLD, face=LIGHT_GOLD, size=8.8)

    # (b) The tangent is counterclockwise; the normal points inward.
    center = np.array([4.44, 4.95])
    radius, distance, phi = .72, .19, .62
    radial = np.array([np.cos(phi), np.sin(phi)])
    tangent = np.array([-np.sin(phi), np.cos(phi)])
    boundary = center + radius * radial
    particle = center + (radius - distance) * radial
    ax.add_patch(Circle(center, radius, fill=False, edgecolor=INK, lw=1.25))
    ax.add_patch(Circle(center, radius - distance, fill=False, edgecolor=BLUE, lw=.8, linestyle="--"))
    ax.plot([particle[0], boundary[0]], [particle[1], boundary[1]], color=GOLD, lw=1.4)
    dot(ax, *particle, color=RED)
    dot(ax, *boundary, color=INK, radius=.026)
    arrow(ax, particle, particle + .42 * tangent, color=BLUE)
    arrow(ax, particle, particle - .32 * radial, color=GREEN)
    arrow(ax, particle, particle + .37 * (.87 * tangent - .49 * radial), color=RED)
    text(ax, 4.65, 5.68, r"$\mathbf{t}_i^n$", size=10.4, color=BLUE)
    text(ax, 4.12, 5.44, r"$\mathbf{u}_i^n$", size=10.4, color=RED)
    text(ax, 4.42, 5.02, "inward\nnormal", size=8.3, color=GREEN, ha="center")
    text(ax, 5.05, 5.42, r"$d_i^n$", size=10.2, color=GOLD)
    text(ax, 5.15, 5.19, r"$\Pi(\mathbf{X}_i^n)$", size=9.0)
    text(ax, 4.89, 4.85, r"$\mathbf{X}_i^n$", size=10.0, color=RED)
    text(ax, 4.84, 4.17, r"$q_i^n=\mathbf{u}_i^n\!\cdot\!\mathbf{t}_i^n$", size=10.0, ha="center")
    text(ax, 4.84, 3.94, r"$q>0$: CCW;  $q<0$: CW", size=8.8, ha="center", color=GRAY)
    text(ax, 4.84, 5.94, r"$\kappa_{d,i}^n=\dfrac{\kappa_{b,i}^n}{1-\kappa_{b,i}^n d_i^n}$", size=10.0, ha="center")

    # (c) This is a selection proxy, not a measured force or causal test.
    text(ax, 1.66, 3.04, "Approximation: nearly tangent motion\nat nearly fixed wall distance.", size=8.6, ha="center", color=GRAY)
    box(ax, .30, 2.32, 2.72, .46,
        "Curve-following requirement\n" + r"$a_{\rm req,\perp}=\kappa_{d,i}^n v^2(q_i^n)^2$", size=9.2)
    box(ax, .30, 1.59, 2.72, .53,
        "Normal acceleration from free rotation\n" + r"$a_{\rm free,\perp}=v\Omega_i^n q_i^n$",
        edge=GREEN, face=LIGHT_GREEN, size=9.0)
    arrow(ax, (1.66, 2.25), (1.66, 2.16), color=GRAY)
    text(ax, 1.66, 1.41, r"$\Omega_i^n$: recomputed microscopic angular rate", size=8.3, ha="center", color=GRAY)
    arrow(ax, (1.66, 1.25), (1.66, 1.14), color=GRAY)
    box(ax, .30, .57, 2.72, .55,
        r"$W_i^n=\dfrac{a_{\rm req,\perp}-a_{\rm free,\perp}}{v}$" + "\n" +
        r"$=\kappa_{d,i}^n v(q_i^n)^2-\Omega_i^n q_i^n$",
        edge=RED, face="#FBF2F2", size=10.0)

    # (d) Exactly the applicable high-alpha criteria, without a force claim.
    box(ax, 3.48, 2.63, 2.72, .48,
        r"$|q_i^n|\geq0.05,\quad d_i^n\leq d_0$" + "\n" +
        r"valid projection; $1-\kappa_{b,i}^n d_i^n>0$", edge=GRAY, face="#F7F8F8", size=9.2)
    arrow(ax, (4.84, 2.56), (4.84, 2.42))
    box(ax, 3.48, 1.73, 1.30, .59, r"$W_i^n>\varepsilon_W$" + "\nRetain candidate", edge=RED, face="#FBF2F2", size=9.0)
    box(ax, 4.90, 1.73, 1.30, .59, r"$W_i^n\leq\varepsilon_W$" + "\nExclude candidate", edge=PURPLE, face=LIGHT_PURPLE, size=9.0)
    text(ax, 4.84, 1.49, r"$\varepsilon_W=0.02v/d_0$", size=10.0, ha="center")
    arrow(ax, (4.12, 1.66), (4.12, 1.24), color=RED)
    box(ax, 3.48, .59, 2.72, .62,
        r"$\mathcal{S}_{ng}\ \longrightarrow\ r_{ng}\ \longrightarrow\ Q_{ng}$" + "\n" +
        "Discard the magnitude of " + r"$W_i^n$." + "\nOnly membership is carried forward.",
        edge=GREEN, face=LIGHT_GREEN, size=9.0)
    text(ax, 3.25, .22, r"$W$ is used only for $\pi/2<\alpha<\pi$: a conditional selection proxy, not a measured force.",
         size=8.6, ha="center", color=GRAY)
    save(fig, ax, "Boundary_Projection_And_W_Filter_Schematic")


def relay_signal():
    fig, ax = page(7.35, "Constructing a relay-compatible boundary signal")
    panel(ax, .16, 6.78, 3.00, 3.02, "a", "One representative per cell")
    panel(ax, 3.34, 6.78, 3.00, 3.02, "b", "Check actual displacement")
    panel(ax, .16, 3.56, 3.00, 3.12, "c", "Relay across two frames")
    panel(ax, 3.34, 3.56, 3.00, 3.12, "d", "Window and time blocks")

    # (a) The closest retained candidate is A, even though B has opposite sign.
    text(ax, 1.66, 6.30, r"$\mathcal{S}_{ng}=\{A,B,C\}$", size=10.0, ha="center")
    for g in range(5):
        x = .39 + .50 * g
        ax.add_patch(Rectangle((x, 4.81), .50, .14, facecolor=LIGHT_BLUE,
                               edgecolor="white", lw=.7))
        text(ax, x + .25, 4.64, rf"$I_{{{g + 1}}}$", size=9.0, ha="center")
    for x, y, name, sign in [(1.49, 5.17, "A", 1), (1.79, 5.59, "B", -1), (1.63, 5.96, "C", 1)]:
        color = BLUE if sign > 0 else PURPLE
        ax.plot([x, x], [4.95, y - .06], color=GRAY, lw=.7, ls=":")
        dot(ax, x, y, color=color)
        arrow(ax, (x, y), (x + sign * .22, y), color=color, head=7)
        text(ax, x + .10, y + .12, name, size=9.0, weight="bold", color=color)
    box(ax, .30, 3.92, 2.72, .51,
        r"$r_{ng}=A,\quad Q_{ng}=\operatorname{sgn}q_A^n=+1$" + "\n" +
        "Opposite-sign nearest tie: " + r"$Q_{ng}=0$.", size=9.0)

    # (b) Only calibration follows the same particle through both frames.
    line_y = 6.04
    ax.plot([3.68, 6.0], [line_y, line_y], color=INK, lw=1.2)
    dot(ax, 4.05, line_y, color=PURPLE)
    dot(ax, 5.50, line_y, color=BLUE)
    arrow(ax, (4.16, line_y + .10), (5.39, line_y + .10), color=BLUE)
    text(ax, 4.05, 6.26, r"$s_i^n$", size=10, ha="center")
    text(ax, 5.50, 6.26, r"$s_i^{n+1}$", size=10, ha="center")
    text(ax, 4.80, 5.83, r"$\Delta s_i^n=\operatorname{wrap}_{[-P/2,P/2)}(s_i^{n+1}-s_i^n)$", size=8.7, ha="center")
    box(ax, 3.48, 4.95, 2.72, .72,
        "Same particle and smooth segment;\nselection and nonzero heading sign retained;\n" +
        r"unique step $|\Delta s_i^n|\leq J<P/2$: $T_{ng}=1$." + "\n" +
        r"$J=1.05v\Delta t_s/(1-\kappa_b d_{\max})$", size=8.5)
    box(ax, 3.48, 4.36, 2.72, .47,
        r"$E_{ng}=T_{ng}\,\mathbf{1}[|\Delta s_i^n|/(v\Delta t_s)\geq0.02]$" + "\n" +
        r"$\times\mathbf{1}[\operatorname{sgn}\Delta s_i^n=Q_{ng}]$",
        edge=GREEN, face=LIGHT_GREEN, size=8.9)
    text(ax, 4.84, 4.04, r"$\sum_nT_{ng}\geq3,\quad C_g=\dfrac{\sum_nE_{ng}}{\sum_nT_{ng}}\geq\dfrac{2}{3}$",
         size=10.0, ha="center")

    # (c) A different carrier can preserve the same local directional signal.
    for y, n, name in [(2.85, "n", "A"), (2.16, "n+1", "B")]:
        text(ax, .34, y, rf"Frame ${n}$", size=9.0)
        ax.add_patch(Rectangle((1.08, y - .16), 1.89, .32,
                               facecolor=LIGHT_BLUE, edgecolor=LINE, lw=.7))
        dot(ax, 1.35, y, color=BLUE)
        arrow(ax, (1.40, y), (1.63, y), color=BLUE, head=7)
        text(ax, 1.79, y, name, size=9.5, color=BLUE, weight="bold")
        text(ax, 2.87, y, r"$Q=+1$", size=9.5, color=BLUE, ha="right")
    arrow(ax, (1.84, 2.59), (1.84, 2.42), color=GREEN)
    text(ax, 1.66, 1.70, "Different carriers; same cell and sign.", size=8.8, ha="center", color=GRAY)
    box(ax, .30, .99, 2.72, .49,
        r"$Q_{ng}=Q_{n+1,g}=+1,\quad A\ne B$" + "\n" + r"$\Longrightarrow\quad Y_{ng}=+1$", size=10.0)
    text(ax, 1.66, .72, "Otherwise " + r"$Y_{ng}=0$." + " No full circuit\nis required of any one particle.",
         size=8.7, ha="center", color=GRAY)

    # (d) The formulas are identical to the saved-frame terminal-window rule.
    box(ax, 3.48, 2.78, 2.72, .37,
        r"$\Delta t_s=\Delta t\,n_{\rm snap},\qquad T_{\rm req}=10P/v$", edge=GRAY, face="#F7F8F8", size=9.8)
    arrow(ax, (4.84, 2.73), (4.84, 2.61))
    box(ax, 3.48, 2.09, 2.72, .48,
        r"$n_{\rm blk}=\lceil(d_0/v)/\Delta t_s\rceil$" + "\n" +
        r"$\Delta T_b=n_{\rm blk}\Delta t_s$", size=10.0)
    arrow(ax, (4.84, 2.03), (4.84, 1.92))
    box(ax, 3.48, 1.24, 2.72, .64,
        r"$N_{\rm avail}=\min(F-1,\lceil T_{\rm req}/\Delta t_s\rceil)$" + "\n" +
        r"$B=\lfloor N_{\rm avail}/n_{\rm blk}\rfloor$" + "\n" + r"$n_0=F-1-Bn_{\rm blk}$",
        edge=PURPLE, face=LIGHT_PURPLE, size=9.5)
    arrow(ax, (4.84, 1.19), (4.84, 1.08))
    box(ax, 3.48, .60, 2.72, .44,
        r"Use frames $n_0,\ldots,F-1$." + "\n" +
        r"$f_T=Bn_{\rm blk}\Delta t_s/T_{\rm req}$", edge=GREEN, face=LIGHT_GREEN, size=9.5)
    text(ax, 3.25, .22, "Calibration follows a particle; the relay signal follows a boundary cell. Illustrative geometry only.",
         size=8.6, ha="center", color=GRAY)
    save(fig, ax, "Relay_Signal_Construction_Schematic")


def stability_measures():
    fig, ax = page(7.45, "From relay signs to two long-time stability measures")
    panel(ax, .16, 6.87, 6.18, 2.14, "a", "Within-cell reduction: relay signs to a block sign")
    panel(ax, .16, 4.53, 3.84, 2.73, "b", "Connected active union")
    panel(ax, 4.18, 4.53, 2.16, 2.73, "c", "Block summaries")
    panel(ax, .16, 1.60, 6.18, 1.17, "d", "Two distinct questions")
    box(ax, .31, 5.90, 2.71, .52,
        r"$Y_{ng}\in\{-1,0,+1\},\quad n\in b$" + "\n" +
        r"$N_{bg}^{\pm}=\sum_{n\in b}\mathbf{1}(Y_{ng}=\pm1)$", size=10.0)
    arrow(ax, (3.08, 6.16), (3.40, 6.16))
    box(ax, 3.47, 5.84, 2.72, .65,
        r"$p_{bg}=\dfrac{|N_{bg}^{+}-N_{bg}^{-}|}{N_{bg}^{+}+N_{bg}^{-}}$" + "\n" +
        "No support: " + r"$p_{bg}=0$.", edge=GOLD, face=LIGHT_GOLD, size=9.7)
    arrow(ax, (4.83, 5.84), (4.83, 5.70))
    box(ax, .31, 4.91, 5.88, .73,
        r"$Z_{bg}=\operatorname{sgn}(N_{bg}^{+}-N_{bg}^{-})$" + "\n" +
        r"if $N_{bg}^{+}+N_{bg}^{-}\geq n_{\min}$, $p_{bg}\geq2/3$, and cell $g$ passes calibration;" + "\n" +
        r"otherwise $Z_{bg}=0$. Main rule: $n_{\min}=1$.", edge=RED, face="#FBF2F2", size=9.5)

    # Exactly the original five-by-eight illustrative array, now truly vector.
    z = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, -1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, -1, 0, 0]])
    left, top, cell_w, cell_h = .67, 3.96, .365, .265
    for col in range(8):
        text(ax, left + (col + .5) * cell_w, 4.10, rf"$g_{col+1}$", size=9.0, ha="center")
    for row in range(5):
        y = top - (row + 1) * cell_h
        text(ax, .43, y + cell_h / 2, rf"$b_{row+1}$", size=9.0, ha="center")
        for col in range(8):
            value = z[row, col]
            face = BLUE if value > 0 else PURPLE if value < 0 else "white"
            ax.add_patch(Rectangle((left + col * cell_w, y), cell_w, cell_h,
                                   edgecolor=LINE, facecolor=face, lw=.6))
            text(ax, left + (col + .5) * cell_w, y + cell_h / 2,
                 f"{value:+d}" if value else "0", size=9.4, ha="center", color="white" if value else GRAY)
    ax.add_patch(Rectangle((left, top - 5 * cell_h), 6 * cell_w, 5 * cell_h,
                           fill=False, edgecolor=GOLD, linestyle="--", lw=1.25))
    text(ax, 2.07, 2.41, r"$\mathcal{A}=\{g_1,\ldots,g_6\}$; illustrative $\ell_g=d_0/2$", size=9.0, ha="center")
    text(ax, 2.07, 2.16, r"$L_{\rm conn}=6\ell_g=3d_0\geq2d_0$", size=10.0, ha="center", color=GOLD)
    text(ax, 2.07, 1.96, "Cells need not be active simultaneously.", size=8.5, ha="center", color=GRAY)

    # The original M and chi values follow directly from this equal-cell array.
    active = np.any(z != 0, axis=1).astype(int)
    support = np.sum(abs(z), axis=1)
    chi = np.divide(z.sum(axis=1), support, out=np.zeros(5), where=support > 0)
    assert active.tolist() == [1, 1, 1, 0, 1]
    assert chi.tolist() == [1., 1., -1., 0., 0.]
    assert active.mean() == .8
    assert abs(chi[active.astype(bool)].mean()) == .25
    for x, label in [(4.53, r"$b$"), (5.16, r"$M_b$"), (5.85, r"$\chi_b$")]:
        text(ax, x, 4.10, label, size=10.0, ha="center")
    for row in range(5):
        y = top - (row + .5) * cell_h
        ax.add_patch(Rectangle((4.37, y - cell_h / 2), 1.75, cell_h,
                               facecolor=LIGHT_GREEN if active[row] else "#F7F8F8", edgecolor="white", lw=.6))
        text(ax, 4.53, y, str(row + 1), size=9.4, ha="center")
        text(ax, 5.16, y, str(active[row]), size=9.4, ha="center")
        text(ax, 5.85, y, f"{chi[row]:+.1f}" if active[row] else "--", size=9.4, ha="center")
    text(ax, 5.26, 2.40, r"$M_b=\mathbf{1}\{\exists g:|Z_{bg}|=1\}$", size=9.0, ha="center")
    text(ax, 5.26, 2.07, r"$\chi_b=\dfrac{\sum_g\ell_g Z_{bg}}{\sum_g\ell_g|Z_{bg}|}$", size=9.8, ha="center")

    box(ax, .31, .53, 2.83, .79, "", edge=GREEN, face=LIGHT_GREEN)
    box(ax, 3.34, .53, 2.85, .79, "", edge=PURPLE, face=LIGHT_PURPLE)
    text(ax, 1.725, 1.20, "Recurrence", size=9.5, ha="center")
    text(ax, 1.725, .96, r"$\Xi_{\rm Persist}=B^{-1}\sum_bM_b$", size=10.0, ha="center")
    text(ax, 1.725, .66, r"$=4/5=0.80$", size=10.0, ha="center")
    text(ax, 4.765, 1.20, "Mean chirality", size=9.5, ha="center")
    text(ax, 4.765, .94, r"$\Xi_{\rm Sign}=\left|\sum_bM_b\chi_b/\sum_bM_b\right|$", size=9.7, ha="center")
    text(ax, 4.765, .66, r"$=|(1+1-1+0)/4|=0.25$", size=9.8, ha="center")
    text(ax, 3.25, .22,
         "Equal active-block weights; the zero-chirality block remains. Illustrative array, not simulation data.",
         size=8.6, ha="center", color=GRAY)
    save(fig, ax, "Long_Time_Stability_Measures_Schematic")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    projection_and_w()
    relay_signal()
    stability_measures()
    (OUT / "schematic_build_manifest.json").write_text(json.dumps(QA, indent=2) + "\n", encoding="utf-8")
    for item in QA:
        print(item["pdf"])
        print(item["png"])
