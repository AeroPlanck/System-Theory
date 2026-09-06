"""Generate vector schematics for the boundary-flow definitions in Methods.

The figures contain illustrative geometry and discrete arrays only.  They do
not read HDF5 files and do not display simulation measurements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle


OUTPUT_DIR = Path("output/Phase_Informed_Boundary_Flow_Refined")
FIGURE_DPI = 320

BLUE = "#2A6F85"
GREEN = "#3E7C4B"
RED = "#A33D3D"
PURPLE = "#76519A"
GOLD = "#9C6B20"
GRAY = "#62676D"
LIGHT_GRAY = "#E8EAED"
INK = "#202124"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.5,
            "axes.titlesize": 12.2,
            "axes.labelsize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


def vector(
    ax: plt.Axes,
    start: tuple[float, float],
    delta: tuple[float, float],
    color: str,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    width: float = 1.8,
) -> None:
    end = (start[0] + delta[0], start[1] + delta[1])
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        ax.text(
            end[0] + label_offset[0],
            end[1] + label_offset[1],
            label,
            color=color,
            ha="center",
            va="center",
        )


def formula_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    width: float,
    height: float,
    edgecolor: str = GRAY,
    facecolor: str = "white",
    fontsize: float = 10,
) -> None:
    x, y = xy
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
            zorder=1,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=2,
    )


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return pdf_path, png_path


def draw_projection_and_w() -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    axes = axes.ravel()
    fig.suptitle("Why $W_i^n$ Is Needed and How It Is Constructed", y=0.985, fontsize=15)

    # (a) Distance alone cannot distinguish boundary support from a nearby vortex.
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.set_title("Distance alone creates a false boundary signal")
    radius, band = 2.15, 0.55
    center = np.array([0.0, 0.0])
    ax.add_patch(Circle(center, radius, fill=False, color=INK, linewidth=2.2))
    ax.add_patch(Circle(center, radius - band, fill=False, color=GRAY, linewidth=1.5, linestyle="--"))
    ax.text(0.0, 2.31, "physical boundary", ha="center", color=INK)
    ax.text(-0.72, 1.12, r"distance band $d\leq d_0$", ha="center", color=GRAY, fontsize=9.4)

    wall_phi = np.deg2rad(28.0)
    wall_particle = (radius - 0.20) * np.array([np.cos(wall_phi), np.sin(wall_phi)])
    wall_tangent = np.array([-np.sin(wall_phi), np.cos(wall_phi)])
    ax.scatter(*wall_particle, s=72, color=RED, edgecolor="white", linewidth=1.2, zorder=4)
    vector(ax, tuple(wall_particle), tuple(0.72 * wall_tangent), RED, width=1.8)
    ax.text(2.22, 1.82, "wall-dependent motion", color=RED, ha="center", fontsize=9.2)
    ax.text(2.22, 1.55, r"$d_i^n\leq d_0$", color=RED, ha="center", fontsize=9.2)

    vortex_center = np.array([-1.12, 0.12])
    vortex_particle = np.array([-1.72, 0.12])
    ax.add_patch(Arc(vortex_center, 1.18, 1.18, theta1=20, theta2=325, color=PURPLE, linewidth=2.0))
    vector(ax, (-0.61, -0.15), (-0.15, -0.28), PURPLE, width=1.6)
    ax.scatter(*vortex_particle, s=72, color=PURPLE, edgecolor="white", linewidth=1.2, zorder=4)
    ax.scatter(*vortex_center, s=20, color=PURPLE, zorder=4)
    ax.text(-1.08, -0.82, "interior vortex", color=PURPLE, ha="center")
    ax.text(-1.08, -1.08, r"passes near the wall: also $d_i^n\leq d_0$", color=PURPLE, ha="center", fontsize=9.2)
    formula_box(
        ax,
        (-2.62, -2.28),
        r"Distance tells where the particle is." "\n"
        r"It does not tell whether the wall supplies the turn.",
        5.24,
        0.55,
        edgecolor=GOLD,
        facecolor="#FBF7EE",
        fontsize=9.5,
    )
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.55, 2.65)
    ax.set_aspect("equal")
    ax.axis("off")

    # (b) Local geometry and tangential motion.
    ax = axes[1]
    panel_label(ax, "(b)")
    ax.set_title("Projection turns a position into local path data")
    local_radius, distance = 1.80, 0.48
    phi = np.deg2rad(34.0)
    boundary_point = local_radius * np.array([np.cos(phi), np.sin(phi)])
    particle = (local_radius - distance) * np.array([np.cos(phi), np.sin(phi)])
    tangent = np.array([-np.sin(phi), np.cos(phi)])
    inward = -np.array([np.cos(phi), np.sin(phi)])
    ax.add_patch(Circle(center, local_radius, fill=False, color=INK, linewidth=2.0))
    ax.add_patch(Circle(center, local_radius - distance, fill=False, color=BLUE, linewidth=1.7, linestyle="--"))
    ax.scatter(*boundary_point, color=INK, s=24, zorder=4)
    ax.scatter(*particle, color=RED, s=58, edgecolor="white", linewidth=1.0, zorder=4)
    ax.plot([boundary_point[0], particle[0]], [boundary_point[1], particle[1]], color=GOLD, linewidth=2)
    ax.text(*(particle + np.array([-0.30, -0.20])), r"$\mathbf{X}_i^n$", color=RED)
    ax.text(*(boundary_point + np.array([0.17, -0.24])), r"$\Pi(\mathbf{X}_i^n)$")
    ax.text(*((boundary_point + particle) / 2 + 0.07 * tangent), r"$d_i^n$", color=GOLD)
    vector(ax, tuple(particle), tuple(0.70 * tangent), BLUE, r"$\mathbf{t}_i^n$", (0.07, 0.03), 1.8)
    vector(ax, tuple(particle), tuple(0.56 * inward), GREEN, r"$\mathbf{n}_i^n$", (0.05, 0.02), 1.8)
    heading = 0.72 * (0.90 * tangent + 0.44 * inward)
    vector(ax, tuple(particle), tuple(heading), RED, r"$\mathbf{u}_i^n$", (0.08, 0.05), 1.8)
    formula_box(
        ax,
        (-2.35, -2.52),
        r"$\Pi(\mathbf{X}_i^n)=(d_i^n,s_i^n,\mathbf{t}_i^n,\kappa_{b,i}^n,e_i^n)$" "\n"
        r"$q_i^n=\mathbf{u}_i^n\!\cdot\!\mathbf{t}_i^n$, $v_{\parallel}=vq_i^n$; "
        r"$\kappa_{d,i}^n=\kappa_{b,i}^n/(1-\kappa_{b,i}^n d_i^n)$",
        5.00,
        0.72,
        edgecolor=BLUE,
        facecolor="#F1F7F9",
        fontsize=9.2,
    )
    ax.text(0.15, -1.50, r"$q>0$: counterclockwise; $q<0$: clockwise", ha="center", color=GRAY)
    ax.set_xlim(-2.55, 2.85)
    ax.set_ylim(-2.65, 2.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # (c) Derivation of the signed turning deficit.
    ax = axes[2]
    panel_label(ax, "(c)")
    ax.set_title("Required turn minus the turn supplied by free dynamics")
    formula_box(
        ax,
        (0.03, 0.74),
        r"Path geometry" "\n"
        r"$q_i^n,\ \kappa_{d,i}^n,\ v$",
        0.44,
        0.16,
        edgecolor=BLUE,
        facecolor="#F1F7F9",
        fontsize=10.2,
    )
    formula_box(
        ax,
        (0.53, 0.69),
        r"Free heading dynamics" "\n"
        r"$\mathcal{N}_i^n=\{j\ne i:\|\mathbf{X}_j^n-\mathbf{X}_i^n\|\leq d_0\}$" "\n"
        r"$\Omega_i^n=\omega_i+K[$" "\n"
        r"$|\mathcal{N}_i^n|^{-1}\!\sum_{j\in\mathcal{N}_i^n}\sin(\theta_j^n-\theta_i^n+\alpha)-\sin\alpha]$",
        0.44,
        0.27,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=7.5,
    )
    vector(ax, (0.25, 0.725), (0.0, -0.075), BLUE, width=1.4)
    vector(ax, (0.75, 0.685), (0.0, -0.075), GREEN, width=1.4)
    formula_box(
        ax,
        (0.03, 0.43),
        r"Curve-following requirement" "\n"
        r"$a_{\rm req,\perp}=\kappa_{d,i}^n(vq_i^n)^2$",
        0.44,
        0.18,
        edgecolor=BLUE,
        facecolor="#F1F7F9",
        fontsize=10.0,
    )
    formula_box(
        ax,
        (0.53, 0.39),
        r"Normal part of free turning" "\n"
        r"$\dot{\mathbf{v}}_{\rm free}=v\Omega_i^n\mathbf{J}\mathbf{u}_i^n$" "\n"
        r"$a_{\rm free,\perp}=\mathbf{n}_i^n\!\cdot\!\dot{\mathbf{v}}_{\rm free}=v\Omega_i^nq_i^n$",
        0.44,
        0.22,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=8.9,
    )
    vector(ax, (0.25, 0.415), (0.20, -0.105), GRAY, width=1.4)
    vector(ax, (0.75, 0.375), (-0.20, -0.065), GRAY, width=1.4)
    formula_box(
        ax,
        (0.13, 0.11),
        r"Missing turn that must be supplied by the wall" "\n"
        r"$W_i^n=\dfrac{a_{\rm req,\perp}-a_{\rm free,\perp}}{v}$"
        r"$=\kappa_{d,i}^n v(q_i^n)^2-\Omega_i^nq_i^n$",
        0.74,
        0.22,
        edgecolor=RED,
        facecolor="#FBF2F2",
        fontsize=10.3,
    )
    ax.text(0.50, 0.015, r"$q^2$ makes the required turn direction-independent; $\Omega q$ keeps the signed chirality.", ha="center", color=GRAY, fontsize=9.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # (d) W is a gate, never a transport weight.
    ax = axes[3]
    panel_label(ax, "(d)")
    ax.set_title("$W$ decides membership, not flow strength")
    formula_box(
        ax,
        (0.08, 0.77),
        r"Near-wall candidate: $d_i^n\leq d_0$" "\n"
        r"valid projection: $1-\kappa_{b,i}^n d_i^n>0$",
        0.84,
        0.16,
        edgecolor=GRAY,
        facecolor="#F7F7F8",
        fontsize=10.0,
    )
    vector(ax, (0.50, 0.755), (0.0, -0.07), GRAY, width=1.4)
    formula_box(
        ax,
        (0.05, 0.47),
        r"$W_i^n>\varepsilon_W=0.02v/d_0$" "\n"
        r"free turning is insufficient" "\n"
        r"$\Rightarrow$ retain as wall-supported",
        0.43,
        0.20,
        edgecolor=RED,
        facecolor="#FBF2F2",
        fontsize=9.5,
    )
    formula_box(
        ax,
        (0.52, 0.47),
        r"$W_i^n\leq\varepsilon_W$" "\n"
        r"free dynamics already explains the turn" "\n"
        r"$\Rightarrow$ do not retain through $W$",
        0.43,
        0.20,
        edgecolor=PURPLE,
        facecolor="#F6F1F9",
        fontsize=9.2,
    )
    vector(ax, (0.265, 0.455), (0.235, -0.095), RED, width=1.4)
    formula_box(
        ax,
        (0.13, 0.17),
        r"Retained particle enters $\mathcal{S}_{ng}\rightarrow r_{ng}\rightarrow Q_{ng}$" "\n"
        r"The value of $W_i^n$ is then discarded: it never weights $Q$, $Y$, $Z$, or $\Xi$.",
        0.74,
        0.18,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=9.4,
    )
    ax.text(0.50, 0.055, r"Current rule: apply $W$ for high-$\alpha$ Pattern states; the exact $\alpha=\pi/2$ analysis omits it.", ha="center", color=GRAY, fontsize=8.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5,
        0.012,
        "Logic: distance locates a candidate; W tests wall dependence; later statistics test motion, relay, extent, and time stability.",
        ha="center",
        color=GRAY,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.078, top=0.91, wspace=0.16, hspace=0.25)
    return save_figure(fig, "Boundary_Projection_And_W_Filter_Schematic")


def draw_relay_signal() -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.1))
    axes = axes.ravel()
    fig.suptitle("From Wall-Supported Candidates to a Relay Signal", y=0.985, fontsize=15)

    # (a) Nearest representative in a boundary cell.
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.set_title("Stage 1 - assign one sign to each cell and frame")
    for g in range(5):
        ax.add_patch(Rectangle((g, 0), 1, 0.25, facecolor=LIGHT_GRAY, edgecolor="white"))
        ax.text(g + 0.5, -0.16, rf"$I_{{{g+1}}}$", ha="center")
    particles = [(2.23, 0.62, "A", +1), (2.66, 1.08, "B", -1), (2.48, 1.52, "C", +1)]
    for x, y, name, sign in particles:
        color = BLUE if sign > 0 else PURPLE
        ax.scatter(x, y, s=85, facecolor=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(x + 0.12, y + 0.02, name, color=color, fontweight="bold")
        vector(ax, (x - 0.16, y), (0.30 * sign, 0.0), color, width=1.4)
        ax.plot([x, x], [0.25, y - 0.08], color=GRAY, linestyle=":", linewidth=1)
    ax.text(2.5, 1.88, r"retained set $\mathcal{S}_{ng}=\{A,B,C\}$", ha="center")
    ax.text(2.5, 1.64, r"each member already passed the applicable geometry/$W$ selection", ha="center", color=GRAY, fontsize=9.2)
    ax.text(2.5, -0.58, r"$r_{ng}=A$ (smallest $d_i^n$)", ha="center", color=RED)
    ax.text(2.5, -0.92, r"$Q_{ng}=\mathrm{sgn}\,q_A^n=+1$", ha="center", color=BLUE)
    ax.text(2.5, -1.20, r"$\ell_g\leq d_0/2$, $\sum_g\ell_g=P_{\rm reg}$", ha="center", color=GRAY)
    ax.set_xlim(-0.15, 5.15)
    ax.set_ylim(-1.42, 2.2)
    ax.axis("off")

    # (c) Two-frame relay signal.
    ax = axes[2]
    panel_label(ax, "(c)")
    ax.set_title("Stage 3 - preserve the sign when the carrier changes")
    rows = [(1.48, r"frame $n$", "A", +1), (0.64, r"frame $n+1$", "B", +1)]
    for y, frame_label, particle_name, sign in rows:
        ax.add_patch(Rectangle((0.3, y - 0.18), 3.2, 0.36, facecolor="#F6F7F8", edgecolor=GRAY))
        ax.text(0.02, y, frame_label, ha="left", va="center")
        ax.scatter(1.72, y, s=95, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(1.88, y + 0.02, f"particle {particle_name}", va="center", color=BLUE)
        vector(ax, (1.40, y), (0.27 * sign, 0.0), BLUE, width=1.5)
        ax.text(3.22, y, r"$Q=+1$", ha="right", va="center", color=BLUE)
    vector(ax, (1.72, 1.25), (0.0, -0.34), GREEN, width=1.6)
    ax.text(2.02, 1.05, "carrier changes", color=GREEN, va="center")
    formula_box(
        ax,
        (0.25, -0.33),
        r"$Q_{ng}=Q_{n+1,g}=+1$ although $A\neq B$" "\n" r"$\Longrightarrow\;Y_{ng}=+1$",
        3.30,
        0.48,
        edgecolor=BLUE,
        facecolor="#F1F7F9",
        fontsize=11,
    )
    ax.text(1.90, -0.62, "The signal persists; no single particle must complete a circuit.", ha="center", color=GRAY)
    ax.set_xlim(-0.05, 3.85)
    ax.set_ylim(-0.82, 1.92)
    ax.axis("off")

    # (b) Direction check from measured displacement.
    ax = axes[1]
    panel_label(ax, "(b)")
    ax.set_title("Stage 2 - check heading against measured motion")
    x = np.linspace(0.35, 3.75, 200)
    y = 0.34 + 0.10 * np.sin(1.1 * x)
    ax.plot(x, y, color=INK, linewidth=2)
    x0, x1 = 1.15, 2.72
    y0 = 0.34 + 0.10 * np.sin(1.1 * x0)
    y1 = 0.34 + 0.10 * np.sin(1.1 * x1)
    ax.scatter([x0, x1], [y0, y1], s=75, color=[PURPLE, BLUE], zorder=3)
    ax.text(x0, y0 + 0.28, r"$s_i^n$", ha="center")
    ax.text(x1, y1 + 0.28, r"$s_i^{n+1}$", ha="center")
    vector(ax, (x0 + 0.12, y0 + 0.03), (x1 - x0 - 0.24, y1 - y0), BLUE, width=1.8)
    ax.text((x0 + x1) / 2, max(y0, y1) + 0.38, r"$\Delta s_i^n>0$", ha="center", color=BLUE)
    formula_box(
        ax,
        (0.20, -0.36),
        r"Trial $T_{ng}=1$: same particle and segment; $|\Delta s_i^n|\leq J$" "\n"
        r"$J=1.05v\Delta t_s/(1-\kappa_b d_{\max})$",
        3.66,
        0.44,
        edgecolor=BLUE,
        facecolor="#F1F7F9",
        fontsize=9.7,
    )
    formula_box(
        ax,
        (0.20, -0.92),
        r"$E_{ng}=T_{ng}\,\mathbf{1}[|\Delta s_i^n|/(v\Delta t_s)\geq0.02]$" "\n"
        r"$\times\mathbf{1}[\mathrm{sgn}(\Delta s_i^n)=Q_{ng}]$",
        3.66,
        0.44,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=9.7,
    )
    ax.text(
        2.03,
        -1.18,
        r"Cell accepted when $\sum_nT_{ng}\geq3$ and "
        r"$C_g=\sum_nE_{ng}/\sum_nT_{ng}\geq2/3$",
        ha="center",
        color=GRAY,
    )
    ax.set_xlim(0, 4.05)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    # (d) Saved frames grouped into terminal time blocks.
    ax = axes[3]
    panel_label(ax, "(d)")
    ax.set_title("Stage 4 - define the terminal window and time blocks")
    formula_box(
        ax,
        (0.05, 0.76),
        r"Inputs: $F$, $\Delta t_s$, $d_0$, $v$, $P$;  $T_{\rm req}=10P/v$",
        0.90,
        0.12,
        edgecolor=GRAY,
        facecolor="#F7F7F8",
        fontsize=10.5,
    )
    vector(ax, (0.50, 0.745), (0.0, -0.055), GRAY, width=1.4)
    formula_box(
        ax,
        (0.05, 0.53),
        r"$n_{\rm blk}=\lceil(d_0/v)/\Delta t_s\rceil$, "
        r"$\Delta T_b=n_{\rm blk}\Delta t_s$",
        0.90,
        0.14,
        edgecolor=BLUE,
        facecolor="#F1F7F9",
        fontsize=10.5,
    )
    vector(ax, (0.50, 0.515), (0.0, -0.055), GRAY, width=1.4)
    formula_box(
        ax,
        (0.05, 0.27),
        r"$N_{\rm avail}=\min(F-1,\lceil T_{\rm req}/\Delta t_s\rceil)$" "\n"
        r"$B=\lfloor N_{\rm avail}/n_{\rm blk}\rfloor$, $n_0=F-1-Bn_{\rm blk}$",
        0.90,
        0.18,
        edgecolor=PURPLE,
        facecolor="#F6F1F9",
        fontsize=9.9,
    )
    vector(ax, (0.50, 0.255), (0.0, -0.055), GRAY, width=1.4)
    formula_box(
        ax,
        (0.05, 0.03),
        r"Use frames $n_0,\ldots,F-1$; $f_T=Bn_{\rm blk}\Delta t_s/T_{\rm req}$",
        0.90,
        0.14,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=10.3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5,
        0.012,
        "Only the displacement calibration tracks one particle; the relay signal may pass from one particle to another.",
        ha="center",
        color=GRAY,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.085, top=0.91, wspace=0.18, hspace=0.25)
    return save_figure(fig, "Relay_Signal_Construction_Schematic")


def draw_stability_measures() -> tuple[Path, Path]:
    fig = plt.figure(figsize=(11.2, 8.2))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.72, 1.55],
        width_ratios=[1.30, 0.98, 1.20],
        hspace=0.28,
        wspace=0.28,
    )
    fig.suptitle("From Relay Signals to Two Long-Time Stability Measures", y=0.985, fontsize=15)

    # Illustrative block-by-cell array.  Rows are blocks, columns are cells.
    z = np.array(
        [
            [+1, +1, +1, 0, 0, 0, 0, 0],
            [0, +1, +1, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, +1, -1, 0, 0],
        ],
        dtype=int,
    )
    cmap = ListedColormap([PURPLE, "white", BLUE])

    # (a) Within-cell, within-block reduction from Y to Z.
    ax = fig.add_subplot(grid[0, :])
    panel_label(ax, "(a)")
    ax.set_title("Stage 5 - reduce the relay signs in cell $g$ and block $b$ to one gated sign")
    formula_box(
        ax,
        (0.05, 0.57),
        r"Input from consecutive cell signs" "\n"
        r"$Y_{ng}\in\{-1,0,+1\}$ for $n\in b$",
        0.36,
        0.27,
        edgecolor=BLUE,
        facecolor="#F1F7F9",
        fontsize=10.5,
    )
    vector(ax, (0.42, 0.705), (0.105, 0.0), GRAY, width=1.5)
    formula_box(
        ax,
        (0.54, 0.55),
        r"$N_{bg}^{+}=\sum_{n\in b}\mathbf{1}(Y_{ng}=+1)$" "\n"
        r"$N_{bg}^{-}=\sum_{n\in b}\mathbf{1}(Y_{ng}=-1)$",
        0.41,
        0.31,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=9.6,
    )
    vector(ax, (0.745, 0.555), (0.0, -0.105), GRAY, width=1.5)
    formula_box(
        ax,
        (0.54, 0.075),
        r"$p_{bg}=\dfrac{|N_{bg}^{+}-N_{bg}^{-}|}{N_{bg}^{+}+N_{bg}^{-}}$" "\n"
        r"$p_{bg}=0$ when $N_{bg}^{+}+N_{bg}^{-}=0$",
        0.41,
        0.32,
        edgecolor=GOLD,
        facecolor="#FBF7EE",
        fontsize=9.5,
    )
    vector(ax, (0.525, 0.245), (-0.105, 0.0), GRAY, width=1.5)
    formula_box(
        ax,
        (0.05, 0.075),
        r"$Z_{bg}=\mathrm{sgn}(N_{bg}^{+}-N_{bg}^{-})$ if" "\n"
        r"$N_{bg}^{+}+N_{bg}^{-}\geq n_{\min}$, $p_{bg}\geq2/3$," "\n"
        r"and displacement calibration accepts cell $g$; otherwise $Z_{bg}=0$",
        0.36,
        0.34,
        edgecolor=RED,
        facecolor="#FBF2F2",
        fontsize=9.3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "(b)")
    ax.set_title(r"Stage 6 - require a connected active length $\geq2d_0$", fontsize=10)
    ax.imshow(z, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="none")
    ax.add_patch(
        Rectangle(
            (-0.48, -0.48),
            5.96,
            4.96,
            fill=False,
            edgecolor=GOLD,
            linewidth=1.5,
            linestyle="--",
        )
    )
    ax.set_xticks(np.arange(z.shape[1]), [rf"$g_{j+1}$" for j in range(z.shape[1])])
    ax.set_yticks(np.arange(z.shape[0]), [rf"$b_{j+1}$" for j in range(z.shape[0])])
    ax.set_xlabel("boundary cell $g$", labelpad=4)
    ax.set_ylabel("time block $b$")
    ax.set_xticks(np.arange(-0.5, z.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, z.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#B9BDC2", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(z.shape[0]):
        for col in range(z.shape[1]):
            value = z[row, col]
            ax.text(col, row, f"{value:+d}" if value else "0", ha="center", va="center", color="white" if value else GRAY)
    ax.text(0.5, -0.19, r"Blue: $+1$; purple: $-1$; white: $0$", transform=ax.transAxes, ha="center", color=GRAY)

    ax = fig.add_subplot(grid[1, 1])
    ax.text(-0.02, 0.98, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=13, fontweight="bold")
    ax.set_title("Stage 7 - summarize each block")
    m = np.array([1, 1, 1, 0, 1])
    chi = [1.0, 1.0, -1.0, np.nan, 0.0]
    y_positions = np.array([4.05, 3.10, 2.15, 1.20, 0.25])
    for row, y in enumerate(y_positions):
        active_color = GREEN if m[row] else LIGHT_GRAY
        ax.add_patch(Rectangle((0.03, y - 0.31), 0.94, 0.62, facecolor=active_color, alpha=0.20, edgecolor=active_color))
        chi_text = "--" if np.isnan(chi[row]) else f"{chi[row]:+.1f}"
        ax.text(0.09, y, rf"$b_{row+1}$", va="center", fontsize=9.5)
        ax.text(0.46, y, rf"$M_b={m[row]}$", ha="center", va="center", fontsize=9.5)
        ax.text(0.93, y, rf"$\chi_b={chi_text}$", ha="right", va="center", fontsize=9.5)
    formula_box(
        ax,
        (0.04, -0.71),
        r"$M_b=\mathbf{1}\{\exists g:\ |Z_{bg}|=1\}$",
        0.92,
        0.46,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=10.5,
    )
    formula_box(
        ax,
        (0.04, -1.78),
        r"$\chi_b=\dfrac{\sum_g\ell_g Z_{bg}}{\sum_g\ell_g|Z_{bg}|}$",
        0.92,
        0.94,
        edgecolor=PURPLE,
        facecolor="#F6F1F9",
        fontsize=10.0,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-2.02, 4.72)
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 2])
    panel_label(ax, "(d)")
    ax.set_title("Stage 8 - answer two different questions")
    formula_box(
        ax,
        (0.05, 0.56),
        r"Recurrence question:" "\n"
        r"Does a directional signal keep returning?" "\n"
        r"$\Xi_{\rm Persist}=B^{-1}\sum_{b=1}^{B}M_b$" "\n"
        r"$=(1+1+1+0+1)/5=0.80$" "\n"
        r"No carrier-fraction multiplier.",
        0.90,
        0.32,
        edgecolor=GREEN,
        facecolor="#F2F8F2",
        fontsize=8.8,
    )
    formula_box(
        ax,
        (0.05, 0.17),
        r"Chirality question:" "\n"
        r"Does the mean boundary sign stay fixed?" "\n"
        r"$\Xi_{\rm Sign}=\left|\dfrac{\sum_b M_b\chi_b}{\sum_b M_b}\right|$" "\n"
        r"$=|(1+1-1+0)/4|=0.25$" "\n"
        r"Every active block has equal weight.",
        0.90,
        0.32,
        edgecolor=PURPLE,
        facecolor="#F6F1F9",
        fontsize=8.6,
    )
    ax.text(
        0.50,
        0.075,
        "High recurrence does not by itself imply a fixed chirality.",
        ha="center",
        va="center",
        color=INK,
        fontsize=9.5,
    )
    ax.text(
        0.50,
        0.008,
        "Illustrative equal cell lengths; not simulation data.",
        ha="center",
        va="bottom",
        color=GRAY,
        fontsize=9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5,
        0.012,
        r"$W_i^n$ is absent from these sums: it selected candidates earlier and is never used as a numerical weight.",
        ha="center",
        color=GRAY,
        fontsize=9.6,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.92, wspace=0.28, hspace=0.30)
    return save_figure(fig, "Long_Time_Stability_Measures_Schematic")


def main() -> None:
    configure_style()
    outputs = [draw_projection_and_w(), draw_relay_signal(), draw_stability_measures()]
    for pdf_path, png_path in outputs:
        print(pdf_path)
        print(png_path)


if __name__ == "__main__":
    main()
