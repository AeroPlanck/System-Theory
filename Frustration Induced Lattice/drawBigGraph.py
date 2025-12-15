import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import os
import shutil
import sys
sys.path.append("..")

randomSeed = 10

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)
colors = ["#5657A4", "#95D3A2", "#FFFFBF", "#F79051", "#A30644"]
cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
cmap_r = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors[::-1])

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns

sns.set_theme(
    style="ticks", 
    font_scale=1.1, rc={
    'figure.figsize': (6, 5),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#000000',
    'figure.titleweight': "bold",
    'xtick.color': '#000000',
    'ytick.color': '#000000'
})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

from main import *
from multiprocessing import Pool

SAVE_PATH = r"F:\MS_ExperimentData\general"


phaseLags = np.linspace(-1, 1, 21) * np.pi
# phaseLags = np.linspace(0, 1, 11) * np.pi
# phaseLags = [0.75 * np.pi]
# phaseLags = [0 * np.pi]
omegaMins = [0] # np.linspace(1e-5, 3, 21)
# randomSeeds = range(10)
randomSeeds = [10]
# strengthKs = np.linspace(4, 20, 7)  # [20]  # 
strengthKs = np.linspace(1e-5, 25, 5)
# strengthKs = [20]
distanceD0s = np.linspace(0.1, 7, 5)  #  np.linspace(0.1, 3, 7)  # [1]
# distanceD0s = [1]
deltaOmegas =  [0, 3]  # [1.0]

models = [
    CollisionBoundaryPatternFormation(
        strengthK=strengthK, distanceD0=distanceD0, phaseLagA0=phaseLag,
        freqDist="uniform", 
        omegaMin=omegaMin, deltaOmega=deltaOmega, 
        agentsNum=2000, dt=0.005,
        tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
        randomSeed=randomSeed, overWrite=False
    )
    for strengthK in strengthKs
    for distanceD0 in distanceD0s
    for omegaMin in omegaMins
    for deltaOmega in deltaOmegas
    for phaseLag in phaseLags
]

def _ensure_sa(model):
    return StateAnalysis(model)

sas = [_ensure_sa(model) for model in tqdm(models)]

sa_map = {}
for sa in sas:
    key = (sa.model.strengthK, sa.model.distanceD0, sa.model.deltaOmega, sa.model.omegaMin, sa.model.phaseLagA0)
    sa_map[key] = sa

for strengthK in strengthKs:
    for distanceD0 in distanceD0s:
        for deltaOmega in deltaOmegas:
            fig, axs = plt.subplots(
                len(strengthKs), len(distanceD0s), 
                figsize=(len(strengthKs) * 4, len(distanceD0s) * 4),
                squeeze=False
            )
            
            # To capture a representative model for filename generation
            rep_sa = None

            for i, omegaMin in enumerate(omegaMins):
                for j, phaseLag in enumerate(phaseLags):
                    key = (strengthK, distanceD0, deltaOmega, omegaMin, phaseLag)
                    if key in sa_map:
                        sa = sa_map[key]
                        rep_sa = sa
                        ax = axs[i, j]
                        
                        colors = ["red"] * (sa.model.freqOmega < 0).sum() + ["#414CC7"] * (sa.model.freqOmega > 0).sum()
                        index = -1
                        sa.plot_spatial(ax, colorsBy="phase", index=index)
                        
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(
                            rf"$\Alpha={(sa.model.phaseLagA0/np.pi):.2f}\pi,"
                            rf"\ \Omega_{{\min}}={sa.model.omegaMin:.2f}$", 
                            fontsize=16, loc="left"
                        )
                        ax.set_aspect("equal")

            plt.tight_layout()
            
            if rep_sa:
                os.makedirs("figs", exist_ok=True)
                plt.savefig(
                    f"figs/{rep_sa.model.__class__.__name__}_"
                    f"K{strengthK:.2f}_D{distanceD0:.2f}_"
                    f"Alpha{phaseLags[0]:.2f}_Del{deltaOmega:.2f}"
                    f"{'initPhaseTheta,' if rep_sa.model.initPhaseTheta is not None else ''}"
                    f"_N{rep_sa.model.agentsNum}_Dist{rep_sa.model.freqDist}.pdf", 
                    bbox_inches="tight"
                )
            plt.close()