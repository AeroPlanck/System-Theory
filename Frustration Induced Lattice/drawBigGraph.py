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

# Use indices for mapping to avoid floating point comparison issues
model_map = {}
# The order must match the list comprehension for 'models' above:
# strengthKs, distanceD0s, omegaMins, deltaOmegas, phaseLags
param_indices = product(
    range(len(strengthKs)),
    range(len(distanceD0s)),
    range(len(omegaMins)),
    range(len(deltaOmegas)),
    range(len(phaseLags))
)

for indices, model in zip(param_indices, models):
    # indices: (i_K, i_D0, i_omegaMin, i_dOmega, i_phaseLag)
    model_map[indices] = model

# Use product to create a single iterable for the outer loops to use with tqdm
# We use indices here too
outer_loops_indices = list(product(
    range(len(strengthKs)), 
    range(len(distanceD0s)), 
    range(len(deltaOmegas))
))

for i_K, i_D0, i_dOmega in tqdm(outer_loops_indices, desc="Generating Plots"):
    strengthK = strengthKs[i_K]
    distanceD0 = distanceD0s[i_D0]
    deltaOmega = deltaOmegas[i_dOmega]

    # Adjusted subplot dimensions to match the inner loops (omegaMin vs phaseLag)
    fig, axs = plt.subplots(
        len(omegaMins), len(phaseLags), 
        figsize=(len(phaseLags) * 4, len(omegaMins) * 4),
        squeeze=False
    )
    
    # To capture a representative model for filename generation
    rep_model = None

    for i_omegaMin, omegaMin in enumerate(omegaMins):
        for i_phaseLag, phaseLag in enumerate(phaseLags):
            # Construct key in the same order as 'models' creation:
            # K, D0, omegaMin, dOmega, phaseLag
            key = (i_K, i_D0, i_omegaMin, i_dOmega, i_phaseLag)
            
            if key in model_map:
                model = model_map[key]
                rep_model = model
                
                ax = axs[i_omegaMin, i_phaseLag]
                
                # Simplified plotting
                sa = StateAnalysis(model)
                sa.plot_spatial(ax, colorsBy="phase", index=-1)
                del sa

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    rf"$\alpha={(model.phaseLagA0/np.pi):.2f}\pi,"
                    rf"\ \Omega_{{\min}}={model.omegaMin:.2f}$", 
                    fontsize=16, loc="left"
                )
                ax.set_aspect("equal")

    if rep_model:
        os.makedirs("figs", exist_ok=True)
        filename_base = (
            f"figs/{rep_model.__class__.__name__}_"
            f"K{strengthK:.2f}_D{distanceD0:.2f}_"
            f"Alpha{phaseLags[0]:.2f}_Del{deltaOmega:.2f}"
            f"{'initPhaseTheta,' if rep_model.initPhaseTheta is not None else ''}"
            f"_N{rep_model.agentsNum}_Dist{rep_model.freqDist}"
        )
        print(f"Saving to {filename_base}.pdf")
        plt.savefig(filename_base + ".pdf", bbox_inches="tight")
    
    plt.close(fig)