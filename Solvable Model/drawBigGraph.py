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

randomSeed = 100

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

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

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

from main import *
from multiprocessing import Pool
import pandas as pd

# Ks = np.linspace(1, 5, 11).round(2)
# Ks = np.linspace(0.1, 1.3, 13).round(2)
# Ks = np.linspace(0.1, 15, 40).round(2) * -1
# Ks = np.linspace(0, 20, 41).round(2)
Ks = np.linspace(20, 40, 41).round(2)
Js = [0]

fig = plt.figure(figsize=(len(Js) * 3, len(Ks) * 3))

idx = 1

SAVE_PATH = "/home/thanmark/MS_DATA"  # r"E:\MS_ExperimentData\general"

for _, K in tqdm(product(Js[::-1], Ks), total=len(Ks) * len(Js)):
    model = ChiralSolvable2D(agentsNum=3000, dt=0.01, K=K, distribution="cauchy", savePath=SAVE_PATH, tqdm=True, overWrite=False)

    class1 = model.omegaValue > 0
    class2 = model.omegaValue < 0

    ax = plt.subplot(len(Ks), len(Js), idx)
    idx += 1

    try:
        sa = StateAnalysis(model)
    except:
        print(f"Error in {model.K}")
        continue
    # print(model.K, sa.TNum)

    positionX, phaseTheta = sa.get_state(index=-1)
    # model.positionX = positionX
    # model.phaseTheta = phaseTheta

    ax.quiver(
        positionX[class1, 0], positionX[class1, 1],
        np.cos(phaseTheta[class1]), np.sin(phaseTheta[class1]), color='red', alpha=0.8
    )
    ax.quiver(
        positionX[class2, 0], positionX[class2, 1],
        np.cos(phaseTheta[class2]), np.sin(phaseTheta[class2]), color='blue', alpha=0.8
    )

    # sa.plot_last_state(ax=ax, withColorBar=False)
    ax.set_title(rf"$K={model.K:.2f}$")

plt.tight_layout()
plt.savefig(f"bigGraph.png", bbox_inches="tight")
plt.close()