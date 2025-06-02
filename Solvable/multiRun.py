
import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import os
import shutil

def run_model(model):
        model.run(6000)

if __name__ == "__main__":

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
    )

    @nb.njit
    def colors_idx(phaseTheta):
        return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

    from main import *
    from multiprocessing import Pool

    # Ks = np.linspace(0.1, 15, 40).round(2)
    # Ks = np.linspace(0.1, 15, 40).round(2)
    # Ks = np.linspace(0, 20, 41).round(2)
    Ks = np.linspace(20, 40, 41).round(2)

    SAVE_PATH = "/home/thanmark/MS_DATA"  # r"E:\MS_ExperimentData\general"

    models = [
        ChiralSolvable2D(agentsNum=3000, dt=0.01, K=K, distribution="cauchy", 
                         randomSeed=0,
                         savePath=SAVE_PATH, tqdm=True, overWrite=True)
        for K in Ks
    ]

    with Pool(41) as p:

        p.map(
            run_model,
            tqdm(models, desc="run models", total=len(models))
        )