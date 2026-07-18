import matplotlib as mpl
mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import subprocess
import gc
import imageio
import os
import shutil
import sys
sys.path.append("..")

from main import *

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

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

import pandas as pd
from multiprocessing import Pool

SAVE_PATH = r"D:\PythonProject\System Theory\Frustration Induced Lattice\data"
MP4_PATH = r"D:\PythonProject\System Theory\Frustration Induced Lattice\mp4"
MP4_TEMP_PATH = r"D:\PythonProject\System Theory\Frustration Induced Lattice\mp4_temp"
BATCH_SIZE = 200
NUM_WORKERS = max(1, (os.cpu_count() or 1) // 2)


def _segment_frames(df: pd.DataFrame) -> list:
    idx = df.index.to_numpy()
    if idx.size == 0:
        return []
    reset = np.where(idx[1:] <= idx[:-1])[0] + 1
    bounds = np.concatenate([[0], reset, [len(idx)]])
    return [df.iloc[bounds[i]:bounds[i + 1]].values for i in range(len(bounds) - 1)]

def _load_variable_frames(model):
    target_path = os.path.join(model.savePath, f"{model}.h5")
    pos_df = pd.read_hdf(target_path, key="positionX")
    theta_df = pd.read_hdf(target_path, key="phaseTheta")
    pos_frames = _segment_frames(pos_df)
    theta_frames = _segment_frames(theta_df)
    return pos_frames, theta_frames

def _apply_boundary_style(ax: plt.Axes, model):
    if hasattr(model, "boundaryVertices"):
        boundary = np.vstack([model.boundaryVertices, model.boundaryVertices[0]])
        ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1.2)
        pad = 0.1
        ax.set_xlim(np.min(model.boundaryVertices[:, 0]) - pad, np.max(model.boundaryVertices[:, 0]) + pad)
        ax.set_ylim(np.min(model.boundaryVertices[:, 1]) - pad, np.max(model.boundaryVertices[:, 1]) + pad)
    else:
        ax.set_xlim(0, model.boundaryLength)
        ax.set_ylim(0, model.boundaryLength)
    ax.set_aspect("equal", adjustable="box")

def draw_frame_dyn(frame: dict):
    idx = frame["index"]
    model = frame["model"]
    positionX = frame["positionX"]
    phaseTheta = frame["phaseTheta"]

    colors = [hexCmap(i) for i in colors_idx(phaseTheta)]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.quiver(
        positionX[:, 0], positionX[:, 1],
        np.cos(phaseTheta), np.sin(phaseTheta),
        scale_units='inches', scale=15.0, width=0.002,
        color=colors
    )
    _apply_boundary_style(ax, model)

    fig.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)
    gc.collect()

def draw_frame(sa: StateAnalysis):
    idx = sa.index
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sa.plot_spatial(ax=ax, colorsBy="phase")
    _apply_boundary_style(ax, sa.model)
    # plt.xticks(
    #     np.arange(0 + xShift, sa.model.boundaryLength + xShift + 1),
    #     np.arange(0, sa.model.boundaryLength + 1))
    # plt.tick_params(length=3, direction="in")
    # plt.xlim(4, 6)
    # plt.ylim(4, 6)

    fig.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)
    gc.collect()


if __name__ == "__main__":

    model = CollisionBoundaryMidpointSpikePatternFormation(
        strengthK=20.75, distanceD0=1.0, phaseLagA0=0.4 * np.pi,
        # initPhaseTheta=np.zeros(1000), 
        freqDist="uniform",
        omegaMin=0, deltaOmega=3, protrusionHeight=1.0, protrusionHalfWidth=0.25,
        agentsNum=2000, dt=0.005,
        tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
        randomSeed=9, overWrite=False
    )

    # model = PhaseLagPatternFormation1D(strengthK=20, distanceD0=1, phaseLagA0=0.6*np.pi, 
    #                                    dt=0.001,
    #                                    tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
    #                                    randomSeed=9, overWrite=True)

    class_name = model.__class__.__name__
    if class_name == "PhaseLagPatternFormationBigArea":
        pos_frames, theta_frames = _load_variable_frames(model)
        total_frames = min(len(pos_frames), len(theta_frames))
    else:
        sa = StateAnalysis(model)

    if os.path.exists(MP4_TEMP_PATH):
        shutil.rmtree(MP4_TEMP_PATH)
    os.mkdir(MP4_TEMP_PATH)
    
    if class_name == "PhaseLagPatternFormationBigArea":
        total_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in tqdm(range(total_batches), desc="Drawing batches", total=total_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, total_frames)
            batch_frames = []
            for i in range(start, end):
                batch_frames.append({
                    "model": model,
                    "positionX": pos_frames[i],
                    "phaseTheta": theta_frames[i].reshape(-1),
                    "index": i
                })
            with Pool(NUM_WORKERS) as p:
                p.map(draw_frame_dyn, batch_frames)
            del batch_frames
            gc.collect()
    else:
        total_batches = (sa.TNum + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in tqdm(range(total_batches), desc="Drawing batches", total=total_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, sa.TNum)
            subSaList = []
            for i in range(start, end):
                subSa = StateAnalysis()
                subSa.totalPositionX = [sa.totalPositionX[i]]
                subSa.totalPhaseTheta = [sa.totalPhaseTheta[i]]
                subSa.model = sa.model
                subSa.index = i
                subSa.model = sa.model
                subSaList.append(subSa)
            with Pool(NUM_WORKERS) as p:
                p.map(draw_frame, subSaList)
            del subSaList
            gc.collect()
    
    if os.path.exists(MP4_PATH + rf"\{model}.mp4"):
        os.remove(rf"{MP4_PATH}\{model}.mp4")
        
    import imageio.v3 as iio
    img = iio.imread(os.path.join(MP4_TEMP_PATH, "0.png"))
    print(img.shape)  # output: (height, width, channels)

    fps = 60
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(MP4_TEMP_PATH, "%d.png"),
        '-vf', f'scale={img.shape[1] // 2 * 2}:{img.shape[0] // 2 * 2}:flags=lanczos', 
        '-c:v', 'libx264',
        '-crf', '20',  # Adjust the quality (lower is better, range 18-28)
        '-pix_fmt', 'yuv420p',
        '-an',  # No audio
        rf"{MP4_PATH}/{model}.mp4"
    ]

    subprocess.run(ffmpeg_command)
    if os.path.exists(MP4_TEMP_PATH):
        shutil.rmtree(MP4_TEMP_PATH)
