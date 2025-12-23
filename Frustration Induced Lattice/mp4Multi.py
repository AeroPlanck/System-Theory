import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import subprocess
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

from multiprocessing import Pool
import pandas as pd

SAVE_PATH = r"D:\PythonProject\System Theory\Frustration Induced Lattice\data"
MP4_PATH = r"D:\PythonProject\System Theory\Frustration Induced Lattice\mp4"
MP4_TEMP_PATH = r"D:\PythonProject\System Theory\Frustration Induced Lattice\mp4_temp"


def _segment_frames(df: pd.DataFrame) -> list:
    idx = df.index.to_numpy()
    if idx.size == 0:
        return []
    reset = np.where(idx[1:] <= idx[:-1])[0] + 1
    bounds = np.concatenate([[0], reset, [len(idx)]])
    return [df.iloc[bounds[i]:bounds[i + 1]].values for i in range(len(bounds) - 1)]

def _load_variable_frames(model) -> list:
    target_path = os.path.join(model.savePath, f"{model}.h5")
    pos_df = pd.read_hdf(target_path, key="positionX")
    theta_df = pd.read_hdf(target_path, key="phaseTheta")
    pos_frames = _segment_frames(pos_df)
    theta_frames = _segment_frames(theta_df)
    T = min(len(pos_frames), len(theta_frames))
    frames = []
    for i in range(T):
        frames.append({
            "model": model,
            "positionX": pos_frames[i],
            "phaseTheta": theta_frames[i].reshape(-1),
            "index": i
        })
    return frames

def draw_frame_dyn(frame: dict):
    idx = frame["index"]
    model = frame["model"]
    positionX = frame["positionX"]
    phaseTheta = frame["phaseTheta"]

    colors = [hexCmap(i) for i in colors_idx(phaseTheta)]
    plt.quiver(
        positionX[:, 0], positionX[:, 1],
        np.cos(phaseTheta), np.sin(phaseTheta),
        scale_units='inches', scale=15.0, width=0.002,
        color=colors
    )
    plt.xlim(0, model.boundaryLength)
    plt.ylim(0, model.boundaryLength)

    plt.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"), bbox_inches='tight', dpi=200)
    plt.close()

def draw_frame(sa: StateAnalysis):
    idx = sa.index
    
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    sa.plot_spatial(ax=None, colorsBy="phase")

    xShift = 0.
    plt.xlim(0 + xShift, sa.model.boundaryLength + xShift)
    plt.ylim(0, sa.model.boundaryLength)
    # plt.xticks(
    #     np.arange(0 + xShift, sa.model.boundaryLength + xShift + 1),
    #     np.arange(0, sa.model.boundaryLength + 1))
    # plt.tick_params(length=3, direction="in")
    # plt.xlim(4, 6)
    # plt.ylim(4, 6)

    plt.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"), bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == "__main__":

    model = PhaseLagPatternFormation(
        strengthK=25, distanceD0=3.5, phaseLagA0=1 * np.pi,
        # initPhaseTheta=np.zeros(1000), 
        freqDist="uniform",
        omegaMin=0, deltaOmega=0,
        agentsNum=1000, dt=0.005,
        tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
        randomSeed=9, overWrite=False
    )

    # model = PhaseLagPatternFormation1D(strengthK=20, distanceD0=1, phaseLagA0=0.6*np.pi, 
    #                                    dt=0.001,
    #                                    tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
    #                                    randomSeed=9, overWrite=True)

    class_name = model.__class__.__name__
    if class_name == "PhaseLagPatternFormationBigArea":
        frames = _load_variable_frames(model)
    else:
        sa = StateAnalysis(model)
        subSaList = list()
        for i in tqdm(range(0, sa.TNum), desc="Processing data"):
            subSa = StateAnalysis()
            subSa.totalPositionX = [sa.totalPositionX[i]]
            subSa.totalPhaseTheta = [sa.totalPhaseTheta[i]]
            subSa.model = sa.model
            subSa.index = i
            subSa.model = sa.model
            subSaList.append(subSa)

    if os.path.exists(MP4_TEMP_PATH):
        shutil.rmtree(MP4_TEMP_PATH)
    os.mkdir(MP4_TEMP_PATH)
    
    with Pool(10) as p:
        if class_name == "PhaseLagPatternFormationBigArea":
            p.map(
                draw_frame_dyn,
                tqdm(frames, desc="Drawing frames", total=len(frames)),
            )
        else:
            p.map(
                draw_frame,
                tqdm(subSaList, desc="Drawing frames", total=sa.TNum),
            )
    
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
