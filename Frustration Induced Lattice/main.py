import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from scipy.spatial import Delaunay
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import json
import sys
import os
import shutil

randomSeed = 10

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

colors = ["#403990", "#3A76D6", "#FFC001", "#F46F43", "#FF0000"]
cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
hex_colors_path = os.path.join(script_dir, "swarmalatorlib", "hex_colors.json")
with open(hex_colors_path, "r", encoding="utf-8") as f:
    hexColors = json.load(f)
hexCmap = mcolors.LinearSegmentedColormap.from_list("cmap", hexColors)
phaseCmap = hexCmap.reversed()
phaseNorm = mcolors.Normalize(vmin=0, vmax=2 * np.pi)
freqCmap = mcolors.LinearSegmentedColormap.from_list(
    "freq", ["#414CC7", "#F8F8F8", "#FF0000"]
)


def _freq_norm(freqOmega: np.ndarray) -> mcolors.Normalize:
    maxAbsFreq = np.max(np.abs(freqOmega)) if freqOmega.size else 1.0
    if maxAbsFreq == 0:
        maxAbsFreq = 1.0
    return mcolors.Normalize(vmin=-maxAbsFreq, vmax=maxAbsFreq)


def _plot_colored_quiver(
    ax: plt.Axes,
    positionX: np.ndarray,
    phaseTheta: np.ndarray,
    freqOmega: np.ndarray,
    colorsBy: str,
    scale: float,
    width: float,
):
    assert colorsBy in ["freq", "phase"], "colorsBy must be 'freq' or 'phase'"

    if colorsBy == "freq":
        colorValues = freqOmega
        colorCmap = freqCmap
        colorNorm = _freq_norm(freqOmega)
    else:
        colorValues = phaseTheta
        colorCmap = phaseCmap
        colorNorm = phaseNorm

    return ax.quiver(
        positionX[:, 0], positionX[:, 1],
        np.cos(phaseTheta), np.sin(phaseTheta),
        colorValues,
        cmap=colorCmap, norm=colorNorm,
        scale_units='inches', scale=scale, width=width
    )


def _add_quiver_colorbar(
    ax: plt.Axes,
    quiver,
    colorsBy: str,
    freqOmega: np.ndarray,
    showColorbar: bool,
) -> None:
    if not showColorbar:
        return

    cbar = ax.figure.colorbar(quiver, ax=ax)
    if colorsBy == "phase":
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    else:
        maxAbsFreq = np.max(np.abs(freqOmega)) if freqOmega.size else 0
        if maxAbsFreq == 0:
            cbar.set_ticks([0])
        else:
            cbar.set_ticks([-maxAbsFreq, 0, maxAbsFreq])

from swarmalatorlib.template import Swarmalators2D


class PhaseLagPatternFormation(Swarmalators2D):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        
        assert freqDist in ["uniform", "cauchy", "identical", "unichiral"]
        
        if freqDist == "cauchy":
            omegaMin = 0.0
            deltaOmega = 0.0
        elif freqDist == "identical":
            # For identical distribution, all agents have the same frequency
            pass

        self.strengthK = strengthK
        self.distanceD0 = distanceD0
        self.phaseLagA0 = phaseLagA0
        self.boundaryLength = boundaryLength
        self.speedV = speedV
        self.freqDist = freqDist
        self.initPhaseTheta = initPhaseTheta
        self.omegaMin = omegaMin
        self.deltaOmega = deltaOmega
        self.agentsNum = agentsNum
        self.dt = dt
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.randomSeed = randomSeed
        self.overWrite = overWrite
        
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        if initPhaseTheta is not None:
            assert len(initPhaseTheta) == agentsNum, "initPhaseTheta must match agentsNum"
            self.phaseTheta = initPhaseTheta
        if freqDist == "uniform":
            posOmega = np.random.uniform(omegaMin, omegaMin + deltaOmega, agentsNum // 2)
        elif freqDist == "identical":
            # All agents have the same frequency (omegaMin)
            posOmega = np.full(agentsNum // 2, omegaMin)
        elif freqDist == "unichiral":
            posOmega = np.full(agentsNum, omegaMin)
        else:  # cauchy
            posOmega = np.abs(np.random.standard_cauchy(agentsNum // 2))
        
        if freqDist == "unichiral":
            self.freqOmega = posOmega
        else:
            self.freqOmega = np.concatenate([
                posOmega, -posOmega
            ])
        self.freqOmega = np.sort(self.freqOmega)
        self.halfBoundaryLength = boundaryLength / 2
        self.counts = 0
        self.dotThetaParams = (
            self.boundaryLength,
            self.halfBoundaryLength,
            self.distanceD0,
            self.strengthK,
            self.phaseLagA0,
        )
    
    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        direction = np.zeros((phaseTheta.shape[0], 2))
        direction[:, 0] = np.cos(phaseTheta)
        direction[:, 1] = np.sin(phaseTheta)
        return direction

    @property
    def dotPosition(self) -> np.ndarray:
        return self.speedV * self._direction(self.phaseTheta)

    @property
    def dotPhase(self) -> np.ndarray:
        # return self._calc_dot_phase(self.deltaTheta, self.A, self.freqOmega, 
        #                             self.strengthK, self.phaseLagA0)
        return self._calc_dot_phase_opti(
                positionX=self.positionX, 
                phaseTheta=self.phaseTheta, 
                freqOmega=self.freqOmega, 
                params=self.dotThetaParams
            )
    
    @staticmethod
    @nb.njit
    def _calc_dot_phase_opti(positionX: np.ndarray, phaseTheta: np.ndarray, 
                         freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            xDiff = np.abs(positionX[:, 0] - positionX[i, 0])
            yDiff = np.abs(positionX[:, 1] - positionX[i, 1])
            neighborIdxs = np.where(
                (xDiff < distanceD0) | (boundaryLength - xDiff < distanceD0) & 
                (yDiff < distanceD0) | (boundaryLength - yDiff < distanceD0)
            )[0]
            if neighborIdxs.size == 0:
                continue

            subX = positionX[i] - positionX[neighborIdxs]
            deltaX = positionX[i] - (
                positionX[neighborIdxs] * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
                (positionX[neighborIdxs] - boundaryLength) * (subX < -halfBoundaryLength) + 
                (positionX[neighborIdxs] + boundaryLength) * (subX > halfBoundaryLength)
            )
            distance = np.sqrt(np.sum(deltaX**2, axis=1))
            A = np.where(distance <= distanceD0)[0]
            if A.size == 0:
                continue

            deltaTheta = phaseTheta[neighborIdxs][A] - phaseTheta[i]
            coupling[i] = np.mean(
                np.sin(deltaTheta + phaseLagA0)
            ) - np.sin(phaseLagA0)
        return strengthK * coupling + freqOmega

    @property
    def deltaX(self) -> np.ndarray:
        return self._delta_x(self.positionX, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return positionX - (
            others * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
            (others - boundaryLength) * (subX < -halfBoundaryLength) + 
            (others + boundaryLength) * (subX > halfBoundaryLength)
        )

    @property
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(self.distance_x(self.deltaX) <= self.distanceD0, 1, 0)

    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1] + phaseLagA0) - np.sin(phaseLagA0)
            )
        return K * coupling + omega

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        
        self.positionX = np.mod(self.positionX + dotPos * self.dt, self.boundaryLength)
        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
    
    def plot(self, ax: plt.Axes = None, colorsBy: str = "phase", showColorbar: bool = True):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        quiver = _plot_colored_quiver(
            ax, self.positionX, self.phaseTheta, self.freqOmega,
            colorsBy, scale=15.0, width=0.002
        )
        _add_quiver_colorbar(ax, quiver, colorsBy, self.freqOmega, showColorbar)
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.1f},"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"wMin={self.omegaMin:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )
    

class PhaseLagPatternFormationBigArea(Swarmalators2D):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 16, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        
        assert freqDist in ["uniform", "cauchy", "identical"]
        
        if freqDist == "cauchy":
            omegaMin = 0.0
            deltaOmega = 0.0
        elif freqDist == "identical":
            # For identical distribution, all agents have the same frequency
            pass

        self.strengthK = strengthK
        self.distanceD0 = distanceD0
        self.phaseLagA0 = phaseLagA0
        self.boundaryLength = boundaryLength
        self.speedV = speedV
        self.freqDist = freqDist
        self.initPhaseTheta = initPhaseTheta
        self.omegaMin = omegaMin
        self.deltaOmega = deltaOmega
        self.agentsNum = agentsNum
        self.dt = dt
        self.tqdm = tqdm
        self.savePath = savePath
        self.store = None
        self.shotsnaps = shotsnaps
        self.randomSeed = randomSeed
        self.overWrite = overWrite
        
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength * 0.5 + boundaryLength * 0.25
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        if initPhaseTheta is not None:
            assert len(initPhaseTheta) == agentsNum, "initPhaseTheta must match agentsNum"
            self.phaseTheta = initPhaseTheta
        if freqDist == "uniform":
            posOmega = np.random.uniform(omegaMin, omegaMin + deltaOmega, agentsNum // 2)
        elif freqDist == "identical":
            # All agents have the same frequency (omegaMin)
            posOmega = np.full(agentsNum // 2, omegaMin)
        else:  # cauchy
            posOmega = np.abs(np.random.standard_cauchy(agentsNum // 2))
        self.freqOmega = np.concatenate([
            posOmega, -posOmega
        ])
        self.freqOmega = np.sort(self.freqOmega)
        self.halfBoundaryLength = boundaryLength / 2
        self.counts = 0
        self.dotThetaParams = (
            self.boundaryLength,
            self.halfBoundaryLength,
            self.distanceD0,
            self.strengthK,
            self.phaseLagA0,
        )
    
    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        direction = np.zeros((phaseTheta.shape[0], 2))
        direction[:, 0] = np.cos(phaseTheta)
        direction[:, 1] = np.sin(phaseTheta)
        return direction

    @property
    def dotPosition(self) -> np.ndarray:
        return self.speedV * self._direction(self.phaseTheta)

    @property
    def dotPhase(self) -> np.ndarray:
        # return self._calc_dot_phase(self.deltaTheta, self.A, self.freqOmega, 
        #                             self.strengthK, self.phaseLagA0)
        return self._calc_dot_phase_opti(
                positionX=self.positionX, 
                phaseTheta=self.phaseTheta, 
                freqOmega=self.freqOmega, 
                params=self.dotThetaParams
            )
    
    @staticmethod
    @nb.njit
    def _calc_dot_phase_opti(positionX: np.ndarray, phaseTheta: np.ndarray, 
                         freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            distances = np.sqrt(np.sum((positionX - positionX[i])**2, axis=1))
            neighborIdxs = np.where((distances <= distanceD0) & (distances > 0))[0]
            if neighborIdxs.size == 0:
                continue

            deltaTheta = phaseTheta[neighborIdxs] - phaseTheta[i]
            coupling[i] = np.mean(
                np.sin(deltaTheta + phaseLagA0)
            ) - np.sin(phaseLagA0)
        return strengthK * coupling + freqOmega

    @property
    def deltaX(self) -> np.ndarray:
        return self.positionX[:, np.newaxis] - self.positionX[np.newaxis, :]

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return positionX - (
            others * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
            (others - boundaryLength) * (subX < -halfBoundaryLength) + 
            (others + boundaryLength) * (subX > halfBoundaryLength)
        )

    @property
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(self.distance_x(self.deltaX) <= self.distanceD0, 1, 0)

    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1] + phaseLagA0) - np.sin(phaseLagA0)
            )
        return K * coupling + omega

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        newPosition = self.positionX + dotPos * self.dt
        inside = (
            (newPosition[:, 0] >= 0) & (newPosition[:, 0] <= self.boundaryLength) &
            (newPosition[:, 1] >= 0) & (newPosition[:, 1] <= self.boundaryLength)
        )
        if inside.any():
            self.positionX = newPosition[inside]
            self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)[inside]
            self.freqOmega = self.freqOmega[inside]
            self.agentsNum = self.positionX.shape[0]
        else:
            self.positionX = np.empty((0, 2))
            self.phaseTheta = np.empty((0,))
            self.freqOmega = np.empty((0,))
            self.agentsNum = 0

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
    
    def plot(self, ax: plt.Axes = None, colorsBy: str = "phase", showColorbar: bool = True):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        quiver = _plot_colored_quiver(
            ax, self.positionX, self.phaseTheta, self.freqOmega,
            colorsBy, scale=15.0, width=0.002
        )
        _add_quiver_colorbar(ax, quiver, colorsBy, self.freqOmega, showColorbar)
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.1f},"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"wMin={self.omegaMin:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


class PhaseLagPatternFormation05pi(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 agentsNum: int = 100, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthK, distanceD0, 0.5 * np.pi, boundaryLength, speedV, 
                         "uniform", None, 0, 0, agentsNum, 
                         dt, tqdm, savePath, shotsnaps, randomSeed, overWrite)
    
    @staticmethod
    @nb.njit
    def _calc_dot_phase_opti(positionX: np.ndarray, phaseTheta: np.ndarray, 
                         freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            xDiff = np.abs(positionX[:, 0] - positionX[i, 0])
            yDiff = np.abs(positionX[:, 1] - positionX[i, 1])
            neighborIdxs = np.where(
                (xDiff < distanceD0) | (boundaryLength - xDiff < distanceD0) & 
                (yDiff < distanceD0) | (boundaryLength - yDiff < distanceD0)
            )[0]
            if neighborIdxs.size == 0:
                continue

            subX = positionX[i] - positionX[neighborIdxs]
            deltaX = positionX[i] - (
                positionX[neighborIdxs] * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
                (positionX[neighborIdxs] - boundaryLength) * (subX < -halfBoundaryLength) + 
                (positionX[neighborIdxs] + boundaryLength) * (subX > halfBoundaryLength)
            )
            distance = np.sqrt(np.sum(deltaX**2, axis=1))
            A = np.where(distance <= distanceD0)[0]
            if A.size == 0:
                continue

            deltaTheta = phaseTheta[neighborIdxs][A] - phaseTheta[i]
            coupling[i] = np.mean(np.cos(deltaTheta)) - 1
        return strengthK * coupling + freqOmega
   

class CollisionBoundaryPatternFormation(Swarmalators2D):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        
        assert freqDist in ["uniform", "cauchy", "identical"]
        
        if freqDist == "cauchy":
            omegaMin = 0.0
            deltaOmega = 0.0
        elif freqDist == "identical":
            # For identical distribution, all agents have the same frequency
            pass

        self.strengthK = strengthK
        self.distanceD0 = distanceD0
        self.phaseLagA0 = phaseLagA0
        self.boundaryLength = boundaryLength
        self.speedV = speedV
        self.freqDist = freqDist
        self.initPhaseTheta = initPhaseTheta
        self.omegaMin = omegaMin
        self.deltaOmega = deltaOmega
        self.agentsNum = agentsNum
        self.dt = dt
        self.tqdm = tqdm
        self.savePath = savePath
        self.store = None
        self.shotsnaps = shotsnaps
        self.randomSeed = randomSeed
        self.overWrite = overWrite
        
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        if initPhaseTheta is not None:
            assert len(initPhaseTheta) == agentsNum, "initPhaseTheta must match agentsNum"
            self.phaseTheta = initPhaseTheta
        if freqDist == "uniform":
            posOmega = np.random.uniform(omegaMin, omegaMin + deltaOmega, agentsNum // 2)
        elif freqDist == "identical":
            # All agents have the same frequency (omegaMin)
            posOmega = np.full(agentsNum // 2, omegaMin)
        else:  # cauchy
            posOmega = np.abs(np.random.standard_cauchy(agentsNum // 2))
        self.freqOmega = np.concatenate([
            posOmega, -posOmega
        ])
        self.freqOmega = np.sort(self.freqOmega)
        self.halfBoundaryLength = boundaryLength / 2
        self.counts = 0
        self.dotThetaParams = (
            self.boundaryLength,
            self.halfBoundaryLength,
            self.distanceD0,
            self.strengthK,
            self.phaseLagA0,
        )
    
    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        direction = np.zeros((phaseTheta.shape[0], 2))
        direction[:, 0] = np.cos(phaseTheta)
        direction[:, 1] = np.sin(phaseTheta)
        return direction

    @property
    def dotPosition(self) -> np.ndarray:
        return self.speedV * self._direction(self.phaseTheta)

    @property
    def dotPhase(self) -> np.ndarray:
        return self._calc_dot_phase_collision(
                positionX=self.positionX, 
                phaseTheta=self.phaseTheta, 
                freqOmega=self.freqOmega, 
                params=self.dotThetaParams
            )
    
    @staticmethod
    @nb.njit
    def _calc_dot_phase_collision(positionX: np.ndarray, phaseTheta: np.ndarray, 
                                  freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            # 对于碰撞边界，不需要考虑周期性，直接计算欧几里得距离
            distances = np.sqrt(np.sum((positionX - positionX[i])**2, axis=1))
            neighborIdxs = np.where((distances <= distanceD0) & (distances > 0))[0]
            
            if neighborIdxs.size == 0:
                continue

            deltaTheta = phaseTheta[neighborIdxs] - phaseTheta[i]
            coupling[i] = np.mean(
                np.sin(deltaTheta + phaseLagA0)
            ) - np.sin(phaseLagA0)
        return strengthK * coupling + freqOmega

    @property
    def deltaX(self) -> np.ndarray:
        # 对于碰撞边界，直接计算欧几里得距离差
        return self.positionX[:, np.newaxis] - self.positionX[np.newaxis, :]

    @property
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(self.distance_x(self.deltaX) <= self.distanceD0, 1, 0)

    @staticmethod
    @nb.njit
    def _handle_collision(positionX: np.ndarray, velocity: np.ndarray, 
                          boundaryLength: float) -> Tuple[np.ndarray, np.ndarray]:
        """处理碰撞边界条件"""
        agentsNum = positionX.shape[0]
        newPositionX = positionX.copy()
        newVelocity = velocity.copy()
        
        for i in range(agentsNum):
            # 检查左边界 (x = 0)
            if newPositionX[i, 0] < 0:
                newPositionX[i, 0] = -newPositionX[i, 0]  # 反射：将超出部分折叠回区域内
                newVelocity[i, 0] = -newVelocity[i, 0]    # 反转x方向速度
            
            # 检查右边界 (x = boundaryLength)
            elif newPositionX[i, 0] > boundaryLength:
                newPositionX[i, 0] = 2 * boundaryLength - newPositionX[i, 0]  # 反射：L - (x - L) = 2L - x
                newVelocity[i, 0] = -newVelocity[i, 0]    # 反转x方向速度
            
            # 检查下边界 (y = 0)
            if newPositionX[i, 1] < 0:
                newPositionX[i, 1] = -newPositionX[i, 1]  # 反射
                newVelocity[i, 1] = -newVelocity[i, 1]    # 反转y方向速度
            
            # 检查上边界 (y = boundaryLength)
            elif newPositionX[i, 1] > boundaryLength:
                newPositionX[i, 1] = 2 * boundaryLength - newPositionX[i, 1]  # 反射
                newVelocity[i, 1] = -newVelocity[i, 1]    # 反转y方向速度
        
        return newPositionX, newVelocity

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        
        # 计算新位置
        newPositionX = self.positionX + dotPos * self.dt
        
        # 处理碰撞边界条件
        self.positionX, correctedVelocity = self._handle_collision(
            newPositionX, dotPos, self.boundaryLength
        )
        
        # 如果发生碰撞，需要更新相位（因为速度方向改变了）
        # 检查哪些粒子发生了碰撞（速度方向改变）
        collision_mask = ~np.isclose(correctedVelocity[:, 0], dotPos[:, 0]) | ~np.isclose(correctedVelocity[:, 1], dotPos[:, 1])
        if np.any(collision_mask):
            # 根据新的速度方向更新相位，但只更新发生碰撞的粒子
            newPhaseTheta = np.arctan2(correctedVelocity[:, 1], correctedVelocity[:, 0])
            self.phaseTheta[collision_mask] = newPhaseTheta[collision_mask]
        
        # 更新相位
        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
    
    def plot(self, ax: plt.Axes = None, colorsBy: str = "phase", showColorbar: bool = True):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        quiver = _plot_colored_quiver(
            ax, self.positionX, self.phaseTheta, self.freqOmega,
            colorsBy, scale=15.0, width=0.002
        )
        _add_quiver_colorbar(ax, quiver, colorsBy, self.freqOmega, showColorbar)
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.1f},"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"wMin={self.omegaMin:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


class CollisionBoundaryMidpointSpikePatternFormation(CollisionBoundaryPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 protrusionHeight: float = 0.8, protrusionHalfWidth: float = None,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        # Use the circular-boundary model as the base dynamics/initialization.
        # The class is kept at its historical location/name for API compatibility;
        # CircularBoundaryPatternFormation is available when instances are created.
        CircularBoundaryPatternFormation.__init__(
            self,
            strengthK=strengthK,
            distanceD0=distanceD0,
            phaseLagA0=phaseLagA0,
            boundaryLength=boundaryLength,
            speedV=speedV,
            freqDist=freqDist,
            initPhaseTheta=initPhaseTheta,
            omegaMin=omegaMin,
            deltaOmega=deltaOmega,
            agentsNum=agentsNum,
            dt=dt,
            tqdm=tqdm,
            savePath=savePath,
            shotsnaps=shotsnaps,
            randomSeed=randomSeed,
            overWrite=overWrite
        )

        if protrusionHalfWidth is None:
            protrusionHalfWidth = max(1e-6, protrusionHeight * 0.35)
        assert protrusionHeight >= 0.0, "protrusionHeight must be non-negative"
        assert protrusionHeight < self.circleRadius, "protrusionHeight must be < circle radius"
        assert 0 < protrusionHalfWidth < self.circleRadius, "protrusionHalfWidth must be in (0, circle radius)"

        self.protrusionHeight = protrusionHeight
        self.protrusionHalfWidth = protrusionHalfWidth
        self.spikeTip, self.spikeBaseLeft, self.spikeBaseRight = self._build_spike_geometry(
            self.circleCenter, self.circleRadius,
            self.protrusionHeight, self.protrusionHalfWidth
        )
        self.boundaryVertices = self._build_spike_boundary_vertices(
            self.circleCenter, self.circleRadius,
            self.protrusionHeight, self.protrusionHalfWidth
        )

        # Circular initialization can place particles in the inward spike cut-out.
        # Rejection-sample only those particles so every initial point is valid.
        valid = self._points_inside_spiked_circle(
            self.positionX, self.circleCenter, self.circleRadius,
            self.spikeBaseLeft, self.spikeTip, self.spikeBaseRight
        )
        while not np.all(valid):
            count = np.count_nonzero(~valid)
            angles = np.random.random(count) * 2 * np.pi
            radii = np.sqrt(np.random.random(count)) * self.circleRadius
            self.positionX[~valid] = self.circleCenter + np.stack(
                [radii * np.cos(angles), radii * np.sin(angles)], axis=1
            )
            valid = self._points_inside_spiked_circle(
                self.positionX, self.circleCenter, self.circleRadius,
                self.spikeBaseLeft, self.spikeTip, self.spikeBaseRight
            )

    @staticmethod
    def _build_spike_geometry(center: np.ndarray, radius: float,
                              protrusionHeight: float,
                              protrusionHalfWidth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build one inward spike at the bottom of the circle."""
        base_radial_distance = np.sqrt(
            radius * radius - protrusionHalfWidth * protrusionHalfWidth
        )
        tip = np.array([
            center[0], center[1] - radius + protrusionHeight
        ], dtype=np.float64)
        base_left = np.array([
            center[0] - protrusionHalfWidth,
            center[1] - base_radial_distance
        ], dtype=np.float64)
        base_right = np.array([
            center[0] + protrusionHalfWidth,
            center[1] - base_radial_distance
        ], dtype=np.float64)
        return tip, base_left, base_right

    @staticmethod
    def _build_spike_boundary_vertices(center: np.ndarray, radius: float,
                                       protrusionHeight: float,
                                       protrusionHalfWidth: float,
                                       arcPoints: int = 257) -> np.ndarray:
        """Vertices for plotting: exact circular arc samples plus one spike tip."""
        tip, _, _ = CollisionBoundaryMidpointSpikePatternFormation._build_spike_geometry(
            center, radius, protrusionHeight, protrusionHalfWidth
        )
        half_angle = np.arcsin(protrusionHalfWidth / radius)
        angles = np.linspace(
            -0.5 * np.pi + half_angle,
            1.5 * np.pi - half_angle,
            arcPoints
        )
        arc = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return np.vstack([arc, tip])

    @staticmethod
    def _points_inside_spiked_circle(points: np.ndarray, center: np.ndarray,
                                     radius: float, baseLeft: np.ndarray,
                                     tip: np.ndarray, baseRight: np.ndarray) -> np.ndarray:
        """Return the mask for the circle after cutting out the spike notch."""
        relative = points - center
        inside_circle = np.sum(relative * relative, axis=1) <= radius * radius

        left_edge = tip - baseLeft
        from_left = points - baseLeft
        left_cross = left_edge[0] * from_left[:, 1] - left_edge[1] * from_left[:, 0]

        right_edge = baseRight - tip
        from_tip = points - tip
        right_cross = right_edge[0] * from_tip[:, 1] - right_edge[1] * from_tip[:, 0]
        # The spike is a concave notch: a point is retained when it lies on the
        # interior side of either spike edge.  Requiring both would incorrectly
        # remove two large wedges on the left and right of the spike.
        return inside_circle & ((left_cross >= -1e-12) | (right_cross >= -1e-12))

    @staticmethod
    @nb.njit
    def _handle_collision_spiked_circle(positionX: np.ndarray, velocity: np.ndarray,
                                         center: np.ndarray, radius: float,
                                         baseLeft: np.ndarray, tip: np.ndarray,
                                         baseRight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reflect particles from a circle whose bottom arc is replaced by one spike."""
        agentsNum = positionX.shape[0]
        newPositionX = positionX.copy()
        newVelocity = velocity.copy()
        radius2 = radius * radius
        tolerance = 1e-12

        for i in range(agentsNum):
            point = newPositionX[i].copy()
            vel = newVelocity[i].copy()

            # More than one reflection is only needed for a large integration step
            # that overshoots a spike side and then the circular arc.
            for _ in range(8):
                dx = point[0] - center[0]
                dy = point[1] - center[1]
                r2 = dx * dx + dy * dy

                left_edge_x = tip[0] - baseLeft[0]
                left_edge_y = tip[1] - baseLeft[1]
                left_point_x = point[0] - baseLeft[0]
                left_point_y = point[1] - baseLeft[1]
                left_cross = left_edge_x * left_point_y - left_edge_y * left_point_x

                right_edge_x = baseRight[0] - tip[0]
                right_edge_y = baseRight[1] - tip[1]
                right_point_x = point[0] - tip[0]
                right_point_y = point[1] - tip[1]
                right_cross = right_edge_x * right_point_y - right_edge_y * right_point_x

                if (r2 <= radius2 + tolerance and
                        (left_cross >= -tolerance or right_cross >= -tolerance)):
                    break

                best_kind = -1  # 0: circular arc, 1: left side, 2: right side
                best_dist2 = 1e300
                best_px = 0.0
                best_py = 0.0

                # The circular projection is a candidate only on the retained
                # (major) arc, never on the arc removed to make the spike.
                r = np.sqrt(r2)
                if r > 1e-15:
                    circle_px = center[0] + radius * dx / r
                    circle_py = center[1] + radius * dy / r

                    circle_left_x = circle_px - baseLeft[0]
                    circle_left_y = circle_py - baseLeft[1]
                    circle_left_cross = (
                        left_edge_x * circle_left_y - left_edge_y * circle_left_x
                    )
                    circle_right_x = circle_px - tip[0]
                    circle_right_y = circle_py - tip[1]
                    circle_right_cross = (
                        right_edge_x * circle_right_y - right_edge_y * circle_right_x
                    )
                    if circle_left_cross >= -tolerance or circle_right_cross >= -tolerance:
                        circle_dist = r - radius
                        best_dist2 = circle_dist * circle_dist
                        best_kind = 0
                        best_px = circle_px
                        best_py = circle_py

                # Compare the closest points on both straight spike sides.
                for side in range(2):
                    if side == 0:
                        ax = baseLeft[0]
                        ay = baseLeft[1]
                        bx = tip[0]
                        by = tip[1]
                    else:
                        ax = tip[0]
                        ay = tip[1]
                        bx = baseRight[0]
                        by = baseRight[1]

                    edge_x = bx - ax
                    edge_y = by - ay
                    edge2 = edge_x * edge_x + edge_y * edge_y
                    projection = (
                        ((point[0] - ax) * edge_x + (point[1] - ay) * edge_y) / edge2
                    )
                    if projection < 0.0:
                        projection = 0.0
                    elif projection > 1.0:
                        projection = 1.0
                    px = ax + projection * edge_x
                    py = ay + projection * edge_y
                    distance_x = point[0] - px
                    distance_y = point[1] - py
                    distance2 = distance_x * distance_x + distance_y * distance_y
                    if distance2 < best_dist2:
                        best_dist2 = distance2
                        best_kind = side + 1
                        best_px = px
                        best_py = py

                if best_kind == 0:
                    # Exact circular reflection, matching CircularBoundaryPatternFormation.
                    nx = dx / r
                    ny = dy / r
                    point[0] = center[0] + (2.0 * radius - r) * nx
                    point[1] = center[1] + (2.0 * radius - r) * ny
                else:
                    nx = point[0] - best_px
                    ny = point[1] - best_py
                    normal_length = np.sqrt(nx * nx + ny * ny)
                    if normal_length < 1e-15:
                        if best_kind == 1:
                            edge_x = tip[0] - baseLeft[0]
                            edge_y = tip[1] - baseLeft[1]
                        else:
                            edge_x = baseRight[0] - tip[0]
                            edge_y = baseRight[1] - tip[1]
                        nx = -edge_y
                        ny = edge_x
                        normal_length = np.sqrt(nx * nx + ny * ny)
                    nx /= normal_length
                    ny /= normal_length
                    signed_distance = (
                        (point[0] - best_px) * nx + (point[1] - best_py) * ny
                    )
                    point[0] -= 2.0 * signed_distance * nx
                    point[1] -= 2.0 * signed_distance * ny

                velocity_normal = vel[0] * nx + vel[1] * ny
                vel[0] -= 2.0 * velocity_normal * nx
                vel[1] -= 2.0 * velocity_normal * ny

            newPositionX[i] = point
            newVelocity[i] = vel

        return newPositionX, newVelocity

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase

        newPositionX = self.positionX + dotPos * self.dt
        self.positionX, correctedVelocity = self._handle_collision_spiked_circle(
            newPositionX, dotPos, self.circleCenter, self.circleRadius,
            self.spikeBaseLeft, self.spikeTip, self.spikeBaseRight
        )

        collision_mask = ~np.isclose(correctedVelocity[:, 0], dotPos[:, 0]) | ~np.isclose(correctedVelocity[:, 1], dotPos[:, 1])
        if np.any(collision_mask):
            newPhaseTheta = np.arctan2(correctedVelocity[:, 1], correctedVelocity[:, 0])
            self.phaseTheta[collision_mask] = newPhaseTheta[collision_mask]

        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)

    def plot(self, ax: plt.Axes = None, colorsBy: str = "phase", showColorbar: bool = True):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        quiver = _plot_colored_quiver(
            ax, self.positionX, self.phaseTheta, self.freqOmega,
            colorsBy, scale=15.0, width=0.002
        )
        _add_quiver_colorbar(ax, quiver, colorsBy, self.freqOmega, showColorbar)

        boundary = np.vstack([self.boundaryVertices, self.boundaryVertices[0]])
        ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1.2)

        pad = 0.02 * self.boundaryLength
        ax.set_xlim(self.circleCenter[0] - self.circleRadius - pad,
                    self.circleCenter[0] + self.circleRadius + pad)
        ax.set_ylim(self.circleCenter[1] - self.circleRadius - pad,
                    self.circleCenter[1] + self.circleRadius + pad)
        ax.set_aspect("equal")

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.1f},"
            f"H={self.protrusionHeight:.3f},W={self.protrusionHalfWidth:.3f},"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"wMin={self.omegaMin:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


class CollisionBoundaryFourSpikePatternFormation(CollisionBoundaryPatternFormation):
    """Square collision boundary with one inward spike at each side midpoint."""
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 protrusionHeight: float = 0.8, protrusionHalfWidth: float = None,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(
            strengthK=strengthK,
            distanceD0=distanceD0,
            phaseLagA0=phaseLagA0,
            boundaryLength=boundaryLength,
            speedV=speedV,
            freqDist=freqDist,
            initPhaseTheta=initPhaseTheta,
            omegaMin=omegaMin,
            deltaOmega=deltaOmega,
            agentsNum=agentsNum,
            dt=dt,
            tqdm=tqdm,
            savePath=savePath,
            shotsnaps=shotsnaps,
            randomSeed=randomSeed,
            overWrite=overWrite
        )

        if protrusionHalfWidth is None:
            protrusionHalfWidth = max(1e-6, protrusionHeight * 0.35)
        assert protrusionHeight >= 0.0, "protrusionHeight must be non-negative"
        assert protrusionHeight < boundaryLength / 2, "protrusionHeight must be < L/2 for inward spikes"
        assert 0 < protrusionHalfWidth < boundaryLength / 2, "protrusionHalfWidth must be in (0, L/2)"

        self.protrusionHeight = protrusionHeight
        self.protrusionHalfWidth = protrusionHalfWidth
        self.boundaryVertices = self._build_spike_boundary_vertices(
            self.boundaryLength, self.protrusionHeight, self.protrusionHalfWidth
        )

    @staticmethod
    def _build_spike_boundary_vertices(boundaryLength: float, protrusionHeight: float,
                                       protrusionHalfWidth: float) -> np.ndarray:
        L = boundaryLength
        h = protrusionHeight
        w = protrusionHalfWidth
        mid = 0.5 * L

        return np.array([
            [0.0, 0.0],
            [mid - w, 0.0],
            [mid, h],
            [mid + w, 0.0],
            [L, 0.0],
            [L, mid - w],
            [L - h, mid],
            [L, mid + w],
            [L, L],
            [mid + w, L],
            [mid, L - h],
            [mid - w, L],
            [0.0, L],
            [0.0, mid + w],
            [h, mid],
            [0.0, mid - w],
        ], dtype=np.float64)

    @staticmethod
    @nb.njit
    def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
        x = point[0]
        y = point[1]
        inside = False
        n = vertices.shape[0]

        for i in range(n):
            j = (i - 1 + n) % n
            xi, yi = vertices[i, 0], vertices[i, 1]
            xj, yj = vertices[j, 0], vertices[j, 1]
            intersects = (yi > y) != (yj > y)
            if intersects:
                x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
                if x < x_cross:
                    inside = not inside
        return inside

    @staticmethod
    @nb.njit
    def _closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        ab2 = ab[0] * ab[0] + ab[1] * ab[1]
        if ab2 < 1e-15:
            return a.copy()
        ap = point - a
        t = (ap[0] * ab[0] + ap[1] * ab[1]) / ab2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return a + t * ab

    @staticmethod
    @nb.njit
    def _reflect_by_polygon(point: np.ndarray, velocity: np.ndarray,
                            vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = vertices.shape[0]
        best_idx = 0
        min_dist2 = 1e30

        for i in range(n):
            j = (i + 1) % n
            a = vertices[i]
            b = vertices[j]
            proj = CollisionBoundaryFourSpikePatternFormation._closest_point_on_segment(point, a, b)
            dx = point[0] - proj[0]
            dy = point[1] - proj[1]
            dist2 = dx * dx + dy * dy
            if dist2 < min_dist2:
                min_dist2 = dist2
                best_idx = i

        a = vertices[best_idx]
        b = vertices[(best_idx + 1) % n]
        proj = CollisionBoundaryFourSpikePatternFormation._closest_point_on_segment(point, a, b)
        normal = point - proj
        norm = np.sqrt(normal[0] * normal[0] + normal[1] * normal[1])

        if norm < 1e-12:
            edge = b - a
            normal = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = np.sqrt(normal[0] * normal[0] + normal[1] * normal[1]) + 1e-15

        unit_normal = normal / norm
        dot_p = (point[0] - proj[0]) * unit_normal[0] + (point[1] - proj[1]) * unit_normal[1]
        reflected_point = point - 2.0 * dot_p * unit_normal
        reflected_velocity = velocity - 2.0 * (
            velocity[0] * unit_normal[0] + velocity[1] * unit_normal[1]
        ) * unit_normal

        return reflected_point, reflected_velocity

    @staticmethod
    @nb.njit
    def _handle_collision_spiked(positionX: np.ndarray, velocity: np.ndarray,
                                 vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        agentsNum = positionX.shape[0]
        newPositionX = positionX.copy()
        newVelocity = velocity.copy()
        n = vertices.shape[0]

        for i in range(agentsNum):
            point = newPositionX[i].copy()
            vel = newVelocity[i].copy()

            # point-in-polygon (ray casting)
            x = point[0]
            y = point[1]
            inside = False
            for k in range(n):
                j = (k - 1 + n) % n
                xi = vertices[k, 0]
                yi = vertices[k, 1]
                xj = vertices[j, 0]
                yj = vertices[j, 1]
                intersects = (yi > y) != (yj > y)
                if intersects:
                    x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
                    if x < x_cross:
                        inside = not inside
            if inside:
                continue

            for _ in range(4):
                # find closest edge to current point
                best_idx = 0
                min_dist2 = 1e30
                for k in range(n):
                    j = (k + 1) % n
                    ax = vertices[k, 0]
                    ay = vertices[k, 1]
                    bx = vertices[j, 0]
                    by = vertices[j, 1]
                    abx = bx - ax
                    aby = by - ay
                    ab2 = abx * abx + aby * aby
                    if ab2 < 1e-15:
                        px = ax
                        py = ay
                    else:
                        apx = point[0] - ax
                        apy = point[1] - ay
                        t = (apx * abx + apy * aby) / ab2
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        px = ax + t * abx
                        py = ay + t * aby
                    dx = point[0] - px
                    dy = point[1] - py
                    dist2 = dx * dx + dy * dy
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        best_idx = k

                # reflect point and velocity by closest edge normal
                j = (best_idx + 1) % n
                ax = vertices[best_idx, 0]
                ay = vertices[best_idx, 1]
                bx = vertices[j, 0]
                by = vertices[j, 1]
                abx = bx - ax
                aby = by - ay
                ab2 = abx * abx + aby * aby
                if ab2 < 1e-15:
                    px = ax
                    py = ay
                else:
                    apx = point[0] - ax
                    apy = point[1] - ay
                    t = (apx * abx + apy * aby) / ab2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    px = ax + t * abx
                    py = ay + t * aby

                nx = point[0] - px
                ny = point[1] - py
                norm = np.sqrt(nx * nx + ny * ny)
                if norm < 1e-12:
                    nx = -aby
                    ny = abx
                    norm = np.sqrt(nx * nx + ny * ny) + 1e-15
                ux = nx / norm
                uy = ny / norm

                dot_p = (point[0] - px) * ux + (point[1] - py) * uy
                point[0] = point[0] - 2.0 * dot_p * ux
                point[1] = point[1] - 2.0 * dot_p * uy

                dot_v = vel[0] * ux + vel[1] * uy
                vel[0] = vel[0] - 2.0 * dot_v * ux
                vel[1] = vel[1] - 2.0 * dot_v * uy

                # check if inside after reflection
                x = point[0]
                y = point[1]
                inside = False
                for k in range(n):
                    jj = (k - 1 + n) % n
                    xi = vertices[k, 0]
                    yi = vertices[k, 1]
                    xj = vertices[jj, 0]
                    yj = vertices[jj, 1]
                    intersects = (yi > y) != (yj > y)
                    if intersects:
                        x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
                        if x < x_cross:
                            inside = not inside
                if inside:
                    break

            newPositionX[i] = point
            newVelocity[i] = vel

        return newPositionX, newVelocity

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase

        newPositionX = self.positionX + dotPos * self.dt
        self.positionX, correctedVelocity = self._handle_collision_spiked(
            newPositionX, dotPos, self.boundaryVertices
        )

        collision_mask = ~np.isclose(correctedVelocity[:, 0], dotPos[:, 0]) | ~np.isclose(correctedVelocity[:, 1], dotPos[:, 1])
        if np.any(collision_mask):
            newPhaseTheta = np.arctan2(correctedVelocity[:, 1], correctedVelocity[:, 0])
            self.phaseTheta[collision_mask] = newPhaseTheta[collision_mask]

        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)

    def plot(self, ax: plt.Axes = None, colorsBy: str = "phase", showColorbar: bool = True):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        quiver = _plot_colored_quiver(
            ax, self.positionX, self.phaseTheta, self.freqOmega,
            colorsBy, scale=15.0, width=0.002
        )
        _add_quiver_colorbar(ax, quiver, colorsBy, self.freqOmega, showColorbar)

        boundary = np.vstack([self.boundaryVertices, self.boundaryVertices[0]])
        ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1.2)

        pad = 0.1
        ax.set_xlim(np.min(self.boundaryVertices[:, 0]) - pad, np.max(self.boundaryVertices[:, 0]) + pad)
        ax.set_ylim(np.min(self.boundaryVertices[:, 1]) - pad, np.max(self.boundaryVertices[:, 1]) + pad)
        ax.set_aspect("equal")

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.1f},"
            f"H={self.protrusionHeight:.3f},W={self.protrusionHalfWidth:.3f},"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"wMin={self.omegaMin:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


class CircularBoundaryPatternFormation(Swarmalators2D):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        
        assert freqDist in ["uniform", "cauchy", "identical"]
        
        if freqDist == "cauchy":
            omegaMin = 0.0
            deltaOmega = 0.0
        elif freqDist == "identical":
            pass

        self.strengthK = strengthK
        self.distanceD0 = distanceD0
        self.phaseLagA0 = phaseLagA0
        self.boundaryLength = boundaryLength
        self.speedV = speedV
        self.freqDist = freqDist
        self.initPhaseTheta = initPhaseTheta
        self.omegaMin = omegaMin
        self.deltaOmega = deltaOmega
        self.agentsNum = agentsNum
        self.dt = dt
        self.tqdm = tqdm
        self.savePath = savePath
        self.store = None
        self.shotsnaps = shotsnaps
        self.randomSeed = randomSeed
        self.overWrite = overWrite
        
        self.halfBoundaryLength = boundaryLength / 2
        self.circleCenter = np.array([self.halfBoundaryLength, self.halfBoundaryLength])
        self.circleRadius = self.halfBoundaryLength
        
        np.random.seed(randomSeed)
        angles = np.random.random(agentsNum) * 2 * np.pi
        radii = np.sqrt(np.random.random(agentsNum)) * self.circleRadius
        self.positionX = self.circleCenter + np.stack(
            [radii * np.cos(angles), radii * np.sin(angles)], axis=1
        )
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        if initPhaseTheta is not None:
            assert len(initPhaseTheta) == agentsNum, "initPhaseTheta must match agentsNum"
            self.phaseTheta = initPhaseTheta
        if freqDist == "uniform":
            posOmega = np.random.uniform(omegaMin, omegaMin + deltaOmega, agentsNum // 2)
        elif freqDist == "identical":
            posOmega = np.full(agentsNum // 2, omegaMin)
        else:
            posOmega = np.abs(np.random.standard_cauchy(agentsNum // 2))
        self.freqOmega = np.concatenate([
            posOmega, -posOmega
        ])
        self.freqOmega = np.sort(self.freqOmega)
        self.counts = 0
        self.dotThetaParams = (
            self.boundaryLength,
            self.halfBoundaryLength,
            self.distanceD0,
            self.strengthK,
            self.phaseLagA0,
        )
    
    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        direction = np.zeros((phaseTheta.shape[0], 2))
        direction[:, 0] = np.cos(phaseTheta)
        direction[:, 1] = np.sin(phaseTheta)
        return direction

    @property
    def dotPosition(self) -> np.ndarray:
        return self.speedV * self._direction(self.phaseTheta)

    @property
    def dotPhase(self) -> np.ndarray:
        return self._calc_dot_phase_collision(
                positionX=self.positionX, 
                phaseTheta=self.phaseTheta, 
                freqOmega=self.freqOmega, 
                params=self.dotThetaParams
            )
    
    @staticmethod
    @nb.njit
    def _calc_dot_phase_collision(positionX: np.ndarray, phaseTheta: np.ndarray, 
                                  freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            distances = np.sqrt(np.sum((positionX - positionX[i])**2, axis=1))
            neighborIdxs = np.where((distances <= distanceD0) & (distances > 0))[0]
            
            if neighborIdxs.size == 0:
                continue

            deltaTheta = phaseTheta[neighborIdxs] - phaseTheta[i]
            coupling[i] = np.mean(
                np.sin(deltaTheta + phaseLagA0)
            ) - np.sin(phaseLagA0)
        return strengthK * coupling + freqOmega

    @property
    def deltaX(self) -> np.ndarray:
        return self.positionX[:, np.newaxis] - self.positionX[np.newaxis, :]

    @property
    def A(self) -> np.ndarray:
        return np.where(self.distance_x(self.deltaX) <= self.distanceD0, 1, 0)

    @staticmethod
    @nb.njit
    def _handle_collision_circle(positionX: np.ndarray, velocity: np.ndarray, 
                                 center: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        agentsNum = positionX.shape[0]
        newPositionX = positionX.copy()
        newVelocity = velocity.copy()
        
        for i in range(agentsNum):
            dx = newPositionX[i, 0] - center[0]
            dy = newPositionX[i, 1] - center[1]
            r = np.sqrt(dx * dx + dy * dy)
            
            if r > radius:
                nx = dx / r
                ny = dy / r
                vdotn = newVelocity[i, 0] * nx + newVelocity[i, 1] * ny
                newVelocity[i, 0] = newVelocity[i, 0] - 2 * vdotn * nx
                newVelocity[i, 1] = newVelocity[i, 1] - 2 * vdotn * ny
                newPositionX[i, 0] = center[0] + (2 * radius - r) * nx
                newPositionX[i, 1] = center[1] + (2 * radius - r) * ny
        
        return newPositionX, newVelocity

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        
        newPositionX = self.positionX + dotPos * self.dt
        
        self.positionX, correctedVelocity = self._handle_collision_circle(
            newPositionX, dotPos, self.circleCenter, self.circleRadius
        )
        
        collision_mask = ~np.isclose(correctedVelocity[:, 0], dotPos[:, 0]) | ~np.isclose(correctedVelocity[:, 1], dotPos[:, 1])
        if np.any(collision_mask):
            newPhaseTheta = np.arctan2(correctedVelocity[:, 1], correctedVelocity[:, 0])
            self.phaseTheta[collision_mask] = newPhaseTheta[collision_mask]
        
        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
    
    def plot(self, ax: plt.Axes = None, colorsBy: str = "phase", showColorbar: bool = True):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        quiver = _plot_colored_quiver(
            ax, self.positionX, self.phaseTheta, self.freqOmega,
            colorsBy, scale=15.0, width=0.002
        )
        _add_quiver_colorbar(ax, quiver, colorsBy, self.freqOmega, showColorbar)
        circle = plt.Circle(
            self.circleCenter, self.circleRadius,
            fill=False, color="white", linewidth=0.8, zorder=3
        )
        ax.add_artist(circle)

        pad = 0.02 * self.boundaryLength
        ax.set_xlim(-pad, self.boundaryLength + pad)
        ax.set_ylim(-pad, self.boundaryLength + pad)
        ax.set_aspect("equal", adjustable="box")

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.1f},"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"wMin={self.omegaMin:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


class CellAndSingleParticle(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 singleParticleDis: float, singleParticleAngle: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 agentsNum: int = 100, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthK, distanceD0, phaseLagA0, boundaryLength, speedV, 
                         "uniform", None, 0, 0, agentsNum, 
                         dt, tqdm, savePath, shotsnaps, randomSeed, overWrite)
        cellPhase = np.linspace(0, 2 * np.pi, self.agentsNum - 1, endpoint=False)
        radius = self.speedV / np.abs(self.strengthK * np.sin(self.phaseLagA0))
        phaseShift = np.pi / 2
        cellPositionX = (
            np.ones((self.agentsNum - 1, 2)) * self.halfBoundaryLength +
            radius * np.array([np.cos(cellPhase + phaseShift), np.sin(cellPhase + phaseShift)]).T
        )
        self.positionX[:-1, :] = cellPositionX
        self.phaseTheta[:-1] = cellPhase
        self.positionX[-1, :] = (
            np.array([self.halfBoundaryLength, self.halfBoundaryLength]) + 
            singleParticleDis * np.array([np.cos(singleParticleAngle), np.sin(singleParticleAngle)])
        )
        self.phaseTheta[-1] = np.pi

        self.singleParticleDis = singleParticleDis
        self.singleParticleAngle = singleParticleAngle

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},"
            f"sPD={self.singleParticleDis:.3f},sPA={self.singleParticleAngle:.3f},"
            f"L={self.boundaryLength:.1f},v={self.speedV:.1f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )
    

class CommonDistPhaseLagPatternFormation(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 meanOmega: float = 0, deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthK, distanceD0, phaseLagA0, boundaryLength, speedV, 
                         freqDist, initPhaseTheta, 0, 0, agentsNum, 
                         dt, tqdm, savePath, shotsnaps, randomSeed, overWrite)
        self.meanOmega = meanOmega

        if freqDist == "uniform":
            self.freqOmega = np.random.uniform(meanOmega - deltaOmega, meanOmega + deltaOmega, agentsNum)
        else:
            self.freqOmega = np.random.standard_cauchy(agentsNum) * deltaOmega + meanOmega

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.strengthK:.3f},D0={self.distanceD0:.3f},"
            f"A0={self.phaseLagA0:.3f},L={self.boundaryLength:.1f},"
            f"v={self.speedV:.1f},dist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"wMean={self.meanOmega:.3f},dw={self.deltaOmega:.3f},"
            f"N={self.agentsNum},dt={self.dt:.3f},"
            f"snap={self.shotsnaps},seed={self.randomSeed}"
            ")"
        )


class HalfInitPhaseLagPatternFormation(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0.1, deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthK, distanceD0, phaseLagA0, boundaryLength, speedV, 
                         freqDist, initPhaseTheta, omegaMin, deltaOmega, agentsNum, 
                         dt, tqdm, savePath, shotsnaps, randomSeed, overWrite)
        self.positionX = np.concatenate([
            np.random.random((agentsNum // 2, 2)) * [self.halfBoundaryLength, boundaryLength],
            np.random.random((agentsNum // 2, 2)) * [self.halfBoundaryLength, boundaryLength]
            + [self.halfBoundaryLength, 0]
        ])


class ChessboardPhaseLagPatternFormation(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0.1, deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthK, distanceD0, phaseLagA0, boundaryLength, speedV, 
                         freqDist, initPhaseTheta, omegaMin, deltaOmega, agentsNum, 
                         dt, tqdm, savePath, shotsnaps, randomSeed, overWrite)
        self.positionX = np.concatenate([
            np.random.random((agentsNum // 4, 2)) * [self.halfBoundaryLength, self.halfBoundaryLength],
            np.random.random((agentsNum // 4, 2)) * [self.halfBoundaryLength, self.halfBoundaryLength]
            + [self.halfBoundaryLength, self.halfBoundaryLength],
            np.random.random((agentsNum // 4, 2)) * [self.halfBoundaryLength, self.halfBoundaryLength]
            + [self.halfBoundaryLength, 0],
            np.random.random((agentsNum // 4, 2)) * [self.halfBoundaryLength, self.halfBoundaryLength]
            + [0, self.halfBoundaryLength]
        ])
    

class PhaseLagPatternFormationNoPeriodic(PhaseLagPatternFormation):
    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        
        self.positionX = np.clip(
            self.positionX + dotPos * self.dt, 
            0, self.boundaryLength
        )
        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)
    
    @property
    def deltaX(self) -> np.ndarray:
        return self.positionX - self.positionX[:, np.newaxis]


class PhaseLagPatternFormationNoCounter(PhaseLagPatternFormation):
    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1] + phaseLagA0)
            )
        return K * coupling + omega


class AdditivePhaseLagPatternFormation(PhaseLagPatternFormation):
    @staticmethod
    @nb.njit
    def _calc_dot_phase_opti(positionX: np.ndarray, phaseTheta: np.ndarray, 
                         freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            xDiff = np.abs(positionX[:, 0] - positionX[i, 0])
            yDiff = np.abs(positionX[:, 1] - positionX[i, 1])
            neighborIdxs = np.where(
                (xDiff < distanceD0) | (boundaryLength - xDiff < distanceD0) & 
                (yDiff < distanceD0) | (boundaryLength - yDiff < distanceD0)
            )[0]
            if neighborIdxs.size == 0:
                continue

            subX = positionX[i] - positionX[neighborIdxs]
            deltaX = positionX[i] - (
                positionX[neighborIdxs] * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
                (positionX[neighborIdxs] - boundaryLength) * (subX < -halfBoundaryLength) + 
                (positionX[neighborIdxs] + boundaryLength) * (subX > halfBoundaryLength)
            )
            distance = np.sqrt(np.sum(deltaX**2, axis=1))
            A = np.where(distance <= distanceD0)[0]
            if A.size == 0:
                continue

            deltaTheta = phaseTheta[neighborIdxs][A] - phaseTheta[i]
            coupling[i] = np.sum(
                np.sin(deltaTheta + phaseLagA0) - np.sin(phaseLagA0)
            )
        return strengthK * coupling + freqOmega


class OnlyCounterPhaseLagPatternFormation(PhaseLagPatternFormation):
    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1]) - np.sin(phaseLagA0)
            )
        return K * coupling + omega


class PurePhaseFrustration(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, phaseLagA0: float, 
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None, 
                 omegaMin: float = 0.1, deltaOmega: float = 1, 
                 agentsNum: int = 1000, dt: float = 0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 randomSeed: int = 10, overWrite: bool = False):
        super().__init__(strengthK, 0, phaseLagA0, 0, 0, 
                         freqDist, initPhaseTheta, 
                         omegaMin, deltaOmega, 
                         agentsNum, dt, 
                         tqdm, savePath, shotsnaps, 
                         randomSeed, overWrite)
        
    @property
    def dotPhase(self) -> np.ndarray:
        return self._calc_dot_phase(self.deltaTheta, None, self.freqOmega, 
                                    self.strengthK, self.phaseLagA0)

    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx] + phaseLagA0) - np.sin(phaseLagA0)
            )
        return K * coupling + omega

    def update(self):
        self.phaseTheta = np.mod(self.phaseTheta + self.dotPhase * self.dt, 2 * np.pi)

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))


class PhaseLagPatternFormation1D(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0.1, deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthK, distanceD0, phaseLagA0,
                         boundaryLength, speedV, freqDist, initPhaseTheta,
                         omegaMin, deltaOmega, agentsNum, dt,
                         tqdm, savePath, shotsnaps, randomSeed, overWrite)

        self.positionX = np.random.random(agentsNum) * boundaryLength

    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        return np.cos(phaseTheta)
    
    @property
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(np.abs(self.deltaX) <= self.distanceD0, 1, 0)
    
    def plot(self, ax: plt.Axes = None, showColorbar: bool = True) -> None:
        colors = [new_cmap(i) for i in
            np.floor(256 - self.phaseTheta / (2 * np.pi) * 256).astype(np.int32)
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        plt.quiver(
            self.positionX, np.zeros(self.agentsNum),
            np.cos(self.phaseTheta), np.sin(self.phaseTheta), 
            color=colors, scale=25, width=0.005,
        )

        plt.plot([0, self.boundaryLength], [0, 0], color="black", lw=1.5)
        plt.xlim(-0.1, 0.1 + self.boundaryLength)
        plt.ylim(-0.9, 0.9)
        plt.yticks([])

        ax.set_aspect('equal', adjustable='box')
        for line in ["top", "right"]:
            ax.spines[line].set_visible(False)

        plt.grid()
        plt.tick_params(direction='in')
        if showColorbar:
            plt.scatter(np.full(self.agentsNum, -2), np.full(self.agentsNum, -2),
                        c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            plt.colorbar(ticks=[0, np.pi, 2*np.pi], ax=ax).ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])


class StateAnalysis:
    def __init__(self, model: PhaseLagPatternFormation = None):
        if model is None:
            return
        self.model = model
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        
        # Original snippet for no data flaw
        # totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
        # TNum = totalPhaseTheta.shape[0] // self.model.agentsNum
        # self.TNum = TNum
        # self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)

        # if isinstance(model, PurePhaseFrustration):
        #     return

        # totalPositionX = pd.read_hdf(targetPath, key="positionX")
        # if isinstance(model, PhaseLagPatternFormation1D):
        #     self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum)
        # else:
        #     self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        
        # Snippet for data flaw
        with pd.HDFStore(targetPath, mode='r') as store:
            totalPhaseTheta = store.select("phaseTheta")
            TNum_theta = totalPhaseTheta.shape[0] // self.model.agentsNum

            if isinstance(model, PurePhaseFrustration):
                self.TNum = TNum_theta
                self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum_theta, self.model.agentsNum)
                return

            totalPositionX = store.select("positionX")
            TNum_pos = totalPositionX.shape[0] // self.model.agentsNum

            # Use the minimum TNum to ensure consistency
            self.TNum = min(TNum_theta, TNum_pos)
            truncate_len = self.TNum * self.model.agentsNum

            # Truncate and reshape
            self.totalPhaseTheta = totalPhaseTheta.values[:truncate_len].reshape(self.TNum, self.model.agentsNum)

            if isinstance(model, PhaseLagPatternFormation1D):
                self.totalPositionX = totalPositionX.values[:truncate_len].reshape(self.TNum, self.model.agentsNum)
            else:
                self.totalPositionX = totalPositionX.values[:truncate_len].reshape(self.TNum, self.model.agentsNum, 2)
        

    def _is_circular_boundary(self) -> bool:
        return isinstance(
            self.model,
            (CircularBoundaryPatternFormation,
             CollisionBoundaryMidpointSpikePatternFormation)
        )

    def _calc_delta_x(self, positionX: np.ndarray, others: np.ndarray) -> np.ndarray:
        if self._is_circular_boundary():
            return positionX - others
        return self.model._delta_x(positionX, others, 
                                   self.model.boundaryLength, 
                                   self.model.halfBoundaryLength)

    def get_state(self, index: int = -1):
        if isinstance(self.model, PurePhaseFrustration):
            positionX = None
        else:
            positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]

        return positionX, phaseTheta
    
    def plot_spatial(self, ax: plt.Axes = None, 
                     colorsBy: str = "phase", index: int = -1, 
                     shift: np.ndarray = np.array([0, 0])):
        assert colorsBy in ["freq", "phase"], "colorsBy must be 'freq' or 'phase'"

        if isinstance(self.model, PhaseLagPatternFormation1D):
            self.plot_spatial_1D(ax, colorsBy, index)
        else:
            self.plot_spatial_2D(ax, colorsBy, index, shift)

    def plot_spatial_2D(self, ax: plt.Axes = None, 
                     colorsBy: str = "freq", index: int = -1, 
                     shift: np.ndarray = np.array([0, 0])):

        positionX, phaseTheta = self.get_state(index)
        if self._is_circular_boundary():
            positionX = positionX + shift
        else:
            positionX = np.mod(positionX + shift, self.model.boundaryLength)

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        if colorsBy == "freq":
            colors = (
                ["red"] * (self.model.freqOmega >= 0).sum() + 
                ["#414CC7"] * (self.model.freqOmega < 0).sum()
            )
        elif colorsBy == "phase":
            # Fix: Ensure indices are within [0, 255] and normalize to [0, 1] for colormap
            indices = np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32) % 256
            colors = [hexCmap(i / 255.0) for i in indices]

        ax.quiver(
            positionX[:, 0], positionX[:, 1],
            np.cos(phaseTheta), np.sin(phaseTheta), 
            scale_units='inches', scale=15.0, width=0.005,
            color=colors
        )
        if isinstance(self.model, CollisionBoundaryMidpointSpikePatternFormation):
            boundary = self.model.boundaryVertices + shift
            boundary = np.vstack([boundary, boundary[0]])
            ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1.0)
        elif self._is_circular_boundary():
            circle = plt.Circle(self.model.circleCenter + shift, self.model.circleRadius, fill=False, lw=1.0)
            ax.add_artist(circle)
        ax.set_xlim(0, self.model.boundaryLength)
        ax.set_ylim(0, self.model.boundaryLength)

    def plot_spatial_1D(self, ax: plt.Axes = None, 
                        colorsBy: str = "freq", index: int = -1):
        positionX, phaseTheta = self.get_state(index)

        colors = [new_cmap(i) for i in
            np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        plt.quiver(
            positionX, np.zeros(self.model.agentsNum),
            np.cos(phaseTheta), np.sin(phaseTheta), 
            color=colors, scale=25, width=0.005,
        )

        plt.plot([0, self.model.boundaryLength], [0, 0], color="black", lw=1.5)
        plt.xlim(-0.1, 0.1 + self.model.boundaryLength)
        plt.ylim(-0.9, 0.9)
        plt.yticks([])

        ax.set_aspect('equal', adjustable='box')
        for line in ["top", "right"]:
            ax.spines[line].set_visible(False)

        plt.grid()
        plt.tick_params(direction='in')
        plt.scatter(np.full(self.model.agentsNum, -2), np.full(self.model.agentsNum, -2),
                    c=phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
        plt.colorbar(ticks=[0, np.pi, 2*np.pi], ax=ax).ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

    def check_state_input(self, positionX: np.ndarray = None, phaseTheta: np.ndarray = None,
                          lookIdx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        if ((positionX is None and phaseTheta is not None) or 
            (positionX is not None and phaseTheta is None)):
            raise ValueError("Both positionX and phaseTheta must be provided or both must be None.")
        if positionX is None:
            positionX, phaseTheta = self.get_state(lookIdx)
        return positionX, phaseTheta

    def calc_dot_theta(self, positionX: np.ndarray = None, phaseTheta: np.ndarray = None,
                       lookIdx: int = -1) -> np.ndarray:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)
        
        if hasattr(self.model, "_calc_dot_phase_collision"):
            return self.model._calc_dot_phase_collision(
                positionX, phaseTheta, self.model.freqOmega, self.model.dotThetaParams
            )
        
        deltaTheta = phaseTheta - phaseTheta[:, np.newaxis]
        deltaX = self._calc_delta_x(positionX, positionX[:, np.newaxis])
        A = np.where(self.model.distance_x(deltaX) <= self.model.distanceD0, 1, 0)
        return self.model._calc_dot_phase(deltaTheta, A, self.model.freqOmega, 
                                          self.model.strengthK, self.model.phaseLagA0)
    
    def calc_rotation_center(self, positionX: np.ndarray = None, phaseTheta: np.ndarray = None,
                       lookIdx: int = -1) -> np.ndarray:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)

        positionx, positiony = positionX[:, 0], positionX[:, 1]
        dotPhase = self.calc_dot_theta(positionX, phaseTheta)

        return np.array([
            positionx - self.model.speedV / dotPhase * np.sin(phaseTheta),
            positiony + self.model.speedV / dotPhase * np.cos(phaseTheta)
        ]).T
    
    def calc_classes_and_centers(self, classDistance: float = 0.1,
                                 positionX: np.ndarray = None,
                                 phaseTheta: np.ndarray = None,
                                 lookIdx: int = -1) -> Tuple[List[List[int]], np.ndarray]:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)
        
        centers = self.calc_rotation_center(positionX, phaseTheta, lookIdx)
        if not self._is_circular_boundary():
            centers = np.mod(centers, self.model.boundaryLength)
        deltaX = self._calc_delta_x(centers, centers[:, np.newaxis])
        totalDistances = self.model.distance_x(deltaX)

        classes = self._calc_classes(centers, classDistance, totalDistances)
        return classes, centers
    
    def calc_classes_based_position(self, classDistance: float = 0.1,
                                    positionX: np.ndarray = None,
                                    phaseTheta: np.ndarray = None,
                                    lookIdx: int = -1, 
                                    withPhase: bool = False) -> Tuple[List[List[int]], np.ndarray]:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)

        if withPhase:
            adjPositionX = np.concatenate([
                positionX, 
                phaseTheta.reshape(-1, 1) / (np.pi * 2) * self.model.boundaryLength
            ], axis=1)
        else:
            adjPositionX = positionX

        deltaX = self._calc_delta_x(adjPositionX, adjPositionX[:, np.newaxis])
        totalDistances = (deltaX ** 2).sum(axis=-1) ** 0.5

        classes = self._calc_classes(adjPositionX, classDistance, totalDistances)
        return classes, positionX
    
    def calc_classes(self, classDistance: float = 0.1,
                     positionX: np.ndarray = None,
                     phaseTheta: np.ndarray = None,
                     lookIdx: int = -1) -> List[List[int]]:
        classes, _ = self.calc_classes_and_centers(
            classDistance, positionX, phaseTheta, lookIdx
        )
        return classes

    @staticmethod
    @nb.njit
    def _calc_classes(centers: np.ndarray, classDistance: float, totalDistances: np.ndarray) -> List[List[int]]:
        classes = [[0]]
        classNum = 1
        nonClassifiedOsci = np.arange(1, centers.shape[0])

        for i in nonClassifiedOsci:
            newClass = True

            for classI in range(len(classes)):
                distance = classDistance
                for j in classes[classI]:
                    if totalDistances[i, j] < distance:
                        distance = totalDistances[i, j]
                if distance < classDistance:
                    classes[classI].append(i)
                    newClass = False
                    break

            if newClass:
                classNum += 1
                classes.append([i])

        newClasses = [classes[0]]

        for subClass in classes[1:]:
            newClass = True
            for newClassI in range(len(newClasses)):
                distance = classDistance
                for i in newClasses[newClassI]:
                    for j in subClass:
                        if totalDistances[i, j] < distance:
                            distance = totalDistances[i, j]
                if distance < classDistance:
                    newClasses[newClassI] += subClass
                    newClass = False
                    break

            if newClass:
                newClasses.append(subClass)
    
        return newClasses
    
    def calc_relative_distance(self, position1: np.ndarray, position2: np.ndarray):  #  -> float | np.ndarray
        deltaX = self._calc_delta_x(position1, position2)
        return np.linalg.norm(deltaX, axis=-1)

    def calc_abslute_distance(self, position1: np.ndarray, position2: np.ndarray) -> float:
        deltaX = position1 - position2
        return np.linalg.norm(deltaX, axis=-1)

    def calc_nearby_edges(self, classCenters: np.ndarray,
                          stdMulti: float = 0.3, 
                          relativeDistance: bool = False) -> Tuple[List[Tuple[int, int]], np.ndarray]:

        rawClassNums = classCenters.shape[0]
        if not self._is_circular_boundary():
            positionShifts = product(
                [-self.model.boundaryLength, 0, self.model.boundaryLength],
                [-self.model.boundaryLength, 0, self.model.boundaryLength]
            )
            periodicCenters = []
            for xShift, yShift in positionShifts:
                periodicCenters.append(
                    np.array([classCenters[:, 0] + xShift, classCenters[:, 1] + yShift]).T
                )
            classCenters = np.concatenate(periodicCenters, axis=0)

        tri = Delaunay(classCenters)
        edges = set()
        
        # get all edges from the Delaunay triangulation
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                edges.add(edge)
        # calculate the lengths of all edges
        edgeLengths = []
        for edge in edges:
            p1 = classCenters[edge[0]]
            p2 = classCenters[edge[1]]
            length = np.linalg.norm(p1 - p2)
            edgeLengths.append(length)
        # calculate mean and std of edge lengths
        meanLength = np.mean(edgeLengths)
        stdLength = np.std(edgeLengths)
        # filter edges based on the mean and std
        filteredEdges = []
        for i, edge in enumerate(edges):
            p1 = classCenters[edge[0]]
            p2 = classCenters[edge[1]]
            length = edgeLengths[i]

            if length <= meanLength + stdMulti * stdLength:
                filteredEdges.append(edge)
    
        if relativeDistance:
            if self._is_circular_boundary():
                return [tuple(edge) for edge in filteredEdges]
            edge = np.unique(np.mod(filteredEdges, rawClassNums), axis=0)
            return [tuple(edge) for edge in edge]
        else:
            return [tuple(edge) for edge in filteredEdges], classCenters

    def select_classIdx_of_line(self, selectClassIdx: int, classCenters: np.ndarray,
                                visualAngle: float, span: float) -> List[int]:
        selectClassPos = classCenters[selectClassIdx]
        deltaX = self._calc_delta_x(selectClassPos, classCenters)
        spaceAngle = np.arctan2(deltaX[:, 1], deltaX[:, 0])
        filterClassIdx = np.where(
            (np.abs(spaceAngle - visualAngle) < span) |
            (np.abs(spaceAngle + np.pi  - visualAngle) < span)
        )[0]
        return filterClassIdx.tolist() + [selectClassIdx]
    
    def calc_order_parameter_R(self, phaseTheta: np.ndarray = None,
                               lookIdx: int = -1) -> float:
        if phaseTheta is None:
            _, phaseTheta = self.get_state(lookIdx)
        
        return np.abs(np.mean(np.exp(1j * phaseTheta)))
    
    def calc_order_parameter_Ra(self, phaseTheta: np.ndarray = None,
                                    lookIdx: int = -1, ajdDistance: float = 1) -> float:
        if phaseTheta is None:
            positionX, phaseTheta = self.get_state(lookIdx)

        adjacent = (
            self.calc_replative_distance(positionX, positionX[:, np.newaxis])
            <= ajdDistance
        )
        Rs = np.zeros(phaseTheta.shape[0])
        for i in range(phaseTheta.shape[0]):
            if not np.any(adjacent[i]):
                continue
            nearbyPhases = phaseTheta[adjacent[i]]
            Rs[i] = np.abs(np.mean(np.exp(1j * nearbyPhases)))

        return Rs.mean()


def calc_lattice_constants(sa: StateAnalysis, plot: bool = False, lookIdx: int = -1):

    sa: StateAnalysis
    model = sa.model
    shift = np.array([0., 0.])
    analysisRadius = model.speedV / np.abs(model.strengthK * np.sin(model.phaseLagA0))

    classes, centers = sa.calc_classes_and_centers(classDistance=analysisRadius, lookIdx=lookIdx)
    if len(classes) > model.agentsNum * 0.2:
        # print(f"Too many classes: {len(classes)} > {model.agentsNum * 0.2}, skipping.")
        return [], []
    numInClasses = np.array([len(c) for c in classes])
    # zScoreNum = stats.zscore(numInClasses)
    # classes = [classes[c] for c in range(len(classes)) 
    #            if (zScoreNum[c] > -0.4) and (numInClasses[c] > 10)]
    numThres = np.median(numInClasses[numInClasses > 10]) * 0.
    classes = [classes[c] for c in range(len(classes))
               if (numInClasses[c] > max(numThres, 10))]
    centers = np.mod(centers + shift, model.boundaryLength)
    if len(classes) <= 1:
        # print("Not enough classes, skipping.")
        return [], []

    classCenters: List[np.ndarray] = []
    for c in classes:
        singleClassCenters = centers[c]

        maxDeltaX = np.abs(singleClassCenters[:, 0] - singleClassCenters[:, 0, np.newaxis]).max()
        subXShift = model.halfBoundaryLength if maxDeltaX > model.halfBoundaryLength else 0
        maxDeltaY = np.abs(singleClassCenters[:, 1] - singleClassCenters[:, 1, np.newaxis]).max()
        subYShift = model.halfBoundaryLength if maxDeltaY > model.halfBoundaryLength else 0

        singleClassCenters = np.mod(singleClassCenters - np.array([subXShift, subYShift]), model.boundaryLength)
        classCenter = np.mod(singleClassCenters.mean(axis=0) + np.array([subXShift, subYShift]), model.boundaryLength)
        classCenters.append(classCenter)
    classCenters: np.ndarray = np.array(classCenters)

    edges, ajdClassCenters = sa.calc_nearby_edges(
        classCenters=classCenters, 
        stdMulti=0.3,
        relativeDistance=False
    )

    classAnalRadius = list()

    for _, oscIdx in enumerate(classes):
        freqOmega: np.ndarray = sa.model.freqOmega[oscIdx]
        meanFreq = freqOmega.mean()
        analRadius = model.speedV / np.abs(meanFreq - model.strengthK * np.sin(model.phaseLagA0))
        
        classAnalRadius.append(analRadius)

    classAnalRadius = np.array(classAnalRadius)
    edgeDistances = np.array([
        sa.calc_replative_distance(ajdClassCenters[edge[0]], ajdClassCenters[edge[1]]) 
        for edge in edges
    ])

    if plot:
        sa.plot_spatial(colorsBy="phase", index=-1, shift=shift)
        plt.scatter(
            classCenters[:, 0], classCenters[:, 1],
            facecolor="white", s=30, edgecolor="black", lw=1.5
        )
        for edge in edges:
            plt.plot(ajdClassCenters[edge, 0], ajdClassCenters[edge, 1],
                    color="black", lw=1.2, alpha=0.3, linestyle=(0, (10, 2)), zorder=0)

    # print(len(classes))

    return classAnalRadius, edgeDistances
