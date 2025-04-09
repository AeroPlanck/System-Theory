import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as ma
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClusterMixin
import os
from itertools import product
from main import ThreeBody
from multiprocessing import Pool

class PeriodicDBSCAN(BaseEstimator, ClusterMixin):
    """考虑周期性边界条件的聚类算法"""
    def __init__(self, eps=0.5, min_samples=5, boundary=5):
        self.eps = eps
        self.min_samples = min_samples
        self.boundary = boundary  # 周期边界长度

    def _adjusted_distance(self, a, b):
        """计算周期性调整后的距离"""
        delta = np.abs(a - b)
        delta = np.where(delta > self.boundary/2, self.boundary - delta, delta)
        return np.linalg.norm(delta)
    
    def fit(self, X):
        self.labels_ = -np.ones(X.shape[0], dtype=int)
        cluster_id = 0
        
        for i in range(X.shape[0]):
            if self.labels_[i] != -1:
                continue
                
            # 寻找邻域粒子
            neighbors = []
            for j in range(X.shape[0]):
                if self._adjusted_distance(X[i], X[j]) <= self.eps:
                    neighbors.append(j)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -2  # 标记为噪声
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        return self
    
    def _expand_cluster(self, X, core_idx, neighbors, cluster_id):
        self.labels_[core_idx] = cluster_id
        queue = list(neighbors)
        
        while queue:
            j = queue.pop(0)
            if self.labels_[j] == -2:  # 噪声转为边界点
                self.labels_[j] = cluster_id
            if self.labels_[j] != -1:
                continue
                
            self.labels_[j] = cluster_id
            new_neighbors = []
            for k in range(X.shape[0]):
                if self._adjusted_distance(X[j], X[k]) <= self.eps:
                    new_neighbors.append(k)
            
            if len(new_neighbors) >= self.min_samples:
                queue.extend(new_neighbors)

def compute_cluster_order_parameters(model, d_th=0.3):
    """计算基于聚类的序参量（含21帧平均）"""
    # 数据加载与预处理
    targetPath = f"./data/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX").values
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta").values
    totalPointTheta = pd.read_hdf(targetPath, key="pointTheta").values
    
    TNum, agentsNum = totalPositionX.shape[0]//model.agentsNum, model.agentsNum
    positionX = totalPositionX.reshape(TNum, agentsNum, 2)
    phaseTheta = totalPhaseTheta.reshape(TNum, agentsNum)
    pointTheta = totalPointTheta.reshape(TNum, agentsNum)
    
    # 计算时间窗口
    frame_window = slice(max(0, TNum-21), TNum)
    valid_frames = positionX[frame_window].shape[0]
    
    k2TNum = 100
    position_X = positionX[-k2TNum:, np.newaxis, :, :]
    others = positionX[-k2TNum:, :, np.newaxis, :]
    deltaX = others - position_X
    k2 = np.sqrt(deltaX[:, :, :, 0] ** 2 + deltaX[:, :, :, 1] ** 2) <= model.distanceD2
    K2 = k2[:, :, np.newaxis, :]*k2[:, :, :, np.newaxis]
    
    # 存储各帧结果
    frame_results = {
        'R_c': np.zeros(valid_frames),
        'Domega': np.zeros(valid_frames),
        'symmetric_ratio': []
    }
    
    # 逐帧处理
    for idx, t in enumerate(range(*frame_window.indices(TNum))):
        # 当前帧数据
        curr_position = positionX[t]
        curr_phase = phaseTheta[t]
        curr_omega = pointTheta[t]
        
        # 计算旋转中心
        X = curr_position[:,0] - model.speedV * np.sin(curr_phase) / curr_omega
        Y = curr_position[:,1] + model.speedV * np.cos(curr_phase) / curr_omega
        centers = np.column_stack([X, Y])
        
        # 聚类分析
        clusterer = PeriodicDBSCAN(eps=d_th, min_samples=5, boundary=model.boundaryLength)
        labels = clusterer.fit_predict(centers)
        
        # 计算簇特征
        cluster_R, cluster_Domega = [], []
        for label in np.unique(labels):
            mask = labels == label
            if mask.sum() == 1:
                cluster_R.append(1.0)
                cluster_Domega.append(0.0)
                continue
                
            theta = curr_phase[mask]
            cluster_R.append(np.abs(np.exp(1j*theta).mean()))

            if mask.sum() > 1:  
                cluster_omegas = curr_omega[mask]
                i, j = np.triu_indices(len(cluster_omegas), k=1)
                squared_diffs = (cluster_omegas[i] - cluster_omegas[j])**2
                domega = np.mean(squared_diffs)

            else:
                domega = 0.0  # 粒子情况设为0
            
            cluster_Domega.append(domega)
        
        # 记录当前帧结果
        frame_results['R_c'][idx] = np.mean(cluster_R) if cluster_R else 0.0
        frame_results['Domega'][idx] = np.mean(cluster_Domega) if cluster_Domega else 0.0

    for t in range(0, k2TNum, 10): 
        # 对称性检测优化算法
        K2_t = K2[t]
        symmetric_count = 0
        unique_triples = list()
        
        # 严格遍历所有i,j,k组合
        aRange = range(agentsNum)
        for i, j, k in tqdm(product(aRange, aRange, aRange)):
            """ if not K2_t[i, j, k]:  # 记录所有True组合（标准化排序）
                continue
            if frozenset([i, j, k]) in unique_triples:
                continue
            unique_triples.append(frozenset([i, j, k])) """
            # 检查轮换对称性
            if K2_t[i,j,k] and K2_t[j,k,i] and K2_t[k,i,j]:
                symmetric_count += 1
        
        # 计算当前帧比例
        total = 1313400
        ratio = symmetric_count / total if total > 0 else 0.0
        frame_results['symmetric_ratio'].append(ratio)
    
    # 返回21帧平均值
    return (
        np.mean(frame_results['R_c']),
        np.mean(frame_results['Domega']),
        np.mean(frame_results['symmetric_ratio'])
    )

""" def plot_2d_phase_diagram(params):
    x_param, y_param, fixed_params = params

    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])
    models = [
        ThreeBody(l1, l2, d1, d2, agentsNum=200, boundaryLength=5,
                tqdm=True, overWrite=False)
        for l1, l2, d1, d2  in product(rangeLambdas, rangeLambdas, distanceDs, distanceDs)
    ]

    # 参数校验
    validate_params = [x_param, y_param] + list(fixed_params.keys())
    for param in validate_params:
        if not hasattr(models[0], param):
            raise ValueError(f"Invalid parameter: {param}")

    # 筛选模型
    TOL = 1e-6
    filtered = []
    for model in models:
        match = all(
            np.isclose(getattr(model, key), val, atol=TOL)
            for key, val in fixed_params.items()
        )
        if match:
            filtered.append(model)

    if not filtered:
        print(f"No data for {fixed_params}")
        return

    # 提取动态数据
    data = {
        x_param: [],
        y_param: [],
        'R_c': [],
        'ΔΩ': [],
        'Symmetric_Ratio': []
    }
    
    for model in filtered:
        R_c, Domega, symmetric_ratio = compute_cluster_order_parameters(model)
        data[x_param].append(getattr(model, x_param))
        data[y_param].append(getattr(model, y_param))
        data['R_c'].append(R_c)
        data['ΔΩ'].append(Domega)
        data['Symmetric_Ratio'].append(symmetric_ratio)

    # 创建网格数据
    df = pd.DataFrame(data)
    try:
        pivot_R = df.pivot(index=y_param, columns=x_param, values='R_c')
        pivot_D = df.pivot(index=y_param, columns=x_param, values='ΔΩ')
        pivot_SR = df.pivot(index=y_param, columns=x_param, values='Symmetric_Ratio')
    except KeyError:
        print("Parameter combination not fully sampled")
        return

    # 生成固定参数标签
    fixed_label = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])

    # 可视化设置
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # R_c热图
    im1 = ax[0].imshow(pivot_R,
                      extent=[df[x_param].min(), df[x_param].max(),
                              df[y_param].min(), df[y_param].max()],
                      origin='lower', 
                      cmap='viridis',
                      aspect='auto')
    ax[0].set_title(f'R_c ({fixed_label})')
    ax[0].set_xlabel(x_param)
    ax[0].set_ylabel(y_param)
    plt.colorbar(im1, ax=ax[0], label='R_c')

    # ΔΩ热图
    im2 = ax[1].imshow(pivot_D,
                      extent=[df[x_param].min(), df[x_param].max(),
                              df[y_param].min(), df[y_param].max()],
                      origin='lower',
                      cmap='plasma',
                      aspect='auto')
    ax[1].set_title(f'ΔΩ ({fixed_label})')
    ax[1].set_xlabel(x_param)
    ax[1].set_ylabel(y_param)
    plt.colorbar(im2, ax=ax[1], label='ΔΩ')

    plt.tight_layout()
    filename = f"phase_{x_param}_{y_param}_" + "_".join([f"{k}{v}" for k,v in fixed_params.items()]) + ".png"
    save_path = "./phase diagram"
    full_path = os.path.join(save_path, filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    im3 = plt.imshow(pivot_SR,
                   extent=[df[x_param].min(), df[x_param].max(),
                           df[y_param].min(), df[y_param].max()],
                   origin='lower',
                   cmap='magma',
                   aspect='auto')
    plt.title(f'Symmetric Ratio ({fixed_label})')
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.colorbar(im3, label='Ratio')
    
    # 保存路径和文件名
    filename_sr = f"symmetric_ratio_{x_param}_{y_param}_" + "_".join([f"{k}{v}" for k,v in fixed_params.items()]) + ".png"
    full_path_sr = os.path.join(save_path, filename_sr)
    plt.savefig(full_path_sr, dpi=300, bbox_inches='tight')
    plt.close()
 """

# 创建自定义颜色映射（取消归一化，使用绝对数值映射）
def create_absolute_colormap(vmin, vmax, colors=['#1a2d5f', '#f7d842', '#c12b2b']):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)  # 禁用自动缩放
    return mcolors.LinearSegmentedColormap.from_list('absolute_cmap', colors), norm

# 增强型绘图函数（完全非归一化）
def plot_absolute_parallel(df, color_col, title):
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    params = ['lambda1', 'lambda2', 'distance1', 'distance2']
    n_params = len(params)

    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])
    models = [
        ThreeBody(l1, l2, d1, d2, agentsNum=200, boundaryLength=5,
                tqdm=True, overWrite=False)
        for l1, l2, d1, d2  in product(rangeLambdas, rangeLambdas, distanceDs, distanceDs)
    ]
    
    data = {
        'lambda1': [],
        'lambda2': [],
        'distance1': [],
        'distance2': [],
        'R_c': [],
        'ΔΩ': [],
        'Symmetric_Ratio': []
    }
    
    for model in models:
        R_c, Domega, symmetric_ratio = compute_cluster_order_parameters(model)
        data['lambda1'].append(getattr(model, 'strengthLambda1'))
        data['lambda2'].append(getattr(model, 'strengthLambda2'))
        data['distance1'].append(getattr(model, 'distanceD1'))
        data['distance2'].append(getattr(model,'distanceD2'))
        data['R_c'].append(R_c)
        data['ΔΩ'].append(Domega)
        data['Symmetric_Ratio'].append(symmetric_ratio)

    # 创建网格数据
    df = pd.DataFrame(data)

    # 创建绝对数值颜色映射
    cmap, norm = create_absolute_colormap(
        vmin=df[color_col].min(),
        vmax=df[color_col].max()
    )
    
    # 绘制每条曲线（使用原始绝对数值）
    for idx in df.index:
        values = df.loc[idx, params].values
        color_val = df.loc[idx, color_col]
        plt.plot(range(n_params), values,
                color=cmap(norm(color_val)),  # 直接传入原始值
                alpha=0.7,
                linewidth=1.5,
                solid_capstyle='round')
    
    # 坐标轴标注（保留原始数值范围）
    ax.set_xticks(range(n_params))
    ax.set_xticklabels([
        r'$\lambda_1$ (raw)', r'$\lambda_2$ (raw)', 
        r'$d_1$ (raw)', r'$d_2$ (raw)'
    ], fontsize=14, rotation=45)
    
    # 添加颜色条（显示绝对数值）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, extend='both')  # 保留超出范围的颜色
    cbar.set_label(f'Absolute {color_col}', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # 增强坐标网格（保持原始数值尺度）
    for i in range(n_params):
        ax.axvline(i, color='gray', alpha=0.2, linestyle='-')
    
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    plt.title(f'{title}\nColor Mapping: Absolute {color_col} Values', 
             fontsize=16, pad=25, weight='bold')
    plt.tight_layout()

# 为每个强度指标生成独立图表
for target in ['R_c', 'ΔΩ', 'Symmetric_Ratio']:
    plot_absolute_parallel(df, target, 'Three-Body System Parameter Space')
    plt.savefig(f'parallel_coords_{target}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    totalParams = []
    totalParams.append(('lambda1', 'lambda2', 'distance1', 'distance2'))

    with Pool(len(totalParams)) as p:
        p.map(plot_absolute_parallel, totalParams) 
        
""" if __name__ == "__main__":
    
    param_pairs = [
        ('strengthLambda1', 'distanceD1'),
        ('strengthLambda1', 'distanceD2'),
        ('strengthLambda2', 'distanceD1'),
        ('strengthLambda2', 'distanceD2'),
        ('strengthLambda1', 'strengthLambda2'),
        ('distanceD1', 'distanceD2'),
    ]

    fixed_defaults = {
        'strengthLambda1': 0.5,
        'distanceD1': 0.5,
        'strengthLambda2': 0.5,
        'distanceD2': 0.5
    }

    totalParams = []

    for x_param, y_param in param_pairs:
    # 仅固定非轴参数
        fixed_params = {
            k:v for k,v in fixed_defaults.items() 
            if k not in [x_param, y_param]
        }  # 排除轴参数
        totalParams.append((x_param, y_param, fixed_params))

    with Pool(len(totalParams)) as p:
        p.map(plot_2d_phase_diagram, totalParams) """