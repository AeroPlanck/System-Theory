import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as ma
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClusterMixin
import os
from itertools import product
from main import ThreeBody
from multiprocessing import Pool
from functools import partial

@nb.njit
def adjusted_distance(a, b, boundary):
    """计算周期性调整后的距离（numba加速版）"""
    delta = np.abs(a - b)
    delta = np.where(delta > boundary/2, boundary - delta, delta)
    return np.sqrt(np.sum(delta**2))  # 使用np.sum代替np.linalg.norm以提高兼容性

class PeriodicDBSCAN(BaseEstimator, ClusterMixin):
    """考虑周期性边界条件的聚类算法"""
    def __init__(self, eps=0.5, min_samples=5, boundary=5):
        self.eps = eps
        self.min_samples = min_samples
        self.boundary = boundary  # 周期边界长度

    def _adjusted_distance(self, a, b):
        """计算周期性调整后的距离"""
        # 调用numba加速的函数
        return adjusted_distance(a, b, self.boundary)
    
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

# 使用numba加速旋转中心计算
@nb.njit
def calculate_rotation_centers(positions, speed, phases, omegas):
    """计算旋转中心（numba加速版）"""
    n = positions.shape[0]
    centers = np.zeros((n, 2))
    
    for i in range(n):
        centers[i, 0] = positions[i, 0] - speed * np.sin(phases[i]) / omegas[i]
        centers[i, 1] = positions[i, 1] + speed * np.cos(phases[i]) / omegas[i]
    
    return centers

# 使用numba加速相位一致性计算
@nb.njit
def calculate_phase_coherence(phases):
    """计算相位一致性（numba加速版）"""
    n = len(phases)
    sum_cos = 0.0
    sum_sin = 0.0
    
    for i in range(n):
        sum_cos += np.cos(phases[i])
        sum_sin += np.sin(phases[i])
    
    return np.sqrt(sum_cos**2 + sum_sin**2) / n

def compute_cluster_order_parameters(model, d_th=0.3):
    """计算聚类序参数（优化版）"""
    # 加载模型数据
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
    
    # 存储各帧结果
    frame_results = {
        'R_c': np.zeros(valid_frames),
        'Domega': np.zeros(valid_frames)
    }
    
    # 逐帧处理
    for idx, t in enumerate(range(*frame_window.indices(TNum))):
        # 当前帧数据
        curr_position = positionX[t]
        curr_phase = phaseTheta[t]
        curr_omega = pointTheta[t] / 0.01
        
        # 使用numba加速计算旋转中心
        centers = calculate_rotation_centers(curr_position, model.speedV, curr_phase, curr_omega)
        
        # 聚类分析
        clusterer = PeriodicDBSCAN(eps=d_th, min_samples=5, boundary=model.boundaryLength)
        labels = clusterer.fit(centers).labels_
        
        # 计算簇特征
        cluster_R, cluster_Domega = [], []
        for label in np.unique(labels):
            if label < 0:  # 跳过噪声点
                continue
                
            mask = labels == label
            if mask.sum() == 1:
                cluster_R.append(1.0)
                cluster_Domega.append(0.0)
                continue
                
            # 使用numba加速计算相位一致性
            theta = curr_phase[mask]
            if len(theta) > 0:
                cluster_R.append(calculate_phase_coherence(theta))
            else:
                cluster_R.append(0.0)

            if mask.sum() > 1:  
                cluster_omegas = curr_omega[mask]
                i, j = np.triu_indices(len(cluster_omegas), k=1)
                squared_diffs = (cluster_omegas[i] - cluster_omegas[j])**2
                domega = np.mean(squared_diffs)
            else:
                domega = 0.0  # 单粒子情况设为0
            
            cluster_Domega.append(domega)
        
        # 记录当前帧结果
        frame_results['R_c'][idx] = np.mean(cluster_R) if cluster_R else 0.0
        frame_results['Domega'][idx] = np.mean(cluster_Domega) if cluster_Domega else 0.0

    return (
        np.mean(frame_results['R_c']),
        np.mean(frame_results['Domega'])
    )

# 使用numba加速对称比率计算
@nb.njit(parallel=True)
def compute_symmetric_ratio_kernel(K2, k2TNum, agentsNum):
    """使用numba加速的对称比率计算核心函数"""
    frame_results = np.zeros((k2TNum + 9) // 10)
    
    for idx in nb.prange((k2TNum + 9) // 10):
        t = idx * 10
        if t >= k2TNum:
            continue
            
        K2_t = K2[t]
        symmetric_count = 0
        valid_triples_count = 0  # 计数K2_t[i,j,k]为真的三元组
        
        # 优化循环，只检查不重复的三元组
        for i in range(agentsNum):
            for j in range(i+1, agentsNum):  # 避免重复
                for k in range(j+1, agentsNum):  # 避免重复
                    # 统计K2_t[i,j,k]为真的三元组
                    if K2_t[i,j,k]:
                        valid_triples_count += 1
                        # 检查轮换对称性
                        if (K2_t[j,k,i] and K2_t[k,i,j]):
                            symmetric_count += 1
        
        # 计算当前帧比例，使用K2_t[i,j,k]为真的三元组数量作为分母
        frame_results[idx] = symmetric_count / valid_triples_count if valid_triples_count > 0 else 0.0
    
    return np.mean(frame_results[:(k2TNum + 9) // 10])

def compute_symmetric_ratio(model):
    """计算对称比率（优化版）"""
    # 加载模型数据
    targetPath = f"./data/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX").values
    
    TNum, agentsNum = totalPositionX.shape[0]//model.agentsNum, model.agentsNum
    positionX = totalPositionX.reshape(TNum, agentsNum, 2)
    
    # 计算K2矩阵
    k2TNum = min(100, TNum)  # 确保不超过TNum
    position_X = positionX[-k2TNum:, np.newaxis, :, :]
    others = positionX[-k2TNum:, :, np.newaxis, :]
    deltaX = others - position_X
    k2 = np.sqrt(deltaX[:, :, :, 0] ** 2 + deltaX[:, :, :, 1] ** 2) <= model.distanceD2
    K2 = k2[:, :, np.newaxis, :]*k2[:, :, :, np.newaxis]
    
    # 调用numba加速的核心函数
    return compute_symmetric_ratio_kernel(K2, k2TNum, agentsNum)

# 创建自定义颜色映射（取消归一化，使用绝对数值映射）
def create_absolute_colormap(vmin, vmax, colors=['#1a2d5f', '#f7d842', '#c12b2b']):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)  # 禁用自动缩放
    return mcolors.LinearSegmentedColormap.from_list('absolute_cmap', colors), norm

# 修改plot_absolute_parallel函数，使其适合多进程调用
def plot_absolute_parallel(params):
    """适合多进程调用的平行坐标图绘制函数"""
    target, df_path, save_path = params
    
    # 加载数据
    try:
        df = pd.read_csv(df_path)
    except:
        print(f"无法加载数据文件: {df_path}")
        return False
    
    if df.empty or len(df) == 0 or target not in df.columns:
        print(f"错误: 无法获取有效数据绘制 {target} 的平行坐标图")
        return False
    
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    params_list = ['lambda1', 'lambda2', 'distance1', 'distance2']
    n_params = len(params_list)
    
    # 创建绝对数值颜色映射
    cmap, norm = create_absolute_colormap(
        vmin=df[target].min(),
        vmax=df[target].max()
    )
    
    # 绘制每条曲线（使用原始绝对数值）
    for idx in df.index:
        values = df.loc[idx, params_list].values
        color_val = df.loc[idx, target]
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
    sm.set_array([])  # 设置空数组以避免警告
    cbar = plt.colorbar(sm, ax=ax, extend='both')  # 明确指定ax参数
    cbar.set_label(f'Absolute {target}', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # 增强坐标网格（保持原始数值尺度）
    for i in range(n_params):
        ax.axvline(i, color='gray', alpha=0.2, linestyle='-')
    
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    plt.title(f'Three-Body System Parameter Space\nColor Mapping: Absolute {target} Values', 
             fontsize=16, pad=25, weight='bold')
    plt.tight_layout()
    
    # 保存图像
    filename = f'parallel_coords_{target}.pdf'
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print(f"已保存图像: {filename}")
    return True

if __name__ == "__main__":
    # 尝试加载已有数据
    save_path = "./parallel coordinate"
    os.makedirs(save_path, exist_ok=True)
    data_path = os.path.join(save_path, "parallel_coords_data.csv")
    
    # 检查数据是否存在
    if not os.path.exists(data_path):
        print("未找到现有数据，将重新计算")
        
        # 这里可以添加数据计算代码
        # 使用较小的参数范围以加快测试
        rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
        ])
        distanceDs = np.concatenate([
            np.arange(0.1, 1, 0.2)
        ])
        
        # 创建模型列表
        models = []
        for l1, l2, d1, d2 in tqdm(list(product(rangeLambdas, rangeLambdas, distanceDs, distanceDs)), 
                                   desc="创建模型"):
            try:
                model = ThreeBody(l1, l2, d1, d2, agentsNum=200, boundaryLength=5,
                                tqdm=False, overWrite=False)
                models.append(model)
            except Exception as e:
                print(f"创建模型出错 ({l1}, {l2}, {d1}, {d2}): {e}")
        
        # 收集数据
        data = {
            'lambda1': [],
            'lambda2': [],
            'distance1': [],
            'distance2': [],
            'R_c': [],
            'ΔΩ': [],
            'Symmetric_Ratio': []
        }
        
        for model in tqdm(models, desc="处理模型"):
            try:
                # 计算聚类序参数
                R_c, Domega = compute_cluster_order_parameters(model)
                
                # 计算对称比率
                symmetric_ratio = compute_symmetric_ratio(model)
                
                data['lambda1'].append(getattr(model, 'strengthLambda1'))
                data['lambda2'].append(getattr(model, 'strengthLambda2'))
                data['distance1'].append(getattr(model, 'distanceD1'))
                data['distance2'].append(getattr(model, 'distanceD2'))
                data['R_c'].append(R_c)
                data['ΔΩ'].append(Domega)
                data['Symmetric_Ratio'].append(symmetric_ratio)
            except Exception as e:
                print(f"处理模型出错 ({model}): {e}")
        
        # 创建新的DataFrame
        df = pd.DataFrame(data)
        
        # 保存数据以便后续使用
        df.to_csv(data_path, index=False)
        print(f"数据已保存至 {data_path}")
    else:
        print("已找到现有数据文件")
        # 加载现有数据
        df = pd.read_csv(data_path)
    
    # 准备多进程参数
    targets = ['R_c', 'ΔΩ', 'Symmetric_Ratio']
    process_params = [(target, data_path, save_path) for target in targets]
    
    # 使用多进程绘制平行坐标图
    print("开始多进程绘制平行坐标图...")
    with Pool(min(len(targets), os.cpu_count())) as p:
        results = p.map(plot_absolute_parallel, process_params)
    
    # 检查结果
    success_count = sum(1 for r in results if r)
    print(f"所有平行坐标图已完成! 成功: {success_count}/{len(targets)}")