import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClusterMixin
import os
from itertools import product
from main import ThreeBody
from multiprocessing import Pool
from functools import partial
import sys

# 从phaseDiagram导入PeriodicDBSCAN类
from phaseDiagram import PeriodicDBSCAN

# 创建数据缓存字典
model_data_cache = {}

def load_model_data(model):
    """加载模型数据并进行预处理"""
    model_key = str(model)
    if model_key in model_data_cache:
        return model_data_cache[model_key]
    
    try:
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

        # 计算K2矩阵
        k2TNum = min(100, TNum)  # 确保不超过TNum
        position_X = positionX[-k2TNum:, np.newaxis, :, :]
        others = positionX[-k2TNum:, :, np.newaxis, :]
        deltaX = others - position_X
        k2 = np.sqrt(deltaX[:, :, :, 0] ** 2 + deltaX[:, :, :, 1] ** 2) <= model.distanceD2
        K2 = k2[:, :, np.newaxis, :]*k2[:, :, :, np.newaxis]
        
        data = {
            'positionX': positionX,
            'phaseTheta': phaseTheta,
            'pointTheta': pointTheta,
            'TNum': TNum,
            'agentsNum': agentsNum,
            'frame_window': frame_window,
            'valid_frames': valid_frames,
            'K2': K2,
            'k2TNum': k2TNum
        }
        
        # 存入缓存
        model_data_cache[model_key] = data
        return data
    except Exception as e:
        print(f"加载数据出错 ({model}): {e}")
        return None

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
    """计算聚类序参数"""
    # 加载数据
    data = load_model_data(model)
    if data is None:
        return 0.0, 0.0
        
    positionX = data['positionX']
    phaseTheta = data['phaseTheta']
    pointTheta = data['pointTheta']
    TNum = data['TNum']
    frame_window = data['frame_window']
    valid_frames = data['valid_frames']
    
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
    frame_results = np.zeros(k2TNum // 10 + 1)
    
    for idx in nb.prange((k2TNum + 9) // 10):
        t = idx * 10
        if t >= k2TNum:
            continue
            
        K2_t = K2[t]
        symmetric_count = 0
        
        # 优化循环，只检查不重复的三元组
        for i in range(agentsNum):
            for j in range(i+1, agentsNum):  # 避免重复
                for k in range(j+1, agentsNum):  # 避免重复
                    # 检查轮换对称性
                    if (K2_t[i,j,k] and K2_t[j,k,i] and K2_t[k,i,j]):
                        symmetric_count += 1
        
        # 计算当前帧比例
        total_triples = agentsNum * (agentsNum - 1) * (agentsNum - 2) // 6  # 组合数C(n,3)
        frame_results[idx] = symmetric_count / total_triples if total_triples > 0 else 0.0
    
    return np.mean(frame_results[:(k2TNum + 9) // 10])

def compute_symmetric_ratio(model):
    """计算对称比率"""
    data = load_model_data(model)
    if data is None:
        return 0.0
        
    K2 = data['K2']
    k2TNum = data['k2TNum']
    agentsNum = data['agentsNum']
    
    # 调用numba加速的核心函数
    return compute_symmetric_ratio_kernel(K2, k2TNum, agentsNum)

def process_fixed_param(param_tuple):
    """处理单个固定参数的函数，用于并行处理"""
    fixed_param_name, fixed_param_value = param_tuple
    try:
        print(f"处理固定参数: {fixed_param_name}={fixed_param_value}")
        plot_3d_phase_diagram(fixed_param_name, fixed_param_value)
        return True
    except Exception as e:
        print(f"处理 {fixed_param_name}={fixed_param_value} 时出错: {e}")
        return False

def plot_3d_phase_diagram(fixed_param_name, fixed_param_value):
    """绘制3D相图"""
    print(f"开始绘制3D相图 (固定 {fixed_param_name}={fixed_param_value})")
    
    # 使用较小的参数范围以加快处理速度
    rangeLambdas = np.concatenate([
        np.arange(0.1, 0.4, 0.2)  # 减少采样点
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 0.4, 0.2)  # 减少采样点
    ])
    
    # 确定坐标轴参数和颜色参数
    param_names = ['strengthLambda1', 'strengthLambda2', 'distanceD1', 'distanceD2']
    variable_params = [p for p in param_names if p != fixed_param_name]
    
    if len(variable_params) != 3:
        raise ValueError("请确保固定1个参数，剩余3个参数作为变量")
    
    x_param, y_param, z_param = variable_params
    
    # 创建模型实例
    models = []
    
    # 根据固定参数筛选参数组合
    filtered_combinations = []
    for l1, l2, d1, d2 in product(rangeLambdas, rangeLambdas, distanceDs, distanceDs):
        params = {
            'strengthLambda1': l1,
            'strengthLambda2': l2,
            'distanceD1': d1,
            'distanceD2': d2
        }
        
        # 检查是否匹配固定参数
        if abs(params[fixed_param_name] - fixed_param_value) < 1e-6:
            filtered_combinations.append((l1, l2, d1, d2))
    
    if not filtered_combinations:
        print(f"没有找到匹配的参数组合 (固定 {fixed_param_name}={fixed_param_value})")
        return
    
    print(f"找到 {len(filtered_combinations)} 个匹配的参数组合")
    
    # 创建模型
    for l1, l2, d1, d2 in tqdm(filtered_combinations, desc="创建模型"):
        try:
            model = ThreeBody(l1, l2, d1, d2, agentsNum=200, boundaryLength=5,
                            tqdm=False, overWrite=False)
            models.append(model)
        except Exception as e:
            print(f"创建模型出错 ({l1}, {l2}, {d1}, {d2}): {e}")
    
    if not models:
        print(f"没有成功创建任何模型 (固定 {fixed_param_name}={fixed_param_value})")
        return
    
    # 收集数据
    data = {
        x_param: [],
        y_param: [],
        z_param: [],
        'R_c': [],
        'Domega': [],
        'Symmetric_Ratio': []
    }
    
    for model in tqdm(models, desc="处理模型"):
        try:
            # 计算序参数
            R_c, Domega = compute_cluster_order_parameters(model)
            
            # 计算对称比率
            symmetric_ratio = compute_symmetric_ratio(model)
            
            # 收集数据
            data[x_param].append(getattr(model, x_param))
            data[y_param].append(getattr(model, y_param))
            data[z_param].append(getattr(model, z_param))
            data['R_c'].append(R_c)
            data['Domega'].append(Domega)
            data['Symmetric_Ratio'].append(symmetric_ratio)
        except Exception as e:
            print(f"处理模型出错 ({model}): {e}")
    
    if not data[x_param]:
        print(f"没有收集到任何数据 (固定 {fixed_param_name}={fixed_param_value})")
        return
    
    # 创建3D图形
    fig = plt.figure(figsize=(18, 6))
    
    # R_c 3D图
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(data[x_param], data[y_param], data[z_param], 
                     c=data['R_c'], cmap='viridis', s=50, alpha=0.8)
    ax1.set_title(f'R_c (Fixed {fixed_param_name}={fixed_param_value})')
    ax1.set_xlabel(x_param)
    ax1.set_ylabel(y_param)
    ax1.set_zlabel(z_param)
    fig.colorbar(sc1, ax=ax1, label='R_c')
    
    # Domega 3D图
    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(data[x_param], data[y_param], data[z_param], 
                     c=data['Domega'], cmap='plasma', s=50, alpha=0.8)
    ax2.set_title(f'ΔΩ (Fixed {fixed_param_name}={fixed_param_value})')
    ax2.set_xlabel(x_param)
    ax2.set_ylabel(y_param)
    ax2.set_zlabel(z_param)
    fig.colorbar(sc2, ax=ax2, label='ΔΩ')
    
    # Symmetric Ratio 3D图
    ax3 = fig.add_subplot(133, projection='3d')
    sc3 = ax3.scatter(data[x_param], data[y_param], data[z_param], 
                     c=data['Symmetric_Ratio'], cmap='magma', s=50, alpha=0.8)
    ax3.set_title(f'Symmetric Ratio (Fixed {fixed_param_name}={fixed_param_value})')
    ax3.set_xlabel(x_param)
    ax3.set_ylabel(y_param)
    ax3.set_zlabel(z_param)
    fig.colorbar(sc3, ax=ax3, label='Symmetric Ratio')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = "./3d_phase_diagram"
    os.makedirs(save_path, exist_ok=True)
    filename = f"3d_phase_fixed_{fixed_param_name}_{fixed_param_value}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存图像: {filename}")
    
    # 保存数据
    df = pd.DataFrame(data)
    data_filename = f"3d_phase_data_{fixed_param_name}_{fixed_param_value}.csv"
    df.to_csv(os.path.join(save_path, data_filename), index=False)
    print(f"已保存数据: {data_filename}")
    
    return data

def create_interactive_3d_plot(fixed_param_name, fixed_param_value, data=None):
    """创建交互式3D图形（可选）"""
    if data is None:
        # 尝试从文件加载数据
        save_path = "./3d_phase_diagram"
        data_filename = f"3d_phase_data_{fixed_param_name}_{fixed_param_value}.csv"
        try:
            data = pd.read_csv(os.path.join(save_path, data_filename))
        except:
            print(f"无法加载数据文件: {data_filename}")
            return
    
    # 确定坐标轴参数
    param_names = ['strengthLambda1', 'strengthLambda2', 'distanceD1', 'distanceD2']
    variable_params = [p for p in param_names if p != fixed_param_name]
    x_param, y_param, z_param = variable_params
    
    # 创建交互式3D图形
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化散点图
    sc = ax.scatter(data[x_param], data[y_param], data[z_param], 
                   c=data['R_c'], cmap='viridis', s=50, alpha=0.8)
    ax.set_title(f'Interactive 3D Phase Diagram (Fixed {fixed_param_name}={fixed_param_value})')
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(z_param)
    fig.colorbar(sc, label='R_c')
    
    # 定义动画函数
    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return [sc]
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=range(0, 360, 2), blit=True)
    
    # 保存动画
    save_path = "./3d_phase_diagram"
    os.makedirs(save_path, exist_ok=True)
    filename = f"3d_phase_interactive_{fixed_param_name}_{fixed_param_value}.gif"
    ani.save(os.path.join(save_path, filename), writer='pillow', fps=15)
    
    plt.close()
    print(f"已保存交互式图像: {filename}")

def process_parameter_difference(param_tuple):
    """处理单个参数类型的差异计算，用于并行处理"""
    param_name, param_values = param_tuple
    
    if len(param_values) < 2:
        print(f"参数 {param_name} 只有一个值，无法计算差异")
        return False
    
    print(f"处理参数 {param_name} 的相邻值差异...")
    
    # 遍历相邻的参数值对
    for i in range(len(param_values) - 1):
        value1 = param_values[i]
        value2 = param_values[i + 1]
        
        try:
            # 加载两个参数值对应的数据
            save_path = "./3d_phase_diagram"
            data_file1 = f"3d_phase_data_{param_name}_{value1}.csv"
            data_file2 = f"3d_phase_data_{param_name}_{value2}.csv"
            
            # 检查文件是否存在
            if not os.path.exists(os.path.join(save_path, data_file1)) or \
               not os.path.exists(os.path.join(save_path, data_file2)):
                print(f"数据文件不存在，跳过 {param_name}: {value1} -> {value2}")
                continue
            
            # 读取数据
            df1 = pd.read_csv(os.path.join(save_path, data_file1))
            df2 = pd.read_csv(os.path.join(save_path, data_file2))
            
            # 确定坐标轴参数
            param_names = ['strengthLambda1', 'strengthLambda2', 'distanceD1', 'distanceD2']
            variable_params = [p for p in param_names if p != param_name]
            x_param, y_param, z_param = variable_params
            
            # 创建合并的数据框
            merged_df = pd.merge(
                df1, df2, 
                on=[x_param, y_param, z_param], 
                suffixes=('_1', '_2'),
                how='inner'
            )
            
            if merged_df.empty:
                print(f"没有匹配的数据点，跳过 {param_name}: {value1} -> {value2}")
                continue
            
            # 计算差异
            merged_df['R_c_diff'] = merged_df['R_c_2'] - merged_df['R_c_1']
            merged_df['Domega_diff'] = merged_df['Domega_2'] - merged_df['Domega_1']
            merged_df['Symmetric_Ratio_diff'] = merged_df['Symmetric_Ratio_2'] - merged_df['Symmetric_Ratio_1']
            
            # 创建3D图形
            fig = plt.figure(figsize=(18, 6))
            
            # R_c 差异3D图
            ax1 = fig.add_subplot(131, projection='3d')
            sc1 = ax1.scatter(merged_df[x_param], merged_df[y_param], merged_df[z_param], 
                             c=merged_df['R_c_diff'], cmap='coolwarm', s=50, alpha=0.8)
            ax1.set_title(f'R_c 差异 ({param_name}: {value1} → {value2})')
            ax1.set_xlabel(x_param)
            ax1.set_ylabel(y_param)
            ax1.set_zlabel(z_param)
            fig.colorbar(sc1, ax=ax1, label='R_c 差异')
            
            # Domega 差异3D图
            ax2 = fig.add_subplot(132, projection='3d')
            sc2 = ax2.scatter(merged_df[x_param], merged_df[y_param], merged_df[z_param], 
                             c=merged_df['Domega_diff'], cmap='coolwarm', s=50, alpha=0.8)
            ax2.set_title(f'ΔΩ 差异 ({param_name}: {value1} → {value2})')
            ax2.set_xlabel(x_param)
            ax2.set_ylabel(y_param)
            ax2.set_zlabel(z_param)
            fig.colorbar(sc2, ax=ax2, label='ΔΩ 差异')
            
            # Symmetric Ratio 差异3D图
            ax3 = fig.add_subplot(133, projection='3d')
            sc3 = ax3.scatter(merged_df[x_param], merged_df[y_param], merged_df[z_param], 
                             c=merged_df['Symmetric_Ratio_diff'], cmap='coolwarm', s=50, alpha=0.8)
            ax3.set_title(f'对称比率差异 ({param_name}: {value1} → {value2})')
            ax3.set_xlabel(x_param)
            ax3.set_ylabel(y_param)
            ax3.set_zlabel(z_param)
            fig.colorbar(sc3, ax=ax3, label='对称比率差异')
            
            plt.tight_layout()
            
            # 保存图像
            diff_dir = os.path.join(save_path, "parameter_differences")
            os.makedirs(diff_dir, exist_ok=True)
            filename = f"diff_{param_name}_{value1}_to_{value2}.png"
            plt.savefig(os.path.join(diff_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存差异图像: {filename}")
            
            # 保存差异数据
            data_filename = f"diff_data_{param_name}_{value1}_to_{value2}.csv"
            merged_df.to_csv(os.path.join(diff_dir, data_filename), index=False)
            print(f"已保存差异数据: {data_filename}")
            
        except Exception as e:
            print(f"处理参数差异时出错 ({param_name}: {value1} -> {value2}): {e}")
    
    return True

def plot_parameter_differences(fixed_params_list):
    """绘制相邻固定参数的3D图数值变化差值（多进程版本）"""
    print("开始绘制参数差值3D图...")
    
    # 按参数类型分组
    param_groups = {}
    for param_name, param_value in fixed_params_list:
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(param_value)
    
    # 对每个参数组内的值排序
    for param_name in param_groups:
        param_groups[param_name].sort()
    
    # 准备多进程任务
    param_tasks = list(param_groups.items())
    
    # 使用进程池并行处理
    print(f"使用多进程处理 {len(param_tasks)} 个参数类型的差异...")
    with Pool(min(4, len(param_tasks))) as p:
        results = p.map(process_parameter_difference, param_tasks)
    
    print("参数差值3D图绘制完成!")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("./3d_phase_diagram", exist_ok=True)
    
    # 定义要测试的固定参数
    fixed_params_to_test = []
    
    # 定义参数范围
    lambda1_values = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ]) 
    lambda2_values = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ])
    d1_values = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])
    d2_values = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])
    
    # 为每个参数添加多个值
    for lambda1 in lambda1_values:
        fixed_params_to_test.append(('strengthLambda1', lambda1))
    
    for lambda2 in lambda2_values:
        fixed_params_to_test.append(('strengthLambda2', lambda2))
    
    for d1 in d1_values:
        fixed_params_to_test.append(('distanceD1', d1))
    
    for d2 in d2_values:
        fixed_params_to_test.append(('distanceD2', d2))
    
    print(f"将处理 {len(fixed_params_to_test)} 个参数组合")
    
    # 添加绘制相邻参数差值的3D图（使用多进程）
    plot_parameter_differences(fixed_params_to_test)
    
    # 串行处理每个固定参数（用于调试）
    for fixed_param in fixed_params_to_test:
        process_fixed_param(fixed_param)
    
    # 如果串行处理正常，可以尝试并行处理
    """
    # 并行处理所有固定参数
    with Pool(min(4, len(fixed_params_to_test))) as p:
        p.map(process_fixed_param, fixed_params_to_test)
    
    # 创建交互式图形（可选）
    for fixed_param in fixed_params_to_test:
        create_interactive_3d_plot(*fixed_param)
    """
    
    print("所有处理完成!")