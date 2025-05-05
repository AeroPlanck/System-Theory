import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import gc
from main import ThreeBody

# 使用numba加速自关联函数计算
@nb.njit
def compute_autocorrelation(time_series, max_lag):
    """
    计算时间序列的自关联函数（numba加速版）
    
    参数:
    time_series: 一维数组，输入时间序列
    max_lag: 最大滞后时间
    
    返回:
    autocorr: 一维数组，自关联函数值
    """
    n = len(time_series)
    mean = np.mean(time_series)
    var = np.var(time_series)
    
    if var == 0:  # 避免除零错误
        return np.zeros(max_lag + 1)
    
    autocorr = np.zeros(max_lag + 1)
    
    # 计算自关联函数
    for lag in range(max_lag + 1):
        # 计算滞后为lag的自关联
        sum_corr = 0.0
        for t in range(n - lag):
            sum_corr += (time_series[t] - mean) * (time_series[t + lag] - mean)
        
        autocorr[lag] = sum_corr / ((n - lag) * var)
    
    return autocorr

# 使用numba加速批量计算多个振子的自关联函数
@nb.njit(parallel=True)
def compute_batch_autocorrelation(data, max_lag):
    """
    批量计算多个振子的自关联函数（numba加速版）
    
    参数:
    data: 二维数组，每行是一个振子的时间序列
    max_lag: 最大滞后时间
    
    返回:
    autocorrs: 二维数组，每行是一个振子的自关联函数
    """
    n_agents = data.shape[0]
    autocorrs = np.zeros((n_agents, max_lag + 1))
    
    for i in nb.prange(n_agents):
        autocorrs[i] = compute_autocorrelation(data[i], max_lag)
    
    return autocorrs

def load_model_data_chunk(model, start_idx, end_idx):
    """
    加载模型数据的指定时间段
    
    参数:
    model: ThreeBody模型实例
    start_idx: 起始时间索引
    end_idx: 结束时间索引
    
    返回:
    chunk_data: 包含指定时间段数据的字典
    """
    try:
        targetPath = f"./data/{model}.h5"
        
        # 读取数据的总行数以确定TNum和agentsNum
        totalPositionX_info = pd.read_hdf(targetPath, key="positionX", start=0, stop=1)
        total_rows = pd.read_hdf(targetPath, key="positionX").shape[0]
        agentsNum = model.agentsNum
        TNum = total_rows // agentsNum
        
        # 确保索引在有效范围内
        start_idx = max(0, min(start_idx, TNum - 1))
        end_idx = max(start_idx + 1, min(end_idx, TNum))
        
        # 计算HDF5文件中的实际行索引
        start_row = start_idx * agentsNum
        end_row = end_idx * agentsNum
        
        # 读取指定范围的数据
        chunk_positionX = pd.read_hdf(targetPath, key="positionX", start=start_row, stop=end_row).values
        chunk_phaseTheta = pd.read_hdf(targetPath, key="phaseTheta", start=start_row, stop=end_row).values
        chunk_pointTheta = pd.read_hdf(targetPath, key="pointTheta", start=start_row, stop=end_row).values
        
        # 重塑数据
        chunk_size = end_idx - start_idx
        positionX = chunk_positionX.reshape(chunk_size, agentsNum, 2)
        phaseTheta = chunk_phaseTheta.reshape(chunk_size, agentsNum)
        pointTheta = chunk_pointTheta.reshape(chunk_size, agentsNum)
        
        # 计算速度（位移的时间导数）
        velocity = np.zeros_like(positionX)
        if chunk_size > 1:
            velocity[1:] = (positionX[1:] - positionX[:-1]) / model.dt
        
        # 相位速度已经存在于pointTheta中
        phase_velocity = pointTheta
        
        chunk_data = {
            'positionX': positionX,
            'velocity': velocity,
            'phaseTheta': phaseTheta,
            'phase_velocity': phase_velocity,
            'chunk_size': chunk_size,
            'agentsNum': agentsNum,
            'dt': model.dt,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'TNum': TNum
        }
        
        return chunk_data
    except Exception as e:
        print(f"加载数据块出错 ({model}, {start_idx}-{end_idx}): {e}")
        return None

def compute_time_window_autocorrelation_chunked(model, window_size=100, max_lag=50, step=10, chunk_size=500):
    """
    分块计算时间窗口的自关联函数，以减少内存使用
    
    参数:
    model: ThreeBody模型实例
    window_size: 每个时间窗口的大小
    max_lag: 最大滞后时间
    step: 时间窗口的步长
    chunk_size: 每次处理的时间步数量
    
    返回:
    results: 包含各时间窗口自关联函数的字典
    """
    try:
        # 获取数据总时间步数
        targetPath = f"./data/{model}.h5"
        total_rows = pd.read_hdf(targetPath, key="positionX").shape[0]
        agentsNum = model.agentsNum
        TNum = total_rows // agentsNum
        
        print(f"模型 {model} 总时间步数: {TNum}, 振子数量: {agentsNum}")
        
        # 确定时间窗口数量
        n_windows = max(1, (TNum - window_size) // step + 1)
        print(f"将计算 {n_windows} 个时间窗口的自关联函数")
        
        # 初始化结果字典
        results = {
            'position_x': np.zeros((n_windows, agentsNum, max_lag + 1)),
            'position_y': np.zeros((n_windows, agentsNum, max_lag + 1)),
            'velocity_x': np.zeros((n_windows, agentsNum, max_lag + 1)),
            'velocity_y': np.zeros((n_windows, agentsNum, max_lag + 1)),
            'phase': np.zeros((n_windows, agentsNum, max_lag + 1)),
            'phase_velocity': np.zeros((n_windows, agentsNum, max_lag + 1)),
            'window_starts': np.zeros(n_windows, dtype=int),
            'window_times': np.zeros(n_windows)
        }
        
        # 确定需要处理的数据块
        chunks = []
        for w in range(n_windows):
            start_idx = w * step
            end_idx = min(start_idx + window_size, TNum)
            
            # 记录窗口起始位置和对应的模拟时间
            results['window_starts'][w] = start_idx
            results['window_times'][w] = start_idx * model.dt
            
            # 确定包含此窗口的数据块
            chunk_start = (start_idx // chunk_size) * chunk_size
            chunk_end = min(((end_idx - 1) // chunk_size + 1) * chunk_size, TNum)
            
            chunks.append((chunk_start, chunk_end, w, start_idx, end_idx))
        
        # 合并重叠的数据块以减少I/O操作
        merged_chunks = []
        if chunks:
            current_chunk = chunks[0]
            for next_chunk in chunks[1:]:
                if next_chunk[0] <= current_chunk[1]:
                    # 合并重叠的块
                    current_chunk = (current_chunk[0], max(current_chunk[1], next_chunk[1]), 
                                    [current_chunk[2], next_chunk[2]], 
                                    [current_chunk[3], next_chunk[3]], 
                                    [current_chunk[4], next_chunk[4]])
                else:
                    # 添加当前块并开始新块
                    if isinstance(current_chunk[2], list):
                        merged_chunks.append(current_chunk)
                    else:
                        merged_chunks.append((current_chunk[0], current_chunk[1], 
                                            [current_chunk[2]], 
                                            [current_chunk[3]], 
                                            [current_chunk[4]]))
                    current_chunk = next_chunk
            
            # 添加最后一个块
            if isinstance(current_chunk[2], list):
                merged_chunks.append(current_chunk)
            else:
                merged_chunks.append((current_chunk[0], current_chunk[1], 
                                    [current_chunk[2]], 
                                    [current_chunk[3]], 
                                    [current_chunk[4]]))
        
        print(f"合并后需要处理 {len(merged_chunks)} 个数据块")
        
        # 处理每个合并后的数据块
        for chunk_start, chunk_end, window_indices, start_indices, end_indices in tqdm(merged_chunks, desc="处理数据块"):
            # 加载数据块
            chunk_data = load_model_data_chunk(model, chunk_start, chunk_end)
            if chunk_data is None:
                continue
            
            # 处理此块中的每个窗口
            for w_idx, start_idx, end_idx in zip(window_indices, start_indices, end_indices):
                # 调整为块内相对索引
                rel_start = start_idx - chunk_start
                rel_end = end_idx - chunk_start
                
                if rel_end - rel_start < window_size // 2:  # 窗口太小，跳过
                    continue
                
                # 提取当前窗口数据并转置为(agents, time)格式
                pos_x = chunk_data['positionX'][rel_start:rel_end, :, 0].T
                pos_y = chunk_data['positionX'][rel_start:rel_end, :, 1].T
                vel_x = chunk_data['velocity'][rel_start:rel_end, :, 0].T
                vel_y = chunk_data['velocity'][rel_start:rel_end, :, 1].T
                phase = chunk_data['phaseTheta'][rel_start:rel_end].T
                phase_vel = chunk_data['phase_velocity'][rel_start:rel_end].T
                
                # 批量计算自关联函数
                results['position_x'][w_idx] = compute_batch_autocorrelation(pos_x, max_lag)
                results['position_y'][w_idx] = compute_batch_autocorrelation(pos_y, max_lag)
                results['velocity_x'][w_idx] = compute_batch_autocorrelation(vel_x, max_lag)
                results['velocity_y'][w_idx] = compute_batch_autocorrelation(vel_y, max_lag)
                results['phase'][w_idx] = compute_batch_autocorrelation(phase, max_lag)
                results['phase_velocity'][w_idx] = compute_batch_autocorrelation(phase_vel, max_lag)
            
            # 清理内存
            del chunk_data
            gc.collect()
        
        return results
    except Exception as e:
        print(f"计算自关联函数出错 ({model}): {e}")
        return None

def plot_autocorrelation_evolution(model, results, save_path="./autocorrelation_plots", show_agents=5):
    """
    绘制自关联函数随时间的演化图
    
    参数:
    model: ThreeBody模型实例
    results: 自关联函数计算结果
    save_path: 图像保存路径
    show_agents: 展示的单个振子数量
    """
    if results is None:
        print(f"无法为模型 {model} 绘制自关联函数演化图：数据为空")
        return False
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 获取时间窗口数量和最大滞后时间
    n_windows = len(results['window_starts'])
    max_lag = results['position_x'].shape[2] - 1
    
    # 创建时间轴（以时间步为单位）
    lag_times = np.arange(max_lag + 1) * model.dt
    
    # 计算每个变量的平均自关联函数（所有振子的平均）
    avg_results = {}
    for key in ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'phase', 'phase_velocity']:
        avg_results[key] = np.mean(results[key], axis=1)  # 对所有振子取平均
    
    # 绘制演化图
    plt.figure(figsize=(20, 15))
    
    # 创建2x3网格的子图
    variables = [
        ('position_x', '位移 X'),
        ('position_y', '位移 Y'),
        ('velocity_x', '速度 X'),
        ('velocity_y', '速度 Y'),
        ('phase', '相位'),
        ('phase_velocity', '相位速度')
    ]
    
    for i, (var_key, var_name) in enumerate(variables):
        plt.subplot(2, 3, i+1)
        
        # 创建颜色映射
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=n_windows-1)
        
        # 绘制每个时间窗口的自关联函数
        for w in range(n_windows):
            color = cmap(norm(w))
            alpha = 0.7 if w == 0 or w == n_windows-1 else 0.3
            plt.plot(lag_times, avg_results[var_key][w], 
                     color=color, alpha=alpha, 
                     label=f'窗口 {results["window_starts"][w]}' if (w == 0 or w == n_windows-1) else None)
        
        plt.title(f'{var_name}的自关联函数演化', fontsize=14)
        plt.xlabel('滞后时间', fontsize=12)
        plt.ylabel('自关联函数', fontsize=12)
        plt.grid(True, alpha=0.3)
        if i == 0 or i == 3:  # 只在左侧子图显示图例
            plt.legend(loc='upper right', fontsize=10)
    
    # 添加总标题
    plt.suptitle(f'三体系统自关联函数演化 (λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2})', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    filename = f"autocorr_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存自关联函数演化图: {filename}")
    
    # 绘制热图展示自关联函数随时间的变化
    plt.figure(figsize=(20, 15))
    
    for i, (var_key, var_name) in enumerate(variables):
        plt.subplot(2, 3, i+1)
        
        # 创建热图数据
        heatmap_data = avg_results[var_key]
        
        # 绘制热图
        im = plt.imshow(heatmap_data, 
                       aspect='auto',
                       origin='lower',
                       extent=[0, max_lag*model.dt, 0, results['window_times'][-1]],
                       cmap='viridis')
        
        plt.colorbar(im, label='自关联函数值')
        plt.title(f'{var_name}的自关联函数热图', fontsize=14)
        plt.xlabel('滞后时间', fontsize=12)
        plt.ylabel('模拟时间', fontsize=12)
    
    # 添加总标题
    plt.suptitle(f'三体系统自关联函数热图 (λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2})', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存热图
    heatmap_filename = f"autocorr_heatmap_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png"
    plt.savefig(os.path.join(save_path, heatmap_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存自关联函数热图: {heatmap_filename}")
    
    # 绘制单个振子的自关联函数
    plt.figure(figsize=(20, 15))
    
    for i, (var_key, var_name) in enumerate(variables):
        plt.subplot(2, 3, i+1)
        
        # 选择最后一个时间窗口和指定数量的振子
        n_agents_to_show = min(show_agents, results[var_key].shape[1])
        for agent_idx in range(n_agents_to_show):
            plt.plot(lag_times, results[var_key][-1, agent_idx], 
                     label=f'振子 {agent_idx+1}')
        
        plt.title(f'{var_name}的单振子自关联函数', fontsize=14)
        plt.xlabel('滞后时间', fontsize=12)
        plt.ylabel('自关联函数', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
    
    # 添加总标题
    plt.suptitle(f'三体系统单振子自关联函数 (λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2})', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存单振子图
    agent_filename = f"autocorr_agents_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png"
    plt.savefig(os.path.join(save_path, agent_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存单振子自关联函数图: {agent_filename}")
    
    # 绘制自关联函数衰减时间分析图
    plt.figure(figsize=(20, 10))
    
    # 计算自关联函数衰减到0.5和0.1的时间
    decay_times_50 = {}
    decay_times_10 = {}
    
    for var_key, var_name in variables:
        decay_times_50[var_key] = []
        decay_times_10[var_key] = []
        
        for w in range(n_windows):
            autocorr = avg_results[var_key][w]
            # 找到自关联函数首次低于0.5的时间
            try:
                idx_50 = np.where(autocorr < 0.5)[0][0]
                decay_times_50[var_key].append(lag_times[idx_50])
            except IndexError:
                decay_times_50[var_key].append(lag_times[-1])  # 如果没有衰减到0.5，使用最大滞后时间
            
            # 找到自关联函数首次低于0.1的时间
            try:
                idx_10 = np.where(autocorr < 0.1)[0][0]
                decay_times_10[var_key].append(lag_times[idx_10])
            except IndexError:
                decay_times_10[var_key].append(lag_times[-1])  # 如果没有衰减到0.1，使用最大滞后时间
    
    # 绘制衰减时间随模拟时间的变化
    plt.subplot(1, 2, 1)
    for var_key, var_name in variables:
        plt.plot(results['window_times'], decay_times_50[var_key], 'o-', label=var_name)
    
    plt.title('自关联函数衰减到0.5的时间', fontsize=14)
    plt.xlabel('模拟时间', fontsize=12)
    plt.ylabel('衰减时间', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    plt.subplot(1, 2, 2)
    for var_key, var_name in variables:
        plt.plot(results['window_times'], decay_times_10[var_key], 'o-', label=var_name)
    
    plt.title('自关联函数衰减到0.1的时间', fontsize=14)
    plt.xlabel('模拟时间', fontsize=12)
    plt.ylabel('衰减时间', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # 添加总标题
    plt.suptitle(f'三体系统自关联函数衰减时间分析 (λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2})', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存衰减时间分析图
    decay_filename = f"autocorr_decay_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png"
    plt.savefig(os.path.join(save_path, decay_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存自关联函数衰减时间分析图: {decay_filename}")
    
    return True

def analyze_single_model(l1, l2, d1, d2, window_size=100, max_lag=50, step=20, chunk_size=500, save_path="./autocorrelation_plots", show_agents=5):
    """
    分析单个模型的自关联函数
    
    参数:
    l1, l2, d1, d2: 模型参数
    window_size: 每个时间窗口的大小
    max_lag: 最大滞后时间
    step: 时间窗口的步长
    chunk_size: 每次处理的时间步数量
    save_path: 图像保存路径
    show_agents: 展示的单个振子数量
    """
    try:
        # 创建模型
        model = ThreeBody(l1, l2, d1, d2, agentsNum=200, boundaryLength=5,
                        tqdm=False, overWrite=False)
        
        print(f"分析模型: λ1={l1}, λ2={l2}, d1={d1}, d2={d2}")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 计算自关联函数
        results = compute_time_window_autocorrelation_chunked(
            model, window_size, max_lag, step, chunk_size)
        
        # 绘制演化图
        success = plot_autocorrelation_evolution(model, results, save_path, show_agents)
        
        # 清理内存
        del results
        gc.collect()
        
        return success
    except Exception as e:
        print(f"分析模型出错 (λ1={l1}, λ2={l2}, d1={d1}, d2={d2}): {e}")
        return False

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析三体系统振子的自关联函数')
    parser.add_argument('--l1', type=float, default=0.5, help='参数λ1的值')
    parser.add_argument('--l2', type=float, default=0.5, help='参数λ2的值')
    parser.add_argument('--d1', type=float, default=0.2, help='参数d1的值')
    parser.add_argument('--d2', type=float, default=0.6, help='参数d2的值')
    parser.add_argument('--window_size', type=int, default=100, help='时间窗口大小')
    parser.add_argument('--max_lag', type=int, default=50, help='最大滞后时间')
    parser.add_argument('--step', type=int, default=20, help='时间窗口步长')
    parser.add_argument('--chunk_size', type=int, default=500, help='数据块大小')
    parser.add_argument('--save_path', type=str, default='./autocorrelation_plots', help='图像保存路径')
    parser.add_argument('--show_agents', type=int, default=5, help='展示的单个振子数量')
    
    args = parser.parse_args()
    
    # 分析单个模型
    success = analyze_single_model(
        args.l1, args.l2, args.d1, args.d2,
        args.window_size, args.max_lag, args.step, args.chunk_size,
        args.save_path, args.show_agents
    )
    
    if success:
        print("分析完成!")
    else:
        print("分析失败!")