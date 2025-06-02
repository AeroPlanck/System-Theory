import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from itertools import product
from multiprocessing import Pool
from functools import partial
from main import ThreeBody
import gc

# 创建数据缓存字典
model_data_cache = {}

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
            'end_idx': end_idx
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
                    # 确保索引列表的一致性
                    w_indices = [current_chunk[2]] if not isinstance(current_chunk[2], list) else current_chunk[2]
                    s_indices = [current_chunk[3]] if not isinstance(current_chunk[3], list) else current_chunk[3]
                    e_indices = [current_chunk[4]] if not isinstance(current_chunk[4], list) else current_chunk[4]
                    
                    # 添加下一个块的索引
                    w_indices.append(next_chunk[2])
                    s_indices.append(next_chunk[3])
                    e_indices.append(next_chunk[4])
                    
                    current_chunk = (current_chunk[0], max(current_chunk[1], next_chunk[1]), 
                                    w_indices, s_indices, e_indices)
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
            
            # 确保所有索引都是列表形式
            if not isinstance(window_indices, list):
                window_indices = [window_indices]
            if not isinstance(start_indices, list):
                start_indices = [start_indices]
            if not isinstance(end_indices, list):
                end_indices = [end_indices]
            
            # 处理此块中的每个窗口
            for i in range(len(window_indices)):
                # 确保索引是整数而不是列表
                w_idx = window_indices[i]
                start_idx = start_indices[i]
                end_idx = end_indices[i]
                
                # 如果索引仍然是列表，则取第一个元素
                if isinstance(w_idx, list):
                    print(f"警告: w_idx 是列表 {w_idx}，取第一个元素")
                    w_idx = w_idx[0]
                if isinstance(start_idx, list):
                    print(f"警告: start_idx 是列表 {start_idx}，取第一个元素")
                    start_idx = start_idx[0]
                if isinstance(end_idx, list):
                    print(f"警告: end_idx 是列表 {end_idx}，取第一个元素")
                    end_idx = end_idx[0]
                
                # 调整为块内相对索引
                try:
                    rel_start = int(start_idx - chunk_start)
                    rel_end = int(end_idx - chunk_start)
                    
                    if rel_end - rel_start < window_size // 2:  # 窗口太小，跳过
                        print(f"跳过窗口 {w_idx}，窗口大小过小: {rel_end - rel_start} < {window_size // 2}")
                        continue
                    
                    if rel_start < 0 or rel_end > chunk_data['chunk_size']:
                        print(f"警告: 窗口 {w_idx} 索引超出范围: rel_start={rel_start}, rel_end={rel_end}, chunk_size={chunk_data['chunk_size']}")
                        rel_start = max(0, rel_start)
                        rel_end = min(rel_end, chunk_data['chunk_size'])
                    
                    # 提取当前窗口数据并转置为(agents, time)格式
                    pos_x = chunk_data['positionX'][rel_start:rel_end, :, 0].T
                    pos_y = chunk_data['positionX'][rel_start:rel_end, :, 1].T
                    vel_x = chunk_data['velocity'][rel_start:rel_end, :, 0].T
                    vel_y = chunk_data['velocity'][rel_start:rel_end, :, 1].T
                    phase = chunk_data['phaseTheta'][rel_start:rel_end].T
                    phase_vel = chunk_data['phase_velocity'][rel_start:rel_end].T
                except Exception as e:
                    print(f"处理窗口 {w_idx} 出错: {e}")
                    print(f"调试信息: start_idx={start_idx}, end_idx={end_idx}, chunk_start={chunk_start}, chunk_end={chunk_end}")
                    continue
                
                # 批量计算自关联函数
                try:
                    # 检查数据维度是否正确
                    if pos_x.shape[0] != agentsNum or pos_x.shape[1] < 2:
                        print(f"警告: 窗口 {w_idx} 数据维度不正确: {pos_x.shape}, 预期第一维为 {agentsNum}")
                        continue
                        
                    results['position_x'][w_idx] = compute_batch_autocorrelation(pos_x, max_lag)
                    results['position_y'][w_idx] = compute_batch_autocorrelation(pos_y, max_lag)
                    results['velocity_x'][w_idx] = compute_batch_autocorrelation(vel_x, max_lag)
                    results['velocity_y'][w_idx] = compute_batch_autocorrelation(vel_y, max_lag)
                    results['phase'][w_idx] = compute_batch_autocorrelation(phase, max_lag)
                    results['phase_velocity'][w_idx] = compute_batch_autocorrelation(phase_vel, max_lag)
                    print(f"成功计算窗口 {w_idx} 的自关联函数")
                except Exception as e:
                    print(f"计算窗口 {w_idx} 的自关联函数出错: {e}")
                    print(f"数据形状: pos_x={pos_x.shape}, phase={phase.shape}")
                    continue
            
            # 清理内存
            del chunk_data
            gc.collect()
        
        return results
    except Exception as e:
        print(f"计算自关联函数出错 ({model}): {e}")
        return None

def plot_autocorrelation_evolution(model, results, save_path="./autocorrelation_plots"):
    """
    绘制自关联函数随时间的演化图
    
    参数:
    model: ThreeBody模型实例
    results: 自关联函数计算结果
    save_path: 图像保存路径
    """
    if results is None:
        print(f"无法为模型 {model} 绘制自关联函数演化图：数据为空")
        return False
    
    # 检查结果字典中是否包含所有必要的键
    required_keys = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'phase', 'phase_velocity', 'window_starts']
    for key in required_keys:
        if key not in results or results[key] is None or results[key].size == 0:
            print(f"无法为模型 {model} 绘制自关联函数演化图：缺少必要的数据键 {key}")
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
        ('position_x', 'Transverse Displacement'),
        ('position_y', 'Longitudinal Displacement'),
        ('velocity_x', 'Transverse Velocity'),
        ('velocity_y', 'Longitudinal Velocity'),
        ('phase', 'Phase'),
        ('phase_velocity', 'Phase Velocity')
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
        
        plt.title(f'{var_name}', fontsize=14)
        plt.xlabel('Lag Time', fontsize=12)
        plt.ylabel('Auto Correlation', fontsize=12)
        plt.grid(True, alpha=0.3)
        if i == 0 or i == 3:  # 只在左侧子图显示图例
            plt.legend(loc='upper right', fontsize=10)
    
    # 添加总标题
    plt.suptitle(f'(λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2})', 
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
                       extent=[0, max_lag*model.dt, 0, n_windows*model.dt*10],
                       cmap='viridis')
        
        plt.colorbar(im, label='Auto Correlation')
        plt.title(f'Heat Map of {var_name}', fontsize=14)
        plt.xlabel('Lag Time', fontsize=12)
        plt.ylabel('Evolution Time', fontsize=12)
    
    # 添加总标题
    plt.suptitle(f'(λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2})', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存热图
    heatmap_filename = f"autocorr_heatmap_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png"
    plt.savefig(os.path.join(save_path, heatmap_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存自关联函数热图: {heatmap_filename}")
    
    return True

def process_model(model, window_size=100, max_lag=50, step=10, chunk_size=500, save_path="./autocorrelation_plots"):
    """
    处理单个模型的自关联函数计算和可视化
    
    参数:
    model: ThreeBody模型实例
    window_size: 每个时间窗口的大小
    max_lag: 最大滞后时间
    step: 时间窗口的步长
    chunk_size: 每次处理的时间步数量
    save_path: 图像保存路径
    
    返回:
    success: 是否成功处理
    """
    try:
        model_info = f"λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2}"
        print(f"处理模型: {model_info}")
        
        # 检查结果文件是否已存在
        result_files = [
            f"autocorr_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png",
            f"autocorr_heatmap_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png",
            f"autocorr_agents_l1_{model.strengthLambda1}_l2_{model.strengthLambda2}_d1_{model.distanceD1}_d2_{model.distanceD2}.png"
        ]
        
        all_exist = all(os.path.exists(os.path.join(save_path, f)) for f in result_files)
        if all_exist:
            print(f"跳过已处理的模型: {model_info}")
            return True
        
        # 检查数据文件是否存在
        data_file = f"./data/{model}.h5"
        if not os.path.exists(data_file):
            print(f"数据文件不存在: {data_file}")
            return False
        
        # 计算自关联函数
        print(f"开始计算模型 {model_info} 的自关联函数...")
        results = compute_time_window_autocorrelation_chunked(
            model, window_size, max_lag, step, chunk_size)
        
        if results is None:
            print(f"模型 {model_info} 的自关联函数计算失败")
            return False
        
        # 检查结果是否有效
        if not all(key in results and results[key] is not None and results[key].size > 0 
                  for key in ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'phase', 'phase_velocity']):
            print(f"模型 {model_info} 的自关联函数计算结果不完整")
            return False
        
        # 绘制演化图
        print(f"开始绘制模型 {model_info} 的自关联函数演化图...")
        success = plot_autocorrelation_evolution(model, results, save_path)
        
        # 清理内存
        del results
        gc.collect()
        
        if success:
            print(f"模型 {model_info} 处理成功")
        else:
            print(f"模型 {model_info} 绘图失败")
        
        return success
    except Exception as e:
        print(f"处理模型出错 ({model}): {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置参数
    save_path = "./autocorrelation_plots"
    os.makedirs(save_path, exist_ok=True)
    
    # 使用较小的参数范围以加快测试
    rangeLambdas = np.array([0.1, 0.5, 0.9])
    distanceDs = np.array([0.1, 0.5])
    
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
    
    # 设置自关联函数计算参数
    window_size = 100  # 时间窗口大小
    max_lag = 50       # 最大滞后时间
    step = 20          # 时间窗口步长
    chunk_size = 500   # 数据块大小
    
    # 使用多进程处理模型
    process_func = partial(process_model, 
                          window_size=window_size, 
                          max_lag=max_lag, 
                          step=step, 
                          chunk_size=chunk_size,
                          save_path=save_path)
    
    print(f"开始计算自关联函数，共 {len(models)} 个模型...")
    
    # 使用多进程加速计算
    with Pool(min(len(models), os.cpu_count())) as p:
        results = list(tqdm(p.imap(process_func, models), total=len(models), desc="处理模型"))
    
    # 检查结果
    success_count = sum(1 for r in results if r)
    print(f"所有自关联函数计算已完成! 成功: {success_count}/{len(models)}")