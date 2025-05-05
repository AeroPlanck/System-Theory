import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial
from main import ThreeBody

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

def load_model_data(model):
    """
    加载模型数据并进行预处理
    
    参数:
    model: ThreeBody模型实例
    
    返回:
    data: 包含预处理数据的字典
    """
    model_key = str(model)
    if model_key in model_data_cache:
        return model_data_cache[model_key]
    
    try:
        targetPath = f"./data/{model}.h5"
        totalPositionX = pd.read_hdf(targetPath, key="positionX").values
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta").values
        totalPointTheta = pd.read_hdf(targetPath, key="pointTheta").values

        TNum, agentsNum = totalPositionX.shape[0]//model.agentsNum, model.agentsNum
        
        # 重塑数据
        positionX = totalPositionX.reshape(TNum, agentsNum, 2)
        phaseTheta = totalPhaseTheta.reshape(TNum, agentsNum)
        pointTheta = totalPointTheta.reshape(TNum, agentsNum)
        
        # 计算速度（位移的时间导数）
        velocity = np.zeros_like(positionX)
        velocity[1:] = (positionX[1:] - positionX[:-1]) / model.dt
        
        # 相位速度已经存在于pointTheta中
        phase_velocity = pointTheta
        
        data = {
            'positionX': positionX,
            'velocity': velocity,
            'phaseTheta': phaseTheta,
            'phase_velocity': phase_velocity,
            'TNum': TNum,
            'agentsNum': agentsNum,
            'dt': model.dt
        }
        
        # 存入缓存
        model_data_cache[model_key] = data
        return data
    except Exception as e:
        print(f"加载数据出错 ({model}): {e}")
        return None

def compute_time_window_autocorrelation(model, window_size=100, max_lag=50, step=10):
    """
    计算分时间窗口的自关联函数
    
    参数:
    model: ThreeBody模型实例
    window_size: 每个时间窗口的大小
    max_lag: 最大滞后时间
    step: 时间窗口的步长
    
    返回:
    results: 包含各时间窗口自关联函数的字典
    """
    # 加载数据
    data = load_model_data(model)
    if data is None:
        return None
    
    TNum = data['TNum']
    agentsNum = data['agentsNum']
    
    # 确定时间窗口数量
    n_windows = max(1, (TNum - window_size) // step + 1)
    
    # 初始化结果字典
    results = {
        'position_x': np.zeros((n_windows, agentsNum, max_lag + 1)),
        'position_y': np.zeros((n_windows, agentsNum, max_lag + 1)),
        'velocity_x': np.zeros((n_windows, agentsNum, max_lag + 1)),
        'velocity_y': np.zeros((n_windows, agentsNum, max_lag + 1)),
        'phase': np.zeros((n_windows, agentsNum, max_lag + 1)),
        'phase_velocity': np.zeros((n_windows, agentsNum, max_lag + 1)),
        'window_starts': np.zeros(n_windows, dtype=int)
    }
    
    # 逐个时间窗口计算
    for w in range(n_windows):
        start_idx = w * step
        end_idx = min(start_idx + window_size, TNum)
        
        if end_idx - start_idx < window_size // 2:  # 窗口太小，跳过
            continue
        
        results['window_starts'][w] = start_idx
        
        # 提取当前窗口数据并转置为(agents, time)格式
        pos_x = data['positionX'][start_idx:end_idx, :, 0].T
        pos_y = data['positionX'][start_idx:end_idx, :, 1].T
        vel_x = data['velocity'][start_idx:end_idx, :, 0].T
        vel_y = data['velocity'][start_idx:end_idx, :, 1].T
        phase = data['phaseTheta'][start_idx:end_idx].T
        phase_vel = data['phase_velocity'][start_idx:end_idx].T
        
        # 批量计算自关联函数
        results['position_x'][w] = compute_batch_autocorrelation(pos_x, max_lag)
        results['position_y'][w] = compute_batch_autocorrelation(pos_y, max_lag)
        results['velocity_x'][w] = compute_batch_autocorrelation(vel_x, max_lag)
        results['velocity_y'][w] = compute_batch_autocorrelation(vel_y, max_lag)
        results['phase'][w] = compute_batch_autocorrelation(phase, max_lag)
        results['phase_velocity'][w] = compute_batch_autocorrelation(phase_vel, max_lag)
    
    return results

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
        return
    
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
                       extent=[0, max_lag*model.dt, 0, n_windows*model.dt*10],
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
    
    return True

def process_model(model, window_size=100, max_lag=50, step=10, save_path="./autocorrelation_plots"):
    """
    处理单个模型的自关联函数计算和可视化
    
    参数:
    model: ThreeBody模型实例
    window_size: 每个时间窗口的大小
    max_lag: 最大滞后时间
    step: 时间窗口的步长
    save_path: 图像保存路径
    
    返回:
    success: 是否成功处理
    """
    try:
        print(f"处理模型: λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, d1={model.distanceD1}, d2={model.distanceD2}")
        
        # 计算自关联函数
        results = compute_time_window_autocorrelation(model, window_size, max_lag, step)
        
        # 绘制演化图
        success = plot_autocorrelation_evolution(model, results, save_path)
        
        return success
    except Exception as e:
        print(f"处理模型出错 ({model}): {e}")
        return False

if __name__ == "__main__":
    # 设置参数
    save_path = "./autocorrelation_plots"
    os.makedirs(save_path, exist_ok=True)
    
    # 使用较小的参数范围以加快测试
    rangeLambdas = np.array([0.1, 0.5, 0.9])
    distanceDs = np.array([0.2, 0.6])
    
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
    
    # 使用多进程处理模型
    process_func = partial(process_model, 
                          window_size=window_size, 
                          max_lag=max_lag, 
                          step=step, 
                          save_path=save_path)
    
    print(f"开始计算自关联函数，共 {len(models)} 个模型...")
    
    # 使用多进程加速计算
    with Pool(min(len(models), os.cpu_count())) as p:
        results = list(tqdm(p.imap(process_func, models), total=len(models), desc="处理模型"))
    
    # 检查结果
    success_count = sum(1 for r in results if r)
    print(f"所有自关联函数计算已完成! 成功: {success_count}/{len(models)}")