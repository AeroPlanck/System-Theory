import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from tqdm import tqdm
import os
from main import ThreeBody

def quick_visualize_K_tensors(model_params=None, data_file=None):
    """
    快速可视化K1和K2张量的演化
    
    Parameters:
    -----------
    model_params : dict, optional
        模型参数字典，包含 strengthLambda1, strengthLambda2, distanceD1, distanceD2
    data_file : str, optional
        指定数据文件路径，如果不提供则使用默认模型参数
    """
    
    # 设置默认参数
    if model_params is None:
        model_params = {
            'strengthLambda1': 0.1,
            'strengthLambda2': 0.1, 
            'distanceD1': 0.5,
            'distanceD2': 0.5,
            'agentsNum': 200,
            'boundaryLength': 5,
            'randomSeed': 10
        }
    
    # 创建模型实例
    model = ThreeBody(
        strengthLambda1=model_params['strengthLambda1'],
        strengthLambda2=model_params['strengthLambda2'],
        distanceD1=model_params['distanceD1'],
        distanceD2=model_params['distanceD2'],
        agentsNum=model_params['agentsNum'],
        boundaryLength=model_params['boundaryLength'],
        randomSeed=model_params['randomSeed'],
        tqdm=False,
        overWrite=False
    )
    
    print(f"正在处理模型: {model}")
    
    # 加载数据
    if data_file is None:
        targetPath = f"./data/{model}.h5"
    else:
        targetPath = data_file
        
    if not os.path.exists(targetPath):
        print(f"数据文件不存在: {targetPath}")
        print("请先运行模型生成数据，或指定正确的数据文件路径")
        return
    
    try:
        # 使用用户指定的数据读取方法
        print("正在加载数据文件...")
        with tqdm(total=3, desc="加载HDF5数据") as pbar:
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            pbar.update(1)
            
            totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
            pbar.update(1)
            
            totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
            pbar.update(1)
        
        print("正在重塑数据格式...")
        TNum = totalPositionX.shape[0] // model.agentsNum
        with tqdm(total=3, desc="重塑数据格式") as pbar:
            positionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
            pbar.update(1)
            
            phaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
            pbar.update(1)
            
            pointTheta = totalPointTheta.values.reshape(TNum, model.agentsNum)
            pbar.update(1)
        
        print(f"数据加载成功: TNum={TNum}, agentsNum={model.agentsNum}")
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return
    
    # 选择几个时间点进行分析
    time_points = [max(0, TNum-20), max(0, TNum-10), TNum-1]  # 最后几个时间点
    
    # 创建保存目录
    save_dir = "./visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n开始计算和可视化K1、K2张量...")
    
    # 为每个时间点计算和可视化
    for t_idx, t in tqdm(enumerate(time_points), total=len(time_points), desc="处理时间点"):
        print(f"\n处理时间点 t={t} ({t_idx+1}/{len(time_points)})")
        
        # 获取当前时刻的位置
        curr_pos = positionX[t]
        
        # 计算K1和K2
        print("  计算K1和K2张量...")
        K1_t, K2_t = compute_K_tensors_at_time(curr_pos, model)
        
        # 可视化K1 (散点图)
        print("  生成K1散点图...")
        visualize_K1_scatter(K1_t, t, model, save_dir)
        
        # 可视化K2 (3D散点图和切片)
        print("  生成K2结构图...")
        visualize_K2_structure(K2_t, t, model, save_dir)
        
        # 分析连接统计
        analyze_connections(K1_t, K2_t, t, model)
    
    # 创建时间演化分析
    print("\n生成时间演化分析...")
    create_temporal_analysis(positionX, model, time_points, save_dir)
    
    print(f"\n所有可视化完成！结果保存在 {save_dir} 目录中")

def compute_K_tensors_at_time(positions, model):
    """
    计算指定时刻的K1和K2张量
    """
    agentsNum = model.agentsNum
    boundaryLength = model.boundaryLength
    halfBoundary = boundaryLength / 2
    
    # 计算周期性边界条件下的距离矩阵
    with tqdm(total=4, desc="    计算距离矩阵", leave=False) as pbar:
        pos_i = positions[:, np.newaxis, :]
        pos_j = positions[np.newaxis, :, :]
        pbar.update(1)
        
        # 周期性距离计算
        delta = pos_i - pos_j
        pbar.update(1)
        
        delta = np.where(delta > halfBoundary, delta - boundaryLength, delta)
        delta = np.where(delta < -halfBoundary, delta + boundaryLength, delta)
        pbar.update(1)
        
        distances = np.sqrt(np.sum(delta**2, axis=2))
        pbar.update(1)
    
    # 排除自连接
    eye_mask = ~np.eye(agentsNum, dtype=bool)
    
    with tqdm(total=2, desc="    计算K1和K2", leave=False) as pbar:
        # 计算K1 (邻接矩阵)
        K1 = (distances <= model.distanceD1) & eye_mask
        pbar.update(1)
        
        # 计算K2 (邻接张量)
        K2_base = (distances <= model.distanceD2) & eye_mask
        K2 = K2_base[:, :, np.newaxis] & K2_base[:, np.newaxis, :]
        pbar.update(1)
    
    return K1, K2

def visualize_K1_scatter(K1, time_step, model, save_dir):
    """
    可视化K1邻接矩阵的散点图
    """
    plt.figure(figsize=(10, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 找到所有连接的位置 (值为1的位置)
    connections = np.where(K1)
    
    if len(connections[0]) > 0:
        # 绘制红色散点表示连接
        plt.scatter(connections[1], connections[0], c='red', s=4, alpha=0.8)
    
    plt.title(f'K1 Adjacency Matrix at t={time_step}\n'
              f'λ1={model.strengthLambda1}, λ2={model.strengthLambda2}, '
              f'd1={model.distanceD1}, d2={model.distanceD2}', fontsize=12)
    plt.xlabel('Agent j', fontsize=12)
    plt.ylabel('Agent i', fontsize=12)
    plt.xlim(-0.5, model.agentsNum-0.5)
    plt.ylim(-0.5, model.agentsNum-0.5)
    plt.gca().invert_yaxis()  # 使y轴从上到下递增
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Connection (1)'),
                      Patch(facecolor='white', edgecolor='gray', label='No Connection (0)')]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 添加连接统计信息
    total_connections = np.sum(K1)
    max_possible = model.agentsNum * (model.agentsNum - 1)
    density = total_connections / max_possible
    
    plt.text(0.02, 0.98, f'Connections: {total_connections}\nDensity: {density:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/K1_scatter_t{time_step}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def visualize_K2_structure(K2, time_step, model, save_dir):
    """
    可视化K2张量的3D结构和切片
    """
    # 找到所有为True的三元组
    i_coords, j_coords, k_coords = np.where(K2)
    
    if len(i_coords) == 0:
        print(f"  警告: t={time_step}时刻K2张量中没有连接")
        return
    
    # 创建图形
    fig = plt.figure(figsize=(16, 6))
    
    # 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    colors = np.arange(len(i_coords))
    scatter = ax1.scatter(i_coords, j_coords, k_coords, 
                         c=colors, cmap='viridis', s=30, alpha=0.7)
    ax1.set_xlabel('Agent i')
    ax1.set_ylabel('Agent j')
    ax1.set_zlabel('Agent k')
    ax1.set_title(f'K2 3D Structure\nt={time_step}')
    
    # 选择两个切片进行2D可视化
    agentsNum = model.agentsNum
    slice_agents = [agentsNum//4, 3*agentsNum//4]
    
    for idx, agent_i in enumerate(slice_agents):
        ax = fig.add_subplot(1, 3, idx+2)
        slice_data = K2[agent_i, :, :].astype(int)
        
        im = ax.imshow(slice_data, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
        ax.set_title(f'K2[{agent_i}, j, k] slice')
        ax.set_xlabel('Agent k')
        ax.set_ylabel('Agent j')
        
        # 添加连接数信息
        connections = np.sum(slice_data)
        ax.text(0.02, 0.98, f'Connections: {connections}', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/K2_structure_t{time_step}.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_connections(K1, K2, time_step, model):
    """
    分析连接的统计特性
    """
    K1_connections = np.sum(K1)
    K2_connections = np.sum(K2)
    
    agentsNum = model.agentsNum
    max_K1 = agentsNum * (agentsNum - 1)
    max_K2 = agentsNum * (agentsNum - 1) * (agentsNum - 1)
    
    K1_density = K1_connections / max_K1
    K2_density = K2_connections / max_K2
    
    print(f"  t={time_step}: K1连接数={K1_connections} (密度={K1_density:.4f}), "
          f"K2连接数={K2_connections} (密度={K2_density:.6f})")

def create_temporal_analysis(positionX, model, time_points, save_dir):
    """
    创建时间演化分析
    """
    print("\n创建时间演化分析...")
    
    K1_stats = []
    K2_stats = []
    
    for t in tqdm(time_points, desc="分析时间演化"):
        K1_t, K2_t = compute_K_tensors_at_time(positionX[t], model)
        K1_stats.append(np.sum(K1_t))
        K2_stats.append(np.sum(K2_t))
    
    # 绘制时间演化图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(time_points, K1_stats, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('K1 Connections')
    ax1.set_title('K1 Connections Evolution')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_points, K2_stats, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('K2 Connections')
    ax2.set_title('K2 Connections Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/temporal_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def demo_with_custom_params():
    """
    使用自定义参数进行演示
    """
    # 示例：不同参数组合
    param_sets = [
        {
            'strengthLambda1': 0.05,
            'strengthLambda2': 0.05,
            'distanceD1': 0.3,
            'distanceD2': 0.5,
            'agentsNum': 200,
            'boundaryLength': 5,
            'randomSeed': 10
        },
        {
            'strengthLambda1': 0.1,
            'strengthLambda2': 0.1,
            'distanceD1': 0.5,
            'distanceD2': 0.7,
            'agentsNum': 200,
            'boundaryLength': 5,
            'randomSeed': 10
        }
    ]
    
    for i, params in tqdm(enumerate(param_sets), total=len(param_sets), desc="参数组合测试"):
        print(f"\n=== 参数组合 {i+1} ===")
        print(f"λ1={params['strengthLambda1']}, λ2={params['strengthLambda2']}")
        print(f"d1={params['distanceD1']}, d2={params['distanceD2']}")
        
        quick_visualize_K_tensors(model_params=params)

if __name__ == "__main__":
    print("=" * 60)
    print("K1和K2张量可视化演示")
    print("=" * 60)
    
    # 基本演示
    print("\n1. 使用默认参数进行可视化...")
    quick_visualize_K_tensors()
    
    # 如果需要测试多个参数组合，取消下面的注释
    # print("\n2. 使用多个参数组合进行对比...")
    # demo_with_custom_params()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)