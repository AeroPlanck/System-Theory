import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ma
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from tqdm import tqdm
import os
from main import ThreeBody

def load_model_data(model):
    """加载模型数据"""
    targetPath = f"./data/{model}.h5"
    if not os.path.exists(targetPath):
        print(f"数据文件不存在: {targetPath}")
        return None
    
    try:
        print("正在加载数据...")
        with tqdm(total=3, desc="加载数据文件") as pbar:
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
        
        return {
            'positionX': positionX,
            'phaseTheta': phaseTheta,
            'pointTheta': pointTheta,
            'TNum': TNum
        }
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def compute_K1_K2_tensors(model, data):
    """计算K1和K2张量的时间序列"""
    positionX = data['positionX']
    TNum = data['TNum']
    agentsNum = model.agentsNum
    
    # 存储K1和K2张量
    print("初始化张量存储空间...")
    with tqdm(total=2, desc="初始化张量") as pbar:
        K1_series = np.zeros((TNum, agentsNum, agentsNum), dtype=bool)
        pbar.update(1)
        
        K2_series = np.zeros((TNum, agentsNum, agentsNum, agentsNum), dtype=bool)
        pbar.update(1)
    
    print("计算K1和K2张量...")
    for t in tqdm(range(TNum), desc="计算张量时间序列"):
        # 计算当前时刻的位置差
        curr_pos = positionX[t]
        
        # 计算周期性边界条件下的距离
        deltaX = compute_periodic_deltaX(curr_pos, model.boundaryLength)
        distances = np.sqrt(deltaX[:, :, 0]**2 + deltaX[:, :, 1]**2)
        
        # 计算K1 (邻接矩阵)
        eyeMask = ~np.eye(agentsNum, dtype=bool)
        K1_t = (distances <= model.distanceD1) * eyeMask
        K1_series[t] = K1_t
        
        # 计算K2 (邻接张量)
        K2_base = (distances <= model.distanceD2) * eyeMask
        K2_t = K2_base[:, :, np.newaxis] * K2_base[:, np.newaxis, :]
        K2_series[t] = K2_t
    
    return K1_series, K2_series

def compute_periodic_deltaX(positionX, boundaryLength):
    """计算周期性边界条件下的位置差"""
    halfBoundary = boundaryLength / 2
    pos_expanded = positionX[:, np.newaxis, :]
    others_expanded = positionX[np.newaxis, :, :]
    
    subX = pos_expanded - others_expanded
    
    # 周期性边界条件调整
    deltaX = pos_expanded - (
        others_expanded * (-halfBoundary <= subX) * (subX <= halfBoundary) + 
        (others_expanded - boundaryLength) * (subX < -halfBoundary) + 
        (others_expanded + boundaryLength) * (subX > halfBoundary)
    )
    
    return deltaX

def visualize_K1_evolution(K1_series, model, save_path="./visualizations"):
    """可视化K1矩阵的演化 (散点图)"""
    os.makedirs(save_path, exist_ok=True)
    TNum = K1_series.shape[0]
    
    # 选择几个关键时刻进行可视化
    time_points = [0, TNum//4, TNum//2, 3*TNum//4, TNum-1]
    
    print("生成K1演化可视化...")
    fig, axes = plt.subplots(1, len(time_points), figsize=(22, 4))
    if len(time_points) == 1:
        axes = [axes]
    
    for i, t in tqdm(enumerate(time_points), total=len(time_points), desc="绘制K1时间点"):
        # 设置白色背景
        axes[i].set_facecolor('white')
        
        # 找到所有连接的位置 (值为1的位置)
        connections = np.where(K1_series[t])
        
        if len(connections[0]) > 0:
            # 绘制红色散点表示连接
             axes[i].scatter(connections[1], connections[0], c='red', s=3, alpha=0.8)
        
        axes[i].set_title(f'K1 at t={t}')
        axes[i].set_xlabel('Agent j')
        axes[i].set_ylabel('Agent i')
        axes[i].set_xlim(-0.5, K1_series.shape[1]-0.5)
        axes[i].set_ylim(-0.5, K1_series.shape[1]-0.5)
        axes[i].invert_yaxis()  # 使y轴从上到下递增
        axes[i].grid(True, alpha=0.3)
    
    # 创建一个虚拟的颜色条用于图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Connection (1)'),
                      Patch(facecolor='white', edgecolor='gray', label='No Connection (0)')]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # 为图例留出空间
    plt.savefig(f"{save_path}/K1_evolution_{model}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"K1演化图已保存至: {save_path}/K1_evolution_{model}.png")

def visualize_K2_3D(K2_series, model, save_path="./visualizations"):
    """可视化K2张量的3D结构（散点图，红色表示连接）"""
    os.makedirs(save_path, exist_ok=True)
    TNum = K2_series.shape[0]
    
    # 选择最后一个时刻的K2张量进行3D可视化
    K2_final = K2_series[-1]
    
    # 找到所有为True的三元组
    i_coords, j_coords, k_coords = np.where(K2_final)
    
    if len(i_coords) == 0:
        print("警告: K2张量中没有连接，跳过3D可视化")
        return
    
    # 创建3D散点图
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    # 所有连接点用红色表示（值为1）
    scatter = ax.scatter(i_coords, j_coords, k_coords, 
                         c='red', s=1, alpha=0.8)
    
    ax.set_xlabel('Agent i')
    ax.set_ylabel('Agent j')
    ax.set_zlabel('Agent k')
    ax.set_title(f'K2 Tensor 3D Structure (t={TNum-1})\nRed points: K2[i,j,k] = 1')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Connection (1)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(f"{save_path}/K2_3D_{model}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"K2 3D结构图已保存至: {save_path}/K2_3D_{model}.png")

def visualize_K2_slices(K2_series, model, save_path="./visualizations"):
    """可视化K2张量的切片 (固定一个维度，散点图)"""
    os.makedirs(save_path, exist_ok=True)
    TNum = K2_series.shape[0]
    agentsNum = K2_series.shape[1]
    
    # 选择最后一个时刻和几个代表性的agent
    K2_final = K2_series[-1]
    agent_indices = [0, agentsNum//4, agentsNum//2, 3*agentsNum//4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, agent_i in enumerate(agent_indices):
        if idx >= len(axes):
            break
            
        # 设置白色背景
        axes[idx].set_facecolor('white')
        
        # K2[i, j, k] 固定i=agent_i，显示j-k平面
        slice_data = K2_final[agent_i, :, :]
        
        # 找到所有连接的位置 (值为True的位置)
        connections = np.where(slice_data)
        
        if len(connections[0]) > 0:
            # 绘制红色散点表示连接
             axes[idx].scatter(connections[1], connections[0], c='red', s=4, alpha=0.8)
        
        axes[idx].set_title(f'K2[{agent_i}, j, k] slice')
        axes[idx].set_xlabel('Agent k')
        axes[idx].set_ylabel('Agent j')
        axes[idx].set_xlim(-0.5, agentsNum-0.5)
        axes[idx].set_ylim(-0.5, agentsNum-0.5)
        axes[idx].invert_yaxis()  # 使y轴从上到下递增
        axes[idx].grid(True, alpha=0.3)
    
    # 创建一个虚拟的颜色条用于图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Connection (1)'),
                      Patch(facecolor='white', edgecolor='gray', label='No Connection (0)')]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # 为图例留出空间
    plt.savefig(f"{save_path}/K2_slices_{model}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"K2切片图已保存至: {save_path}/K2_slices_{model}.png")

def analyze_K_statistics(K1_series, K2_series, model, save_path="./visualizations"):
    """分析K1和K2的统计特性"""
    os.makedirs(save_path, exist_ok=True)
    TNum = K1_series.shape[0]
    
    # 计算连接密度随时间的变化
    K1_density = np.mean(K1_series, axis=(1, 2))
    K2_density = np.mean(K2_series, axis=(1, 2, 3))
    
    # 计算连接数量
    K1_count = np.sum(K1_series, axis=(1, 2))
    K2_count = np.sum(K2_series, axis=(1, 2, 3))
    
    # 绘制统计图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('K1 and K2 Tensor Statistics with Definitions', fontsize=16, fontweight='bold')
    
    # K1密度
    axes[0, 0].plot(K1_density, linewidth=2)
    axes[0, 0].set_title('K1 Connection Density over Time\n' + 
                        r'$\rho_{K1}(t) = \frac{1}{N^2} \sum_{i,j} K1_{ij}(t)$', fontsize=12)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    # K2密度
    axes[0, 1].plot(K2_density, linewidth=2, color='orange')
    axes[0, 1].set_title('K2 Connection Density over Time\n' + 
                        r'$\rho_{K2}(t) = \frac{1}{N^3} \sum_{i,j,k} K2_{ijk}(t)$', fontsize=12)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    
    # K1连接数
    axes[1, 0].plot(K1_count, linewidth=2, color='green')
    axes[1, 0].set_title('K1 Total Connections over Time\n' + 
                        r'$C_{K1}(t) = \sum_{i,j} K1_{ij}(t)$', fontsize=12)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # K2连接数
    axes[1, 1].plot(K2_count, linewidth=2, color='red')
    axes[1, 1].set_title('K2 Total Connections over Time\n' + 
                        r'$C_{K2}(t) = \sum_{i,j,k} K2_{ijk}(t)$', fontsize=12)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加定义说明文本框
    textstr = '\n'.join([
        'Definitions:',
        r'$K1_{ij}$: Binary adjacency matrix (pairwise connections)',
        r'$K2_{ijk}$: Binary tensor (three-body connections)',
        r'$N$: Number of agents',
        r'$\rho$: Connection density (normalized)',
        r'$C$: Total connection count'
    ])
    
    # 在图的右下角添加定义框
    fig.text(0.02, 0.02, textstr, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为定义框留出空间
    plt.savefig(f"{save_path}/K_statistics_{model}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"K统计分析图已保存至: {save_path}/K_statistics_{model}.png")
    
    # 返回统计数据
    return {
        'K1_density': K1_density,
        'K2_density': K2_density,
        'K1_count': K1_count,
        'K2_count': K2_count
    }

def create_K2_animation(K2_series, model, save_path="./visualizations"):
    """创建K2张量演化的动画（散点图，MP4格式）"""
    os.makedirs(save_path, exist_ok=True)
    TNum = K2_series.shape[0]
    agentsNum = K2_series.shape[1]
    
    # 选择一个固定的agent进行切片动画
    fixed_agent = agentsNum // 2
    
    print("开始生成K2演化动画...")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    def animate(frame):
        ax.clear()
        ax.set_facecolor('white')
        
        slice_data = K2_series[frame, fixed_agent, :, :]
        
        # 找到所有连接的位置
        connections = np.where(slice_data)
        
        if len(connections[0]) > 0:
            # 绘制红色散点表示连接
             ax.scatter(connections[1], connections[0], c='red', s=1, alpha=0.8)
        
        ax.set_title(f'K2[{fixed_agent}, j, k] at t={frame}')
        ax.set_xlabel('Agent k')
        ax.set_ylabel('Agent j')
        ax.set_xlim(-0.5, agentsNum-0.5)
        ax.set_ylim(-0.5, agentsNum-0.5)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        return ax.collections
    
    # 创建动画（每10帧取一帧以减少文件大小）
    frames = range(0, TNum, max(1, TNum//50))
    print(f"生成动画帧数: {len(frames)}")
    
    with tqdm(total=len(frames), desc="生成K2动画帧") as pbar:
        def animate_with_progress(frame):
            result = animate(frame)
            pbar.update(1)
            return result
        
        ani = ma.FuncAnimation(fig, animate_with_progress, frames=frames, interval=200, blit=False)
        
        # 保存为MP4格式动画
        print("正在保存MP4动画文件...")
        ani.save(f"{save_path}/K2_evolution_{model}.mp4", writer='ffmpeg', fps=5, 
                extra_args=['-vcodec', 'libx264'])
    
    plt.close()
    print(f"K2演化动画已保存至: {save_path}/K2_evolution_{model}.mp4")

def main():
    """主函数"""
    print("=" * 60)
    print("K1和K2张量可视化分析")
    print("=" * 60)
    
    # 创建一个示例模型（需要根据实际情况调整参数）
    model = ThreeBody(
        strengthLambda1=0.1, 
        strengthLambda2=0.1, 
        distanceD1=0.5, 
        distanceD2=0.5, 
        agentsNum=200,  # 使用较小的agent数量以便可视化
        boundaryLength=5,
        tqdm=False, 
        overWrite=False
    )
    
    print(f"正在处理模型: {model}")
    
    # 加载数据
    data = load_model_data(model)
    if data is None:
        print("无法加载数据，请确保数据文件存在")
        return
    
    # 计算K1和K2张量
    K1_series, K2_series = compute_K1_K2_tensors(model, data)
    
    # 创建可视化
    print("\n开始创建可视化...")
    visualization_tasks = [
        ("K1演化可视化", lambda: visualize_K1_evolution(K1_series, model)),
        ("K2 3D结构可视化", lambda: visualize_K2_3D(K2_series, model)),
        ("K2切片可视化", lambda: visualize_K2_slices(K2_series, model)),
        ("统计分析", lambda: analyze_K_statistics(K1_series, K2_series, model))
    ]
    
    stats = None
    for task_name, task_func in tqdm(visualization_tasks, desc="执行可视化任务"):
        print(f"\n正在执行: {task_name}")
        result = task_func()
        if task_name == "统计分析":
            stats = result
    
    # 自动生成K2演化动画（MP4格式）
    print("\n=== 动画生成阶段 ===")
    create_K2_animation(K2_series, model)
    
    print("\n" + "=" * 60)
    print("所有可视化已完成！")
    print(f"结果保存在 ./visualizations/ 目录中")
    print("=" * 60)
    
    # 打印一些基本统计信息
    if stats:
        print(f"\n基本统计信息:")
        print(f"时间步数: {data['TNum']}")
        print(f"Agent数量: {model.agentsNum}")
        print(f"最终K1连接密度: {stats['K1_density'][-1]:.4f}")
        print(f"最终K2连接密度: {stats['K2_density'][-1]:.6f}")
        print(f"最终K1连接数: {stats['K1_count'][-1]}")
        print(f"最终K2连接数: {stats['K2_count'][-1]}")
        print("=" * 60)

if __name__ == "__main__":
    main()