import numpy as np
import pandas as pd
from scipy.stats import entropy
from itertools import combinations
from tqdm import tqdm
import os
from main import ThreeBody

def load_K2_data(model, model_name):
    """
    加载K2张量数据
    
    Parameters:
    -----------
    model : ThreeBody
        模型实例
    model_name : str
        模型名称，用于构建数据文件路径
    
    Returns:
    --------
    K2_series : np.ndarray
        形状为 (TNum, agentsNum, agentsNum, agentsNum) 的K2张量时间序列
    """
    import pandas as pd
    import os
    from tqdm import tqdm
    from visualize_K_tensors import compute_K1_K2_tensors
    
    # 直接实现数据加载逻辑，避免 visualize_K_tensors.load_model_data 的设计问题
    targetPath = f"./data/{model_name}.h5"
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
        
        data = {
            'positionX': positionX,
            'phaseTheta': phaseTheta,
            'pointTheta': pointTheta,
            'TNum': TNum
        }
        
        # 计算K2张量
        _, K2_series = compute_K1_K2_tensors(model, data)
        
        return K2_series
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def compute_M_vector(K2_series):
    """
    计算M向量：每个agent在每个时刻的连接总数
    
    Parameters:
    -----------
    K2_series : np.ndarray
        形状为 (TNum, agentsNum, agentsNum, agentsNum) 的K2张量
    
    Returns:
    --------
    M_series : np.ndarray
        形状为 (TNum, agentsNum) 的M向量时间序列
    """
    print("计算M向量...")
    # M_i = sum over j,k of K2[i,j,k]
    M_series = np.sum(K2_series, axis=(2, 3))
    return M_series

def compute_probability_distribution(M_series, num_bins=10):
    """
    计算M向量的概率分布
    
    Parameters:
    -----------
    M_series : np.ndarray
        形状为 (TNum, agentsNum) 的M向量时间序列
    num_bins : int
        离散化的bin数量
    
    Returns:
    --------
    M_discrete : np.ndarray
        离散化后的M向量
    bin_edges : np.ndarray
        bin的边界
    """
    print("计算概率分布...")
    
    # 找到M的范围
    M_min = np.min(M_series)
    M_max = np.max(M_series)
    
    # 创建bin边界
    bin_edges = np.linspace(M_min, M_max + 1e-10, num_bins + 1)
    
    # 离散化M向量
    M_discrete = np.digitize(M_series, bin_edges) - 1
    M_discrete = np.clip(M_discrete, 0, num_bins - 1)
    
    return M_discrete, bin_edges

def compute_mutual_information(X, Y, Z=None):
    """
    计算互信息 I(X;Y) 或条件互信息 I(X;Y|Z)
    
    Parameters:
    -----------
    X, Y : np.ndarray
        输入变量。X可以是多维的（多个变量），Y应该是一维的
    Z : np.ndarray, optional
        条件变量，应该是一维的
    
    Returns:
    --------
    mi : float
        互信息值
    """
    # 确保Y是一维的
    if Y.ndim > 1:
        Y = Y.flatten()
    
    # 处理X：如果是多维的，需要将每行作为一个样本
    if X.ndim == 1:
        X_data = X.reshape(-1, 1)
    else:
        X_data = X
    
    # 确保Z是一维的（如果存在）
    if Z is not None and Z.ndim > 1:
        Z = Z.flatten()
    
    # 验证长度一致性
    n_samples = len(Y)
    if X_data.shape[0] != n_samples:
        raise ValueError(f"X和Y的样本数不匹配: X有{X_data.shape[0]}个样本，Y有{n_samples}个样本")
    if Z is not None and len(Z) != n_samples:
        raise ValueError(f"Z和Y的样本数不匹配: Z有{len(Z)}个样本，Y有{n_samples}个样本")
    
    if Z is None:
        # 计算 I(X;Y)
        # 将X的每一行转换为元组，以便作为字典键
        if X_data.shape[1] == 1:
            X_tuples = X_data.flatten()
        else:
            X_tuples = [tuple(row) for row in X_data]
        
        # 构建联合分布
        xy_pairs = list(zip(X_tuples, Y))
        xy_unique, xy_counts = np.unique(xy_pairs, axis=0, return_counts=True)
        p_xy = xy_counts / n_samples
        
        # 构建边际分布
        x_unique, x_counts = np.unique(X_tuples, axis=0, return_counts=True)
        p_x = x_counts / n_samples
        
        y_unique, y_counts = np.unique(Y, return_counts=True)
        p_y = y_counts / n_samples
        
        # 计算互信息
        mi = 0.0
        for i, (x_val, y_val) in enumerate(xy_unique):
            p_xy_val = p_xy[i]
            
            # 找到对应的边际概率
            x_idx = np.where(x_unique == x_val)[0]
            y_idx = np.where(y_unique == y_val)[0]
            
            if len(x_idx) > 0 and len(y_idx) > 0:
                p_x_val = p_x[x_idx[0]]
                p_y_val = p_y[y_idx[0]]
                
                if p_xy_val > 0 and p_x_val > 0 and p_y_val > 0:
                    mi += p_xy_val * np.log2(p_xy_val / (p_x_val * p_y_val))
    else:
        # 计算 I(X;Y|Z)
        # 将X的每一行转换为元组
        if X_data.shape[1] == 1:
            X_tuples = X_data.flatten()
        else:
            X_tuples = [tuple(row) for row in X_data]
        
        # 构建三元组 - 确保数据类型一致
        xyz_triples = []
        for i in range(n_samples):
            if X_data.shape[1] == 1:
                x_val = X_tuples[i]
            else:
                x_val = X_tuples[i]
            xyz_triples.append((x_val, Y[i], Z[i]))
        
        # 使用pandas或手动方式处理unique，避免numpy的inhomogeneous问题
        xyz_dict = {}
        for triple in xyz_triples:
            if triple in xyz_dict:
                xyz_dict[triple] += 1
            else:
                xyz_dict[triple] = 1
        
        xyz_unique = list(xyz_dict.keys())
        xyz_counts = np.array(list(xyz_dict.values()))
        p_xyz = xyz_counts / n_samples
        
        # 构建二元组 - 使用手动方式处理unique
        xz_dict = {}
        for i in range(n_samples):
            if X_data.shape[1] == 1:
                x_val = X_tuples[i]
            else:
                x_val = X_tuples[i]
            xz_pair = (x_val, Z[i])
            if xz_pair in xz_dict:
                xz_dict[xz_pair] += 1
            else:
                xz_dict[xz_pair] = 1
        
        xz_unique = list(xz_dict.keys())
        xz_counts = np.array(list(xz_dict.values()))
        p_xz = xz_counts / n_samples
        
        yz_dict = {}
        for i in range(n_samples):
            yz_pair = (Y[i], Z[i])
            if yz_pair in yz_dict:
                yz_dict[yz_pair] += 1
            else:
                yz_dict[yz_pair] = 1
        
        yz_unique = list(yz_dict.keys())
        yz_counts = np.array(list(yz_dict.values()))
        p_yz = yz_counts / n_samples
        
        z_unique, z_counts = np.unique(Z, return_counts=True)
        p_z = z_counts / n_samples
        
        # 计算条件互信息
        mi = 0.0
        for i, (x_val, y_val, z_val) in enumerate(xyz_unique):
            p_xyz_val = p_xyz[i]
            
            # 找到对应的边际概率
            xz_target = (x_val, z_val)
            yz_target = (y_val, z_val)
            
            # 手动查找匹配的索引
            xz_idx = -1
            for j, xz_pair in enumerate(xz_unique):
                if xz_pair == xz_target:
                    xz_idx = j
                    break
            
            yz_idx = -1
            for j, yz_pair in enumerate(yz_unique):
                if yz_pair == yz_target:
                    yz_idx = j
                    break
            
            z_idx = np.where(z_unique == z_val)[0]
            
            if xz_idx >= 0 and yz_idx >= 0 and len(z_idx) > 0:
                p_xz_val = p_xz[xz_idx]
                p_yz_val = p_yz[yz_idx]
                p_z_val = p_z[z_idx[0]]
                
                if p_xyz_val > 0 and p_xz_val > 0 and p_yz_val > 0 and p_z_val > 0:
                    mi += p_xyz_val * np.log2((p_xyz_val * p_z_val) / (p_xz_val * p_yz_val))
    
    return mi

def compute_dynamic_O_information(M_discrete, target_idx, driver_indices, order=1):
    """
    计算动态O信息
    
    Parameters:
    -----------
    M_discrete : np.ndarray
        离散化的M向量时间序列，形状为 (TNum, agentsNum)
    target_idx : int
        目标变量的索引
    driver_indices : list
        驱动变量的索引列表
    order : int
        时间延迟阶数
    
    Returns:
    --------
    dO_info : float
        动态O信息值
    """
    TNum, agentsNum = M_discrete.shape
    n = len(driver_indices)
    
    if n < 2:
        return 0.0
    
    # 确保有足够的时间步数
    if TNum <= order:
        return 0.0
    
    # 构建时间序列 - 确保所有数组长度一致
    # Y(t) = target(t+order) - 目标变量的未来值
    Y = M_discrete[order:, target_idx]
    
    # Y_0(t) = target(t) - 目标变量的当前值（历史）
    Y_0 = M_discrete[:-order, target_idx] if order > 0 else None
    
    # X_j(t) = driver_j(t) - 驱动变量的当前值
    X = []
    for j in driver_indices:
        X_j = M_discrete[:-order, j] if order > 0 else M_discrete[:, j]
        X.append(X_j)
    
    # 验证数组长度一致性
    expected_length = len(Y)
    if Y_0 is not None and len(Y_0) != expected_length:
        print(f"警告：Y_0长度不匹配 - Y: {len(Y)}, Y_0: {len(Y_0)}")
        return 0.0
    
    for i, X_j in enumerate(X):
        if len(X_j) != expected_length:
            print(f"警告：X[{i}]长度不匹配 - Y: {len(Y)}, X[{i}]: {len(X_j)}")
            return 0.0
    
    # 计算动态O信息
    # dΩ_n = (1-n) * I(Y; X | Y_0) + Σ_j I(Y; X\X_j | Y_0)
    
    # 第一项：(1-n) * I(Y; X | Y_0)
    if Y_0 is not None:
        # 条件互信息
        X_all = np.column_stack(X)
        term1 = (1 - n) * compute_mutual_information(X_all, Y, Y_0)
    else:
        X_all = np.column_stack(X)
        term1 = (1 - n) * compute_mutual_information(X_all, Y)
    
    # 第二项：Σ_j I(Y; X\X_j | Y_0)
    term2 = 0.0
    for j in range(n):
        # X\X_j: 除了X_j之外的所有驱动变量
        X_without_j = [X[k] for k in range(n) if k != j]
        if len(X_without_j) > 0:
            X_without_j_combined = np.column_stack(X_without_j)
            if Y_0 is not None:
                term2 += compute_mutual_information(X_without_j_combined, Y, Y_0)
            else:
                term2 += compute_mutual_information(X_without_j_combined, Y)
    
    dO_info = term1 + term2
    return dO_info

def compute_total_dynamic_O_information(M_discrete, max_drivers=5):
    """
    计算总动态O信息：每个变量作为目标变量的动态O信息之和
    
    Parameters:
    -----------
    M_discrete : np.ndarray
        离散化的M向量时间序列，形状为 (TNum, agentsNum)
    max_drivers : int
        每个目标变量考虑的最大驱动变量数量
    
    Returns:
    --------
    total_dO_info : float
        总动态O信息
    individual_dO_info : list
        每个变量作为目标的动态O信息列表
    """
    TNum, agentsNum = M_discrete.shape
    individual_dO_info = []
    
    print(f"计算总动态O信息（{agentsNum}个变量）...")
    
    for target_idx in tqdm(range(agentsNum), desc="计算各目标变量的动态O信息"):
        # 选择驱动变量（除了目标变量之外的所有变量）
        all_drivers = [i for i in range(agentsNum) if i != target_idx]
        
        # 如果驱动变量太多，随机选择一部分
        if len(all_drivers) > max_drivers:
            np.random.seed(42)  # 保证可重复性
            driver_indices = np.random.choice(all_drivers, max_drivers, replace=False).tolist()
        else:
            driver_indices = all_drivers
        
        # 计算该目标变量的动态O信息
        dO_info = compute_dynamic_O_information(M_discrete, target_idx, driver_indices)
        individual_dO_info.append(dO_info)
    
    # 计算总和
    total_dO_info = np.sum(individual_dO_info)
    
    return total_dO_info, individual_dO_info

def analyze_dynamic_O_information(model, model_name, num_bins=10, max_drivers=5):
    """
    完整的动态O信息分析流程
    
    Parameters:
    -----------
    model : ThreeBody
        模型实例
    model_name : str
        模型名称
    num_bins : int
        离散化的bin数量
    max_drivers : int
        每个目标变量考虑的最大驱动变量数量
    
    Returns:
    --------
    results : dict
        包含分析结果的字典
    """
    print(f"开始分析模型 {model_name} 的动态O信息")
    print("="*50)
    
    # 1. 加载K2数据
    print("步骤1: 加载K2张量数据")
    K2_series = load_K2_data(model, model_name)
    if K2_series is None:
        print("无法加载数据")
        return None
    
    print(f"K2张量形状: {K2_series.shape}")
    
    # 2. 计算M向量
    print("\n步骤2: 计算M向量")
    M_series = compute_M_vector(K2_series)
    print(f"M向量形状: {M_series.shape}")
    print(f"M向量统计: 最小值={np.min(M_series)}, 最大值={np.max(M_series)}, 平均值={np.mean(M_series):.2f}")
    
    # 3. 计算概率分布
    print("\n步骤3: 计算概率分布")
    M_discrete, bin_edges = compute_probability_distribution(M_series, num_bins)
    print(f"离散化后的M向量形状: {M_discrete.shape}")
    print(f"Bin边界: {bin_edges}")
    
    # 4. 计算总动态O信息
    print("\n步骤4: 计算总动态O信息")
    total_dO_info, individual_dO_info = compute_total_dynamic_O_information(M_discrete, max_drivers)
    
    # 5. 结果汇总
    results = {
        'model_name': model_name,
        'K2_shape': K2_series.shape,
        'M_series': M_series,
        'M_discrete': M_discrete,
        'bin_edges': bin_edges,
        'total_dynamic_O_information': total_dO_info,
        'individual_dynamic_O_information': individual_dO_info,
        'num_bins': num_bins,
        'max_drivers': max_drivers
    }
    
    print("\n" + "="*50)
    print("分析结果:")
    print(f"总动态O信息: {total_dO_info:.6f}")
    print(f"平均每变量动态O信息: {np.mean(individual_dO_info):.6f}")
    print(f"动态O信息标准差: {np.std(individual_dO_info):.6f}")
    print(f"最大动态O信息: {np.max(individual_dO_info):.6f}")
    print(f"最小动态O信息: {np.min(individual_dO_info):.6f}")
    
    return results

def save_results(results, save_path="./dynamic_O_results"):
    """
    保存分析结果
    
    Parameters:
    -----------
    results : dict
        分析结果字典
    save_path : str
        保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    model_name = results['model_name']
    
    # 保存为numpy文件
    np.savez(f"{save_path}/{model_name}_dynamic_O_results.npz", **results)
    
    # 保存文本报告
    with open(f"{save_path}/{model_name}_dynamic_O_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"动态O信息分析报告\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"K2张量形状: {results['K2_shape']}\n")
        f.write(f"离散化bins数量: {results['num_bins']}\n")
        f.write(f"最大驱动变量数: {results['max_drivers']}\n")
        f.write(f"\n总动态O信息: {results['total_dynamic_O_information']:.6f}\n")
        f.write(f"平均每变量动态O信息: {np.mean(results['individual_dynamic_O_information']):.6f}\n")
        f.write(f"动态O信息标准差: {np.std(results['individual_dynamic_O_information']):.6f}\n")
        f.write(f"最大动态O信息: {np.max(results['individual_dynamic_O_information']):.6f}\n")
        f.write(f"最小动态O信息: {np.min(results['individual_dynamic_O_information']):.6f}\n")
    
    print(f"\n结果已保存到: {save_path}")

if __name__ == "__main__":
    # 创建模型实例（与main函数参数保持一致）
    model = ThreeBody(
        strengthLambda1=0.03, 
        strengthLambda2=0.01,
        distanceD1=0.5,
        distanceD2=0.3,
        agentsNum=200
    )
    
    # 自动生成模型名称
    model_name = str(model)
    print(f"使用模型: {model_name}")
    
    # 分析动态O信息
    results = analyze_dynamic_O_information(
        model=model,
        model_name=model_name,
        num_bins=10,
        max_drivers=5
    )
    
    if results is not None:
        # 保存结果
        save_results(results)
        
        print("\n分析完成！")
    else:
        print("分析失败，请检查数据文件是否存在。")