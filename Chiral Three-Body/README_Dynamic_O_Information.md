# 动态O信息计算脚本说明

## 概述

本脚本实现了基于信息论的动态O信息（Dynamic O-Information）计算，用于分析复杂系统中多变量间的高阶相互作用和信息传递模式。动态O信息是一种衡量多变量系统中协同效应和冗余效应平衡的信息论度量。

## 理论基础

### 1. 互信息（Mutual Information）

互信息是信息论中衡量两个随机变量之间相互依赖程度的基本度量：

$$I(X;Y) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)}$$

其中：
- $p(x,y)$ 是联合概率分布
- $p(x)$, $p(y)$ 是边际概率分布

### 2. 条件互信息（Conditional Mutual Information）

条件互信息衡量在给定第三个变量Z的条件下，X和Y之间的相互信息：

$$I(X;Y|Z) = \sum_{x,y,z} p(x,y,z) \log_2 \frac{p(x,y|z)}{p(x|z)p(y|z)}$$

等价地：

$$I(X;Y|Z) = \sum_{x,y,z} p(x,y,z) \log_2 \frac{p(x,y,z)p(z)}{p(x,z)p(y,z)}$$

### 3. 动态O信息（Dynamic O-Information）

对于一个包含n个驱动变量$\{X_1, X_2, \ldots, X_n\}$和一个目标变量Y的系统，动态O信息定义为：

$$\Omega_n(Y; X_1, \ldots, X_n | Y_0) = (1-n) \cdot I(Y; X_1, \ldots, X_n | Y_0) + \sum_{j=1}^{n} I(Y; X_1, \ldots, X_{j-1}, X_{j+1}, \ldots, X_n | Y_0)$$

其中：
- $Y$ 是目标变量在时刻 $t+\tau$ 的值（$\tau$ 为时间延迟）
- $X_j$ 是第j个驱动变量在时刻 $t$ 的值
- $Y_0$ 是目标变量在时刻 $t$ 的历史值（条件变量）
- $n$ 是驱动变量的数量

### 4. 动态O信息的解释

- **第一项** $(1-n) \cdot I(Y; X_1, \ldots, X_n | Y_0)$：衡量所有驱动变量对目标变量的总体信息贡献，系数$(1-n)$使其为负值
- **第二项** $\sum_{j=1}^{n} I(Y; X_1, \ldots, X_{j-1}, X_{j+1}, \ldots, X_n | Y_0)$：衡量去除每个驱动变量后剩余变量的信息贡献之和

当$\Omega_n > 0$时，表示系统表现出**协同效应**（synergy）；当$\Omega_n < 0$时，表示系统表现出**冗余效应**（redundancy）。

## 算法实现

### 1. 数据预处理

#### K2张量计算
从三体系统的位置和相位数据计算K2张量：
```
K2[i,j,k] = 三体相互作用强度
```

#### M向量计算
对每个智能体i，计算其连接度：
$$M_i(t) = \sum_{j,k} K2[i,j,k](t)$$

#### 离散化
将连续的M向量离散化为有限状态空间：
```
M_discrete = digitize(M_series, bins)
```

### 2. 互信息计算

#### 概率分布估计
使用频率统计方法估计概率分布：
- 联合分布：$\hat{p}(x,y,z) = \frac{\text{count}(x,y,z)}{N}$
- 边际分布：$\hat{p}(x) = \frac{\text{count}(x)}{N}$

#### 数值稳定性
为避免$\log(0)$，只计算概率大于0的项。

### 3. 动态O信息计算流程

1. **时间序列构建**：
   - $Y(t) = M_{\text{target}}(t+\text{order})$ （目标变量的未来值）
   - $Y_0(t) = M_{\text{target}}(t)$ （目标变量的当前值）
   - $X_j(t) = M_{\text{driver}_j}(t)$ （驱动变量的当前值）

2. **第一项计算**：
   $$\text{term1} = (1-n) \cdot I(Y; X_1, \ldots, X_n | Y_0)$$

3. **第二项计算**：
   $$\text{term2} = \sum_{j=1}^{n} I(Y; X_{\setminus j} | Y_0)$$
   其中$X_{\setminus j}$表示除$X_j$外的所有驱动变量。

4. **最终结果**：
   $$\Omega_n = \text{term1} + \text{term2}$$

### 4. 总动态O信息

对系统中的每个变量作为目标变量计算动态O信息，然后求和：

$$\Omega_{\text{total}} = \sum_{i=1}^{N} \Omega_n^{(i)}$$

其中$\Omega_n^{(i)}$是以第i个变量为目标的动态O信息。

## 参数说明

- `num_bins`: 离散化的bin数量，影响概率估计的精度
- `max_drivers`: 每个目标变量考虑的最大驱动变量数量，用于控制计算复杂度
- `order`: 时间延迟阶数，通常设为1

## 计算复杂度

- 时间复杂度：$O(T \cdot N \cdot 2^{\text{max_drivers}})$
- 空间复杂度：$O(T \cdot N + \text{num_bins}^{\text{max_drivers}})$

其中T是时间步数，N是变量数量。

## 应用场景

1. **神经网络分析**：识别神经元间的协同和竞争关系
2. **生态系统研究**：分析物种间的相互作用模式
3. **金融市场**：研究资产间的信息传递和风险传染
4. **社会网络**：理解个体间的影响机制
5. **复杂物理系统**：如本脚本应用的三体手性系统

## 注意事项

1. **样本量要求**：需要足够的时间序列数据以获得可靠的概率估计
2. **离散化影响**：bin数量的选择会影响结果，需要在精度和统计可靠性间平衡
3. **计算资源**：高维系统的计算量随驱动变量数量指数增长
4. **统计显著性**：建议进行置换检验以评估结果的统计显著性

## 参考文献

1. Rosas, F. E., et al. (2019). Quantifying high-order interdependencies via multivariate extensions of the mutual information. Physical Review E, 100(3), 032305.
2. Timme, N., et al. (2014). Synergy, redundancy, and multivariate information measures: an experimentalist's perspective. Journal of computational neuroscience, 36(2), 119-140.
3. Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv preprint arXiv:1004.2515.