# 圆边界手性 Lattice：未成晶、圆盘矩阵与 \(K\) 扫描综合结论

## 一句话结论

1. “未成晶”必须限定为“未通过手性边界多团簇 Lattice 判据”；部分 \(\alpha>0.5\pi\)
   条件实际形成了体内涡旋/团簇阵列，不能称为完全无晶体。
2. Dispersion.py 的体 \(3\times3\) 矩阵加圆边界后变成每个整数 \(m\) 一个三分量径向
   积分--微分算符；仅做 \(k\to m/R\) 是不成立的。
3. 完整加入邻居归一化和镜面低阶矩边界后，当前均匀三场圆盘闭合在
   \(N_r=41\)--81 间没有收敛的 \(m_*\)（0/8 个 \(K\times D\) 单元收敛），因此现在
   不能声称该线性闭合已经对应实验。
4. 粒子实验表明 \(K\) 会改变整数平台和形成动力学，但不会让晶格常数按 \(v/K\)
   缩放：有效弧长始终约 \(1.0\)--\(1.07d_0\)。主尺度仍由 \(d_0\) 控制，\(K\) 负责
   径向贴墙程度、锁定速度及相邻整数平台切换。

## 1. 未通过判据的参数

详细逐条清单及形貌分类见
[Boundary_Lattice_Noncrystallized_Inventory.md](Boundary_Lattice_Noncrystallized_Inventory.md)。

- 原临界/近临界量子化数据中有 6 条严格失败；
- 长时间 Alpha Sweep 中除 \(\alpha=0.5\pi\) 外的 12 组均未形成目标边界 Lattice，
  但其中 \(\alpha=0.6\pi\) 是明确的体涡旋阵列，\(\alpha=0.8\pi\) 也高度结构化；
- 新 \(K\) 扫描中 7/24 条在 \(t=250\) 前未严格成晶；所有失败都应写成
  “by \(t=250\)”，不能外推为无限时间不存在该吸引子。

## 2. 圆盘线性算符

定义

\[
p_+=p_x+ip_y=P_+(r)e^{i(m+1)\phi},\qquad
p_-=p_x-ip_y=P_-(r)e^{i(m-1)\phi},
\]

\[
\delta\rho=R_m(r)e^{im\phi},
\quad
\mathcal D_\ell^\uparrow=\partial_r-\ell/r,
\quad
\mathcal D_\ell^\downarrow=\partial_r+\ell/r.
\]

圆盘算符为

\[
\mathbb L_m=
\begin{pmatrix}
0&-\frac v2\mathcal D_{m+1}^{\downarrow}
&-\frac v2\mathcal D_{m-1}^{\uparrow}\\
-\frac v2\mathcal D_m^{\uparrow}
&\mathcal A_{m+1}-i\mathcal B_{m+1}&0\\
-\frac v2\mathcal D_m^{\downarrow}
&0&\mathcal A_{m-1}+i\mathcal B_{m-1}
\end{pmatrix}.
\]

粒子一致的墙边归一化卷积是

\[
\mathcal C_\ell h(r)=
\frac{\int_0^Rr'dr'\,G_\ell(r,r')h(r')}
{\int_0^Rr'dr'\,G_0(r,r')},
\]

\[
\mathcal A_\ell=\frac K2\cos\alpha\,\mathcal C_\ell,
\]

\[
\mathcal B_\ell=(-\omega+K\sin\alpha)I
-\frac K2\sin\alpha\,\mathcal C_\ell
-\frac{v^2}{4D_K}\Delta_\ell,
\qquad
D_K=2\omega-2K\sin\alpha.
\]

镜面反射给出

\[
P_-(R)=-P_+(R),
\]

\[
\left[\partial_r+\frac{m-1}{R}\right]P_-(R)=
-\left[\partial_r-\frac{m+1}{R}\right]P_+(R).
\]

完整推导、top-hat 圆盘核和原点正则条件见
[Circular_Boundary_Matrix_Derivation.md](Circular_Boundary_Linear_Operator/Circular_Boundary_Matrix_Derivation.md)。

### 能否对应实验？

目前答案是**不能定量对应**。理由有两层：

- 简单体谱离散在原五个直径预测 \((8,9,10,11,13)\)，实验为
  \((9,10,12,13,15)\)，逐样本精确符合率为 0；
- 上述完整均匀三场圆盘闭合的径向数值求谱在五个分辨率下 0/8 个单元得到收敛
  \(m_*\)，候选实增长通常只有 \(10^{-7}\)--\(10^{-4}\) 且随网格漂移。

因此下一步不是继续提高 bulk 的 \(k\) 网格，而是：

\[
\bar\rho(r),\quad \bar p_r=0,\quad\bar p_\phi(r)\ne0,
\]

先求非均匀环流基态，再解

\[
\sigma u_m=\mathbb J_m[\bar\rho,\bar p_\phi]u_m,
\]

或保留更多角谐波做 kinetic 墙面谱。当前团簇 Lattice 很可能是边界环流的方位调制
不稳定及其非线性锁定，而不是均匀态的一次体失稳。

数值诊断见
[Disk_Operator_Numerical_Diagnostic.md](Circular_Boundary_Linear_Operator/Disk_Operator_Numerical_Diagnostic.md)。

## 3. \(K\) 扫描结果

固定 \(N=2000,d_0=1,v=3,\alpha=\pi/2,dt=0.005\)，每条 50000 步；
\(K=(8,12,20.75,40)\)、\(D=(3.30,4.58)\)、共同 seeds \(=(9,10,11)\)，共 24 条。

只统计严格成晶样本：

| \(K\) | 成晶/6 | \(D=3.30\) 的 \(m\) | \(D=4.58\) 的 \(m\) | 中位有效弧长 | 中位实际弦长 | 中位 \(q=m/R_{\rm eff}\) | 中位锁定时间 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4/6 | 9 | 13 | 1.0595 | 1.0381 | 5.9305 | 140 |
| 12 | 4/6 | 9 | 13 | 1.0694 | 1.0533 | 5.8765 | 110 |
| 20.75 | 3/6 | 10 | 14 | 1.0087 | 0.9930 | 6.2290 | 62.5 |
| 40 | 6/6 | 10 | 14 | 1.0186 | 1.0061 | 6.1688 | 45 |

三个直接结论：

1. \(K\) 的确改变整数选择：在两个直径上都出现
   \[
   (9,13)\longrightarrow(10,14)
   \]
   的同步 \(+1\) 平台切换，位置在 \(K=12\) 与 \(20.75\) 之间。
2. 有效波长不是严格常数，但只在约 6% 范围内变化；与此同时 \(v/K\) 从 0.375
   降到 0.075，变化 5 倍。因此不支持 \(a\propto v/K\)，支持“\(d_0\) 主导、
   \(K\) 微调并触发整数换台阶”。
3. \(K\) 增大时团簇更贴墙：团簇到墙的中位距离约从 0.130 降到
   0.102、0.045、0.025；已成晶样本的锁定时间也从约 140 降到 45。

体谱右极限波长为 \(1.187,1.246,1.233,1.226\)，始终比实测有效弧长长
约 11%--18%；17 条成晶样本中仅 3 条（都属于 \(K=8,D=3.30\)）的整数模式与
体谱离散预测相同。粒子的 \(K\) 响应不能由裸体谱解释。

原始测量与图见：

- [Boundary_Lattice_K_Sweep_Measurements.csv](Boundary_Lattice_K_Sweep/Boundary_Lattice_K_Sweep_Measurements.csv)
- [Boundary_Lattice_K_Sweep_Analysis.png](Boundary_Lattice_K_Sweep/Boundary_Lattice_K_Sweep_Analysis.png)
- [Boundary_Lattice_K_Sweep_Kinetics.png](Boundary_Lattice_K_Sweep/Boundary_Lattice_K_Sweep_Kinetics.png)

## 4. \(K=40\) 步长收敛

对 seed 9 的两个直径，把 \(dt\) 从 0.005 减半到 0.0025，同时把步数从 50000
加倍到 100000，保持物理时间 \(t=250\)：

| \(D\) | 粗步长 \(m\) | 半步长 \(m\) | 有效弧长相对变化 | 实测弦长相对变化 |
|---:|---:|---:|---:|---:|
| 3.30 | 10 | 10 | \(-4.39\times10^{-7}\) | \(-2.0\times10^{-5}\) |
| 4.58 | 14 | 14 | \(-2.65\times10^{-5}\) | \(-2.5\times10^{-5}\) |

所以 \(K=40\) 的整数平台和间距不是 \(dt=0.005\) 的积分伪影。详细对照见
[K40_DT_Convergence.md](Boundary_Lattice_K_Sweep/K40_DT_Convergence.md)。

## 5. 统计边界

每个 \(K\times D\) 只有三个共同 seed，因此本报告给出描述性平台和范围，不给
\(p\) 值或 bootstrap 置信区间。已成晶样本在同一单元内的整数 \(m\) 完全一致，
足以证明本批数据中存在平台切换；但要精确定位临界 \(K_c\) 或估计形成概率，需要在
\(K=12\)--20.75 间加密并将每个单元扩展到至少 10 个 seed。

## 6. 参考文件完整性

Dispersion.py、PRL.tex 和 Methods Appendix.tex 全程只读，最终 SHA-256 与分析前
一致：

- Dispersion.py: A1FC299F4AB13F9997BDF0EBA993C6BA12054500134A8617180F572F3732B89D
- PRL.tex: A7020AB8327BDB0045F8D4A5147FC2A49FEC2732060B51AEEA3DE379C9873937
- Methods Appendix.tex: 63B20123EEDA9A1DE865A3F766D3A7973F0D7C3113E1E4E9EB3A960CDF14784F
