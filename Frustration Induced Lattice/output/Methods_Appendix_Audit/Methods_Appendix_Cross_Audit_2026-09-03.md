# Methods Appendix：理论—代码多智能体交叉审查

日期：2026-09-03。结论：**归一化补充通过；全文尚未通过，存在重大问题。**

## 1. 审查范围、版本与修改边界

规范源为 [Methods Appendix.tex](<D:/LaTex/Boundary Flow/Methods Appendix.tex>)，并参照 [PRL.tex](<D:/LaTex/Boundary Flow/PRL.tex>)。覆盖附录全部理论/算法章节、粒子模拟实现、边界统计及晶格尺度分析，以及另一个本地目录中的色散/Chern/strip 实现。

三名专项审查者分别负责理论、拓扑实现和粒子/统计方法，主审通读全文、逐项复核并独立重跑核心反例。EP 反例经理论、拓扑和主审重复确认；strip 计数及伪 line-gap 反例经拓扑和主审重复确认。数据核对的范围在下文明确限定，不将元数据通过误称为全部轨迹重算通过。

最终反向复核：理论、拓扑、粒子/统计三位审查者均已明确确认本报告通过复核。最后一轮收紧了谐波截断、端点渐近、UV 反例及 W 阈值的适用条件。**这表示报告的证据与结论通过交叉核验，不表示论文或生产代码已通过审查。**

按用户最终授权，只补充归一化说明；未修正下列其他问题，未修改生产代码、PRL.tex 或原始数据。审查用脚本属于独立输出，不改变生产算法。

- 原附录：2314 行，SHA-256 C85CA4C50C8BD2C667BFB698A9979567DAC69C3457CBC185CEE82931E511CE41。
- 修改后：2358 行，SHA-256 8305CD3BE1834B5494CB2055A7DE8E367722B6C26C4754DA9E46448C1B03398B。
- 唯一正文改动：原第104行后新增44行；本报告所有附录链接均使用修改后的行号。
- [原稿备份](<D:/PythonProject/System Theory/Frustration Induced Lattice/output/Methods_Appendix_Audit/Methods Appendix.before-normalization.tex>)。
- [只读复验脚本](<D:/PythonProject/System Theory/Frustration Induced Lattice/output/Methods_Appendix_Audit/reproduce_audit_probes.py>)。

### 已通过的归一化补充

[新增说明](<D:/LaTex/Boundary Flow/Methods Appendix.tex:106>)保留全文数密度约定 \(\rho=\int f\,d\theta\)，明确连续均匀参考背景中的面积平均为
\[
\frac{1}{\widehat G(0)}\int G\,d^2x'\int(\cdots)\frac{f}{\rho_0}\,d\theta',
\qquad
\lambda=\frac{K}{\rho_0\widehat G(0)}
=\frac{K}{\rho_0\pi d_0^2}\simeq\frac{K}{N'}.
\]
因此 \(\lambda\rho_0\widehat G(0)=K\)、\(\omega=0\) 时 \(D_0=-2K\sin\alpha\)。这与 [PRL 第92行](<D:/LaTex/Boundary Flow/PRL.tex:92>)及实际谱分析参数构造一致。没有把数密度 \(f\) 静默改为归一化概率密度。

补文也明确：冻结均匀参考邻居数与保留瞬时局部人数，在均匀无极化体态线性阶一致，但一般不能在非线性密度变化或截断边界邻域中视为完全相同的模型。

新增段落经独立理论审核通过。XeLaTeX 两轮编译通过，交叉引用解析正常，无 Overfull/Underfull 告警；保留原有 inputenc 在 XeTeX 下被忽略的提示。仅生成审查目录中的 XDV/辅助文件，未更新出版 PDF。

## 2. 重大问题：暂不能通过

### P1-1　二阶谐波非线性闭合有两处确定的符号错误

定位：[Qx/Qy 闭合](<D:/LaTex/Boundary Flow/Methods Appendix.tex:283>)，具体第287、298行；依据为[原模方程](<D:/LaTex/Boundary Flow/Methods Appendix.tex:188>)。

令 \(z=p_x+ip_y\)，在附录已经采用的 \(|m|\ge3\) 截断下，由其傅里叶约定及模方程得到
\[
\partial_tQ=-\frac v2(\partial_{x_1}+i\partial_{x_2})z
+iDQ+\lambda e^{i\alpha}(\mathcal Gz)z.
\]
若 \((\mathcal Gz)z=A+iB\)，正确代数解为
\[
Q_x=\frac{\frac v2(\partial_{x_1}p_y+\partial_{x_2}p_x)
-\lambda A\sin\alpha-\lambda B\cos\alpha}{D},
\]
\[
Q_y=\frac{-\frac v2(\partial_{x_1}p_x-\partial_{x_2}p_y)
+\lambda A\cos\alpha-\lambda B\sin\alpha}{D}.
\]
这里 A、B 仅为本报告的代数缩写，不建议作为新增正文符号。

现稿 Qx 的 \(A\sin\alpha\) 符号、Qy 外层负号内的 \(A\cos\alpha\) 符号均相反。

最小反例：无空间梯度，\(\widehat G(0)=1,\rho=4,\lambda=1,\omega=0,\alpha=\pi/2,z=1\)，故 \(D=-8\)。正文给 \(Q=-1/8\)，代回模方程残差为 \(2i\)；正确 \(Q=+1/8\) 给零。主审复验的残差绝对值分别为 2.0 和 0.0。

影响：非线性闭合及饱和解释需要纠正；**不会改变后文纯线性色散矩阵**，因为错误项均为极化的二阶项。

### P1-2　有限波数 EP 使部分开区间参数的单带 Chern 数无定义，代码却仍返回整数

定位：[极点推导及其全局 gap 前提](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1930>)；[Chern 网格及匹配](<D:/PrivatePythonProject/Math/Lattice/ChernNumberCompute.py:273>)；[无条件取整返回](<D:/PrivatePythonProject/Math/Lattice/ChernNumberCompute.py:484>)。

取与圆边界均匀参数映射一致的
\[
v=3,\quad \omega=0,\quad
\lambda\rho_0\widehat G(0)=20.75,\quad d_0=1,\quad \alpha=0.99\pi .
\]
此时 \(D_0=-1.3035465017\ne0\)，但：

- 原点谱为 \(0,-10.36988056\pm0.32588663i\)。
- \(b(k)=0\) 于 \(k=0.4396928321\)，此处三根全实：
  \(-0.0866984983,-10.0345917969,-10.1212902952\)。
- 三次特征多项式判别式在 \(k=0.4132665534\) 和 \(0.4722166740\) 为零；原先上下虚部支在有限 k 发生并合。

特征多项式可直接写为
\[
\sigma^3-2a\sigma^2+
(a^2+b^2+v^2k^2/2)\sigma-a v^2k^2/2.
\]
因此这不是仅凭“虚部排序不方便”提出的疑虑，而是实际的有限 k 谱带合并。在 \(\omega=0\)、固定 \(K>0,v>0\) 及有限 \(d_0>0\) 的盘核模型中，足够靠近 \(\alpha=0,\pi\) 都有类似机制：令 \(s=\sin\alpha\to0\)，则 \(b=0\) 的小根满足 \(k=2Ks/v+O(s^3)\)，且 \(a^2-2v^2k^2=K^2/4+O(s^2)>0\)。

现有扫描 \(Q=60,N_\theta=71,\delta=10^{-3}\) 的第一个非零 k 已是 1.3461941，会整个越过上述 EP 区。函数仍给
\[
C_{\rm raw}=1.3314\times10^{-11},\quad C_{\rm int}=0,
\]
并报告 min_sigma=0.8631、unresolved_bad=0。min_sigma 是左右本征矢重叠矩阵的奇异值，**不是目标谱簇与补谱之间的间隙**；诊断中根本没有 gap-valid 标志。

边界：附录第1968–1970行已经把极点结论限定于全局有隙情况，所以条件公式本身不是错误。错误在于未查明该前提实际成立的参数范围，以及数值代码会对失去隔离的单带仍输出貌似合法的整数。把合并的两带作为一个簇可以改变问题的定义，但不能自动保留原先非零的单带/互补簇 Chern 数。

通过要求：先验证有限 k 的 target/complement 分离，细化小 k 网格；无定义处返回 NA/invalid；只在有效参数域讨论对应投影的 Chern 平台。

### P1-3　strip 所用的 Im σ=±10 穿过体谱，不是该计算的 bulk line gap

定位：[bulk line-gap 定义](<D:/LaTex/Boundary Flow/Methods Appendix.tex:2125>)；[报告的 ±10 计数](<D:/LaTex/Boundary Flow/Methods Appendix.tex:2268>)；[实际 strip 参数](<D:/PrivatePythonProject/Math/Lattice/SpectralFlow.py:312>)。

实际默认参数是 \(v=3,\omega=1.5,\lambda\rho_0G_0=20,d_0=2,\alpha=\pi/2\)。独立求根给
\[
k_x=0,\quad k_y=0.5523194411551681,\quad
\operatorname{Spec}M\simeq\{0,+10i,-10i\}.
\]
SpectralFlow 的独立矩阵与 Dispersion 的矩阵在该点逐元素差为 0。因此两条参考线都直接穿过体谱，违反附录自己的“所有有限 k 均不得相交”定义。仅将 \(\omega\) 改为 0 也不能解决：此时原点谱已含 \(\pm10i\)。

**离散计数本身复现成功**：完整默认 strip 对两条线均给 \((\mathrm{SF}_L,\mathrm{SF}_R)=(2,-2)\)。不能报告为“无法复现数值”；应报告为“该数值不能据此充当同一 bulk line gap 的体边对应检验”。

文献中的 line-gap 分类确实以谱中的空线区域为前提，而不是任选一条画图参考线：[Kawabata 等，PRX 2019](https://arxiv.org/abs/1812.09133)。

### P1-4　D0≠0 不是快弛豫条件；低频闭合的 UV 外推不足以证明微观拓扑

定位：[relaxation denominator](<D:/LaTex/Boundary Flow/Methods Appendix.tex:44>)；[fast variables 与绝热消元](<D:/LaTex/Boundary Flow/Methods Appendix.tex:227>)；[无穷远紧化](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1716>)。

均匀态在 \(k=0\) 的二阶谐波线性本征值是 \(\mp iD_0\)，没有负实部、没有弛豫。\(D_0\ne0\) 只保证零频代数方程可逆。动态消元实际涉及 \((\sigma\mp iD_0)^{-1}\)，还需要低频/尺度分离条件，例如 \(|\sigma|/|D_0|\ll1\)、\(vk/|D_0|\ll1\)，并处理不会自动衰减的自由二阶谐波。现稿没有证明这些控制条件。

这不等于宣布振荡平均闭合不可能；它意味着当前“controlled”“fast relaxation”措辞证据不足。

更关键的是，无穷远 Chern 极点恰由消元产生的 \(k^2/D_0\) 项决定，却在 \(k\to\infty\) 使用了该低频闭合。严格反例：取主参数设定 \(\omega=0,K=\lambda\rho_0\widehat G(0)>0,v>0\) 及 \(\alpha=\pi/2\)，将
\[
b(k)\mapsto b(k)+\gamma k^4,\qquad \gamma>0.
\]
它不改变截至 k² 的长波展开，但改变无穷远极点。三根仍为
\[
0,\quad \pm i\sqrt{[b(k)+\gamma k^4]^2+v^2k^2/2},
\]
在任意有限 k 均分离；下虚部带的极点 Chern 数可由 −2 变为 0。变化来自 UV 完成，而非有限 k 间隙关闭。

结论边界：给定三场矩阵及其有效隔离/紧化条件时，极点计算是正确数学结果；目前尚不能将它无条件提升为粒子模型或完整动理学的拓扑不变量。需要披露有效理论/UV 假设，或补充更高角谐波及微观兼容 UV 完成的验证。

### P1-5　高 α 的 W gate 强烈条件化手性，Xi_Sign≈1 不能作为独立的无偏保护性检验

定位：[W 定义及近切向限制](<D:/LaTex/Boundary Flow/Methods Appendix.tex:637>)；[保护性解释](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1125>)；[代码筛选](<D:/PythonProject/System Theory/Frustration Induced Lattice/phase_informed_boundary_flow_analysis.py:507>)。

公式与实现相符，但筛选不保持两种手性的对称进入。直边上 \(W=-\Omega q\)；若 \(\Omega<0\)，则 \(W>\varepsilon_W\) 等价于 \(q>\varepsilon_W/|\Omega|>0\)：负 q 必然被排除，正 q 也须超过该幅值阈值才能入选，然后再对筛选后的 q 计算手性稳定性。

对 \(\alpha=0.6\pi\)，在每个实际终窗内均匀抽取7帧：

| 几何 | 实际终窗时间 | W前候选 q+/q− | W后候选 q+/q− |
|---|---:|---:|---:|
| 圆 | 73.15 | 3021 / 3745 | 3021 / 0 |
| 单缺陷圆 | 91.35 | 4011 / 5045 | 4011 / 0 |
| 方形 | 93.10 | 3241 / 3668 | 3241 / 0 |
| 四缺陷方形 | 114.10 | 3077 / 3681 | 3077 / 0 |

这是稀疏抽样验证，**不是全窗无 W 的 headline 重算**。筛选后还有31–34%的候选满足 |q|<0.5，未真正实施加速度解释要求的近切向条件。

\(\eta_W\) 统计 cell occupancy：同一格内负 q 被正 q 替换后 occupancy 仍可接近1，不能显露上述方向条件化。

通过要求：明确这是“通过 W 的条件统计”；补无 W 对照、按符号分解的保留率、近切向子集敏感性。此问题削弱该统计作为独立保护证据的力度，**并不凭此推翻实际粒子流的存在或方向**。

## 3. 实质性文码差异与可复现性问题

### P2-1　相关函数的“先平均再取峰”与“先取峰再平均”不一致

定位：[附录定义](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1197>)；[逐帧相关与取峰代码](<D:/PythonProject/System Theory/Frustration Induced Lattice/boundary_arc_correlation_analysis.py:84>)；[逐帧结果汇总](<D:/PythonProject/System Theory/Frustration Induced Lattice/boundary_arc_correlation_analysis.py:201>)。

文稿先定义 \(\langle A(s,t)\rangle_t/\langle A(0,t)\rangle_t\)，再对时间平均相关函数找峰；代码每帧归一化相关、用该帧平均半径换算弧长、每帧找峰，最后平均峰位置。这些非线性操作一般不能交换。

真实数据例：\(\alpha=\pi\)、seed 8 的 CCW stream，现算法给 1.3979492409d0；按文稿先平均相关再取峰、使用同一找峰算法及平均半径，给 1.4276458958d0，相差0.0296966549d0（约2.12%）。

若保留现有数据，应将文稿改成逐帧相关 \(C_f(s,t)\)、逐帧峰 \(a_f(t)\)，再报告其时间均值和样本标准差（ddof=1）。不要静默改变算法以迁就文稿。

### P2-2　奇异 α=π 目前被实际计算，不是明确排除后的绘图占位

定位：[附录占位声明](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1606>)；[ChernParamScan 的端点配置](<D:/PrivatePythonProject/Math/Lattice/ChernParamScan.py:33>)；[无排除分支的计算与绘图](<D:/PrivatePythonProject/Math/Lattice/ChernParamScan.py:226>)。

代码把 π 纳入扫描，直接调用 compute_topology 并绘制返回值。浮点 sin(π) 不为精确零，导致 \(D_0=-4.8985872\times10^{-15}\)，k=1 的矩阵最大元素约 \(4.5932\times10^{14}\)，函数仍返回 C≈0。现有零值是对奇异输入数值运算的结果，不能描述为代码明确排除计算后的显示占位。

通过要求：对理论奇异参数显式判 invalid/NA，将计算状态与图形占位分开。

### P2-3　无穷远 cap 按 band index 硬编码，泛参数调用可能给错 Chern 数

定位：[默认无穷远基底](<D:/PrivatePythonProject/Math/Lattice/ChernNumberCompute.py:333>)。

band 0 永远使用 \(u_+\)，未由 \(D_0\)、连续分支和极点自适应决定。反例 \(v=3,\omega=30,K_{\rm eff}=20,d_0=1,\alpha=\pi/2\)：D0=20，正确下虚部无穷远极点为 \(u_-\)，单带全程分离。默认函数给 C≈0，仅通过现有 infty_basis 接口传入 \(u_-\) 后给 C=2.0000000000001017。

该特定反例不属于当前 ω=0 的主参数范围；它证明通用函数及通用算法叙述具有未声明的参数域限制，不证明所有主图数值因此错误。

### P2-4　strip 参数不是圆边界粒子对照参数；单组计数尚未构成收敛验证

定位：[附录复现参数](<D:/LaTex/Boundary Flow/Methods Appendix.tex:2262>)；[生产 strip 默认参数](<D:/PrivatePythonProject/Math/Lattice/SpectralFlow.py:312>)；[圆边界参数](<D:/PythonProject/System Theory/Frustration Induced Lattice/boundary_defect_analysis.py:49>)。

| 参数 | 圆边界粒子对照 | strip 默认算例 |
|---|---:|---:|
| v | 3 | 3 |
| ω | 0 | 1.5 |
| K 或 λρ0G0 | 20.75 | 20 |
| d0 | 1 | 2 |
| α | 多点扫描 | π/2 |

附录列出了数值截断参数，但没有在该段披露这组不同的物理参数。它至多是另一参数算例，不能默认视为相同粒子参数的验证。

固定其余数值参数的单参数敏感性复跑：

| kc | ky截止 | Ny | (SF左,SF右)，c=10 |
|---:|---:|---:|---:|
| 40 | 50 | 101 | (2,−2) |
| 40 | 25 | 51 | (1,−1) |
| 80 | 50 | 101 | (1,−1) |
| 20 | 50 | 101 | (0,0) |

这表明有限计数依赖截止，不能取代联合收敛测试。改变 kc 也改变辅助格距和条带物理宽度，故**不能由此断言连续极限不存在**。需要按固定物理宽度/边缘宽度等有意义的联合方案检查 Nx、kc、Nk、Rmax、ky截止和步长。

### P2-5　正文要求的退化子空间跟踪与双侧边缘标签条件未实现

定位：[附录数值流程](<D:/LaTex/Boundary Flow/Methods Appendix.tex:2248>)；[实际逐列匹配](<D:/PrivatePythonProject/Math/Lattice/SpectralFlow.py:129>)；[交叉分类](<D:/PrivatePythonProject/Math/Lattice/SpectralFlow.py:253>)。

- 代码始终逐本征矢做 Hungarian 匹配，没有在退化/近退化处切换为谱投影或子空间跟踪，也未实现文中要求的相应奇异值检查。
- 流程第2251–2252行要求 crossing 两侧边缘标签均持续存在；代码只检查两侧平均边缘权重。反例 \(W_L=(0.9,0.01),W_R=(0.01,0.01)\)，σ从−i到+i，η=0.45：代码仍计左边交叉1。
- 附录第2160行的“平均权重”反而与代码一致，因此工作流与前文也存在内部冲突。

### P2-6　晶格尺度提取缺少决定结果的操作与阈值

定位：[附录涡旋中心距离说明](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1250>)；[中心识别](<D:/PythonProject/System Theory/Frustration Induced Lattice/alpha06_bulk_hex_lattice_analysis.py:174>)；[第一壳层](<D:/PythonProject/System Theory/Frustration Induced Lattice/alpha06_bulk_hex_lattice_analysis.py:233>)。

0.6π 实际流程还包括：由自由 Ω 求瞬时旋转中心；\(|\Omega|\ge0.2K|\sin\alpha|\)；中心半径≤R+0.2d0；DBSCAN 的 ε=v/(K|sinα|)、min_samples=10；至少15粒子的簇；坐标中位数中心；排除距壁<0.5d0者；保留距离≤该帧中位最近邻距离×1.45的无序对；每帧先平均再等权时间平均。

相关函数还用2048个角度 bins、密度 Gaussian smoothing σ=2.5 bins、相关 smoothing σ=2.0 bins。π 家族分离还使用
\(\beta=\frac12\arg\langle e^{2i(\theta-\varphi)}\rangle\) 和 \(|\cos(\theta-\varphi-\beta)|\ge0.5\)。

这些是估计量定义的一部分，不能只以“第一壳层邻近涡旋中心距离”或“两轴向家族”代替。DBSCAN 用于定位中心与“没有使用 cluster-number observable”不矛盾；后一句不属于错误。

### P2-7　相关峰高度的抛物线插值有因子2错误，但未发现改变当前分类

定位：[峰插值](<D:/PythonProject/System Theory/Frustration Induced Lattice/boundary_arc_correlation_analysis.py:117>)。

峰位置 offset 正确，峰高应为
\[
y_M-\tfrac14(y_L-y_R)\,\mathrm{offset},
\]
而代码使用系数1/2，使插值增益翻倍。例如三点(0.9,1,0.8)的正确峰高是1.0041667，代码给1.0083333。

这是确定的代码错误，但当前 resolved/unresolved 分类在峰高阈值0.25–0.35内稳定，尚无证据表明此小误差改变现有分类或峰位置。

## 4. 次要但确定的表述/数值问题

1. [第822行](<D:/LaTex/Boundary Flow/Methods Appendix.tex:822>)称 fT<1 直接说明轨迹不足；完整文件也因舍去不满一块的余数而略小于1。实现以≥0.995作图形分界，应披露离散余数。
2. [第727–730行](<D:/LaTex/Boundary Flow/Methods Appendix.tex:727>)的 dmax 分段漏掉 α=π/2；代码在该点使用低α分支，第二个条件应包含等号。
3. [第1060–1061行](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1060>)最小 fcal 数字不准确：square 0.3π 实际为0.875000，单缺陷圆0.1π为0.875520；不能统一写最小0.876。
4. [第1147行](<D:/LaTex/Boundary Flow/Methods Appendix.tex:1147>)的 α=0.2 漏 π，应为0.2π。
5. [第990–993行](<D:/LaTex/Boundary Flow/Methods Appendix.tex:990>)的 finite-values 检查范围写得过大：所有文件检查结构；有限值仅检查参与计算的所读终窗；两个端点在加载终窗前返回。并未检验44文件的全部历史帧。

## 5. 已核对一致的内容与范围

| 内容 | 审查结果 |
|---|---|
| 角傅里叶约定、对流/对齐模方程、ρ/p/Q定义 | 一致 |
| 未消元密度—极化方程、Q的线性梯度项、M各矩阵元 | 一致；不受P1-1非线性错误影响 |
| 盘核傅里叶变换、旋转协变、生成元、群速度负号 | 一致 |
| 相锁半径/直径公式 | 在ω=0、局部完整均匀相位采样等已声明假设下成立；本身是现象学ansatz |
| Riesz、双正交规范、投影曲率、极点公式 | 数学形式一致；必须保留隔离/紧化前提 |
| determinant-link Fukui 基本操作 | 与公式基本一致；不能以整数化替代gap验证 |
| Q→Y→Z、最近代表、并列置零、同粒子校准/不同粒子relay | 一致 |
| block=7、0.35时间、整块终窗、等权active-block均值、NA规则 | 一致 |
| 36行72个headline值与现有CSV四位小数 | 全部一致 |
| 44个精确参数HDF5路径、schema、帧数 | 44/44一致；不是全部历史帧finite复检 |
| 32个近完整窗/4个短窗、RHS最大误差1.0658e−14、no-cross-join | 与输出一致 |
| 十种子范围、均值、4.8%/10.3%/4.5%、未解析种子标识 | 与当前CSV/JSON一致 |
| 三图统一纵轴跨度0.50d0、时间标准差而非SEM | 一致 |
| Toeplitz块、右矢边缘权重、Hungarian成本、离散交叉符号 | 基本一致；默认(2,−2)计数可复现 |

Fukui 方法本身是离散规范不变的 Chern 计算方法；这里的问题不是否定该方法，而是输入的谱投影、gap、紧化和网格是否满足适用条件：[Fukui、Hatsugai、Suzuki 2005](https://arxiv.org/abs/cond-mat/0503172)。

未执行：重新生成所有2000粒子长时间轨迹、全部历史HDF5有限值遍历、完整相图的EP/谱隙边界定位、所有参数的联合strip收敛极限、更高角谐波动理学误差验证。报告不将这些未执行项目描述为通过。

## 6. 通过门槛与下一步权限

审查结论已收敛，但论文/代码不能诚实标为“全部通过”。当前未解决问题不会因重复同一审查自动消失。

优先顺序建议：

1. 确认后修正 Q 闭合的两个符号；同步区分非线性与线性影响。
2. 确定有效单带/谱簇区间，明确排除奇异端点和EP；补计算状态。
3. 重建真实有隙条件下的体边比较，或将现有strip结果降为有限截止离散计数；披露实际物理参数。
4. 明确闭合/UV的有效理论定位；补充W条件性与对照。
5. 统一相关峰估计量定义，补全测量参数，修正其余小错。

除归一化说明外，以上均等待用户批准后再改。任何需要重新计算的图表，应保留现有原始数据与旧输出，不通过删去不利样本或改变判据来获取“通过”。
