# 从体色散矩阵到圆盘边界算符

## 0. 结论先行

将 Dispersion.py 的体空间矩阵放入半径 \(R=D/2\) 的圆盘后，结果**不是**把
\(k\) 机械替换成 \(m/R\) 后得到另一个 \(3\times3\) 数值矩阵。平移对称性消失而旋转
对称性保留，因此每个整数角动量 \(m\) 对应一个三分量径向积分--微分本征问题；用
\(N_r\) 个径向点离散后才成为约 \(3N_r\times3N_r\) 的矩阵。

圆盘化是必要步骤，但不能仅凭这一步声称已经解释粒子实验：

1. 精确 \(\alpha=\pi/2\) 的体谱实部为零，没有体空间的唯一最快波数；
2. 粒子在墙边按实际邻居数归一化，而参考连续化用均匀体态的典型邻居数；
3. 镜面墙产生 kinetic 边界层，高角谐波在墙边未必可忽略；
4. 实验测到的是有限振幅环流晶格，可能是非均匀边界流的二次方位不稳定。

正确检验量是圆盘边缘支的

\[
\Gamma_m(D,K,\alpha)
=\max_{j\in\mathrm{edge}}\operatorname{Re}\sigma_{m,j},
\qquad
m_*=\arg\max_m\Gamma_m,
\]

而不是 \(m\simeq k_*D/2\)。

## 1. 体空间矩阵及其实空间形式

参考 [Dispersion.py](D:/PrivatePythonProject/Math/Lattice/Dispersion.py) 第 7--68 行和
[Methods Appendix.tex](D:/LaTex/Boundary%20Flow/Methods%20Appendix.tex) 第 307--370 行：

\[
M(\mathbf k)=
\begin{pmatrix}
0&-ivk_x&-ivk_y\\
-\frac{iv}{2}k_x&a(k)&b(k)\\
-\frac{iv}{2}k_y&-b(k)&a(k)
\end{pmatrix},
\]

\[
a(k)=\frac{\lambda\rho_0}{2}\widehat G(k)\cos\alpha,
\]

\[
b(k)=-\omega+\lambda\rho_0G_0\sin\alpha
-\frac{\lambda\rho_0}{2}\widehat G(k)\sin\alpha
+\frac{v^2k^2}{4D_0},
\qquad
D_0=2\omega-2\lambda\rho_0G_0\sin\alpha .
\]

令 \(J_{\rm cw}(p_x,p_y)=(p_y,-p_x)\)，对应的实空间方程为

\[
\partial_t\rho=-v\,\boldsymbol{\nabla}\cdot\mathbf p,
\]

\[
\partial_t\mathbf p
=-\frac v2\boldsymbol{\nabla}\rho+\mathcal A\mathbf p
+\mathcal B J_{\rm cw}\mathbf p,
\]

\[
\mathcal A=\frac{\lambda\rho_0\cos\alpha}{2}\mathcal G,
\]

\[
\mathcal B=
\left(-\omega+\lambda\rho_0G_0\sin\alpha\right)I
-\frac{\lambda\rho_0\sin\alpha}{2}\mathcal G
-\frac{v^2}{4D_0}\boldsymbol{\nabla}^2 .
\]

最后一项可用 \(\boldsymbol{\nabla}^2\mapsto-k^2\) 核对符号。

## 2. 圆偏振变量与总角动量

定义

\[
p_+=p_x+ip_y=2\pi f_{-1},\qquad
p_-=p_x-ip_y=2\pi f_1,
\]

\[
\partial_\pm=\partial_x\pm i\partial_y
=e^{\pm i\phi}
\left(\partial_r\pm\frac{i}{r}\partial_\phi\right).
\]

由于 \(p_\pm\) 自带自旋 \(\pm1\)，总角动量 \(m\) 的三个分量必须写成

\[
\delta\rho=R_m(r)e^{im\phi},
\qquad
p_+=P_+(r)e^{i(m+1)\phi},
\qquad
p_-=P_-(r)e^{i(m-1)\phi}.
\]

定义

\[
\mathcal D_\ell^\uparrow=\partial_r-\frac{\ell}{r},
\qquad
\mathcal D_\ell^\downarrow=\partial_r+\frac{\ell}{r},
\]

\[
\Delta_\ell=\partial_r^2+\frac1r\partial_r-\frac{\ell^2}{r^2}.
\]

则每个 \(m\) 的圆盘径向算符为

\[
\boxed{
\mathbb L_m=
\begin{pmatrix}
0&-\frac v2\mathcal D_{m+1}^{\downarrow}
&-\frac v2\mathcal D_{m-1}^{\uparrow}\\[3pt]
-\frac v2\mathcal D_m^{\uparrow}
&\mathcal A_{m+1}-i\mathcal B_{m+1}&0\\[3pt]
-\frac v2\mathcal D_m^{\downarrow}
&0&\mathcal A_{m-1}+i\mathcal B_{m-1}
\end{pmatrix}.
}
\]

因此本征问题

\[
\sigma
\begin{pmatrix}R_m\\P_+\\P_-\end{pmatrix}
=\mathbb L_m
\begin{pmatrix}R_m\\P_+\\P_-\end{pmatrix}
\]

是三分量径向问题，而不是普通的 \(3\times3\) 本征问题。

## 3. 圆盘非局域核与粒子一致归一化

在半径 \(R\) 的圆盘上，

\[
(\mathcal G_Dh)(r,\phi)=
\int_0^Rr'dr'\int_0^{2\pi}d\phi'\,
G(s)h(r',\phi'),
\]

\[
s=\sqrt{r^2+r'^2-2rr'\cos(\phi'-\phi)}.
\]

角动量仍然对角化：

\[
(\mathcal G_D h_\ell e^{i\ell\phi})(r,\phi)
=e^{i\ell\phi}\int_0^Rr'dr'\,G_\ell(r,r')h_\ell(r'),
\]

\[
G_\ell(r,r')=
\int_{-\pi}^{\pi}
G\!\left(\sqrt{r^2+r'^2-2rr'\cos\chi}\right)e^{i\ell\chi}d\chi.
\]

对 top-hat 核 \(G(s)=\mathbf1_{s\le d_0}\)，部分相交时

\[
\chi_0=\arccos\frac{r^2+r'^2-d_0^2}{2rr'},
\]

\[
G_0=2\chi_0,\qquad
G_{\ell\ne0}=\frac{2\sin(\ell\chi_0)}{\ell}.
\]

粒子模型见 [main.py](../../main.py) 第 1465--1482 行，使用实际邻居均值。定义

\[
g(r)=(\mathcal G_D1)(r),
\]

\[
\mathcal C_\ell h=
\frac1{g(r)}\int_0^Rr'dr'\,G_\ell(r,r')h(r').
\]

这保证 \(\mathcal C_0 1=1\)。对于这一粒子一致版本，

\[
D_K=2\omega-2K\sin\alpha,
\]

\[
\mathcal A_\ell=\frac K2\cos\alpha\,\mathcal C_\ell,
\]

\[
\boxed{
\mathcal B_\ell=
(-\omega+K\sin\alpha)I
-\frac K2\sin\alpha\,\mathcal C_\ell
-\frac{v^2}{4D_K}\Delta_\ell .
}
\]

若仅把 Dispersion.py 直接限制到圆盘，则应取
\(K\to\lambda\rho_0G_0\)、\(\mathcal C_\ell\to\mathcal G_\ell/G_0\)。
若未归一化理论又严格保留墙边截断，二阶谐波消去分母会变成

\[
D(r)=2\omega-2\lambda\rho_0\sin\alpha\,g(r),
\]

二阶项不能再写成常系数 \(\Delta_\ell\)。三种模型不能混用。

## 4. 镜面反射边界条件

粒子反射见 [main.py](../../main.py) 第 1492--1531 行。连续 kinetic 条件为

\[
\theta\longmapsto\pi+2\phi-\theta,
\qquad
f(R,\phi,\theta)=f(R,\phi,\pi+2\phi-\theta).
\]

因此

\[
f_n(R,\phi)=(-1)^ne^{-2in\phi}f_{-n}(R,\phi).
\]

### 一阶矩：无穿透

\[
\boxed{P_-(R)=-P_+(R),}
\]

等价于 \(p_r(R)=0\)。

### 二阶矩：闭合所需的第二条条件

由 Methods Appendix 第 300--305 行，

\[
Q_+=-i\frac{v}{2D}\partial_+p_+,
\qquad
Q_-=+i\frac{v}{2D}\partial_-p_-.
\]

结合 \(Q_-=e^{-4i\phi}Q_+\)，得到

\[
\boxed{
\left[\partial_r+\frac{m-1}{R}\right]P_-(R)
=-
\left[\partial_r-\frac{m+1}{R}\right]P_+(R).
}
\]

与第一条条件合并也可写成

\[
\boxed{
P_+'(R)+P_-'(R)-\frac{2m}{R}P_+(R)=0.
}
\]

原点正则性为

\[
R_m=O(r^{|m|}),\qquad
P_+=O(r^{|m+1|}),\qquad
P_-=O(r^{|m-1|}).
\]

## 5. 现有粒子数据已经排除的最简单解释

对 \(K=20.75,d_0=1,v=3,\alpha=\pi/2\)，已成晶样本为：

| \(D\) | 实验 \(m\) | 体谱右极限离散预测 \(m_{\rm bulk}\) | 差值 |
|---:|---:|---:|---:|
| 3.0 | 9 | 8 | +1 |
| 3.5 | 10 | 9 | +1 |
| 4.0 | 12 | 10 | +2 |
| 4.5 | 13 | 11 | +2 |
| 5.0 | 15 | 13 | +2 |

逐样本精确符合率为 0。实验波数约 \(q_{\rm edge}=5.8\)--\(6.0\)，而体谱一侧极限为
\(k_*^+=5.094669\)。所以已有数据排除了“只把 \(k\) 离散为 \(m/R\)”。

精确 \(\alpha=\pi/2\) 时，\(\mathcal A\propto\cos\alpha=0\)。在平移不变体空间中三个
本征值实部均为零；在所有空间算符共享同一自伴内积的理想未归一化圆盘压缩中，这一
中性结构也不会自动给出唯一最快 \(m\)。粒子一致的
\(g(r)^{-1}\mathcal G_D\) 与 streaming 一般不对易，原则上可能产生边界谱修正，但必须
数值求解 \(\mathbb L_m\) 后才能判断。

## 6. 怎样才算与实验对应

可证伪的圆盘谱比较至少应满足：

1. 径向网格和角积分收敛；
2. 候选本征矢量在墙边局域，而不是原点奇异或网格 Nyquist 模；
3. 对 \(D=(3,3.5,4,4.5,5)\) 给出 \(m=(9,10,12,13,15)\)；
4. 改变 \(K\) 后的 \(m\) 与 \(q=m/R_{\rm eff}\) 同受控粒子扫描一致；
5. 对闭合阶数和墙面处理的小变化稳定。

若完整归一化圆盘谱仍接近体谱的 \((8,9,10,11,13)\)，下一步应先求非均匀环流基态

\[
\bar\rho(r),\qquad \bar p_r(r)=0,\qquad \bar p_\phi(r)\ne0,
\]

再研究

\[
\delta U(r,\phi,t)=u_m(r)e^{im\phi+\sigma t},
\qquad
\sigma u_m=\mathbb J_m[\bar\rho,\bar p_\phi]u_m.
\]

这时团簇数属于“边界环流的方位调制不稳定”，而不是均匀态的直接体失稳。

## 7. \(K\) 的可区分预测

在 \(\alpha=\pi/2+0^+\) 的体谱中：

| \(K\) | \(v/K\) | \(k_*^+\) | \(2\pi/k_*^+\) |
|---:|---:|---:|---:|
| 8 | 0.3750 | 5.292991 | 1.187076 |
| 12 | 0.2500 | 5.043657 | 1.245760 |
| 20.75 | 0.1446 | 5.094671 | 1.233286 |
| 40 | 0.0750 | 5.123384 | 1.226374 |

\(v/K\) 改变 5 倍，而体谱波长只改变约 5%。受控粒子扫描可区分：

- 若终态间距仍约为 \(d_0\)，主尺度更可能由相互作用截断和非线性团簇几何控制；
- 若间距随 \(1/K\) 漂移，优先检验 \(a_{\rm eff}=a_0+b/K\)；
- 未通过成晶判据的轨迹只用于报告形成概率，不能作为波长样本。

## 8. 参考文件完整性

参考文件均为只读，分析前 SHA-256 为：

- Dispersion.py: A1FC299F4AB13F9997BDF0EBA993C6BA12054500134A8617180F572F3732B89D
- PRL.tex: A7020AB8327BDB0045F8D4A5147FC2A49FEC2732060B51AEEA3DE379C9873937
- Methods Appendix.tex: 63B20123EEDA9A1DE865A3F766D3A7973F0D7C3113E1E4E9EB3A960CDF14784F

## 9. 数值圆盘闭合的收敛检验

使用归一化 \(\mathcal C_\ell\)、上述两条镜面矩条件和 Chebyshev--Lobatto 径向离散，
对 \(N_r=41,51,61,71,81\) 做了独立求谱。边缘候选要求外侧 \(0.25d_0\) 内的加权
模态能量至少为 50%。

八个 \(K\times D\) 单元中，收敛的 \(m_*\) 数量为 **0/8**。例如：

- \(K=20.75,D=3.30\)：候选 \(m_*=13,12,9,9,8\)；
- \(K=20.75,D=4.58\)：候选 \(m_*=13,12,14,9,10\)；
- \(K=40,D=4.58\)：候选 \(m_*=12,12,17,8,15\)。

候选实增长率通常仅 \(10^{-7}\)--\(10^{-4}\)，并随网格不规则漂移；孤立的较大值在
下一分辨率消失。因此不能把其中任一个 \(m\) 当作物理预测。详细数据见
[Disk_Operator_Numerical_Diagnostic.md](Disk_Operator_Numerical_Diagnostic.md) 和
[Disk_Operator_Convergence.csv](Disk_Operator_Convergence.csv)。

这个负结果排除了“在均匀三场闭合上加圆盘积分核和两条低阶矩边界条件，就能稳定预测
实验团簇数”的说法。它并不排除完整 kinetic 圆盘谱可以产生选择；它说明墙边高角谐波
或非均匀环流基态必须进入下一层理论。
