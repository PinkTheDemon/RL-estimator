### 230804 总述

RL 估计器 主要参考22.Linear observer learning by temporal difference一文，把该文献中在线性系统上用 RL 学习协方差矩阵 P 的想法拓展到非线性系统。

系统动态来自于22.Moving horizon estimation for nonlinear and non-Gaussian stochastic disturbances一文。
$$
\begin{align}
&x_{1,k+1} = 0.99x_{1,k} + 0.2x_{2,k}
\\
&x_{2,k+1} = -0.1x_{1,k} + \frac{0.5x_{2,k}}{1+x_{2,k}^2} + w_k
\\
&y_k \quad\ \ = x_{1,k} - 3x_{2,k} + v_k
\\
&w_k \sim \mathcal{N}(0,1), v_k \sim \mathcal{N}(0,0.01)
\end{align}
$$
首先，明确我们的问题，是估计arrival cost。假定当前时刻为 k 时刻，时间窗口长度为 q ，那么MHE问题做的是已知窗口初始分布 $x_{k-q} \sim P(x_{k-q})$ 和窗口观测序列 $ \{y_{k-q+1},y_{k-q+2}\dots,y_{k}\} $ 的条件下，给出当前时刻的估计 $\hat{x}_{k}$ ，而arrival cost 的估计就是基于初始时刻分布 $x_0 \sim P(x_0)$ 和过去时刻的观测序列 $\{y_1, y_2, \dots, y_{k-q}\}$ ，估计窗口初始分布 $P(x_{k-q})$ 。

下一时刻 k+1 时刻，窗口往后移动，MHE问题变为已知 $P_{k-q+1}$ （ $P(x_{k-q+1})$ 的简写）和观测序列 $\{y_{k-q+2},\dots,y_{k+1}\}$ ，估计 $\hat{x}_{k+1}$ ，arrival cost 做的仍然是基于初始分布 $P_0$ 和过去的观测序列 $\{y_1,\dots,y_{k-q+1}\}$ 得到 $P_{k-q+1}$ 。

可以看到，MHE的主体优化问题部分，结构维度形式都没有发生变化。而arrival cost 的估计则是增加了一维，