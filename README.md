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
首先，明确我们的问题，是估计arrival cost。假定当前时刻为 k 时刻，时间窗口长度为 q ，那么MHE问题做的是已知窗口初始状态近似服从分布 $x_{k-q} \sim \hat{P}(x_{k-q})$ 和窗口观测序列 $ \{y_{k-q+1},y_{k-q+2}\dots,y_{k}\} $ 的条件下，给出当前时刻的估计 $\hat{x}_{k}$ ，而arrival cost 的估计就是基于初始时刻分布 $x_0 \sim P(x_0)$ 和过去时刻的观测序列 $\{y_1, y_2, \dots, y_{k-q}\}$ ，估计窗口初始状态PDF $\hat{P}(x_{k-q})$ 。

下一时刻 k+1 时刻，窗口往后移动，MHE问题变为已知 $\hat{P}_{k-q+1}$ （ $\hat{P}(x_{k-q+1})$ 的简写）和观测序列 $\{y_{k-q+2},\dots,y_{k+1}\}$ ，估计 $\hat{x}_{k+1}$ ，arrival cost 做的仍然是基于初始分布 $P_0$ 和过去的观测序列 $\{y_1,\dots,y_{k-q+1}\}$ 得到 $\hat{P}_{k-q+1}$ 。

可以看到，MHE的主体优化问题部分，结构维度形式都没有发生变化。而arrival cost 的估计则是增加了一维。但是在实际计算过程中，由于过去的观测序列维度较高难以处理，因此一般对arrival cost 的近似是以递归的方式进行（可以用MHE估计arrival cost吗，会有啥提升吗？），即假定前面的估计是最优的情况下，只做当前的单步估计，基于 $\hat{P}_{k-q}$ 和观测 $y_{k-q+1}$ 估计 $\hat{P}_{k-q+1}$ 。

以上都有概率意义下的详细解释，具体可以参考22.Moving horizon estimation for nonlinear and non-Gaussian stochastic disturbances一文。

那么我们所希望的是能够更快更好地得到窗口初始状态的估计（即arrival cost的估计），并且为了方便后续MHE优化问题的求解，限制其为高斯分布（对应的arrival cost为二次型）。当然如果有别的形式能更好拟合非高斯分布，并且对应的arrival cost形式也能方便呢优化问题的求解，那就更好了。也就是说，在 k 到 k+1 步，强化学习智能体需要基于 $\{\hat{x}_{k-q},\hat{P}_{k-q},y_{k-q+1}\}$ （即，上一时刻分布的估计，由于限定形式为高斯分布，所以对应到均值和协方差，以及这一时刻的观测），给出初始状态分布的估计（均值由非线性系统动态给出，实际上只需要给出协方差矩阵 $\hat{P}_{k-q+1}$ ）。这其实就是一个单步估计的问题，所以我们可以先把它剥离出MHE的体系，只是基于 $\{\hat{x}_{k},\hat{P}_{k},y_{k+1}\}$ 来估计 $\hat{P}_{k+1}$ 。



### 20230804 系统动态

step函数：{

功能：系统动态方程，由当前时刻状态得到下一时刻状态

输入：x—2维k时刻状态量，disturb—1维过程扰动项，noise—1维测量噪声

输出：x_next—2维k+1时刻状态量，y—1维k+1时刻观测量

备注：使用时需注意，相同的随机数种子会生成相同的噪声，因此在不同的时间步需要不同的随机数种子输入。并且在不同次的仿真过程中，随机数种子的顺序应该不同。由于需要保证可复现性，因此噪声序列对于指定的一次仿真过程应该是确定的，即指定仿真序号 i 后，噪声序列就确定了。那么在仿真执行过程中，从头到尾需要保存一个噪声序列。

}

gen_noise函数：{

功能：为第 i 次仿真过程生成噪声序列

输入：sim_num—整数型仿真序列号，maxstep—仿真最大时间步长，#disturb_mu—扰动项均值，#disturb_P—扰动项方差，#noise_mu—噪声项均值，#noise_Q—噪声项协方差

输出：disturb_list—扰动序列，noise_list—噪声序列

备注：#为可选参数。目前扰动和噪声都是一维的，所以PQ都是浮点数，而不是协方差矩阵，后续如果有需要再做修改。

}