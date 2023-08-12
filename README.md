### 230804 总述

RL 估计器 主要参考22.Linear Observer Learning by Temporal Difference一文，把该文献中在线性系统上用 RL 学习协方差矩阵 P 的想法拓展到非线性系统。

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

后续是代码说明文档，在代码文件中，双井号（##）索引的注释一般是未解决的疑问，需要不时进行回顾。



### 20230804 系统动态

dynamic.py

step函数：{

功能：系统动态方程，由当前时刻状态得到下一时刻状态

输入：x—2维k时刻状态量(ndarray, (2, ))，disturb—2维过程扰动项(ndarray, (2, ))，noise—1维测量噪声(float64)

输出：x_next—2维k+1时刻状态量(ndarray, (2, ))，y_next—1维k+1时刻观测量(float64)

备注：使用时需注意，相同的随机数种子会生成相同的噪声，因此在不同的时间步需要不同的随机数种子输入。并且在不同次的仿真过程中，随机数种子的顺序应该不同。由于需要保证可复现性，因此初始状态和噪声序列对于指定的一次仿真过程应该是确定的，即指定仿真序号 i 后，初始状态和噪声序列就确定了。那么在仿真执行过程中，从头到尾需要保存一个噪声序列。

}

reset函数：{

功能：为第 i 次仿真过程生成初始状态和噪声序列

输入：sim_num—仿真序列号(int)，maxstep—仿真最大时间步长(int)，#x0_mu—初始状态均值(ndarray, (2, ))，#P0—初始状态协方差矩阵(ndarray, (2,2))，#disturb_mu—扰动项均值(ndarray, (2, ))，#disturb_Q—扰动项协方差矩阵(ndarray, (2,2))，#noise_mu—噪声项均值(float)，#noise_R—噪声项协方差(float)

输出：initial_state—初始状态(ndarray, (2, ))，disturb_list—扰动序列(ndarray, (200,2))，noise_list—噪声序列(ndarray, (200, ))

备注：#为可选参数。目前扰动和噪声都是一维的，所以PQ都是浮点数，而不是协方差矩阵，后续如果有需要再做修改。disturb_list是从w0开始的，而noise_list是从v1开始的，即disturb_list[0]=w0，noise_list[0]=v1。disturb_mu和disturb_Q的默认值偷懒没写成ndarray类型而是直接用的list型，但是传进去ndarray型也能正常运行。

}



### 20230805 EKF和simulation

estimator.py

inv函数：{

功能：矩阵求逆

输入：M—矩阵(ndarray, (n,n))

输出：M矩阵的逆(ndarray, (n,n))

备注：由于numpy提供的矩阵求逆的方法不能对1维矩阵求逆，并且有两个前缀写起来不方便，因此简单整合了一下。矩阵M的维度一定要是n*n，即使n=1，因为里面的判断条件是M.shape==(1,1)，而且一维矩阵求逆肯定是为了矩阵运算的，在矩阵运算中也会要求矩阵有两个维度，而不是只有一个维度。

}

EKF函数：{

功能：EKF做单步估计

输入：state—当前状态（的估计）(ndarray, (2, ))，P—当前状态的协方差矩阵（的估计）(ndarray, (2,2))，obs_next—下一时刻的观测(float64)，#Q—过程噪声协方差矩阵(ndarray, (2,2))，#R—测量噪声方差(ndarray, (1,1))

输出：state_hat—下一时刻状态的估计(ndarray, (2, ))，P_hat—下一时刻状态的协方差矩阵的估计(ndarray, (2,2))

备注：预测和更新都是直接照搬的12.A Fresh Look at the Kalman Filter P12最下方的公式。做出来的效果与22.Moving horizon estimator for nonlinear and non-Gaussian stochastic disturbances中P243 case Ia接近，初步认为写的EKF没什么大问题。

}

RL_estimator.py

simulate函数：{

功能：进行一次状态估计的仿真过程

输入：args—一些全局的参数，sim_num—仿真序号（对应该次仿真的随机数种子）(int)，STATUS—指明用哪种估计器（EKF、NLS-EKF、NLS-RLF或者train）(str)

输出：none

备注：当STATUS!='train'时，会绘制状态估计的图以及误差的图。

}



### 20230808 非线性最小二乘估计器

estimator.py

block_diag函数：{

功能：将多个矩阵拼成一个块对角矩阵

输入：matrix_list—矩阵列表(list)

输出：bd_M—块对角矩阵(ndarray)

备注：矩阵列表中每一个元素都是一个ndarray，对其维度没有限制。块对角矩阵是按照从左上到右下来排列列表中的矩阵，实际使用时，可以直接将多个矩阵用方括号括起来作为输入，例如[A, B, C]。

}

NLSF函数：{

功能：非线性最小二乘求解单步优化问题得到状态估计

输入：state_mu—k时刻状态估计(ndarray, (2, ))，P—k时刻状态协方差估计(ndarray, (2,2))，obs_next—k+1时刻观测(float64)，Q—（k+1时刻）过程噪声协方差矩阵(ndarray, (2,2))，R—（k+1）时刻测量噪声协方差矩阵(ndarray, (2,2))

输出：result.x—状态估计值(ndarray, (4, ))

备注：输出的是k时刻状态估计的修正值（前2维）以及k+1时刻状态的估计值（后2维）。基于EKF估计P矩阵的NLS-EKF方法效果比单纯EKF好一点，目前初步认为NLSF函数写的没问题。

}

residual_fun函数：{

功能：非线性最小二乘求解器中需要的残差向量

输入：x—待优化变量(ndarray, (4, ))，state_mu、P、obs_next、Q、R—NLSF函数的输入直接传过来的。

输出：f—残差(ndarray, (5, ))

备注：残差函数实际上是
$$
[\hat{x}_k - \bar{x}_k, \hat{x}_{k+1} - f(\hat{x}_k), y_{k+1} - h(\hat{x}_{k+1})]
diag(P_k^{-1}, Q_{k+1}^{-1}, R_{k+1}^{-1})
\triangle ^\top
$$
所以残差向量是前面这一串向量乘上中间的对角矩阵的cholesky分解。

}

jac_fun函数：{

功能：非线性最小二乘求解器中需要的雅可比矩阵

输入：x—待优化变量(ndarray, (4, ))，state_mu、P、obs_next、Q、R—NLSF函数的输入直接传过来的。

输出：J—雅可比矩阵(ndarray, (5,4))

备注：第一行是残差向量第一个分量关于四个决策变量求偏导，后续同理。

}

同时对之前的内容进行了一些修订。另外，由于RL部分好像与常规DDPG算法不太相同，在本例中Critic函数好像是有解析的表达形式，而不是像一般的算法中需要用另一个神经网络去拟合，因此关于RL部分的代码暂时不在本文档中进行说明。

在class中，如果在init方法中指定随机数种子，那么反复进入该类的其余方法时，会生成不同的随机数，但这组随机数是可复现的，也就是对应于init方法中指定的随机数种子。因此可以把dynamic.step函数改成class中的一个方法，可能是更简单的写法。



### 20230812 代码修正

对之前的代码中的部分错误进行修正：

- estimator.py中对R的初始值设定都是1，args中对R的默认值也是1，然而实际上应该是0.01。

但是修正之后，NLS-EKF突然效果变得很差，检查了一下jac矩阵应该也是没问题的，不过实际上把非线性最小二乘中的jac参数删去之后，效果就跟原来一样好了，这一点让我很疑惑。