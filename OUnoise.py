import numpy as np


class OUnoise : 
    def __init__(self, dim:int=1, mu=None, theta=0.15, sigma=0.2, dt=1e-2, x0=None, rand_num=111) -> None:
        if mu is None : 
            self.dim = dim 
            self.mu  = np.zeros(self.dim)
        else : 
            mu = np.array(mu)
            self.dim = mu.size 
            self.mu  = mu         
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        np.random.seed(rand_num)
        self.reset()

    def reset(self) : 
        self.x_prev = self.x0 if self.x0 is not None else self.mu

    def noise(self) : 
        if isinstance(self.sigma, int) or isinstance(self.sigma, float) : 
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                np.sqrt(self.dt) * self.sigma * np.random.normal(size=self.dim)
        else : 
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                np.sqrt(self.dt) * self.sigma @ np.random.normal(size=self.dim)
        self.x_prev = x
        return x

    def noise_list(self, lenth:int) : 
        x_list = []
        for _ in range(lenth) : 
            x_list.append(self.noise())
        return x_list


'''
参数        含义              数据类型         取值范围    说明
---------   ---------------   --------------   ---------   --------------
dim         需要的噪声维度    int              >0          默认值为1 会根据mu的维度进行调整
mu          噪声的均值        np向量           --          默认值为0 优先以mu的维度为准 1维输入可以为float
theta       幅度参数          float或np向量    >=0         调整噪声的连续性 越大越不连续 接近高斯噪声的样子
sigma       方差参数          float或nd矩阵    >=0         调整噪声的取值范围 不改变噪声的整体形状 相当于乘一个系数 矩阵就是协方差
dt          时间间隔          float            >=0         相同theta dt越大噪声的连续性也越不好 因为时间相关性弱了
x0          初始取值          np向量           --          噪声的初始值 默认取值为均值mu
rand_num    随机数种子        int              >=0         随机数种子决定整个噪声序列的取值 
'''

'''
方法          功能                           输入                  输出           说明
------        ---------------------------   ------                ---------      ------
reset         初始化噪声 使其从x0重新开始    无                    无             无
noise         取噪声值                       无                    x:nd向量       无
noise_list    生成一个噪声序列               lenth:int:序列长度    x_list:list    如果外部不修改seed值 这个方法跟循环生成lenth长度的noise方法得到的结果一样
'''