import numpy as np
from scipy.integrate import solve_ivp


def f(x, u=None, disturb=None, sample_time=0.01, batch_first:bool=True) : 

    # disceret form
    # if batch_first is True : x = x.T
    # x_next = np.zeros_like(x)
    # # dynamics 1 #########################################
    # x_next[0] = 0.99*x[0] + 0.2*x[1]
    # x_next[1] = -0.1*x[0] + 0.5*x[1]/(1+x[1]**2)
    # # ####################################################
    # # dynamics 2 #########################################
    # # x_next[0] = 1.19*x[1]
    # # x_next[1] = -0.1*x[0] + 0.5*x[1]/(1+x[1]**2)
    # # ####################################################
    # x = x_next
    # if batch_first is True : x = x.T

    # continuous form
    xdot = lambda t, y : ( np.array([
        # dynamics 3 #####################################
        # -x[1],
        # -0.2*(1-x[0]**2)*x[1] + x[0]
        # ################################################
        # dynamics 4 ####################################
        10*(-y[0]+y[1]),
        28*y[0] - y[1] - y[0]*y[2],
        -8/3*y[2] + y[0]*y[1]
        # ###############################################
        ]) )
    if x.ndim == 1 : 
        x = solve_ivp(xdot, [0,0+sample_time], x).y.T[-1]
    elif x.ndim == 2 : 
        x_next = np.array([]).reshape(0, x[0].size)
        for y in x : 
            x_next = np.append(x_next, [solve_ivp(xdot, [0,0+sample_time], y).y.T[-1]], axis=0)
        x = x_next

    if disturb is not None : x = x + disturb
    return x

def F(x, u=None, sample_time=0.01) : 
    # jac 1 ##############################################
    # F = np.array([[.99, .2], 
    #               [-.1, .5*(1-x[1]**2)/(1+x[1]**2)**2]])
    # ####################################################
    # jac 2 ##############################################
    # F = np.array([[0  , 1.19], 
    #               [-.1, .5*(1-x[1]**2)/(1+x[1]**2)**2]])
    # ####################################################
    # jac 3 ##############################################
    # F = np.eye(x.size) + sample_time * \
    #     np.array([[0, -1], 
    #               [0.4*x[0]*x[1]+1, 0.2*(x[0]**2-1)]])
    # ####################################################
    # jac 4 ##############################################
    F = np.eye(x.size) + sample_time * \
        np.array([[-10    , 10  , 0    ], 
                  [28-x[2], -1  , -x[0]],
                  [x[1]   , x[0], -8/3 ]])
    # ####################################################
    return F

def f_real(x, u=None, disturb=None, sample_time=0.01, batch_first:bool=True) : 
    if batch_first is True : x = x.T
    # continuous form
    xdot = np.zeros_like(x)
    dt = 0.01
    for _ in range(int(sample_time/dt)) : 
        # dynamics 4 #####################################
        xdot[0] = 10*(-x[0]+x[1])
        xdot[1] = 28*x[0] - x[1] - x[0]*x[2]
        xdot[2] = -8/3*x[2] + x[0]*x[1]
        # ################################################
        x = x + dt*xdot
    if batch_first is True : x = x.T
    if disturb is not None : x = x + disturb
    return x

def h(x, noise=None, batch_first:bool=True) : 
    if batch_first is True : x = x.T
    # measurement equation 1 #############################
    # y = x[0] - 3*x[1]
    # ####################################################
    # measurement equation 3 #############################
    # y = x
    # ####################################################
    # measurement equation 4 #############################
    y = np.array([np.sqrt(x[0]**2+x[1]**2+x[2]**2), x[0]])
    # ####################################################
    if batch_first is True : y = y.T
    if noise is not None : y = y + noise
    return y

def H(x) : 
    # jac 1 #############################################
    # H = np.array([[1, -3]])
    # ###################################################
    # jac 1 #############################################
    # H = np.array([[1,0],[0,1]])
    # ###################################################
    # jac 4 #############################################
    y = np.linalg.norm(x)
    H = np.array([[x[0]/y, x[1]/y, x[2]/y], [1,0,0]])
    # ###################################################
    return H

def h_real(x, noise=None, batch_first:bool=True) : 
    if batch_first is True : x = x.T
    # measurement equation 4 #############################
    y = np.array([np.linalg.norm(x), x[0]])
    # ####################################################
    if batch_first is True : y = y.T
    if noise is not None : y = y + noise
    return y

# system dynamics
def step(x, u=None, disturb=None, noise=None) : 
    x_next = f(x, u=u, disturb=disturb)
    y_next = h(x_next, noise)
    return x_next, y_next

def step_real(x, u=None, disturb=None, noise=None) : 
    x_next = f_real(x, u=u, disturb=disturb)
    y_next = h_real(x_next, noise)
    return x_next, y_next


# generate noise list for ith simulation
def reset(sim_num, maxstep, x0_mu, P0, disturb_Q, noise_R, 
          disturb_mu=None, noise_mu=None) : 
    np.random.seed(sim_num)
    
    if P0.size == 0 : 
        initial_state = x0_mu
    else :
        initial_state = x0_mu + (np.random.multivariate_normal(np.zeros_like(x0_mu), P0))

    if disturb_mu is None : 
        disturb_mu = np.zeros(disturb_Q.shape[0])
    if noise_mu is None : 
        noise_mu = np.zeros(noise_R.shape[0])

    if disturb_Q.size == 0 : 
        disturb_list = np.zeros((maxstep, disturb_mu.size))
    else : 
        disturb_list = np.random.multivariate_normal(disturb_mu, disturb_Q, maxstep)
    if noise_R.size == 0 : 
        noise_list = np.zeros((maxstep, noise_mu.size))
    else : 
        noise_list = np.random.multivariate_normal(noise_mu, noise_R, maxstep)

    return initial_state, disturb_list, noise_list


'''
f
状态方程 微分方程通过RK45方法数值求解
--------------------------------------------------
输入           含义        数据类型    取值范围    说明
x              状态        ndarray     --          无
#u             控制        ndarray     --          默认无控制
disturb        扰动        ndarray     --          默认无扰动
time_sample    采样时间    float       >0          默认为0.01 连续形式的状态方程对应的采样时间
                                                   内部进行离散化的时间单元是1e-3 这个参数只是外部采样间隔 不影响方程离散化计算的精度
batch_first    批量优先    bool        bool        默认值True 指定x的格式是否为批量优先 一般的批量数据的形状都是批量优先 方便索引
--------------------------------------------------
输出    含义    数据类型    取值范围    说明
x       输出    ndarray     --          无
'''

'''
h
观测方程
--------------------------------------------------
输入           含义        数据类型    取值范围    说明
x              状态        ndarray     --          无
noise          噪声        ndarray     --          默认无噪声
batch_first    批量优先    bool        bool        默认值True 指定x的格式是否为批量优先 一般的批量数据的形状都是批量优先 方便索引
--------------------------------------------------
输出    含义    数据类型    取值范围    说明
y       观测    ndarray     --          无
'''

'''
step
步进函数 整合f和h
--------------------------------------------------
输入       含义    数据类型    取值范围    说明
x          状态    ndarray     --          无
disturb    扰动    ndarray     --          无
noise      噪声    ndarray     --          无
--------------------------------------------------
输出      含义    数据类型    取值范围    说明
x_next    状态    ndarray     --          无
y_next    观测    ndarray     --          无
'''

'''
reset
初始化函数 生成随机初始状态以及噪声序列
--------------------------------------------------
输入          含义              数据类型    取值范围    说明
rand_num      随机数种子号      int         >=0         可以用仿真序号作为随机数种子输入
maxstep       仿真步长          int         >0          生成对应长度的噪声序列
x0_mu         初始状态均值      ndarray     --          仅用于生成仿真实验中的不同初始状态 不用于估计问题
P0            初始状态协方差    ndarray     --          为空集时 初始状态不随机 确定为均值
disturb_Q     扰动协方差        ndarray     --          方差为0对应生成全0的扰动
noise_R       噪声协方差        ndarray     --          方差为0对应生成全0的噪声
disturb_mu    扰动均值          ndarray     --          默认0
noise_mu      噪声均值          ndarray     --          默认0
--------------------------------------------------
输出             含义        数据类型    取值范围    说明
initial_state    初始状态    ndarray     --          无
disturb_list     扰动序列    ndarray     --          无
noise_list       噪声序列    ndarray     --          无
'''