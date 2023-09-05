import numpy as np

def f(x, disturb=[]) : 
    # nolinear dynamics equation
    xdot = np.zeros_like(x)

    Ts = 0.1
    dt = 0.01
    for _ in range(int(Ts/dt)) : 
        xdot[0] = -x[1]
        xdot[1] = -0.2*(1-x[0]**2)*x[1] + x[0]
        x = x + dt*xdot

    if len(disturb) == 0 : disturb = np.zeros_like(x)
    x = x + disturb
    return x

def h(x, noise=[]) : 
    # measurement equation 
    y = x
    if len(noise) == 0 : noise = np.zeros_like(y)
    y = y + noise
    return y

# system dynamics ## 我感觉这个系统动态可能需要更换一个更合适的例子
def step(x, disturb=[], noise=[]) : 
    x_next = f(x, disturb)
    y_next = h(x_next, noise)

    return x_next, y_next

# generate noise list for ith simulation
def reset(sim_num, maxstep, x0_mu, P0, disturb_Q, noise_R, 
          disturb_mu=None, noise_mu=None) : 
    np.random.seed(sim_num)
    
    if P0.size == 0 : 
        initial_state = x0_mu
    else :
        initial_state = np.random.multivariate_normal(x0_mu, P0)

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


# if __name__ == '__main__' : 
    # x0, wlist, vlist = reset(1,10)
    # print(wlist,'\n',vlist) # 测试噪声序列能否正常生成——能

    # x = [1,2]
    # x_next,y = step(x,wlist[0],vlist[0])
    # print(x,y,x_next) # 测试step正常功能——无误
    # x_next,y = step(x,wlist[1],vlist[1]) # 测试反复进入step会不会得到相同的噪声——不会
    # print(y)

    # state = np.concatenate((x, [y])) # 测试拼接数组——可行
    # print(state)

    # 测试相同随机数种子会不会采样到相同的编号——不会
    # np.random.seed(1)
    # a = [0,1,2,3,4,5,6,7,8,9]
    # b = np.random.choice(a, 3, False) # [2 9 6]
    # a.pop(0)
    # a.append(10)
    # c = np.random.choice(a, 3, False) # [10 6 4]
    # print(a,'\n',b,'\n',c)

    # x = [0.927, -0.213]
    # x1, _ = step(x, [0,0], 0)
    # print(x1)