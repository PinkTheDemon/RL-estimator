import numpy as np

def f(x, disturb=[], time_sample=.1) : 
    x = x.T
    # parameters for dynamic 3 ##############################
    # beta0 = -.59783
    # H0    = 13.406
    # Gm0   = 3.9860e5
    # R0    = 6374
    # #######################################################

    # dynamics 1 #########################################
    x[0] = 0.99*x[1] + 0.2*x[1]
    x[1] = -0.1*x[0] + 0.5*x[1]/(1+x[1]**2)
    # ####################################################

    # xdot = np.zeros_like(x)
    # dt = 0.01
    # for _ in range(int(time_sample/dt)) : 
    #     # dynamics 2 #########################################
    #     # xdot[0] = -x[1]
    #     # xdot[1] = -0.2*(1-x[0]**2)*x[1] + x[0]
    #     # ####################################################

    #     # dynamics 3 #########################################
    #     # betak = beta0 * np.exp(x[4])
    #     # rk = np.sqrt((x[0]-xr)**2 + (x[1]-yr)**2)
    #     # Rk = np.sqrt(x[0]**2 + x[1]**2)
    #     # Vk = np.sqrt(x[2]**2 + x[3]**2)
    #     # Dk = -betak * np.exp((R0 - Rk)/H0) * Vk
    #     # Gk = -
    #     # xdot[0] = x[2]
    #     # xdot[1] = x[3]
    #     # xdot[2] = 
    #     # ####################################################
    #     x = x + dt*xdot
    x = x.T

    if len(disturb) == 0 : disturb = np.zeros_like(x)
    x = x + disturb
    return x

def h(x, noise=[]) : 
    # measurement equation 1 ###############################
    x = x.T
    y = x[0] - 3*x[1]
    # ######################################################

    # measurement equation 2 ###############################
    # y = x
    # ######################################################
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
          disturb_mu=[0,0], noise_mu=0.0) : 
    np.random.seed(sim_num)
    
    initial_state = np.random.multivariate_normal(x0_mu, P0)
    disturb_list  = np.random.multivariate_normal(disturb_mu, disturb_Q, maxstep)
    noise_list    = np.random.normal(noise_mu, noise_R.item(), maxstep)

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