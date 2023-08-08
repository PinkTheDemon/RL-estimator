import numpy as np

# system dynamics
def step(x, disturb=[0,0], noise=0) : 
    # nolinear dynamics
    x_next = np.copy(x)
    x_next[0] = 0.99*x[0] + 0.2*x[1] + disturb[0]
    x_next[1] = -0.1*x[0] + 0.5*x[1]/(1+x[1]**2) + disturb[1] 
    y_next = x_next[0] - 3*x_next[1] + noise 

    return x_next, y_next

# generate noise list for ith simulation
def reset(sim_num, maxstep, x0_mu=[0,0], P0=[[1,0],[0,1]],
          disturb_mu=[0,0], disturb_Q=[[0.0001,0],[0,1]], noise_mu=0.0, noise_R=0.01) : 
    np.random.seed(sim_num)
    
    initial_state = np.random.multivariate_normal(x0_mu, P0)
    disturb_list  = np.random.multivariate_normal(disturb_mu, disturb_Q, maxstep)
    noise_list    = np.random.normal(noise_mu, noise_R, maxstep)

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