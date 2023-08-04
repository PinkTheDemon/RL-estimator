import numpy as np

# system dynamics
def step(x, disturb, noise) : 
    # nolinear dynamics
    x_next = list.copy(x)
    x_next[0] = 0.99*x[0] + 0.2*x[1] 
    x_next[1] = -0.1*x[0] + 0.5*x[1]/(1+x[1]**2) + disturb 
    y = x_next[0] - 3*x_next[1] + noise 

    return x_next, y

# generate noise list for ith simulation
def gen_noise(sim_num, maxstep, disturb_mu=0, disturb_P=1, noise_mu=0, noise_Q=0.01) : 
    np.random.seed(sim_num)
    disturb_list = np.random.normal(disturb_mu, disturb_P, maxstep)
    noise_list   = np.random.normal(noise_mu, noise_Q, maxstep)

    return disturb_list, noise_list


# if __name__ == '__main__' : 
    # (wlist,vlist) = gen_noise(1,10)
    # print(wlist,'\n',vlist) # 测试噪声序列能否正常生成——能

    
    # x = [1,2]
    # x_next,y = step(x,wlist[0],vlist[0])
    # print(x,y,x_next) # 测试step正常功能——无误
    # x_next,y = step(x,wlist[1],vlist[1]) # 测试反复进入step会不会得到相同的噪声——不会
    # print(y)