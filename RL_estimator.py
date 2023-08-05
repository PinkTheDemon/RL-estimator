import numpy as np
import matplotlib.pyplot as plt
import dynamics as dyn
import estimator as est

# global varibles
MAXSTEP = 200

def simulate(sim_num, x0_mu=[0,0], P0=[[1,0],[0,1]], STATUS='test') : 
    # set random seed
    np.random.seed(sim_num)
    # generate disturbance and noise
    w_list, v_list = dyn.gen_noise(sim_num, MAXSTEP)
    # generate initial state x0
    x0 = np.random.multivariate_normal(x0_mu, P0)
    x0_hat = np.random.multivariate_normal(x0_mu, P0)

    x_seq = np.zeros((MAXSTEP,2))
    x_hat_seq = np.zeros((MAXSTEP,2))
    y_seq = np.zeros((MAXSTEP,1))
    error = np.zeros((MAXSTEP,2))
    x = x0
    x_hat = x0_hat
    P_hat = P0
    MSE = 0
    t_seq = range(0, MAXSTEP)
    # main circle
    for t in t_seq : 
        # real state
        x_next,y_next = dyn.step(x,w_list[t],v_list[t])
        x_seq[t] = x 
        y_seq[t] = y_next

        # estimator
        x_hat_next, P_hat_next = est.EKF(x_hat, P_hat, y_next)
        x_hat_seq[t] = x_hat

        # error evaluate, MSE
        MSE += (x - x_hat)**2 / MAXSTEP

        # move forward
        x_seq[t] = x 
        y_seq[t] = y_next
        x_hat_seq[t] = x_hat
        error[t] = x - x_hat
        t += 1
        x = x_next
        x_hat = x_hat_next
        P_hat = P_hat_next
    
    if STATUS == 'test' : 
        # plot
        fig, axs = plt.subplots(2,1)
        axs[0].plot(t_seq, x_seq[:,0], label='x_real', color='tab:blue')
        axs[0].plot(t_seq, x_hat_seq[:,0], label='x_hat', color='tab:red')
        axs[0].set_xlim(0, MAXSTEP)
        axs[0].set_ylabel('x1')
        axs[0].legend()
        axs[1].plot(t_seq, x_seq[:,1], color='blue')
        axs[1].plot(t_seq, x_hat_seq[:,1], color='red')
        axs[1].set_xlim(0, MAXSTEP)
        axs[1].set_xlabel('step')
        axs[1].set_ylabel('x2')

        fig, ax = plt.subplots()
        ax.plot(t_seq, error[:,0], label='x1', color='tab:blue')
        ax.plot(t_seq, error[:,1], label='x2', color='tab:red')
        ax.set_xlim(0, MAXSTEP)
        ax.set_xlabel('step')
        ax.set_ylabel('error')
        ax.set_title(f'MSE = {MSE}')
        ax.legend()
        plt.show()






if __name__ == '__main__' : 
    # (wlist,vlist) = dyn.gen_noise(1,10)
    # print(wlist,'\n',vlist) 
    # x = [1,2]
    # x_next,y = dyn.step(x,wlist[0],vlist[0])
    # print(x,y,x_next) 
    # x_next,y = dyn.step(x,wlist[1],vlist[1]) 
    # print(y) # 测试跨文件函数是否正确执行并得到相同结果——无误

    # x0 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 2)[0]
    # x1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 2)[1]
    # print(x0)

    simulate(1)