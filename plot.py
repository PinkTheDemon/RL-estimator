import numpy as np
import matplotlib.pyplot as plt

from functions import checkFilename


def plotReward(rewardSeq, filename=None) -> None : 
    tSeq = range(len(rewardSeq))
    smoothedReward = [rewardSeq[0]]
    alpha = 1
    for i in range(1, len(rewardSeq)) : 
        # # 指数平滑
        smoothedValue = alpha*rewardSeq[i] + (1-alpha)*smoothedReward[i-1]
        smoothedReward.append(smoothedValue)
        # # ----------
        # 滑动窗口
        # windowLength = 50
        # if i < windowLength : smoothedReward.append(sum(rewardSeq[:i+1])/(i+1))
        # else : smoothedReward.append(sum(rewardSeq[i-windowLength+1:i+1])/windowLength)
        # ----------
    plt.figure()
    plt.plot(tSeq, smoothedReward)
    plt.title("reward curve")
    plt.xlabel("train times")
    plt.ylabel("train reward")
    if filename is not None : 
        filename = checkFilename(filename)
        plt.savefig(filename)


def plotTrajectory(x_seq, x_hat_seq, STATUS="None") -> None:
    x_seq = np.array(x_seq)
    x_hat_seq = np.array(x_hat_seq)
    ds = x_seq[0].size
    max_steps = len(x_seq)
    t_seq = range(max_steps)
    error = np.abs(x_seq - x_hat_seq)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    plt.rcParams['font.size'] = 28
    fig, axs = plt.subplots(ds,1)
    for i in range(ds) : 
        ax = axs[i] if ds > 1 else axs
        ax.plot(t_seq, x_seq.T[i], 'o', label='x_real', color='blue', linestyle='-')
        ax.plot(t_seq, x_hat_seq.T[i], 'o', label='x_hat', color='red', linestyle='-')
        ax.set_xlim(0, max_steps)
        ax.set_ylabel(f'x{i+1}')
    # axs[0].set_title(f'{STATUS}')
    if ds > 1:
        axs[0].set_title(f'{STATUS}状态估计效果图')
        axs[0].legend()
        axs[-1].set_xlabel('时间步')
    else :
        axs.set_title(f'{STATUS}状态估计效果图')
        axs.legend()
        axs.set_xlabel('时间步')
    plt.grid(True)

    fig, ax = plt.subplots()
    color = ['b','g','r','c','m','y','k']
    for i in range(ds) : 
        ax.plot(t_seq, error[:,i], 'o', label=f'x{i+1}', color=color[i], linestyle='--')
        ax.plot(t_seq, np.average(error[:,i])*np.ones_like(t_seq), color=color[i])
    ax.set_xlim(0, max_steps)
    ax.set_xlabel('时间步')
    ax.set_ylabel('绝对值误差')
    # ax.set_title(f'MSE = {MSE}')
    ax.set_title('绝对值误差')
    ax.legend()
    plt.grid(True)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x_seq[:,0], x_seq[:,1], x_seq[:,2], label='x_real', color='blue')
    # ax.scatter(*x_seq[0], marker='o', color='blue')
    # ax.plot3D(x_hat_seq[:,0], x_hat_seq[:,1], x_hat_seq[:,2], label='x_hat', color='red')
    # ax.scatter(*x_hat_seq[0], marker='o', color='red')
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('x3')
    # ax.legend()
    # ax.set_title('状态轨迹')

if __name__ == "__main__" : 
    rewardSeq = [
    ]

    plotReward(rewardSeq) #, filename="picture/train_RMSE.png"
    plt.show()