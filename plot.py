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
        ax.plot(t_seq, x_seq.T[i], label='x_real', color='blue', linestyle='-') #, 'o'
        ax.plot(t_seq, x_hat_seq.T[i], label='x_hat', color='red', linestyle='-')#, 'o'
        ax.set_xlim(0, max_steps)
        ax.set_ylabel(f'x{i+1}')
        ax.xaxis.set_visible(False)
        ax.grid(True)
    # axs[0].set_title(f'{STATUS}')
    if ds > 1:
        axs[0].set_title(f'{STATUS}状态估计效果图')
        axs[0].legend()
        axs[-1].set_xlabel('时间步')
        axs[-1].xaxis.set_visible(True)
    else :
        axs.set_title(f'{STATUS}状态估计效果图')
        axs.legend()
        axs.set_xlabel('时间步')
        axs.xaxis.set_visible(True)

    fig, axs = plt.subplots(ds,1)
    color = ['b','g','r','c','m','y','k']
    for i in range(ds) : 
        ax = axs[i] if ds > 1 else axs
        ax.plot(t_seq, error[:,i], label=f'x{i+1}', color='red', linestyle='--') #, 'o'
        ax.plot(t_seq, np.average(error[:,i])*np.ones_like(t_seq), color='blue')
        ax.legend()
        ax.xaxis.set_visible(False)
        ax.grid(True)
    if ds > 1:
        axs[0].set_title(f'{STATUS}误差绝对值')
        axs[-1].set_xlabel('时间步')
        axs[-1].xaxis.set_visible(True)
    else :
        axs.set_title(f'{STATUS}误差绝对值')
        axs.set_xlabel('时间步')
        axs.xaxis.set_visible(True)

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
    # rewardSeq = [
    # ]
    # plotReward(rewardSeq) #, filename="picture/train_RMSE.png"

    from gendata import getData
    x_batch, y_batch = getData(modelName="Continuous2", steps=40000, episodes=1, randSeed=10086)
    x_seq = x_batch[0]
    plotTrajectory(x_seq=x_seq, x_hat_seq=x_seq)

    plt.show()
