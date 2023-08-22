import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import argparse
import os
import sys

import dynamics as dyn
import estimator_copy as est


class OUnoise : # DDPG算法中常用OUnoise，而不是Gaussian noise，在这个问题中是否有影响？——在目前的算法中，并不需要用到OUnoise，反而高斯噪声更合适
    def __init__(self, state_dim, theta=0.15, mu=None, sigma=0.2, dt=1e-2, x0=None, rand_num=111) -> None:
        self.output_dim = int(state_dim*(state_dim+1)/2 + 1)
        self.theta = theta
        self.mu = mu if mu is not None else np.zeros(self.output_dim)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        np.random.seed(rand_num)
        self.reset()

    def reset(self) : 
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def noise(self) : 
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class Actor(nn.Module) : 
    def __init__(self, state_dim, obs_dim, h1=300, h2=300, h3=300, h4=300) -> None : 
        super(Actor, self).__init__()

        self.input_dim = state_dim + obs_dim
        self.output_dim = int(state_dim*(state_dim+1)/2 + 1)

        self.fc1 = nn.Linear(self.input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, self.output_dim)

    def forward(self, input) : 
        input = torch.tensor(input, dtype=torch.float32)
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        f4 = F.relu(self.fc4(f3))
        action = self.fc5(f4)
        return action

    def update_weight(self, bin, bot, lr=1e-3) : 
        output_batch = self.forward(bin)
        loss = F.mse_loss(output_batch, bot)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class ReplayBuffer : 
    def __init__(self, maxsize:int) -> None:
        self.maxsize = maxsize
        self.size = 0
        self.size_init = 0
        self.count = 0
        self.input = list()
        self.output = list()
        self.input_init = list()
        self.output_init = list()

    def push_init(self, state_pre, obs, Pinv_next, h_next) : # 初始样本人为确保正确性，就不判断正定了
        input = np.hstack((state_pre, obs))
        output = self.P2o(Pinv_next, h_next)
        self.input_init.append(input)
        self.output_init.append(output)
        self.size_init += 1

    def push(self, state_pre, obs, Pinv_next, h_next) : 
        input = np.hstack((state_pre, obs))
        output = self.P2o(Pinv_next, h_next)
        if len(output) == 0 : # Pinv_next非正定
            return

        if self.size < self.maxsize : 
            self.input.append(input)
            self.output.append(output)
            self.size += 1
        else : 
            self.input[self.count] = input
            self.output[self.count] = output
            self.count += 1
            self.count = int(self.count % self.maxsize)

    def sample(self, n:int) : 
        indices = np.random.randint(self.size+self.size_init, size=n)
        bin = []
        bot = []
        for i in indices : 
            if i < self.size : 
                bin.append(self.input[i])
                bot.append(self.output[i])
            else : 
                bin.append(self.input_init[i-self.size])
                bot.append(self.output_init[i-self.size])
        bin = torch.FloatTensor(bin)
        bot = torch.FloatTensor(bot)

        return bin, bot
    
    def P2o(self, P, h) : 
        # check if Pinv_next is semi-definit
        try : 
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError : # 矩阵非正定
            print('new P matrix is not positive definite')
            return []

        out = L.reshape(L.size)
        out = out[out != 0]
        out = np.append(out, h)
        return out

class RL_estimator : 
    def __init__(self, state_dim, obs_dim, noise:OUnoise, rand_num=111, STATUS='train') -> None:
        self.state_dim = state_dim
        torch.manual_seed(rand_num)
        self.policy = Actor(state_dim, obs_dim)
        self.OUnoise = noise
        self.STATUS = STATUS

    def value(self, state, state_pre, Pinv, h) : 
        Q = (state - state_pre) @ Pinv @ (state - state_pre).T + h
        return Q

    def get_Pinv(self, state_pre, obs) : 
        noise = self.OUnoise.noise()
        noise = (self.STATUS=='train')*noise
        input = np.hstack((state_pre, obs))
        output = self.policy(input).detach().numpy() + noise
        L = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.state_dim) : 
            L[i][ :i+1] = np.copy(output[ :i+1])
            output = output[i+1: ]
        P_next_inv = L @ L.T
        h_next = output[-1]

        return P_next_inv, h_next

    def reset_noise(self, noise:OUnoise) : 
        self.noise = noise


def train(args, agent:RL_estimator, replay_buffer:ReplayBuffer) : 
    sys.stdout = open(args.output_file, 'w')

    MSE_min = np.zeros((2))
    for i in range(args.max_episodes) : 
        # noise = OUnoise(args.state_dim, rand_num=i)
        # agent.reset_noise(noise) ## 是否需要每次都基于当前最好的模型来训练，但是当前最好的模型也有可能只是在当前这一集上表现好。评价好坏可能需要与EKF的MSE作对比。

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        np.random.seed(i)
        v0 = np.random.normal(0, args.R).item() # 看一下i变化之后生成的随机噪声有没有变——没有——修改随机数种子后解决
        y = dyn.h(x, v0)
        x_pre = args.x0_mu
        P_pre_inv = est.inv(args.P0)
        # x_EKF = args.x0_mu
        # P_EKF = args.P0
        h = 0
        MSE = np.zeros((args.max_episodes, 2))
        for t in range(args.max_steps) : 
            # dynamic, x is unobservable, y is observable
            x_next, y_next = dyn.step(x, w_list[t], v_list[t])

            # get covarience matrix P 
            P_next_pre_inv, h_next = agent.get_Pinv(x_pre, y)

            # solve optimization problem, get x_next_hat
            result1 = est.NLSF(x_pre, est.inv(P_pre_inv), y, args.Q, args.R)
            result = est.OPTF(x_pre, est.inv(P_pre_inv), y, args.Q, args.R)
            x_hat = result[ :2]
            x_next_pre = result[2: ]

            # training
            # if t > 0 : ## 可能随机探索要放到回放训练里面做，而不是在存储样本之前做。文献中这样写是因为它是一个episode只整个做一次更新，就只需要对每个预测值做一次探索。
            x_next_noise = x_next_pre + np.random.multivariate_normal(np.zeros((args.state_dim, )), args.explore_Cov) # 不同的t会得到相同的采样值吗？——不相同，但是能保证可重现
            target_Q = args.gamma * agent.value(x_hat, x_pre, P_pre_inv, h) + \
                    (x_next_noise - dyn.f(x_hat))@est.inv(args.Q)@(x_next_noise - dyn.f(x_hat)).T + \
                    (y - dyn.h(x_hat))@est.inv(args.R)@(y - dyn.h(x_hat)).T
            Q = agent.value(x_next_noise, x_next_pre, P_next_pre_inv, h_next)
            delta = Q - target_Q  ## 说实在的，现在我并没有深入理解这个算法的理论依据是什么
            P_next_new_inv = P_next_pre_inv - args.lr_value * delta * ((x_next_noise - x_next_pre)@(x_next_noise - x_next_pre).T) ## 梯度下降不能保证正定-不正定就不做更新直接跳过
            h_next_new = h_next - args.lr_value * delta
            replay_buffer.push(x_next_pre, y_next, P_next_new_inv, h_next_new)
            if replay_buffer.size > args.warmup_size : 
                input_batch, output_batch = replay_buffer.sample(args.batch_size)
                agent.policy.update_weight(input_batch, output_batch, lr=args.lr_policy)

            # error evaluate, MSE
            MSE[i] += (x - x_hat)**2 / args.max_steps

            x = x_next
            y = y_next
            x_pre = x_next_pre
            P_pre_inv = P_next_pre_inv

        print(i, ': MSE = ', MSE[i], '\n')
        sys.stdout.flush()
        if (MSE[i] <= MSE_min).all() or i == 0 : 
            MSE_min = MSE[i]
            save_path = os.path.join(args.output_dir, "model.bin")
            torch.save(agent.policy.state_dict(), save_path)

    sys.stdout.close()
    sys.stdout = sys.__stdout__


def simulate(args, sim_num, STATUS='EKF') : 
    # set random seed
    np.random.seed(sim_num)
    # generate disturbance and noise
    x0, w_list, v_list = dyn.reset(sim_num, args.max_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
    v0 = np.random.normal(0, args.R).item()
    y0 = dyn.h(x0, v0)

    if STATUS == 'NLS-RLF' : 
        noise = OUnoise(state_dim=args.state_dim, rand_num=sim_num)
        agent = RL_estimator(args.state_dim, args.obs_dim, noise, STATUS='test')
        model_path = os.path.join(args.output_dir, "model.bin")
        agent.policy.load_state_dict(torch.load(model_path))

    x_seq = np.zeros((args.max_steps,2))
    x_hat_seq = np.zeros((args.max_steps,2))
    x_pre_seq = np.zeros((args.max_steps,2))
    y_seq = np.zeros((args.max_steps,1))
    P_seq = [np.zeros((2,2)) for _ in range(args.max_steps)]
    error = np.zeros((args.max_steps,2))
    x = x0
    y = y0
    x_pre = args.x0_mu
    P_pre = args.P0
    MSE = 0
    t_seq = range(args.max_steps)
    # main circle
    for t in t_seq : 
        # real state
        x_next,y_next = dyn.step(x,w_list[t],v_list[t])

        if STATUS == 'EKF' or STATUS=='init': 
            # estimator Extended Kalman Filter
            x_hat, x_next_pre, P_next_pre = est.EKF(x_pre, P_pre, y, args.Q, args.R)
        elif STATUS == 'NLS-EKF' : 
            # estimator Nonlinear Least Square-Extended Kalman Filter
            result = est.NLSF(x_pre, P_pre, y, args.Q, args.R)
            result1 = est.OPTF(x_pre, P_pre, y, args.Q, args.R)
            x_hat = result[ :2]
            x_next_pre = result[2: ]
            _, _, P_next_pre = est.EKF(x_pre, P_pre, y, args.Q, args.R)
        elif STATUS == 'NLS-RLF' : 
            # estimator Nonlinear Least Square-Reinforcement Learning Filter
            P_next_pre_inv, _ = agent.get_Pinv(x_pre, y)
            P_next_pre = est.inv(P_next_pre_inv)
            result = est.NLSF(x_pre, P_pre, y, args.Q, args.R)
            x_hat = result[ :2]
            x_next_pre = result[2: ]

        # error evaluate, MSE
        MSE += (x - x_hat)**2 / args.max_steps

        # move forward
        x_seq[t] = x
        y_seq[t] = y
        x_hat_seq[t] = x_hat
        x_pre_seq[t] = x_next_pre
        P_seq[t] = P_next_pre
        error[t] = x - x_hat
        t += 1
        x = x_next
        y = y_next
        x_pre = x_next_pre
        P_pre = P_next_pre
        # P_hat = P_next_hat

    if STATUS != 'init' : 
        # plot
        fig, axs = plt.subplots(2,1)
        axs[0].plot(t_seq, x_seq[:,0], label='x_real', color='tab:blue')
        axs[0].plot(t_seq, x_hat_seq[:,0], label='x_hat', color='tab:red')
        axs[0].set_xlim(0, args.max_steps)
        axs[0].set_ylabel('x1')
        axs[0].legend()
        axs[1].plot(t_seq, x_seq[:,1], color='blue')
        axs[1].plot(t_seq, x_hat_seq[:,1], color='red')
        axs[1].set_xlim(0, args.max_steps)
        axs[1].set_xlabel('step')
        axs[1].set_ylabel('x2')

        fig, ax = plt.subplots()
        ax.plot(t_seq, error[:,0], label='x1', color='tab:blue')
        ax.plot(t_seq, error[:,1], label='x2', color='tab:red')
        ax.set_xlim(0, args.max_steps)
        ax.set_xlabel('step')
        ax.set_ylabel('error')
        ax.set_title(f'MSE = {MSE}')
        ax.legend()
        plt.show()

    return x_pre_seq, y_seq, P_seq


def main() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", default=500, type=int, help="max train episodes")
    parser.add_argument("--max_steps", default=200, type=int, help="max simulation steps")
    parser.add_argument("--buffer_size", default=1e4, type=int, help="max size of replay buffer")
    parser.add_argument("--batch_size", default=32, type=int, help="number of samples for batch update")
    parser.add_argument("--warmup_size", default=200, type=int, help="decide when to start the training of the NN")
    parser.add_argument("--state_dim", default=2, type=int, help="dimension of state variable x")
    parser.add_argument("--obs_dim", default=1, type=int, help="dimension of measurement y")
    parser.add_argument("--x0_mu", default=np.array([0,0]), help="average of initial state distribution")
    parser.add_argument("--P0", default=np.array([[1,0],[0,1]]), help="Covariance of initial state distribution")
    parser.add_argument("--Q", default=np.array([[0.0001,0],[0,1]]), help="Covariance of process disturbance")
    parser.add_argument("--R", default=np.array([[0.01]]), help="Covariance of measurement noise")
    parser.add_argument("--gamma", default=0.9, type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-3, type=float, help="learning rate of policy net")
    parser.add_argument("--explore_Cov", default=np.array([[0.01,0],[0,0.01]]), help="the covariance of Guassian distribution added to predicted state")
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log_copy.txt", type=str, help="file to save training messages")

    args = parser.parse_args()

    noise = OUnoise(args.state_dim)
    agent = RL_estimator(state_dim=args.state_dim, obs_dim=args.obs_dim, noise=noise, STATUS='test')
    # init policy network ## 可以尝试没有初始样本的——没有初始样本的目前看来不太行
    x_pre_seq, y_seq, P_next_pre_seq = simulate(args, sim_num=22222, STATUS='init')
    x_pre_seq = np.insert(x_pre_seq, 0, args.x0_mu, axis=0)
    replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
    for t in range(args.max_steps) : 
        replay_buffer.push_init(x_pre_seq[t], y_seq[t], P_next_pre_seq[t], 0)
    input_batch, output_batch = replay_buffer.sample(args.max_steps)
    agent.policy.update_weight(input_batch, output_batch, lr=args.lr_policy)
    # save_path = os.path.join(args.output_dir, "model.bin")
    # torch.save(agent.policy.state_dict(), save_path)

    train(args, agent, replay_buffer)

    simulate(args, sim_num=10086, STATUS='NLS-RLF')


if __name__ == '__main__' : 
    main()