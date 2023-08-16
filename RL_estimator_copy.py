import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torch
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import argparse
import os
import sys
from dataclasses import dataclass, field

import dynamics as dyn
import estimator_copy as est


class OUnoise : ## DDPG算法中常用OUnoise，而不是Gaussian noise，在这个问题中是否有影响？
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
    def __init__(self, state_dim, obs_dim, h1=400, h2=300) -> None : 
        """
        param:action_lim: used to limit action space in [-action_lim, action_lim]
        """
        super(Actor, self).__init__()

        self.input_dim = state_dim + obs_dim
        self.output_dim = int(state_dim*(state_dim+1)/2 + 1)

        self.fc1 = nn.Linear(self.input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, self.output_dim)

    def forward(self, state_pre, obs) : 
        input = np.hstack((state_pre, obs))
        input = torch.tensor(input, dtype=torch.float32)
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        action = self.fc3(f2)
        return action

    def update_weight(self, bsp, bo, bPni, bh) : 
        for state_pre, obs, P_next_inv, h_next in zip(bsp, bo, bPni, bh) : 
            action = self.forward(state_pre, obs)
            try : 
                L_new = np.linalg.cholesky(P_next_inv)
            except np.linalg.LinAlgError : 
                print('update error')
                return
            action_new = L_new.reshape(L_new.size)
            action_new = action_new[action_new != 0]
            action_new = np.append(action_new, h_next)
            action_new = torch.tensor(action_new, dtype=torch.float32)

            loss = F.mse_loss(action, action_new)
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class ReplayBuffer : 
    def __init__(self, maxsize:int) -> None:
        self.maxsize = maxsize
        self.size = 0
        self.size_init = 0
        self.count = 0
        self.state_pre = list()
        self.obs = list()
        self.Pinv = list()
        self.h = list()
        self.state_pre_init = list()
        self.obs_init = list()
        self.P_next_inv_init = list()
        self.h_next_init = list()

    def push_init(self, state_pre, obs, Pinv_next, h_next) : 
        self.state_pre_init.append(state_pre)
        self.obs_init.append(obs)
        self.P_next_inv_init.append(Pinv_next)
        self.h_next_init.append(h_next)
        self.size_init += 1

    def push(self, state_pre, obs, Pinv_next, h_next) : 
        if self.size < self.maxsize : 
            self.state_pre.append(state_pre)
            self.obs.append(obs)
            self.Pinv.append(Pinv_next)
            self.h.append(h_next)
            self.size += 1
        else : 
            self.state_pre[self.count] = state_pre
            self.obs[self.count] = obs
            self.Pinv[self.count] = Pinv_next
            self.h[self.count] = h_next
        self.count += 1
        self.count = self.count % self.maxsize

    def sample(self, n) : 
        indices = np.random.randint(self.size+self.size_init, size=n)
        bsp = []
        bo  = []
        bPi = []
        bh  = []
        for i in indices : 
            if i < self.size : 
                bsp.append(self.state_pre[i])
                bo.append(self.obs[i])
                bPi.append(self.Pinv[i])
                bh.append(self.h[i])
            else : 
                bsp.append(self.state_pre_init[i-self.size])
                bo.append(self.obs_init[i-self.size])
                bPi.append(self.P_next_inv_init[i-self.size])
                bh.append(self.h_next_init[i-self.size])                

        return bsp, bo, bPi, bh


class RL_estimator : 
    def __init__(self, state_dim, obs_dim, noise:OUnoise, rand_num=111, STATUS='train') -> None:
        self.state_dim = state_dim
        torch.manual_seed(rand_num)
        self.policy = Actor(state_dim, obs_dim)
        self.OUnoise = noise
        self.STATUS = STATUS

    def value(self, state, state_hat, Pinv, h) : 
        Q = (state - state_hat) @ Pinv @ (state - state_hat).T + h
        return Q

    def get_Pinv(self, state_pre, obs) : 
        noise = self.OUnoise.noise()
        noise = (self.STATUS=='train')*noise
        action = self.policy(state_pre, obs).detach().numpy() + noise
        L = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.state_dim) : 
            L[i][ :i+1] = np.copy(action[ :i+1])
            action = action[i+1: ]
        P_next_inv = L @ L.T
        h_next = action[-1]

        return P_next_inv, h_next

    def reset_noise(self, noise:OUnoise) : 
        self.noise = noise


def train(args, agent:RL_estimator, replay_buffer:ReplayBuffer) : 
    sys.stdout = open(args.output_file, 'w')

    MSE_min = np.zeros((2))
    for i in range(args.max_episodes) : 
        noise = OUnoise(args.state_dim, rand_num=i)
        agent.reset_noise(noise)

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        np.random.seed(i)
        v0 = np.random.normal(0, args.R).item() ## 看一下i变化之后生成的随机噪声有没有变——没有
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

            # # EKF for comparison
            # x_next_EKF, P_next_EKF = est.EKF(x_EKF, P_EKF, y_next, args.Q, args.R)
            # x_EKF = x_next_EKF
            # P_EKF = P_next_EKF

            # get covarience matrix P 
            P_next_pre_inv, h_next = agent.get_Pinv(x_pre, y)

            # solve optimization problem, get x_next_hat
            result1 = est.NLSF(x_pre, est.inv(P_pre_inv), y, args.Q, args.R)
            result = est.OPTF(x_pre, est.inv(P_pre_inv), y, args.Q, args.R)
            x_hat = result[ :2]
            x_next_pre = result[2: ]

            # training
            # if t > 0 : ## 论文中的决策变量是在xhat邻域内随机选择的，然后通过优化问题求出另一个决策变量，这样好像能够保证梯度不会差太多
            x_next_noise = x_next_pre + np.random.multivariate_normal(np.zeros((args.state_dim, )), args.explore_Cov) ## 不同的t会得到相同的采样值吗？
            target_Q = args.gamma * agent.value(x_hat, x_pre, P_pre_inv, h) + \
                    (x_next_noise - dyn.f(x_hat))@est.inv(args.Q)@(x_next_noise - dyn.f(x_hat)).T + \
                    (y - dyn.h(x_hat))@est.inv(args.R)@(y - dyn.h(x_hat)).T
            Q = agent.value(x_next_noise, x_next_pre, P_next_pre_inv, h_next)
            delta = Q - target_Q
            P_next_new_inv = P_next_pre_inv - args.lr_value * delta * ((x_next_noise - x_next_pre)@(x_next_noise - x_next_pre).T) # 梯度下降不能保证正定-不正定就不做更新直接跳过
            h_next_new = h_next - args.lr_value * delta
            replay_buffer.push(x_next_pre, y_next, P_next_new_inv, h_next_new)
            if replay_buffer.size > args.warmup_size : 
                bxp, by, bPni, bh = replay_buffer.sample(args.batch_size)
                agent.policy.update_weight(bxp, by, bPni, bh)

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


def simulate(args, sim_num, STATUS='EKF') : ## 后续可以也改成用args传参数
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
    parser.add_argument("--buffer_size", default=1e5, type=int, help="max size of replay buffer")
    parser.add_argument("--batch_size", default=64, type=int, help="number of samples for batch update")
    parser.add_argument("--warmup_size", default=200, type=int, help="decide when to start the training of the NN")
    parser.add_argument("--state_dim", default=2, type=int, help="dimension of state variable x")
    parser.add_argument("--obs_dim", default=1, type=int, help="dimension of measurement y")
    parser.add_argument("--x0_mu", default=np.array([0,0]), help="average of initial state distribution")
    parser.add_argument("--P0", default=np.array([[1,0],[0,1]]), help="Covariance of initial state distribution")
    parser.add_argument("--Q", default=np.array([[0.0001,0],[0,1]]), help="Covariance of process disturbance")
    parser.add_argument("--R", default=np.array([[0.01]]), help="Covariance of measurement noise")
    parser.add_argument("--gamma", default=1.0, type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-3, type=float, help="learning rate of policy net")
    parser.add_argument("--explore_Cov", default=np.array([[0.01,0],[0,0.01]]), help="the covariance of Guassian distribution added to predicted state")
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log_copy.txt", type=str, help="file to save training messages")

    args = parser.parse_args()

    noise = OUnoise(args.state_dim)
    agent = RL_estimator(state_dim=args.state_dim, obs_dim=args.obs_dim, noise=noise, STATUS='test')
    # init policy network
    x_pre_seq, y_seq, P_next_pre_seq = simulate(args, sim_num=22222, STATUS='init')
    x_pre_seq = np.insert(x_pre_seq, 0, args.x0_mu, axis=0)
    replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
    for t in range(args.max_steps) : 
        replay_buffer.push_init(x_pre_seq[t], y_seq[t], P_next_pre_seq[t], 0)
    agent.policy.update_weight(x_pre_seq, y_seq, P_next_pre_seq, np.zeros((200,)))
    # save_path = os.path.join(args.output_dir, "model.bin")
    # torch.save(agent.policy.state_dict(), save_path)

    train(args, agent, replay_buffer)

    simulate(args, sim_num=10086, STATUS='NLS-RLF')


if __name__ == '__main__' : 
    main()