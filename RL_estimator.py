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

import dynamics as dyn
import estimator as est


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

    def forward(self, state_hat, obs_next) : 
        input = np.hstack((state_hat, obs_next))
        input = torch.tensor(input, dtype=torch.float32)
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        action = self.fc3(f2)
        return action
    
    def update_weight(self, state_hat, obs_next, P_new, h_new) : 
        action = self.forward(state_hat, obs_next)
        try : 
            L_new = np.linalg.cholesky(P_new)
        except np.linalg.LinAlgError : 
            return
        action_new = L_new.reshape(L_new.size)
        action_new = action_new[action_new != 0]
        action_new = np.append(action_new, h_new)
        action_new = torch.tensor(action_new, dtype=torch.float32)
        
        loss = F.mse_loss(action, action_new)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    def get_Pinv(self, state_hat, obs_next) : 
        noise = self.OUnoise.noise()
        noise = (self.STATUS=='train')*noise
        action = self.policy(state_hat, obs_next).detach().numpy() + noise
        L = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.state_dim) : 
            L[i][ :i+1] = np.copy(action[ :i+1])
            action = action[i+1: ]
        Pinv = L @ L.T
        h = action[-1]

        return Pinv, h
    
    def reset_noise(self, noise:OUnoise) : 
        self.noise = noise

def train(args, agent:RL_estimator) : 
    sys.stdout = open(args.output_file, 'w')

    MSE_min = np.zeros((2))
    for i in range(args.max_episodes) : 
        noise = OUnoise(args.state_dim, rand_num=i)
        agent.reset_noise(noise)

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        x_hat = args.x0_mu
        P_inv = est.inv(args.P0)
        x_EKF = args.x0_mu
        P_EKF = args.P0
        h = 0
        MSE = np.zeros((args.max_episodes, 2))
        for t in range(args.max_steps) : 
            # dynamic, x is unobservable, y is observable
            x_next, y_next = dyn.step(x, w_list[t], v_list[t])

            # EKF for comparison
            x_next_EKF, P_next_EKF = est.EKF(x_EKF, P_EKF, y_next, args.Q, args.R)
            x_EKF = x_next_EKF
            P_EKF = P_next_EKF

            # get covarience matrix P 
            P_inv_next, h_next = agent.get_Pinv(x_hat, y_next) 

            # solve optimization problem, get x_next_hat
            result = est.NLSF(x_hat, est.inv(P_inv), y_next, args.Q, args.R)
            x_correct = result[ :2]
            x_next_hat = result[2: ]

            if t > 0 : 
                target_Q = args.gamma * agent.value(x_last_correct, x_last_hat, P_inv_last, h_last) + \
                        (x_correct - dyn.f(x_last_correct))@est.inv(args.Q)@(x_correct - dyn.f(x_last_correct)).T + \
                        (y - dyn.h(x_correct))@est.inv(args.R)@(y - dyn.h(x_correct)).T
                Q = agent.value(x_correct, x_hat, P_inv, h)
                delta = Q - target_Q
                P_inv_new = P_inv - args.lr_value * delta * ((x_correct - x_hat)@(x_correct - x_hat).T) # 梯度下降不能保证正定
                h_new = h - args.lr_value * delta
                agent.policy.update_weight(x_last_hat, y, P_inv_new, h_new)

            # error evaluate, MSE
            MSE[i] += (x_next - x_next_hat)**2 / args.max_steps

            x_last_correct = x_correct
            x_last_hat = x_hat
            x_hat = x_next_hat
            x = x_next
            y = y_next
            P_inv_last = P_inv
            P_inv = P_inv_next
            h_last = h
            h = h_next
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

    if STATUS == 'NLS-RLF' : 
        noise = OUnoise(state_dim=args.state_dim, rand_num=sim_num)
        agent = RL_estimator(args.state_dim, args.obs_dim, noise, STATUS='test')
        model_path = os.path.join(args.output_dir, "model.bin")
        agent.policy.load_state_dict(torch.load(model_path))

    x_seq = np.zeros((args.max_steps,2))
    x_hat_seq = np.zeros((args.max_steps,2))
    y_seq = np.zeros((args.max_steps,1))
    P_seq = [np.zeros((2,2)) for _ in range(args.max_steps)]
    error = np.zeros((args.max_steps,2))
    x = x0
    x_hat = args.x0_mu
    P_hat = args.P0
    MSE = 0
    t_seq = range(args.max_steps)
    # main circle
    for t in t_seq : 
        # real state
        x_next,y_next = dyn.step(x,w_list[t,:],v_list[t])

        if STATUS == 'EKF' or STATUS=='init': 
            # estimator Extended Kalman Filter
            x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
        elif STATUS == 'NLS-EKF' : 
            # estimator Nonlinear Least Square-Extended Kalman Filter
            result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
            x_hat = result[ :2]
            x_next_hat = result[2: ]
            _, P_next_hat = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
        elif STATUS == 'NLS-RLF' : 
            # estimator Nonlinear Least Square-Reinforcement Learning Filter
            P_inv_next, _ = agent.get_Pinv(x_hat, y_next)
            P_next_hat = est.inv(P_inv_next)
            result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
            x_hat = result[ :2]
            x_next_hat = result[2: ]


        # error evaluate, MSE
        MSE += (x_next - x_next_hat)**2 / args.max_steps

        # move forward
        x_seq[t] = x_next 
        y_seq[t] = y_next
        x_hat_seq[t] = x_next_hat
        P_seq[t] = P_next_hat
        error[t] = x_next - x_next_hat
        t += 1
        x = x_next
        x_hat = x_next_hat
        P_hat = P_next_hat
    
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

    return x_hat_seq, y_seq, P_seq

def main() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", default=500, type=int, help="max train episodes")
    parser.add_argument("--max_steps", default=200, type=int, help="max simulation steps")
    parser.add_argument("--state_dim", default=2, type=int, help="dimension of state variable x")
    parser.add_argument("--obs_dim", default=1, type=int, help="dimension of measurement y")
    parser.add_argument("--x0_mu", default=np.array([0,0]), help="average of initial state distribution")
    parser.add_argument("--P0", default=np.array([[1,0],[0,1]]), help="Covariance of initial state distribution")
    parser.add_argument("--Q", default=np.array([[0.0001,0],[0,1]]), help="Covariance of process disturbance")
    parser.add_argument("--R", default=np.array([[0.01]]), help="Covariance of measurement noise")
    parser.add_argument("--gamma", default=1.0, type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-2, type=float, help="learning rate of policy net")
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log.txt", type=str, help="file to save training messages")

    args = parser.parse_args()

    # noise = OUnoise(args.state_dim)
    # agent = RL_estimator(state_dim=args.state_dim, obs_dim=args.obs_dim, noise=noise, STATUS='test')
    # # init policy network
    # x_hat_seq, y_next_seq, P_next_seq = simulate(args, sim_num=22222, STATUS='init')
    # x_hat_seq = np.insert(x_hat_seq, 0, args.x0_mu, axis=0)
    # for t in range(args.max_steps) : 
    #     agent.policy.update_weight(x_hat_seq[t], y_next_seq[t], P_next_seq[t], 0)
    # # save_path = os.path.join(args.output_dir, "model.bin")
    # # torch.save(agent.policy.state_dict(), save_path)

    # train(args, agent)

    simulate(args, sim_num=10086, STATUS='EKF')

if __name__ == '__main__' : 
    main()