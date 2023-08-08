import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torch
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import argparse

import dynamics as dyn
import estimator as est


class Actor(nn.Module) : 
    def __init__(self, state_dim, action_dim, h1=400, h2=300) -> None : 
        """
        param:action_lim: used to limit action space in [-action_lim, action_lim]
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_dim)

    def forward(self, state) : 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action
    
class Critic(nn.Module) : ## critic 可能不是一个神经网络，而是基于状态和动作的一个确定性网络
    def __init__(self, state_dim, action_dim, h1=200, h2=300) -> None:
        super(Critic).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(action_dim, h1)
        self.fc3 = nn.Linear(h1+h1, h2)
        self.fc4 = nn.Linear(h2, 1)

    def forward(self, state, action) : 
        x = F.relu(self.fc1(state))
        y = F.relu(self.fc2(action))
        h = torch.cat((x,y), dim=1)
        h = F.relu(self.fc3(h))
        Q = self.fc4(h)

class ReplayBuffer(object) : 
    def __init__(self, buffer_size, rand_num=111) -> None:
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        np.random.seed(rand_num)
        ## 在init中设定seed，后续方法中采样也是按照这个seed来吗？

    def add(self, s, a, r, d, s2) : 
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size : 
            self.buffer.append(experience)
            self.count += 1
        else : # buffer 已满，非优先采样，旧样本出新样本进
            self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self) : 
        return self.count
    
    def sample_batch(self, batch_size) : 
        if self.count < batch_size : # 按理来说不会出现count > batch_size的情况，可能这样代码鲁棒性更强
            batch = np.random.choice(self.buffer, self.count, replace=False)
        else : 
            batch = np.random.choice(self.buffer, batch_size, replace=False)

        bs  = np.array([_[0] for _ in batch])
        ba  = np.array([_[1] for _ in batch])
        br  = np.array([_[2] for _ in batch])
        bd  = np.array([_[3] for _ in batch])
        bns = np.array([_[4] for _ in batch])

        bs  = torch.tensor(bs, dtype=torch.float32)
        ba  = torch.tensor(ba, dtype=torch.float32)
        br  = torch.tensor(br, dtype=torch.float32)
        bd  = torch.tensor(bd, dtype=torch.float32)
        bns = torch.tensor(bns, dtype=torch.float32)
        return bs, ba, br, bd, bns

class OUnoise : ## DDPG算法中常用OUnoise，而不是Gaussian noise，在这个问题中是否有影响？
    def __init__(self, action_dim, theta=0.15, mu=0, sigma=0.2, dt=1e-2, x0=None) -> None:
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self) : 
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def noise(self) : 
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class DDPG : 
    def __init__(self, state_dim, action_dim, gamma) -> None:
        self.policy = Actor(state_dim, action_dim)
        self.value = Critic(state_dim, action_dim)
        self.target_policy = Actor(state_dim, action_dim)
        self.target_value = Critic(state_dim, action_dim)
        self.target_policy.load_statr_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())
        self.gamma = gamma

    def get_action(self, state) : 
        action = self.policy(state)
        return action 
    
    def compute_policy_loss(self, bs, ba, br, bd, bns) : 
        predicted_action = self.get_action(bs)
        loss = -self.value(bs, predicted_action).mean()
        return loss
    
    def compute_value_loss(self, bs, ba, br, bd, bns) : 
        with torch.no_grad() : 
            predicted_bna = self.target_policy(bns)
            target_value = self.gamma * self.target_value(bns, predicted_bna).squeeze * (1-bd) + br

        value = self.value(bs, ba).squeeze()
        loss = F.mse_loss(value, target_value)
        return loss
    
    def soft_update(self, tau=0.01) : 
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()) : 
            target_param.data.copy_(target_param.data*(1-tau) + param.data*tau)

        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()) : 
            target_param.data.copy_(target_param.data*(1-tau) + param.data*tau)

def train(args, env, agent:DDPG, noise:OUnoise) : 
    policy_optimizer = Adam(agent.policy.parameters(), lr = args.lr_policy)
    value_optimizer  = Adam(agent.value.parameters(),  lr = args.lr_value)

    replay_buffer = ReplayBuffer(buffer_size=args.buffer_size)

    log = defaultdict(list)

    episode_reward = 0
    episode_length = 0
    max_episode_reward = -float('inf')
    value_loss_list = [0]
    policy_loss_list = [0]

    x_seq = np.zeros((args.max_steps,2))
    x_hat_seq = np.zeros((args.max_steps,2))
    y_seq = np.zeros((args.max_steps,1))
    error = np.zeros((args.max_steps,2))
    x, w_list, v_list = dyn.reset()
    x_hat = np.random.multivariate_normal(args.x0_mu, args.P0)
    for t in range(args.max_steps) : 
        # dynamic, x is unobservable, y is observable
        x_next, y_next = dyn.step(x, w_list[t], v_list[t])

        # get state = [x_hat, y_next]
        state = np.concatenate((x_hat, [y_next]))

        # get action with noise
        action = agent.get_action(torch.tensor(state))
        action = action + noise.noise()

        # get covarience matrix P ## 还有一个常数项 h，常数项不影响优化问题，应该可以不要
        # action to P directly
        P = action.reshape((x.size(), x.size()))
        # # action to L, then P = L.T @ L
        # for i in range(x.size()) : 
        #     L = 

        # solve optimization problem, get x_hat_next

        





def simulate(args, sim_num, STATUS='test') : ## 后续可以也改成用args传参数
    # set random seed
    np.random.seed(sim_num)
    # generate disturbance and noise
    x0, w_list, v_list = dyn.reset(sim_num, args.max_steps, x0_mu=args.x0_mu, P0=args.P0)

    x_seq = np.zeros((args.max_steps,2))
    x_hat_seq = np.zeros((args.max_steps,2))
    y_seq = np.zeros((args.max_steps,1))
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
        x_seq[t] = x 
        y_seq[t] = y_next

        # # estimator EKF
        # x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_next)
        # estimator NLS-EKF
        result = est.NLSF(x_hat, P_hat, y_next)
        x_hat = result[:2]
        x_next_hat = result[2:]
        _, P_next_hat = est.EKF(x_hat, P_hat, y_next)

        # error evaluate, MSE
        # MSE += (x - x_hat)**2 / args.max_steps
        MSE += (x_next - x_next_hat)**2 / args.max_steps

        # move forward
        x_seq[t] = x_next 
        y_seq[t] = y_next
        x_hat_seq[t] = x_next_hat
        error[t] = x - x_hat
        t += 1
        x = x_next
        x_hat = x_next_hat
        P_hat = P_next_hat

    # # save data to a file for error analysis
    # a = np.hstack((x_seq, x_hat_seq, np.insert(y_seq,0,0)[:-1].reshape((200,1))))
    # np.set_printoptions(suppress=True)
    # with open('data.txt', 'w') as file : 
    #     file.write(np.array2string(a, separator=','))
    
    if STATUS == 'test' : 
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



if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", default=200, type=int, help="max simulation steps")
    parser.add_argument("--x0_mu", default=np.array([0,0]), help="average of initial state distribution")
    parser.add_argument("--P0", default=np.array([[1,0],[0,1]]), help="Covariance of initial state distribution")

    args = parser.parse_args()

    simulate(args, sim_num=1)