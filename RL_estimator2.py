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
import time
from filterpy.kalman import MerweScaledSigmaPoints,UnscentedKalmanFilter

import dynamics as dyn
import estimator as est

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
    def __init__(self, state_dim, obs_dim, h=[200]) -> None : 
        super(Actor, self).__init__()

        self.input_dim = state_dim + obs_dim
        self.output_dim = int(state_dim*(state_dim+1)/2 + 1)

        self.fc = nn.ModuleList()
        input_size = self.input_dim+self.output_dim-1
        for hidden_size in h : 
            self.fc.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.fc.append(nn.Linear(h[-1], self.output_dim))

    def forward(self, input) : 
        input = torch.tensor(input, dtype=torch.float32)
        output = self.fc[0](input)
        for fc in self.fc[1:] : 
            output = fc(F.relu(output))
        return output

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
        self.x_hat          = []
        self.x_hat_new      = []
        self.x_next_hat     = []
        self.y_next         = []
        self.P_hat_inv      = []
        self.P_next_hat_inv = []
        self.h              = []
        self.h_next         = []
        self.input_init = list()
        self.output_init = list()

    def push_init(self, state_pre, obs, P_inv, Pinv_next, h_next) : # 初始样本人为确保正确性，就不判断正定了
        output_last = P2o(P_inv)
        input = np.hstack((state_pre, obs, output_last))
        output = P2o(Pinv_next, h_next)
        self.input_init.append(input)
        self.output_init.append(output)
        self.size_init += 1

    def push(self, x_hat, x_hat_new, x_next_hat, y_next, P_hat_inv, P_next_hat_inv, h, h_next) : 
        try : 
            np.linalg.cholesky(P_next_hat_inv)
        except np.linalg.LinAlgError : # 矩阵非正定
            print('new P matrix is not positive definite')
            return

        if self.size < self.maxsize : 
            self.x_hat.append(x_hat)
            self.x_hat_new.append(x_hat_new)
            self.x_next_hat.append(x_next_hat)
            self.y_next.append(y_next)
            self.P_hat_inv.append(P_hat_inv)
            self.P_next_hat_inv.append(P_next_hat_inv)
            self.h.append(h)
            self.h_next.append(h_next)
            self.size += 1
        else : 
            self.x_hat[self.count]          = x_hat
            self.x_hat_new[self.count]      = x_hat_new
            self.x_next_hat[self.count]     = x_next_hat
            self.y_next[self.count]         = y_next
            self.P_hat_inv[self.count]      = P_hat_inv
            self.P_next_hat_inv[self.count] = P_next_hat_inv
            self.h[self.count]              = h
            self.h_next[self.count]         = h_next
            self.count += 1
            self.count = int(self.count % self.maxsize)

    def sample(self, n:int, args) : 
        indices = np.random.randint(self.size+self.size_init, size=n)
        x_hat_batch          = []
        x_hat_new_batch      = []
        x_next_hat_batch     = []
        y_next_batch         = []
        P_hat_inv_batch      = []
        P_next_hat_inv_batch = []
        h_batch              = []
        h_next_batch         = []
        bin = []
        bot = []
        for i in indices : 
            if i < self.size : 
                x_hat_batch.append(self.x_hat[i])
                x_hat_new_batch.append(self.x_hat_new[i])
                x_next_hat_batch.append(self.x_next_hat[i])
                y_next_batch.append(self.y_next[i])
                P_hat_inv_batch.append(self.P_hat_inv[i])
                P_next_hat_inv_batch.append(self.P_next_hat_inv[i])
                h_batch.append(self.h[i])
                h_next_batch.append(self.h_next[i])
            else : 
                bin.append(self.input_init[i-self.size])
                bot.append(self.output_init[i-self.size])
        size = len(x_hat_batch)
        x_hat_batch          = np.array(x_hat_batch)
        x_hat_new_batch      = np.array(x_hat_new_batch)
        x_next_hat_batch     = np.array(x_next_hat_batch)
        y_next_batch         = np.array(y_next_batch)
        P_hat_inv_batch      = np.array(P_hat_inv_batch)
        P_next_hat_inv_batch = np.array(P_next_hat_inv_batch)
        h_batch              = np.array(h_batch)
        h_next_batch         = np.array(h_next_batch)
        Q_inv_batch          = np.tile(est.inv(args.Q), (size, 1, 1))
        R_inv_batch          = np.tile(est.inv(args.R), (size, 1, 1))

        if size > 0 : 
            x_next_noise_batch = x_next_hat_batch + np.random.multivariate_normal(np.zeros((args.state_dim, )), args.explore_Cov, size) # 不同的t会得到相同的采样值吗？——不相同，但是能保证可重现
            target_Q_batch = args.gamma * quad_value(x_hat_batch, x_hat_new_batch, P_hat_inv_batch, h_batch) + \
                                quad_value(x_next_noise_batch, dyn.f(x_hat_new_batch), Q_inv_batch) + \
                                quad_value(y_next_batch, dyn.h(x_next_noise_batch), R_inv_batch) ## 这里以前写错了，写成了h(x_hat_batch)，应该是h(x_next_noise_batch)但在错误的情况下效果好像比正确时要好？
            Q_batch = quad_value(x_next_noise_batch, x_next_hat_batch, P_next_hat_inv_batch, h_next_batch)
            delta = Q_batch - target_Q_batch  ## 说实在的，现在我并没有深入理解这个算法的理论依据是什么
            P_next_new_inv_batch = P_next_hat_inv_batch - args.lr_value * np.array([delta[index] * \
                                    quad_value_T(x_next_noise_batch, x_next_hat_batch)[index] for index in range(size)]) ## 梯度下降不能保证Pnew正定-不正定就不做更新直接跳过
            h_next_new_batch = h_next_batch - args.lr_value * delta
            input_batch  = np.concatenate((x_hat_batch, y_next_batch, [P2o(P_hat_inv_batch[index]) for index in range(size)]), axis=1).tolist() ## 这里以前写错了，写成了x_next_hat_batch，应该是x_hat_batch
            output_batch = [P2o(P_next_new_inv_batch[index], h_next_new_batch[index]) for index in range(size)]
            del_list = []
            for n in range(size) : 
                if output_batch[n] == [] : 
                    del_list.append(n)
            if del_list != [] :
                input_batch  = [input_batch[index]  for index in range(n) if index not in del_list]
                output_batch = [output_batch[index] for index in range(n) if index not in del_list]
            bin = bin + input_batch
            bot = bot + output_batch


        bin = torch.FloatTensor(bin)
        bot = torch.FloatTensor(bot)

        return bin, bot


# Pinv(n,n) to L(n,n) to output(1,n*(n+1)/2)
def P2o(P, h=None) : 
    # check if Pinv_next is semi-definit
    try : 
        L = np.linalg.cholesky(P)
    except np.linalg.LinAlgError : # 矩阵非正定
        print('new P matrix is not positive definite')
        return []

    out = []
    for i in range(L.shape[0]) : 
        out.extend(L[i, :i+1])
    out = np.array(out).flatten()
    if h is not None : out = np.append(out, h)
    return out


class RL_estimator : 
    def __init__(self, state_dim, obs_dim, noise:OUnoise, hidden_layer=[200,200,200,200,200,200,200,200,200,200,200],
                 rand_num=111, STATUS='train') -> None:
        self.state_dim = state_dim
        torch.manual_seed(rand_num)
        self.policy = Actor(state_dim, obs_dim, h=hidden_layer)
        self.OUnoise = noise
        self.STATUS = STATUS

    def get_Pinv(self, state_pre, obs, Pinv_now) : 
        noise = self.OUnoise.noise()
        noise = (self.STATUS=='train')*noise
        output_last = P2o(Pinv_now)
        input = np.hstack((state_pre, obs, output_last))
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


def quad_value(state, state_pre, Pinv, h=None) : 
    quad = lambda x1, x2, W, h : (x1 - x2) @ W @ (x1 - x2).T + h

    batch_size = state.shape[0]
    if h is None : h = np.zeros((batch_size, ))
    if batch_size == 1 : 
        Q = quad(state, state_pre, Pinv, h)
    elif batch_size >= 2 : 
        Q = [quad(state[i], state_pre[i], Pinv[i], h[i]) for i in range(batch_size)]
    return np.array(Q)

def quad_value_T(state, state_pre, h=None) : 
    quad = lambda x1, x2, h : (x1 - x2).T @ (x1 - x2) + h

    batch_size = state.shape[0]
    if h is None : h = np.zeros((batch_size, 2,2))
    if batch_size == 1 : 
        Q = quad(state, state_pre, h)
    elif batch_size >= 2 : 
        Q = [quad(np.tile(state[i],(1,1)), np.tile(state_pre[i],(1,1)), h[i]) for i in range(batch_size)]
    return np.array(Q)


def train(args, agent:RL_estimator, replay_buffer:ReplayBuffer) : 
    sys.stdout = open(args.output_file, 'w')

    MSE_min = np.zeros((2))
    num_noupdate = 0
    for i in range(args.max_episodes) : 
        # noise = OUnoise(args.state_dim, rand_num=i)
        # agent.reset_noise(noise) ## 是否需要每次都基于当前最好的模型来训练，但是当前最好的模型也有可能只是在当前这一集上表现好。评价好坏可能需要与EKF的MSE作对比。

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_train_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        x_hat = args.x0_hat
        x_hat_EKF = args.x0_hat
        P_hat_EKF = args.P0_hat
        P_hat_inv = est.inv(args.P0_hat)
        h = 0
        MSE = np.zeros((args.state_dim, ))
        MSE_EKF = np.zeros((args.state_dim, ))
        for t in range(args.max_train_steps) : 
            # dynamic, x is unobservable, y is observable
            if args.MODEL_MISMATCH == False : 
                x_next,y_next = dyn.step(x,w_list[t],v_list[t])
            else : 
                x_next = real_fx(x, w_list[t])
                y_next = real_hx(x_next, v_list[t])

            # get covarience matrix P 
            P_next_hat_inv, h_next = agent.get_Pinv(x_hat, y_next, P_hat_inv)

            # solve optimization problem, get x_next_hat
            result = est.NLSF(x_hat, est.inv(P_hat_inv), y_next, args.Q, args.R)
            # result = est.OPTF(x_pre, est.inv(P_pre_inv), y, args.Q, args.R)
            x_hat_new = result[ :args.state_dim]
            x_next_hat = result[args.state_dim: ]
            # EKF for comparison
            x_next_hat_EKF, P_next_hat_EKF = est.EKF(x_hat_EKF, P_hat_EKF, y_next, args.Q, args.R)

            # push experience into replay buffer
            replay_buffer.push(x_hat, x_hat_new, x_next_hat, y_next, P_hat_inv, P_next_hat_inv, h, h_next)

            # training ## 如果要把随机探索放到回放训练里面，那么就是采样之后再做这个新P和新h的计算
            if replay_buffer.size > args.warmup_size : 
                input_batch, output_batch = replay_buffer.sample(args.batch_size, args)
                agent.policy.update_weight(input_batch, output_batch, lr=args.lr_policy)

            # error evaluate, MSE
            MSE += (x - x_hat)**2 / args.max_train_steps
            MSE_EKF += (x - x_hat_EKF)**2 / args.max_train_steps
            MSE_ratio = MSE / MSE_EKF
        
            x = x_next
            x_hat_EKF = x_next_hat_EKF
            P_hat_EKF = P_next_hat_EKF
            x_hat = x_next_hat
            P_hat_inv = P_next_hat_inv
            h = h_next

        print(i, ': MSE = ', MSE, '\n')
        num_noupdate += 1
        if (MSE_ratio <= MSE_min).all() or i == 0 : 
            MSE_min = MSE_ratio
            save_path = os.path.join(args.output_dir, args.model_file)
            torch.save(agent.policy.state_dict(), save_path)
            num_noupdate = 0
        elif num_noupdate > 50 and (args.lr_policy/2) > args.lr_policy_min : 
            args.lr_policy /= 2
            print("lr_policy update")
            num_noupdate = 0
        sys.stdout.flush()
    save_path = os.path.join(args.output_dir, args.modelend_file)
    torch.save(agent.policy.state_dict(), save_path)
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def real_fx(x, disturb=[], time_sample=.1) : 
    x = x.T
    x_next = np.copy(x)
    x_next[0] = 1.1*x[1] + 0.25*x[1]
    x_next[1] = -0.15*x[0] + 0.55*x[1]/(1+x[1]**2)
    x = x_next
    if len(disturb) == 0 : disturb = np.zeros_like(x)
    x = x + disturb
    return x

def real_hx(x, noise=[]) : 
    x = x.T
    y = x[0] - 3*x[1]
    if len(noise) == 0 : noise = np.zeros_like(y)
    y = y + noise
    return y


def simulate(args, sim_num=1, rand_num=1111, STATUS='EKF') : 
    if STATUS == 'NLS-RLF' or STATUS == 'RLF' : 
        noise = OUnoise(state_dim=args.state_dim, rand_num=rand_num)
        agent = RL_estimator(args.state_dim, args.obs_dim, noise, hidden_layer=args.hidden_layer, STATUS='test')
        model_path = os.path.join(args.output_dir, args.model_file)
        agent.policy.load_state_dict(torch.load(model_path))
    # if STATUS == 'UKF' : 
        # points = MerweScaledSigmaPoints(2, alpha=1., beta=2., kappa=0.)
        # ukf = UnscentedKalmanFilter(dim_x=args.state_dim, dim_z=args.obs_dim, dt=.1, fx=fx, hx=hx, points=points)
        # ukf.x = args.x0_hat # initial state
        # ukf.P = args.P0_hat # initial uncertainty
        # ukf.R = args.R
        # ukf.Q = args.Q
    if STATUS == 'PF' : 
        pf = est.Particle_Filter(args.state_dim, args.obs_dim, int(1e4), dyn.f, dyn.h, args.x0_mu, args.P0)

    # initial set for criterion
    error = np.zeros((args.max_sim_steps,args.state_dim))
    MSE = 0
    execution_time = 0
    for i in range(sim_num) : 
        # set random seed
        np.random.seed(rand_num+i)
        # generate disturbance and noise
        x0, w_list, v_list = dyn.reset(rand_num+i, args.max_sim_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        # initial set for each simulation
        x_seq = np.zeros((args.max_sim_steps,args.state_dim))
        x_hat_seq = np.zeros((args.max_sim_steps,args.state_dim))
        y_seq = np.zeros((args.max_sim_steps,args.obs_dim))
        P_seq = [np.zeros_like(args.P0_hat) for _ in range(args.max_sim_steps)]
        x = x0
        x_hat = args.x0_hat
        P_hat = args.P0_hat
        t_seq = range(args.max_sim_steps)
        # main circle
        for t in t_seq : 
            # real state
            if args.MODEL_MISMATCH == False : 
                x_next,y_next = dyn.step(x,w_list[t],v_list[t])
            else : 
                x_next = real_fx(x, w_list[t])
                y_next = real_hx(x_next, v_list[t])

            # time record 
            start_time = time.process_time()

            # choose filter
            if STATUS == 'EKF' or STATUS=='init': 
                # estimator Extended Kalman Filter
                x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
            elif STATUS == 'UKF' : 
                x_next_hat, P_next_hat = est.UKF(x_hat, P_hat, y_next, args.Q, args.R)
                # ukf.predict()
                # ukf.update(y_next)
                # x_next_hat = ukf.x
                # P_next_hat = ukf.P
            elif STATUS == 'PF' : 
                pf.predict(args.Q)
                pf.update(y_next, args.R)
                x_next_hat, P_next_hat = pf.estimate()
            elif STATUS == 'RLF' : 
                x_next_hat, _ = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
                P_inv_next, _ = agent.get_Pinv(x_hat, y_next, est.inv(P_hat))
                P_next_hat = est.inv(P_inv_next)
            elif STATUS == 'NLS-EKF' : 
                # estimator Nonlinear Least Square-Extended Kalman Filter
                result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
                # result1 = est.OPTF(x_hat, P_hat, y_next, args.Q, args.R)
                x_hat = result[ :2]
                x_next_hat = result[2: ]
                _, P_next_hat = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
            elif STATUS == 'NLS-UKF' : 
                result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
                x_hat = result[ :2]
                x_next_hat = result[2: ]
                _, P_next_hat = est.UKF(x_hat, P_hat, y_next, args.Q, args.R)
            elif STATUS == 'NLS-RLF' : 
                # estimator Nonlinear Least Square-Reinforcement Learning Filter
                result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
                x_hat = result[ :2]
                x_next_hat = result[2: ]
                P_inv_next, _ = agent.get_Pinv(x_hat, y_next, est.inv(P_hat))
                P_next_hat = est.inv(P_inv_next)

            # time evaluate, ms
            end_time = time.process_time()
            execution_time += 1000 * (end_time - start_time) / args.max_sim_steps / sim_num

            # error evaluate, MSE
            MSE += (x_next - x_next_hat)**2 / args.max_sim_steps / sim_num

            # move forward
            x_seq[t] = x_next 
            y_seq[t] = y_next
            x_hat_seq[t] = x_next_hat
            P_seq[t] = P_next_hat
            error[t] += np.abs(x_next - x_next_hat) / sim_num
            x = x_next
            x_hat = x_next_hat
            P_hat = P_next_hat

    if STATUS != 'init' : 
        # evaluation criterion print
        print(f"average cpu time of {STATUS}: {execution_time}")
        print(f"MSE of {STATUS}: {MSE}")
        print(f"lr_policy : {args.lr_policy}")

        # plot
        fig, axs = plt.subplots(2,1)
        axs[0].plot(t_seq, x_seq[:,0], label='x_real', color='blue')
        axs[0].plot(t_seq, x_hat_seq[:,0], label='x_hat', color='red')
        axs[0].set_xlim(0, args.max_sim_steps)
        axs[0].set_ylabel('x1')
        axs[0].set_title(f'{STATUS}')
        axs[0].legend()
        axs[1].plot(t_seq, x_seq[:,1], color='blue')
        axs[1].plot(t_seq, x_hat_seq[:,1], color='red')
        axs[1].set_xlim(0, args.max_sim_steps)
        axs[1].set_xlabel('step')
        axs[1].set_ylabel('x2')

        fig, ax = plt.subplots()
        ax.plot(t_seq, error[:,0], label='x1', color='blue')
        ax.plot(t_seq, error[:,1], label='x2', color='red')
        ax.set_xlim(0, args.max_sim_steps)
        ax.set_xlabel('step')
        ax.set_ylabel('error')
        ax.set_title(f'MSE = {MSE}')
        ax.legend()
        plt.show()

    return x_hat_seq, y_seq, P_seq

def main() : 
    parser = argparse.ArgumentParser()
    # system parameters
    parser.add_argument("--state_dim", default=2, type=int, help="dimension of state variable x")
    parser.add_argument("--obs_dim", default=1, type=int, help="dimension of measurement y")
    parser.add_argument("--x0_mu", default=np.array([0, 0]), help="average of initial state distribution")
    parser.add_argument("--P0", default=np.array([[1., 0],[0, 1.]]), help="covariance of initial state distribution")
    parser.add_argument("--x0_hat", default=np.array([0, 0]), help="estimation of initial state distribution average")
    parser.add_argument("--P0_hat", default=np.array([[1., 0],[0, 1.]]), help="estimation of initial state distribution covariance")
    parser.add_argument("--Q", default=np.array([[.01, 0],[0, 1.]]), help="covariance of process disturbance")
    parser.add_argument("--R", default=np.array([[1e-2]]), help="covariance of measurement noise")
    parser.add_argument("--MODEL_MISMATCH", default=False, type=bool, help="choose whether to apply model mismatch")

    # training parameters
    parser.add_argument("--max_episodes", default=5000, type=int, help="max train episodes")
    parser.add_argument("--max_train_steps", default=200, type=int, help="max simulation steps")
    parser.add_argument("--max_sim_steps", default=1000, type=int, help="max simulation steps")
    parser.add_argument("--buffer_size", default=1e4, type=int, help="max size of replay buffer")
    parser.add_argument("--batch_size", default=16, type=int, help="number of samples for batch update")
    parser.add_argument("--warmup_size", default=200, type=int, help="decide when to start the training of the NN")
    parser.add_argument("--hidden_layer", default=[200,200,200,200,200,200,200,200,200,200,200,200,200,200], help="FC layers of NN")
    parser.add_argument("--gamma", default=.9, type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1.e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-2, type=float, help="learning rate of policy net")
    parser.add_argument("--lr_policy_delta", default=5e-4, type=float, help="learning rate reduction every time")
    parser.add_argument("--lr_policy_min", default=1e-6, type=float, help="learning rate of policy net")
    parser.add_argument("--explore_Cov", default=np.array([[.001,0],[0,.001]]), help="the covariance of Guassian distribution added to predicted state")

    # file path
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log.txt", type=str, help="file to save training messages")
    parser.add_argument("--model_file", default="model.bin", type=str, help="trained model")
    parser.add_argument("--modelend_file", default="modelend.bin", type=str, help="trained model")

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise = OUnoise(args.state_dim)
    agent = RL_estimator(state_dim=args.state_dim, obs_dim=args.obs_dim, noise=noise, hidden_layer=args.hidden_layer, STATUS='test')
    agent.policy.to(args.device)
    # init policy network ## 可以尝试没有初始样本的——没有初始样本的目前看来不太行
    x_hat_seq, y_seq, P_hat_seq = simulate(args, rand_num=22222, STATUS='init')
    x_hat_seq = np.insert(x_hat_seq, 0, args.x0_hat, axis=0)
    P_hat_seq = np.insert(P_hat_seq, 0, args.P0_hat, axis=0)
    replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
    for t in range(args.max_train_steps) : 
        replay_buffer.push_init(x_hat_seq[t], y_seq[t], est.inv(P_hat_seq[t]), est.inv(P_hat_seq[t+1]), 0)
    input_batch, output_batch = replay_buffer.sample(args.max_train_steps, args)
    agent.policy.update_weight(input_batch, output_batch, lr=1e-4)
    # save_path = os.path.join(args.output_dir, "model.bin")
    # torch.save(agent.policy.state_dict(), save_path)
    train(args, agent, replay_buffer)

    simulate(args, sim_num=50, rand_num=10086, STATUS='NLS-RLF')

if __name__ == '__main__' : 
    main()