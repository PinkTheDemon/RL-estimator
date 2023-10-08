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
from OUnoise import OUnoise
from actor import * # 包括torch等
from functions import * # 包括np


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

    # def push_init(self, state_pre, obs, P_inv, h, Pinv_next, h_next) : # 初始样本人为确保正确性，就不判断正定了
    #     output_last = P2o(P_inv, h)
    #     input = np.hstack((state_pre, obs, output_last))
    #     output = P2o(Pinv_next, h_next)
    #     self.input_init.append(input)
    #     self.output_init.append(output)
    #     self.size_init += 1

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
        Q_inv_batch          = np.tile(inv(args.Q), (size, 1, 1))
        R_inv_batch          = np.tile(inv(args.R), (size, 1, 1))

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
            input_batch  = np.concatenate((x_hat_batch, y_next_batch, [P2o(P_hat_inv_batch[index], h_batch[index]) for index in range(size)]), axis=1).tolist() ## 这里以前写错了，写成了x_next_hat_batch，应该是x_hat_batch
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


class RL_estimator : 
    def __init__(self, state_dim, obs_dim, noise:OUnoise, hidden_layer=[200],
                 rand_num=111, STATUS='train') -> None:
        self.dim_output = ds2do(state_dim)
        self.dim_input = state_dim + obs_dim + self.dim_output
        self.policy = Actor(self.dim_input, self.dim_output, h=hidden_layer, rand_num=rand_num)
        self.OUnoise = noise
        self.STATUS = STATUS

    def value(state, state_pre, Pinv, h=None, Transpose=False) : 
        if Transpose : 
            quad = lambda x1, x2, W : (x1 - x2).T @ W @ (x1 - x2)
        else : 
            quad = lambda x1, x2, W : (x1 - x2) @ W @ (x1 - x2).T

        batch_size = state.shape[0]
        Q = [quad(state[i], state_pre[i], Pinv[i]) for i in range(batch_size)]
        Q = np.array(Q)
        if h is not None : Q += h
        return Q

    def get_Pinv(self, state_pre, obs, Pinv_now, h) : 
        noise = self.OUnoise.noise()
        noise = (self.STATUS=='train')*noise
        output_last = P2o(Pinv_now, h)
        input = np.hstack((state_pre, obs, output_last))
        output = self.policy(input).detach().numpy() # + noise
        ds = do2ds(output.size)
        L = np.zeros((ds, ds))
        for i in range(ds) : 
            L[i][ :i+1] = np.copy(output[ :i+1])
            output = output[i+1: ]
        P_next_inv = L @ L.T
        h_next = output[-1]

        return P_next_inv, h_next


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

    ds = args.state_dim
    MSE_min = np.zeros((ds))
    num_noupdate = 0
    for i in range(args.max_episodes) : 
        # noise = OUnoise(ds, rand_num=i)
        # agent.reset_noise(noise) ## 是否需要每次都基于当前最好的模型来训练，但是当前最好的模型也有可能只是在当前这一集上表现好。

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_train_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        np.random.seed(i)
        x_hat = np.random.multivariate_normal(args.x0_hat, args.P0_hat)
        x_hat_EKF = x_hat
        P_hat_EKF = args.P0_hat
        P_hat_inv = inv(args.P0_hat)
        h = 0
        MSE = np.zeros((ds, ))
        for t in range(args.max_train_steps) : 
            # dynamic, x is unobservable, y is observable
            if args.MODEL_MISMATCH == False : 
                x_next, y_next = dyn.step(x, w_list[t], v_list[t])
            else : 
                x_next, y_next = dyn.step_real(x, w_list[t], v_list[t])

            # get covarience matrix P 
            P_next_hat_inv, h_next = agent.get_Pinv(x_hat, y_next, P_hat_inv, h)

            # solve optimization problem, get x_next_hat
            result = est.NLSF(x_hat, inv(P_hat_inv), y_next, args.Q, args.R)
            x_hat_new = result[ :ds]
            x_next_hat = result[ds: ]

            # push experience into replay buffer
            replay_buffer.push(zip(x_hat, x_hat_new, x_next_hat, y_next, P_hat_inv, P_next_hat_inv, h, h_next), None)

            # training ## 采样之后再做新P和新h的计算
            if replay_buffer.size > args.warmup_size : 
                in_list, ot_list, is_init = replay_buffer.sample(args.batch_size)
                bin = [input for input,judge in zip(in_list,is_init) if judge]
                bot = [output for output,judge in zip(ot_list,is_init) if judge]
                size = is_init.count(True)
                if size > 0 : 
                    x_hat_batch, x_hat_new_batch, x_next_hat_batch, y_next_batch, P_hat_inv_batch, P_next_hat_inv_batch, h_batch, h_next_batch = \
                        zip(*[input for input,judge in zip(in_list,is_init) if not judge])
                    Q_inv_batch = np.tile(inv(args.Q), (size, 1, 1))
                    R_inv_batch = np.tile(inv(args.R), (size, 1, 1))
                    x_next_noise_batch = x_next_hat_batch + np.random.multivariate_normal(np.zeros((args.state_dim, )), args.explore_Cov, size)
                    target_Q_batch = args.gamma * agent.value(x_hat_batch, x_hat_new_batch, P_hat_inv_batch, h_batch) + \
                                     agent.value(x_next_noise_batch, dyn.f(x_hat_new_batch), Q_inv_batch) + \
                                     agent.value(y_next_batch, dyn.h(x_next_noise_batch), R_inv_batch) ## 这里以前写错了，写成了h(x_hat_batch)，应该是h(x_next_noise_batch)
                    Q_batch = agent.value(x_next_noise_batch, x_next_hat_batch, P_next_hat_inv_batch, h_next_batch)
                    delta = Q_batch - target_Q_batch
                    P_next_new_inv_batch = P_next_hat_inv_batch - args.lr_value * np.array([delta[index] * \
                                            agent.value(x_next_noise_batch, x_next_hat_batch, np.ones(size,1), Transpose=True)[index] for index in range(size)])
                    h_next_new_batch = h_next_batch - args.lr_value * delta
                    input_batch = np.concatenate(x_hat_batch, y_next_batch, [P2o(P_hat_inv_batch[index], h_batch[index]) for index in range(size)], axis=1).tolist()
                    output_batch = [P2o(P_next_new_inv_batch[index], h_next_new_batch[index]) for index in range(size)]
                    del_list = []
                    for n in range(size) : 
                        if output_batch[n] is None : 
                            del_list.append(n)
                    if del_list != [] :
                        input_batch  = [input_batch[index]  for index in range(n) if index not in del_list]
                        output_batch = [output_batch[index] for index in range(n) if index not in del_list]
                bin += input_batch
                bot += output_batch
                agent.policy.update_weight(bin, bot, args.device, lr=args.lr_policy)

            # error evaluate, MSE
            MSE += (x - x_hat)**2 / args.max_train_steps
        
            x = x_next
            x_hat = x_next_hat
            P_hat_inv = P_next_hat_inv
            h = h_next

        print(i, ': MSE = ', MSE, '\n')
        num_noupdate += 1
        if (MSE < MSE_min).any() or i == 0 : 
            if i == 0 : MSE_min = MSE 
            else : MSE_min[MSE < MSE_min] = MSE[MSE < MSE_min]
            save_path = os.path.join(args.output_dir, args.model_file)
            torch.save(agent.policy.state_dict(), save_path)
            num_noupdate = 0
        elif num_noupdate >= 50 and (args.lr_policy/2) >= args.lr_policy_min : 
            args.lr_policy /= 2
            print("lr_policy update")
            num_noupdate = 0
        sys.stdout.flush()

    save_path = os.path.join(args.output_dir, args.modelend_file)
    torch.save(agent.policy.state_dict(), save_path)
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def simulate(args, sim_num=1, rand_num=1111, STATUS='EKF') : 
    ds = args.state_dim
    if STATUS == 'NLS-RLF' or STATUS == 'RLF' : 
        noise = OUnoise(state_dim=ds, rand_num=rand_num)
        agent = RL_estimator(ds, args.obs_dim, noise, hidden_layer=args.hidden_layer, STATUS='test')
        model_path = os.path.join(args.output_dir, args.model_test)
        agent.policy.load_state_dict(torch.load(model_path))
    if STATUS == 'UKF' : 
        points = MerweScaledSigmaPoints(2, alpha=.3, beta=2., kappa=0.)
        ukf = UnscentedKalmanFilter(dim_x=ds, dim_z=args.obs_dim, dt=.1, fx=dyn.f, hx=dyn.h, points=points)
        ukf.x = args.x0_hat # initial state
        ukf.P = args.P0_hat # initial uncertainty
        ukf.R = args.R
        ukf.Q = args.Q
    if STATUS == 'PF' : 
        pf = est.Particle_Filter(ds, args.obs_dim, int(1e4), dyn.f, dyn.h, args.x0_mu, args.P0)

    # initial set for criterion
    error = np.zeros((args.max_sim_steps,ds))
    MSE = 0
    execution_time = 0
    for i in range(sim_num) : 
        # set random seed
        # np.random.seed(rand_num+i) # 这个不要可能也能得到相同的结果，但肯定不能在之前已经跑过的代码上试
        # generate disturbance and noise
        x0, w_list, v_list = dyn.reset(rand_num+i, args.max_sim_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        # initial set for each simulation
        x_seq = np.zeros((args.max_sim_steps,ds))
        x_hat_seq = np.zeros((args.max_sim_steps,ds))
        y_seq = np.zeros((args.max_sim_steps,args.obs_dim))
        P_seq = [np.zeros_like(args.P0_hat) for _ in range(args.max_sim_steps)]
        x = x0
        x_hat = args.x0_hat
        P_hat = args.P0_hat
        y_next_list = []
        h = 0
        t_seq = range(args.max_sim_steps)
        # main circle
        for t in t_seq : 
            # real state
            if args.MODEL_MISMATCH == False : 
                x_next,y_next = dyn.step(x,w_list[t],v_list[t])
            else : 
                x_next, y_next = dyn.step_real(x, w_list[t], v_list[t])

            # time record 
            start_time = time.process_time()

            # choose filter
            if STATUS == 'EKF' or STATUS=='init': 
                # estimator Extended Kalman Filter
                x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
            elif STATUS == 'UKF' : 
                # x_next_hat, P_next_hat = est.UKF(x_hat, P_hat, y_next, args.Q, args.R)
                ukf.predict()
                ukf.update(y_next)
                x_next_hat = ukf.x
                P_next_hat = ukf.P
            elif STATUS == 'PF' : 
                pf.predict(args.Q)
                pf.update(y_next, args.R)
                x_next_hat, P_next_hat = pf.estimate()
            elif STATUS == 'RLF' : 
                x_next_hat, _ = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
                P_inv_next, h_next = agent.get_Pinv(x_hat, y_next, inv(P_hat), h)
                h = h_next
                P_next_hat = inv(P_inv_next)
            elif STATUS == 'NLS-EKF' : 
                # estimator Nonlinear Least Square-Extended Kalman Filter
                result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
                # result1 = est.OPTF(x_hat, P_hat, y_next, args.Q, args.R)
                x_hat = result[ :ds]
                x_next_hat = result[ds: ]
                _, P_next_hat = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
            elif STATUS == 'NLS-UKF' : 
                result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
                x_hat = result[ :ds]
                x_next_hat = result[ds: ]
                _, P_next_hat = est.UKF(x_hat, P_hat, y_next, args.Q, args.R)
                ukf.predict()
                ukf.update(y_next)
                P_next_hat = ukf.P
            elif STATUS == 'NLS-RLF' : 
                # estimator Nonlinear Least Square-Reinforcement Learning Filter
                result = est.NLSF(x_hat, P_hat, y_next, args.Q, args.R)
                x_hat = result[ :ds]
                x_next_hat = result[ds: ]
                P_inv_next, h_next = agent.get_Pinv(x_hat, y_next, inv(P_hat), h)
                P_next_hat = inv(P_inv_next)
                h = h_next
            elif STATUS == 'FULL' : 
                y_next_list.append(y_next)
                result = est.NLSF(args.x0_hat, args.P0_hat, y_next_list, args.Q, args.R)
                x_next_hat = result[-ds: ]
                P_next_hat = args.P0_hat

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
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False 
        # plt.rcParams['font.size'] = 28
        fig, axs = plt.subplots(ds,1)
        for i in range(ds) : 
            axs[i].plot(t_seq, x_seq[:,i], label='x_real', color='blue')
            axs[i].plot(t_seq, x_hat_seq[:,i], label='x_hat', color='red')
            axs[i].set_xlim(0, args.max_sim_steps)
            axs[i].set_ylabel(f'x{i+1}')
        axs[0].set_title(f'{STATUS}')
        # axs[0].set_title('状态估计效果图')
        axs[0].legend()
        axs[-1].set_xlabel('时间步')

        fig, ax = plt.subplots()
        color = ['b','g','r','c','m','y','k']
        for i in range(ds) : 
            ax.plot(t_seq, error[:,i], label=f'x{i+1}', color=color[i], linestyle='--')
            ax.plot(t_seq, np.average(error[:,i])*np.ones_like(t_seq), color=color[i])
        ax.set_xlim(0, args.max_sim_steps)
        ax.set_xlabel('时间步')
        ax.set_ylabel('绝对值误差')
        ax.set_title(f'MSE = {MSE}')
        # ax.set_title('绝对值误差')
        ax.legend()
        plt.show()

    return x_hat_seq, y_seq, P_seq
