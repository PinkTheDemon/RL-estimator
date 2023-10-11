from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time

import dynamics as dyn
import estimator as est
from OUnoise import OUnoise
from functions import * # 包括np
from replay_buffer import ReplayBuffer


class Actor(nn.Module) : 
    def __init__(self, dim_input, dim_output, h1=[200], h2=[200], rand_num=111) -> None : 
        super(Actor, self).__init__()
        torch.manual_seed(rand_num)
        self.dim_input = dim_input
        self.hidden_dim = h2[0]
        self.dim_output = dim_output

        self.fc1 = nn.ModuleList()
        self.rnn = nn.GRU(h1[-1], h2[0], batch_first=True)
        self.fc2 = nn.ModuleList()
        input_size = self.dim_input
        for output_size in h1 : 
            self.fc1.append(nn.Linear(input_size, output_size))
            input_size = output_size
        if not self.rnn.bidirectional : 
            self.num_directions = 1
            input_size = h2[0]
        else : 
            self.num_directions = 2
            input_size = h2[0] * 2
        for output_size in h2[1: ] : 
            self.fc2.append(nn.Linear(input_size, output_size))
        self.fc2.append(nn.Linear(h2[-1], self.dim_output))

    # input_seq: time x batch x state fanguolai
    # output_seq:time x batch x output fanguolai 
    def forward(self, input_seq, hidden, device, batch_size=1) : 
        if hidden is None : 
            hidden = torch.zeros((1, batch_size, self.hidden_dim), device=device)

        if isinstance(input_seq, np.ndarray) : 
            input_seq = torch.tensor(np.tile(input_seq, (1,1,1)), dtype=torch.float32, device=device)
        output = input_seq
        for fc1 in self.fc1 : 
            output = F.relu(fc1(output))
        output, hidden = self.rnn(output, hidden) # 要不要激活函数？
        for fc2 in self.fc2 : 
            output = fc2(F.relu(output))

        return output, hidden

    ## rnn做批量更新好像有问题，因为同时计算不同批量的隐状态
    def update_weight(self, input_seq, output_seq, batch_size, num_steps, device, lr=1e-3) : 
        input_seq = torch.FloatTensor(input_seq)
        output_seq = torch.FloatTensor(output_seq)
        hidden = None
        input_seq, output_seq = input_seq.to(device), output_seq.to(device)
        output_seq_hat, hidden = self.forward(input_seq, hidden, device=device, batch_size=batch_size)
        loss = F.mse_loss(output_seq.reshape(-1), output_seq_hat.reshape(-1))
        optimizer = Adam(self.parameters(), lr=lr)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_clipping(self, 10)
        optimizer.step()


def grad_clipping(net, theta) : 
    if isinstance(net, nn.Module) : 
        params = [p for p in net.parameters() if p.requires_grad]
    else : 
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))  ## p.grad 的括号是否可以去掉？
    if norm > theta : 
        for param in params : 
            param.grad[:] *= theta / norm


class RL_estimator : 
    def __init__(self, state_dim, obs_dim, device, noise:OUnoise, hidden_layer=([200],[200]), STATUS='train') -> None:
        self.state_dim = state_dim
        dim_input = state_dim + obs_dim
        dim_output = ds2do(state_dim)
        self.policy = Actor(dim_input, dim_output, h1=hidden_layer[0], h2=hidden_layer[1]).to(device)
        self.device = device
        self.OUnoise = noise
        self.STATUS = STATUS

    def value(self, state, state_pre, Pinv, h=None, Transpose=False) : 
        if Transpose : 
            quad = lambda x1, x2, W : (x1 - x2).T @ W @ (x1 - x2)
        else : 
            quad = lambda x1, x2, W : (x1 - x2) @ W @ (x1 - x2).T

        batch_size = state.shape[0]
        Q = [quad(np.tile(state[i],(1,1)), np.tile(state_pre[i],(1,1)), Pinv[i]) for i in range(batch_size)]
        Q = np.squeeze(np.array(Q))
        if h is not None : Q += h
        return Q

    def get_Pinv(self, state_pre, obs, hidden) : 
        input = np.hstack((state_pre, obs))
        output, hidden = self.policy(input, hidden, self.device)
        output = output.detach().cpu().numpy().reshape(-1)
        L = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.state_dim) : 
            L[i][ :i+1] = np.copy(output[ :i+1])
            output = output[i+1: ]
        P_next_inv = L @ L.T
        h_next = output[-1]

        return P_next_inv, h_next, hidden

    def reset_noise(self, noise:OUnoise) : 
        self.noise = noise


def train(args, agent:RL_estimator, replay_buffer:ReplayBuffer) : 
    sys.stdout = open(args.output_file, 'w')

    ds = args.state_dim
    MSE_min = np.zeros((ds))
    num_noupdate = 0
    for i in range(args.max_episodes) : 
        # noise = OUnoise(ds, rand_num=i)
        # agent.reset_noise(noise) ## 是否需要每次都基于当前最好的模型来训练，但是当前最好的模型也有可能只是在当前这一集上表现好。评价好坏可能需要与EKF的MSE作对比。

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_train_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        np.random.seed(i)
        x_hat = np.random.multivariate_normal(args.x0_hat, args.P0_hat)
        P_hat_inv = est.inv(args.P0_hat)
        h = 0
        hidden = None
        seq = False
        MSE = np.zeros((ds, ))
        for t in range(args.max_train_steps) : 
            # dynamic, x is unobservable, y is observable
            if args.MODEL_MISMATCH == False : 
                x_next, y_next = dyn.step(x, w_list[t], v_list[t])
            else : 
                x_next, y_next = dyn.step_real(x, w_list[t], v_list[t])

            # get covarience matrix P 
            P_next_hat_inv, h_next, hidden = agent.get_Pinv(x_hat, y_next, hidden)

            # solve optimization problem, get x_next_hat
            result = est.NLSF(x_hat, est.inv(P_hat_inv), [y_next], args.Q, args.R)
            x_hat_new = result[ :ds]
            x_next_hat = result[ds: ]

            # push experience into replay buffer
            if t == 0 : seq = not seq
            if t == args.max_train_steps-args.num_steps : seq = not seq
            replay_buffer.push((x_hat, x_hat_new, x_next_hat, y_next, P_hat_inv, P_next_hat_inv, h, h_next), seq)

            # training ## 如果要把随机探索放到回放训练里面，那么就是采样之后再做这个新P和新h的计算
            if replay_buffer.size > args.warmup_size : 
                exp_list, inf_list, is_init = replay_buffer.sample_seq(args.batch_size, args.num_steps)
                bin = []
                bot = []
                size = 0
                for exp_seq, inf_seq, judge in zip(exp_list, inf_list, is_init) : 
                    if judge : 
                        size += 1
                        bin.append([exp[0] for exp in exp_seq])
                        bot.append([exp[1] for exp in exp_seq])
                    elif inf_seq[0] : 
                        size += 1
                        x_hat_batch          = []
                        x_hat_new_batch      = []
                        x_next_hat_batch     = []
                        y_next_batch         = []
                        P_hat_inv_batch      = []
                        P_next_hat_inv_batch = []
                        h_batch              = []
                        h_next_batch         = []
                        for exp in exp_seq : 
                            x_hat_batch.append(exp[0])
                            x_hat_new_batch.append(exp[1])
                            x_next_hat_batch.append(exp[2])
                            y_next_batch.append(exp[3])
                            P_hat_inv_batch.append(exp[4])
                            P_next_hat_inv_batch.append(exp[5])
                            h_batch.append(exp[6])
                            h_next_batch.append(exp[7])
                        x_hat_batch          = np.array(x_hat_batch)
                        x_hat_new_batch      = np.array(x_hat_new_batch)
                        x_next_hat_batch     = np.array(x_next_hat_batch)
                        y_next_batch         = np.array(y_next_batch)
                        P_hat_inv_batch      = np.array(P_hat_inv_batch)
                        P_next_hat_inv_batch = np.array(P_next_hat_inv_batch)
                        h_batch              = np.array(h_batch)
                        h_next_batch         = np.array(h_next_batch)
                        Q_inv_batch = np.tile(inv(args.Q), (args.num_steps, 1, 1))
                        R_inv_batch = np.tile(inv(args.R), (args.num_steps, 1, 1))
                        x_next_noise_batch = x_next_hat_batch + np.random.multivariate_normal(np.zeros((ds, )), args.explore_Cov, args.num_steps)
                        target_Q_batch = args.gamma * agent.value(x_hat_batch, x_hat_new_batch, P_hat_inv_batch, h_batch) + \
                                        agent.value(x_next_noise_batch, dyn.f(x_hat_new_batch), Q_inv_batch) + \
                                        agent.value(y_next_batch, dyn.h(x_next_noise_batch), R_inv_batch) ## 这里以前写错了，写成了h(x_hat_batch)，应该是h(x_next_noise_batch)
                        Q_batch = agent.value(x_next_noise_batch, x_next_hat_batch, P_next_hat_inv_batch, h_next_batch)
                        delta = Q_batch - target_Q_batch
                        P_next_new_inv_batch = P_next_hat_inv_batch - args.lr_value * np.array([delta[index] * \
                                                agent.value(x_next_noise_batch, x_next_hat_batch, np.ones((args.num_steps,1,1)), Transpose=True)[index] for index in range(args.num_steps)])
                        h_next_new_batch = h_next_batch - args.lr_value * delta
                        input_batch = np.concatenate((x_hat_batch, y_next_batch), axis=1).tolist()
                        output_batch = [P2o(P_next_new_inv_batch[index], h_next_new_batch[index]) for index in range(args.num_steps)]
                        del_list = []
                        for n in range(args.num_steps) : 
                            if output_batch[n] is None : 
                                output_batch[n] = P2o(P_next_hat_inv_batch[n], h_next_new_batch[n])
                        if del_list != [] :
                            input_batch  = [input_batch[index]  for index in range(n) if index not in del_list]
                            output_batch = [output_batch[index] for index in range(n) if index not in del_list]
                        bin.append(input_batch)
                        bot.append(output_batch)
                agent.policy.update_weight(bin, bot, size, args.num_steps, args.device, lr=args.lr_policy)

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
        elif num_noupdate > 50 and (args.lr_policy/2) > args.lr_policy_min : 
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
        noise = OUnoise(dim=ds, rand_num=rand_num)
        agent = RL_estimator(ds, args.obs_dim, args.device, noise, hidden_layer=args.hidden_layer, STATUS='test')
        model_path = os.path.join(args.output_dir, args.model_test)
        agent.policy.load_state_dict(torch.load(model_path))
    # if STATUS == 'UKF' : 
        # points = MerweScaledSigmaPoints(2, alpha=1., beta=2., kappa=0.)
        # ukf = UnscentedKalmanFilter(dim_x=ds, dim_z=args.obs_dim, dt=.1, fx=fx, hx=hx, points=points)
        # ukf.x = args.x0_hat # initial state
        # ukf.P = args.P0_hat # initial uncertainty
        # ukf.R = args.R
        # ukf.Q = args.Q
    if STATUS == 'PF' : 
        pf = est.Particle_Filter(ds, args.obs_dim, int(1e4), dyn.f, dyn.h, args.x0_mu, args.P0)

    # initial set for criterion
    error = np.zeros((args.max_sim_steps,ds))
    MSE = 0
    execution_time = 0
    for i in range(sim_num) : 
        # set random seed
        # np.random.seed(rand_num+i)
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
        hidden = None
        t_seq = range(args.max_sim_steps)
        # main circle
        for t in t_seq : 
            # real state
            if args.MODEL_MISMATCH == False : 
                x_next, y_next = dyn.step(x, w_list[t], v_list[t])
            else : 
                x_next, y_next = dyn.step_real(x, w_list[t], v_list[t])

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
                P_inv_next, h_next, hidden = agent.get_Pinv(x_hat, y_next, hidden)
                P_next_hat = est.inv(P_inv_next)
                h = h_next
            elif STATUS == 'NLS-EKF' : 
                # estimator Nonlinear Least Square-Extended Kalman Filter
                result = est.NLSF(x_hat, P_hat, [y_next], args.Q, args.R)
                # result1 = est.OPTF(x_hat, P_hat, y_next, args.Q, args.R)
                x_hat = result[ :ds]
                x_next_hat = result[ds: ]
                _, P_next_hat = est.EKF(x_hat, P_hat, y_next, args.Q, args.R)
            elif STATUS == 'NLS-UKF' : 
                result = est.NLSF(x_hat, P_hat, [y_next], args.Q, args.R)
                x_hat = result[ :ds]
                x_next_hat = result[ds: ]
                _, P_next_hat = est.UKF(x_hat, P_hat, y_next, args.Q, args.R)
            elif STATUS == 'NLS-RLF' : 
                # estimator Nonlinear Least Square-Reinforcement Learning Filter
                result = est.NLSF(x_hat, P_hat, [y_next], args.Q, args.R)
                x_hat = result[ :ds]
                x_next_hat = result[ds: ]
                P_inv_next, h_next, hidden = agent.get_Pinv(x_hat, y_next, hidden)
                P_next_hat = est.inv(P_inv_next)
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
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False 
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

def main() : 
    parser = argparse.ArgumentParser()
    # system parameters
    parser.add_argument("--state_dim", default=3, type=int, help="dimension of state variable x")
    parser.add_argument("--obs_dim", default=2, type=int, help="dimension of measurement y")
    parser.add_argument("--x0_mu", default=np.array([.2, .2, 8]), help="average of initial state distribution")
    parser.add_argument("--P0", default=.6*np.eye(3), help="covariance of initial state distribution")
    parser.add_argument("--x0_hat", default=np.array([.5, .5, .5]), help="estimation of initial state distribution average")
    parser.add_argument("--P0_hat", default=np.eye(3), help="estimation of initial state distribution covariance")
    parser.add_argument("--Q", default=np.array([[.001,0,0],[0,.001,0],[0,0,1.]]), help="covariance of process disturbance")
    parser.add_argument("--R", default=np.array([[1.,0], [0,1.]]), help="covariance of measurement noise")
    parser.add_argument("--MODEL_MISMATCH", default=False, type=bool, help="choose whether to apply model mismatch")

    # training parameters
    parser.add_argument("--max_episodes", default=1000, type=int, help="max train episodes")
    parser.add_argument("--max_train_steps", default=80, type=int, help="max simulation steps")
    parser.add_argument("--max_sim_steps", default=100, type=int, help="max simulation steps")
    parser.add_argument("--buffer_size", default=1e4, type=int, help="max size of replay buffer")
    parser.add_argument("--batch_size", default=16, type=int, help="number of samples for batch update")
    parser.add_argument("--num_steps", default=8, type=int, help="number of steps in one sample")
    parser.add_argument("--warmup_size", default=200, type=int, help="decide when to start the training of the NN")
    parser.add_argument("--hidden_layer", default=([500,500,500,500],[500,500,500,500]), help="FC layers of NN")
    parser.add_argument("--gamma", default=.9, type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-2, type=float, help="learning rate of policy net")
    parser.add_argument("--lr_policy_delta", default=5e-4, type=float, help="learning rate reduction every time")
    parser.add_argument("--lr_policy_min", default=1e-5, type=float, help="learning rate of policy net")
    parser.add_argument("--explore_Cov", default=np.array([[.001,0,0],[0,.001,0],[0,0,.001]]), help="the covariance of Guassian distribution added to predicted state")

    # file path
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log.txt", type=str, help="file to save training messages")
    parser.add_argument("--model_file", default="modelend.bin", type=str, help="trained model")
    parser.add_argument("--modelend_file", default="modelend.bin", type=str, help="trained model")
    parser.add_argument("--model_test", default="modelend.bin", type=str, help="trained model")

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise = OUnoise(args.state_dim)
    agent = RL_estimator(state_dim=args.state_dim, obs_dim=args.obs_dim, device=args.device, noise=noise, hidden_layer=args.hidden_layer, STATUS='test')
    agent.policy.to(args.device)
    # init policy network ## 可以尝试没有初始样本的——没有初始样本的目前看来不太行
    x_hat_seq, y_seq, P_hat_seq = simulate(args, rand_num=22222, STATUS='init')
    x_hat_seq = np.insert(x_hat_seq, 0, args.x0_hat, axis=0)
    P_hat_seq = np.insert(P_hat_seq, 0, args.P0_hat, axis=0)
    replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
    for t in range(args.max_train_steps) : 
        input = np.hstack((x_hat_seq[t], y_seq[t]))
        output = P2o(est.inv(P_hat_seq[t+1]), 0)
        replay_buffer.push_init((input, output))
    exp_list, _, _ = replay_buffer.sample_seq(1, args.max_train_steps-1)
    input_batch = torch.unsqueeze(torch.FloatTensor([exp[0] for exp in exp_list[0]]),dim=0)
    output_batch = torch.unsqueeze(torch.FloatTensor([exp[1] for exp in exp_list[0]]),dim=0)
    agent.policy.update_weight(input_batch, output_batch, batch_size=1, num_steps=args.max_train_steps, device=args.device, lr=1e-4)
    # save_path = os.path.join(args.output_dir, "model.bin")
    # torch.save(agent.policy.state_dict(), save_path)
    train(args, agent, replay_buffer)

    simulate(args, sim_num=50, rand_num=10086, STATUS='NLS-RLF')

if __name__ == '__main__' : 
    main()