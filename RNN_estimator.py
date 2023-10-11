import matplotlib.pyplot as plt
import os
import sys
import argparse
import time
from filterpy.kalman import MerweScaledSigmaPoints,UnscentedKalmanFilter

import dynamics as dyn
import estimator as est
from functions import * # 包括np
from OUnoise import OUnoise
from replay_buffer import ReplayBuffer
from actor import * # 包括torch等
from RL_estimatorRNN import Actor as ActorRNN


class RL_estimator : 
    def __init__(self, state_dim, obs_dim, device, noise:OUnoise, hidden_layer=([200],[200]), STATUS='train') -> None:
        self.state_dim = state_dim
        self.dim_output = ds2do(state_dim)
        self.dim_input = state_dim + obs_dim
        self.policy = ActorRNN(self.dim_input, self.dim_output, h1=hidden_layer[0], h2=hidden_layer[1]).to(device)
        self.device = device
        self.OUnoise = noise
        self.STATUS = STATUS

    def value(self, state, state_pre, Pinv, h=None) : 
        x = np.tile((state - state_pre), (1,1))
        Q = x @ Pinv @ x.T
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


def train(args, agent:RL_estimator, replay_buffer:ReplayBuffer) : 
    sys.stdout = open(args.output_file, 'w')

    ds = args.state_dim
    MSE_min = np.zeros((ds))
    num_noupdate = 0
    for i in range(args.max_episodes) : 
        # noise = OUnoise(ds, rand_num=i)
        # agent.reset_noise(noise) ## 是否需要每次都基于当前最好的模型来训练，但是当前最好的模型也有可能只是在当前这一集上表现好。评价好坏可能需要与EKF的MSE作对比。

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_train_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        x_hat = args.x0_hat
        P_hat_inv = inv(args.P0_hat)
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
            result = est.NLSF(x_hat, inv(P_hat_inv), [y_next], args.Q, args.R)
            x_hat_new = result[ :ds]
            x_next_hat = result[ds: ]

            # training 
            x_next_noise = x_next_hat + np.random.multivariate_normal(np.zeros((ds, )), args.explore_Cov) ## 这里用OUnoise会不会提升训练效率？
            target_Q = args.gamma * agent.value(x_hat, x_hat_new, P_hat_inv, h) + \
                    (x_next_noise - dyn.f(x_hat))@inv(args.Q)@(x_next_noise - dyn.f(x_hat)).T + \
                    (y_next - dyn.h(x_next_noise))@inv(args.R)@(y_next - dyn.h(x_next_noise)).T 
            Q = agent.value(x_next_noise, x_next_hat, P_next_hat_inv, h_next)
            delta = Q - target_Q
            P_next_new_inv = P_next_hat_inv - args.lr_value * delta * agent.value(x_next_noise.reshape(ds,1), x_next_hat.reshape(ds,1), np.eye(1)) ## 
            h_next_new = h_next - args.lr_value * delta

            input = np.hstack((x_hat, y_next))
            output = P2o(P_next_new_inv, h_next_new)
            while output is None : # Pnew不正定，减小学习率，但不能减小全局的学习率
                P_next_new_inv = (P_next_new_inv + P_next_hat_inv) / 2
                output = P2o(P_next_new_inv, h_next_new)
            if t == 0 : seq = not seq
            if t == args.max_train_steps-args.num_steps : seq = not seq
            replay_buffer.push((input, output), seq) 

            if replay_buffer.size > args.warmup_size : 
                bin = []
                bot = []
                size = 0
                exp_list, inf_list, _ = replay_buffer.sample_seq(args.batch_size, args.num_steps)
                for exp_seq, inf_seq in zip(exp_list, inf_list) : 
                    if inf_seq[0] : 
                        bin.append([exp[0] for exp in exp_seq])
                        bot.append([exp[1] for exp in exp_seq])
                        size += 1
                agent.policy.update_weight(bin, bot, size, args.num_steps, args.device, lr=args.lr_policy)

            # error evaluate, MSE
            MSE += (x - x_hat)**2 / args.max_train_steps

            # time delay
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
        noise = OUnoise(dim=ds, rand_num=rand_num)
        agent = RL_estimator(ds, args.obs_dim, args.device, noise, hidden_layer=args.hidden_layer, STATUS='test')
        model_path = os.path.join(args.output_dir, args.model_test)
        agent.policy.load_state_dict(torch.load(model_path))
    if STATUS == 'UKF' : 
        points = MerweScaledSigmaPoints(ds, alpha=1., beta=2., kappa=0.)
        ukf = UnscentedKalmanFilter(dim_x=ds, dim_z=args.obs_dim, dt=.01, fx=dyn.f, hx=dyn.h, points=points)
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
        np.random.seed(rand_num+i)
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
                P_inv_next, h_next, hidden = agent.get_Pinv(x_hat, y_next, hidden)
                P_next_hat = inv(P_inv_next)
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
                # _, P_next_hat = est.UKF(x_hat, P_hat, y_next, args.Q, args.R)
                ukf.predict()
                ukf.update(y_next)
                P_next_hat = ukf.P
            elif STATUS == 'NLS-RLF' : 
                # estimator Nonlinear Least Square-Reinforcement Learning Filter
                result = est.NLSF(x_hat, P_hat, [y_next], args.Q, args.R)
                x_hat = result[ :ds]
                x_next_hat = result[ds: ]
                P_inv_next, h_next, hidden = agent.get_Pinv(x_hat, y_next, hidden)
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
    parser.add_argument("--hidden_layer", default=([500,500,500],[500,500,500]), help="FC layers of NN")
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
        replay_buffer.push_init((input, output), True)
    exp_list, _, _ = replay_buffer.sample_seq(1, args.max_train_steps-1)
    input_batch = torch.unsqueeze(torch.FloatTensor([exp[0] for exp in exp_list[0]]),dim=0)
    output_batch = torch.unsqueeze(torch.FloatTensor([exp[1] for exp in exp_list[0]]),dim=0)
    agent.policy.update_weight(input_batch, output_batch, batch_size=1, num_steps=args.max_train_steps, device=args.device, lr=1e-4)
    # save_path = os.path.join(args.output_dir, "model.bin")
    # torch.save(agent.policy.state_dict(), save_path)
    # train(args, agent, replay_buffer)

    simulate(args, sim_num=50, rand_num=10086, STATUS='NLS-RLF')

if __name__ == '__main__' : 
    main()