import matplotlib.pyplot as plt
import os
import sys
import time
# from filterpy.kalman import MerweScaledSigmaPoints,UnscentedKalmanFilter

import dynamics as dyn
import estimator as est
from functions import *
from OUnoise import *
from replay_buffer import *
from actor import *


class RL_estimator : 
    def __init__(self, state_dim, obs_dim, noise:OUnoise, hidden_layer=[200,200,200,200,200,200,200,200,200,200,200],
                 rand_num=111, STATUS='train') -> None:
        self.state_dim = state_dim
        self.policy = Actor(state_dim, obs_dim, h=hidden_layer, rand_num=rand_num)
        self.OUnoise = noise
        self.STATUS = STATUS

    def value(self, state, state_pre, Pinv, h) : 
        Q = (state - state_pre) @ Pinv @ (state - state_pre).T + h
        return Q

    def get_Pinv(self, state_pre, obs, Pinv_now, h) : 
        noise = self.OUnoise.noise()
        noise = (self.STATUS=='train')*noise
        output_last = P2o(Pinv_now, h)
        input = np.hstack((state_pre, obs, output_last))
        output = self.policy(input).detach().numpy() # + noise
        dim_state = do2ds(output.size)
        L = np.zeros((dim_state, dim_state))
        for i in range(dim_state) : 
            L[i][ :i+1] = np.copy(output[ :i+1])
            output = output[i+1: ]
        P_next_inv = L @ L.T
        h_next = output[-1]

        return P_next_inv, h_next


def train(args, agent:RL_estimator, replay_buffer:ReplayBuffer) : 
    sys.stdout = open(args.output_file, 'w')

    MSE_min = np.zeros((args.state_dim))
    num_noupdate = 0
    for i in range(args.max_episodes) : 
        # noise = OUnoise(args.state_dim, rand_num=i)
        # agent.reset_noise(noise) ## 是否需要每次都基于当前最好的模型来训练，但是当前最好的模型也有可能只是在当前这一集上表现好。评价好坏可能需要与EKF的MSE作对比。

        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_train_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        x_hat = args.x0_hat
        P_hat_inv = est.inv(args.P0_hat)
        h = 0
        MSE = np.zeros((args.max_episodes, args.state_dim))
        for t in range(args.max_train_steps) : 
            # dynamic, x is unobservable, y is observable
            if args.MODEL_MISMATCH == False : 
                x_next,y_next = dyn.step(x,w_list[t],v_list[t])
            else : 
                x_next = real_fx(x, w_list[t])
                y_next = real_hx(x_next, v_list[t])

            # get covarience matrix P 
            P_next_hat_inv, h_next = agent.get_Pinv(x_hat, y_next, P_hat_inv, h)

            # solve optimization problem, get x_next_hat
            result = est.NLSF(x_hat, est.inv(P_hat_inv), [y_next], args.Q, args.R)
            x_hat_new = result[ :args.state_dim]
            x_next_hat = result[args.state_dim: ]

            # training 
            x_next_noise = x_next_hat + np.random.multivariate_normal(np.zeros((args.state_dim, )), args.explore_Cov) ## 这里用OUnoise会不会提升训练效率？
            target_Q = args.gamma * agent.value(x_hat, x_hat_new, P_hat_inv, h) + \
                    (x_next_noise - dyn.f(x_hat))@est.inv(args.Q)@(x_next_noise - dyn.f(x_hat)).T + \
                    (y_next - dyn.h(x_next_hat))@est.inv(args.R)@(y_next - dyn.h(x_next_hat)).T 
            Q = agent.value(x_next_noise, x_next_hat, P_next_hat_inv, h_next)
            delta = Q - target_Q
            P_next_new_inv = P_next_hat_inv - args.lr_value * delta * ((x_next_noise - x_next_hat)@(x_next_noise - x_next_hat).T) ## 梯度下降不能保证正定-不正定就不做更新直接跳过
            h_next_new = h_next - args.lr_value * delta

            input = np.hstack((x_hat, y_next, P2o(P_hat_inv, h)))
            output = P2o(P_next_new_inv, h_next_new)
            while output is None : # Pnew不正定，减小学习率，但不能减小全局的学习率
                P_next_new_inv = (P_next_new_inv + P_next_hat_inv) / 2
                output = P2o(P_next_new_inv, h_next_new)
            replay_buffer.push(input, output) 
            if replay_buffer.size > args.warmup_size : 
                input_batch, output_batch = replay_buffer.sample(args.batch_size)
                input_batch = torch.FloatTensor(input_batch)
                output_batch = torch.FloatTensor(output_batch)
                agent.policy.update_weight(input_batch, output_batch, lr=args.lr_policy)

            # error evaluate, MSE
            MSE[i] += (x - x_hat)**2 / args.max_train_steps

            # time delay
            x = x_next
            y = y_next
            x_hat = x_next_hat
            P_hat_inv = P_next_hat_inv
            h = h_next

        print(i, ': MSE = ', MSE[i], '\n')
        num_noupdate += 1
        if (MSE[i] <= MSE_min).all() or i == 0 : 
            MSE_min = MSE[i]
            save_path = os.path.join(args.output_dir, args.model_file)
            torch.save(agent.policy.state_dict(), save_path)
            num_noupdate = 0
        elif num_noupdate >= 50 and (args.lr_policy/2) > args.lr_policy_min : 
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
    ds = args.state_dim
    if STATUS == 'NLS-RLF' or STATUS == 'RLF' : 
        noise = OUnoise(dim=ds, rand_num=rand_num)
        dim_input = ds + args.obs_dim
        dim_output = int(ds*(ds+1)/2 + 1)
        dim_input = dim_input + dim_output
        agent = RL_estimator(dim_input, dim_output, noise, hidden_layer=args.hidden_layer, STATUS='test')
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
                P_inv_next, h_next = agent.get_Pinv(x_hat, y_next, est.inv(P_hat), h)
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
                P_inv_next, h_next = agent.get_Pinv(x_hat, y_next, est.inv(P_hat), h)
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
