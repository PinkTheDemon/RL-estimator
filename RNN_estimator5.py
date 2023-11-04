import matplotlib.pyplot as plt
import os
import sys
import time

import dynamics as dyn
import estimator as est
from functions import * # 包括np
from OUnoise import OUnoise
from replay_buffer import ReplayBuffer
from actor import * # 包括torch等
from def_param import def_param


class ActorRNN(nn.Module) : 
    def __init__(self, dim_input, dim_output, h1=[200], h2=[200], rand_num=111) -> None : 
        super(ActorRNN, self).__init__()
        torch.manual_seed(rand_num)
        self.dim_input = dim_input
        self.hidden_dim = h2[0]
        self.dim_output = dim_output

        self.fc1 = nn.ModuleList()
        self.rnn = nn.GRU(h1[-1], self.hidden_dim, batch_first=True)
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
            input_size = output_size
        self.fc2.append(nn.Linear(input_size, self.dim_output))

    # input_seq: time x batch x state fanguolai
    # output_seq:time x batch x output fanguolai 
    def forward(self, input_seq, hidden, device, batch_size=1) : 
        if hidden is None : 
            hidden = torch.zeros((self.num_directions, batch_size, self.hidden_dim), device=device)

        if isinstance(input_seq, np.ndarray) : 
            input_seq = torch.tensor(np.tile(input_seq, (1,1,1)), dtype=torch.float32, device=device)
        output = input_seq
        for fc1 in self.fc1 : 
            output = torch.tanh(fc1(output))
        output, hidden = self.rnn(output, hidden) # 要不要激活函数？
        for fc2 in self.fc2 : 
            output = fc2(torch.tanh(output))
        # output[-1] = F.relu(output[-1]) ## 不一定好，但先试出较好的网络结构再来去掉这个，加上这个更可能出现求逆的问题
        # output1 = torch.clone(output)
        # output1[:,:,0] = F.softplus(output[:,:,0])
        # output1[:,:,2] = F.softplus(output[:,:,2])
        # output1[:,:,5] = F.softplus(output[:,:,5])

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
        '''真实轨迹'''
        x, w_list, v_list = dyn.reset(sim_num=i, maxstep=args.max_train_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        x_seq = []
        y_seq = []
        for t in range(args.max_train_steps) : 
            # dynamic, x is unobservable, y is observable
            if args.MODEL_MISMATCH == False : 
                x_next, y_next = dyn.step(x, w_list[t], v_list[t])
            else : 
                x_next, y_next = dyn.step_real(x, w_list[t], v_list[t])
            x = x_next
            x_seq.append(x_next)
            y_seq.append(y_next)

        '''状态估计'''
        h = 0
        hidden = None
        x_hat = args.x0_hat
        x0_NLSF = args.x0_hat
        P0_NLSF = args.P0_hat
        P_hat_inv = inv(args.P0_hat)
        x_hat_seq = []
        P_hat_seq = []
        y_list = []
        seq = False
        np.random.seed(i)
        for t in range(args.max_train_steps) : 
            y_list.append(y_seq[t])
            if len(y_list) > args.train_window : del y_list[0]
            '''求解非线性最小二乘，得到 x_next_hat 和 x_hat_new'''
            result = est.NLSF(x0_NLSF, P0_NLSF, y_list, args.Q, args.R)
            x_hat_new = result[ :ds]
            x_next_hat = result[ds:2*ds]
            x_hat_seq.append(x_next_hat)
            if t < args.train_window -1 : 
                x0_NLSF = result[:ds]
                P0_NLSF = args.P0_hat
            else : 
                x0_NLSF = result[ds:2*ds]
                '''基于 x_hat_new 和 y_next 得到 P_next_hat_inv''' 
                P_next_hat_inv, h_next, hidden = agent.get_Pinv(x_hat_new, y_seq[t-args.window+1], hidden)
                P0_NLSF = inv(P_next_hat_inv)
                # training
                target_Q_list = []
                Q_list = []
                for _ in range(args.aver_num) : 
                    x_next_noise = x_next_hat + np.random.multivariate_normal(np.zeros((ds, )), args.explore_Cov)
                    # x_hat_min = est.NLSF_xt(x_hat_new, P_hat_inv, [], x_next_noise, inv(args.Q), inv(args.R))
                    target_Q = args.gamma * agent.value(x_hat, x_hat_new, P_hat_inv, h) + \
                            (x_next_noise - dyn.f(x_hat_new))@inv(args.Q)@(x_next_noise - dyn.f(x_hat_new)).T + \
                            (y_seq[t] - dyn.h(x_next_noise))@inv(args.R)@(y_seq[t] - dyn.h(x_next_noise)).T 
                    Q = agent.value(x_next_noise, x_next_hat, P_next_hat_inv, h_next)
                    target_Q_list.append(target_Q)
                    Q_list.append(Q)
                target_Q = sum(target_Q_list)/len(target_Q_list)
                Q = sum(Q_list)/len(Q_list)
                delta = Q - target_Q
                P_next_new_inv = P_next_hat_inv - args.lr_value * delta * agent.value(x_next_noise.reshape(ds,1), x_next_hat.reshape(ds,1), np.eye(1)) ## 这里的x_next_noise是上面的最后一次的，这完全是错误的写法呀
                h_next_new = h_next - args.lr_value * delta

                input = np.hstack((x_hat, y_seq[t]))
                output = P2o(P_next_new_inv, h_next_new)
                output_judge = 0
                while output is None : # Pnew不正定，减小学习率，但不能减小全局的学习率
                    if output_judge < 5 : 
                        P_next_new_inv = (P_next_new_inv + P_next_hat_inv) / 2
                        output = P2o(P_next_new_inv, h_next_new)
                        output_judge += 1
                    else : 
                        output = P2o(P_next_hat_inv, h_next_new)
                        break
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
                    agent.policy.update_weight(np.array(bin), np.array(bot), size, args.num_steps, args.device, lr=args.lr_policy)
                # time delay
                x_hat = x_next_hat
                P_hat_inv = P_next_hat_inv
                h = h_next

        # error evaluate, MSE
        x_seq = np.array(x_seq)
        x_hat_seq = np.array(x_hat_seq)
        MSE = np.square(x_seq - x_hat_seq).sum(0) / args.max_train_steps

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
    if 'RLF' in STATUS : 
        noise = OUnoise(dim=ds, rand_num=rand_num)
        agent = RL_estimator(ds, args.obs_dim, args.device, noise, hidden_layer=args.hidden_layer, STATUS='test')
        model_path = os.path.join(args.output_dir, args.model_test)
        agent.policy.load_state_dict(torch.load(model_path))
    if STATUS == 'PF' : 
        pf = est.Particle_Filter(ds, args.obs_dim, int(1e4), dyn.f, dyn.h, args.x0_mu, args.P0)

    # initial set for criterion
    error = np.zeros((args.max_sim_steps,ds))
    MSE = 0
    execution_time = 0
    for i in range(sim_num) : 
        '''set random seed'''
        np.random.seed(rand_num+i)
        '''generate disturbance and noise'''
        x, w_list, v_list = dyn.reset(rand_num+i, args.max_sim_steps, x0_mu=args.x0_mu, P0=args.P0, disturb_Q=args.Q, noise_R=args.R)
        t_seq = range(args.max_sim_steps)

        x_seq = []
        y_seq = []
        for t in t_seq : # 真实轨迹
            if args.MODEL_MISMATCH == False : 
                x_next,y_next = dyn.step(x,w_list[t],v_list[t])
            else : 
                x_next, y_next = dyn.step_real(x, w_list[t], v_list[t])
            x = x_next
            x_seq.append(x_next)
            y_seq.append(y_next)

        start_time = time.process_time() # time record

        '''state estimation - different filter'''
        x_hat = args.x0_hat
        P_hat = args.P0_hat
        x_hat_seq = []
        P_hat_seq = []
        if STATUS == 'FIE' : 
            result = est.NLSF(args.x0_hat, args.P0_hat, y_seq, args.Q, args.R)
            for t in t_seq : 
                x_hat_seq.append(result[ds:2*ds])
                result = result[ds:]

        elif 'MHE' not in STATUS : 
            for t in t_seq : 
                if STATUS == 'EKF' or STATUS=='init': 
                    x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_seq[t], args.Q, args.R)
                elif STATUS == 'UKF' : 
                    x_next_hat, P_next_hat = est.UKF(x_hat, P_hat, y_seq[t], args.Q, args.R)
                elif STATUS == 'PF' : 
                    pf.predict(args.Q)
                    pf.update(y_seq[t], args.R)
                    x_next_hat, P_next_hat = pf.estimate()
                x_hat_seq.append(x_next_hat)
                P_hat_seq.append(P_next_hat)
                x_hat = x_next_hat
                P_hat = P_next_hat

        elif 'MHE' in STATUS : 
            hidden = None
            x0_NLSF = args.x0_hat
            y_list = []
            for t in t_seq : 
                y_list.append(y_seq[t])
                if len(y_list) > args.window : del y_list[0]
                result = est.NLSF(x0_NLSF, P_hat, y_list, args.Q, args.R)
                x_next_hat = result[-ds: ]
                if t < args.window - 1 : # before first full window 
                    x0_NLSF = result[ :ds]
                    P_next_hat = args.P0_hat
                else : 
                    x_hat_new = result[ :ds]
                    x0_NLSF = result[ds:2*ds]
                    if 'EKF' in STATUS : _, P_next_hat = est.EKF(x_hat_new, P_hat, y_seq[t-args.window+1], args.Q, args.R)
                    if 'UKF' in STATUS : _, P_next_hat = est.UKF(x_hat_new, P_hat, y_seq[t-args.window+1], args.Q, args.R)
                    if 'RLF' in STATUS : 
                        P_inv_next, _, hidden = agent.get_Pinv(x_hat_new, y_seq[t-args.window+1], hidden)
                        P_next_hat = inv(P_inv_next)
                x_hat_seq.append(x_next_hat)
                P_hat_seq.append(P_next_hat)
                P_hat = P_next_hat

        # time evaluate, ms
        end_time = time.process_time()
        execution_time += 1000 * (end_time - start_time) / args.max_sim_steps / sim_num

        # error evaluate, MSE
        x_seq = np.array(x_seq)
        x_hat_seq = np.array(x_hat_seq)
        error += np.abs(x_seq - x_hat_seq) / sim_num
        MSE += np.square(x_seq - x_hat_seq).sum(0) / args.max_sim_steps / sim_num

    if STATUS != 'init' : 
        # evaluation criterion print
        print(f"average cpu time of {STATUS}: {execution_time}")
        print(f"MSE of {STATUS}: {MSE}")
        print(f"lr_policy : {args.lr_policy}")

        # # plot
        # # plt.rcParams['font.sans-serif'] = ['SimHei']
        # # plt.rcParams['axes.unicode_minus'] = False 
        # # plt.rcParams['font.size'] = 28
        # fig, axs = plt.subplots(ds,1)
        # for i in range(ds) : 
        #     axs[i].plot(t_seq, x_seq[:,i], label='x_real', color='blue')
        #     axs[i].plot(t_seq, x_hat_seq[:,i], label='x_hat', color='red')
        #     axs[i].set_xlim(0, args.max_sim_steps)
        #     axs[i].set_ylabel(f'x{i+1}')
        # axs[0].set_title(f'{STATUS}')
        # # axs[0].set_title('状态估计效果图')
        # axs[0].legend()
        # # axs[-1].set_xlabel('时间步')

        # fig, ax = plt.subplots()
        # color = ['b','g','r','c','m','y','k']
        # for i in range(ds) : 
        #     ax.plot(t_seq, error[:,i], label=f'x{i+1}', color=color[i], linestyle='--')
        #     ax.plot(t_seq, np.average(error[:,i])*np.ones_like(t_seq), color=color[i])
        # ax.set_xlim(0, args.max_sim_steps)
        # # ax.set_xlabel('时间步')
        # # ax.set_ylabel('绝对值误差')
        # ax.set_title(f'MSE = {MSE}')
        # # ax.set_title('绝对值误差')
        # ax.legend()
        # plt.show()

    return x_hat_seq, y_seq, P_hat_seq

def main() : 
    for i in range(1) : 
        args = def_param()
        args.max_episodes = 500 # 即使提到1000估计也并不会有太大提升吧
        args.hidden_layer = ([500],[500])
        args.num_steps = 8
        args.batch_size = 16
        args.window = 1
        args.train_window = 1
        args.aver_num = 20 # 5*i+5
        # if i == 1 : args.explore_Cov = np.array([[.001,0,0],[0,.01,0],[0,0,.001]])
        # args.model_test = 'model3.bin'
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # args.gamma = 0.82+0.3*i

        noise = OUnoise(args.state_dim)
        agent = RL_estimator(state_dim=args.state_dim, obs_dim=args.obs_dim, device=args.device, noise=noise, hidden_layer=args.hidden_layer, STATUS='test')
        agent.policy.to(args.device)
        # init policy network 
        x_hat_seq, y_seq, P_hat_seq = simulate(args, rand_num=22222, STATUS='init')
        x_hat_seq = np.insert(x_hat_seq, 0, args.x0_hat, axis=0)
        P_hat_seq = np.insert(P_hat_seq, 0, args.P0_hat, axis=0)
        replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
        for t in range(args.max_train_steps) : 
            input = np.hstack((x_hat_seq[t], y_seq[t]))
            output = P2o(est.inv(P_hat_seq[t+1]), 0)
            replay_buffer.push_init((input, output), True)
        exp_list, _, _ = replay_buffer.sample_seq(1, args.max_train_steps-1)
        input_batch = torch.unsqueeze(torch.FloatTensor(np.array([exp[0] for exp in exp_list[0]])),dim=0)
        output_batch = torch.unsqueeze(torch.FloatTensor(np.array([exp[1] for exp in exp_list[0]])),dim=0)
        agent.policy.update_weight(input_batch, output_batch, batch_size=1, num_steps=args.max_train_steps, device=args.device, lr=1e-4)
        train(args, agent, replay_buffer)

        simulate(args, sim_num=50, rand_num=10086, STATUS=f'MHE-RLF{i}')

if __name__ == '__main__' : 
    main()
    print('change: 双向RNN、确保满秩都不行')