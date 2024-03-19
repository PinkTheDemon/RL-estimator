import os
import time

from model import create_model
import estimator as est
from functions import * # 包括np
from replay_buffer import ReplayBuffer
from actor import * # 包括torch等
from def_param2 import def_param2


class ActorRNN(nn.Module) : 
    def __init__(self, dim_input, dim_output, dim_fc1=[256], dim_fc2=[256], type_activate='tanh', 
                 type_rnn='gru', dim_rnn_hidden=32, num_rnn_layers=1, batch_first=True,
                 rand_seed=111, device='cpu') -> None : 
        super(ActorRNN, self).__init__()

        torch.manual_seed(rand_seed) # 固定网络初始化权重
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_rnn_layers = num_rnn_layers
        self.dim_rnn_hidden = dim_rnn_hidden
        self.device = device

        # RNN层之前的全连接层，可以为空列表
        self.fc1 = nn.ModuleList()
        dim_in = self.dim_input
        for dim_out in dim_fc1 : 
            self.fc1.append(nn.Linear(dim_in, dim_out))
            if type_activate.lower() == 'relu' : 
                self.fc1.append(nn.ReLU())
            elif type_activate.lower() == 'tanh' : 
                self.fc1.append(nn.Tanh())
            else : 
                raise ValueError("No such activation layer type defined")
            dim_in = dim_out

        # RNN层
        if type_rnn.lower() == 'rnn' : 
            self.rnn = nn.RNN(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=batch_first)
        elif type_rnn.lower() == 'gru' : 
            self.rnn = nn.GRU(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=batch_first)
        elif type_rnn.lower() == 'lstm' : 
            self.rnn = nn.LSTM(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=batch_first)
        else : 
            raise ValueError("No such RNN type defined")

        # RNN层之后的全连接层，可以为空列表
        self.fc2 = nn.ModuleList()
        dim_in = dim_rnn_hidden
        for dim_out in dim_fc2 : 
            self.fc2.append(nn.Linear(dim_in, dim_out))
            if type_activate.lower() == 'relu' : 
                self.fc2.append(nn.ReLU())
            elif type_activate.lower() == 'tanh' : 
                self.fc2.append(nn.Tanh())
            else : 
                raise ValueError("No such activation layer type defined")
            dim_in = dim_out

        # 输出层
        self.fc2.append(nn.Linear(dim_in, dim_output))

    # input_seq: time x batch x state fanguolai
    # output_seq:time x batch x output fanguolai
    def forward(self, input_seq, hidden=None, batch_size=1) : 
        # 隐藏层输出
        if hidden is None : 
            hidden = torch.zeros((self.num_rnn_layers, batch_size, self.dim_rnn_hidden), device=self.device)

        output = input_seq.to(self.device)
        for fc1 in self.fc1 : 
            output = fc1(output)
        output, hidden = self.rnn(output, hidden)
        output = torch.tanh(output)
        for fc2 in self.fc2 : 
            output = fc2(output)
        diag_indices = (0,2,5)
        output[...,diag_indices] = F.softplus(output[...,diag_indices])

        return output, hidden

    def update_weight(self, input_seq, output_seq, batch_size, hidden_state=None, num_steps=None, lr=1e-3) : 
        input_seq = torch.FloatTensor(input_seq)
        output_seq = torch.FloatTensor(output_seq)
        input_seq, output_seq = input_seq.to(self.device), output_seq.to(self.device)
        output_seq_hat, _ = self.forward(input_seq, hidden_state, batch_size=batch_size)
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
    def __init__(self, dim_state, dim_obs, rnn_params_dict, device='cpu', STATUS='train') -> None:
        self.dim_state = dim_state
        self.dim_input = dim_state + dim_obs
        self.dim_output = ds2do(dim_state)
        self.device = device
        self.policy = ActorRNN(dim_input=self.dim_input, dim_output=self.dim_output, **rnn_params_dict).to(self.device)
        self.STATUS = STATUS

    def value(self, state, state_pre, Pinv, h=None) : 
        x = np.tile((state - state_pre), (1,1))
        Q = x @ Pinv @ x.T
        if h is not None : Q += h
        return Q

    def get_Pinv(self, state_pre, obs, hidden) : 
        input = np.tile(np.hstack((state_pre, obs)), (1,1,1))
        input = torch.from_numpy(input).float() # .float转成torch.float32类型
        output, hidden = self.policy(input, hidden)
        L = torch.zeros((output.shape[:-1])+(self.dim_state, self.dim_state), device=self.device)
        indices = torch.tril_indices(row=3, col=3, offset=0) # 获取下三角矩阵的索引
        L[..., indices[0], indices[1]] = output[..., :-1]
        P_next_inv = L @ L.permute(*range(L.dim() - 2), -1, -2)
        h_next = output[..., -1]
        return P_next_inv, h_next, hidden

    def save_model(self, save_path) : 
        torch.save(self.policy.state_dict(), save_path)

    def load_model(self, save_path) : 
        self.policy.load_state_dict(torch.load(save_path))


def train(model, args, agent:RL_estimator, replay_buffer:ReplayBuffer) : 
    log_file = LogFile(args.output_file, args.rename_option)

    ds = model.dim_state
    MSE_min = np.zeros((ds))
    num_noupdate = 0
    for i in range(args.max_episodes) : 
        '''真实轨迹'''
        x_seq, y_seq = model.generate_data(args.max_train_steps, is_mismatch=args.MODEL_MISMATCH, rand_seed=i)

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
            result = est.NLSF(x0_NLSF, P0_NLSF, y_list, model.Q, model.R)
            # x_hat_new = result[ :ds]
            x_next_hat = result[ds:2*ds]
            x_hat_seq.append(x_next_hat)
            if t < args.train_window -1 : 
                # x0_NLSF = result[:ds]
                P0_NLSF = args.P0_hat
            else : 
                x0_NLSF = x_hat_seq[t-args.train_window+1]#result[ds:2*ds]
                '''基于 x_hat_new 和 y_next 得到 P_next_hat_inv''' 
                P_next_hat_inv, h_next, hidden_next = agent.get_Pinv(x_hat, y_seq[t-args.train_window+1], hidden)
                P_next_hat_inv = P_next_hat_inv.detach().cpu().numpy().squeeze()
                h_next = h_next.detach().cpu().numpy().squeeze()
                P0_NLSF = inv(P_next_hat_inv)
                # training
                for _ in range(args.aver_num) : 
                    x_next_noise = x_next_hat + np.random.multivariate_normal(np.zeros((ds, )), args.explore_Cov)
                    x_hat_min, min_value = est.NLSF_xt(x_hat, P_hat_inv, [], x_next_noise, inv(model.Q), inv(model.R))
                    target_Q = min_value@min_value + h + \
                            (y_seq[t] - model.h(x_next_noise))@inv(model.R)@(y_seq[t] - model.h(x_next_noise)).T 
                    Q = agent.value(x_next_noise, x_next_hat, P_next_hat_inv, h_next)
                    delta = (Q - target_Q)/args.aver_num
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
                replay_buffer.push((input, output, hidden), seq)

                if replay_buffer.size > args.warmup_size : 
                    bin = []
                    bot = []
                    bhd = []
                    size = 0
                    exp_list, inf_list, isinit_batch = replay_buffer.sample_seq(args.batch_size, args.num_steps)
                    for exp_seq, inf_seq, isinit in zip(exp_list, inf_list, isinit_batch) : 
                        if inf_seq[0] : 
                            bin.append([exp[0] for exp in exp_seq])
                            bot.append([exp[1] for exp in exp_seq])
                            bhd.append(exp_seq[0][2].squeeze().detach() if exp_seq[0][2] is not None else torch.zeros((agent.policy.dim_rnn_hidden), device=agent.device))
                            size += 1
                    agent.policy.update_weight(np.array(bin), np.array(bot), size, hidden_state=torch.stack(bhd).unsqueeze(0), num_steps=args.num_steps, lr=args.lr_policy)
            # time delay
            x_hat = x_next_hat
            P_hat_inv = P_next_hat_inv
            h = h_next
            hidden = hidden_next

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
        log_file.flush()

    args.modelend_file = checkFilename(args.modelend_file)
    save_path = os.path.join(args.output_dir, args.modelend_file)
    agent.save_model(save_path=save_path)
    args_dict = vars(args)
    for key, value in args_dict.items() : 
        print(f"{key}: {value}")
    log_file.endLog()


def simulate(model, args, agent=None, estimator_dict=None, sim_num=1, rand_seed=1111, STATUS='EKF') : 
    ds = model.dim_state
    if STATUS == 'PF' : 
        pf = est.Particle_Filter(ds, model.dim_obs, int(1e4), model.f, model.h, model.x0_mu, model.P0)

    # initial set for criterion
    error = np.zeros((args.max_sim_steps,ds))
    MSE = 0
    execution_time = 0
    for i in range(sim_num) : 
        '''set random seed'''
        np.random.seed(rand_seed+i)
        '''generate disturbance and noise'''
        t_seq = range(args.max_sim_steps)
        x_seq, y_seq = model.generate_data(args.max_sim_steps, is_mismatch=args.MODEL_MISMATCH, rand_seed=rand_seed+i)

        start_time = time.process_time() # time record

        '''state estimation - different filter'''
        x_hat = args.x0_hat
        P_hat = args.P0_hat
        x_hat_seq = []
        P_hat_seq = []
        if STATUS == 'FIE' : 
            initial_x = [args.x0_hat]
            for t in t_seq : 
                result = est.NLSF(args.x0_hat, args.P0_hat, y_seq[:t+1], model.Q, model.R, initial_x=initial_x)
                x_hat_seq.append(result[-ds:])
                initial_x = list(result.reshape(-1,3))[1:]

        elif 'MHE' not in STATUS : 
            for t in t_seq : 
                if STATUS == 'EKF' or STATUS=='init': 
                    x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_seq[t], model.Q, model.R)
                elif STATUS == 'UKF' : 
                    x_next_hat, P_next_hat = est.UKF(x_hat, P_hat, y_seq[t], model.Q, model.R)
                elif STATUS == 'PF' : 
                    pf.predict(model.Q)
                    pf.update(y_seq[t], model.R)
                    x_next_hat, P_next_hat = pf.estimate()
                x_hat_seq.append(x_next_hat)
                P_hat_seq.append(P_next_hat)
                x_hat = x_next_hat
                P_hat = P_next_hat

        elif 'MHE' in STATUS : 
            hidden = None
            x0_NLSF = args.x0_hat
            x_hat = args.x0_hat
            initial_x = [args.x0_hat]
            y_list = []
            for t in t_seq : 
                y_list.append(y_seq[t])
                if len(y_list) > args.window : del y_list[0]
                result = est.NLSF_uniform(P_hat, y_list, model.Q, model.R, x0=initial_x, state_hat=x0_NLSF)
                x_next_hat = result[-ds: ]
                x_hat_seq.append(x_next_hat)
                initial_x = list(result.reshape(-1,3))[1:]
                if t < args.window - 1 : # before first full window 
                    #都不更新x0_NLSF# x0_NLSF = result[ :ds]
                    P_next_hat = args.P0_hat
                else : 
                    # x_hat_new = result[ :ds]
                    x0_NLSF = x_hat_seq[t-args.window+1]# result[ds:2*ds]# 不用最新的x而是用以前的x，考虑时序
                    if 'EKF' in STATUS : _, P_next_hat = est.EKF(x_hat, P_hat, y_seq[t-args.window+1], model.Q, model.R)
                    if 'UKF' in STATUS : _, P_next_hat = est.UKF(x_hat, P_hat, y_seq[t-args.window+1], model.Q, model.R)
                    if 'RLF' in STATUS : 
                        P_inv_next, _, hidden = agent.get_Pinv(x_hat, y_seq[t-args.window+1], hidden)
                        P_inv_next = P_inv_next.detach().cpu().numpy().squeeze()
                        P_next_hat = inv(P_inv_next)
                P_hat_seq.append(P_next_hat)
                x_hat = x_next_hat
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
        print(f"average cpu time of {STATUS}: {execution_time} ms")
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
    args, model_paras_dict, estimator_paras_dict = def_param2()
    args.num_steps = 8
    args.batch_size = 16
    args.aver_num = 20 # 5*i+5
    args.STATUS = "EKF-MHE"
    estimator_paras_dict["rnn_params_dict"]["dim_fc1"] = [64]
    estimator_paras_dict["rnn_params_dict"]["dim_rnn_hidden"] = 32
    estimator_paras_dict["rnn_params_dict"]["dim_fc2"] = [128]
    print("simulate method: ", args.STATUS)
    print("hidden_layer: ", args.hidden_layer)

    model = create_model(**model_paras_dict)
    agent = RL_estimator(**estimator_paras_dict, STATUS='test')
    # init policy network 
    x_hat_seq, y_seq, P_hat_seq = simulate(model, args, rand_seed=22222, STATUS='init')
    x_hat_seq = np.insert(x_hat_seq, 0, args.x0_hat, axis=0)
    P_hat_seq = np.insert(P_hat_seq, 0, args.P0_hat, axis=0)
    replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
    for t in range(args.max_train_steps) : 
        input = np.hstack((x_hat_seq[t], y_seq[t]))
        output = P2o(est.inv(P_hat_seq[t+1]), 0)
        replay_buffer.push_init((input, output, None), True)
    exp_list, _, _ = replay_buffer.sample_seq(1, args.max_train_steps-1)
    input_batch = torch.unsqueeze(torch.FloatTensor(np.array([exp[0] for exp in exp_list[0]])),dim=0)
    output_batch = torch.unsqueeze(torch.FloatTensor(np.array([exp[1] for exp in exp_list[0]])),dim=0)
    agent.policy.update_weight(input_batch, output_batch, batch_size=1, num_steps=args.max_train_steps, lr=1e-4)
    if 'RLF' in args.STATUS : train(model, args, agent, replay_buffer)
    # 加载模型（不训练时才需要加载）
    # args.model_test = args.modelend_file
    # model_path = os.path.join(args.output_dir, args.model_test)
    # agent.load_model(model_path)
    simulate(model, args, agent, sim_num=50, rand_seed=10086, STATUS=args.STATUS)

if __name__ == '__main__' : 
    main()

# 基于19，使代码能够用脚本运行, 修改MHE及FIE算法的窗口初始时刻状态的均值, 采用老的估计值而非新的更新值, 存储hidden_state在回放缓存中
    # 19: 基于5，修改代码结构

# 调试：调整网络结构以及学习率