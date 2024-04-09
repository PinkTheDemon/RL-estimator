import torch
from torch import nn 
from torch.optim import Adam 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import estimator as est
from simulate import simulate
from functions import * # 包括np
from def_param2 import def_param2, set_params
from model import Model, create_model
from plot import plotReward


class ActorRNN(nn.Module):
    def __init__(self, dim_input, dim_output, dim_fc1=[256], dim_fc2=[256], type_activate='tanh', 
                 type_rnn='gru', dim_rnn_hidden=32, num_rnn_layers=1, batch_first=True, 
                 rand_seed=111, device='cpu') -> None:
        super(ActorRNN, self).__init__()
        #region 属性定义以及固定随机数种子(固定网络初始化权重)
        torch.manual_seed(rand_seed)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_rnn_layers = num_rnn_layers
        self.dim_rnn_hidden = dim_rnn_hidden
        self.device = device
        #endregion
        #region RNN层之前的全连接层，可以为空列表
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
        #endregion
        #region RNN层
        if type_rnn.lower() == 'rnn' : 
            self.rnn = nn.RNN(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=batch_first)
        elif type_rnn.lower() == 'gru' : 
            self.rnn = nn.GRU(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=batch_first)
        elif type_rnn.lower() == 'lstm' : 
            self.rnn = nn.LSTM(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=batch_first)
        else : 
            raise ValueError("No such RNN type defined")
        #endregion
        #region RNN层之后的全连接层，可以为空列表
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
        #endregion
        #region 输出层
        self.fc2.append(nn.Linear(dim_in, dim_output))
        #endregion
    # end function __init__
    def forward(self, input_seq, hidden=None, batch_size=1):
        '''
        param : input_seq : batch_size x time_steps x dim_input
        output: output_seq: batch_size x time_steps x dim_output
        '''
        #region 隐藏层初始化
        if hidden is None : 
            hidden = torch.zeros((self.num_rnn_layers, batch_size, self.dim_rnn_hidden), device=self.device)
        #endregion
        #region 前向传播
        output = input_seq.to(self.device)
        for fc1 in self.fc1 : 
            output = fc1(output)
        output, hidden = self.rnn(output, hidden)
        # output = torch.tanh(output) ## rnn内部有tanh激活函数，所以不需要再次加tanh
        for fc2 in self.fc2 : 
            output = fc2(output)
        #endregion
        #region 确保对角线元素为非零 ## 是否需要
        diag_indices = (0,2,5)
        output[...,diag_indices] = F.softplus(output[...,diag_indices])
        #endregion
        #region 将输出转换成矩阵形式
        ds = do2ds(self.dim_output)
        L = torch.zeros((output.shape[:-1])+(ds, ds), device=self.device)
        indices = torch.tril_indices(row=ds, col=ds, offset=0) # 获取下三角矩阵的索引
        L[..., indices[0], indices[1]] = output[..., :-1]
        P_next_inv = L @ L.permute(*range(L.dim() - 2), -1, -2)
        h_next = output[..., -1]
        #endregion
        return P_next_inv, h_next, hidden
    # end function forward

def grad_clipping(net, theta) -> None:
    if isinstance(net, nn.Module) : 
        params = [p for p in net.parameters() if p.requires_grad]
    else : 
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))  ## p.grad 的括号是否可以去掉？
    if norm > theta : 
        for param in params : 
            param.grad[:] *= theta / norm
# end function grad_clipping

class RL_estimator():
    def __init__(self, dim_state, dim_obs, lr, lr_min, rnn_params_dict, device='cpu') -> None:
        self.dim_state = dim_state
        self.dim_input = dim_state + dim_obs
        self.dim_output = ds2do(dim_state)
        self.device = device
        self.policy = ActorRNN(dim_input=self.dim_input, dim_output=self.dim_output, **rnn_params_dict).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=50, factor=0.5, min_lr=lr_min, verbose=True)
    # end function __init__
    def reset(self, x0_hat, P0_hat) -> None:
        self.x_hat = x0_hat
        # 保持后续计算的统一性给P矩阵转成tensor
        self.P_inv = torch.FloatTensor(inv(P0_hat))
        self.hidden = None
    # end function reset
    def estimate(self, y, Q, R):
        result = est.NLSF_uniform(self.P_inv.detach().squeeze().cpu().numpy(), y_seq=[y], Q=Q, R=R, mode="quadratic", x0=[self.x_hat], x0_bar=self.x_hat)
        self.status = result.status
        input = np.tile(np.hstack((self.x_hat, y)), (1,1,1))
        input = torch.from_numpy(input).float().to(self.device)
        self.P_inv, h, self.hidden = self.policy.forward(input, self.hidden)
        self.x_hat = result.x[-self.dim_state: ]
        return self.x_hat, self.P_inv, h
    # end function estimate
    def value(self, x, x_bar, P_inv, h=None):
        x = torch.Tensor((x - x_bar)).unsqueeze(0)
        Q = x @ P_inv.squeeze() @ x.T
        if h is not None:Q += h.squeeze()
        return Q
    # end function value
    def save_model(self, fileName) -> None:
        torch.save(self.policy.state_dict(), fileName)
        baseName, _ = os.path.splitext(fileName)
        torch.save(self.optimizer.state_dict(), baseName+".opt")
        torch.save(self.scheduler.state_dict(), baseName+".sch")
        print(f"save model at {fileName}")
    # end function save_model
    def load_model(self, fileName) -> None:
        self.policy.load_state_dict(torch.load(fileName))
        baseName, _ = os.path.splitext(fileName)
        self.optimizer.load_state_dict(torch.load(baseName+".opt"))
        self.scheduler.load_state_dict(torch.load(baseName+".sch"))
    # end function load_model

def train(model:Model, agent:RL_estimator, args) -> None:
    #region 重定位系统输出
    log_file = LogFile(args.output_file, args.rename_option)
    #endregion
    #region 记录损失
    loss_seq = []
    #endregion
    for i in range(args.max_episodes) : 
        #region 生成真实轨迹
        x_seq, y_seq = model.generate_data(args.max_train_steps, is_mismatch=args.MODEL_MISMATCH, rand_seed=i)
        #endregion
        #region 状态估计
        agent.reset(args.x0_hat, args.P0_hat)
        x_hat_seq = []
        P_inv_seq = []
        y_list = []
        h = torch.Tensor([0])
        targetQ_seq = []
        Q_seq = []
        for t in range(args.max_train_steps) : 
            #region 获取真实观测值
            y_list.append(y_seq[t])
            if len(y_list) > args.train_window : del y_list[0]
            #endregion
            #region 求解窗口长度为1的非线性最小二乘，得到 x_next_hat
            x_next_hat, P_inv_next, h_next = agent.estimate(y_seq[t], model.Q, model.R)
            x_hat_seq.append(x_next_hat)
            P_inv_seq.append(P_inv_next.detach().squeeze().cpu().numpy())
            #endregion
            #region 计算targetQ和Q
            targetQ_list = []
            Q_list = []
            for _ in range(args.aver_num) : 
                x_next_noise = x_next_hat + np.random.multivariate_normal(np.zeros((model.dim_state, )), args.explore_Cov)
                if t < args.train_window : # 窗口未满或刚满，窗口初始为args.x0_hat
                    result = est.NLSF_uniform(inv(args.P0_hat), y_seq=y_list[ :-1], Q=model.Q, R=model.R, mode="quadratic-end", 
                                                    x0=[args.x0_hat]+x_hat_seq[:-1], x0_bar=args.x0_hat, xend=x_next_noise)
                else : # 窗口已满，窗口初始为x_hat_seq[t-args.train_window]
                    result = est.NLSF_uniform(P_inv_seq[t-args.train_window], y_seq=y_list[ :-1], Q=model.Q, R=model.R, mode="quadratic-end", 
                                                    x0=x_hat_seq[t-args.train_window:-1], x0_bar=x_hat_seq[t-args.train_window], xend=x_next_noise)
                # end if t(step)
                min_fun_value = result.fun
                targetQ = min_fun_value@min_fun_value + (y_list[-1] - model.h(x_next_noise))@inv(model.R)@(y_list[-1] - model.h(x_next_noise)) + h.item()
                Q = agent.value(x=x_next_noise, x_bar=x_next_hat, P_inv=P_inv_next, h=h_next)
                targetQ_list.append(targetQ)
                Q_list.append(Q)
            targetQ_seq.append(targetQ_list)
            Q_seq.append(Q_list)
            h = h_next
            #endregion
        # end for t(step)
        #endregion 状态估计
        #region 网络参数更新
        Q_seq = torch.stack([torch.stack(Q_list) for Q_list in Q_seq]).squeeze()
        targetQ_seq = torch.from_numpy(np.stack(targetQ_seq)).float().squeeze().to(device=agent.device)
        loss = F.mse_loss(Q_seq, targetQ_seq)
        agent.optimizer.zero_grad(set_to_none=True) ## 
        loss.backward()
        grad_clipping(agent.policy, 10)
        agent.optimizer.step()
        loss_seq.append(loss.item())
        #endregion
        #region MSE指标计算并打印（RMSE是计算所有误差的平方平均然后开方）
        x_seq = np.array(x_seq)
        x_hat_seq = np.array(x_hat_seq)
        MSE = np.square(x_seq - x_hat_seq).sum(0) / args.max_train_steps
        RMSE = np.sqrt(np.mean(MSE))
        print(f"{i}: MSE = {MSE}, RMSE = {RMSE}, loss = {loss.item()}\n")
        log_file.flush()
        #endregion
        #region 学习率更新
        agent.scheduler.step(loss) ## RMSE
        #endregion
        #region 保存学习过程中loss最低的模型
        if len(loss_seq) < 2 or loss.item() < min(loss_seq[:-1]) : 
            agent.save_model(args.model_file)
        #endregion
    # end for i(episode)
    #region 保存网络参数文件，以及打印相关参数，绘制损失曲线
    agent.save_model(args.modelend_file)
    for key, value in vars(args).items() : 
        print(f"{key}: {value}")
    print("optimizer params: ")
    print(f"betas: {agent.optimizer.param_groups[0]['betas']}")
    print(f"weight_decay: {agent.optimizer.param_groups[0]['weight_decay']}")
    log_file.flush()
    log_file.endLog()
    plotReward(loss_seq, filename="picture/train_loss.png")
    #endregion
# end function train

def main():
    #region 加载相关参数
    args = def_param2()
    args.aver_num = 5 # 有提升但提升不大，或许最后可以靠这个提高一点点性能
    args.train_window = 1
    args.max_episodes = 5000
    args.output_file = "output/log29.txt"
    args.model_file = checkFilename("output/model.mdl")
    args.modelend_file = checkFilename("output/modelend.mdl")
    args.rename_option = True
    args.STATUS = "RLF-MHE"
    args.hidden_layer = ([256], 64, [64])
    print("simulate method: ", args.STATUS)
    print("hidden_layer: ", args.hidden_layer)
    model_paras_dict, estimator_paras_dict = set_params(args=args)
    model = create_model(**model_paras_dict)
    agent = RL_estimator(**estimator_paras_dict)
    #endregion
    #region 策略网络初始化 
    x_hat_seq, y_seq, P_hat_seq = simulate(model, args, rand_seed=22222, STATUS='init')
    x_hat_seq = np.insert(x_hat_seq, 0, args.x0_hat, axis=0)
    P_hat_seq = np.insert(P_hat_seq, 0, args.P0_hat, axis=0)
    input_seq = []
    target_Pinv_seq = []
    target_h_seq = []
    for t in range(args.max_train_steps) : 
        input_seq.append(np.hstack((x_hat_seq[t], y_seq[t])))
        target_Pinv_seq.append(est.inv(P_hat_seq[t+1]))
    input_seq = torch.FloatTensor(np.stack(input_seq)).unsqueeze(0).to(agent.device)
    Pinv_seq, h_seq, _ = agent.policy.forward(input_seq, None)
    target_Pinv_seq = torch.FloatTensor(np.stack(target_Pinv_seq)).unsqueeze(0).to(agent.device)
    target_h_seq = torch.zeros_like(h_seq, device=agent.device)
    loss = F.mse_loss(Pinv_seq, target_Pinv_seq)+F.mse_loss(h_seq, target_h_seq)
    optimizer = Adam(agent.policy.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_clipping(agent.policy, 10)
    optimizer.step()
    #endregion
    #region 模型训练
    if 'RLF' in args.STATUS : train(model, agent, args)
    #endregion
    #region 加载模型（不训练时才需要加载）
    # agent.load_model("output/modelend(256 64 64).mdl")
    #endregion
    #region 测试
    simulate(model, args, agent, sim_num=50, rand_seed=10086, STATUS=args.STATUS)
    #endregion
# end function main

if __name__ == '__main__':
    main()

# 基于24, 做FIE（或长视窗）的学习
