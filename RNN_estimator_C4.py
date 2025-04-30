import numpy as np
import os
import torch
from torch import nn 
from torch.optim import Adam 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import params as pm
import functions as fun
import estimator as est
from plot import plotReward
from gendata import getData
from simulate import simulate
from model import Model, getModel, Continuous4

class ActorRNN(nn.Module):
    def __init__(self, dim_input, dim_output, dim_fc1=[256], dim_fc2=[256], type_activate='tanh', 
                 type_rnn='gru', dim_rnn_hidden=32, num_rnn_layers=1, dropout=0, rand_seed=111, device='cpu') -> None: # 修改：增加dropout
        super().__init__()
        #region 属性定义以及固定随机数种子(固定网络初始化权重)
        torch.manual_seed(rand_seed)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_rnn_layers = num_rnn_layers
        self.dim_rnn_hidden = dim_rnn_hidden
        self.type_activate = type_activate.lower()
        self.device = device
        #endregion
        #region RNN层之前的全连接层，可以为空列表
        self.fc1 = nn.ModuleList()
        dim_in = self.dim_input
        for dim_out in dim_fc1 : 
            self.fc1.append(nn.Linear(dim_in, dim_out))
            if type_activate.lower() == 'relu' : 
                self.fc1.append(nn.ReLU())
            elif type_activate.lower() == 'leaky_relu' : 
                self.fc1.append(nn.LeakyReLU(negative_slope=0.05))
            elif type_activate.lower() == 'prelu' : 
                self.fc1.append(nn.PReLU())
            elif type_activate.lower() == 'elu' : 
                self.fc1.append(nn.ELU(alpha=1.0))
            elif type_activate.lower() == 'tanh' : 
                self.fc1.append(nn.Tanh())
            elif type_activate.lower() == 'sigmoid' : 
                self.fc1.append(nn.Sigmoid())
            else : 
                raise ValueError("No such activation layer type defined")
            dim_in = dim_out
        #endregion
        #region RNN层
        if type_rnn.lower() == 'rnn' : 
            self.rnn = nn.RNN(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=True, dropout=dropout)
        elif type_rnn.lower() == 'gru' : 
            self.rnn = nn.GRU(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=True, dropout=dropout)
        elif type_rnn.lower() == 'lstm' : 
            self.rnn = nn.LSTM(input_size=dim_in, hidden_size=dim_rnn_hidden, num_layers=num_rnn_layers, batch_first=True, dropout=dropout)
        else : 
            raise ValueError("No such RNN type defined")
        #endregion
        #region RNN层之后的全连接层，可以为空列表
        self.fc2 = nn.ModuleList()
        dim_in = dim_rnn_hidden
        for dim_out in dim_fc2 : 
            self.fc2.append(nn.Linear(dim_in, dim_out)) # 这个可以尝试换位置
            if type_activate.lower() == 'relu' : 
                self.fc2.append(nn.ReLU())
            elif type_activate.lower() == 'leaky_relu' : 
                self.fc2.append(nn.LeakyReLU(negative_slope=0.05))
            elif type_activate.lower() == 'prelu' : 
                self.fc2.append(nn.PReLU())
            elif type_activate.lower() == 'elu' : 
                self.fc2.append(nn.ELU(alpha=1.0))
            elif type_activate.lower() == 'tanh' : 
                self.fc2.append(nn.Tanh())
            elif type_activate.lower() == 'sigmoid' : 
                self.fc2.append(nn.Sigmoid())
            else : 
                raise ValueError("No such activation layer type defined")
            dim_in = dim_out
        #endregion
        #region 输出层
        self.out = nn.ModuleList()
        self.out.append(nn.Linear(dim_in, self.dim_output)) # L.flatten
        self.out.append(nn.Linear(dim_in, 1)) # c
        self.weight_init()
        #endregion
    # end function __init__
    def forward(self, input_seq, hidden=None): # hidden=None时会自动初始化为全零张量
        '''
        param : input_seq : batch_size x time_steps x dim_input
        output: output_seq: batch_size x time_steps x dim_output
        '''
        #region 前向传播
        output = input_seq.to(self.device)
        for fc1 in self.fc1 : 
            output = fc1(output)
        output, hidden = self.rnn(output, hidden)
        # output = torch.tanh(output) ## rnn内部有tanh激活函数，所以不需要再次加tanh
        for fc2 in self.fc2 : 
            output = fc2(output)
        outputs = []
        for fc in self.out :
            outputs.append(fc(output))
        #endregion
        #region 确保对角线元素为非零 ## 是否需要，以及加上之后自动计算梯度是否还生效
        # diag_indices = (0,2,5)
        # output[0][...,diag_indices] = F.softplus(output[0][...,diag_indices])
        #endregion
        #region 将输出转换成矩阵形式
        ds = fun.do2ds(self.dim_output)
        L = torch.zeros((outputs[0].shape[:-1])+(ds, ds), device=self.device)
        indices = torch.tril_indices(row=ds, col=ds, offset=0) # 获取下三角矩阵的索引
        L[..., indices[0], indices[1]] = outputs[0]
        P_next_inv = L @ L.permute(*range(L.dim() - 2), -1, -2)
        h_next = outputs[1]
        #endregion
        return P_next_inv, h_next, hidden
    # end function forward
    def weight_init(self) : 
        if self.type_activate != "elu" and self.type_activate != "prelu":
            nonlinearity = self.type_activate
        else :
            nonlinearity = "leaky_relu"
        # for fc in self.fc1 : 
        #     if isinstance(fc, nn.Linear) : 
        #         nn.init.kaiming_uniform_(fc.weight, mode="fan_in", nonlinearity=nonlinearity)
        # for fc in self.fc2 : 
        #     if isinstance(fc, nn.Linear) : 
        #         nn.init.kaiming_uniform_(fc.weight, mode="fan_in", nonlinearity=nonlinearity)
        # for fc in self.out : 
        #     if isinstance(fc, nn.Linear) : 
        #         nn.init.kaiming_uniform_(fc.weight, mode="fan_in", nonlinearity=nonlinearity)
        for name, param in self.named_parameters():
            if 'weight' in name:  # 处理权重
                if 'rnn' in name:  # RNN层权重
                    if 'hh' in name:  # 循环权重 (hidden-to-hidden)
                        nn.init.orthogonal_(param)  # 正交初始化
                    elif 'ih' in name:  # 输入权重 (input-to-hidden)
                        nn.init.xavier_uniform_(param)  # Xavier初始化（适用于tanh）
                elif 'fc' in name:  # 全连接层权重
                    if nonlinearity == "tanh" or nonlinearity == "sigmoid":
                        nn.init.xavier_uniform_(param)
                    else :
                        nn.init.kaiming_uniform_(param, nonlinearity=nonlinearity)  # He初始化（适用于ReLU）
            elif 'bias' in name:  # 处理偏置
                nn.init.zeros_(param)  # 偏置初始化为0
    # end function weight_init
    def detachHidden(self, hidden):
        if isinstance(hidden, tuple):
            for i in range(len(hidden)):
                hidden[i].detach_()
        else :
            hidden.detach_()
    # end function detachHidden


def grad_clipping(net, theta) -> None:
    if isinstance(net, nn.Module) : 
        params = [p for p in net.parameters() if p.requires_grad]
    else : 
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta : 
        for param in params : 
            param.grad[:] *= theta / norm
# end function grad_clipping

class RL_estimator(est.Estimator):
    def __init__(self, model:Continuous4, lr, lr_min, nnParams:dict, gamma=1.0, device='cpu', x0_hat=None, P0_hat=None) -> None:
        self.model = model
        dim_input = model.dim_state + model.dim_obs
        dim_output = fun.ds2do(model.dim_state)
        self.device = device
        self.gamma = gamma
        self.policy = ActorRNN(dim_input=dim_input, dim_output=dim_output, device=self.device, **nnParams).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=50, factor=0.75, min_lr=lr_min)
        self.lr = lr
        super().__init__(name="RL_estimator", x0_hat=x0_hat, P0_hat=P0_hat)
    # end function __init__
    def reset(self, x0_hat, P0_hat) -> None:
        self.x_hat = x0_hat
        self.P_hat = P0_hat
        if P0_hat is not None:
            # 保持后续计算的统一性给P矩阵转成tensor
            self.P_inv = torch.FloatTensor(fun.inv(P0_hat))
        self.hidden = None
        if hasattr(self, "y_pre") : del self.y_pre
        self.policy.train()
    # end function reset
    def estimate(self, y, Q, R, isEval:bool=False):
        #q的初始估计
        if not hasattr(self, "y_pre") :
            self.y_pre = y
            y_a = y[3:6]
            y_m = y[6:9]
            roll_accmag_0 = np.arctan2(y_a[1], y_a[2])
            pitch_accmag_0 = -np.arctan2(y_a[0], np.sqrt(y_a[1]**2 + y_a[2]**2))
            C_n_b__0 = self.model.rot(angle=0, axis="z") @ self.model.rot(angle=pitch_accmag_0, axis="Y") @ self.model.rot(angle=roll_accmag_0, axis="X")
            y_m_0_NEW = C_n_b__0.T @ y_m.reshape(-1,1)
            yaw_accmag_0 = -np.arctan2(y_m_0_NEW[1], y_m_0_NEW[0]).item()
            # q_m: Estimated-from-measurements quaternion (i.e., q from y_g (measured gyro output))
            q = self.model.RotMat2quat(self.model.rot(angle=yaw_accmag_0, axis="Z") @
                                       self.model.rot(angle=pitch_accmag_0, axis="Y") @
                                       self.model.rot(angle=roll_accmag_0, axis="X"))
            self.y_hat = self.model.h(self.x_hat)
            self.y_hat[0:3] = y[0:3]
            # self.y_seq.append(y)
            self.x_hat = np.hstack(( q, np.zeros((6,)) ))
            # self.x_hat = q
            # self.x0_bar_seq[0] = self.x_hat
            return 
        #正常情况
        ds = self.model.dim_state
        P_inv = self.P_inv.detach().squeeze().cpu().numpy().reshape((ds,-1)) ##+ np.eye(ds)*1e-4
        result = est.NLSFForC4(model=self.model, x0_bar=self.x_hat, y_seq=[self.y_pre, y], P_inv=P_inv,
                               Q=Q, R=R, x0=[self.x_hat], gamma=self.gamma)
        # self.status = result.status
        input = np.tile(np.hstack((self.x_hat, y)), (1,1,1))
        input = torch.from_numpy(input).float().to(self.device)
        if isEval:
            input.requires_grad_(False)
            self.policy.eval()
        self.P_inv, c, self.hidden = self.policy.forward(input, self.hidden) ## 这里不一定没计算梯度，可能还是要用到torch.no_grad()或者torch.inference_mode()
        self.x_hat = result.x[-ds: ]
        self.x_hat[0:4] /= np.linalg.norm(self.x_hat[0:4])
        self.y_hat = self.model.h(self.x_hat)
        self.y_hat[0:3] = y[0:3]
        self.P_hat = fun.inv(self.P_inv.detach().squeeze().cpu().numpy())
        return self.x_hat, self.P_inv, c
    # end function estimate
    def value(self, x, x_bar, P_inv, c=None):
        x = torch.Tensor((x - x_bar)).unsqueeze(0)
        Q = x @ P_inv.squeeze().reshape((self.model.dim_state,-1)) @ x.T
        if c is not None:Q += c.squeeze()
        return Q
    # end function value
    def save_network(self, baseName) -> None:
        torch.save(self.policy.state_dict(), baseName+'.mdl')
        torch.save(self.optimizer.state_dict(), baseName+".opt")
        torch.save(self.scheduler.state_dict(), baseName+".sch")
        print(f"save model at {baseName}")
    # end function save_network
    def load_network(self, baseName) -> None:
        self.policy.load_state_dict(torch.load(baseName+".mdl"))
        self.optimizer.load_state_dict(torch.load(baseName+".opt"))
        self.scheduler.load_state_dict(torch.load(baseName+".sch"))
    # end function load_network
    def train(self, x_batch_test:list, y_batch_test:list, trainParams:dict, estParams:dict) -> None:
        ds = self.model.dim_state
        x_batch, y_batch = getData(modelName=self.model.name, steps=trainParams["steps"], 
                             episodes=trainParams["episodes"], randSeed=trainParams["randSeed"])
        x0_hat = estParams["x0_hat"]
        P0_hat = estParams["P0_hat"]
        Q = estParams["Q"]
        R = estParams["R"]
        train_window = trainParams["train_window"]
        self.cov = trainParams['cov'] if isinstance(trainParams['cov'], np.ndarray) else trainParams['cov']*np.eye(ds)
        saveFile = fun.checkFilename(filename=trainParams["saveFile"], suffix='.mdl')
        self.noiseGen = fun.RandomGenerator(randomFun=np.random.multivariate_normal, rand_num=222)
        #region 初始化参数
        loss_seq = []
        min_loss = 0
        #endregion
        for i in range(len(y_batch)):
            self.reset(x0_hat, P0_hat)
            y_seq = y_batch[i]
            x_hat_seq = []
            y_hat_seq = []
            P_inv_seq = []
            y_list = []
            c = torch.Tensor([0])
            targetQ_list = []
            Q_list = []
            for t in range(len(y_seq)):
                #region 获取真实观测值
                y = y_seq[t]
                y_list.append(y)
                if len(y_list) > train_window+1 : del y_list[0]
                #endregion
                #region 求解窗口长度为1的非线性最小二乘，得到 x_next_hat
                if t == 0 :  # t=0求初始的q，不做其他操作
                    self.estimate(y, Q, R, isEval=False)
                    continue
                x_next_hat, P_inv_next, c_next = self.estimate(y, Q, R, isEval=False)
                x_hat_seq.append(x_next_hat)
                y_hat_seq.append(self.model.h(x_next_hat))
                P_inv_seq.append(P_inv_next.detach().squeeze().cpu().numpy().reshape((ds,-1)))
                #endregion
                #region 计算targetQ和Q
                if t >= 5: # 窗口大于指定长度开始训练（修改：窗口长度小于指定长度的数据不要）## 是不是等大于多一点的窗口再开始训练好一点？
                    for _ in range(trainParams["aver_num"]): # 为了实现函数拟合，取多个值计算arrival cost值
                        x_next_noise = x_next_hat + self.noiseGen.getRandom(mean=np.zeros((ds, )), cov=self.cov)
                        result = est.NLSFForC4(model=self.model, x0_bar=x_hat_seq[t-train_window], y_seq=y_list[:-1], P_inv=P_inv_seq[t-train_window], Q=Q, R=R,
                                               x0=x_hat_seq[t-train_window:-1], gamma=self.gamma, xend=x_next_noise)
                        # end if t(step)
                        min_fun_value = result.fun
                        targetQ = min_fun_value@min_fun_value + (y_list[-1] - self.model.h(x_next_noise))[3:9]@fun.inv(R[3:9,3:9])@(y_list[-1] - self.model.h(x_next_noise))[3:9] + c.item()
                        Qvalue = self.value(x=x_next_noise, x_bar=x_next_hat, P_inv=P_inv_next, c=c_next)
                        targetQ_list.append(targetQ)
                        Q_list.append(Qvalue)
                #endregion
                if len(Q_list) >= trainParams["seq_len"]*trainParams["aver_num"]: # 数据达到指定长度，做一次网络参数更新(修改：截断训练)
                    Q_list = torch.stack(Q_list).squeeze()
                    targetQ_list = torch.from_numpy(np.stack(targetQ_list)).float().squeeze().to(device=self.device)
                    loss = F.mse_loss(Q_list, targetQ_list)
                    self.optimizer.zero_grad(set_to_none=True) ## 
                    loss.backward()
                    # grad_clipping(self.policy, 10) ##
                    self.optimizer.step()
                    self.policy.detachHidden(self.hidden)
                    Q_list = []
                    targetQ_list = []
                    print(f"loss: {loss.item()}", flush=True)
                    loss_seq.append(loss.item())
                    self.scheduler.step(loss) # 学习率更新
                    new_lr = self.scheduler.get_last_lr()
                    if self.lr != new_lr :
                        print(f"learning rate updated: {self.lr} -> {new_lr}")
                        self.lr = new_lr
                    if loss.item() < min_loss or min_loss == 0: # 保存学习过程中loss最低的模型
                        min_loss = loss.item()
                        self.save_network(saveFile)
                c = c_next
            #region MSE指标计算并打印
            MSE, RMSE = fun.calMSE(x_batch=[x_batch[i][1:]], xhat_batch=[x_hat_seq])
            print(f"state MSE of batch {i}: MSE = {MSE}, RMSE = {RMSE}\n")
            # MSE, RMSE = fun.calMSE(x_batch=[y_batch[i]], xhat_batch=[y_hat_seq])
            # print(f"obs MSE of batch {i}: MSE = {MSE}, RMSE = {RMSE}\n", flush=True)
            #endregion
        # end for i(episode)
        #region 保存网络参数文件，以及打印相关参数，绘制损失曲线
        self.save_network(f'{saveFile}end')
        plotReward(loss_seq, filename="picture/train_loss.png")
        #endregion
        #region 测试
        self.policy.eval()
        simulate(agent=self, estParams=estParams, x_batch=x_batch_test, y_batch=y_batch_test, isPrint=True)
        #endregion
    # end function train
    # def initialize(self, x_batch_init:list, y_batch_init:list, estParams:dict):
    #     optimizer = Adam(self.policy.parameters(), lr=5e-4)
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.75, min_lr=1e-8, verbose=True)
    #     ekf = est.EKFForC4(model=self.model)
    #     loss_seq = []
    #     for _ in range(5):
    #         for i in range(len(y_batch_init)):
    #             x_hat_batch, P_hat_batch = simulate(agent=ekf, estParams=estParams, x_batch=x_batch_init[i:i+1], y_batch=y_batch_init[i:i+1])
    #             x_hat_seq = x_hat_batch[0]
    #             P_hat_seq = P_hat_batch[0]
    #             y_seq = y_batch_init[i]
    #             x_hat_seq = np.insert(x_hat_seq, 0, estParams["x0_hat"], axis=0)
    #             P_hat_seq = np.insert(P_hat_seq, 0, estParams["P0_hat"], axis=0)
    #             input_seq = []
    #             target_Pinv_seq = []
    #             # target_c_seq = []
    #             # c = 0
    #             for t in range(len(y_seq)):
    #                 input_seq.append(np.hstack((x_hat_seq[t], y_seq[t])))
    #                 Ptp1_inv = fun.inv(P_hat_seq[t+1])
    #                 target_Pinv_seq.append(Ptp1_inv)
    #                 # xt = x_hat_seq[t].reshape(-1,1)
    #                 # xtp1 = x_hat_seq[t+1].reshape(-1,1)
    #                 # dytp1 = (y_seq[t] - self.model.h(x=self.model.f(x=x_hat_seq[t]))).reshape(-1,1)
    #                 # Ft = self.model.F(x=x_hat_seq[t])
    #                 # Ht = self.model.H(x=self.model.f(x=x_hat_seq[t]))
    #                 # Rinv = fun.inv(estParams["R"])
    #                 # c = c + xt.T@Ft.T@Ptp1_inv@Ft@xt - xtp1.T@Ptp1_inv@xtp1 + dytp1.T@Rinv@dytp1 + 2*xt.T@Ft.T@Ht.T@Rinv@dytp1
    #                 # target_c_seq.append(c.reshape(-1))
    #             input_seq = torch.FloatTensor(np.stack(input_seq)).unsqueeze(0).to(self.device)
    #             Pinv_seq, c_seq, _ = self.policy.forward(input_seq, None)
    #             target_Pinv_seq = torch.FloatTensor(np.stack(target_Pinv_seq)).unsqueeze(0).to(self.device)
    #             target_c_seq = torch.zeros_like(c_seq, device=self.device)#torch.FloatTensor(np.stack(target_c_seq)).unsqueeze(0).to(self.device)#
    #             loss = F.mse_loss(Pinv_seq, target_Pinv_seq)+F.mse_loss(c_seq, target_c_seq)
    #             optimizer.zero_grad(set_to_none=True)
    #             loss.backward()
    #             grad_clipping(self.policy, 10)
    #             optimizer.step()
    #             scheduler.step(loss)
    #             if i % 10 == 0: print(f"loss: {loss.item()}")
    #             loss_seq.append(loss.item())
    #             if len(loss_seq) > 10: 
    #                 del loss_seq[0]
    #                 if fun.isConverge(loss_seq): break
    # # end function initialize
    def initialize(self, x_batch_init:list, y_batch_init:list, estParams:dict):
        optimizer = Adam(self.policy.parameters(), lr=1e-3)
        lr = 1e-2
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.75, min_lr=1e-6)
        agent = est.EKFForC4(model=self.model)
        loss_seq = []
        for i in range(len(y_batch_init)):
            x_hat_batch, P_hat_batch = simulate(agent=agent, estParams=estParams, x_batch=x_batch_init[i:i+1], y_batch=y_batch_init[i:i+1])
            # x_hat_seq = np.insert(x_hat_seq, 0, estParams["x0_hat"], axis=0)
            # P_hat_seq = np.insert(P_hat_seq, 0, estParams["P0_hat"], axis=0)
            for j in range(5):
                x_hat_seq = x_hat_batch[0]
                P_hat_seq = P_hat_batch[0]
                y_seq = y_batch_init[i]
                input_seq = []
                target_Pinv_seq = []
                target_c_seq = []
                c = 0
                hiddenState = None
                for t in range(len(y_seq)-1):
                    input_seq.append(np.hstack((x_hat_seq[t], y_seq[t])))
                    Ptp1_inv = fun.inv(P_hat_seq[t+1])
                    target_Pinv_seq.append(Ptp1_inv)
                    # xt = x_hat_seq[t].reshape(-1,1)
                    # xtp1 = x_hat_seq[t+1].reshape(-1,1)
                    # Ptp1 = P_hat_seq[t+1]
                    # dytp1 = (y_seq[t] - self.model.h(x=self.model.f(x=x_hat_seq[t]))).reshape(-1,1)
                    # # Ft = self.model.F(x=x_hat_seq[t])
                    # Ht = self.model.H(x=self.model.f(x=x_hat_seq[t], omega_pre=y_seq[t][0:3]))
                    # Rinv = fun.inv(estParams["R"])
                    # c = self.gamma * c + dytp1.T @ (Rinv-Rinv@Ht@Ptp1@Ht.T@Rinv) @ dytp1
                    # target_c_seq.append(c.reshape(-1))
                    if ((t+1) % 4999) == 0: # 每500时间步训练一次而不是整个轨迹训练一次
                        input_seq = torch.FloatTensor(np.stack(input_seq)).unsqueeze(0).to(self.device)
                        Pinv_seq, c_seq, hiddenState = self.policy.forward(input_seq, hiddenState)
                        target_Pinv_seq = torch.FloatTensor(np.stack(target_Pinv_seq)).unsqueeze(0).to(self.device)
                        target_c_seq = torch.zeros_like(c_seq, device=self.device)#torch.FloatTensor(np.stack(target_c_seq)).unsqueeze(0).to(self.device)#
                        loss = F.mse_loss(Pinv_seq[:,1:], target_Pinv_seq[:,1:])+F.mse_loss(c_seq, target_c_seq)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        # grad_clipping(self.policy, 10)
                        optimizer.step()
                        self.policy.detachHidden(hiddenState)
                        scheduler.step(loss)
                        new_lr = scheduler.get_last_lr()
                        if new_lr != lr : 
                            print(f"learning rate update: {lr} -> {new_lr}")
                            lr = new_lr
                        input_seq = []
                        target_Pinv_seq = []
                        target_c_seq = []
                        if t == 4998 and j == 0: print(f"loss: {loss.item()}")
                        loss_seq.append(loss.item())
                        if len(loss_seq) > 10: 
                            del loss_seq[0]
                            if fun.isConverge(loss_seq): break
    # end function initialize

def main():
    #region 测试的模型和参数
    model = getModel(modelName="Continuous4")
    steps = 5000
    episodes = 1
    randSeed = 10086
    initsteps = 5000 # 初始化网络用的
    initepisodes = 100 # 初始化网络用的
    initrandSeed = 0 # 初始化网络用的
    args = pm.parseParams()
    estParams = pm.getEstParams(modelName=model.name)
    trainParams = pm.getTrainParams(estorName="RL_estimator", cov=args.cov, gamma=args.gamma)
    nnParams = pm.getNNParams(netName="ActorRNN", hidden_layer=args.hidden_layer, dropout=args.dropout, num_rnn_layers=args.num_layer, type_activate=args.act_fun)
    #endregion
    #region 修改参数以便人工测试（自动测试时注释掉，否则参数无法自动变化）
    trainParams["lr"] = 5e-3
    trainParams["lr_min"] = 1e-6
    trainParams["gamma"] = 1.0
    args.hidden_layer = ([64,64,64], 64, [64]) # 这几个要同步修改
    nnParams["dim_fc1"] = [64,64,64] # 这几个要同步修改
    nnParams["dim_rnn_hidden"] = 64 # 这几个要同步修改
    nnParams["dim_fc2"] = [64] # 这几个要同步修改
    nnParams["dropout"] = 0.3
    nnParams["num_rnn_layers"] = 10
    nnParams["type_activate"] = "relu"
    nnParams["type_rnn"] = "lstm"
    #endregion
    # 定义估计器类以及获取测试数据
    agent = RL_estimator(model=model, lr=trainParams["lr"], lr_min=trainParams["lr_min"], nnParams=nnParams, 
                         gamma=trainParams["gamma"], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    x_batch_test, y_batch_test = getData(modelName=model.name, steps=steps, episodes=episodes, randSeed=randSeed)
    #region 策略网络初始化
    initNetName = f"net/c4{nnParams['type_rnn']}_{nnParams['type_activate']}_dropout{nnParams['dropout']}_layer{nnParams['num_rnn_layers']}_"\
                  f"{args.hidden_layer}_steps{initsteps}_epis{initepisodes}_randseed{initrandSeed}.mdl"
    if os.path.exists(initNetName):
        agent.policy.load_state_dict(torch.load(initNetName))
    # else :
        # agent.policy.load_state_dict(torch.load("net/c4lstm_relu_dropout0.3_layer10_([64, 64, 64], 64, [64])_steps5000_epis700_randseed22222.mdl"))
        x_batch_init, y_batch_init = getData(modelName=model.name, steps=initsteps, episodes=initepisodes, randSeed=initrandSeed)
        agent.initialize(x_batch_init=x_batch_init, y_batch_init=y_batch_init, estParams=estParams)
        torch.save(agent.policy.state_dict(), initNetName)
    #endregion
    # 加载模型
    # agent.load_network("net/RNN_net(54)end")
    #region 相关训练参数打印以及模型训练（训练结束后会自动进行测试）
    # logfile = fun.LogFile(fileName="output/log.txt", rename_option=True)
    # print("estimator params: ")
    # for key, value in estParams.items() : 
    #     print(f"{key}: {value}")
    # print("train params: ")
    # for key, value in trainParams.items() : 
    #     print(f"{key}: {value}")
    # print("network params: ")
    # for key, value in nnParams.items() : 
    #     print(f"{key}: {value}")
    # print("optimizer params: ")
    # print(f"betas: {agent.optimizer.param_groups[0]['betas']}")
    # print(f"weight_decay: {agent.optimizer.param_groups[0]['weight_decay']}\n", flush=True)
    # print("Before train, the test result: ") # 训练前测试一次估计性能
    # simulate(agent=agent, estParams=estParams, x_batch=x_batch_test, y_batch=y_batch_test, isPrint=True)
    # agent.train(x_batch_test=x_batch_test, y_batch_test=y_batch_test, trainParams=trainParams, estParams=estParams)
    # logfile.endLog()
    #endregion
    # 测试
    simulate(agent=agent, estParams=estParams, x_batch=x_batch_test, y_batch=y_batch_test, isPrint=True, isPlot=False)
# end function main

if __name__ == '__main__':
    main()
