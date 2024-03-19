import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import estimator as est
from functions import inv

def simulate(model, args, agent=None, sim_num=1, rand_seed=1111, STATUS='EKF', plot_flag=False) : 
    # 常用变量读取
    ds = model.dim_state
    # ---------
    # 初始化评价指标变量
    MSE_avg = 0
    RMSE_avg = 0
    execution_time = 0
    error = np.zeros((args.max_sim_steps,ds)) # 可能需要绘制均值和方差大小，保存的话，可能batch×steps×ds
    # ---------
    # 估计器模型初始化
    if STATUS == 'PF' : 
        pf = est.Particle_Filter(ds, model.dim_obs, int(1e4), model.f, model.h, model.x0_mu, model.P0)
    # ----------
    # 生成多条测试轨迹
    for i in range(sim_num) : 
        # 设置随机数种子
        np.random.seed(rand_seed+i)
        # ----------
        # 随机生成一条测试轨迹
        t_seq = range(args.max_sim_steps)
        x_seq, y_seq = model.generate_data(args.max_sim_steps, is_mismatch=args.MODEL_MISMATCH, rand_seed=rand_seed+i)
        # ----------
        # 记录一条轨迹状态估计所需的cpu时间开始点
        start_time = time.process_time()
        # ----------
        # 状态估计，不同的估计器
        x_hat_seq = []
        P_hat_seq = []
        if STATUS.upper() == 'FIE' : # 全信息估计
            x_hat = args.x0_hat
            P_hat = args.P0_hat
            initial_x = [args.x0_hat]
            for t in t_seq : 
                result = est.NLSF_uniform(args.P0_hat, y_seq=y_seq[:t+1], Q=model.Q, R=model.R, x0=initial_x, mode="quadratic", x0_bar=args.x0_hat)
                x_hat_seq.append(result[-ds:])
                initial_x = list(result.reshape(-1,3))[1:]
            # end for t(step)
        elif 'MHE' not in STATUS.upper() : # 单步状态估计，EKF、UKF、PF
            x_hat = args.x0_hat
            P_hat = args.P0_hat
            for t in t_seq : 
                if STATUS.upper() == 'EKF' or STATUS.upper()=='init': 
                    x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_seq[t], model.Q, model.R)
                elif STATUS.upper() == 'UKF' : 
                    x_next_hat, P_next_hat = est.UKF(x_hat, P_hat, y_seq[t], model.Q, model.R)
                elif STATUS.upper() == 'PF' : 
                    pf.predict(model.Q)
                    pf.update(y_seq[t], model.R)
                    x_next_hat, P_next_hat = pf.estimate()
                # end if STATUS
                x_hat_seq.append(x_next_hat)
                P_hat_seq.append(P_next_hat)
                x_hat = x_next_hat
                P_hat = P_next_hat
            # end for t(step)
        elif 'RLF' in STATUS.upper() and 'MHE' in STATUS.upper() : # RL更新arrival cost的MHE
            hidden = None
            initial_x = [args.x0_hat]
            y_list = []
            for t in t_seq : 
                # 获取真实观测值
                y_list.append(y_seq[t])
                if len(y_list) > args.train_window : del y_list[0]
                # ----------
                if t < args.train_window -1 : # 窗口未满
                    # 求解非线性最小二乘，得到 x_next_hat
                    result, _ = est.NLSF_uniform(args.P0_hat, y_seq=y_list, Q=model.Q, R=model.R, mode="quadratic", x0=initial_x, x0_bar=args.x0_hat)
                    x_next_hat = result[-ds: ]
                    x_hat_seq.append(x_next_hat)
                    initial_x = np.vsplit(result[:].reshape(-1,ds), 1)
                    # ----------
                elif t == args.train_window -1 : # 窗口刚满，arrival cost从0到1
                    # 求解非线性最小二乘，得到 x_next_hat
                    result, _ = est.NLSF_uniform(args.P0_hat, y_seq=y_list, Q=model.Q, R=model.R, mode="quadratic", x0=initial_x, x0_bar=args.x0_hat)
                    x_next_hat = result[-ds: ]
                    x_hat_seq.append(x_next_hat)
                    initial_x = np.vsplit(result[ds: ].reshape(-1,ds), 1)
                    # ----------
                    # 基于 x_k|k 和 y_k+1 得到 L0_NLSF
                    inputTensor = torch.FloatTensor(np.hstack((args.x0_hat, y_seq[t-args.train_window+1])), device=agent.device).reshape(1,1,-1)
                    L0_NLSF, hidden = agent.policy.forward(input_seq=inputTensor, hidden=hidden, batch_size=1)
                    L0_NLSF = L0_NLSF.squeeze().detach().cpu().numpy()
                    # ----------
                else : # 窗口已满，arrival cost从k到k+1
                    # 求解非线性最小二乘，得到 x_next_hat
                    result, _ = est.NLSF_uniform(L0_NLSF, y_seq=y_list, Q=model.Q, R=model.R, mode="sumofsquares", x0=initial_x)
                    x_next_hat = result[-ds: ]
                    x_hat_seq.append(x_next_hat)
                    initial_x = np.vsplit(result[ds: ].reshape(-1,ds), 1)
                    # ----------
                    # 基于 x_k|k 和 y_k+1 得到 L0_NLSF
                    inputTensor = torch.FloatTensor(np.hstack((x_hat_seq[t-args.train_window], y_seq[t-args.train_window+1])), device=agent.device).reshape(1,1,-1)
                    L0_NLSF, hidden = agent.policy.forward(input_seq=inputTensor, hidden=hidden, batch_size=1)
                    L0_NLSF = L0_NLSF.squeeze().detach().cpu().numpy()
                    # ----------
                # end if t(step)
            # end for t(step)
        elif 'MHE' in STATUS.upper() : # 传统方法（EKF、UKF）更新arrival cost的MHE
            x0_NLSF = args.x0_hat
            P0_NLSF = args.P0_hat
            initial_x = [args.x0_hat]
            y_list = []
            for t in t_seq : 
                # 获取真实观测值
                y_list.append(y_seq[t])
                if len(y_list) > args.window : del y_list[0]
                # ----------
                # 求解非线性最小二乘，得到 x_next_hat
                result = est.NLSF_uniform(P0_NLSF, y_seq=y_list, Q=model.Q, R=model.R, x0=initial_x, mode="quadratic", x0_bar=x0_NLSF)
                x_next_hat = result[-ds: ]
                x_hat_seq.append(x_next_hat)
                # ----------
                if t < args.window - 1 : # 窗口未满，x0_NLSF 和 P0_NLSF无需更新
                    initial_x = list(result.reshape(-1,3))
                else : # 窗口已满，用不同的方法更新x0_NLSF 和 P0_NLSF
                    initial_x = list(result.reshape(-1,3))[1:]
                    if 'EKF' in STATUS : 
                        F = model.F(model.f(x0_NLSF))
                        H = model.H(model.f(x0_NLSF))
                        P0_NLSF = F@P0_NLSF@F.T + model.Q - F@P0_NLSF@H.T@ inv(H@P0_NLSF@H.T + model.R) @ H@P0_NLSF@F.T
                    if 'UKF' in STATUS : ## 尚未修改正确
                        _, P0_NLSF = est.UKF(x0_NLSF, P0_NLSF, y_seq[t-args.window+1], model.Q, model.R)
                    x0_NLSF = x_hat_seq[t-args.window+1] # x_hat_k|k
                # end if t(step)
                P_hat_seq.append(P0_NLSF)
            #end for t(step)
        # end if STATUS
        # 单步估计所需平均cpu时间，单位ms
        end_time = time.process_time()
        execution_time += 1000 * (end_time - start_time) / args.max_sim_steps / sim_num
        # ----------
        # 计算MSE指标
        x_seq = np.array(x_seq)
        x_hat_seq = np.array(x_hat_seq)
        error += np.abs(x_seq - x_hat_seq) / sim_num
        MSE = np.square(x_seq - x_hat_seq).sum(0) / args.max_sim_steps
        RMSE = np.sqrt(np.mean(MSE))
        MSE_avg += MSE / sim_num
        RMSE_avg += RMSE / sim_num
        # ----------
    # end for i(sim_num)
    # 结果输出或绘制
    if STATUS.upper() == 'INIT' : 
        return x_hat_seq, y_seq, P_hat_seq
    else : 
        # 打印MSE和时间指标
        print(f"MSE of {STATUS.upper()}: {MSE_avg}, RMSE: {RMSE_avg}")
        print(f"average cpu time of {STATUS.upper()}: {execution_time} ms")
        # ----------
        # 绘图
        if plot_flag : 
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False 
            plt.rcParams['font.size'] = 28
            fig, axs = plt.subplots(ds,1)
            for i in range(ds) : 
                axs[i].plot(t_seq, x_seq[:,i], label='x_real', color='blue')
                axs[i].plot(t_seq, x_hat_seq[:,i], label='x_hat', color='red')
                axs[i].set_xlim(0, args.max_sim_steps)
                axs[i].set_ylabel(f'x{i+1}')
            axs[0].set_title(f'{STATUS}')
            # axs[0].set_title('状态估计效果图')
            axs[0].legend()
            # axs[-1].set_xlabel('时间步')

            fig, ax = plt.subplots()
            color = ['b','g','r','c','m','y','k']
            for i in range(ds) : 
                ax.plot(t_seq, error[:,i], label=f'x{i+1}', color=color[i], linestyle='--')
                ax.plot(t_seq, np.average(error[:,i])*np.ones_like(t_seq), color=color[i])
            ax.set_xlim(0, args.max_sim_steps)
            # ax.set_xlabel('时间步')
            # ax.set_ylabel('绝对值误差')
            ax.set_title(f'MSE = {MSE}')
            # ax.set_title('绝对值误差')
            ax.legend()
            plt.show()
        # ----------
    # end if STATUS(== INIT)
# end function simulate