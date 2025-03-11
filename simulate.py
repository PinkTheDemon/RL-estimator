import time
import matplotlib.pyplot as plt

import estimator as est
import functions as fc
from model import getModel
from gendata import getData
from plot import plotTrajectory
from params import getEstParams, getModelParams


def getSysFuns(model, modelErr):
    if modelErr:
        return model.f_real, model.h_real, model.F_real, model.H_real
    else :
        return model.f, model.h, model.F, model.H

# 对外接口
def simulate(agent:est.Estimator, estParams, x_batch, y_batch, isPrint=False, isPlot=False):
    # 识别参数
    episodes = len(x_batch)
    steps = len(x_batch[0])
    # 变量初始化
    xhat_batch = []
    yhat_batch = []
    Phat_batch = []
    execution_time = 0
    #region 状态估计
    for y_seq in y_batch:
        # 变量初始化
        xhat_seq = []
        yhat_seq = []
        Phat_seq = []
        seq_time = 0
        agent.reset(x0_hat=estParams["x0_hat"], P0_hat=estParams["P0_hat"])
        # 单条轨迹估计
        for y in y_seq:
            timeStart = time.time()
            agent.estimate(y=y, Q=estParams["Q"], R=estParams["R"])
            timeEnd = time.time()
            seq_time += timeEnd - timeStart
            xhat_seq.append(agent.x_hat)
            yhat_seq.append(agent.y_hat)
            Phat_seq.append(agent.P_hat)
        execution_time += 1000 * seq_time / steps / episodes
        xhat_batch.append(xhat_seq)
        yhat_batch.append(yhat_seq)
        Phat_batch.append(Phat_seq)
        agent.reset(x0_hat=estParams["x0_hat"], P0_hat=estParams["P0_hat"])
    #endregion 状态估计
    #region 保存数据temp
    # import pickle
    # if agent.name == "EKF":
    #     fileName = f"data/EKF"
    #     with open(file=fileName+".bin", mode="wb") as f:
    #         pickle.dump(xhat_batch, f)
    # elif agent.name == "MHE":
    #     fileName = f"data/MHE_window{estParams['window']}"
    #     with open(file=fileName+".bin", mode="wb") as f:
    #         pickle.dump(xhat_batch, f)
    # elif agent.name == "RL_estimator":
    #     fileName = f"data/RA-MHE"
    #     with open(file=fileName+".bin", mode="wb") as f:
    #         pickle.dump(xhat_batch, f)
    #endregion 保存数据
    # 打印
    if isPrint:
        # 计算性能指标
        MSE_x, RMSE_x = fc.calMSE(x_batch=[x_seq[:] for x_seq in x_batch], xhat_batch=[x_seq[:] for x_seq in xhat_batch])
        # MSE_y, RMSE_y = fc.calMSE(x_batch=y_batch, xhat_batch=yhat_batch)
        print(f"state MSE of {agent.name}: {MSE_x}, RMSE: {RMSE_x}")
        # print(f"observation MSE of {agent.name}: {MSE_y}, RMSE: {RMSE_y}")
        print(f"average cpu time of {agent.name}: {execution_time} ms")
    else :
        return xhat_batch, Phat_batch
    # 绘图
    if isPlot:
        plotTrajectory(x_seq=x_batch[-1][1:], x_hat_seq=xhat_seq[1:], STATUS=agent.name)
        plotTrajectory(x_seq=y_seq, x_hat_seq=y_seq, STATUS=agent.name)
        plt.show()

if __name__ == "__main__" : 
    # 选择模型、仿真步数以及轨迹条数
    model = getModel(modelName="Continuous4")
    # model.sampleTime=0.01
    # model.N_sample=20000 # 要想改生成的数据，必须去模型定义那里改
    steps = 5000
    episodes = 10
    randSeed = 10086
    modelErr = False
    isPrint = True
    isPlot = False
    # 选择执行测试的方法
    test_options = ["EKF", "EKF-MHE"] # , "UKF", "PF", "UKF-MHE", "FIE", "IEKF"
    # 生成数据以及参数
    x_batch, y_batch = getData(modelName=model.name, steps=steps, episodes=episodes, randSeed=randSeed)
    modelParams = getModelParams(modelName=model.name)
    estParams = getEstParams(modelName=model.name, modelErr=modelErr)
    # 重定向系统输出以及打印仿真信息
    logfile = fc.LogFile("output/test_results.txt", rename_option=False)
    print("model params:")
    for key, val in modelParams.items():
        print(f"{key}: {val}")
    print("estimator params:")
    for key, val in estParams.items():
        print(f"{key}: {val}")
    print("********************")
    logfile.flush()
    #endregion
    #region 测试
    for status in test_options:
        print(f"{status.upper()}:", flush=True)
        if status.upper() == "EKF-MHE":
            for i in range(1,6):
                estParams["window"] = i
                print("window length:", estParams["window"])
                # logfile.flush()
                # 生成EKF-MHE类
                if model.name == "Continuous2" :
                    agent = est.MHEForQuat(model=model, window=estParams["window"])
                elif model.name == "Continuous4" :
                    agent = est.MHEForC4(model=model, window=estParams["window"])
                else :
                    f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
                    agent = est.MHE(f_fn=f, h_fn=h, F_fn=F, H_fn=H, window=estParams["window"])
                simulate(agent=agent, estParams=estParams, x_batch=x_batch, y_batch=y_batch, isPrint=isPrint, isPlot=isPlot)
                print("********************")
        elif status.upper() == "EKF":
            # 生成EKF类
            if model.name == "Continuous2" :
                agent = est.EKFForQuat(model=model)
            elif model.name == "Continuous4" :
                agent = est.EKFForC4(model=model)
            else :
                f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
                agent = est.EKF(f_fn=f, h_fn=h, F_fn=F, H_fn=H)
            simulate(agent=agent, estParams=estParams, x_batch=x_batch, y_batch=y_batch, isPrint=isPrint, isPlot=isPlot)
            print("********************")
        elif status.upper() == "UKF":
            # 生成UKF类
            if model.name == "Continuous2" :
                agent = est.EKFForQuat(model=model)
            elif model.name == "Continuous4" :
                agent = est.EKFForC4(model=model)
            else :
                f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
                agent = est.UKF(dim_x=model.dim_state, dim_z=model.dim_obs, dt=None, hx=h, fx=f)
            simulate(agent=agent, estParams=estParams, x_batch=x_batch, y_batch=y_batch, isPrint=isPrint, isPlot=isPlot)
            print("********************")
        elif status.upper() == "PF":
            # 生成PF类
            if model.name == "Continuous2" :
                agent = est.EKFForQuat(model=model)
            elif model.name == "Continuous4" :
                agent = est.EKFForC4(model=model)
            else :
                f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
                agent = est.ParticleFilter(state_dim=model.dim_state, obs_dim=model.dim_obs, num_particles=1000, fx=f, hx=h)
            simulate(agent=agent, estParams=estParams, x_batch=x_batch, y_batch=y_batch, isPrint=isPrint, isPlot=isPlot)
            print("********************")
        logfile.flush()
    logfile.endLog()
    #endregion