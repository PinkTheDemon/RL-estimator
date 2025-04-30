import numpy as np
import pickle
import os

from functions import LogFile
from model import getModel
from params import getModelParams

def generate_data(model, modelParam, steps, randSeed):
    np.random.seed(randSeed)
    # 生成初始状态
    if modelParam["P0"] is None:
        initial_state = modelParam["x0_mu"]
    else :
        initial_state = modelParam["x0_mu"] + np.random.multivariate_normal(np.zeros_like(modelParam["x0_mu"]), modelParam["P0"])
    # 生成噪声序列
    if "Q" not in modelParam or modelParam["Q"] is None:
        disturb_list = [None for _ in range(steps)]
    else : 
        if "disturbMu" not in modelParam or modelParam["disturbMu"] is None:
            modelParam["disturbMu"] = np.zeros(model.dim_state)
        disturb_list = np.random.multivariate_normal(modelParam["disturbMu"], modelParam["Q"], steps)
    if "R" not in modelParam or modelParam["R"] is None:
        noise_list = [None for _ in range(steps)]
    else : 
        if "noiseMu" not in modelParam or modelParam["noiseMu"] is None:
            modelParam["noiseMu"] = np.zeros(model.dim_obs)
        noise_list = np.random.multivariate_normal(modelParam["noiseMu"], modelParam["R"], steps)
    # 生成状态序列和观测序列
    t_seq = range(steps)
    x_seq = []
    y_seq = []
    x = initial_state
    for t in t_seq : # 真实轨迹
        x_next, y_next = model.step(x, disturb=disturb_list[t], noise=noise_list[t], isReal=True)
        x = x_next
        x_seq.append(x_next)
        y_seq.append(y_next)
    # 额外的外部扰动（未知且不希望有的）
    if hasattr(model, "ext_y") :
        ext_y = model.ext_y()
        y_seq = [y+e_y for y,e_y in zip(y_seq, ext_y)]
    return x_seq, y_seq

# 生成并保存数据轨迹
def generate_trajectories(modelName, steps, episodes, randSeed, isSave=True):
    #region 判断数据是否已经存在
    fileName = f"data/{modelName}_steps{steps}_episodes{episodes}_randomSeed{randSeed}"
    if os.path.isfile(fileName+".bin") :
        print("Data file already exist, input \"y\" to regenerate : ", end="")
        char = input()
        if char != "y" :
            return 
    #endregion
    trajs = {
        "x_batch": [],
        "y_batch": [],
    }
    model = getModel(modelName=modelName)
    modelParam = getModelParams(modelName=modelName)
    for num in range(episodes):
        if hasattr(model, "t") : model.t = 0
        x_seq, y_seq = generate_data(model=model, modelParam=modelParam, steps=steps, randSeed=randSeed+num)
        trajs["x_batch"].append(x_seq)
        trajs["y_batch"].append(y_seq)

    if isSave:
        print("Data saving, please wait...", flush=True)
        #region save trajectories and relative information
        with open(file=fileName+".bin", mode="wb") as f:
            pickle.dump(trajs, f)
        # relative information
        log = LogFile(fileName=fileName+".txt")
        for key, val in modelParam.items():
            print(f"{key}: {val}")
        log.endLog()
        #endregion
        print("successfully generate data")
    else :
        return trajs

# 根据相关信息获取数据，对外接口
def getData(modelName, steps, episodes, randSeed):
    isReverse = False
    if "Reverse" in modelName:
        isReverse = True
        modelName = "Dynamics" + modelName[7:]
    fileName = f"data/{modelName}_steps{steps}_episodes{episodes}_randomSeed{randSeed}.bin"
    if os.path.isfile(fileName) :
        with open(file=fileName, mode="rb") as f:
            trajs = pickle.load(f)
    else :
        trajs = generate_trajectories(modelName=modelName, steps=steps, episodes=episodes, randSeed=randSeed, isSave=False)
    x_batch = trajs["x_batch"]
    y_batch = trajs["y_batch"]
    if isReverse :
        x_batch = [x_seq[::-1] for x_seq in x_batch]
        y_batch = [y_seq[::-1] for y_seq in y_batch]
    return x_batch, y_batch

if __name__ == "__main__":
    generate_trajectories(modelName="Continuous1", steps=100, episodes=50, randSeed=10086)
