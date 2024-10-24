import argparse
import numpy as np

# 解析输入参数
def parseParams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", type=str, default="1e-4") # 希望它长什么样就输入什么就行
    parser.add_argument("--goodInit", type=bool, default=False) # ""或不指定表示False，其他都是True
    parser.add_argument("--gamma", type=float, default=0.4)
    args = parser.parse_args()
    return args

# 仅用于生成数据
def getModelParams(modelName):
    if modelName == "Discrete1":
        modelParams = {
            "x0_mu": np.array([10, 10]),
            "P0": np.diag((1., 1.)),
            "Q": np.diag((1e0, 1e0)),
            "R": np.array([[1e-2]]),
            "disturbMu": None,
            "noiseMu": None,
        }
    elif modelName == "Continuous1":
        modelParams = {
            "x0_mu": np.array([10, 10, 10]),
            "P0": [[1,0,0],
                   [0,1,0],
                   [0,0,1]], # 只能在modelParam中的P0用list格式，别的只能用ndarray
            "Q": np.diag((1e-2, 1e-2, 1e-2)),
            "R": np.diag((1e-2, 1e-2)),
            "disturbMu": None,
            "noiseMu": None,
        }
    return modelParams

# 状态估计的初始参数
def getEstParams(modelName, **args):
    if modelName == "Discrete1":
        estParams = {
            "x0_hat": np.array([0, 0]),
            "P0_hat": np.diag((10., 10.)),
            "Q": np.array([[1,0],[1,0]]),
            "R": np.array([[0.1]]),
        }
    elif modelName == "Continuous1":
        estParams = {
            "x0_hat": np.array([10, 10, 10]),
            "P0_hat": np.diag((1., 1., 1.)),
            "Q": np.diag((1e-2, 1e-2, 1e-2)),
            "R": np.diag((1e-2, 1e-2)),
        }
    estParams |= args
    return estParams

def getTrainParams(estorName, **args):
    if estorName == "RL_Observer":
        trainParams = {
            "trainEpis": 100,
            "steps": 30,
            "episodes": 50,
            "randSeed": 0,
        }
    trainParams |= args
    return trainParams