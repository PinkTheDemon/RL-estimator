import argparse
import numpy as np
from functions import block_diag

# 解析输入参数
def parseParams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", type=str, default="1e-2") # 希望它长什么样就输入什么就行
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hidden_layer", default=([256], 32, [256]))
    parser.add_argument("--dropout", type=float, default=0, help="no effect when num_layer=1")
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--act_fun", type=str, default="relu")
    args = parser.parse_args()
    args.cov = eval(args.cov)
    if isinstance(args.hidden_layer, str) : args.hidden_layer = eval(args.hidden_layer)
    return args

# 仅用于生成数据
def getModelParams(modelName):
    if modelName == "Discrete1":
        modelParams = {
            "x0_mu": np.array([10, 10]),
            "P0": np.diag((1., 1.)),
            "Q": np.diag((1e0, 1e0)),
            "R": np.array([[1e0]]),
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
    elif modelName == "Continuous2":
        modelParams = {
            "x0_mu": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "P0": None,
            "Q": None,
            "disturbMu": None,
            "R": block_diag(( 0.000036*np.eye(3), 0.002304*np.eye(3), 4*np.eye(3) )),
            "noiseMu": np.array((0, 0, 0, 0.07, 0.033, -0.044, -0.019, 0.013, -0.006)),
        }
    elif modelName == "Continuous4":
        modelParams = {
            "x0_mu": np.array([0.94563839, -0.03449911,  0.21051564,  0.24548119, 0.07, 0.033, -0.044, -0.019, 0.013, -0.006]),#
            "P0": None,
            "Q": None,
            "disturbMu": None,
            "R": block_diag(( 0.000036*np.eye(3), 0.002304*np.eye(3), 4*np.eye(3) )), # None, #
            "noiseMu": None,
        }
    return modelParams

# 状态估计的初始参数
def getEstParams(modelName, **kwargs):
    if modelName == "Discrete1":
        estParams = {
            "x0_hat": np.array([0, 0]),
            "P0_hat": np.diag((10., 10.)),
            "Q": np.array([[1e-2,0],[0,1e-2]]),
            "R": np.array([[0.01]]),
        }
    elif modelName == "Continuous1":
        estParams = {
            "x0_hat": np.array([1, 1, 1]),
            "P0_hat": np.diag((10., 10., 10.)),
            "Q": np.diag((1e-2, 1e-2, 1e-2)),
            "R": np.diag((1e-2, 1e-2)),
        }
    elif modelName == "Continuous2":
        estParams = {
            "x0_hat": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), #
            "P0_hat": block_diag(( 4e-2*np.eye(3), 1e-6*np.eye(3), 4e-4*np.eye(1) )),
            "Q": block_diag(( 9e-6*np.eye(3), 1e-6*np.eye(3), 1e-6*np.eye(3) )),
            "R": block_diag(( 0.000036*np.eye(3), 0.002304*np.eye(3), 4*np.eye(3) )),
        }
    elif modelName == "Continuous4":
        estParams = {
            "x0_hat": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float64'),#
            "P0_hat": block_diag(( 1e-4*np.eye(4), 1e-4*np.eye(3), 1e-4*np.eye(3) )),#
            "Q": block_diag(( 9e-6*np.eye(4), 1e-6*np.eye(3), 1e-6*np.eye(3) )),#
            "R": block_diag(( 0.000036*np.eye(3), 0.002304*np.eye(3), 4*np.eye(3) )),#
        }
    estParams |= kwargs
    return estParams

def getTrainParams(estorName, **kwargs):
    if estorName == "RL_estimator":
        trainParams = {
            "steps": 5000,
            "episodes": 10,
            "randSeed": 0,
            "lr": 5e-3,
            "lr_min": 1e-5,
            "train_window": 4,
            "aver_num": 15,
            "seq_len": 15,
            "saveFile": "net/RNN_net", #base name without suffix
        }
    trainParams |= kwargs
    return trainParams

def getNNParams(netName, hidden_layer, **kwargs):
    if netName == "ActorRNN":
        nnParams = {
            "dim_fc1": hidden_layer[0],
            "dim_fc2": hidden_layer[2],
            "type_activate": 'relu',
            "type_rnn": 'lstm',
            "dim_rnn_hidden": hidden_layer[1],
            "num_rnn_layers": 1,
            "rand_seed": 111,
        }
    nnParams |= kwargs
    return nnParams