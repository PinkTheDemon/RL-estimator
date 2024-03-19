import argparse
import torch
import numpy as np

def def_param2() : 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    '''simulate parameters'''
    parser.add_argument("--STATUS", default="EKF", type=str, help="simulate method")
    parser.add_argument("--x0_hat", default=np.array([0., 0., 1.]), help="estimation of initial state distribution average")
    parser.add_argument("--P0_hat", default=np.eye(3), help="estimation of initial state distribution covariance")
    parser.add_argument("--MODEL_MISMATCH", default=False, type=bool, help="choose whether to apply model mismatch")
    parser.add_argument("--window", default=1, type=int, help="MHE window length")
    '''training parameters'''
    parser.add_argument("--max_episodes", default=500, type=int, help="max train episodes")
    parser.add_argument("--max_train_steps", default=100, type=int, help="max simulation steps")
    parser.add_argument("--max_sim_steps", default=100, type=int, help="max simulation steps")
    parser.add_argument("--buffer_size", default=1e5, type=int, help="max size of replay buffer")
    parser.add_argument("--batch_size", default=50, type=int, help="number of samples for batch update")
    parser.add_argument("--warmup_size", default=200, type=int, help="decide when to start the training of the NN")
    parser.add_argument("--gamma", default=1., type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-3, type=float, help="learning rate of policy net")
    parser.add_argument("--lr_policy_min", default=1e-7, type=float, help="minimum learning rate of policy net")
    parser.add_argument("--hidden_layer", default=([256], 32, [256]), help="hidden layers of NN")
    parser.add_argument("--explore_Cov", default=np.array([[.001,0,0],[0,.001,0],[0,0,.001]]), help="the covariance of Guassian distribution added to predicted state")
    parser.add_argument("--train_window", default=1, type=int, help="MHE window length")
    '''file path'''
    parser.add_argument("--rename_option", default=False, type=bool, help="whether to rename the output file")
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log.txt", type=str, help="file to save training messages")
    parser.add_argument("--model_file", default="model.bin", type=str, help="trained model")
    parser.add_argument("--modelend_file", default="modelend.bin", type=str, help="trained model")
    parser.add_argument("--model_test", default="modelend.bin", type=str, help="trained model")
    args = parser.parse_args()
    if isinstance(args.hidden_layer, str) : args.hidden_layer = eval(args.hidden_layer)
    model_paras_dict = {
        "dim_state": 3,
        "dim_obs": 2,
        "x0_mu": np.array([10, 10, 10]),
        "P0": .1*np.eye(3),
        "Q": np.array([[.001,0,0],[0,.001,0],[0,0,1.]]),
        "R": np.array([[1.,0], [0,1.]]),
    }
    estimator_dict = {
        "dim_state": 3,
        "dim_obs": 2,
        "lr": args.lr_policy,
        "lr_min": args.lr_policy_min,
        "device": device,
        "rnn_params_dict": {
            "dim_fc1": args.hidden_layer[0],
            "dim_fc2": args.hidden_layer[2],
            "type_activate": 'tanh',
            "type_rnn": 'gru',
            "dim_rnn_hidden": args.hidden_layer[1],
            "num_rnn_layers": 1,
            "batch_first": True,
            "rand_seed": 111,
            "device": device,
        },
    }
    return args, model_paras_dict, estimator_dict
