import argparse
import numpy as np

def def_param() : 
    parser = argparse.ArgumentParser()
    '''system parameters'''
    parser.add_argument("--state_dim", default=3, type=int, help="dimension of state variable x")
    parser.add_argument("--obs_dim", default=2, type=int, help="dimension of measurement y")
    parser.add_argument("--dim_output", default=6, type=int, help="dimension of phi(x)")
    parser.add_argument("--x0_mu", default=np.array([10, 10, 10]), help="average of initial state distribution")
    parser.add_argument("--P0", default=.1*np.eye(3), help="covariance of initial state distribution")
    parser.add_argument("--x0_hat", default=np.array([0., 0., 1.]), help="estimation of initial state distribution average")
    parser.add_argument("--P0_hat", default=np.eye(3), help="estimation of initial state distribution covariance")
    parser.add_argument("--Q", default=np.array([[.001,0,0],[0,.001,0],[0,0,1.]]), help="covariance of process disturbance")
    parser.add_argument("--R", default=np.array([[1.,0], [0,1.]]), help="covariance of measurement noise")
    parser.add_argument("--MODEL_MISMATCH", default=False, type=bool, help="choose whether to apply model mismatch")
    parser.add_argument("--window", default=1, type=bool, help="MHE window length")

    '''training parameters'''
    parser.add_argument("--max_episodes", default=1000, type=int, help="max train episodes")
    parser.add_argument("--max_train_steps", default=100, type=int, help="max simulation steps")
    parser.add_argument("--max_sim_steps", default=100, type=int, help="max simulation steps")
    parser.add_argument("--buffer_size", default=1e5, type=int, help="max size of replay buffer")
    parser.add_argument("--batch_size", default=50, type=int, help="number of samples for batch update")
    parser.add_argument("--warmup_size", default=200, type=int, help="decide when to start the training of the NN")
    parser.add_argument("--hidden_layer", default=[500,500,500,500,500,500,500,500,500,500], help="FC layers of NN")
    parser.add_argument("--gamma", default=1., type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-2, type=float, help="learning rate of policy net")
    parser.add_argument("--lr_policy_min", default=1e-5, type=float, help="minimum learning rate of policy net")
    parser.add_argument("--explore_Cov", default=np.array([[.001,0,0],[0,.001,0],[0,0,.001]]), help="the covariance of Guassian distribution added to predicted state")

    '''file path'''
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log.txt", type=str, help="file to save training messages")
    parser.add_argument("--model_file", default="model.bin", type=str, help="trained model")
    parser.add_argument("--modelend_file", default="modelend.bin", type=str, help="trained model")
    parser.add_argument("--model_test", default="modelend.bin", type=str, help="trained model")

    args = parser.parse_args()
    return args
