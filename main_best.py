import argparse
from RL_estimator_best import *


def main() : 
    parser = argparse.ArgumentParser()
    # system parameters
    parser.add_argument("--state_dim", default=3, type=int, help="dimension of state variable x")
    parser.add_argument("--obs_dim", default=2, type=int, help="dimension of measurement y")
    parser.add_argument("--x0_mu", default=np.array([.2, .2, 8]), help="average of initial state distribution")
    parser.add_argument("--P0", default=.6*np.eye(3), help="covariance of initial state distribution")
    parser.add_argument("--x0_hat", default=np.array([.5, .5, .5]), help="estimation of initial state distribution average")
    parser.add_argument("--P0_hat", default=np.eye(3), help="estimation of initial state distribution covariance")
    parser.add_argument("--Q", default=np.array([[.001,0,0],[0,.001,0],[0,0,1.]]), help="covariance of process disturbance")
    parser.add_argument("--R", default=np.array([[1.,0], [0,1.]]), help="covariance of measurement noise")
    parser.add_argument("--MODEL_MISMATCH", default=False, type=bool, help="choose whether to apply model mismatch")

    # training parameters
    parser.add_argument("--max_episodes", default=1000, type=int, help="max train episodes")
    parser.add_argument("--max_train_steps", default=50, type=int, help="max simulation steps")
    parser.add_argument("--max_sim_steps", default=100, type=int, help="max simulation steps")
    parser.add_argument("--buffer_size", default=1e4, type=int, help="max size of replay buffer")
    parser.add_argument("--batch_size", default=50, type=int, help="number of samples for batch update")
    parser.add_argument("--warmup_size", default=200, type=int, help="decide when to start the training of the NN")
    parser.add_argument("--hidden_layer", default=[500,500,500,500,500,500,500,500,500,500], help="FC layers of NN")
    parser.add_argument("--gamma", default=.9, type=float, help="discount factor in value function")
    parser.add_argument("--lr_value", default=1e-3, type=float, help="learning rate of value function")
    parser.add_argument("--lr_policy", default=1e-2, type=float, help="learning rate of policy net")
    parser.add_argument("--lr_policy_delta", default=5e-4, type=float, help="learning rate reduction every time")
    parser.add_argument("--lr_policy_min", default=1e-5, type=float, help="minimum learning rate of policy net")
    parser.add_argument("--explore_Cov", default=np.array([[.001,0,0],[0,.001,0],[0,0,.001]]), help="the covariance of Guassian distribution added to predicted state")

    # file path
    parser.add_argument("--output_dir", default="output", type=str, help="path for files to save outputs such as model")
    parser.add_argument("--output_file", default="output/log.txt", type=str, help="file to save training messages")
    parser.add_argument("--model_file", default="model.bin", type=str, help="trained model")
    parser.add_argument("--modelend_file", default="modelend.bin", type=str, help="trained model")
    parser.add_argument("--model_test", default="modelend.bin", type=str, help="trained model")

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise = OUnoise(args.state_dim, sigma=args.explore_Cov)
    agent = RL_estimator(args.state_dim, args.obs_dim, noise=noise, hidden_layer=args.hidden_layer, STATUS='test')
    agent.policy.to(args.device)
    # init policy network ## 可以尝试没有初始样本的——没有初始样本的目前看来不太行
    x_hat_seq, y_seq, P_hat_seq = simulate(args, rand_num=22222, STATUS='init')
    x_hat_seq = np.insert(x_hat_seq, 0, args.x0_hat, axis=0)
    P_hat_seq = np.insert(P_hat_seq, 0, args.P0_hat, axis=0)
    replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
    for t in range(args.max_train_steps) : 
        input = np.hstack((x_hat_seq[t], y_seq[t], P2o(est.inv(P_hat_seq[t]), 0)))
        output = P2o(est.inv(P_hat_seq[t+1]), 0)
        replay_buffer.push_init((input, output))
    exp_list, _, _ = replay_buffer.sample(args.max_train_steps)
    input_batch = torch.FloatTensor([exp[0] for exp in exp_list])
    output_batch = torch.FloatTensor([exp[1] for exp in exp_list])
    agent.policy.update_weight(input_batch, output_batch, lr=1e-4)
    # save_path = os.path.join(args.output_dir, "model.bin")
    # torch.save(agent.policy.state_dict(), save_path)
    train(args, agent, replay_buffer)

    simulate(args, sim_num=50, rand_num=10086, STATUS='NLS-RLF')

if __name__ == '__main__' : 
    main()