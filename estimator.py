import numpy as np
from scipy.optimize import least_squares
import dynamics as dyn

# inverse of matrix M
def inv(M) : 
    if M.shape == (1,1) : 
        return 1/M 
    else :
        return np.linalg.inv(M)
    
# transfer matrix list to one block-diag matrix
def block_diag(matrix_list) : 
    bd_M = matrix_list[0]
    for M in matrix_list[1:] : 
        bd_M = np.block([[bd_M, np.zeros((bd_M.shape[0], M.shape[1]))],
                         [np.zeros((M.shape[0], bd_M.shape[1])), M]])
    return bd_M

# Extended Kalman Filter
def EKF (state, P, obs_next, Q=np.array([[0.0001,0],[0,1]]), R=np.array([[1]])) : 
    # linearization system matrix
    F = np.array([[0.99,0.2],[-0.1,0.5*(1-state[1]**2)/(1+state[1]**2)**2]])
    H = np.array([[1,-3]])
    
    # predict
    P_pre = F @ P @ F.T + Q
    state_pre, obs_pre = dyn.step(state)
    # update
    P_hat = inv(inv(P_pre) + H.T@inv(R)@H)
    state_hat = state_pre - (P_hat@H.T@inv(R)@(obs_pre - obs_next).reshape((1,1))).T
    state_hat = state_hat.reshape((2,))

    return state_hat, P_hat

# solve one-step optimization problem to get state estimation, nonlinear least square filter
def NLSF(state_mu, P, obs_next, Q=np.array([[0.0001,0],[0,1]]), R=np.array([[1]])) : 
    state_next_mu, _ = dyn.step(state_mu)
    x0 = np.hstack((state_mu, state_next_mu))
    result = least_squares(residual_fun, x0, jac_fun, method='lm', args=(state_mu, P, obs_next, Q, R))
    return result.x

# residual function in NLSF
def residual_fun(x, state_mu, P, obs_next, Q, R) : 
    x_hat = x[:2]
    x_next_hat = x[2:]
    fx, _ = dyn.step(x_hat)
    hx_next = x_next_hat[0] - 3*x_next_hat[1] ## 由于系统动态函数中动态方程和观测方程不是分开写的，所以这里只能单独把观测方程搬过来

    L = block_diag([inv(P), inv(Q), inv(R)])
    L = np.linalg.cholesky(L)

    f1 = (x_hat - state_mu)
    f2 = (x_next_hat - fx)
    f3 = (obs_next - hx_next)
    f = np.hstack((f1, f2, f3))
    f = f @ L

    return f

def jac_fun(x, state_mu, P, obs_next, Q, R) : 
    L = block_diag([inv(P), inv(Q), inv(R)])
    L = np.linalg.cholesky(L)

    J = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [-0.99,-0.2,1,0],
                  [0.1,-0.5*(1-x[1]**2)/(1+x[1]**2)**2,0,1],
                  [0,0,-1,3]])
    return (L @ J)


# if __name__ == '__main__' : 
    # residual_fun(np.array([0,1,2,3]), np.array([1,2]), np.array([[1,0],[0,1]]), 3)

    # state_mu = np.array([1,2])
    # P = np.array([[1,0],[0,1]])
    # obs_next = 3
    # result = least_squares(residual_fun, np.array([1,2,1,2]), args=(state_mu, P, obs_next))
    # print(result.x)