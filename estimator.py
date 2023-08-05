import numpy as np
import dynamics as dyn

# Extended Kalman Filter
def EKF (state, Cov, obs_next) : 
    # linearization system matrix
    F = np.array([[0.99,0.2],[-0.1,0.5*(1-state[1]**2)/(1+state[1]**2)**2]])
    Q = np.array([[0,0],[0,1]])
    H = np.array([[1,-3]])
    R = np.array([[1]])
    
    # predict
    Cov_pre = F @ Cov @ F.T + Q
    state_pre, obs_pre = dyn.step(state, 0, 0)
    # update
    Cov_hat = inv(inv(Cov_pre) + H.T@inv(R)@H)
    state_hat = state_pre - (Cov_hat@H.T@inv(R)@(obs_pre - obs_next).reshape((1,1))).T
    state_hat = state_hat.reshape((2,))

    return state_hat, Cov_hat

# inverse of matrix M
def inv(M) : 
    if M.shape == (1,1) : 
        return 1/M 
    else :
        return np.linalg.inv(M)