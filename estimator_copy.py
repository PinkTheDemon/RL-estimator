import numpy as np
from scipy.optimize import least_squares, minimize
import dynamics as dyn

# inverse of matrix M
def inv(M) : 
    if M.shape == (1,1) : 
        return 1/M 
    else :
        return np.linalg.inv(M)
    
# transfer matrix list to one block-diag matrix
def block_diag(matrix_list) : 
    delete = []
    for i in range(len(matrix_list)) : 
        if matrix_list[i].size == 0 : delete.append(i)
    for _ in delete : matrix_list.pop(_)

    bd_M = matrix_list[0]
    for M in matrix_list[1:] : 
        bd_M = np.block([[bd_M, np.zeros((bd_M.shape[0], M.shape[1]))],
                         [np.zeros((M.shape[0], bd_M.shape[1])), M]])
    return bd_M

# Extended Kalman Filter
def EKF (state_pre, P_pre, obs, Q, R) : 
    H = np.array([[1,-3]])
    # update
    P_hat = inv(inv(P_pre) + H.T@inv(R)@H)
    state_hat = state_pre - (P_hat@H.T@inv(R)@(dyn.h(state_pre) - obs).reshape((1,1))).T
    state_hat = state_hat.reshape((2,))
    # linearization system matrix
    F = np.array([[0.99,0.2],[-0.1,0.5*(1-state_hat[1]**2)/(1+state_hat[1]**2)**2]])
    # predict
    state_next_pre = dyn.f(state_hat)
    P_next_pre = F @ P_hat @ F.T + Q

    return state_hat, state_next_pre, P_next_pre

# solve one-step optimization problem to get state estimation, nonlinear least square filter
def NLSF(state_pre, P_pre, obs, Q, R) : 
    state_next_pre = dyn.f(state_pre)
    x0 = np.hstack((state_pre, state_next_pre))
    result = least_squares(residual_fun, x0, jac_fun, method='lm', args=(state_pre, P_pre, obs, Q, R)) # , max_nfev=8
    return result.x

# residual function in NLSF
def residual_fun(x, state_pre, P, obs, Q, R) : 
    x_hat = x[:2]
    x_next_pre = x[2:]
    fx = dyn.f(x_hat)
    hx = dyn.h(x_hat)

    L = block_diag([inv(P), inv(Q), inv(R)])
    L = np.linalg.cholesky(L)

    f1 = (x_hat - state_pre)
    f2 = (x_next_pre - fx)
    f3 = (obs - hx)
    f = np.hstack((f1, f2, f3))
    f = f @ L

    return f

def jac_fun(x, state_pre, P_pre, obs, Q, R) : 
    L = block_diag([inv(P_pre), inv(Q), inv(R)])
    L = np.linalg.cholesky(L)

    J = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [-0.99,-0.2,1,0],
                  [0.1,-0.5*(1-x[1]**2)/(1+x[1]**2)**2,0,1],
                  [-1,3,0,0]])
    return (L @ J)

# unknown error in jac_fun, using optimize.minimize methods for comparision to find where the error is
def OPTF(state_pre, P_pre, obs, Q, R) : 
    state_next_pre = dyn.f(state_pre)
    x0 = np.hstack((state_pre, state_next_pre))
    result = minimize(obj_fun, x0, args=(state_pre, P_pre, obs, Q, R), method='Newton-CG', jac=opt_jac) # , options={'maxiter':8}
    return result.x

def obj_fun(x, state_pre, P_pre, obs, Q, R) : 
    x_hat = x[:2]
    x_next_pre = x[2:]
    fx = dyn.f(x_hat)
    hx = dyn.h(x_hat)

    W = block_diag([inv(P_pre), inv(Q), inv(R)])

    f1 = (x_hat - state_pre)
    f2 = (x_next_pre - fx)
    f3 = (obs - hx)
    f = np.hstack((f1, f2, f3))
    f = f @ W @ f.T

    return f

def opt_jac(x, state_pre, P_pre, obs, Q, R) : 
    x_hat = x[:2]
    x_next_pre = x[2:]
    fx = dyn.f(x_hat)
    hx = dyn.h(x_hat)
    f1 = (x_hat - state_pre)
    f2 = (x_next_pre - fx)
    f3 = (obs - hx)
    f = np.hstack((f1, f2, f3))

    W = block_diag([inv(P_pre), inv(Q), inv(R)])

    J = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [-0.99,-0.2,1,0],
                  [0.1,-0.5*(1-x[1]**2)/(1+x[1]**2)**2,0,1],
                  [-1,3,0,0]])

    return 2*f @ W @ J

# if __name__ == '__main__' : 
    # residual_fun(np.array([0,1,2,3]), np.array([1,2]), np.array([[1,0],[0,1]]), 3)

    # state_mu = np.array([1,2])
    # P = np.array([[1,0],[0,1]])
    # obs_next = 3
    # result = least_squares(residual_fun, np.array([1,2,1,2]), args=(state_mu, P, obs_next))
    # print(result.x)