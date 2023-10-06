import numpy as np
from functions import block_diag, inv, delete_empty
from scipy.optimize import least_squares, minimize
from scipy.stats import multivariate_normal
import dynamics as dyn
import casadi

# Extended Kalman Filter
def EKF (x, P, y_next, Q, R) : 
    # linearization system matrix 1 #####################
    F = dyn.F(x)
    H = dyn.H(x)
    # ###################################################

    # predict
    P_pre = F @ P @ F.T
    if Q.size != 0 : P_pre = P_pre + Q
    x_pre, y_pre = dyn.step(x)
    # update
    P_hat = inv(inv(P_pre) + H.T@inv(R)@H)
    x_hat = x_pre - (P_hat@H.T@inv(R)@(y_pre - y_next).T).T
    x_hat = np.squeeze(x_hat)

    return x_hat, P_hat

# Unscented Kalman Filter
def UKF(state, P, obs_next, Q, R, alpha=.3, beta=2., kappa=-1.) : 
    n = state.size
    nw = Q.shape[1]
    nv = R.shape[1]
    na = n + nw + nv
    lamda = alpha**2 * (na + kappa) - na

    # calculate sigma points and weights
    xa = np.hstack((state, np.zeros((nw, )), np.zeros((nv, ))))
    xa_sigma = np.tile(xa, (2*na+1, 1))
    M = (na+lamda)*block_diag([P, Q, R])
    M = np.linalg.cholesky(M)
    xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
    xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
    xx_sigma = xa_sigma[:, :n]
    xw_sigma = xa_sigma[:,n:n+nw]
    xv_sigma = xa_sigma[:,n+nw: ]
    Wc = np.ones((2*na+1, )) * 0.5 / (na+lamda)
    Wm = np.ones((2*na+1, )) * 0.5 / (na+lamda)
    Wc[0] = lamda / (na + lamda)
    Wm[0] = lamda / (na + lamda) + 1 - alpha**2 + beta

    # time update
    x_next_pre = np.array([dyn.f(xx_sigma[i], xw_sigma[i]).reshape((n,1)) for i in range(2*na+1)])
    x_next_pre_aver = np.average(x_next_pre, weights=Wm, axis=0)
    P_next_pre = np.zeros((n,n))
    for i in range(2*na+1) : 
        P_next_pre += Wc[i] * (x_next_pre[i] - x_next_pre_aver) @ (x_next_pre[i] - x_next_pre_aver).T

    # measurement update ## 有一种是直接用上面的sigma点做y的预测的，还有一种是用上面算出来的x_pre_aver和P_pre重新选择sigma点做y预测的，下面先采用前者简单方式
    y_next_pre = np.array([dyn.h(np.squeeze(x_next_pre[i]), xv_sigma[i]).reshape((nv,1)) for i in range(2*na+1)])
    y_next_pre_aver = np.average(y_next_pre, weights=Wm, axis=0)
    P_yy = np.zeros_like(R)
    P_xy = np.zeros((n, nv))
    for i in range(2*na+1) : 
        P_yy += Wc[i] * (y_next_pre[i] - y_next_pre_aver) @ (y_next_pre[i] - y_next_pre_aver).T # 这里到底加不加R
        P_xy += Wc[i] * (x_next_pre[i] - x_next_pre_aver) @ (y_next_pre[i] - y_next_pre_aver).T
    K = P_xy @ inv(P_yy)
    x_next_hat = x_next_pre_aver + K @ (obs_next[:, np.newaxis] - y_next_pre_aver)
    P_next_hat = P_next_pre - K @ P_yy @ K.T

    return x_next_hat.reshape((n, )), P_next_hat

# solve one-step optimization problem to get state estimation, nonlinear least square filter
def NLSF(state_hat, P_pre, obs_next_list, Q, R) : 
    x0 = [state_hat]
    dim_state = state_hat.size
    for i in range(len(obs_next_list)) : x0.append(dyn.f(x0[-1]))
    x0 = np.array(x0).reshape(-1)
    result = least_squares(residual_fun, x0, method='lm', jac=jac_fun, args=(state_hat, dim_state, P_pre, obs_next_list, Q, R)) # , max_nfev=8
    return result.x

# residual function in NLSF
def residual_fun(x, state_hat, ds, P_pre, obs_next_list, Q, R) : 
    num_var = len(obs_next_list) + 1
    x_hat = []
    fx    = []
    hx    = []
    for i in range(num_var) : 
        x_hat.append(x[ds*i:ds*(i+1)])
        fx.append(dyn.f(x_hat[i]))
        if i > 0 : hx.append(dyn.h(x_hat[i]))

    f1 = np.tile(np.array(x_hat[0] - state_hat), (1,1))
    f2 = np.array([(x_hat[i] - fx[i-1]) for i in range(1, num_var)])
    f3 = np.array([(obs_next_list[i] - hx[i]) for i in range(num_var-1)]).reshape((1,-1))

    Q, delete_list = delete_empty(Q)
    f2 = np.delete(f2, delete_list, axis=1)
    R, delete_list = delete_empty(R)
    f3 = np.delete(f3, delete_list, axis=1)
    f = np.hstack((f1, f2.reshape(1,-1), f3.reshape(1,-1)))

    Q = block_diag([Q for i in range(num_var-1)])
    R = block_diag([R for i in range(num_var-1)])
    L = block_diag([inv(P_pre), inv(Q), inv(R)])
    L = np.linalg.cholesky(L)

    f = (f @ L).reshape(-1)

    return f


def jac_fun(x, state_hat, ds, P_pre, obs_next_list, Q, R) : 
    num_obs = len(obs_next_list)
    do = obs_next_list[0].size
    J = np.eye(ds)
    jadd = lambda x0, x1 : np.hstack((np.pad(-dyn.F(x0), ((0,do),(0,0))), np.vstack((np.eye(ds), -dyn.H(x1)))))

    # delete_list = []
    # for i in range(Q.shape[0]) : 
    #     if Q[i,i] == 0 : delete_list.append(i)
    # Q = np.delete(Q, delete_list, axis=0)
    # Q = np.delete(Q, delete_list, axis=1)
    # J = np.delete(J, (state_pre.size+np.array(delete_list)).tolist(), axis=0)
    # delete_list = []
    # for i in range(R.shape[0]) : 
    #     if R[i,i] == 0 : delete_list.append(i)
    # R = np.delete(R, delete_list, axis=0)
    # R = np.delete(R, delete_list, axis=1)
    # J = np.delete(J, (state_pre.size+Q.shape[0]+np.array(delete_list)).tolist(), axis=0)

    for i in range(num_obs) : 
        J = np.pad(J, ((0,0), (0,ds))) # (axis0(前,后) axis1(前,后))
        Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
        J = np.vstack((J, Jadd))

    Q = block_diag([Q for i in range(num_obs)])
    R = block_diag([R for i in range(num_obs)])
    L = block_diag([inv(P_pre), inv(Q), inv(R)])
    L = np.linalg.cholesky(L).T

    return (L @ J)

# unknown error in jac_fun, using optimize.minimize methods for comparision to find where the error is
# def OPTF(state_pre, P_pre, obs_next, Q, R) : 
#     state_next_pre = dyn.f(state_pre)
#     x0 = np.hstack((state_pre, state_next_pre))
#     result = minimize(obj_fun, x0, args=(state_pre, P_pre, obs_next, Q, R), method='Newton-CG', jac=opt_jac) # , options={'maxiter':8}
#     return result.x

# def obj_fun(x, state_pre, P_pre, obs_next, Q, R) : 
#     x_hat = x[:2]
#     x_next_hat = x[2:]
#     fx = dyn.f(x_hat)
#     hx_next = dyn.h(x_next_hat)

#     W = block_diag([inv(P_pre), inv(Q), inv(R)])

#     f1 = (x_hat - state_pre)
#     f2 = (x_next_hat - fx)
#     f3 = (obs_next - hx_next)
#     f = np.hstack((f1, f2, f3))
#     f = f @ W @ f.T

#     return f

# def opt_jac(x, state_pre, P_pre, obs_next, Q, R) : 
#     x_hat = x[:2]
#     x_next_hat = x[2:]
#     fx = dyn.f(x_hat)
#     hx_next = dyn.h(x_next_hat)
#     f1 = (x_hat - state_pre)
#     f2 = (x_next_hat - fx)
#     f3 = (obs_next - hx_next)
#     f = np.hstack((f1, f2, f3))

#     W = block_diag([inv(P_pre), inv(Q), inv(R)])

#     # jac1 #####################################################
#     J = np.array([[1,0,0,0],
#                   [0,1,0,0],
#                   [-.99,-.2,1,0],
#                   [.1,-.5*(1-x[1]**2)/(1+x[1]**2)**2,0,1],
#                   [0,0,-1,3]])
#     # ##########################################################

#     # jac2 #####################################################
#     # J = np.array([[1,0,0,0],
#     #               [0,1,0,0],
#     #               [-1,0.1,1,0],
#     #               [-0.04*x[0]*x[1]-0.1,-0.02*x[0]**2-0.98,0,1],
#     #               [0,0,-1,0],
#     #               [0,0,0,-1]])
#     # ##########################################################

#     return 2*f @ W @ J


# class Particle_Filter() : 
    # def __init__(self, state_dim:int, obs_dim:int, num_particles:int, fx, hx, x0_mu, P0, threshold=None, rand_num=1111) -> None:
    #     self.state_dim = state_dim
    #     self.obs_dim   = obs_dim
    #     self.N         = num_particles
    #     self.fx        = fx
    #     self.hx        = hx
    #     self.threshold = self.N * 0.5 if threshold is None else threshold
    #     np.random.seed(seed=rand_num)
    #     self.create_gaussian_particles(x0_mu, P0, self.N)

    # def create_uniform_particles(self, state_dim, state_range, N) : 
    #     self.particles = np.empty((N, state_dim))
    #     for i in range(state_dim) : 
    #         self.particles[:, i] = np.random.uniform(state_range[i][0], state_range[i][1], size=N)
    #     self.weight = np.ones((N, ))/N

    # def create_gaussian_particles(self, mean, cov, N) : 
    #     self.particles = np.random.multivariate_normal(mean, cov, N)
    #     self.weight = np.ones((N, ))/N
    
    # def predict(self, noise_Q, noise_mu=None, dt=.1) : 
    #     if noise_mu is None : noise_mu = np.zeros((self.state_dim, ))
    #     process_noise = np.random.multivariate_normal(noise_mu, noise_Q, self.N)
    #     self.particles = self.fx(self.particles, process_noise, dt)

    # def update(self, observation, obs_noise_R) : 
    #     for i in range(self.N) : 
    #         self.weight[i] *= multivariate_normal(self.hx(self.particles[i]), obs_noise_R).pdf(observation)
        
    #     self.weight += 1.e-300          # avoid round-off to zero
    #     self.weight /= sum(self.weight) # normalize

    # def estimate(self) : 
    #     state_hat = np.average(self.particles, weights=self.weight, axis=0)
    #     Cov_hat = np.cov(self.particles, rowvar=False, aweights=self.weight)

    #     if self.neff() >= self.threshold : 
    #         print(f'resample, Wneff={self.neff()}')
    #         self.simple_resample()
    #     return state_hat, Cov_hat
    
    # def simple_resample(self) : 
    #     cumulative_sum = np.cumsum(self.weight)
    #     cumulative_sum[-1] = 1.  # avoid round-off error
    #     indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))

    #     # resample according to indexes
    #     self.particles = self.particles[indexes]
    #     self.weight = np.ones((self.N, ))/self.N

    # def neff(self) : 
    #     return 1. / np.sum(np.square(self.weight))