import numpy as np
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

from functions import block_diag, inv, delete_empty, cholesky4semi
import dynamics as dyn

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
def UKF(state, P, obs_next, Q, R, alpha=.5, beta=2., kappa=-5.) : 
    n = state.size
    nw = Q.shape[1]
    nv = R.shape[1]
    na = n + nw + nv
    lamda = alpha**2 * (na + kappa) - na

    # calculate sigma points and weights
    xa = np.hstack((state, np.zeros((nw, )), np.zeros((nv, ))))
    xa_sigma = np.tile(xa, (2*na+1, 1))
    M = (na+lamda)*block_diag([P, Q, R])
    M = cholesky4semi(M)
    xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
    xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
    xx_sigma = xa_sigma[:, :n]
    xw_sigma = xa_sigma[:,n:n+nw]
    xv_sigma = xa_sigma[:,n+nw: ]
    Wc = np.ones((2*na+1, )) * 0.5 / (na+lamda)
    Wm = np.ones((2*na+1, )) * 0.5 / (na+lamda)
    Wc[0] = lamda / (na + lamda) + 1 - alpha**2 + beta
    Wm[0] = lamda / (na + lamda)

    # time update
    x_next_pre = dyn.f(xx_sigma)
    x_next_pre_aver = np.average(x_next_pre, weights=Wm, axis=0)
    P_next_pre = np.zeros((n,n))
    for i in range(2*na+1) : 
        P_next_pre += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (x_next_pre[i] - x_next_pre_aver).reshape(1,-1)
    P_next_pre += Q
    
    # resample sigma points
    xa = np.hstack((x_next_pre_aver, np.zeros((nw, )), np.zeros((nv, ))))
    xa_sigma = np.tile(xa, (2*na+1, 1))
    M = (na+lamda)*block_diag([P_next_pre, Q, R])
    M = cholesky4semi(M)
    xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
    xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
    xx_sigma = xa_sigma[:, :n]
    xw_sigma = xa_sigma[:,n:n+nw]
    xv_sigma = xa_sigma[:,n+nw: ]

    # measurement update ## 有一种是直接用上面的sigma点做y的预测的，还有一种是用上面算出来的x_pre_aver和P_pre重新选择sigma点做y预测的，下面先采用前者简单方式
    y_next_pre = dyn.h(xx_sigma)
    y_next_pre_aver = np.average(y_next_pre, weights=Wm, axis=0)
    P_yy = np.zeros_like(R)
    P_xy = np.zeros((n, nv))
    for i in range(2*na+1) : 
        P_yy += Wc[i] * (y_next_pre[i] - y_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
        P_xy += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
    P_yy += R
    K = P_xy @ inv(P_yy)
    x_next_hat = x_next_pre_aver + K @ (obs_next - y_next_pre_aver)
    P_next_hat = P_next_pre - K @ P_yy @ K.T

    return x_next_hat.reshape(-1), P_next_hat


def NLSF_uniform(P_inv, y_seq, Q, R, mode:str="quadratic", x0=None, **args) : 
    if "sumofsquares" in mode.lower() : 
        fun = SumOfSquares()
        params = [P_inv, y_seq, Q, R]
    elif "quadratic" in mode.lower() : 
        fun = Quadratic()
        params = [P_inv, y_seq, Q, R, args["x0_bar"]]
    
    if "xend" in mode : params.append(args["xend"])

    ds = Q.shape[0]
    if x0 is None : 
        x0 = np.zeros((ds*(len(y_seq)+1), ))
    else : 
        while (len(x0) <= len(y_seq)) : 
            x0.append(dyn.f(x0[-1]))
        x0 = np.array(x0).reshape(-1)
    result = least_squares(fun.res_fun, x0, method='lm', jac=fun.jac_fun, args=params) # , max_nfev=8
    return result.x, result.fun

class SumOfSquares() : 
    def __init__(self) -> None:
        pass

    def res_fun(self, x, LP, y_seq, Q, R, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))

        LQ = np.linalg.cholesky(inv(Q))
        LR = np.linalg.cholesky(inv(R))
        f = np.insert(x[:ds], 0, 1)[np.newaxis,:]
        L = LP[:]
        for i in range(num_obs) : 
            f = np.hstack((f, x[ds*(i+1):ds*(i+2)]-dyn.f(x[ds*(i):ds*(i+1)])[np.newaxis,:], 
                              y_seq[i]-dyn.h(x[ds*(i+1):ds*(i+2)])[np.newaxis,:]))
            L = block_diag((L, LQ, LR))
        
        if xend is not None : 
            f = np.hstack((f, (xend-dyn.f(x[-ds:]))[np.newaxis,:]))
            L = block_diag((L, LQ))
        
        return (f@L).reshape(-1)

    def jac_fun(self, x, LP, y_seq, Q, R, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : np.vstack((np.hstack((-dyn.F(x0), np.eye(ds))), np.pad(-dyn.H(x1), ((0,0),(ds,0)))))

        LQ = np.linalg.cholesky(inv(Q))
        LR = np.linalg.cholesky(inv(R))
        J = np.pad(np.eye(ds), ((1,0),(0,0)))
        L = LP[:]
        for i in range(num_obs) : 
            J = np.pad(J, ((0,0), (0,ds)))
            Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
            J = np.vstack((J, Jadd))
            L = block_diag((L, LQ, LR))

        if xend is not None : 
            Jadd = np.pad(-dyn.F(xend), ((0,0),(ds*num_obs,0)))
            J = np.vstack((J, Jadd))
            L = block_diag((L, LQ))

        return L.T@J

class Quadratic() : 
    def __init__(self) -> None:
        pass

    def res_fun(self, x, P_inv, y_seq, Q, R, x0_bar, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))

        f = np.tile(np.array(x[:ds] - x0_bar), (1,1))
        M = P_inv[:]
        for i in range(num_obs) : 
            f = np.hstack((f, x[ds*(i+1):ds*(i+2)]-dyn.f(x[ds*(i):ds*(i+1)])[np.newaxis,:], 
                              y_seq[i]-dyn.h(x[ds*(i+1):ds*(i+2)])[np.newaxis,:]))
            M = block_diag((M, inv(Q), inv(R)))
        
        if xend is not None : 
            f = np.hstack((f, (xend-dyn.f(x[-ds:]))[np.newaxis,:]))
            M = block_diag((M, inv(Q)))
        
        L = np.linalg.cholesky(M)
        return (f@L).reshape(-1)

    def jac_fun(self, x, P_inv, y_seq, Q, R, x0_bar, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : np.vstack((np.hstack((-dyn.F(x0), np.eye(ds))), np.pad(-dyn.H(x1), ((0,0),(ds,0)))))

        J = np.eye(ds)
        M = P_inv[:]
        for i in range(num_obs) : 
            J = np.pad(J, ((0,0), (0,ds)))
            Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
            J = np.vstack((J, Jadd))
            M = block_diag((M, inv(Q), inv(R)))

        if xend is not None : 
            Jadd = np.pad(-dyn.F(xend), ((0,0),(ds*num_obs,0)))
            J = np.vstack((J, Jadd))
            M = block_diag((M, inv(Q)))

        L = np.linalg.cholesky(M).T
        return L@J


# solve one-step optimization problem to get state estimation, nonlinear least square filter
def NLSF(state_hat, P_pre, obs_next_list, Q, R, initial_x=None) : 
    dim_state = state_hat.size
    if not initial_x : # 这个判断语句对空列表也成立
        initial_x = [state_hat]
    while (len(initial_x) <= len(obs_next_list)) : 
        initial_x.append(dyn.f(initial_x[-1]))
    initial_x = np.array(initial_x).reshape(-1)
    result = least_squares(residual_fun, initial_x, method='lm', jac=jac_fun, args=(state_hat, dim_state, P_pre, obs_next_list, Q, R)) # , max_nfev=8
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
    f3 = np.array([(obs_next_list[i] - hx[i]) for i in range(num_var-1)])

    Q, delete_list = delete_empty(Q)
    f2 = np.delete(f2, delete_list, axis=1)
    R, delete_list = delete_empty(R)
    f3 = np.delete(f3, delete_list, axis=1)
    f = np.hstack((f1, f2.reshape(1,-1), f3.reshape(1,-1)))

    Q_block = block_diag([Q for i in range(num_var-1)])
    R_block = block_diag([R for i in range(num_var-1)])
    L = block_diag([inv(P_pre), inv(Q_block), inv(R_block)])
    L = np.linalg.cholesky(L)

    f = (f @ L).reshape(-1)

    return f

def jac_fun(x, state_hat, ds, P_pre, obs_next_list, Q, R) : 
    num_obs = len(obs_next_list)
    do = obs_next_list[0].size
    J = np.eye(ds)
    jaddQ = lambda x0 : np.hstack((-dyn.F(x0), np.eye(ds)))
    jaddR = lambda x1 : -dyn.H(x1)

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

    J = np.pad(J, ((0,0), (0,num_obs*ds))) # (axis0(前,后) axis1(前,后))
    for i in range(num_obs) : 
        JaddQ = np.pad(jaddQ(x[ds*i:ds*(i+1)]), ((0,0), (i*ds,(num_obs-i-1)*ds)))
        J = np.vstack((J, JaddQ))
    for i in range(num_obs) : 
        JaddR = np.pad(jaddR(x[ds*(i+1):ds*(i+2)]), ((0,0), ((i+1)*ds,(num_obs-i-1)*ds)))
        J = np.vstack((J, JaddR))

    Q_block = block_diag([Q for i in range(num_obs)])
    R_block = block_diag([R for i in range(num_obs)])
    L = block_diag([inv(P_pre), inv(Q_block), inv(R_block)])
    L = np.linalg.cholesky(L).T

    # jac_com = approx_derivative(residual_fun, x, args=(state_hat, ds, P_pre, obs_next_list, Q, R))

    return (L @ J)

def NLSF_xt(xt_hat, Pt_inv, y_list, xte, Q_inv, R_inv) : 
    x0 = [xt_hat]
    for i in range(len(y_list)) : x0.append(dyn.f(x0[-1]))
    x0 = np.array(x0).reshape(-1)
    result = least_squares(res_fun_xt, x0, method='lm', jac=jac_fun_xt, args=(xt_hat, Pt_inv, y_list, xte, Q_inv, R_inv))
    return result.x, result.fun

def res_fun_xt(x, xt_bar, Pt_inv, y_list, xte, Q_inv, R_inv) : 
    num_var = len(y_list) + 1
    ds = xt_bar.size
    x_hat = []
    fx_hat = []
    hx_next_hat = []
    for i in range(num_var) : 
        x_hat.append(x[ds*i:ds*(i+1)])
        fx_hat.append(dyn.f(x[ds*i:ds*(i+1)]))
        if i > 0 : hx_next_hat.append(dyn.h(x[ds*i:ds*(i+1)]))

    f1 = np.tile(np.array(x_hat[0] - xt_bar), (1,1))
    f2 = np.array([(x_hat[i] - fx_hat[i-1]) for i in range(1, num_var)])
    f3 = np.array(xte - fx_hat[-1]).reshape(1,-1)
    f4 = np.array([(y_list[i] - hx_next_hat[i]) for i in range(num_var-1)])
    f = np.hstack((f1, f2.reshape(1,-1), f3, f4.reshape(1,-1)))

    Q_block = block_diag([Q_inv for _ in range(num_var)])
    R_block = block_diag([R_inv for _ in range(num_var-1)])
    L = block_diag([Pt_inv, Q_block, R_block])
    L = cholesky4semi(L)

    f = (f @ L).reshape(-1)
    return f

def jac_fun_xt(x, xt_hat, Pt_inv, y_list, xte, Q_inv, R_inv) : 
    ds = xt_hat.size
    num_obs = len(y_list)
    J = np.eye(ds)
    jaddQ = lambda x0 : np.hstack((-dyn.F(x0), np.eye(ds)))
    jaddR = lambda x1 : -dyn.H(x1)

    J = np.pad(J, ((0,0), (0,num_obs*ds))) # (axis0(前,后) axis1(前,后))
    for i in range(num_obs) : 
        JaddQ = np.pad(jaddQ(x[ds*i:ds*(i+1)]), ((0,0), (i*ds,(num_obs-i-1)*ds)))
        J = np.vstack((J, JaddQ))
    J = np.vstack((J, np.pad(-dyn.F(x[-ds:]), ((0,0),(num_obs*ds,0)))))
    for i in range(num_obs) : 
        JaddR = np.pad(jaddR(x[ds*(i+1):ds*(i+2)]), ((0,0), ((i+1)*ds,(num_obs-i-1)*ds)))
        J = np.vstack((J, JaddR))

    Q_block = block_diag([Q_inv for _ in range(num_obs+1)])
    R_block = block_diag([R_inv for _ in range(num_obs)])
    L = block_diag([Pt_inv, Q_block, R_block])
    L = cholesky4semi(L).T

    return (L @ J)


def NLSF_sos(state_hat, P_pre, obs_next_list, Q, R) : 
    x0 = [state_hat]
    dim_state = state_hat.size
    for i in range(len(obs_next_list)) : x0.append(dyn.f(x0[-1]))
    x0 = np.array(x0).reshape(-1)
    result = least_squares(residual_fun_sos, x0, method='lm', jac=jac_fun_sos, args=(state_hat, dim_state, P_pre, obs_next_list, Q, R)) # , max_nfev=8
    return result.x

# residual function in NLSF
def residual_fun_sos(x, state_hat, ds, P_pre, obs_next_list, Q, R) : 
    num_var = len(obs_next_list) + 1
    x_hat = []
    fx    = []
    hx    = []
    for i in range(num_var) : 
        x_hat.append(x[ds*i:ds*(i+1)])
        fx.append(dyn.f(x_hat[i]))
        if i > 0 : hx.append(dyn.h(x_hat[i]))

    f1 = np.tile(np.array(x_hat[0] - state_hat), (1,1))
    f2 = f1**2
    f3 = f1**3
    f4 = np.array([(x_hat[i] - fx[i-1]) for i in range(1, num_var)])
    f5 = np.array([(obs_next_list[i] - hx[i]) for i in range(num_var-1)]).reshape((1,-1))

    f = np.hstack((f1, f2, f3, f4.reshape(1,-1), f5.reshape(1,-1)))

    Q = block_diag([Q for i in range(num_var-1)])
    R = block_diag([R for i in range(num_var-1)])
    L = block_diag([inv(P_pre), inv(Q), inv(R)])
    L = cholesky4semi(L)

    f = (f @ L).reshape(-1)

    return f

def jac_fun_sos(x, state_hat, ds, P_pre, obs_next_list, Q, R) : 
    num_obs = len(obs_next_list)
    do = obs_next_list[0].size
    J = np.eye(ds)
    J1 = np.diag(2*(x[:ds]-state_hat))
    J2 = np.diag(3*(x[:ds]-state_hat)**2)
    J = np.vstack((J, J1, J2))
    jadd = lambda x0, x1 : np.hstack((np.pad(-dyn.F(x0), ((0,do),(0,0))), np.vstack((np.eye(ds), -dyn.H(x1)))))

    for i in range(num_obs) : 
        J = np.pad(J, ((0,0), (0,ds))) # (axis0(前,后) axis1(前,后))
        Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
        J = np.vstack((J, Jadd))

    Q = block_diag([Q for i in range(num_obs)])
    R = block_diag([R for i in range(num_obs)])
    L = block_diag([inv(P_pre), inv(Q), inv(R)])
    L = cholesky4semi(L).T

    return (L @ J)


class Particle_Filter() : 
    def __init__(self, state_dim:int, obs_dim:int, num_particles:int, fx, hx, x0_mu, P0, threshold=None, rand_num=1111) -> None:
        self.state_dim = state_dim
        self.obs_dim   = obs_dim
        self.N         = num_particles
        self.fx        = fx
        self.hx        = hx
        self.threshold = self.N * 0.5 if threshold is None else threshold
        np.random.seed(seed=rand_num)
        self.create_gaussian_particles(x0_mu, P0, self.N)

    def create_uniform_particles(self, state_dim, state_range, N) : 
        self.particles = np.empty((N, state_dim))
        for i in range(state_dim) : 
            self.particles[:, i] = np.random.uniform(state_range[i][0], state_range[i][1], size=N)
        self.weight = np.ones((N, ))/N

    def create_gaussian_particles(self, mean, cov, N) : 
        self.particles = np.random.multivariate_normal(mean, cov, N)
        self.weight = np.ones((N, ))/N
    
    def predict(self, noise_Q, noise_mu=None, dt=.1) : 
        if noise_mu is None : noise_mu = np.zeros((self.state_dim, ))
        process_noise = np.random.multivariate_normal(noise_mu, noise_Q, self.N)
        self.particles = self.fx(self.particles, process_noise, dt)

    def update(self, observation, obs_noise_R) : 
        for i in range(self.N) : 
            self.weight[i] *= multivariate_normal(self.hx(self.particles[i]), obs_noise_R).pdf(observation)
        
        self.weight += 1.e-300          # avoid round-off to zero
        self.weight /= sum(self.weight) # normalize

    def estimate(self) : 
        state_hat = np.average(self.particles, weights=self.weight, axis=0)
        Cov_hat = np.cov(self.particles, rowvar=False, aweights=self.weight)

        if self.neff() >= self.threshold : 
            print(f'resample, Wneff={self.neff()}')
            self.simple_resample()
        return state_hat, Cov_hat
    
    def simple_resample(self) : 
        cumulative_sum = np.cumsum(self.weight)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))

        # resample according to indexes
        self.particles = self.particles[indexes]
        self.weight = np.ones((self.N, ))/self.N

    def neff(self) : 
        return 1. / np.sum(np.square(self.weight))