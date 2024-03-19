import numpy as np

import dynamics as dyn

class Model() : 
    def __init__(self, dim_state, dim_obs, f, h, F, H, x0_mu, P0, Q, R, f_real=None, h_real=None) -> None:
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.f = f
        self.h = h
        self.x0_mu = x0_mu
        self.P0 = P0
        self.Q = Q
        self.R = R
        self.f_real = f if f_real is None else f_real
        self.h_real = h if h_real is None else h_real

    def step(self, x, disturb=None, noise=None) : 
        x_next = self.f(x, disturb)
        y_next = self.h(x_next, noise)
        return x_next, y_next
    
    def step_real(self, x, disturb=None, noise=None) : 
        x_next = self.f_real(x, disturb)
        y_next = self.h_real(x_next, noise)
        return x_next, y_next
    
    def generate_data(self, maxsteps, disturb_mu=None, noise_mu=None, is_mismatch=False, rand_seed=123) : 
        np.random.seed(rand_seed)
        # 生成初始状态
        if self.P0.size == 0 : 
            initial_state = self.x0_mu
        else :
            initial_state = self.x0_mu + (np.random.multivariate_normal(np.zeros_like(self.x0_mu), self.P0))
        # 噪声是否有偏
        if disturb_mu is None : 
            disturb_mu = np.zeros(self.Q.shape[0])
        if noise_mu is None : 
            noise_mu = np.zeros(self.R.shape[0])
        # 生成噪声序列
        if self.Q.size == 0 : 
            disturb_list = np.zeros((maxsteps, disturb_mu.size))
        else : 
            disturb_list = np.random.multivariate_normal(disturb_mu, self.Q, maxsteps)
        if self.R.size == 0 : 
            noise_list = np.zeros((maxsteps, noise_mu.size))
        else : 
            noise_list = np.random.multivariate_normal(noise_mu, self.R, maxsteps)
        # 生成状态序列和观测序列
        t_seq = range(maxsteps)
        x_seq = []
        y_seq = []
        x = initial_state
        for t in t_seq : # 真实轨迹
            if is_mismatch : 
                x_next, y_next = dyn.step_real(x, disturb_list[t], noise_list[t])
            else : 
                x_next, y_next = dyn.step(x, disturb_list[t], noise_list[t])
            x = x_next
            x_seq.append(x_next)
            y_seq.append(y_next)
        return x_seq, y_seq


def create_model(dim_state, dim_obs, x0_mu, P0, Q, R, f_real=None, h_real=None) : 
    # 验证模型和给定维度是否一致
    x = np.ones(dim_state)
    try : 
        assert dyn.f(x).shape == (dim_state, )
        assert dyn.h(x).shape == (dim_obs, )
    except ValueError : 
        raise ValueError("dynamic or obvious functions do not match the model")
    
    # 创建模型
    model = Model(dim_state, dim_obs, dyn.f, dyn.h, dyn.F, dyn.H, x0_mu, P0, Q, R, f_real, h_real)

    return model