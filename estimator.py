import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import multivariate_normal
from scipy.linalg import expm

from functions import block_diag, inv, delete_empty, isConverge
from model import Continuous2, Continuous4

class Estimator:
    def __init__(self, name, x0_hat=None, P0_hat=None) -> None:
        self.name = name
        self.reset(x0_hat=x0_hat, P0_hat=P0_hat)

    def reset(self, x0_hat, P0_hat):
        self.x_hat = x0_hat
        self.y_hat = None
        self.P_hat = P0_hat

    def estimate(self):
        pass

# calculate (P^-1)*
def cal_Poptim(A:np.ndarray, C:np.ndarray, Q:np.ndarray, R:np.ndarray, B=None, P0=None, gamma=1.0, tol=1e-4) -> np.ndarray:
    '''LQR
    c(x,u) = u.T@R@u + y.T@Q@y
    V(x) = x.T@P@x = c(x,u) + gamma@V'(x)
    '''
    if P0 is None:
        P0 = np.eye(A.shape[0])
    if B is None :
        B = np.eye(A.shape[0])
    P_list = [P0]
    while True:
        P = P_list[-1]
        P = C.T@Q@C + gamma*A.T@P@A - gamma**2*A.T@P@B@inv(R + gamma*B.T@P@B)@B.T@P@A # A.T @ inv(inv(gamma*P) + B@inv(R)@B.T) @ A #
        P_list.append(P)
        # Pinv = A@inv(P + C.T@Q@C)@A.T + inv(R)
        # P_list.append(inv(Pinv))
        if len(P_list) >= 5:
            del P_list[0]
            if isConverge(matrices=P_list, criterion=np.linalg.norm, tol=tol, ord="fro"):
                break
    return P_list[-1]

class EKF(Estimator):
    def __init__(self, f_fn, h_fn, F_fn, H_fn, x0_hat=None, P0_hat=None) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn
        super().__init__(name="EKF", x0_hat=x0_hat, P0_hat=P0_hat)

    def predict(self, Q, u=None):
        F = self.F_fn(x=self.x_hat, u=u)
        self.x_hat = self.f_fn(x=self.x_hat, u=u)
        self.P_hat = F@self.P_hat@F.T
        if Q.size != 0 : self.P_hat += Q

    def update(self, y, R):
        y_pre = self.h_fn(x=self.x_hat)
        H = self.H_fn(x=self.x_hat)
        self.P_hat = self.P_hat - self.P_hat@H.T@inv(R+H@self.P_hat@H.T)@H@self.P_hat
        self.x_hat = self.x_hat - self.P_hat@H.T@inv(R)@(y_pre - y)
        self.y_hat = self.h_fn(x=self.x_hat)

    def estimate(self, y, Q, R, u=None):
        self.predict(Q=Q, u=u)
        self.update(y=y, R=R)


class EKFForC4(Estimator) :
    def __init__(self, model:Continuous4, x0_hat=None, P0_hat=None) -> None:
        self.model = model
        super().__init__(name="EKFForC4", x0_hat=x0_hat, P0_hat=P0_hat)

    def reset(self, x0_hat, P0_hat):
        if hasattr(self, "y_prev") : del self.y_prev
        if hasattr(self, "thetam") : del self.thetam
        self.judgeR_seq = []
        super().reset(x0_hat, P0_hat)

    def estimate(self, y, Q, R, u=None):
        if not hasattr(self, "y_prev") : # y_pre，注意在reset中删去
            self.y_prev = y
            self.y_hat = y
            return
        if (self.x_hat[0:4] == np.zeros((4,))).all(): # 跟参考代码保持一致，用第二个y算q的初始猜测
            y_a = y[3:6]
            y_m = y[6:9]
            roll_accmag_0 = np.arctan2(y_a[1], y_a[2])
            pitch_accmag_0 = -np.arctan2(y_a[0], np.sqrt(y_a[1]**2 + y_a[2]**2))
            C_n_b__0 = self.model.rot(angle=0, axis="z") @ self.model.rot(angle=pitch_accmag_0, axis="Y") @ self.model.rot(angle=roll_accmag_0, axis="X")
            y_m_0_NEW = C_n_b__0.T @ y_m.reshape(-1,1)
            yaw_accmag_0 = -np.arctan2(y_m_0_NEW[1], y_m_0_NEW[0]).item()
            # q_m: Estimated-from-measurements quaternion (i.e., q from y_g (measured gyro output))
            self.x_hat[0:4] = self.model.RotMat2quat(self.model.rot(angle=yaw_accmag_0, axis="Z") @
                                                     self.model.rot(angle=pitch_accmag_0, axis="Y") @
                                                     self.model.rot(angle=roll_accmag_0, axis="X"))
        # omega_pre = self.y_prev[0:3]# - self.x_hat[4:7] # 注意x_hat[4:7]全过程有没有发生变化，没有的话都可以去掉了
        # dt = self.model.sampleTime
        F = self.model.F(x=self.x_hat, omega_pre=self.y_prev[0:3])
        self.x_hat = self.model.f(x=self.x_hat, omega_pre=self.y_prev[0:3])
        self.P_hat = F@self.P_hat@F.T + Q
        y_pre = self.model.h(x=self.x_hat)
        # 调整R矩阵
        # a_hat = y[3:6]# - self.x_hat[4:7]
        # m_hat = y[6:9]# - self.x_hat[7:10]
        # if not hasattr(self, "thetam") : 
        #     self.thetam = self.calTheta(m=self.model.m, g=self.model.g)
        # if abs(np.linalg.norm(a_hat) - np.linalg.norm(self.model.g)) > 0.2:
        #     self.judgeR_seq.append(False)
        # else :
        #     self.judgeR_seq.append(True)
        # if len(self.judgeR_seq) > int(0.1/self.model.sampleTime) :
        #     del self.judgeR_seq[0]
        # if True not in self.judgeR_seq :
        #     R[3:6,3:6] = np.zeros((3,3))
        # if abs(np.linalg.norm(m_hat) - np.linalg.norm(self.model.m)) > 0.1 or \
        #    abs(self.calTheta(m=m_hat, g=a_hat) - self.thetam) > 10:
        #         R[6:9,6:9] = np.zeros((3,3))
        # 修正
        H = self.model.H(x=self.x_hat)
        self.P_hat = self.P_hat - self.P_hat@H.T@inv(R+H@self.P_hat@H.T)@H@self.P_hat
        self.x_hat = self.x_hat - self.P_hat@H.T@inv(R)@((y_pre - y))
        self.x_hat[0:4] /= np.linalg.norm(self.x_hat[0:4])
        # if self.x_hat[0] < 0: self.x_hat[0:4] = -self.x_hat[0:4]
        # self.y_hat = self.model.h(x=self.x_hat)
        self.y_hat = y

    def calTheta(self, m, g) :
        return np.arccos( np.dot(m, g) / np.linalg.norm(m) / np.linalg.norm(g) )* 180 / np.pi

class MHE(Estimator):
    def __init__(self, f_fn, h_fn, F_fn, H_fn, window, x0_hat=None, P0_hat=None) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn
        self.window = window
        super().__init__(name="MHE", x0_hat=x0_hat, P0_hat=P0_hat)

    def reset(self, x0_hat=None, P0_hat=None):
        self.x_hat = x0_hat
        if x0_hat is not None : self.dim_state = x0_hat.size
        self.P_hat = P0_hat
        self.y_seq = []
        self.x0_bar_seq = [x0_hat]

    def estimate(self, y, Q, R, u=None, P_inv=None):
        self.y_seq.append(y)
        if len(self.y_seq) > self.window : del self.y_seq[0]
        if P_inv is None : P_inv = inv(self.P_hat)
        result = NLSF_uniform(P_inv=P_inv, y_seq=self.y_seq, Q=Q, R=R, f=self.f_fn, h=self.h_fn, F=self.F_fn, H=self.H_fn, 
                              mode="quadratic", x0=self.x0_bar_seq[:], x0_bar=self.x0_bar_seq[0])
        self.x_hat = result.x[-self.dim_state:]
        self.y_hat = self.h_fn(x=self.x_hat)
        # EKF方法更新P(直观解释：x0被删去的时候才需要更新P)
        if len(self.x0_bar_seq) == self.window: 
            x0_hat = self.x0_bar_seq[0]
            x1_pre = self.f_fn(x=x0_hat) # 先不加u了很麻烦，有u的话再说吧
            F = self.F_fn(x=x0_hat)
            P_pre = F@self.P_hat@F.T + Q
            H = self.H_fn(x=x1_pre)
            self.P_hat = P_pre - P_pre@H.T@inv(R+H@P_pre@H.T)@H@P_pre
        # 更新x0_bar_seq
        # self.x0_bar_seq = list(result.x.reshape(-1, self.dim_state)) # 要用新的就都用新的
        self.x0_bar_seq.append(self.x_hat) # 要不用新的就都不用新的，训练的时候是不用新的所以测试也不应该用新的
        if len(self.x0_bar_seq) > self.window : del self.x0_bar_seq[0]

class MHEForC4(Estimator) :
    def __init__(self, model:Continuous4, window, x0_hat=None, P0_hat=None) -> None:
        self.model = model
        self.window = window
        super().__init__("MHEForC4", x0_hat, P0_hat)

    def reset(self, x0_hat, P0_hat):
        self.x_hat = x0_hat
        self.P_hat = P0_hat
        self.y_seq = []
        self.x0_bar_seq = [x0_hat]
    
    def estimate(self, y, Q, R, gamma=1.0, xend=None):
        # 定义W0
        ds = self.model.dim_state
        #region 数据处理
        if len(self.y_seq) == 0 : # y0用来计算四元数q的初始估计
            y_a = y[3:6]
            y_m = y[6:9]
            roll_accmag_0 = np.arctan2(y_a[1], y_a[2])
            pitch_accmag_0 = -np.arctan2(y_a[0], np.sqrt(y_a[1]**2 + y_a[2]**2))
            C_n_b__0 = self.model.rot(angle=0, axis="z") @ self.model.rot(angle=pitch_accmag_0, axis="Y") @ self.model.rot(angle=roll_accmag_0, axis="X")
            y_m_0_NEW = C_n_b__0.T @ y_m.reshape(-1,1)
            yaw_accmag_0 = -np.arctan2(y_m_0_NEW[1], y_m_0_NEW[0]).item()
            q = self.model.RotMat2quat(self.model.rot(angle=yaw_accmag_0, axis="Z") @
                                       self.model.rot(angle=pitch_accmag_0, axis="Y") @
                                       self.model.rot(angle=roll_accmag_0, axis="X"))
            self.y_hat = y
            self.y_seq.append(y)
            self.x_hat = np.hstack(( q, np.zeros((6,))))#q#
            self.x0_bar_seq[0] = self.x_hat
            return 
        # 保存y到y_seq
        self.y_seq.append(y)
        if len(self.y_seq) > self.window+1 : del self.y_seq[0]
        #endregion
        result = NLSFForC4(model=self.model, x0_bar=self.x0_bar_seq[0], y_seq=self.y_seq, P_inv=inv(self.P_hat), Q=Q, R=R, gamma=gamma, xend=xend)
        self.x_hat = result.x[-ds:]
        self.x_hat[0:4] /= np.linalg.norm(self.x_hat[0:4])
        self.y_hat = y
        # EKF方法更新P(直观解释：x0被删去的时候才需要更新P)
        if len(self.x0_bar_seq) == self.window: 
            x0_hat = self.x0_bar_seq[0]
            x1_pre = self.model.f(x=x0_hat, omega_pre=self.y_seq[0][0:3])
            F = self.model.F(x=x0_hat, omega_pre=self.y_seq[0][0:3])
            P_pre = F@self.P_hat@F.T + Q
            H = self.model.H(x=x1_pre)
            self.P_hat = P_pre - P_pre@H.T@inv(R+H@P_pre@H.T)@H@P_pre
        # 更新x0_bar_seq
        self.x0_bar_seq.append(self.x_hat)
        if len(self.x0_bar_seq) > self.window: del self.x0_bar_seq[0]

#region NLSFForC4
def NLSFForC4(model:Continuous4, x0_bar, y_seq, P_inv, Q, R, x0=None, gamma=1.0, xend=None):
    # 生成初始值
    if x0 is None : 
        x0 = [x0_bar]
    for i in range(len(x0)-1, len(y_seq)-1) : 
        x0.append(model.f(x=x0[i], omega_pre=y_seq[i][0:3]))
    x0 = np.array(x0).reshape(-1)
    # 参数打包
    params = (model, x0_bar, y_seq, P_inv, Q, R, gamma, xend)
    # 计算最小二乘问题
    result = least_squares(fun=resForC4, x0=x0, args=params, method='lm', jac=jacForC4) #, max_nfev=10
    return result

def resForC4(x, model:Continuous4, x0_bar, y_seq, P_inv, Q, R, gamma=1.0, xend=None):
    num_y = len(y_seq)
    ds = 10

    f = (x[0:ds] - x0_bar)[np.newaxis,:]
    M = np.copy(P_inv)
    for i in range(num_y-1) :
        f = np.hstack(( f, (x[ds*(i+1):ds*(i+2)] - model.f(x=x[ds*(i):ds*(i+1)], omega_pre=y_seq[i][0:3]))[np.newaxis,:], 
                           (y_seq[i+1][3:] - model.h(x=x[ds*(i+1):ds*(i+2)])[3:])[np.newaxis,:] ))
        M = block_diag(( M*gamma, inv(Q), inv(R[3:,3:]) ))

    if xend is not None:
        f = np.hstack(( f, (xend-model.f(x=x[-ds:], omega_pre=y_seq[-1][0:3]))[np.newaxis,:] ))
        M = block_diag(( M*gamma, inv(Q) ))

    L = np.linalg.cholesky(M)
    return (f@L).reshape(-1)

def jacForC4(x, model:Continuous4, x0_bar, y_seq, P_inv, Q, R, gamma=1.0, xend=None):
    num_y = len(y_seq)
    ds = 10
    jadd = lambda x0, y0, x1 : np.vstack(( np.hstack(( -model.F(x=x0, omega_pre=y0[0:3]), np.eye(ds) )), np.pad(-model.H(x=x1)[3:], ((0,0),(ds,0))) ))

    J = np.eye(ds)
    M = np.copy(P_inv)
    for i in range(num_y-1):
        J = np.pad(J, ((0,0),(0,ds)))
        Jadd = np.pad(jadd(x0=x[ds*i:ds*(i+1)], y0=y_seq[i], x1=x[ds*(i+1):ds*(i+2)]), ((0,0),(i*ds,0)))
        J = np.vstack(( J, Jadd ))
        M = block_diag(( M*gamma, inv(Q), inv(R[3:,3:]) ))

    if xend is not None :
        Jadd = np.pad(-model.F(x=xend, omega_pre=y_seq[-1][0:3]), ((0,0),(ds*(num_y-1),0)))
        J = np.vstack(( J, Jadd ))
        M = block_diag(( M*gamma, inv(Q) ))

    L = np.linalg.cholesky(M).T
    return L@J
#endregion

def NLSF_uniform(P_inv, y_seq, Q, R, f, h, F, H, mode:str="quadratic", x0=None, **args) : 
    if "sumofsquares" in mode.lower() : 
        fun = SumOfSquares(f_fn=f, h_fn=h, F_fn=F, H_fn=H)
        params = [P_inv, y_seq, Q, R]
    elif "quadratic" in mode.lower() : 
        fun = Quadratic(f_fn=f, h_fn=h, F_fn=F, H_fn=H)
        params = [P_inv, y_seq, Q, R, args["x0_bar"]]

    if "gamma" in args.keys() : params.append(args["gamma"])
    if "end" in mode.lower() : params.append(args["xend"])

    ds = Q.shape[0]
    if x0 is None : 
        x0 = np.zeros((ds*(len(y_seq)+1), ))
    else : 
        while (len(x0) <= len(y_seq)) : 
            x0.append(f(x0[-1]))
        x0 = np.array(x0).reshape(-1)
    
    # 判断传入的参数是否包含上下界信息
    if "lb" in args.keys() :
        lb = args["lb"]
        lb0 = np.copy(lb)
    else :
        lb = np.full((ds,), -np.inf)
        lb0 = np.full((ds,), -np.inf)
    if "ub" in args.keys() :
        ub = args["ub"]
        ub0 = np.copy(ub)
    else :
        ub = np.full((ds,), np.inf)
        ub0 = np.full((ds,), np.inf)
    # 人为设置约束
    # lb = np.array((-20, -30, 0))
    # lb0 = np.array((-20, -30, 0))
    # ub = np.array((20, 30, 50))
    # ub0 = np.array((20, 30, 50))
    for _ in range(len(y_seq)) : 
        lb = np.hstack((lb, lb0))
        ub = np.hstack((ub, ub0))
    result = least_squares(fun.res_fun, x0, method='trf', jac=fun.jac_fun, args=params) # , max_nfev=30, bounds=(lb, ub)
    return result

class SumOfSquares() : 
    def __init__(self, f_fn, h_fn, F_fn, H_fn) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn

    def res_fun(self, x, LP, y_seq, Q, R, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))

        LQ = np.linalg.cholesky(inv(Q))
        LR = np.linalg.cholesky(inv(R))
        f = np.insert(x[:ds], 0, 1)[np.newaxis,:]
        L = LP[:] * np.sqrt(gamma)
        for i in range(num_obs) : 
            f = np.hstack((f, x[ds*(i+1):ds*(i+2)]-self.f_fn(x[ds*(i):ds*(i+1)])[np.newaxis,:], 
                              y_seq[i]-self.h_fn(x[ds*(i+1):ds*(i+2)])[np.newaxis,:]))
            L = block_diag((L, LQ, LR))
        
        if xend is not None : 
            f = np.hstack((f, (xend-self.f_fn(x[-ds:]))[np.newaxis,:]))
            L = block_diag((L, LQ))
        
        return (f@L).reshape(-1)

    def jac_fun(self, x, LP, y_seq, Q, R, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : np.vstack((np.hstack((-self.F_fn(x0), np.eye(ds))), np.pad(-self.H_fn(x1), ((0,0),(ds,0)))))

        LQ = np.linalg.cholesky(inv(Q))
        LR = np.linalg.cholesky(inv(R))
        J = np.pad(np.eye(ds), ((1,0),(0,0)))
        L = LP[:] * np.sqrt(gamma)
        for i in range(num_obs) : 
            J = np.pad(J, ((0,0), (0,ds)))
            Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
            J = np.vstack((J, Jadd))
            L = block_diag((L, LQ, LR))

        if xend is not None : 
            Jadd = np.pad(-self.F_fn(xend), ((0,0),(ds*num_obs,0)))
            J = np.vstack((J, Jadd))
            L = block_diag((L, LQ))

        return L.T@J

class Quadratic() : 
    def __init__(self, f_fn, h_fn, F_fn, H_fn) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn

    def res_fun(self, x, P_inv, y_seq, Q, R, x0_bar, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = x.size // (num_obs+1)

        f = np.array(x[:ds] - x0_bar)[np.newaxis,:]
        M = np.copy(P_inv)
        for i in range(num_obs) : 
            f = np.hstack((f, x[ds*(i+1):ds*(i+2)]-self.f_fn(x[ds*(i):ds*(i+1)])[np.newaxis,:], 
                              y_seq[i]-self.h_fn(x[ds*(i+1):ds*(i+2)])[np.newaxis,:]))
            M = block_diag((M * gamma, inv(Q), inv(R)))
        
        if xend is not None : 
            f = np.hstack((f, (xend-self.f_fn(x[-ds:]))[np.newaxis,:]))
            M = block_diag((M * gamma, inv(Q)))

        # 如果M有全零行，把该行以及行号相同的列剔除并且剔除f中对应的元素
        non_zeros = ~np.all(M == 0, axis=1)
        M = M[non_zeros]
        M = M[:, non_zeros]
        f = f[:, non_zeros]
        
        L = np.linalg.cholesky(M)
        return (f@L).reshape(-1)

    def jac_fun(self, x, P_inv, y_seq, Q, R, x0_bar, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : np.vstack((np.hstack((-self.F_fn(x0), np.eye(ds))), np.pad(-self.H_fn(x1), ((0,0),(ds,0)))))

        J = np.eye(ds)
        M = P_inv[:]
        for i in range(num_obs) : 
            J = np.pad(J, ((0,0), (0,ds)))
            Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
            J = np.vstack((J, Jadd))
            M = block_diag((M * gamma, inv(Q), inv(R)))

        if xend is not None : 
            Jadd = np.pad(-self.F_fn(xend), ((0,0),(ds*num_obs,0)))
            J = np.vstack((J, Jadd))
            M = block_diag((M * gamma, inv(Q)))

        # 如果M有全零行，把该行以及行号相同的列剔除并且剔除J中对应的行
        non_zeros = ~np.all(M == 0, axis=1)
        M = M[non_zeros]
        M = M[:, non_zeros]
        J = J[non_zeros]

        L = np.linalg.cholesky(M).T
        return L@J


from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
class UKF(UnscentedKalmanFilter, Estimator):
    def __init__(self, dim_x, dim_z, dt, hx, fx, n=3, alpha=.1, beta=2., kappa=-1, sqrt_fn=None, x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None, x0_hat=None, P0_hat=None):
        points = MerweScaledSigmaPoints(n, alpha, beta, kappa)
        UnscentedKalmanFilter.__init__(self, dim_x, dim_z, dt, hx, fx, points, sqrt_fn, x_mean_fn, z_mean_fn, residual_x, residual_z)
        Estimator.__init__(self, name="UKF", x0_hat=x0_hat, P0_hat=P0_hat)

    def reset(self, x0_hat, P0_hat):
        self.x = x0_hat
        self.P = P0_hat

    def estimate(self, y, Q, R):
        self.Q = Q
        self.R = R
        self.predict()
        self.update(y)
        self.x_hat = self.x
        self.y_hat = self.hx(self.x_hat)
        self.P_hat = self.P
