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
        # tempM = inv(self.P_hat) + H.T@inv(R)@H
        # gamma = min(np.linalg.eigvals(tempM))*0.6
        # self.P_hat = inv(inv(self.P_hat) + H.T@inv(R)@H - gamma*np.eye(2))
        self.x_hat = self.x_hat - self.P_hat@H.T@inv(R)@(y_pre - y)
        self.y_hat = self.h_fn(x=self.x_hat)

    def estimate(self, y, Q, R, u=None):
        self.predict(Q=Q, u=u)
        self.update(y=y, R=R)

class EKFForQuat(Estimator) :
    def __init__(self, model:Continuous2, x0_hat=None, P0_hat=None) -> None:
        self.model = model
        super().__init__(name="EKFForQuat", x0_hat=x0_hat, P0_hat=P0_hat)

    def reset(self, x0_hat, P0_hat):
        self.model.q = None
        self.lamda = np.zeros((3,3))
        self.mu = np.zeros((3,3))
        self.r_a = np.zeros((3,3))
        if hasattr(self, "Q"):del self.Q
        if hasattr(self, "y_prev"):del self.y_prev
        super().reset(x0_hat, P0_hat)

    def estimate(self, y, Q, R, u=None):
        if not hasattr(self, "y_prev") :
            self.y_prev = y # y_pre，注意在reset中删去
            self.y_hat = y
            return np.zeros((9,))
        if self.model.q is None : # 计算四元数q的初始估计
            y_a = y[3:6]
            y_m = y[6:9]
            roll_accmag_0 = np.arctan2(y_a[1], y_a[2])
            pitch_accmag_0 = -np.arctan2(y_a[0], np.sqrt(y_a[1]**2 + y_a[2]**2))
            C_n_b__0 = self.model.rot(angle=0, axis="z") @ self.model.rot(angle=pitch_accmag_0, axis="Y") @ self.model.rot(angle=roll_accmag_0, axis="X")
            y_m_0_NEW = C_n_b__0.T @ y_m.reshape(-1,1)
            yaw_accmag_0 = -np.arctan2(y_m_0_NEW[1], y_m_0_NEW[0]).item()
            # q_m: Estimated-from-measurements quaternion (i.e., q from y_g (measured gyro output))
            self.model.q = self.model.RotMat2quat(self.model.rot(angle=yaw_accmag_0, axis="Z") @
                                                  self.model.rot(angle=pitch_accmag_0, axis="Y") @
                                                  self.model.rot(angle=roll_accmag_0, axis="X"))
        omega = y[:3] # 由于有了qe中间值，可以直接把yg当做角速度数据而无需减掉状态量中的gyro偏差
        dt = self.model.sampleTime
        # predict
        F = self.model.F(x=self.x_hat, omega=omega)
        self.model.q_step(omega_pre=self.y_prev[0:3], omega=omega)
        self.x_hat = self.model.f(x=self.x_hat, omega=omega)
        self.P_hat = F@self.P_hat@F.T
        A = np.vstack(( np.hstack(( -0*self.model.skewSymmetric(omega), -0.5*np.eye(3), np.zeros((3,3)) )), np.zeros((6,9)) ))
        if not hasattr(self, "Q") :
            self.Q = Q # 初始时刻Q，注意在reset中删去
        # self.Q = self.Q*dt + 0.5*A@self.Q + 0.5*self.Q@A.T
        if self.Q.size != 0 : self.P_hat += self.Q
        # correct(1)
        #0.
        Cnb = self.model.quat2RotMat(self.model.q)
        #1.
        H_a = self.model.H1() # np.hstack(( (self.model.Cnb_prime(q=np.insert(self.x_hat[0:3], 0, 1))[1:4]@(self.model.quat2RotMat(q=self.model.q)@self.model.g)).T, np.zeros((3,3)), np.eye(3) )) # 
        #2.个人注：实际上就是加速度偏差项
        z_a = y[3:6] - Cnb@self.model.g
        #3.Sab 影响性能
        # if abs(np.linalg.norm(y[3:6]) - np.linalg.norm(self.model.g)) < 0.25 : 
        #     Q_hat_a_b = np.zeros((3,3))
        # else :
        #     Q_hat_a_b = 10*np.eye(3)
        # Suh
        R_a = R[3:6,3:6]
        # Q_hat_a_b = self.estimateExtAccCov_Suh(H_a=H_a, R_a=R_a)
        # zero
        Q_hat_a_b = np.zeros((3,3))
        #4.
        S_a = H_a @ self.P_hat @ H_a.T + R_a + Q_hat_a_b
        #5.
        K_a = self.P_hat @ H_a.T @ inv(S_a)
        #6.
        r_a = z_a - H_a @ self.x_hat # y[3:6] - self.model.quat2RotMat(self.model.otimes(q1=self.model.q, q2=np.insert(self.x_hat[0:3], 0, 1)))@self.model.g + self.x_hat[6:9] # 
        self.r_a[1:3] = self.r_a[0:2]
        self.r_a[0] = r_a
        #7.
        self.x_hat = self.x_hat + K_a @ r_a
        #8.
        temp = np.eye(9) - K_a @ H_a
        self.P_hat = temp @ self.P_hat @ temp.T + K_a @ (R_a + Q_hat_a_b) @ K_a.T
        #9.
        self.model.q += self.model.Omega(self.x_hat[:3]) @ self.model.q
        self.model.q = self.model.q / np.linalg.norm(self.model.q) # 四元数范数归一化
        # if self.model.q[0] < 0: self.model.q = -self.model.q # 确保四元数的标量位大于0
        self.x_hat[:3] = np.zeros_like(self.x_hat[:3]) ## 个人注：为什么？这个必须要有，对性能影响很大
        # correct(2)
        #0.个人注：这里确实得再算一遍因为上面更新了q
        Cnb = self.model.quat2RotMat(self.model.q)
        #1.
        H_m = self.model.H2()
        #2.
        z_m = y[6:9] - Cnb@self.model.m
        #3. 个人注：P_m实际上没用，因为H_m的后6列本来就为0
        # P_m = block_diag(( self.P_hat[:3,:3], np.zeros((6,6)) ))
        #4.
        R_m = R[6:9,6:9]
        S_m = H_m @ self.P_hat @ H_m.T + R_m
        #5. r_3限制磁力计的修正仅作用于偏航分量，这个影响不是很大
        r_3 = Cnb[:,2].reshape(-1,1)#np.array((1,1,1))
        Suh = block_diag(( r_3@r_3.T, np.zeros((6,6)) ))
        K_m = Suh @ self.P_hat @ H_m.T @ inv(S_m)
        #6.
        r_m = z_m - H_m @ self.x_hat
        #7.
        self.x_hat = self.x_hat + K_m @ r_m
        #8.
        temp = np.eye(9) - K_m @ H_m
        self.P_hat = temp @ self.P_hat @ temp.T + K_m @ R_m @ K_m.T
        # Increment time istant
        self.y_prev = y
        return Q_hat_a_b.reshape(-1)

    def estimateExtAccCov_Suh(self, H_a, R_a):
        U = 0
        for r_a in self.r_a:
            r_a = r_a.reshape(-1,1)
            U += r_a @ r_a.T / 3
        
        self.lamda[1:3] = self.lamda[0:2]
        self.mu[1:3] = self.mu[0:2]

        vals, vecs = np.linalg.eigh(U)
        self.lamda[0] = vals
        u = vecs.T # 分解出来的是列向量，转成行向量处理

        M = H_a @ self.P_hat @ H_a.T + R_a
        for i in range(3):
            ui = u[i].reshape(-1,1)
            self.mu[0,i] = ui.T @ M @ ui
        
        Q_hat_a_b = np.zeros((3,3)) # 默认mode为1
        if max([max(err) for err in (self.lamda - self.mu)]) >= 0.1: # mdoe=2    0.1是gamma，可调参数
            Q_hat_a_b = 0
            for i in range(3):
                ui = u[i].reshape(-1,1)
                Q_hat_a_b += max(self.lamda[0,i]-self.mu[0,i], 0) * ui @ ui.T
        return Q_hat_a_b

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
        # if len(self.y_seq) == 0 : # y0用来计算四元数q的初始估计
        #     y_a = y[3:6]
        #     y_m = y[6:9]
        #     roll_accmag_0 = np.arctan2(y_a[1], y_a[2])
        #     pitch_accmag_0 = -np.arctan2(y_a[0], np.sqrt(y_a[1]**2 + y_a[2]**2))
        #     C_n_b__0 = self.model.rot(angle=0, axis="z") @ self.model.rot(angle=pitch_accmag_0, axis="Y") @ self.model.rot(angle=roll_accmag_0, axis="X")
        #     y_m_0_NEW = C_n_b__0.T @ y_m.reshape(-1,1)
        #     yaw_accmag_0 = -np.arctan2(y_m_0_NEW[1], y_m_0_NEW[0]).item()
        #     q = self.model.RotMat2quat(self.model.rot(angle=yaw_accmag_0, axis="Z") @
        #                                self.model.rot(angle=pitch_accmag_0, axis="Y") @
        #                                self.model.rot(angle=roll_accmag_0, axis="X"))
        #     self.y_hat = y
        #     self.y_seq.append(y)
        #     self.x_hat = q#np.hstack(( q, np.zeros((6,))))
        #     self.x0_bar_seq[0] = self.x_hat
        #     return 
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
            x1_pre = self.model.f(x=x0_hat, omega_pre=self.y_seq[0][0:3]) # 先不加u了很麻烦，有u的话再说吧
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
    result = least_squares(fun=resForC4, x0=x0, args=params, method='lm', jac=jacForC4, max_nfev=10) #
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

class MHEForQuat(Estimator):
    def __init__(self, model:Continuous2, window, x0_hat=None, P0_hat=None) -> None:
        self.model = model
        self.window = window
        super().__init__("MHEForQuat", x0_hat, P0_hat)

    def reset(self, x0_hat, P0_hat):
        self.Wm = np.eye(3)
        self.Wv = np.eye(3)
        self.Wu = 1e-6*np.eye(3) # 由于β的状态转移矩阵是单位阵，Wu就不变了
        self.x_hat = x0_hat
        self.P_hat = P0_hat
        self.y_seq = []
        self.x0_bar_seq = [x0_hat]
        if hasattr(self, "Q"):del self.Q
        if hasattr(self, "y_pre"):del self.y_prev
        self.model.q = None

    def estimate(self, y, W0=None, u=None, gamma=1.0, xend=None, Q=None, R=None):
        # 定义W0
        ds = 7
        W0 = np.eye(ds)
        self.Wm = R[6:9,6:9]
        self.Wv = R[0:3,0:3]
        #region 数据处理
        if self.model.q is None : # y0用来计算四元数q的初始估计
            y_a = y[3:6]
            y_m = y[6:9]
            roll_accmag_0 = np.arctan2(y_a[1], y_a[2])
            pitch_accmag_0 = -np.arctan2(y_a[0], np.sqrt(y_a[1]**2 + y_a[2]**2))
            C_n_b__0 = self.model.rot(angle=0, axis="z") @ self.model.rot(angle=pitch_accmag_0, axis="Y") @ self.model.rot(angle=roll_accmag_0, axis="X")
            y_m_0_NEW = C_n_b__0.T @ y_m.reshape(-1,1)
            yaw_accmag_0 = -np.arctan2(y_m_0_NEW[1], y_m_0_NEW[0]).item()
            # q_m: Estimated-from-measurements quaternion (i.e., q from y_g (measured gyro output))
            self.model.q = self.model.RotMat2quat(self.model.rot(angle=yaw_accmag_0, axis="Z") @
                                                  self.model.rot(angle=pitch_accmag_0, axis="Y") @
                                                  self.model.rot(angle=roll_accmag_0, axis="X"))
            self.y_hat = y
            self.y_seq.append(y)
            self.x_hat = np.hstack(( self.model.q, np.zeros((3,))))
            self.x0_bar_seq[0] = self.x_hat
            return 
        # 保存y到y_seq
        self.y_seq.append(y)
        if len(self.y_seq) > self.window+1 : del self.y_seq[0]
        #endregion
        if W0 is None : W0 = inv(self.P_hat)
        result = minForQuat(model=self.model, x0_bar=self.x0_bar_seq[0], y_seq=self.y_seq, W0=W0, Wm=self.Wm, Wv=self.Wv, Wu=self.Wu, gamma=gamma, xend=xend)
        self.x_hat = result.x[-ds:]
        self.x_hat[:4] /= np.linalg.norm(self.x_hat[:4])
        self.model.q = self.x_hat[:4]
        self.y_hat = None
        # 更新P: EKF
        if len(self.x0_bar_seq) == self.window:
            F = block_diag(( expm(self.model.Omega(w=self.y_seq[0][0:3])*self.model.sampleTime), np.eye(3)))
            self.P_hat = F @ self.P_hat @ F.T + Q[0:7,0:7]
            H = np.hstack(( (self.model.Cnb_prime(q=self.x0_bar_seq[0][0:4])@self.model.m).T, np.zeros((3,3)) ))
            self.P_hat = self.P_hat - self.P_hat @ H.T @ inv(R[6:9,6:9] + H@self.P_hat@H.T) @ H @ self.P_hat
        # 更新x0_bar_seq
        self.x0_bar_seq.append(self.x_hat)
        if len(self.x0_bar_seq) > self.window: del self.x0_bar_seq[0]

#region minForQuat
def minForQuat(model:Continuous2, x0_bar, y_seq, W0, Wm, Wv, Wu, x0=None, gamma=1.0, xend=None):
    # 生成初始值
    if x0 is None : 
        x0 = [x0_bar]
        for i in range(len(y_seq)-1) : 
            x0.append(np.hstack(( model.q_step(q=x0[i][0:4], omega_pre=y_seq[i][0:3], omega=y_seq[i+1][0:3]), x0[i][4:7] )))
    x0 = np.array(x0).reshape(-1)
    # 参数打包
    params = (model, x0_bar, y_seq, W0, Wm, Wv, Wu, gamma, xend)
    # 约束条件
    cons = {
        "type": "eq",
        "fun": consForQuat,
        "jac": consJacForQuat,
    }
    # 计算最小二乘问题
    result = minimize(fun=resForQuat, x0=x0, args=params, method='SLSQP', constraints=cons, options={"maxiter":10,}) # jac先不写了
    return result

def resForQuat(x, model:Continuous2, x0_bar, y_seq, W0, Wm, Wv, Wu, gamma=1.0, xend=None):
    num_y = len(y_seq)
    ds = 7

    f = (x[0:ds] - x0_bar)[np.newaxis,:]
    W = np.copy(W0)
    f1 = lambda x, y: (y[6:9] - model.quat2RotMat(x[:4])@model.m)[np.newaxis,:]
    f2 = lambda x1, x2, y: (y[0:3] - model.sampleTime*0.25*inv(model.Ksi(x1[:4]) + model.Ksi(x2[:4]))@(x2[:4]-x1[:4]) - x1[4:7])[np.newaxis,:]
    f3 = lambda x1, x2: (x2[4:7] - x1[4:7])[np.newaxis,:]

    for i in range(num_y-1) :
        f = np.hstack(( f, f1(x[(i+1)*ds:(i+2)*ds],y_seq[i+1]), 
                           f2(x[i*ds:(i+1)*ds],x[(i+1)*ds:(i+2)*ds],y_seq[i]), 
                           f3(x[i*ds:(i+1)*ds],x[(i+1)*ds:(i+2)*ds]) ))
        W = block_diag(( W*gamma, Wm, Wv, Wu ))

    if xend is not None:
        f = np.hstack(( f, f2(x[-ds:],xend,y_seq[-1]), f3(x[-ds:],xend) ))
        W = block_diag(( W*gamma, Wv, Wu ))

    # L = np.linalg.cholesky(W)
    return f@W@f.T

def jacForQuat(x, model:Continuous2, x0_bar, y_seq, W0, Wm, Wv, Wu, gamma=1.0, xend=None):
    num_y = len(y_seq)
    ds = 7

    J = np.eye(7)
    W = np.copy(W0)
    Jadd = lambda x1, x2 : np.vstack(( np.pad(model.Cnb_prime(x2)@model.m, pad_width=((0,0),(ds,0))),
                                       model.omega_hat_prime(x1, x2)))

    for i in range(num_y-1):
        J = np.pad(J, pad_width=((0,0),(0,ds)))

def consForQuat(x):
    ds = 7
    num_x = x.size // 7

    fc = np.empty((0,))
    for i in range(num_x):
        fc = np.hstack(( fc, np.linalg.norm(x[i*ds:(i+1)*ds])-1 ))
    return fc

def consJacForQuat(x):
    ds = 7
    num_x = x.size // 7

    Jac = np.empty((0, x.size))
    for i in range(num_x):
        Jac = np.vstack(( Jac, 2*np.pad(x[i*ds:(i+1)*ds][np.newaxis,:], ((0,0),(i*ds, (num_x-i-1)*ds))) ))
    return Jac
#endregion

#region not use
# def IEKF(x, P, y_next, Q, R, times=10):
#     # predict
#     F = dyn.F(x)
#     P_pre = F @ P @ F.T
#     if Q.size != 0 : P_pre = P_pre + Q
#     x_pre0 = dyn.f(x)
#     x_pre = x_pre0
#     # update
#     for _ in range(times):
#         H = dyn.H(x=x_pre)
#         y_pre = dyn.h(x=x_pre)
#         P_hat = inv(inv(P_pre) + H.T@inv(R)@H)
#         x_hat = x_pre0 - (P_hat@H.T@inv(R)@(y_pre - y_next).T).T
#         x_hat = np.squeeze(x_hat)
#         x_pre = x_hat
#     return x_hat, P_hat

# Unscented Kalman Filter
# def UKF(state, P, obs_next, Q, R, alpha=.5, beta=2., kappa=-5.) : 
#     n = state.size
#     nw = Q.shape[1]
#     nv = R.shape[1]
#     na = n + nw + nv
#     lamda = alpha**2 * (na + kappa) - na

#     # calculate sigma points and weights
#     xa = np.hstack((state, np.zeros((nw, )), np.zeros((nv, ))))
#     xa_sigma = np.tile(xa, (2*na+1, 1))
#     M = (na+lamda)*block_diag([P, Q, R])
#     M = np.linalg.cholesky(M)
#     xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
#     xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
#     xx_sigma = xa_sigma[:, :n]
#     xw_sigma = xa_sigma[:,n:n+nw]
#     xv_sigma = xa_sigma[:,n+nw: ]
#     Wc = np.ones((2*na+1, )) * 0.5 / (na+lamda)
#     Wm = np.ones((2*na+1, )) * 0.5 / (na+lamda)
#     Wc[0] = lamda / (na + lamda) + 1 - alpha**2 + beta
#     Wm[0] = lamda / (na + lamda)

#     # time update
#     x_next_pre = dyn.f(xx_sigma)
#     x_next_pre_aver = np.average(x_next_pre, weights=Wm, axis=0)
#     P_next_pre = np.zeros((n,n))
#     for i in range(2*na+1) : 
#         P_next_pre += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (x_next_pre[i] - x_next_pre_aver).reshape(1,-1)
#     P_next_pre += Q
    
#     # resample sigma points
#     xa = np.hstack((x_next_pre_aver, np.zeros((nw, )), np.zeros((nv, ))))
#     xa_sigma = np.tile(xa, (2*na+1, 1))
#     M = (na+lamda)*block_diag([P_next_pre, Q, R])
#     M = np.linalg.cholesky(M)
#     xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
#     xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
#     xx_sigma = xa_sigma[:, :n]
#     xw_sigma = xa_sigma[:,n:n+nw]
#     xv_sigma = xa_sigma[:,n+nw: ]

#     # measurement update ## 有一种是直接用上面的sigma点做y的预测的，还有一种是用上面算出来的x_pre_aver和P_pre重新选择sigma点做y预测的，下面先采用前者简单方式
#     y_next_pre = dyn.h(xx_sigma)
#     y_next_pre_aver = np.average(y_next_pre, weights=Wm, axis=0)
#     P_yy = np.zeros_like(R)
#     P_xy = np.zeros((n, nv))
#     for i in range(2*na+1) : 
#         P_yy += Wc[i] * (y_next_pre[i] - y_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
#         P_xy += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
#     P_yy += R
#     K = P_xy @ inv(P_yy)
#     x_next_hat = x_next_pre_aver + K @ (obs_next - y_next_pre_aver)
#     P_next_hat = P_next_pre - K @ P_yy @ K.T

#     return x_next_hat.reshape(-1), P_next_hat
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
    result = least_squares(fun.res_fun, x0, method='lm', jac=fun.jac_fun, args=params) # , max_nfev=8
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


# class ParticleFilter() : 
#     def __init__(self, state_dim:int, obs_dim:int, num_particles:int, fx, hx, threshold=None, rand_num=1111) -> None: #, x0_mu, P0
#         self.state_dim = state_dim
#         self.obs_dim   = obs_dim
#         self.N         = num_particles
#         self.fx        = fx
#         self.hx        = hx
#         self.threshold = self.N * 0.5 if threshold is None else threshold
#         np.random.seed(seed=rand_num)
#         self.name = "PF"
#         # self.reset(x0_mu, P0)
#         # self.create_gaussian_particles(x0_mu, P0, self.N)

#     def reset(self, x0_hat, P0_hat):
#         self.create_gaussian_particles(x0_hat, P0_hat, self.N)

#     def create_uniform_particles(self, state_dim, state_range, N) : 
#         self.particles = np.empty((N, state_dim))
#         for i in range(state_dim) : 
#             self.particles[:, i] = np.random.uniform(state_range[i][0], state_range[i][1], size=N)
#         self.weight = np.ones((N, ))/N

#     def create_gaussian_particles(self, mean, cov, N) : 
#         self.particles = np.random.multivariate_normal(mean, cov, N)
#         self.weight = np.ones((N, ))/N
    
#     def predict(self, noise_Q, noise_mu=None, dt=.1) : 
#         if noise_mu is None : noise_mu = np.zeros((self.state_dim, ))
#         process_noise = np.random.multivariate_normal(noise_mu, noise_Q, self.N)
#         for i in range(self.N):
#             self.particles[i] = self.fx(self.particles[i], dt) + process_noise[i]
#         # self.particles = self.fx(self.particles, process_noise, dt)

#     def update(self, observation, obs_noise_R) : 
#         for i in range(self.N) : 
#             # self.weight[i] *= multivariate_normal(self.hx(self.particles[i]), obs_noise_R).pdf(observation)
#             log_weight = -0.5 * ((observation - self.hx(self.particles[i])).reshape(1,-1) @ np.linalg.inv(obs_noise_R) @ (observation - self.hx(self.particles[i])).reshape(-1,1))
#             self.weight[i] *= np.exp(log_weight)
        
#         self.weight += 1.e-300          # avoid round-off to zero
#         self.weight /= sum(self.weight) # normalize

#     def estimate(self, y, Q, R) : 
#         self.predict(Q)
#         self.update(y, R)

#         state_hat = np.average(self.particles, weights=self.weight, axis=0)
#         Cov_hat = np.cov(self.particles, rowvar=False, aweights=self.weight)
#         Cov_hat = np.clip(Cov_hat, a_min=-1e5, a_max=1e5)  # 限制协方差的范围

#         if self.neff() <= self.threshold : 
#             print(f'resample, Wneff={self.neff()}')
#             self.simple_resample()
        
#         self.x_hat = state_hat
#         self.y_hat = self.hx(state_hat)
#         self.P_hat = Cov_hat
#         # return state_hat, Cov_hat
    
#     def simple_resample(self) : 
#         cumulative_sum = np.cumsum(self.weight)
#         cumulative_sum[-1] = 1.  # avoid round-off error
#         indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))

#         # resample according to indexes
#         self.particles = self.particles[indexes]
#         self.weight = np.ones((self.N, ))/self.N

#     def neff(self) : 
#         return 1. / np.sum(np.square(self.weight))
