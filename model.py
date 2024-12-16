import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from functions import block_diag

class Model:
    def __init__(self, name, ds, do) -> None:
        self.name = name
        self.dim_state = ds
        self.dim_obs = do
        self.modelErr = False

    # 打印非函数成员变量
    def printAttr(self) -> None:
        model = vars(self)
        for key, val in model.items():
            print(f"{key}: {val}")

    # system dynamics
    def step(self, x, disturb=None, noise=None, isReal=False) : 
        f = self.f_real if isReal else self.f
        h = self.h_real if isReal else self.h
        x_next = f(x=x)
        if disturb is not None : x_next += disturb
        y_next = h(x=x_next)
        if noise is not None : y_next += noise
        return x_next, y_next

    #region 虚函数，子类具体实现，子类没有的话可以直接raise error
    def f(self):
        pass
    def h(self):
        pass
    def F(self):
        pass
    def H(self):
        pass
    # 注意，如果子类不实现real函数，那么在使用real函数时，需要传递关键字参数而非位置参数
    def f_real(self, **args):
        return self.f(**args)
    def h_real(self, **args):
        return self.h(**args)
    def F_real(self, **args):
        return self.F(**args)
    def H_real(self, **args):
        return self.H(**args)
    #endregion

class Discrete1(Model):
    def __init__(self) -> None:
        super().__init__("Discrete1", 2, 1)
        self.modelErr = False

    def f(self, x, **args) : 
        x_next = np.zeros_like(x)
        x_next[0] = 0.99*x[0] + 0.2*x[1]
        x_next[1] = -0.1*x[0] + 0.5*x[1]/(1+x[1]**2)
        return x_next

    def F(self, x, **args) : 
        return np.array([[.99, .2], 
                  [-.1, .5*(1-x[1]**2)/(1+x[1]**2)**2]])

    def h(self, x, **args) : 
        y = np.array((x[0] - 3*x[1], ))
        return y

    def H(self, x, **args) : 
        return np.array([[1, -3]])

# 洛伦兹吸引子
class Continuous1(Model):
    def __init__(self, sampleTime=0.1) -> None:
        super().__init__("Continuous1", 3, 2)
        self.modelErr = False
        self.sampleTime = sampleTime

    def f(self, x, **args) : 
        xdot = lambda t, y : ( np.array([
            10*(-y[0]+y[1]),
            28*y[0] - y[1] - y[0]*y[2],
            -8/3*y[2] + y[0]*y[1]
            ]) )
        x = solve_ivp(xdot, [0,0+self.sampleTime], x).y.T[-1]
        return x

    def F(self, x, **args) : 
        return np.eye(x.size) + self.sampleTime * \
                                np.array([[-10    , 10  , 0    ], 
                                        [28-x[2], -1  , -x[0]],
                                        [x[1]   , x[0], -8/3 ]])

    def h(self, x, **args) : 
        return np.array([np.sqrt(x[0]**2+x[1]**2+x[2]**2), x[0]])

    def H(self, x, **args) : 
        y = np.linalg.norm(x)
        return np.array([[x[0]/y, x[1]/y, x[2]/y], [1,0,0]])

# 四元数姿态估计
class Continuous2(Model):
    def __init__(self, sampleTime=0.02, N_sample=10000, q0=np.array([0.94563839, -0.03449911,  0.21051564,  0.24548119]), 
                 g=np.array((0,0,9.805185)), m=np.array((23.4820, 0, -40.8496))) -> None:
        super().__init__("Continuous2", 9, 9)
        self.modelErr = False
        self.sampleTime = sampleTime
        self.N_sample = N_sample
        self.q = q0
        self.g = g
        self.m = m
        self.n = 0 # 离散步
        self.t = 0 # 仿真时间, t = n*sampleTime

    def f(self, x, omega=None, omega_next=None, **args) :
        dt = self.sampleTime
        if omega is None : # 数据生成阶段，生成q_seq即可
            self.q_step(omega_pre=self.omega(self.t), omega=self.omega(self.t+dt))
            # 时间步进
            self.t += self.sampleTime
            return self.q
        x = self.F(x=x, omega=omega) @ x
        return x

    def q_step(self, omega_pre, omega=None, q=None):
        dt = self.sampleTime
        # 参考代码的三阶近似
        if omega is None: omega = omega_pre
        if q is None: 
            q = self.q
            ret = False
        else : ret = True
        q = (np.eye(4) + 3/4 * self.Omega(omega) * dt - 1/4 * self.Omega(omega_pre) * dt \
            - 1/6 * np.linalg.norm(omega)**2 * dt**2 - 1/24 * self.Omega(omega) @ self.Omega(omega_pre) * dt**2 \
            - 1/48 * np.linalg.norm(omega)**2 * self.Omega(omega) * dt**3) @ q
        # 微分方程解
        # self.q = expm(0.5*self.Omega(omega_pre)*dt) @ self.q
        # 变号以及归一化，变号可以不要了
        # if q[0] < 0:
        #     q = -q
        q = q / np.linalg.norm(q)
        if not ret :
            self.q = q
        else :
            return q

    def F(self, x, omega, **args) : 
        dt = self.sampleTime
        A = np.hstack(( -0*self.skewSymmetric(omega), -0.5*np.eye(3), np.zeros((3,3)) ))
        A = np.vstack(( A, np.zeros((6,9)) ))
        # 微分方程解
        # F = expm(A*dt)
        # 参考代码一阶近似
        F = np.eye(9) + A*dt + 0.5*A@A*dt**2
        return F

    def h(self, x, **args) : 
        if x.size == 4 : # 生成数据阶段
            x = np.zeros((9,))
        Cnbq = self.quat2RotMat(self.q)
        w = self.omega(t=self.t)
        a = Cnbq @ self.g + x[6:9]
        m = Cnbq @ self.m
        y = np.hstack((w, a, m))
        return y

    # def H(self, x, **args) : 
    #     Hw = np.hstack(( np.zeros((3,4)), np.eye(3), np.zeros((3,3)) ))
    #     qnorm = np.linalg.norm(self.q)
    #     Cnbq = self.quat2RotMat(self.q)
    #     q4, q1, q2, q3 = self.q
    #     Cnbq_prime = 1/(qnorm**2) * np.array([
    #         2*qnorm*np.array([
    #             [q1,  q2,  q3],
    #             [q2, -q1,  q4],
    #             [q3, -q4, -q1]
    #         ]) - q1*Cnbq,
    #         2*qnorm*np.array([
    #             [-q2,  q1, -q4],
    #             [ q1,  q2,  q3],
    #             [ q4,  q3, -q2]
    #         ]) - q2*Cnbq,
    #         2*qnorm*np.array([
    #             [-q3,  q4,  q1],
    #             [-q4, -q3,  q2],
    #             [ q1,  q2,  q3]
    #         ]) - q3*Cnbq,
    #         2*qnorm*np.array([
    #             [ q4,  q3, -q2],
    #             [-q3,  q4,  q1],
    #             [ q2, -q1,  q4]
    #         ]) - q4*Cnbq
    #     ])
    #     Ha = np.hstack(( (Cnbq_prime@self.g).T, np.zeros((3,3)), np.eye(3) ))
    #     Hm = np.hstack(( (Cnbq_prime@self.m).T, np.zeros((3,6)) ))
    #     return np.vstack((Hw, Ha, Hm))

    def H1(self) :
        return np.hstack(( 2*self.skewSymmetric(self.quat2RotMat(self.q)@self.g), np.zeros((3,3)), np.eye(3) ))
    
    def H2(self) :
        return np.hstack(( 2*self.skewSymmetric(self.quat2RotMat(self.q)@self.m), np.zeros((3,6)) ))

    def omega(self, t, rateMode="mult_ramp") : # 生成真实角速度数据
        N = self.N_sample
        n = int(t / self.sampleTime)
        if rateMode.lower() == "mult_ramp" :
            if (0 <= n and n < 5/20 * N) :
                w =np.array((0, 0, 0))
            
            elif (5/20 * N <= n and n < 7/20 * N) :
                w =np.array(((n - 5/20 * N + 2) / N * 20, 0, 0))
            elif (7/20 * N <= n and n < 8/20 * N) :
                w =np.array((0, 0, 0))
                
            elif (8/20 * N <= n and n < 10/20 * N) :
                w =np.array((0, (n - 8/20 * N + 2) / N * 10, 0))
            elif (10/20 * N <= n and n < 11/20 * N) :
                w =np.array((0, 0, 0))
                
            elif (11/20 * N <= n and n < 13/20 * N) :
                w =np.array((0, 0, (n - 11/20 * N + 2) / N * 10))
            elif (13/20 * N <= n and n < 14/20 * N) :
                w =np.array((0, 0, 0))
                
            elif (14/20 * N <= n and n < 17/20 * N) :
                w =np.array(((n - 8/20 * N + 2) / N * (10), (n - 8/20 * N + 2) / N * (-30), (n - 8/20 * N + 2) / N * (-20)))
            
            elif (17/20 * N <= n and n <= N) :
                w =np.array((0, 0, 0))
        return w

    def Omega(self, w) :
        return np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0],  0,  w[2], -w[1]],
            [w[1], -w[2],  0,  w[0]],
            [w[2],  w[1], -w[0],  0]
        ])

    def Ksi(self, q) :
        return np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]]
        ])

    def omega_hat_prime(self, q1, q2) :
        # omega_hat = inv(Ksi(q1) + Ksi(q2)) @ (q2 - q1)
        pass

    def rot(self, angle, axis:str) :
        '''angle: rad'''
        sa = np.sin(angle)
        ca = np.cos(angle)
        if axis.lower() == "x" :
            C_x = np.array([
                [1,   0,  0],
                [0,  ca, sa],
                [0, -sa, ca]
            ])
        elif axis.lower() == 'y' :
            C_x = np.array([
                [ca, 0, -sa],
                [0,  1,   0],
                [sa, 0,  ca]
            ])
        elif axis.lower() == 'z' :
            C_x = np.array([
                [ca,  sa, 0],
                [-sa, ca, 0],
                [0,    0, 1]
            ])
        else :
            raise ValueError("axis must be x/y/z !")
        return C_x

    def skewSymmetric(self, p:np.ndarray) :
        if p.size == 3: # p是角速度
            p1, p2, p3 = p
        elif p.size == 4: # p是四元数
            p1, p2, p3 = p[1:]
        return np.array([
            [ 0, -p3,  p2],
            [ p3, 0 , -p1],
            [-p2, p1,  0 ]
        ])

    def RotMat2quat(self, C) :
        #region check C
        if abs(1 - np.linalg.norm(C, ord=2)) > 1e-15 :
            raise Warning(f"C的二范数不为1:    norm(C) = {np.linalg.norm(C, ord=2)},    norm(C) - 1 = {np.linalg.norm(C, ord=2) -1}")
        if (abs(C) > 1).any() :
            raise Warning("C中的某个元素绝对值大于1")
        #endregion
        c11, c12, c13 = C[0]
        c21, c22, c23 = C[1]
        c31, c32, c33 = C[2]
        if (c33 < 0) :
            if (c11 > c22) :
                t = 1 + c11 - c22 - c33
                q = np.array((t, c12 + c21, c31 + c13, c23 - c32))
            else :
                t = 1 - c11 + c22 - c33
                q = np.array((c12 + c21, t, c23 + c32, c31 - c13))
        else :
            if (c11 < -c22) :
                t = 1 - c11 - c22 + c33
                q = np.array((c31 + c13, c23 + c32, t, c12 - c21))
            else :
                t = 1 + c11 + c22 + c33
                q =np.array((c23 - c32, c31 - c13, c12 - c21, t))
        # 归一化
        q = q * 0.5 / np.sqrt(t)
        # 确保标量项大于0
        if (q[3] < 0) :
            q = -q
        # 返回值
        q = np.array((q[3], q[0], q[1], q[2]))
        return q

    def quat2RotMat(self, q) : # Cnb(q)
        # q = q / np.linalg.norm(q) # 四元数范数归一化
        q4, q1, q2, q3 = q
        C = np.array([
            [q1*q1 - q2*q2 - q3*q3 + q4*q4,   2 * (q1*q2 + q3*q4),              2 * (q1*q3 - q2*q4)],
            [2 * (q1*q2 - q3*q4),           - q1*q1 + q2*q2 - q3*q3 + q4*q4,    2 * (q2*q3 + q1*q4)],
            [2 * (q1*q3 + q2*q4),           2 * (q2*q3 - q1*q4),              - q1*q1 - q2*q2 + q3*q3 + q4*q4]
        ])
        return C

    def Cnb_prime(self, q) :
        q4, q1, q2, q3 = q
        C_hat = 2*np.array([
            [
                [q4, q3, -q2],
                [-q3, q4, q1],
                [q2, -q1, q4]
            ],
            [
                [q1, q2, q3],
                [q2, -q1, q4],
                [q3, -q4, -q1]
            ],
            [
                [-q2, q1, -q4],
                [q1, q2, q3],
                [q4, q3, -q2]
            ],
            [
                [-q3, q4, q1],
                [-q4, -q3, q2],
                [q1, q2, q3],
            ]
        ])
        return C_hat

    def otimes(self, q1, q2) :
        return (q2[0]*np.eye(4) + self.Omega(w=q2[1:4]))@q1

    def ext_y(self) :
        N1 = int(0.4*self.N_sample)
        N2 = int(0.6*self.N_sample)
        N3 = int(0.7*self.N_sample)
        deltaN = int(0.0025*self.N_sample) # 需要N_sample是400的倍数
        ext_yseq = [np.zeros((self.dim_obs,)) for _ in range(self.N_sample)]
        dt = self.sampleTime
        for i in range(N1-deltaN, N1+3*deltaN):
            ext_yseq[i][3:6] = np.array((10 * (1 - abs(i*dt - 80.5)), 
                                        5 * (1 - abs(i*dt - 80.5)), 
                                        20 * (1 - abs(i*dt - 80.5)) ))
        for i in range(N2-deltaN, N2+3*deltaN):
            ext_yseq[i][3:6] = np.array((0, 
                                        -7 * (1 - abs(i*dt - 120.5)), 
                                        0))
        for i in range(N3-deltaN, N3+3*deltaN):
            ext_yseq[i][3:6] = np.array((-4 * (1 - abs(i*dt - 140.5)), 
                                        -3 * (1 - abs(i*dt - 140.5)), 
                                        8 * (1 - abs(i*dt - 140.5)) ))
        return ext_yseq

# 四元数姿态估计，标准MHE做法
class Continuous4(Model):
    def __init__(self, sampleTime=0.02, N_sample=10000, g=np.array((0,0,9.805185)), m=np.array((23.4820, 0, -40.8496))) -> None:
        super().__init__("Continuous4", 10, 9)
        self.modelErr = False
        self.g = g
        self.m = m
        self.bg = np.zeros((3,)) # np.array((-0.019, 0.013, -0.006))
        self.sampleTime = sampleTime
        self.N_sample = N_sample
        self.t = 0 # 仿真时间

    def f(self, x, omega_pre=None) :
        if omega_pre is None: # 生成数据阶段
            omega = self.omega(self.t)
            F = block_diag(( expm(0.5*self.Omega(omega)*self.sampleTime), np.eye(6) ))
            x = F @ x
            x[0:4] /= np.linalg.norm(x[0:4])
            # if x[0] < 0: 
            #     x[0:4] = -x[0:4]
            return x
        F = self.F(x=x, omega_pre=omega_pre)
        x = F @ x
        x[0:4] /= np.linalg.norm(x[0:4])
        # if x[0] < 0: 
        #     x[0:4] = -x[0:4]
        return x

    def F(self, x, omega_pre) : 
        dt = self.sampleTime
        F = np.hstack(( expm(0.5*self.Omega(omega_pre)*dt), np.zeros((4,6)) ))
        F = np.vstack(( F, np.hstack(( np.zeros((6,4)), np.eye(6) )) ))
        return F

    def h(self, x, **args) : 
        yg = self.omega(t=self.t)
        ya = self.quat2RotMat(q=x[0:4])@self.g + x[4:7]
        ym = self.quat2RotMat(q=x[0:4])@self.m + x[7:10]
        self.t += self.sampleTime
        return np.hstack(( yg, ya, ym ))

    def H(self, x, **args) : 
        Hg = np.zeros((3,10)) # np.hstack(( np.zeros((3,4)), np.eye(3), np.zeros((3,3)) ))
        Ha = np.hstack(( (self.Cnb_prime(q=x[0:4])@self.g).T, np.eye(3), np.zeros((3,3)) ))
        Hm = np.hstack(( (self.Cnb_prime(q=x[0:4])@self.m).T, np.zeros((3,3)), np.eye(3) ))
        return np.vstack(( Hg, Ha, Hm ))

    def Omega(self, w) :
        return np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0],  0,  w[2], -w[1]],
            [w[1], -w[2],  0,  w[0]],
            [w[2],  w[1], -w[0],  0]
        ])

    def rot(self, angle, axis:str) :
        '''angle: rad'''
        sa = np.sin(angle)
        ca = np.cos(angle)
        if axis.lower() == "x" :
            C_x = np.array([
                [1,   0,  0],
                [0,  ca, sa],
                [0, -sa, ca]
            ])
        elif axis.lower() == 'y' :
            C_x = np.array([
                [ca, 0, -sa],
                [0,  1,   0],
                [sa, 0,  ca]
            ])
        elif axis.lower() == 'z' :
            C_x = np.array([
                [ca,  sa, 0],
                [-sa, ca, 0],
                [0,    0, 1]
            ])
        else :
            raise ValueError("axis must be x/y/z !")
        return C_x

    def RotMat2quat(self, C) :
        #region check C
        if abs(1 - np.linalg.norm(C, ord=2)) > 1e-15 :
            raise Warning(f"C的二范数不为1:    norm(C) = {np.linalg.norm(C, ord=2)},    norm(C) - 1 = {np.linalg.norm(C, ord=2) -1}")
        if (abs(C) > 1).any() :
            raise Warning("C中的某个元素绝对值大于1")
        #endregion
        c11, c12, c13 = C[0]
        c21, c22, c23 = C[1]
        c31, c32, c33 = C[2]
        if (c33 < 0) :
            if (c11 > c22) :
                t = 1 + c11 - c22 - c33
                q = np.array((t, c12 + c21, c31 + c13, c23 - c32))
            else :
                t = 1 - c11 + c22 - c33
                q = np.array((c12 + c21, t, c23 + c32, c31 - c13))
        else :
            if (c11 < -c22) :
                t = 1 - c11 - c22 + c33
                q = np.array((c31 + c13, c23 + c32, t, c12 - c21))
            else :
                t = 1 + c11 + c22 + c33
                q =np.array((c23 - c32, c31 - c13, c12 - c21, t))
        # 归一化
        q = q * 0.5 / np.sqrt(t)
        # 确保标量项大于0
        if (q[3] < 0) :
            q = -q
        # 返回值
        q = np.array((q[3], q[0], q[1], q[2]))
        return q

    def quat2RotMat(self, q) : # Cnb(q)
        # q = q / np.linalg.norm(q) # 四元数范数归一化
        q4, q1, q2, q3 = q
        C = np.array([
            [q1*q1 - q2*q2 - q3*q3 + q4*q4,   2 * (q1*q2 + q3*q4),              2 * (q1*q3 - q2*q4)],
            [2 * (q1*q2 - q3*q4),           - q1*q1 + q2*q2 - q3*q3 + q4*q4,    2 * (q2*q3 + q1*q4)],
            [2 * (q1*q3 + q2*q4),           2 * (q2*q3 - q1*q4),              - q1*q1 - q2*q2 + q3*q3 + q4*q4]
        ])
        return C

    def Cnb_prime(self, q) : ## 注意确认正确性
        q4, q1, q2, q3 = q
        # q_norm = np.linalg.norm(q)
        # C_hat = -1/q_norm**3*np.kron(np.array((q1, q2, q3, q4)).reshape(4,1,1), self.quat2RotMat(q=q))
        # C_hat += 2/q_norm*np.array([
        C_hat = 2*np.array([
            [
                [q4, q3, -q2],
                [-q3, q4, q1],
                [q2, -q1, q4]
            ],
            [
                [q1, q2, q3],
                [q2, -q1, q4],
                [q3, -q4, -q1]
            ],
            [
                [-q2, q1, -q4],
                [q1, q2, q3],
                [q4, q3, -q2]
            ],
            [
                [-q3, q4, q1],
                [-q4, -q3, q2],
                [q1, q2, q3],
            ]
        ])
        return C_hat

    def omega(self, t, rateMode="mult_ramp") : # 生成真实角速度数据
        N = self.N_sample
        n = int(t / self.sampleTime)
        if rateMode.lower() == "mult_ramp" :
            if (0 <= n and n < 5/20 * N) :
                w =np.array((0, 0, 0))
            
            elif (5/20 * N <= n and n < 7/20 * N) :
                w =np.array(((n - 5/20 * N + 2) / N * 20, 0, 0))
            elif (7/20 * N <= n and n < 8/20 * N) :
                w =np.array((0, 0, 0))
                
            elif (8/20 * N <= n and n < 10/20 * N) :
                w =np.array((0, (n - 8/20 * N + 2) / N * 10, 0))
            elif (10/20 * N <= n and n < 11/20 * N) :
                w =np.array((0, 0, 0))
                
            elif (11/20 * N <= n and n < 13/20 * N) :
                w =np.array((0, 0, (n - 11/20 * N + 2) / N * 10))
            elif (13/20 * N <= n and n < 14/20 * N) :
                w =np.array((0, 0, 0))
                
            elif (14/20 * N <= n and n < 17/20 * N) :
                w =np.array(((n - 8/20 * N + 2) / N * (10), (n - 8/20 * N + 2) / N * (-30), (n - 8/20 * N + 2) / N * (-20)))
            
            elif (17/20 * N <= n and n <= N) :
                w =np.array((0, 0, 0))
        return w

    # def ext_y(self) :
    #     N1 = int(0.4*self.N_sample)
    #     N2 = int(0.6*self.N_sample)
    #     N3 = int(0.7*self.N_sample)
    #     deltaN = int(0.0025*self.N_sample) # 需要N_sample是400的倍数
    #     ext_yseq = [np.zeros((self.dim_obs,)) for _ in range(self.N_sample)]
    #     dt = self.sampleTime
    #     for i in range(N1-deltaN, N1+3*deltaN):
    #         ext_yseq[i][3:6] = np.array((10 * (1 - abs(i*dt - 80.5)), 
    #                                     5 * (1 - abs(i*dt - 80.5)), 
    #                                     20 * (1 - abs(i*dt - 80.5)) ))
    #     for i in range(N2-deltaN, N2+3*deltaN):
    #         ext_yseq[i][3:6] = np.array((0, 
    #                                     -7 * (1 - abs(i*dt - 120.5)), 
    #                                     0))
    #     for i in range(N3-deltaN, N3+3*deltaN):
    #         ext_yseq[i][3:6] = np.array((-4 * (1 - abs(i*dt - 140.5)), 
    #                                     -3 * (1 - abs(i*dt - 140.5)), 
    #                                     8 * (1 - abs(i*dt - 140.5)) ))
    #     return ext_yseq







# xy平面无人船
class Continuous3(Model):
    def __init__(self, sampleTime=0.1) -> None:
        super().__init__("Continuous3", 6, 8)
        self.modelErr = False
        self.sampleTime = sampleTime

    def f(self, x, **args) : 
        T = self.rotz(x[2])
        
        xdot = lambda t, y : ( np.array([
            10*(-y[0]+y[1]),
            28*y[0] - y[1] - y[0]*y[2],
            -8/3*y[2] + y[0]*y[1]
            ]) )
        x = solve_ivp(xdot, [0,0+self.sampleTime], x).y.T[-1]
        return x

    def F(self, x, **args) : 
        return np.eye(x.size) + self.sampleTime * \
                                np.array([[-10    , 10  , 0    ], 
                                        [28-x[2], -1  , -x[0]],
                                        [x[1]   , x[0], -8/3 ]])

    def h(self, x, **args) : 
        return np.array([np.sqrt(x[0]**2+x[1]**2+x[2]**2), x[0]])

    def H(self, x, **args) : 
        y = np.linalg.norm(x)
        return np.array([[x[0]/y, x[1]/y, x[2]/y], [1,0,0]])
    
    def rotz(self, angle) :
        '''
        机体系到惯性系
        angle: rad
        '''
        sa = np.sin(angle)
        ca = np.cos(angle)
        C_x = np.array([
            [ca, -sa, 0],
            [sa,  ca, 0],
            [0,    0, 1]
        ])
        return C_x

# 对外接口
def getModel(modelName) :
    if modelName == "Discrete1": return Discrete1()
    if modelName == "Continuous1": return Continuous1()
    if modelName == "Continuous2": return Continuous2()
    if modelName == "Continuous3": return Continuous3()
    if modelName == "Continuous4": return Continuous4()