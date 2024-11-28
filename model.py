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
    def __init__(self, sampleTime=0.005, N_sample=40000, q0=np.array([0.94563839, -0.03449911,  0.21051564,  0.24548119]), 
                 g=np.array((0,0,9.805185)), m=np.array((23.4820, 0, -40.8496))) -> None:
        super().__init__("Continuous2", 9, 9)
        self.modelErr = False
        self.sampleTime = sampleTime
        self.N_sample = N_sample
        self.q = q0
        self.g = g
        self.m = m
        self.n = 0 # 离散步
        self.t = 0 # 仿真时间, t = N*sampleTime

    def f(self, x, omega=None, omega_next=None, **args) :
        dt = self.sampleTime
        if omega is None : # 数据生成阶段，生成q_seq即可
            self.q_step(omega_pre=self.omega(self.t), omega=self.omega(self.t+dt))
            # 时间步进
            self.t += self.sampleTime
            return self.q
        x = self.F(x=x, omega=omega) @ x
        return x

    def q_step(self, omega_pre, omega=None):
        dt = self.sampleTime
        # 参考代码的三阶近似
        if omega is None: omega = omega_pre
        self.q = (np.eye(4) + 3/4 * self.Omega(omega) * dt - 1/4 * self.Omega(omega_pre) * dt \
            - 1/6 * np.linalg.norm(omega)**2 * dt**2 - 1/24 * self.Omega(omega) @ self.Omega(omega_pre) * dt**2 \
            - 1/48 * np.linalg.norm(omega)**2 * self.Omega(omega) * dt**3) @ self.q
        # 微分方程解
        # self.q = expm(0.5*self.Omega(omega_pre)*dt) @ self.q
        # 变号以及归一化
        if self.q[0] < 0:
            self.q = -self.q
        self.q = self.q / np.linalg.norm(self.q)

    def F(self, x, omega, **args) : 
        dt = self.sampleTime
        A = np.hstack(( -self.skewSymmetric(omega), -0.5*np.eye(3), np.zeros((3,3)) ))
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
            
            elif (17/20 * N <= n and n < N) :
                w =np.array((0, 0, 0))
        return w

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

    def quat2RotMat(self, q) :
        q = q / np.linalg.norm(q) # 四元数范数归一化
        q4, q1, q2, q3 = q
        C = np.array([
            [q1*q1 - q2*q2 - q3*q3 + q4*q4,   2 * (q1*q2 + q3*q4),              2 * (q1*q3 - q2*q4)],
            [2 * (q1*q2 - q3*q4),           - q1*q1 + q2*q2 - q3*q3 + q4*q4,    2 * (q2*q3 + q1*q4)],
            [2 * (q1*q3 + q2*q4),           2 * (q2*q3 - q1*q4),              - q1*q1 - q2*q2 + q3*q3 + q4*q4]
        ])
        return C

    def ext_y(self) :
        ext_yseq = [np.zeros((self.dim_obs,)) for _ in range(self.N_sample)]
        dt = self.sampleTime
        for i in range(int(79.5/dt), int(81.5/dt)): # 需要sampleTime能被0.5整除
            ext_yseq[i][3:6] = np.array((10 * (1 - abs(i*dt - 80.5)), 
                                        5 * (1 - abs(i*dt - 80.5)), 
                                        20 * (1 - abs(i*dt - 80.5)) ))
        for i in range(int(119.5/dt), int(121.5/dt)):
            ext_yseq[i][3:6] = np.array((0, 
                                        -7 * (1 - abs(i*dt - 120.5)), 
                                        0))
        for i in range(int(139.5/dt), int(141.5/dt)):
            ext_yseq[i][3:6] = np.array((-4 * (1 - abs(i*dt - 140.5)), 
                                        -3 * (1 - abs(i*dt - 140.5)), 
                                        8 * (1 - abs(i*dt - 140.5)) ))
        return ext_yseq

    def judgeQR(self, y, Q, R) :
        dt = self.sampleTime
        omega = y[:3]
        A = block_diag(( 0.5*self.Omega(omega*dt), np.zeros((6,6)) ))
        Q = Q*dt + 0.5*A@Q + 0.5*Q@A.T
        if abs(np.linalg.norm(y[3:6]) - np.linalg.norm(self.g)) > 0.2 : 
            R[3:6] = np.zeros((3,9))
        # else :
        #     Q[7:10] = np.hstack(( np.zeros((3,7)), 10*np.eye(3) ))
        return Q, R













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