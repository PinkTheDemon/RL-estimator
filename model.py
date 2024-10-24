import numpy as np
from scipy.integrate import solve_ivp

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
        y = x[0] - 3*x[1]
        return y

    def H(self, x, **args) : 
        return np.array([[1, -3]])

# 洛伦兹吸引子
class Continuous1(Model):
    def __init__(self, sampleTime=0.05) -> None:
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

# 对外接口
def getModel(modelName) :
    if modelName == "Discrete1": return Discrete1()
    if modelName == "Continuous1": return Continuous1()