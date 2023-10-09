import numpy as np


class ReplayBuffer : 
    def __init__(self, maxsize:int, rand_num:int=111) -> None:
        self.maxsize = maxsize
        self.size = 0
        self.size_init = 0
        self.count = 0
        self.experience = []
        self.information = []
        self.experience_init = []
        self.information_init = []
        np.random.seed(rand_num)

    def push_init(self, exp, inf=None) : 
        self.experience_init.append(exp)
        self.information_init.append(inf)
        self.size_init += 1

    def push(self, exp, inf=None) : 
        if self.size < self.maxsize : 
            self.experience.append(exp)
            self.information.append(inf)
            self.size += 1
        else : 
            self.experience[self.count] = exp
            self.information[self.count] = inf
            self.count += 1
            self.count = int(self.count % self.maxsize)

    def sample(self, n:int) : 
        indices = np.random.randint(self.size+self.size_init, size=n)
        exp_list = []
        inf_list = []
        is_init = []
        for i in indices : 
            if i < self.size : 
                exp_list.append(self.experience[i])
                inf_list.append(self.information[i])
                is_init.append(False)
            else : 
                exp_list.append(self.experience_init[i-self.size])
                inf_list.append(self.information_init[i-self.size])
                is_init.append(True)

        return exp_list, inf_list, is_init
    
    def sample_seq(self, batch_size:int, num_steps:int) : 
        indices = np.random.randint(self.size + self.size_init - num_steps, size=batch_size)
        exp_list = []
        inf_list = []
        is_init = []
        for i in indices : 
            if i < self.size - num_steps : 
                exp_list += [self.experience[i:i+num_steps]]
                inf_list += [self.information[i:i+num_steps]]
                is_init.append(False)
            elif i < self.size : 
                exp_list += [self.experience[i: ] + self.experience[ :i+num_steps-self.size]]
                inf_list += [self.information[i: ] + self.information[ : i+num_steps-self.size]]
                is_init.append(False)
            else : 
                exp_list += [self.experience_init[i-self.size:i-self.size+num_steps]]
                inf_list += [self.information_init[i-self.size:i-self.size+num_steps]]
                is_init.append(True)
        return exp_list, inf_list, is_init

'''
参数         含义          数据类型    取值范围    说明
----------   -----------   ---------   ---------   ----------
maxsize      容量          int         >0          初始样本不算在容量中
#rand_num    随机数种子    int         >=0         默认值111 种子相同的buffer 采样过程将会是一致的
'''

'''
push_init
放入初始化样本 初始化样本在经验回放过程中不会丢弃 请人为确保初始样本的正确性
存入数量不限 也可以中途加入 但每次只存储一个样本 不支持list形式批量存储
--------------------------------------------------
输入    含义        数据类型    取值范围    说明
exp     经验样本    --          --          进来后将被放入list中存储 随意存入即可
#inf    其他信息    --          --          默认为None
'''

'''
push
存入样本 达到最大容量后再存入会丢弃最早的样本 存入的样本不会做任何修改 请人为确保样本的正确性
--------------------------------------------------
输入    含义        数据类型    取值范围    说明
exp     经验样本    --          --          进来后将被放入list中存储 随意存入即可
#inf    其他信息    --          --          默认为None
'''

'''
sample
采样样本 在现有样本和初始样本中均匀随机采样
--------------------------------------------------
输入    含义        数据类型    取值范围    说明
n       采样数量    int         >0          无
--------------------------------------------------
输出        含义                数据类型    取值范围    说明
exp_list    样本列表            list        --          无
inf_list    信息列表            list        --          无
is_init     是否来自初始样本    list        bool        位置与输入、输出列表对应
'''

'''
sample_seq
采样序列样本 在现有样本和初始样本中均匀随机采样
--------------------------------------------------
输入          含义        数据类型    取值范围    说明
batch_size    批量大小    int         >0          无
num_steps     序列长度    int         >0          无
--------------------------------------------------
输出        含义                数据类型    取值范围    说明
exp_list    样本列表            list        --          长度为batch_size
inf_list    信息列表            list        --          每个元素都是长度为num_steps的list
is_init     是否来自初始样本    list        bool        长度为batch_size
'''