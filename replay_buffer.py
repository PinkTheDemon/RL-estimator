import numpy as np


class ReplayBuffer : 
    def __init__(self, maxsize:int) -> None:
        self.maxsize = maxsize
        self.size = 0
        self.size_init = 0
        self.count = 0
        self.input = []
        self.output = []
        self.input_init = []
        self.output_init = []

    def push_init(self, input, output) : 
        self.input_init.append(input)
        self.output_init.append(output)
        self.size_init += 1

    def push(self, input, output) : 
        if self.size < self.maxsize : 
            self.input.append(input)
            self.output.append(output)
            self.size += 1
        else : 
            self.input[self.count] = input
            self.output[self.count] = output
            self.count += 1
            self.count = int(self.count % self.maxsize)

    def sample(self, n:int) : 
        indices = np.random.randint(self.size+self.size_init, size=n)
        in_list = []
        ot_list = []
        is_init = []
        for i in indices : 
            if i < self.size : 
                in_list.append(self.input[i])
                ot_list.append(self.output[i])
                is_init.append(False)
            else : 
                in_list.append(self.input_init[i-self.size])
                ot_list.append(self.output_init[i-self.size])
                is_init.append(True)

        return in_list, ot_list, is_init
    
    def sample_seq(self, batch_size:int, num_steps:int) : 
        indices = np.random.randint(self.size + self.size_init - num_steps, size=batch_size)
        in_list = []
        ot_list = []
        is_init = []
        for i in indices : 
            if i < self.size - num_steps : 
                in_list += [self.input[i:i+num_steps]]
                ot_list += [self.output[i:i+num_steps]]
                is_init += False
            elif i < self.size : 
                in_list += [self.input[i: ] + self.input[ :i+num_steps-self.size]]
                ot_list += [self.output[i: ] + self.output[ : i+num_steps-self.size]]
                is_init += False
            else : 
                in_list += [self.input_init[i-self.size:i-self.size+num_steps]]
                ot_list += [self.output_init[i-self.size:i-self.size+num_steps]]
                is_init += True
        return in_list, ot_list, is_init

'''
参数       含义    数据类型    取值范围    说明
--------   -----   ---------   ---------   ----------
maxsize    容量    int         >0          初始样本不算在容量中
'''

'''
push_init
放入初始化样本 初始化样本在经验回放过程中不会丢弃 请人为确保初始样本的正确性
存入数量不限 也可以中途加入 但每次只存储一个样本 不支持list形式批量存储
--------------------------------------------------
输入      含义    数据类型    取值范围    说明
input     输入    --          --          进来后将被放入list中存储 随意存入即可
output    输出    --          --          同input
'''

'''
push
存入样本 达到最大容量后再存入会丢弃最早的样本 存入的样本不会做任何修改 请人为确保样本的正确性
--------------------------------------------------
输入      含义    数据类型    取值范围    说明
input     输入    --          --          进来后将被放入list中存储 随意存入即可
output    输出    --          --          同input
'''

'''
sample
采样样本 在现有样本和初始样本中均匀随机采样
--------------------------------------------------
输入    含义        数据类型    取值范围    说明
n       采样数量    int         >0          无
--------------------------------------------------
输出       含义                数据类型    取值范围    说明
in_list    输入列表            list        --          无
ot_list    输出列表            list        --          无
is_init    是否来自初始样本    list        bool        位置与输入、输出列表对应
'''

'''
sample_seq
采样序列样本 在现有样本和初始样本中均匀随机采样
--------------------------------------------------
输入          含义        数据类型    取值范围    说明
batch_size    批量大小    int         >0          无
num_steps     序列长度    int         >0          无
--------------------------------------------------
输出       含义                数据类型    取值范围    说明
in_list    输入列表            list        --          长度为batch_size
ot_list    输出列表            list        --          每个元素都是长度为num_steps的list
is_init    是否来自初始样本    list        bool        长度为batch_size
'''