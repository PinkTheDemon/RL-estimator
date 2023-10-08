import torch
from torch import nn 
from torch.optim import Adam 
import torch.nn.functional as F


class Actor(nn.Module) : 
    def __init__(self, dim_input:int, dim_output:int, h=[200], rand_num=111) -> None : 
        super(Actor, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        torch.manual_seed(rand_num)
        self.fc = nn.ModuleList()
        size_input = self.dim_input
        for size_hidden in h : 
            self.fc.append(nn.Linear(size_input, size_hidden))
            size_input = size_hidden
        self.fc.append(nn.Linear(h[-1], self.dim_output))

    def forward(self, input) : 
        input = torch.FloatTensor(input)
        output = self.fc[0](input)
        for fc in self.fc[1:] : 
            output = fc(F.relu(output))
        # output = torch.clamp(output, min=-1e3, max=1e3) # 对输出进行限幅
        output[-1] = F.relu(output[-1]) # 限定h为正
        return output

    def update_weight(self, bin, bot, lr=1e-3) : 
        bin = torch.FloatTensor(bin)
        bot = torch.FloatTensor(bot)
        output_batch = self.forward(bin)
        loss = F.mse_loss(output_batch, bot)
        optimizer = Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        grad_clipping(self, 10)
        optimizer.step()
        return loss


def grad_clipping(net, theta) : 
    if isinstance(net, nn.Module) : 
        params = [p for p in net.parameters() if p.requires_grad]
    else : 
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta : 
        for param in params : 
            param.grad[:] *= theta / norm



'''
参数          含义              数据类型    取值范围    说明
-----------   ---------------   ---------   ---------   ----------
dim_input     网络输入的维度    int         >0          无
dim_output    网络输出的维度    int         >0          无
h             网络隐藏层        int list    >0          默认单隐藏层200单元 不能输入空集
rand_num      随机数种子        int         >=0         神经网络随机初始化的种子号
'''

'''
forward
前向传播 使用时可以直接用类对象的名字 当然加上.forward也行
--------------------------------------------------
输入     含义        数据类型           取值范围    说明
input    网络输入    tensor或ndarray    --          也可以是批量输入 size_batch*dim
--------------------------------------------------
输出      含义        数据类型           取值范围    说明
output    网络输出    tensor或ndarray    --          无
'''

'''
update_weight
更新网络参数
--------------------------------------------------
输入    含义          数据类型           取值范围    说明
bin     批量输入      tensor或ndarray    --          size_batch*dim
bot     批量标签值    tensor或ndarray    --          无
#lr     学习率        float              0< <1       默认值1e-3
--------------------------------------------------
输出      含义        数据类型           取值范围    说明
output    网络输出    tensor或ndarray    --          无
'''

'''
grad_clipping
梯度裁剪
--------------------------------------------------
输入     含义                      数据类型    取值范围    说明
net      需要进行梯度裁剪的网络    神经网络    --          需要有parameter属性
theta    梯度限幅值                float       >0          限制梯度的范数小于theta 且不改变梯度的方向
'''