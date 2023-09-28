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

    def push_init(self, input, output) : # 初始样本人为确保正确性，就不判断正定了
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
        bin = []
        bot = []
        for i in indices : 
            if i < self.size : 
                bin.append(self.input[i])
                bot.append(self.output[i])
            else : 
                bin.append(self.input_init[i-self.size])
                bot.append(self.output_init[i-self.size])

        return bin, bot
