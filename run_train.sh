#!/bin/bash

# 定义参数列表
cov_list=(
    # "1e-6"
    # "1e-4"
    "1e-2"
    # "1e0"
    # "1e2"
)
gamma_list=(
    # best ##########
    # 0.8（修改之前）(rnn25)
    # 1.0 (after) (rnn34)300集训练后加载网络再练300集，学习率依然是最小的学习率(38)
    # 1.0 # lr衰减因子0.5->0.75(39)
    # 1.0 # lr衰减因子0.5->0.75，继续训练但cov减小(41)
    # try ###########
    1.0
    # used #########
    # 0.99 (after)
    # 0.9 (after)
    # 0.8 (after)
    # 0.7 (after)
    # 0.6 (after)
    # 1.0 (before)
    # 0.99 (before)
    # 0.9 (before)
    # 0.8 (before)
)
hidden_list=(
    # 目前最优
    # "([], 64, [128])" # quad
    # "([], 128, [])" # SOS
    # 尝试
    "([], 128, [])" # C4
    # 尝试过
    # "([], 128, [128])" # C4
    # "([], 64, [64])" #C4
    # "([], 128, [128,128])" #C4
    # "([], 128, [128,128,128])" #C4
    # "([], 128, [128])" # SOS
    # "([128], 128, [])" # SOS
    # "([128], 128, [128])" # SOS
    # "([], 64, [64])"
    # "([], 128, [128,128])"
    # "([], 128, [128,128,128])"
    # "([], 128, [64])"
)
dropout_list=(
    # 0
    # 0.1
    0.2
    # 0.3
    0.4
)
numLayer_list=(
    # 1
    6
    5
    4
    3
    2
)
actFun_list=(
    # best#########
    # "elu" #1
    # try##########
    # "elu" #1.0
    # used#########
    # "elu" #1.2
    # "prelu"
    "relu"
    # "leaky_relu" #0.01,0.05
    # "tanh"
    # "sigmoid"
)

# 循环执行Python脚本
for cov in "${cov_list[@]}"
do
    for gamma in "${gamma_list[@]}"
    do 
        for hidden in "${hidden_list[@]}"
        do
            for dropout in "${dropout_list[@]}"
            do
                for numLayer in "${numLayer_list[@]}"
                do
                    for actFun in "${actFun_list[@]}"
                    do
                        python ./RNN_estimator_C4.py --cov "$cov" --gamma $gamma --hidden_layer "$hidden" --dropout $dropout --num_layer $numLayer --act_fun "$actFun"
                    done
                done
            done
        done
    done
done