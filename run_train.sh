#!/bin/bash

# 定义参数列表
cov_list=(
    # "1e-1"
    "1e0"
    # "1e1"
)
gamma_list=(
    # 1.0
    # 0.99
    # 0.9
    0.8
)
hidden_list=(
    # 目前最优
    "([], 64, [128])" # quad
    # "([], 128, [])" # SOS
    # 尝试
    # 尝试过
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
    0.1
    # 0.2
    # 0.3
    # 0.4
)
numLayer_list=(
    # 1
    # 2
    3
)
actFun_list=(
    # "relu"
    "leaky_relu"
    "prelu"
    "elu"
    "tanh"
    "sigmoid"
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
                        python ./RNN_estimator29.py --cov "$cov" --gamma $gamma --hidden_layer "$hidden" --dropout $dropout --num_layer $numLayer --act_fun "$actFun"
                    done
                done
            done
        done
    done
done