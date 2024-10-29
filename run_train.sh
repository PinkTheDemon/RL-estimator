#!/bin/bash

# 定义参数列表
cov_list=(
    "1e-2"
    # "1e0"
    # "1e2"
)
gamma_list=(
    1.0
    # 0.99
    # 0.9
    # 0.8
)
hidden_list=(
    "([], 64, [])"
    # "([64], 64, [])"
    # "([], 64, [64])"
    # "([64], 64, [64])"
)
dropout_list=(
    0
    # 0.1
    # 0.2
    # 0.3
    # 0.4
)
numLayer_list=(
    1
    # 2
    # 3
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
                    python ./RNN_estimator29.py --cov "$cov" --gamma $gamma --hidden_layer "$hidden" --dropout $dropout --num_layer $numLayer
                done
            done
        done
    done
done