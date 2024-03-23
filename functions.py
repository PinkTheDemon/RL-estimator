import numpy as np
import sys
import os

# Pinv(n,n) to L(n,n) to output(1,n*(n+1)/2)
def P2o( P, h=None) : 
    # check if Pinv_next is semi-definit
    try : 
        L = np.linalg.cholesky(P)
    except np.linalg.LinAlgError : # 矩阵非正定
        return None

    out = []
    for i in range(L.shape[0]) : 
        out.extend(L[i, :i+1])
    out = np.array(out).flatten()
    if h is not None : out = np.append(out, h)
    return out


# transfer matrix list to one block-diag matrix
def block_diag(matrix_list) : 
    # 去掉空矩阵
    no_empty_list = (matrix for matrix in matrix_list if matrix.size != 0)

    bd_M = np.empty(shape=(0,0))
    for M in no_empty_list : 
        bd_M = np.block([[bd_M, np.zeros((bd_M.shape[0], M.shape[1]))],
                         [np.zeros((M.shape[0], bd_M.shape[1])), M]])
    return bd_M


# inverse of lower triangular matrix M
def inv(M:np.ndarray) : 
    if M.shape == (1,1) : 
        return 1/M 
    else :
        L = np.linalg.cholesky(M)
        L_inv = np.linalg.inv(L)
        M_inv = L_inv.transpose(*range(L_inv.ndim - 2), -1, -2) @ L_inv
        return M_inv


def delete_empty(M) : 
    delete_list = []
    for i in range(M.shape[0]) : 
        if M[i,i] == 0 : delete_list.append(i)
    M = np.delete(M, delete_list, axis=0)
    M = np.delete(M, delete_list, axis=1)
    return M, delete_list


# input dim_state and output dim_output
def ds2do(dim_input:int) : 
    lookup_table = {
        1: 2,
        2: 4,
        3: 7,
        4: 11,
        5: 16,
        6: 22,
        7: 29,
        8: 37,
        9: 46,
    }
    if dim_input in lookup_table : 
        return lookup_table[dim_input]
    else : 
        return None

def do2ds(dim_output:int) : 
    lookup_table = {
        2 : 1,
        4 : 2,
        7 : 3,
        11: 4,
        16: 5,
        22: 6,
        29: 7,
        37: 8,
        46: 9,
    }
    if dim_output in lookup_table : 
        return lookup_table[dim_output]
    else : 
        return None


def checkFilename(filename:str) -> str : 
    baseName, extension = os.path.splitext(filename)
    index = 1
    while (os.path.exists(filename)) : 
        # 如果文件存在，自动更名
        filename = f'{baseName}({index}){extension}'
        index += 1
    return filename


class LogFile() : 
    def __init__(self, fileName='output/log.txt', rename_option=False) -> None:
        # 文件已经存在则自动更名
        if rename_option : 
            baseName, extension = os.path.splitext(fileName)
            index = 1
            while (os.path.exists(fileName)) : 
                # 如果文件存在，自动更名
                fileName = f'{baseName}({index}){extension}'
                index += 1
        # ----------
        self.fileName = fileName
        sys.stdout = open(self.fileName, 'w')

    def flush(self) -> None : 
        sys.stdout.flush()

    # def addParam(self, args) -> None : 
    #     with open('param.py', 'r', encoding='utf-8') as source : 
    #         with open(self.fileName, 'a') as target : 
    #             target.write(source.read())

    def endLog(self) -> None : 
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        del self


def P3dtoP4d(x_bar, P3d, h3d=None) : 
    x_bar = x_bar[np.newaxis,:]
    P4d = np.array([[(x_bar@P3d@x_bar.T).item(), -(x_bar@P3d[:,0]).item(), -(x_bar@P3d[:,1]).item(), -(x_bar@P3d[:,2]).item()],
                    [-(P3d[0]@x_bar.T).item(),    P3d[0,0], P3d[0,1], P3d[0,2]],
                    [-(P3d[1]@x_bar.T).item(),    P3d[1,0], P3d[1,1], P3d[1,2]],
                    [-(P3d[2]@x_bar.T).item(),    P3d[2,0], P3d[2,1], P3d[2,2]]])
    if h3d is not None : P4d[0,0] += h3d
    return P4d

def cholesky_unique(A): # 半正定矩阵的cholesky分解之一（可能因为舍入误差出现nan）
        A_temp = np.copy(A)
        L = np.zeros_like(A_temp)
        L[0,0] = np.sqrt(A_temp[0,0])

        for i in range(1, L.shape[0]) : 
            L[i:,i-1] = A_temp[i:,i-1]/L[i-1,i-1]
            A_temp[i:,i:] = A_temp[i:,i:] - A_temp[i:,i-1].reshape(-1,1)@A_temp[i:,i-1].reshape(1,-1)/A_temp[i-1,i-1]
            L[i,i] = np.sqrt(A_temp[i,i])

        # if L.shape[0] > 1 : 
        #     L[1:,0] = A_temp[1:,0]/L[0,0]
        #     A_temp = np.copy(A_temp)
        #     A_temp[1:,1:] = A_temp[1:,1:] - A_temp[1:, 0].reshape(-1,1)@A_temp[1:, 0].reshape(1,-1)/A_temp[0,0]
        #     L[1:,1:] = cholesky_unique(A_temp[1:, 1:])
        return L

'''
P2o
将P矩阵和h转换成输出向量格式 先把P做cholesky分解 然后排成向量 在末尾加上h
--------------------------------------------------
输入    含义        数据类型    取值范围    说明
P       系数矩阵    ndarray     --          无
#h      常数项      float       --          默认无h项
--------------------------------------------------
输出    含义    数据类型    取值范围    说明
out     输出    ndarray     --          无
'''

'''
block_diag
生成块对角矩阵
--------------------------------------------------
输入           含义        数据类型    取值范围    说明
matrix_list    矩阵列表    list        --          列表中可以有空矩阵 会自动删除
--------------------------------------------------
输出    含义          数据类型    取值范围    说明
bd_M    块对角矩阵    ndarray     --          无
'''

'''
inv
矩阵求逆 由于np的函数不能求(1,1)矩阵的逆 因此做个整合 另外利用M的对称性 先做cholesky分解再对下三角求逆
--------------------------------------------------
输入    含义    数据类型    取值范围    说明
M       矩阵    ndarray     --          非正定的异常不做处理 直接报错
--------------------------------------------------
输出    含义        数据类型    取值范围    说明
--      矩阵的逆    ndarray     --          无
'''

'''
delete_empty
删除方阵中对角线元素为0的行和列
--------------------------------------------------
输入    含义    数据类型    取值范围    说明
M       方阵    ndarray     --          无
--------------------------------------------------
输入           含义            数据类型    取值范围    说明
M              方阵            ndarray     --          删除后的方阵
delete_list    删除的索引号    list        --          对应在原方阵中的索引
'''

'''
ds2do
用查找表做维度转换 
--------------------------------------------------
输入         含义        数据类型    取值范围    说明
dim_state    状态维度    int         1~9         查找表只做了1~9 应该足以应对大部分情况 如果有特殊的也可以往表里加
--------------------------------------------------
输出    含义        数据类型    取值范围    说明
--      输出维度    int         --          对应1~9的输入
'''

'''
do2ds
用查找表做维度转换 
--------------------------------------------------
输入          含义        数据类型    取值范围    说明
dim_output    输出维度    int         --          对应1~9的输入
--------------------------------------------------
输出    含义        数据类型    取值范围    说明
--      状态维度    int         1~9         无
'''