import numpy as np
import copy
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
        M_inv = 0 if M.item() == 0 else 1/M
    elif np.linalg.matrix_rank(M) < M.shape[0]: # 不满秩方阵
        U, S, V = np.linalg.svd(M)
        S = np.diag([1/s if s != 0 else 0 for s in S])
        M_inv = V.T @ S @ U.T
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
    return dim_input*(dim_input+1) // 2

def do2ds(dim_output:int) : 
    return int(np.sqrt(2*(dim_output)))


def checkFilename(filename:str, suffix:str=None) -> str : 
    if suffix is None:
        baseName, extension = os.path.splitext(filename)
    else :
        baseName = filename
        extension = suffix
        filename = f'{baseName}{suffix}'
    index = 1
    while (os.path.exists(filename)) : 
        # 如果文件存在，自动更名
        filename = f'{baseName}({index}){extension}'
        index += 1
    if suffix is None :
        return filename
    else :
        baseName, _ = os.path.splitext(filename)
        return baseName

class LogFile() : 
    def __init__(self, fileName='output/log.txt', rename_option=False) -> None:
        # 保留现场
        
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

    def __del__(self) -> None:
        self.endLog()


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
            L[i:,i-1] = A_temp[i:,i-1]/(L[i-1,i-1]+1e-8)
            A_temp[i:,i:] = A_temp[i:,i:] - A_temp[i:,i-1].reshape(-1,1)@A_temp[i:,i-1].reshape(1,-1)/(A_temp[i-1,i-1]+1e-8)
            L[i,i] = np.sqrt(A_temp[i,i])

        # if L.shape[0] > 1 : 
        #     L[1:,0] = A_temp[1:,0]/L[0,0]
        #     A_temp = np.copy(A_temp)
        #     A_temp[1:,1:] = A_temp[1:,1:] - A_temp[1:, 0].reshape(-1,1)@A_temp[1:, 0].reshape(1,-1)/A_temp[0,0]
        #     L[1:,1:] = cholesky_unique(A_temp[1:, 1:])
        return L

def isConverge(matrices:list, criterion=None, tol:float = 1e-4, **kwargs):
    if criterion is None:
        candidates = matrices[:]
    else :
        candidates = [criterion(matrix, **kwargs) for matrix in matrices]
    candidate0 = candidates[0]
    for candidate in candidates :
        if (np.abs(candidate0 - candidate) < tol).all():
            continue
        else :
            return False
    return True

class RandomGenerator : 
    def __init__(self, randomFun:np.random, rand_num=111) -> None:
        self.fun = randomFun
        np.random.seed(rand_num)

    def getRandom(self, **args) : 
        result = self.fun(**args)
        return result

    def getRandomList(self, length, **args) : 
        randomList = []
        for _ in range(length) : 
            randomList.append(self.getRandom(**args))
        return randomList
    
def vectorize(M:np.array) -> np.array:
    '''
    对称矩阵化为向量,矩阵维度为n*n,向量长度为n(n+1)/2
    矩阵维度为m*n,记a=min(m,n),向量长度为m*n-a(a-1)/2
    不会影响M的值
    '''
    # 深拷贝
    matrix = copy.deepcopy(M)
    # 统一转成胖矩阵处理
    if matrix.shape[0] > matrix.shape[1] :
        matrix = matrix.T
    # 对称的部分加到一起
    if matrix.shape[0] > 1:
        indices = np.triu_indices(matrix.shape[0]-1)
        matrix[indices[0], indices[1]+1] *= 2
    # 取上三角矩阵的前几行索引号
    indices = np.triu_indices(matrix.shape[1])
    indices0 = indices[0][indices[0] < matrix.shape[0]]
    indices1 = indices[1][range(len(indices0))]
    # 取出对应的元素
    vec = matrix[indices0, indices1]
    return vec

def vec2mat(vec:np.array, m=None, n=None) -> np.array:
    '''
    向量化为对称矩阵,向量长度为n(n+1)/2,矩阵维度为n*n
    可能会改变向量vec的值
    '''
    # 判断需要的是胖矩阵还是瘦矩阵
    transpose = False
    if m is None:
        m = do2ds(vec.size)
        n = m
    else :
        if m > n :
            transpose = True
            m = m + n
            n = m - n
            m = m - n
    # 取上三角矩阵的前m行索引号
    indices = np.triu_indices(n)
    indices0 = indices[0][indices[0] < m]
    indices1 = indices[1][range(len(indices0))]
    # 构建m*n空矩阵并填入元素
    matrix = np.empty(shape=(m,n))
    matrix[indices0, indices1] = vec
    # 还原对称部分的元素
    if m > 1:
        indices = np.triu_indices(m-1)
        matrix[indices[0], indices[1]+1] /= 2
        matrix[indices[1]+1, indices[0]] = matrix[indices[0], indices[1]+1]
    return matrix.T if transpose else matrix

def EVD(M:np.ndarray, rank=None):
    '''实矩阵特征值分解'''
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # 筛去近0特征值（可能是由于数值计算误差产生的非常小的值）
    threshold = 1e-10
    valid_indices = np.where(np.abs(eigenvalues) > threshold)[0]
    eigenvalues = eigenvalues[valid_indices]
    # 特征值分解：M = U @ (E @ E.T) @ U.T
    U = eigenvectors[:, valid_indices]
    E = np.linalg.cholesky(np.diag(v=eigenvalues))
    L = U @ E
    if rank is not None:
        L = np.pad(L, ((0,rank-L.shape[0]),(0,0)))
    return L

def calMSE(x_batch, xhat_batch):
    x_batch = np.array(x_batch)
    xhat_batch = np.array(xhat_batch)
    SE = np.square(x_batch - xhat_batch)
    MSE = np.mean(np.mean(SE, axis=0), axis=0)
    RMSE = np.sqrt(np.mean(MSE))
    return MSE, RMSE

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