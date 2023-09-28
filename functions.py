import numpy as np


# Pinv(n,n) to L(n,n) to output(1,n*(n+1)/2)
def P2o( P, h=None) : 
    # check if Pinv_next is semi-definit
    try : 
        L = np.linalg.cholesky(P)
    except np.linalg.LinAlgError : # 矩阵非正定
        return 'error'

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
