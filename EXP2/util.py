# coding=utf-8
import numpy as np
import pyten.tenclass

def create(siz,r, aux=None):
    """
    r: rank of core tensor
    size: shape of tensor
    """
    dims = len(siz)
    if aux is None:
        aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]

        #random.random生成的是0-1之间的均匀分布随机数
        epsilon = [np.random.random([r[n], 2]) for n in range(dims)]
        #print(epsilon)
        # Solution Decomposition Matrices
        tmp = []
        for n in range(dims):
            #tmp.append(np.array([range(1, siz[n] + 1), np.ones(siz[n])]).T)
            # modified 2019/7/20
            tmp.append(np.array([range(1, siz[n] + 1), np.ones(siz[n])]).T)
        u = [np.dot(tmp[n], epsilon[n].T) for n in range(dims)]
    else:
        # Solution Decomposition Matrices
        u = [np.random.multivariate_normal(np.zeros(siz[n]), aux[n], r[n]).T for n in range(dims)]
    syn_lambda = np.ones(r[0])
    sol = pyten.tenclass.Ktensor(syn_lambda, u)
    return(sol.totensor(),aux)


def create2(siz,r, aux=None):
    """
    拿weights加起来为1的两个均匀分布去构造
    r: rank of core tensor
    size: shape of tensor
    """
    dims = len(siz)
    rho = 0.9
    if aux is None:
        aux =[]
        for n in range(dims):
            auxn = np.zeros((siz[n],siz[n]))
            for nn in range(1,siz[n]):
                auxn = auxn + np.diag(np.ones(siz[n]-nn),-nn)*rho**nn+np.diag(np.ones(siz[n]-nn),nn)*rho**nn
            aux.append(auxn)
        #aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]
        #random.random生成的是0-1之间的均匀分布随机数
        epsilon = [np.random.uniform(0,2,size=[r[n], 2]) for n in range(dims)]

        # Solution Decomposition Matrices
        tmp = []
        for n in range(dims):
            #tmp.append(np.array([range(1, siz[n] + 1), np.ones(siz[n])]).T)
            # modified 2019/7/20
            tmp.append(np.array([np.arange(0,1,1.0/siz[n]), 1-np.arange(0,1,1.0/siz[n])]).T)
        u = [np.dot(tmp[n], epsilon[n].T)  for n in range(dims)]
        aux2 = [np.abs(np.corrcoef(u[n])) for n in range(dims)]


    else:
        # Solution Decomposition Matrices
        u = [np.random.multivariate_normal(np.zeros(siz[n]), aux[n], r[n]).T for n in range(dims)]
    syn_lambda = np.ones(r[0])
    sol = pyten.tenclass.Ktensor(syn_lambda, u)
    return(sol.totensor(),aux,aux2)



#x, y =create2([10,10,10],[4,4,4])

#为了Binomial准备的生成方式
def create3(siz,r, aux=None):
    """
    拿weights加起来为1的两个均匀分布去构造
    r: rank of core tensor
    size: shape of tensor
    """
    dims = len(siz)
    rho = 0.9
    if aux is None:
        # aux =[]
        # for n in range(dims):
        #     auxn = np.zeros((siz[n],siz[n]))
        #     for nn in range(1,siz[n]):
        #         auxn = auxn + np.diag(np.ones(siz[n]-nn),-nn)*rho**nn+np.diag(np.ones(siz[n]-nn),nn)*rho**nn
        #     aux.append(auxn)
        aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]
        #aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]
        #random.normal生成正态分布
        epsilon = [np.random.normal(0,2,size=[r[n], 2]) for n in range(dims)]

        # Solution Decomposition Matrices
        tmp = []
        for n in range(dims):
            #tmp.append(np.array([range(1, siz[n] + 1), np.ones(siz[n])]).T)
            # modified 2019/7/20
            tmp.append(np.array([np.arange(0,1,1.0/siz[n]), 1-np.arange(0,1,1.0/siz[n])]).T)

        u = [np.dot(tmp[n], epsilon[n].T)  for n in range(dims)]
    else:
        # Solution Decomposition Matrices
        u = [np.random.multivariate_normal(np.zeros(siz[n]), aux[n], r[n]).T for n in range(dims)]
    syn_lambda = np.ones(r[0])
    sol = pyten.tenclass.Ktensor(syn_lambda, u)
    return(sol.totensor(),aux)

#为了negative binomial准备的生成方式
def create4(siz,r, aux=None):
    """
    拿weights加起来为1的两个均匀分布去构造
    r: rank of core tensor
    size: shape of tensor
    """
    dims = len(siz)
    rho = 0.9
    if aux is None:
        # aux =[]
        # for n in range(dims):
        #     auxn = np.zeros((siz[n],siz[n]))
        #     for nn in range(1,siz[n]):
        #         auxn = auxn + np.diag(np.ones(siz[n]-nn),-nn)*rho**nn+np.diag(np.ones(siz[n]-nn),nn)*rho**nn
        #     aux.append(auxn)
        aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]
        #aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]
        #random.normal生成正态分布
        epsilon = [np.random.uniform(-1,0,size=[r[n], 2]) for n in range(dims)]

        # Solution Decomposition Matrices
        tmp = []
        for n in range(dims):
            #tmp.append(np.array([range(1, siz[n] + 1), np.ones(siz[n])]).T)
            # modified 2019/7/20
            tmp.append(np.array([np.arange(0,1,1.0/siz[n]), 1-np.arange(0,1,1.0/siz[n])]).T)

        u = [np.dot(tmp[n], epsilon[n].T)  for n in range(dims)]
    else:
        # Solution Decomposition Matrices
        u = [np.random.multivariate_normal(np.zeros(siz[n]), aux[n], r[n]).T for n in range(dims)]
    syn_lambda = np.ones(r[0])
    sol = pyten.tenclass.Ktensor(syn_lambda, u)
    return(sol.totensor(),aux)