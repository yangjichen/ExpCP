#coding=utf-8
import numpy as np
import pyten
from scipy import stats
from pyten.method.PoissonAirCP import PoissonAirCP
from pyten.method import AirCP
from pyten.tools import tenerror
from pyten.method import cp_als
from pyten.method import falrtc,TNCP
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#这部分想完成的事，不使用AuxInfomation的和使用AuxInfo的对比

#参数设置
missList = np.around(np.arange(0.1,0.9,0.1),2)
duplicate=1
prespecifyrank = 5
para_alpha = [1,1,1]
para_lmbda = 1
def normalize(mat):
    '''
    将矩阵每一列都标准化，不然在计算余弦相似度时都非常相近
    :param mat:
    :return:
    '''
    X_mean = mat.mean(axis=0)
    # standardize X
    X1 = (mat - X_mean)
    return(X1)

def cons_similarity(dat):
    siz = dat.shape
    temp = np.sum(dat, axis=1)

    tagvector = normalize(np.sum(dat, axis=1))
    cos_dist = 1 - cosine_similarity(tagvector)
    aux0 = np.exp(-(cos_dist**2))

    # 2时间相似性用AR(1)模型的acf去做
    from statsmodels.tsa.arima_model import ARMA
    ts = np.sum(np.sum(dat, axis=0),axis = 1)
    order = (1,0)
    tempModel = ARMA(ts,order).fit()
    rho = np.abs(tempModel.arparams)
    aux1 = np.diag(np.ones(siz[1]))
    for nn in range(1, siz[1]):
        aux1 = aux1 + np.diag(np.ones(siz[1] - nn), -nn) * rho ** nn + np.diag(np.ones(siz[1] - nn), nn) * rho ** nn

    # 3话题之间相关性
    aux2 = np.diag(np.ones(siz[2]))
    Pl = np.sum(temp, axis=1) / np.sum(temp)
    for i in range(siz[2]):
        for j in range(siz[2]):
            aux2[i,j] = np.exp(-np.sum((((temp[:, i] - temp[:, j]) / np.max(temp, 1)) ** 2) * Pl))
    aux = [aux0, aux1, aux2]
    return (aux)


# def cons_similarity(dat):
#     '''
#     根据现有数据构造相似性矩阵
#     :param dat:
#     :return:
#     '''
#     siz = dat.shape
#     # 1国家之间相关性通过构造20维的话题向量来计算余项相似度
#     tagvector = normalize(np.sum(dat, axis=1))
#     aux0 = 0.5+0.5*cosine_similarity(tagvector)
#
#     # 2时间相似性用AR(1)模型的acf去做
#     from statsmodels.tsa.arima_model import ARMA
#     ts = np.sum(np.sum(dat, axis=0),axis = 1)
#     order = (1,0)
#     tempModel = ARMA(ts,order).fit()
#     rho = tempModel.arparams
#     aux1 = np.diag(np.ones(siz[1]))
#     for nn in range(1, siz[1]):
#         aux1 = aux1 + np.diag(np.ones(siz[1] - nn), -nn) * rho ** nn + np.diag(np.ones(siz[1] - nn), nn) * rho ** nn
#     aux1 = 0.5+0.5*aux1
#
#     #3话题之间使用协方差矩阵
#     aux2 = np.corrcoef(np.sum(dat, axis=1).T)
#     aux2 = 0.5+0.5*aux2
#
#     aux = [aux0,aux1,aux2]
#     return(aux)










def convertMon(mat):
    '''
    将数据从daily_data转化为monthly_data
    :param mat:
    :return:
    '''
    monthdat = []
    month = range(0, 365, 30)
    for i in range(12):
        monthdat.append(np.sum(mat[:, month[i]:month[i + 1]], axis=1))
    monthdat = np.array(monthdat)
    monthdat = monthdat.transpose((1, 0, 2))
    return(monthdat)




dat =np.load('newbuild_tensor.npy')
#预处理，先筛选一次国家，0太多的的不纳入考虑，只剩下235->195个
idx = np.sum(np.sum(dat ==0,axis = 1),axis=1)>1000
dat = dat[idx]

#可供选择的调整方法，整理成月数据
dat = convertMon(dat)

siz = dat.shape
true_data = dat.copy()
true_data = pyten.tenclass.tensor.Tensor(true_data)


# list1是使用辅助信息，list2是不使用
finalList1 = []
finalList2 = []

for miss in missList:
    aux = [np.diag(np.ones(siz[0])), np.diag(np.ones(siz[1])), np.diag(np.ones(siz[2]))]
    RE1 = []
    RE2 = []
    for dup in range(duplicate):
        np.random.seed(823)
        #每次都用同一份数据去做
        data = dat.copy()
        #观测值：丢失部分数据的
        Omega = (np.random.random(siz) > miss) * 1
        data[Omega == 0] -= data[Omega == 0]
        data = pyten.tenclass.tensor.Tensor(data)

        #补全时候用的rank
        print('missing ratio: {0}'.format(miss))
        #补全时候用的rank
        com_rank = prespecifyrank

        simerror = 1
        Iter = 1
        while (simerror > 1e-2 and Iter < 10):
            self1 = PoissonAirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                                 OnlyObs=True ,TrueValue = true_data,sim_mats=aux,alpha = para_alpha,lmbda=para_lmbda)
            self1.run()
            temp_aux = cons_similarity(self1.X.data)
            simerror = np.max((np.linalg.norm(aux[0] - temp_aux[0]),
                        np.linalg.norm(aux[1] - temp_aux[1]),
                        np.linalg.norm(aux[2] - temp_aux[2])))
            aux = temp_aux
            Iter = Iter+1
            print('ExpAirCP loop with similarity error: {0}'.format(simerror))
            [EEr, EReEr1, EReEr2] = tenerror(self1.X, true_data, Omega)
            print(EReEr1)
#到这里为止

        self2 = PoissonAirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                             OnlyObs=True ,TrueValue = true_data,lmbda=para_lmbda)
        self2.run()

        [EEr, EReEr1, EReEr2] = tenerror(self1.X, true_data, Omega)
        [EErr, EReErr1, EReErr2] = tenerror(self2.X, true_data, Omega)
        print ('ExpAirCP with aux: {0}, {1}, {2}'.format(EEr, EReEr1, EReEr2))
        print ('ExpAirCP without aux : {0}, {1}, {2}'.format(EErr, EReErr1, EReErr2))
        RE1.append(EReEr1)
        RE2.append(EReErr1)

    finalList1.append(np.mean(RE1))
    finalList2.append(np.mean(RE2))

result = [finalList1,finalList2]
np.savetxt('Effects of Aux Info.csv',result,fmt='%.4f',delimiter=',')