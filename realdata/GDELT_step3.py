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


#参数设置
missList = [0.7]

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

from sklearn.metrics.pairwise import cosine_similarity
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


# 这里是为了画图比较
finalList1 = []
finalList22 = []
finalList2 = []
finalListTNCP=[]
finalListfal = []



for miss in missList:
    aux = [np.diag(np.ones(siz[0])), np.diag(np.ones(siz[1])), np.diag(np.ones(siz[2]))]
    RE2 = []
    RE22 = []
    for dup in range(duplicate):
        np.random.seed(dup*4)
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

        # 这部分引入了更新辅助矩阵的算法
        simerror = 1
        Iter = 1
        while (simerror > 1e-2 and Iter < 10):
            self2 = PoissonAirCP(data, omega=Omega, rank=com_rank, max_iter=3000, tol=1e-5,
                                 OnlyObs=True, TrueValue=true_data, sim_mats=aux, alpha=para_alpha, lmbda=para_lmbda)
            self2.run()
            temp_aux = cons_similarity(self2.X.data)
            simerror = np.max((np.linalg.norm(aux[0] - temp_aux[0]),
                               np.linalg.norm(aux[1] - temp_aux[1]),
                               np.linalg.norm(aux[2] - temp_aux[2])))
            aux = temp_aux
            Iter = Iter + 1

            print('ExpAirCP loop with similarity error: {0}'.format(simerror))
            [EEr, EReEr1, EReEr2] = tenerror(self2.X, true_data, Omega)
            if Iter ==2:
                RE22.append(EReEr1)
            print(EReEr1)
        # 到这里为止
        [EErr, EReErr1, EReErr2] = tenerror(self2.X, true_data, Omega)
        print ('ExpAirCP Completion Error: {0}, {1}, {2}'.format(EErr, EReErr1, EReErr2))

        RE2.append(EReErr1)

    finalList22.append(np.mean(RE22))
    finalList2.append(np.mean(RE2))


for miss in missList:
    aux = [np.diag(np.ones(siz[0])), np.diag(np.ones(siz[1])), np.diag(np.ones(siz[2]))]
    RE1 = []
    RE11 = []
    for dup in range(duplicate):
        np.random.seed(dup*4)
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

        # 这部分引入了更新辅助矩阵的算法
        simerror = 1
        Iter = 1
        while (simerror > 1e-2 and Iter < 10):
            self = AirCP(data, omega=Omega, rank=com_rank, max_iter=3000, tol=1e-5, sim_mats=aux, alpha=para_alpha, lmbda=para_lmbda)
            self.run()
            temp_aux = cons_similarity(self.X.data)
            simerror = np.max((np.linalg.norm(aux[0] - temp_aux[0]),
                               np.linalg.norm(aux[1] - temp_aux[1]),
                               np.linalg.norm(aux[2] - temp_aux[2])))
            aux = temp_aux
            Iter = Iter + 1
            print('AirCP loop with similarity error: {0}'.format(simerror))
            [EEr, EReEr1, EReEr2] = tenerror(self.X, true_data, Omega)
            print(EReEr1)
        # 到这里为止

        #这里看对原始数据的补全准不准
        [Err, ReErr1, ReErr2] = tenerror(self.X, true_data, Omega)
        print ('AirCP Completion Error: {0}, {1}, {2}'.format(Err, ReErr1, ReErr2))
        RE1.append(ReErr1)
    finalList1.append(np.mean(RE1))


# for miss in missList:
#     RETNCP = []
#
#     for dup in range(duplicate):
#         np.random.seed(dup*4)
#         #每次都用同一份数据去做
#         data = dat.copy()
#         #观测值：丢失部分数据的
#         Omega = (np.random.random(siz) > miss) * 1
#         data[Omega == 0] -= data[Omega == 0]
#         data = pyten.tenclass.tensor.Tensor(data)
#
#         #补全时候用的rank
#         print('missing ratio: {0}'.format(miss))
#         #补全时候用的rank
#         com_rank = prespecifyrank
#         self3 = TNCP(data, Omega, rank=com_rank,alpha = para_alpha, lmbda=para_lmbda)
#         self3.run()
#         [EErrr, EReErrr1, EReErrr2] = tenerror(self3.X, true_data, Omega)
#         print ('TNCP Completion Error: {0}, {1}, {2}'.format(EErrr, EReErrr1, EReErrr2))
#         RETNCP.append(EReErrr1)
#     finalListTNCP.append(np.mean(RETNCP))
#
#
# #对于fal不受到rank改变的影响，所以单独写出来
# for miss in missList:
#     REfal = []
#     for dup in range(duplicate):
#         np.random.seed(dup*4)
#         #每次都用同一份数据去做
#         data = dat.copy()
#         #观测值：丢失部分数据的
#         Omega = (np.random.random(siz) > miss) * 1
#         data[Omega == 0] -= data[Omega == 0]
#         data = pyten.tenclass.tensor.Tensor(data)
#         print('missing ratio: {0}'.format(miss))
#         rX1 = falrtc(data, Omega, max_iter=100)
#         [Errfal, ReErrfal, ReErr2fal] = tenerror(rX1, true_data, Omega)
#         print ('falrtc Completion Error: {0}, {1}, {2}'.format(Errfal, ReErrfal, ReErr2fal))
#         REfal.append(ReErrfal)
#     finalListfal.append(np.mean(REfal))
#

print(finalList1)
print(finalList2)
print(finalListTNCP)
print(finalListfal)

result = [finalList1,finalList2,finalListTNCP]
result_name = 'prerank='+str(prespecifyrank)+'.csv'
#np.savetxt(result_name,result,fmt='%.4f',delimiter=',')
