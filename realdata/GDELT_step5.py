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
#这部分想完成的事，Effects of Regularization Parameters，
# 测试不同的正则化参数，测试alpha=【】，lambda=【】，二维结果就够了
#参数设置
miss = [0.5]
duplicate=1
prespecifyrank = 5
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




finalList1 = []
paralist = [0.001,0.01,0.1,1,10]

result = np.zeros((len(paralist), len(paralist),len(paralist)))
for i in range(len(paralist)):
    for j in range(len(paralist)):
        for k in range(len(paralist)):
            RE1 = []
            para_alpha = [paralist[i],paralist[j],paralist[k]]
            aux = [np.diag(np.ones(siz[0])), np.diag(np.ones(siz[1])), np.diag(np.ones(siz[2]))]
            for dup in range(duplicate):
                np.random.seed(dup*4)
                #每次都用同一份数据去做
                data = dat.copy()
                #观测值：丢失部分数据的
                Omega = (np.random.random(siz) > miss) * 1
                data[Omega == 0] -= data[Omega == 0]
                data = pyten.tenclass.tensor.Tensor(data)

                #补全时候用的rank
                print('parameter: {0}'.format((paralist[i],paralist[j],paralist[k])))
                #补全时候用的rank
                com_rank = prespecifyrank

                # 这部分引入了更新辅助矩阵的算法
                simerror = 1
                Iter = 1
                while (simerror > 1e-3 and Iter < 10):
                    self1 = PoissonAirCP(data, omega=Omega, rank=com_rank, max_iter=3000, tol=1e-5,
                                         OnlyObs=True, TrueValue=true_data, sim_mats=aux, alpha=para_alpha,
                                         lmbda=para_lmbda)
                    self1.run()
                    temp_aux = cons_similarity(self1.X.data)
                    simerror = np.max((np.linalg.norm(aux[0] - temp_aux[0]),
                                       np.linalg.norm(aux[1] - temp_aux[1]),
                                       np.linalg.norm(aux[2] - temp_aux[2])))
                    aux = temp_aux
                    Iter = Iter + 1
                    print(simerror)
                # 到这里为止

                [EEr, EReEr1, EReEr2] = tenerror(self1.X, true_data, Omega)
                print ('ExpAirCP with aux: {0}, {1}, {2}'.format(EEr, EReEr1, EReEr2))
                RE1.append(EReEr1)
            result[i,j,k] = np.mean(RE1)

print(result)
np.save('GDELTstep5.npy', result)