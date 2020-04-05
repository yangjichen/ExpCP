# coding=utf-8

import numpy as np
import scipy.optimize as sco
from scipy.special import loggamma

import pyten
from pyten.method.PoissonAirCP import PoissonAirCP
from pyten.method.Neg_binAirCP import Neg_binAirCP
from pyten.tools import tenerror
import util

siz = [20,20,20]
rank = [5,5,5]
missList = [0.2,0.5,0.8]

np.random.seed(23)
#failuresR = np.random.randint(1, high=100, size=siz)
failuresR = 100*np.ones(siz)
theta, simMat= util.create4(siz = siz, r = rank)

def obj_func(r,mulist,plist):
    f = np.sum(loggamma(r) - loggamma(r+mulist) - r*np.log(plist))
    return(f)
def mleR(data):
    '''
    用观测位置的数据估计参数r
    :param data:
    :return:
    '''
    sigmalist = np.array([np.var(data[i][data[i]!=0]) for i in range(siz[0])])
    mulist = np.array([np.mean(data[i][data[i]!=0]) for i in range(siz[0])])
    plist = mulist / sigmalist
    opt = sco.minimize(fun=obj_func, x0=0, args=(mulist,plist,),method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})
    print(opt.x)
    return(opt.x)

# 这里用theta生成数据矩阵
dat = np.zeros(siz)
for i in range(siz[0]):
    for j in range(siz[1]):
        for k in range(siz[2]):
            prob = np.exp(theta.data[i, j, k]) / (1 + np.exp(theta.data[i, j, k]))
            dat[i, j, k] = np.random.negative_binomial(failuresR[i,j,k], prob)

true_data = dat.copy()
true_data = pyten.tenclass.tensor.Tensor(true_data)




duplicate = 1
# 这里是为了画图比较
finalList1 = np.zeros((duplicate,len(missList)))
finalList2 = np.zeros((duplicate,len(missList)))
finalList3 = np.zeros((duplicate,len(missList)))
ind = 0

for miss in missList:
    RE1 = []
    RE2 = []
    RE3 = []
    for dup in range(duplicate):
        np.random.seed(dup*4)
        #每次都用同一份数据去做
        data = dat.copy()
        #观测值：丢失部分数据的
        Omega = (np.random.random(siz) > miss) * 1
        data[Omega == 0] -= data[Omega == 0]

        failuresR2 = mleR(data)*np.ones(siz)
        #补全时候用的rank
        print('missing ratio: {0}'.format(miss))
        com_rank = 5
        data = pyten.tenclass.tensor.Tensor(data)

        self2 = Neg_binAirCP(data, failuresR=failuresR,omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                           OnlyObs=True,TrueValue = true_data,sim_mats = simMat,alpha = [3,3,3], lmbda=3)

        self3 = Neg_binAirCP(data, failuresR=failuresR2,omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                           OnlyObs=True,TrueValue = true_data,sim_mats = simMat,alpha = [3,3,3], lmbda=3)

        self2.run()
        self3.run()

        #这里看对原始数据的补全准不准
        [EErr, EReErr1, EReErr2] = tenerror(self2.X, true_data, Omega)
        [EErr2, EReErr3, EReErr4] = tenerror(self3.X, true_data, Omega)

        RE2.append(EReErr1)
        RE3.append(EReErr3)

        # #
        # x = self1.X.data-true_data.data
        # y = self2.X.data-true_data.data
        # all_data=[x.reshape(np.prod(siz)),y.reshape(np.prod(siz))]
        #
        # figure,axes=plt.subplots() #得到画板、轴
        # axes.boxplot(all_data) #描点上色
        # plt.show()

    finalList2[:, ind] = RE2
    finalList3[:, ind] = RE3
    ind = ind + 1

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df1 = pd.DataFrame(finalList1.reshape(len(missList)*duplicate))
df1['method']='Poisson'
df1['MR'] = np.tile(missList, duplicate)

df2 = pd.DataFrame(finalList2.reshape(len(missList)*duplicate))
df2['method']='negBinTrue'
df2['MR'] = np.tile(missList, duplicate)

df3 = pd.DataFrame(finalList3.reshape(len(missList)*duplicate))
df3['method']='negBinR=1000'
df3['MR'] = np.tile(missList, duplicate)

df = df1.append(df2).append(df3)
df.columns = ['value', 'method', 'MR']
fig, axs = plt.subplots(1, 1)
sns.boxplot(data=df,x='MR',y='value',hue='method')
plt.title('NegBinData')
plt.show()
plt.savefig('Exp3(1).png')