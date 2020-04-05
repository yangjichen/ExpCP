# coding=utf-8
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import pyten
from scipy import stats
from pyten.method.Neg_binAirCP import Neg_binAirCP
from pyten.method import AirCP, TNCP,falrtc
from pyten.tools import tenerror
from pyten.method import cp_als
import util

import matplotlib.pyplot as plt


#调大这个降低relative error

siz = [20,20,20]
rank = [3,3,3]
#missList = np.around(np.arange(0.05,0.9,0.1),2)  # Missing Percentage
missList = [0.85]

#R的生成也要放在seed之后
np.random.seed(13)
failuresR = np.random.randint(1, high=100, size=siz)
theta, simMat= util.create4(siz = siz, r = rank)

# 这里用theta生成数据矩阵
dat = np.zeros(siz)
for i in range(siz[0]):
    for j in range(siz[1]):
        for k in range(siz[2]):
            prob = np.exp(theta.data[i, j, k]) / (1 + np.exp(theta.data[i, j, k]))
            dat[i, j, k] = np.random.negative_binomial(failuresR[i,j,k], prob)

# plt.hist((np.exp(theta.data) / (1 + np.exp(theta.data))).reshape(np.prod(siz)))
# # plt.hist(dat.reshape(np.prod(siz)))
# #plt.hist(theta.data.reshape(np.prod(siz)))
# plt.show()


# #保存原始数据，其他code也需要用
np.save('./negBinomial/negBinomialdata.npy',dat)
# np.save('./negBinomial/negBinomialdataETF.npy',dat.reshape(50*50*50))

true_data = dat.copy()
true_data = pyten.tenclass.tensor.Tensor(true_data)


duplicate = 1
# 这里是为了画图比较
finalList1 = np.zeros((duplicate,len(missList)))
finalList2 = np.zeros((duplicate,len(missList)))
finalListTNCP = np.zeros((duplicate,len(missList)))
finalListfal = np.zeros((duplicate,len(missList)))

ThetaErList = np.zeros((duplicate,len(missList)))

ind = 0

for miss in missList:
    RE1 = []
    RE2 = []
    RETNCP = []
    RETheta = []
    REfal = []
    for dup in range(duplicate):
        np.random.seed(dup*4)
        #每次都用同一份数据去做
        data = dat.copy()
        #观测值：丢失部分数据的
        Omega = (np.random.random(siz) > miss) * 1
        data[Omega == 0] -= data[Omega == 0]


        #补全时候用的rank
        print('missing ratio: {0}'.format(miss))
        com_rank = 3
        data = pyten.tenclass.tensor.Tensor(data)

        self1 = AirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,sim_mats = simMat,alpha = [3,3,3], lmbda=3)
        self2 = Neg_binAirCP(data, failuresR=failuresR, omega=Omega, rank=com_rank, max_iter=3000, tol=1e-5,
                             OnlyObs=True, TrueValue=true_data, sim_mats=simMat, alpha = [3,3,3], lmbda=3)
        self3 = TNCP(data, Omega, rank=com_rank,alpha = [3,3,3], lmbda=3)
        self1.run()
        self2.run()
        self3.run()

        #rX1 = falrtc(data, Omega, max_iter=100)



        #这里看对原始数据的补全准不准

        [Err, ReErr1, ReErr2] = tenerror(self1.X, true_data, Omega)
        [EErr, EReErr1, EReErr2] = tenerror(self2.X, true_data, Omega)
        [Errcp, ReErrcp, ReErr2cp] = tenerror(self3.X, true_data, Omega)
        #[Errfal, ReErrfal, ReErr2fal] = tenerror(rX1, true_data, Omega)



        print ('AirCP Completion Error: {0}, {1}, {2}'.format(Err, ReErr1, ReErr2))
        print ('ExpAirCP Completion Error: {0}, {1}, {2}'.format(EErr, EReErr1, EReErr2))
        print ('TNCP Completion Error: {0}, {1}, {2}'.format(Errcp, ReErrcp, ReErr2cp))
        #print ('falrtc Completion Error: {0}, {1}, {2}'.format(Errfal, ReErrfal, ReErr2fal))

        pred2 = pyten.tenclass.Ktensor(np.ones(com_rank), us=self2.U).totensor()
        print ('Theta Error for ExpAirCP: {0}'.format(np.linalg.norm(pred2.data-theta.data)/np.linalg.norm(theta.data)))

        RETNCP.append(ReErrcp)
        RE1.append(ReErr1)
        RE2.append(EReErr1)
        #REfal.append(ReErrfal)
        ThetaRE = np.linalg.norm(pred2.data - theta.data) / np.linalg.norm(theta.data)
        RETheta.append(ThetaRE)

        # #
        x = self1.X.data-true_data.data
        y = self2.X.data-true_data.data
        all_data=[x.reshape(np.prod(siz)),y.reshape(np.prod(siz))]

        figure,axes=plt.subplots() #得到画板、轴
        axes.boxplot(all_data) #描点上色
        plt.show()

    finalListTNCP[:, ind] = RETNCP
    finalList1[:, ind] = RE1
    finalList2[:, ind] = RE2
    ThetaErList[:, ind] = RETheta
#    finalListfal[:, ind] = REfal

    ind = ind + 1


print(np.mean(finalListTNCP,0))
print(np.mean(finalList1,0))
print(np.mean(finalListfal,0))
print(np.mean(finalList2,0))
print(np.mean(ThetaErList,0))

# plt.hist(pred2.data.reshape(np.prod(siz)))
# plt.show()



# np.savetxt('./negBinomial/TNCPresult.csv',finalListTNCP,delimiter=',')
# np.savetxt('./negBinomial/AirCPresult.csv',finalList1,delimiter=',')
# np.savetxt('./negBinomial/falrtcresult.csv',finalListfal,delimiter=',')
# np.savetxt('./negBinomial/ExpAirCPresult.csv',finalList2,delimiter=',')
# np.savetxt('./negBinomial/ThetaErList.csv',ThetaErList,delimiter=',')


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df1 = pd.DataFrame(finalList1.reshape(len(missList)*duplicate))
df1['method']='AirCP'
df1['MR'] = np.tile(missList, duplicate)

df2 = pd.DataFrame(finalList2.reshape(len(missList)*duplicate))
df2['method']='ExpAirCP'
df2['MR'] = np.tile(missList, duplicate)

df3 = pd.DataFrame(finalListTNCP.reshape(len(missList)*duplicate))
df3['method']='TNCP'
df3['MR'] = np.tile(missList, duplicate)

df4 = pd.DataFrame(finalListfal.reshape(len(missList)*duplicate))
df4['method']='falrtc'
df4['MR'] = np.tile(missList, duplicate)

df = df1.append(df3).append(df2)
df.columns = ['value', 'method', 'MR']
fig, axs = plt.subplots(1, 1)
sns.boxplot(data=df,x='MR',y='value',hue='method')
plt.show()
#
# plt.savefig('./negBinomial/NegBino_size50rank3.png')