# coding=utf-8
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import pyten
from scipy import stats
from pyten.method.BernoulliAirCP import BernoulliAirCP
from pyten.method import AirCP, TNCP,falrtc
from pyten.tools import tenerror
from pyten.method import cp_als
from util import create2, create3
import time

import matplotlib.pyplot as plt


#调大这个降低relative error
binomialn = 100
siz = [20,20,20]
rank = [3,3,3]
#missList = np.around(np.arange(0.15,0.9,0.1),2)  # Missing Percentage
missList = [0.85]
np.random.seed(203)
theta, simMat = create3(siz=siz, r=rank)

# 这里用theta生成数据矩阵
dat = np.zeros(siz)
for i in range(siz[0]):
    for j in range(siz[1]):
        for k in range(siz[2]):
            prob = np.exp(theta.data[i, j, k]) / (1 + np.exp(theta.data[i, j, k]))
            dat[i, j, k] = np.random.binomial(binomialn, prob)

# #保存原始数据，其他code也需要用
# np.save('./Binomial/Binomialdata.npy',dat)
# np.save('./Binomial/BinomialdataETF.npy',dat.reshape(50*50*50))
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
        self2 = BernoulliAirCP(data, binomialn = binomialn,omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                           OnlyObs=True,TrueValue = true_data,sim_mats = simMat,alpha = [3,3,3], lmbda=3)
        self3 = TNCP(data, Omega, rank=com_rank, alpha = [3,3,3], lmbda=3)

        #rX1 = falrtc(data, Omega, max_iter=100)

        self1.run()
        self2.run()
        self3.run()


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

        #
        # x = self2.X.data-true_data.data
        # y = self3.X.data-true_data.data
        # all_data=[x.reshape(np.prod(siz)),y.reshape(np.prod(siz))]
        #
        # figure,axes=plt.subplots() #得到画板、轴
        # axes.boxplot(all_data) #描点上色
        # plt.show()
    finalListTNCP[:, ind] = RETNCP
    finalList1[:, ind] = RE1
    finalList2[:, ind] = RE2
    ThetaErList[:, ind] = RETheta
    #finalListfal[:, ind] = REfal

    ind = ind + 1

print(np.mean(finalListTNCP,0))
print(np.mean(finalList1,0))
print(np.mean(finalListfal,0))
print(np.mean(finalList2,0))
print(np.mean(ThetaErList,0))

# np.savetxt('./Binomial/TNCPresult.csv',finalListTNCP,delimiter=',')
# np.savetxt('./Binomial/AirCPresult.csv',finalList1,delimiter=',')
# np.savetxt('./Binomial/falrtcresult.csv',finalListfal,delimiter=',')
# np.savetxt('./Binomial/ExpAirCPresult.csv',finalList2,delimiter=',')
# np.savetxt('./Binomial/ThetaErList.csv',ThetaErList,delimiter=',')

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
#
# df4 = pd.DataFrame(finalListfal.reshape(len(missList)*duplicate))
# df4['method']='falrtc'
# df4['MR'] = np.tile(missList, duplicate)

df = df1.append(df3).append(df2)
df.columns = ['value', 'method', 'MR']
fig, axs = plt.subplots(1, 1)
sns.boxplot(data=df,x='MR',y='value',hue='method')
plt.show()
#
# plt.savefig('./Binomial/Bino_100_size50rank3.png')




# #这里画最终的比较图
# import matplotlib.pyplot as plt
# plt.plot(missList, finalListCP,label="RE of CPALS",color="green",linestyle = 'dashed')
# plt.plot(missList,finalList1, label="RE of AirCP",color="red",linestyle = 'dashed')
# plt.plot(missList, finalList2,label="RE of ExpAirCP",color="blue")
#
# plt.xlabel("Missing Ratio")
# #Y轴的文字
# plt.ylabel("RE")
# #图表的标题
# plt.title("Relative error")
# plt.legend()
# plt.show()
# plt.savefig('ttt.png')



# #这里画一下histogram
# print(self2.disList)
#
# import matplotlib.pyplot as plt
# #
# # # 这里画与真实值之间差距
# # plt.plot(self2.disList,label="Error of ExpAirCP",color="blue")
# # plt.xlabel("Time(s)")
# # #Y轴的文字
# # plt.ylabel("log(distance)")
# # #图表的标题
# # plt.title("Distance")
# # plt.legend()
# #
# # plt.show()
#
# # 这里画收敛速度的图
# plt.plot(np.log(self.errList),label="Error of AirCP",color="red",linestyle = 'dashed')
# plt.plot(np.log(self2.errList),label="Error of ExpAirCP",color="blue")
# plt.xlabel("Time(s)")
# #Y轴的文字
# plt.ylabel("Err")
# #图表的标题
# plt.title("ErrList")
# plt.legend()
# plt.show()
#
# x = self.X.data-true_data.data
# y = self2.X.data-true_data.data
# all_data=[x.reshape(np.prod(siz)),y.reshape(np.prod(siz))]
#
# #直方图
# bins = np.linspace(-50, 50, 100)
# plt.hist(all_data, bins, alpha=0.7, label=['AirCP', 'ExpAirCP'])
# plt.legend(loc='upper right')
# plt.show()
# #箱线图
# figure,axes=plt.subplots() #得到画板、轴
# axes.boxplot(all_data) #描点上色
# plt.show()



