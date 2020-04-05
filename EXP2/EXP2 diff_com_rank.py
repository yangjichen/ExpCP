# coding=utf-8
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pyten
from scipy import stats
from pyten.method.PoissonAirCP import PoissonAirCP
import util
import matplotlib.pyplot as plt
from pyten.tools import tenerror
import pandas as pd


siz = [20,20,20]
rank = [4,4,4]
missList = np.around(np.arange(0.15,0.7,0.1),2)
#missList = [0.35]
np.random.seed(53)
# 要不要用协方差矩阵去生成
theta, simMat,simMat2= util.create2(siz = siz, r = rank)


# 这里用theta生成数据矩阵
dat = np.zeros(siz)
for i in range(siz[0]):
    for j in range(siz[1]):
        for k in range(siz[2]):
            lamb = np.exp(theta.data[i, j, k])
            dat[i, j, k] = np.random.poisson(lam=lamb, size=1)[0]



true_data = dat.copy()
true_data = pyten.tenclass.tensor.Tensor(true_data)

# 这里是为了画图比较
def get_RE(com_rank,duplicate=10):
    ind=0
    XErList = []
    ThetaErList = np.zeros((duplicate,len(missList)))
    for miss in missList:
        RETheta = []
        REX = []
        for dup in range(duplicate):
            np.random.seed(dup*4)
            #每次都用同一份数据去做
            data = dat.copy()
            #观测值：丢失部分数据的
            Omega = (np.random.random(siz) > miss) * 1
            data[Omega == 0] -= data[Omega == 0]

            #补全时候用的rank
            print('missing ratio: {0}'.format(miss))

            data = pyten.tenclass.tensor.Tensor(data)
            self2 = PoissonAirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                                 OnlyObs=True ,TrueValue = true_data,alpha = [10,10,10], lmbda=10,sim_mats=simMat2)
            self2.run()


            #这里看对原始数据的补全准不准
            [EErr, EReErr1, EReErr2] = tenerror(self2.X, true_data, Omega)
            REX.append(EReErr1)

            #这里看对原始theta矩阵的估计准不准
            pred2 = pyten.tenclass.Ktensor(np.ones(com_rank), us=self2.U).totensor()
            ThetaRE = np.linalg.norm(pred2.data-theta.data)/np.linalg.norm(theta.data)
            RETheta.append(ThetaRE)



        # print(np.mean(RECP),np.mean(RE1),np.mean(RE2))
        XErList.append(np.mean(REX))
        ThetaErList[:, ind] = RETheta
        ind = ind + 1

    print(XErList)
    print(ThetaErList)
    return ((XErList,ThetaErList))


dup1, dup2, dup3 = 10,10,10
r4X, r4Theta = get_RE(com_rank=3,duplicate=dup1)
r6X, r6Theta = get_RE(com_rank=4,duplicate=dup2)
r7X, r7Theta = get_RE(com_rank=6,duplicate=dup3)
r8X, r8Theta = get_RE(com_rank=8,duplicate=dup3)




import seaborn as sns
df1 = pd.DataFrame(r4Theta.reshape(len(missList)*dup1))
df1['completion rank']='Rank=3'
df1['MR'] = np.tile(missList, dup1)

df2 = pd.DataFrame(r6Theta.reshape(len(missList)*dup2))
df2['completion rank']='Rank=4'
df2['MR'] = np.tile(missList, dup2)
#
df3 = pd.DataFrame(r7Theta.reshape(len(missList)*dup3))
df3['completion rank']='Rank=6'
df3['MR'] = np.tile(missList, dup3)

df4 = pd.DataFrame(r8Theta.reshape(len(missList)*dup3))
df4['completion rank']='Rank=8'
df4['MR'] = np.tile(missList, dup3)

df = df1.append(df2).append(df3).append(df4)
#df = df1.append(df2)
df.columns = ['Relative Error', 'Pre-specified Rank', 'Fraction of Missing Data']
fig, axs = plt.subplots(1, 1)
sns.boxplot(data=df,x='Fraction of Missing Data',y='Relative Error',hue='Pre-specified Rank')
plt.show()
plt.savefig('Exp2_size20(53).png',dpi=300)


