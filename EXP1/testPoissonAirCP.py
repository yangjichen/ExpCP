# coding=utf-8
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import pyten
from scipy import stats
from pyten.method.PoissonAirCP import PoissonAirCP
from pyten.method import AirCP
from pyten.tools import tenerror
from pyten.method import falrtc,TNCP,cp_als
import util
import matplotlib.pyplot as plt

siz = [50,50,50]
rank = [4,4,4]
#missList = np.around(np.arange(0.05,0.9,0.1),2)
missList = [0.05]

# Theta的第三种构造方法，结合辅助信息去生成Theta，这是最终想要的
#
# 生成相同的数据
np.random.seed(823)
# 要不要用协方差矩阵去生成
theta, simMat,simMat2 = util.create2(siz = siz, r = rank)

# 这里用theta生成数据矩阵
dat = np.zeros(siz)
for i in range(siz[0]):
    for j in range(siz[1]):
        for k in range(siz[2]):
            lamb = np.exp(theta.data[i, j, k])
            dat[i, j, k] = np.random.poisson(lam=lamb, size=1)[0]

# #保存原始数据，其他code也需要用
# np.save('./Poisson/Poissondata.npy',dat)
# np.save('./Poisson/PoissondataETF.npy',dat.reshape(50*50*50))


true_data = dat.copy()
true_data = pyten.tenclass.tensor.Tensor(true_data)


#重复做试验的次数
duplicate = 1
# 这里是为了画图比较
finalListAirCP = np.zeros((duplicate,len(missList)))
finalListExpAirCP = np.zeros((duplicate,len(missList)))
finalListTNCP = np.zeros((duplicate,len(missList)))
finalListfal = np.zeros((duplicate,len(missList)))
ThetaErList = np.zeros((duplicate,len(missList)))

ind = 0

for miss in missList:
    REAirCP = []
    REExpAirCP = []
    RETNCP = []
    REfal = []
    RETheta = []
    for dup in range(duplicate):
        np.random.seed(dup*4)
        #每次都用同一份数据去做
        data = dat.copy()
        #观测值：丢失部分数据的
        Omega = (np.random.random(siz) > miss) * 1
        data[Omega == 0] -= data[Omega == 0]


        print('missing ratio: {0}'.format(miss))
        #补全时候用的rank
        com_rank = 4
        data = pyten.tenclass.tensor.Tensor(data)
        #print(data.data)
        #所使用的补全方法
        self1 = AirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,alpha = [3,3,3], lmbda=3,sim_mats=simMat2)
        self2 = PoissonAirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                             OnlyObs=True ,TrueValue = true_data,alpha = [5,5,5], lmbda=5,sim_mats=simMat2,eta=1e-4)

        self3 = TNCP(data,Omega, rank=com_rank,alpha = [3,3,3], lmbda=3)

        rX1 = falrtc(data, Omega, max_iter=100)
        try:
            #print(data.data)
            self1.run()
            #print(data.data)
            self2.run()
            self3.run()
        except:
            continue

        #这里看对原始数据的补全准不准
        [Err, ReErr1, ReErr2] = tenerror(self1.X, true_data, Omega)
        [EErr, EReErr1, EReErr2] = tenerror(self2.X, true_data, Omega)
        [ErrTNCP, ReErrTNCP, ReErr2TNCP] = tenerror(self3.X, true_data, Omega)
        #[Errfal, ReErrfal, ReErr2fal] = tenerror(rX1, true_data, Omega)


        print ('AirCP Completion Error: {0}, {1}, {2}'.format(Err, ReErr1, ReErr2))
        print ('ExpAirCP Completion Error: {0}, {1}, {2}'.format(EErr, EReErr1, EReErr2))
        print ('TNCP Completion Error: {0}, {1}, {2}'.format(ErrTNCP, ReErrTNCP, ReErr2TNCP))
        #print ('falrtc Completion Error: {0}, {1}, {2}'.format(Errfal, ReErrfal, ReErr2fal))

        #这里看对原始theta矩阵的估计准不准
        pred2 = pyten.tenclass.Ktensor(np.ones(com_rank), us=self2.U).totensor()
        print ('Theta Error for ExpAirCP: {0}'.format(np.linalg.norm(pred2.data-theta.data)/np.linalg.norm(theta.data)))

        RETNCP.append(ReErrTNCP)
        REAirCP.append(ReErr1)
        REExpAirCP.append(EReErr1)
        #REfal.append(ReErrfal)

        ThetaRE = np.linalg.norm(pred2.data-theta.data)/np.linalg.norm(theta.data)
        RETheta.append(ThetaRE)


        # x = self1.X.data-true_data.data
        # y = self2.X.data-true_data.data
        # all_data=[x.reshape(np.prod(siz)),y.reshape(np.prod(siz))]
        #
        # figure,axes=plt.subplots() #得到画板、轴
        # axes.boxplot(all_data) #描点上色
        # plt.show()


    # print(np.mean(RECP),np.mean(REAirCP),np.mean(REExpAirCP))
    finalListTNCP[:,ind] = RETNCP
    finalListAirCP[:,ind] = REAirCP
    finalListExpAirCP[:,ind] = REExpAirCP
    #finalListfal[:,ind] = REfal
    ThetaErList[:,ind] = RETheta

    ind = ind+1

print(np.mean(finalListTNCP,0))
print(np.mean(finalListAirCP,0))
#print(np.mean(finalListfal,0))
print(np.mean(finalListExpAirCP,0))
print(np.mean(ThetaErList,0))

# np.savetxt('./Poisson/TNCPresult.csv',finalListTNCP,delimiter=',')
# np.savetxt('./Poisson/AirCPresult.csv',finalListAirCP,delimiter=',')
# np.savetxt('./Poisson/falrtcresult.csv',finalListfal,delimiter=',')
# np.savetxt('./Poisson/ExpAirCPresult.csv',finalListExpAirCP,delimiter=',')
# np.savetxt('./Poisson/ThetaErList.csv',ThetaErList,delimiter=',')

#这里画最终的比较图
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df1 = pd.DataFrame(finalListAirCP.reshape(len(missList)*duplicate))
df1['method']='AirCP'
df1['MR'] = np.tile(missList, duplicate)

df2 = pd.DataFrame(finalListExpAirCP.reshape(len(missList)*duplicate))
df2['method']='ExpAirCP'
df2['MR'] = np.tile(missList, duplicate)

df3 = pd.DataFrame(finalListTNCP.reshape(len(missList)*duplicate))
df3['method']='TNCP'
df3['MR'] = np.tile(missList, duplicate)

# df4 = pd.DataFrame(finalListfal.reshape(len(missList)*duplicate))
# df4['method']='falrtc'
# df4['MR'] = np.tile(missList, duplicate)


df = df1.append(df3).append(df2)
df.columns = ['value', 'method', 'MR']
fig, axs = plt.subplots(1, 1)
sns.boxplot(data=df,x='MR',y='value',hue='method')
plt.show()

# plt.savefig('./Poisson/size50rank4.png')


