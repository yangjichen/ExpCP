# coding=utf-8
import numpy as np
import pyten
from scipy import stats
from pyten.method.PoissonAirCP import PoissonAirCP
from pyten.method import AirCP
from pyten.tools import tenerror
from pyten.method import falrtc,TNCP,cp_als
import util
import matplotlib.pyplot as plt
import time

'''
这个实验是为了比较不同算法的运行速度
'''
siz = [20,20,20]
rank = [4,4,4]
#missList = np.around(np.arange(0.05,0.9,0.1),2)
missList = [0.05]

# Theta的第三种构造方法，结合辅助信息去生成Theta，这是最终想要的
#
# 生成相同的数据
np.random.seed(222)
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
                             OnlyObs=True ,TrueValue = true_data,alpha = [3,3,3], lmbda=3,sim_mats=simMat2,eta=1e-4)

        self3 = TNCP(data,Omega, rank=com_rank,alpha = [3,3,3], lmbda=3)
        start1 = time.time()
        rX1 = falrtc(data, Omega, max_iter=100)
        start2 = time.time()
        self1.run()
        start3 = time.time()
        self2.run()
        start4 = time.time()
        self3.run()
        start5 = time.time()


print('falrtc cost {0}'.format(start2-start1))
print('AirCP cost {0}'.format(start3-start2))
print('ExpAirCP cost {0}'.format(start4-start3))
print('TNCP cost {0}'.format(start5-start4))
