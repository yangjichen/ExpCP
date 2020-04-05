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




'''
这个实验是为了比较模拟数据集上引入辅助信息能否提升准确率
'''
siz = [20,20,20]
rank = [4,4,4]
#missList = np.around(np.arange(0.05,0.9,0.1),2)
missList = [0.85]

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

true_data = dat.copy()
true_data = pyten.tenclass.tensor.Tensor(true_data)


#重复做试验的次数
duplicate = 10
# 这里是为了画图比较
listWithAux = []
listWithoutAux = []

ind = 0

for miss in missList:
    REwithout = []
    REwith = []

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

        self1 = PoissonAirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                             OnlyObs=True ,TrueValue = true_data, lmbda=3)
        self2 = PoissonAirCP(data, omega = Omega,rank=com_rank,max_iter=3000,tol = 1e-5,
                             OnlyObs=True ,TrueValue = true_data,alpha = [3,3,3], lmbda=3,sim_mats=simMat2)

        self1.run()
        self2.run()
        #这里看对原始数据的补全准不准
        [Err, ReErr1, ReErr2] = tenerror(self1.X, true_data, Omega)
        [EErr, EReErr1, EReErr2] = tenerror(self2.X, true_data, Omega)
        #[Errfal, ReErrfal, ReErr2fal] = tenerror(rX1, true_data, Omega)


        print ('ExpAirCP without Aux: {0}, {1}, {2}'.format(Err, ReErr1, ReErr2))
        print ('ExpAirCP with Aux: {0}, {1}, {2}'.format(EErr, EReErr1, EReErr2))

        REwithout.append(ReErr1)
        REwith.append(EReErr1)


    listWithoutAux.append(np.mean(REwithout))

    listWithAux.append(np.mean(REwith))

print(listWithoutAux)
print(listWithAux)

result = [listWithoutAux,listWithAux]
#np.savetxt('Effects of Aux Info.csv',result,fmt='%.4f',delimiter=',')
# plt.plot(listWithoutAux,label="without",color="red",linestyle = 'dashed')
# plt.plot(listWithAux,label="with",color="blue")
# plt.xlabel("Time(s)")
# #Y轴的文字
# plt.ylabel("Err")
# #图表的标题
# plt.title("ErrList")
# plt.legend()
# plt.show()

