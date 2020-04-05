# coding=utf-8
import numpy as np
from scipy.sparse import csgraph
import pyten.tenclass
import pyten.tools
from pyten.method import cp_als
import time


class PoissonAirCP(object):
    """ This routine solves the Exponential_Family_Air_CP Tensor
      completion via Alternation Direction Method of Multipliers (ADMM)."""

    def __init__(self, obser,omega=None, rank=4, tol=1e-5, max_iter=500, sim_mats=None, alpha=None, lmbda=None,
                 eta=1e-4, rho=1.05, printitn=50):
        if not obser:
            raise ValueError("AirCP: observed Tensor cannot be empty!")
        elif type(obser) != pyten.tenclass.Tensor and type(obser) != np.ndarray:
            raise ValueError("AirCP: cannot recognize the format of observed Tensor!")
        elif type(obser) == np.ndarray:
            self.T = pyten.tenclass.Tensor(obser)
        else:
            self.T = obser

        if omega is None:
            self.omega = self.T.data * 0 + 1

        if type(omega) != pyten.tenclass.Tensor and type(omega) != np.ndarray:
            raise ValueError("AirCP: cannot recognize the format of indicator Tensor!")
        elif type(omega) == np.ndarray:
            self.omega = pyten.tenclass.Tensor(omega)
        else:
            self.omega = omega

        if not self.omega:
            raise ValueError("AirCP: indicator Tensor cannot be empty!")

        if type(rank) == list or type(rank) == tuple:
            rank = rank[0]

        self.ndims = self.T.ndims
        self.shape = self.T.shape

        if sim_mats is None:
            self.simMats = np.array([np.identity(self.shape[i]) for i in range(self.ndims)])
            #print( self.simMats[0])
        elif type(sim_mats) != np.ndarray and type(sim_mats) != list:
            raise ValueError("AirCP: cannot recognize the format of similarity matrices from auxiliary information!")
        else:
            self.simMats = np.array(sim_mats)
        self.L = np.array([csgraph.laplacian(simMat, normed=False) for simMat in self.simMats])

        if alpha is None:
            self.alpha = np.ones(self.ndims)
            self.alpha = self.alpha / sum(self.alpha)
        else:
            self.alpha = alpha

        self.rank = rank

        if lmbda is None:
            self.lmbda = 1 / np.sqrt(max(self.shape))
        else:
            self.lmbda = lmbda

        if printitn == 0:
            printitn = max_iter

        self.maxIter = max_iter
        self.tol = tol
        self.eta = eta
        self.rho = rho
        self.errList = []
        self.X = None
        self.X_pre = None
        #修改停止准则
        self.Theta_pre = None
        self.U = None
        self.Y = None
        self.Z = None
        self.II = None
        self.normT = np.linalg.norm(self.T.data)
        self.printitn = printitn


    def initializeLatentMatrices(self):
        # 这里设定随机数种子是为了让和AirCP有相同的初值U，方便比较
        np.random.seed(50)
        self.U = [np.random.rand(self.shape[i], self.rank) for i in range(self.ndims)]
        print(self.U)
        self.Y = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.Z = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.II = pyten.tools.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
        #初始化时候需要根据指数分布族调整，X与U有关,是否要加exp()

        midpara = self.II.copy()
        for i in range(self.ndims):
            midpara = midpara.ttm(self.U[i], i + 1)
        # 选用与AirCP一样的初始化方法
        self.X = self.T.data + (1 - self.omega.data) * (self.T.norm() / self.T.size())
        # 初始化方法2
        #self.X = self.T.data + (1 - self.omega.data) * fir_dev(midpara.data)
        # 使用这句可以将nparray转换成tensor类
        self.X = pyten.tenclass.Tensor(self.X)
        self.X_pre = self.X.copy()
        #修改停止准则
        self.Theta_pre = midpara.copy()

    def run(self):

        self.errList = []
        self.initializeLatentMatrices()

        for k in range(self.maxIter):
            # update step eta
            self.eta *= self.rho

            # update Z
            for i in range(self.ndims):
                temp_1 = self.eta * self.U[i] - self.Y[i]
                temp_2 = self.eta * np.identity(self.shape[i]) + self.alpha[i] * self.L[i]
                self.Z[i] = np.dot(np.linalg.inv(temp_2 + 0.00001 * np.identity(self.shape[i])), temp_1)

                #检查哪些值一开始会引入负数Y，说明这两项都不会引入负数
                #print(np.min(self.[i]))
                #print(np.min(self.Z[i]))

            # # update U
            # for i in range(self.ndims):
            #     # calculate intermedian Tensor and its mode-n unfolding
            #     midT = self.II.copy()
            #     # calculate Kronecker product of U(1), ..., U(i-1),U(i+1), ...,U(n)
            #     for j in range(self.ndims):
            #         if j == i:
            #             continue
            #         midT = midT.ttm(self.U[j], j + 1)

            #     # B^(n)的展开
            #     unfoldD_temp = pyten.tenclass.Tenmat(midT, i + 1)
            #     #
            #     temp_Z = self.eta * self.Z[i] + self.Y[i]
            #     temp_B = np.dot(unfoldD_temp.data, unfoldD_temp.data.T)
            #     temp_B += self.eta * np.identity(self.rank) + self.lmbda * np.identity(self.rank)
            #     temp_B += 0.00001 * np.identity(self.rank)
            #     #这一部分是X的（mold-i）展开,从论文中可以看出来,计算X的那一步
            #     temp_C = pyten.tenclass.Tenmat(self.X, i + 1)
            #     temp_D = np.dot(temp_C.data, unfoldD_temp.data.T)
            #     self.U[i] = np.dot((temp_D + temp_Z), np.linalg.inv(temp_B))


            # 这部分是ExpAirCP中更新U的过程
            # 现根据别的量计算tilde(W)与tilde(X),假定先使用poisson distribution
            def fir_dev(varx):
                return (np.exp(varx))

            def sed_dev(varx):
                return (np.exp(varx))



            for n in range(self.ndims):
                # 根据当前U计算当前Theta,跟原始算法更新X步code相同,这部分检查过没错

                midTheta = self.II.copy()
                for i in range(self.ndims):
                    midTheta = midTheta.ttm(self.U[i], i + 1)

                # print(midTheta.data)
                # 根据当前theta计算tildeW tildeX，下面暂时都是nparray
                # 这里比较疑惑用上一轮补全值去做还是原始值去做
                tildeU = -self.X.data + fir_dev(midTheta.data)
                tildeW = sed_dev(midTheta.data)
                tildeX = midTheta.data - (tildeU / tildeW)

                # #2019/7/12是否要只考虑观测到的位置,经过测试发现missing比例小的时候二者效果差不多，但是比例大的时候只考虑观测值的话效果很差，需要让lambda变大
                # tildeW = tildeW * self.omega.data

                #print(np.min(tildeX))

                tildeX = pyten.tenclass.Tensor(tildeX)
                tildeW = pyten.tenclass.Tensor(tildeW)


                #第一部分，计算Mn
                #先计算W_(n),X_(n)，这部分和注释中temp_C的展开写的类似，应该没错误
                Wn = pyten.tenclass.Tenmat(tildeW, n + 1)

                Xn = pyten.tenclass.Tenmat(tildeX, n + 1)
                # print(np.min(self.U[i]))
                # print(np.min(midTheta.data))
                # print(np.min(Wn.data))


                # 计算B^(n)
                midT = self.II.copy()
                # calculate Kronecker product of U(1), ..., U(i-1),U(i+1), ...,U(n)
                for jj in range(self.ndims):
                    if jj == n:
                        continue
                    midT = midT.ttm(self.U[jj], jj + 1)
                # mold i unfolding 这就是B^(n)
                unfoldD_temp = pyten.tenclass.Tenmat(midT, n + 1)

                Mn = self.eta * self.Z[n] + self.Y[n] + np.dot((Wn.data * Xn.data), unfoldD_temp.data.T)
                #print(np.min(Mn))
                #Mn计算完毕

                #下面第二部分计算Lambda，与i有关，与第三部分放在一起，直观一点，后期再优化
                #第三部分，一共这么多行，一行一行更新，注意，行索引为i
                for i in range(self.shape[n]):

                    #先计算对应这行的Lambda矩阵,一个元素一个元素计算
                    Lami = np.zeros((self.rank,self.rank))

                    for row in range(self.rank):
                        for col in range(self.rank):
                            # 2019/7/10 
                            # 这里计算进行了优化，减少了一层循环  
                            item = np.sum(unfoldD_temp.data[row,] * unfoldD_temp.data[col,] * Wn.data[i,])
                            Lami[row,col] = item

                    #print(Lami)

                    #现在可以更新Un的第i行了
                    #method 1
                    temp_uni = (self.lmbda+self.eta)*np.identity(self.rank) + Lami
                    #print(np.linalg.inv(temp_uni))
                    #print(np.dot(np.linalg.inv(temp_uni),Mn[i, ].T))
                    aaa = np.dot(np.linalg.inv(temp_uni),Mn[i, ].T)
                    self.U[n][i,] = aaa.T

                    # # # method 2
                    # temp_uni = (self.lmbda+self.eta)*np.identity(self.rank) + Lami
                    # des = np.dot(temp_uni.T, temp_uni)
                    # dess = np.dot( np.linalg.inv(des), temp_uni.T )
                    # betaa = np.dot(dess, Mn[i, ].T)
                    # # n是dim的索引，通常是三维，i是行索引
                    # # print(np.min(temp_uni.T))
                    # self.U[n][i,] = betaa.T




                #print(self.U[n])


            # update X
            # 这里是计算tensor的方法,特定poisson分布时候，加上np.exp()
            midT = self.II.copy()
            for i in range(self.ndims):
                midT = midT.ttm(self.U[i], i + 1)
            self.X = midT.copy()

            self.X.data = self.T.data * self.omega.data + fir_dev(self.X.data) * (1 - self.omega.data)

            # Modified for outlier, 2019.7.4
            #考虑更新X时候加入99%分位数的阈值来避免异常值点的出现
            TH = np.percentile(self.X.data,95)
            THidx = self.X.data>TH
            self.X.data[THidx] =TH

            # 为了查看离群点位置是否每次变化，记得删除
            # yy = self.X.data-self.T.data
            # idx = np.where(yy==np.max(yy))
            # #print(idx)
            # print(self.X.data[idx])

            # update Lagrange multiper
            for i in range(self.ndims):
                self.Y[i] = self.Y[i] + self.eta * (self.Z[i] - self.U[i])

            # checking the stop criteria
            # # 准则：前后两次x的差值是不是足够小

            # error = np.linalg.norm(self.X_pre.data - self.X.data) / self.normT
            # self.X_pre = self.X.copy()
            # self.errList.append(error)
            midT = self.II.copy()
            for i in range(self.ndims):
                midT = midT.ttm(self.U[i], i + 1)

            #如果这里检查收敛时使用Theta
            error = np.linalg.norm(self.Theta_pre.data - midT.data) 
            self.Theta_pre = midT.copy()
            self.errList.append(error)

            if (k + 1) % self.printitn == 0:
                print ('ExpAirCP: iterations={0}, difference={1}'.format(k + 1, self.errList[-1]))
            elif error < self.tol:
                print ('ExpAirCP: iterations={0}, difference={1}'.format(k + 1, self.errList[-1]))

            if error < self.tol:
                # # update X
                # # 这里是计算tensor的方法,特定poisson分布时候，加上np.exp()
                # midT = self.II.copy()
                # for i in range(self.ndims):
                #     midT = midT.ttm(self.U[i], i + 1)
                # self.X = midT.copy()

                # self.X.data = self.T.data * self.omega.data + fir_dev(self.X.data) * (1 - self.omega.data)

                # # Modified for outlier, 2019.7.4
                # #考虑更新X时候加入99%分位数的阈值来避免异常值点的出现
                # TH = np.percentile(self.X.data,95)
                # THidx = self.X.data>TH
                # self.X.data[THidx] =TH

                break


# 上面是针对poisson distribution写的更新过程。换别的分布时候需要更新的两个东西分别是
# 1. 更新U时候的一阶导数和二阶导数
# 2. update X时候的最后一步