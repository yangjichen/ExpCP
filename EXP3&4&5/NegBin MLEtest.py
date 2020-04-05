# coding=utf-8
import numpy as np
import scipy.optimize as sco
from scipy.special import loggamma
#
# #这个实验中想看Negative binomial参数是否能估计准确
# siz = [20,20,20]
# rank = [3,3,3]
# missList = [0.25]
#
# np.random.seed(13)
# failuresR = np.ones((siz))*1000
# theta, simMat= util.create4(siz = siz, r = rank)
#
#
# # 这里用theta生成数据矩阵
# dat = np.zeros(siz)
# for i in range(siz[0]):
#     for j in range(siz[1]):
#         for k in range(siz[2]):
#             prob = np.exp(theta.data[i, j, k]) / (1 + np.exp(theta.data[i, j, k]))
#             dat[i, j, k] = np.random.negative_binomial(failuresR[i,j,k], prob)
#
# MLEdata = np.array([np.mean(dat[i]) for i in range(siz[0])])
# MLEsigmasquare = np.array([np.var(dat[i])  for i in range(siz[0])])
# MLEmu = np.array([np.mean(dat[i]) for i in range(siz[0])])
# MLEp = (MLEmu)/MLEsigmasquare
#
#
# import scipy.optimize as sco
# from scipy.special import gamma
#
# def obj_func(r):
#     f = np.sum(loggamma(r) - loggamma(r+MLEdata) - r*np.log(MLEp))
#     return(f)
#
# opt = sco.minimize(fun=obj_func, x0=10,method='nelder-mead',
#           options={'xtol': 1e-8, 'disp': True})
# print('Estimator for r is {0}'.format(opt.x))




# #
'''
STEP1
这里想看如果用同p和r生成一层数据，是否可以用矩估计对p估计的比较准，确实是比较准确
'''
# failuresR = 100
# prob=np.random.rand(1)
# data = np.random.negative_binomial(failuresR, prob,size = (100,100))
#
#
# sigmasquare = np.var(data)
# mu = np.mean(data)
# p = (mu)/sigmasquare
# print('true prob. is {0}, estimation is {1}'.format(prob,p))
# #每一层数据，单独用MLE去对r进行估计
#
# data = data.reshape(10000)
# pseudop = np.repeat(p,10000)
# def obj_func(r):
#     f = np.sum(loggamma(r)- loggamma(r+data) - r*np.log(pseudop))
#     return(f)
# opt = sco.minimize(fun=obj_func, x0=10,method='nelder-mead',
#           options={'xtol': 1e-8, 'disp': True})
# print('Estimator for r is {0}'.format(opt.x))

'''
STEP2
更进一步如果用全局统一的r，每一层各自有一个p去构造数据，看p是否可以估计准确
结论：p与r估计比较准
'''

failuresR = 1000
problist = np.random.rand(10)
data = np.array([np.random.negative_binomial(failuresR, prob,size = (100,100)) for prob in problist])
sigmalist = np.array([np.var(data[i])  for i in range(10)])
mulist = np.array([np.mean(data[i]) for i in range(10)])
plist =  mulist/sigmalist


def obj_func(r):
    f = np.sum(loggamma(r) - loggamma(r+mulist) - r*np.log(plist))
    return(f)

opt = sco.minimize(fun=obj_func, x0=10)
print('Estimator for r is {0}'.format(opt.x))

'''
STEP3
更进一步如果用全局统一的r，所有位置p都不一样，然后把每层看做一个数据，求出p，然后用mle去求r
结论：这样就求不出了
'''
# failuresR = 10
# problist = np.random.rand(100,100,100)
# data = np.random.negative_binomial(failuresR, problist)
# sigmalist = np.array([np.var(data[i])  for i in range(10)])
# mulist = np.array([np.mean(data[i]) for i in range(10)])
# plist =  mulist/sigmalist
#
# print('estimation of p is {0}'.format(plist))
#
#
# def obj_func(r):
#     f = np.sum(loggamma(r) - loggamma(r+mulist) - r*np.log(plist))
#     return(f)
#
# opt = sco.minimize(fun=obj_func, x0=10)
# print('Estimator for r is {0}'.format(opt.x))


