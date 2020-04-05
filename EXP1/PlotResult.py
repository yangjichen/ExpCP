# coding=utf-8
import numpy as np
import pandas as pd
from scipy import stats
import util
import matplotlib.pyplot as plt
import os
import re

def get_dir_list(file_dir):
    file_list = []
    for i in os.listdir(file_dir):
        if (i.find('csv') != -1):
            file_list.append(file_dir + '/'+i)
    return(file_list)

# 'Binomial'   'negBinomial' 'Poisson'
exptype = 'Binomial'

file_dir  = './'+exptype

file_list = get_dir_list(file_dir)

# ['ThetaErList', 'TNCPresult', 'ExpAirCPresult', 'falrtcresult', 'AirCPresult']
resultlist = []
for i in range(len(file_list)):
    name = re.findall(".*/(.*).csv.*",file_list[i])
    resultlist.append(name[0])
    locals()[name[0]] = pd.read_csv(file_list[i],header = None)

missList = np.around(np.arange(0.05,0.9,0.1),2)

tSVDresult = tSVDresult.T
#先画折线图
import matplotlib.pyplot as plt
plt.plot(missList, np.mean(TNCPresult,0),label="RE of TNCP",color="green",linestyle = 'dashed')
plt.plot(missList,np.mean(falrtcresult,0), label="RE of falrt",color="black",linestyle = 'dashed')
plt.plot(missList,np.mean(AirCPresult,0), label="RE of AirCP",color="purple",linestyle = 'dashed')
plt.plot(missList,np.mean(tSVDresult,0), label="RE of tSVD",color="red",linestyle = 'dashed')
plt.plot(missList, np.mean(ExpAirCPresult,0),label="RE of ExpAirCP",color="blue")

plt.xlabel("Missing Ratio")
#Y轴的文字
plt.ylabel("RE")
#图表的标题
plt.title(exptype+' '+"Relative error")
plt.legend()

plt.savefig(file_dir+'/plot1.png')
plt.show()

#再画箱线图

import seaborn as sns
duplicate =TNCPresult.shape[0]
df1 = pd.DataFrame(np.array(AirCPresult).reshape(len(missList)*duplicate))
df1['method']='AirCP'
df1['MR'] = np.tile(missList, duplicate)

df2 = pd.DataFrame(np.array(ExpAirCPresult).reshape(len(missList)*duplicate))
df2['method']='ExpAirCP'
df2['MR'] = np.tile(missList, duplicate)

df3 = pd.DataFrame(np.array(TNCPresult).reshape(len(missList)*duplicate))
df3['method']='TNCP'
df3['MR'] = np.tile(missList, duplicate)

df4 = pd.DataFrame(np.array(falrtcresult).reshape(len(missList)*duplicate))
df4['method']='falrtc'
df4['MR'] = np.tile(missList, duplicate)

df = df4.append(df3).append(df1).append(df2)
df.columns = ['value', 'method', 'MR']
fig, axs = plt.subplots(1, 1)
sns.boxplot(data=df,x='MR',y='value',hue='method')


plt.savefig(file_dir+'/plot2.png')
plt.show()



