#coding=utf-8
import pandas as pd
import numpy as np
import os
import datetime
starttime = datetime.datetime.now()

'''
这部分是为了将数据整理成tensor形式保存下来进行第三步
'''


def get_dir_list(file_dir):
    file_list = []
    for i in os.listdir(file_dir):
        if (i.find('data') != -1):
            file_list.append(i)
    return(file_list)


def constuct(x, dict1, dict2, dict3, geo_list, time_list, event_list):
    tensorr = np.zeros((len(geo_list), len(time_list), len(event_list)))
    if pd.isna(x['Actor1Geo_CountryCode']):
        location = x['Actor2Geo_CountryCode']
    else:
        location = x['Actor1Geo_CountryCode']
    time = x['SQLDATE']
    event = x['EventRootCode']
    tensorr[dict1[location], dict2[time], dict3[event]] += 1
    return (tensorr)

def build_tensor(data,printitn=10000):
    #step1 清理数据
    # 删除时间不合适的值/
    data = data[(data['SQLDATE'] <= 20181231) & (data['SQLDATE'] >= 20180101)]
    # 删除event不合适的列/
    data = data[data['EventRootCode'] != '--']
    # 将'EventRootCode'全部变成int类型
    data['EventRootCode'] = data['EventRootCode'].astype(int)
    # 删除地理信息同时为na的行
    a = pd.isna(data['Actor1Geo_CountryCode']) & pd.isna(data['Actor2Geo_CountryCode'])
    data = data[~a]

    #step2 构建列表，且删除nan值
    geo_list = list(set(pd.unique(data['Actor1Geo_CountryCode'])).union(set(pd.unique(data['Actor2Geo_CountryCode']))))
    time_list = list(pd.unique(data['SQLDATE']))
    event_list = list(pd.unique(data['EventRootCode']))

    geo_list = [x for x in geo_list if str(x) != 'nan']
    time_list = np.sort([x for x in time_list if str(x) != 'nan'])
    event_list =np.sort([x for x in event_list if str(x) != 'nan'])

    dict1 = dict(zip(geo_list, range(len(geo_list))))
    dict2 = dict(zip(np.sort(time_list), range(len(time_list))))
    dict3 = dict(zip(np.sort(event_list), range(len(event_list))))

    # step3 构建张量方法一：所用时间短，占用内存大
    # tensorseries = data.apply(constuct,axis = 1, args = (dict1, dict2, dict3, geo_list, time_list, event_list))
    # tensorr = np.sum(tensorseries)
    # return(tensorr,geo_list,time_list,event_list)

    #step3 构建张量方法二：所用时间长，占用内存小
    tensorr = np.zeros( (len(geo_list), len(time_list), len(event_list)))
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['Actor1Geo_CountryCode']):
            location = data.iloc[i]['Actor2Geo_CountryCode']
        else:
            location = data.iloc[i]['Actor1Geo_CountryCode']
        time = data.iloc[i]['SQLDATE']
        event = data.iloc[i]['EventRootCode']

        if (i + 1) % printitn == 0:
            print ('Finish: iterations={0}, Percentage={1}'.format(i+1, (i+1)/len(data)))

        tensorr[dict1[location], dict2[time], dict3[event]] += 1

    return(tensorr,geo_list,time_list,event_list)

if __name__ == '__main__':

    filelist = get_dir_list('../realdata')
    dataset = pd.DataFrame()

    # print(filelist)
    # for i in filelist:
    #     dataset = dataset.append(pd.read_csv(i))
    #     print(i)
    dataset = dataset.append(pd.read_csv('data.csv',dtype = {'EventRootCode': object}))
    tensorr,geo_list,time_list,event_list = build_tensor(dataset)
    print(geo_list)
    print(time_list)
    print(event_list)


    #np.save('build_tensor.npy', tensorr)

    #long running
    endtime = datetime.datetime.now()

    print ((endtime - starttime).seconds)