#coding=utf-8
import gdelt
import pandas as pd

'''
这部分是为了爬取2018年GDELT的数据
'''

# Version 1 queries
gd1 = gdelt.gdelt(version=1)
# pull events table, range, output to json format
Monlist = [['2018 1 1','2018 1 31'],
        ['2018 2 1','2018 2 27'],
        ['2018 3 1','2018 3 31'],
        ['2018 4 1','2018 4 30'],
        ['2018 5 1','2018 5 31'],
        ['2018 6 1','2018 6 30'],
        ['2018 7 1','2018 7 31'],
        ['2018 8 1','2018 8 31'],
        ['2018 9 1','2018 9 30'],
        ['2018 10 1','2018 10 31'],
        ['2018 11 1','2018 11 30'],
        ['2018 12 1','2018 12 31'] ]
Infolist = ['SQLDATE', 'Actor1Geo_CountryCode', 'Actor2Geo_CountryCode', 'EventRootCode', 'Actor1Geo_Lat',
                 'Actor1Geo_Long', 'Actor2Geo_Lat', 'Actor2Geo_Long']

# for i in range(1):
#     results = gd1.Search(Monlist[i],coverage=True,table='events')
#     print(len(results))

result = gd1.Search(['2018 1 1','2018 1 2'],coverage=True,table='events')
