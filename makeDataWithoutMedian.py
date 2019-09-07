import pandas
import json
import numpy as np
import ctypes
from scipy import stats

#df = pandas.read_json('meow.json')

data = json.load(open('Sigma.march1.json'))
df = pandas.io.json.json_normalize(data)
print(df.head(20))


datf = pandas.read_csv('xycoordinates.csv')

df = df[df.beaconId != '']
df = df[df.relayNo != '']

patternDel = "[a-zA-Z]"
filter = df['beaconId'].str.contains(patternDel)
df = df[~filter]



df = df.reindex(sorted(df.columns), axis=1)
df["beaconId"] = df["beaconId"].astype(int)
df["relayNo"] = df["relayNo"].astype(int)
df = df[df.beaconId < 1000]
df = df[df.beaconId > 0]


#mediandf = df.groupby(['beaconId','relayNo'], as_index=False)['rssiVal'].median()



def compareRSSIToMedian(dataset):
    global mediandf
    relay = dataset['relayNo']
    beacon = dataset['beaconId']
    firstCond = mediandf[mediandf['beaconId']==beacon]
    currentMedianRSSIDF = firstCond[firstCond['relayNo']==relay].iloc[0]
    #print(currentMedianRSSIDF)
    if((currentMedianRSSIDF['rssiVal']>(dataset['rssiVal']+2)) or (currentMedianRSSIDF['rssiVal']<(dataset['rssiVal']-2))):
        return False
    else:
        return True


#df['withinMedian'] = df.apply(compareRSSIToMedian,axis=1)
#print(df)
#df.to_csv('medianSeparated.csv')
#df = df[df.withinMedian==True]
df['timeStamp'] = pandas.to_datetime(df['timeStamp'],yearfirst=True,utc=True)
df['timeStamp'] = df['timeStamp'].dt.floor('2min')
df = pandas.pivot_table(df,index=['timeStamp','beaconId'],columns=['relayNo'],values='rssiVal',aggfunc=np.median)
#df = df.drop([0],axis=1)
datf.columns = ['beaconId','X-Coordinate','Y-Coordinate']


df = df.reset_index()
df = df.fillna(0)
df = df.merge(datf,how='inner',on='beaconId')

temp= df.head(100)
temp.to_csv('showcase.csv')
print(df.shape)
df.to_csv('myfile.csv')