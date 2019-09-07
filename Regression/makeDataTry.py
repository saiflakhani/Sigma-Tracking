import pandas
import json
import numpy as np
from scipy import stats
import ctypes

# df = pandas.read_json('meow.json')

#data = json.load(open('Sigma.march1.json'))
#df = pandas.io.json.json_normalize(data, errors='ignore')
# print(df)

df = pandas.read_csv('cleaned_1_march.csv')


datf = pandas.read_csv('xycord.csv')
#relayCords = pandas.read_csv('relayCords.csv')


#df = df[df.rssi != '00']
#df = df[df.beaconId != '']
#df = df[df.relayNo != '']
#df = df[~df['relayNo'].isnull()]
# print(df['relayNo'].unique())
# Y contained some other garbage, so null check was not enough
# df = df[df['rssiVal'].str.isnumeric()]
# df = df[df.beaconId != '1e50']
# df = df[df.beaconId != '9e48']
# df = df[df.beaconId != '52e9']
#patternDel = "[a-zA-Z]"
#filter = df['beaconId'].str.contains(patternDel)
#df = df[~filter]
# df['rssiVal'] = df['rssiVal'].str.extract('(\d+)', expand=False)
# df['rssiVal'] = pandas.to_numeric(df.rssiVal, errors='ignore')
# df = df.sort_values(by=['beaconId'])
# df = df.reset_index()
# df.index.name = 'Index'


#df = df.reindex(sorted(df.columns), axis=1)
#df["beaconId"] = df["beaconId"].astype(int)
#df["relayNo"] = df["relayNo"].astype(int)


# df["beaconId"] = df["beaconId"].cat.codes
# df['timeStamp'] = df['timeStamp.$date']##YOU MIGHT NEED TO COMMENT THESE TWO LINES
df = df.drop(['_id.$oid','Unnamed: 0'],axis=1)
df['timeStamp'] = pandas.to_datetime(df['timeStamp'],yearfirst=True,utc=True)
df['timeStamp'] = df['timeStamp'].dt.floor('2min')

df = pandas.pivot_table(df,index=['timeStamp', 'beaconId', 'X-Coordinate', 'Y-Coordinate'], columns=['relayNo'], values='rssiVal', aggfunc=(lambda x: stats.mode(x)[0][0]))




#print(df.head(20))
#df = df.drop([0],axis=1)
#datf.columns = ['beaconId', 'X-Coordinate', 'Y-Coordinate']

df = df.reset_index()
df = df.fillna(0)
#df = df.merge(datf,how='inner',on='beaconId')



temp = df.head(100)
print(df.head(10))
temp.to_csv('distancedShow.csv')
print("Shape of DF = ")
print(df.shape)
df.to_csv('myfileDistanced.csv')
