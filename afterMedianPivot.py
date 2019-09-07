import pandas
import numpy as np

df = pandas.read_csv('medianSeparated.csv')
df = df[df.withinMedian==True]
#top3df = pandas.read_csv('top3relays.csv')
#top3df.columns = ['beaconId','relayNo','count']
#df = df.merge(top3df,how='inner',on=['beaconId','relayNo'])

datf = pandas.read_csv('xycoordinates.csv')
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
df.to_csv('myfileMedian2min.csv')