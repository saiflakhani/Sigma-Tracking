import pandas
import json
import numpy as np
import ctypes
from scipy import stats

#df = pandas.read_json('meow.json')

data = json.load(open('meow.json'))
df = pandas.io.json.json_normalize(data)
print(df)
#df = df[df.rssi != '00']
#df = df[df.rssiVal != '00000']
#df = df[~df['rssiVal'].isnull()]

# Y contained some other garbage, so null check was not enough
#df = df[df['rssiVal'].str.isnumeric()]
#df = df[df.beaconId != '0000']
#df['rssiVal'] = df['rssiVal'].str.extract('(\d+)', expand=False)
#df['rssiVal'] = pandas.to_numeric(df.rssiVal, errors='ignore')
#df = df.sort_values(by=['beaconId'])
#df = df.reset_index()

df["beaconId"] = df["beaconId"].astype('category')
df["beaconId"] = df["beaconId"].cat.codes
df['timeStamp'] = df['timeStamp.$date']
df['timeStamp'] = pandas.to_datetime(df['timeStamp'],yearfirst=True,utc=True)
df['timeStamp'] = df['timeStamp'].dt.floor('1min')
df = pandas.pivot_table(df,index=['timeStamp','beaconId'],columns=['relayNo'],values='rssi',aggfunc=np.mean)
df = df.reset_index()
df = df.fillna(0)
print(df)
df.to_csv('myfile.csv')