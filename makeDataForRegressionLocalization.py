import pandas
import json
import numpy as np
from scipy import stats
import ctypes
from scipy.spatial import distance

def calculateEuclidianDistance(df):
    p1 = (df["X-Coordinate"], df["Y-Coordinate"])
    p2 = (df["relayX"], df["relayY"])
    mdistance = distance.euclidean(p1, p2)
    return mdistance



pandas.set_option('display.width', 1000)
data = json.load(open('Sigma.march1.json'))
df = pandas.io.json.json_normalize(data, errors='ignore')
datf = pandas.read_csv('xycord.csv')
relayCords = pandas.read_csv('relayCords.csv')
df = df.reindex(sorted(df.columns), axis=1)
df["beaconId"] = df["beaconId"].astype(int)
df["relayNo"] = df["relayNo"].astype(int)
datf.columns = ['beaconId', 'X-Coordinate', 'Y-Coordinate']
relayCords.columns = ['relayNo', 'relayX', 'relayY']
df = df.merge(datf,how='inner',on='beaconId')
df = df.merge(relayCords, how='inner', on='relayNo')
df["X-Coordinate"] = df["X-Coordinate"].astype(int)
df["Y-Coordinate"] = df["Y-Coordinate"].astype(int)
df["relayX"] = df["relayX"].astype(int)
df["relayY"] = df["relayY"].astype(int)
df['distance'] = df.apply(calculateEuclidianDistance,axis=1)

df.to_csv('distancedFile.csv')
