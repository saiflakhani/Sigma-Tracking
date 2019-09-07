import pandas

df = pandas.read_csv('medianSeparated.csv')
#filter = df.beaconId==193 or df.beaconId==170 or df.beaconId==850
#df = df[df.beaconId==872]
#df.to_csv('temporaryMedianTesting.csv')
df = df[df.withinMedian==True]

#df['relayCountPerBeacon'] = df['relayNo'].groupby(df['beaconId']).value_counts()

#df = df.groupby(['beaconId','relayNo'], as_index=False)['relayNo'].count()

df = df.groupby(df['beaconId'])['relayNo'].apply(lambda x: x.value_counts().head(3))
print(df)


#print(s.groupby(level=[1,0]).nlargest(3))
#df = df.groupby(['beaconId']).relayNo.value_counts().nlargest(4)
#print(df)
df.to_csv('top3relays.csv')