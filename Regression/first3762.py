import pandas

df= pandas.read_csv('distancedFile.csv')
meow = df[df['beaconId']==193]
meow.to_csv('first3762.csv')