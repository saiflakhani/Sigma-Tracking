import pandas
import json

df = pandas.read_json('tagsmasters.json')
df = df[df.rssiVal != '00']
df = df[df.rssiVal != '00000']
df = df[df.beaconId != '0000']
df['rssiVal'] = df['rssiVal'].str.extract('(\d+)', expand=False)
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df = df.sort_values(by=['beaconId'])
df = df.reset_index()
#print(df.beaconId)

#df.to_csv("updatedCSV")
end = df.beaconId.count()
#print(end)
counter = 0
index = 0
finalObject = []
takenKeys = []

while index<len(df):
    beacon = df.loc[index].beaconId
    currenttDataFrame = df.loc[df['beaconId'] == beacon]
    skipBy = currenttDataFrame.beaconId.count()
    #currenttDataFrame = currenttDataFrame.sort_values(by=['relayNo'])
    #currenttDataFrame = currenttDataFrame.reset_index()
    takenKeys = []
    j=0
    print('Now processing beacon : ' + beacon + "| Final size = " + str(len(finalObject)))
    print("Current length = "+str(len(currenttDataFrame)))

    for y in range(1,24):
        if y==5 or y==10 or y==22:
            continue
        else:
            thisRelay = y
            if()






    while j<len(currenttDataFrame):
        foundFive = 0
        myDict = {}
        myRelays = []
        print('j = '+str(j))
        for k in range(j+1,len(currenttDataFrame)):
            key = currenttDataFrame.loc[k].relayNo
            if currenttDataFrame.loc[k].relayNo in myRelays:
                continue
            else:
                if currenttDataFrame.loc[k]._id not in takenKeys:
                    myRelays.append(key)
                    foundFive = foundFive+1
                    key = currenttDataFrame.loc[k].relayNo
                    myDict[key] = currenttDataFrame.loc[k].rssiVal
                    myDict['beaconId'] = beacon
                    takenKeys.append(currenttDataFrame.loc[k]._id)
                    #foundAt.append(k)

        if foundFive>=5:
            print("Found five = "+str(foundFive))
            finalObject.append(myDict)
        else:
            break
        j= j+1

    index = index + skipBy
    print(finalObject)

















