# Load libraries
import pandas
from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
from scipy import stats
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools
from scipy.spatial import distance
from itertools import combinations
from flask import Flask
from flask import request
import random
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import paho.mqtt.client as mqtt
from datetime import datetime
from numpy import argmax
import _pickle as cPickle

# from keras.utils import to_categorical

lookupSeries = []
app = Flask(__name__)
stupidList = []
permanentCords = pandas.read_csv('relayCords.csv')
print('Now creating Performing Prediction based on regression model')
try:
    with open('regressor_1_march.pkl', 'rb') as fid:
        regressor = cPickle.load(fid)
        print("Opened Regressor Successfully")
except:
    print("Regressor not found")
    exit(1)


def calculateDistancesFromRSSI(listOfThree):
    distances = []
    ##CONSTANTS
    measuredPower = -76.0  # -80
    n = 2.5  # 2.1
    ##END CONSTANTS
    for i in range(0, 3):
        rssi = float(listOfThree[i]['rssi'])
        poly_features = PolynomialFeatures(degree=3)
        rssi = [[rssi]]
        # transforms the existing features to higher degree features.
        higher = poly_features.fit_transform(rssi)
        # rssi = -1*rssi
        # print(rssi)

        mdistance = regressor.predict(higher)

        # print("Predicted Distance = ",mdistance[0][0])
        # mdistance = 10.0**((measuredPower-rssi)/(10.0*n))
        distances.append(mdistance[0][0])
    return distances


def calculateTrilateration(listofThree):
    global permanentCords
    global invalidVal
    permX = []
    permY = []
    finalCords = []
    ## WHAT ARE THE PERMANENT COORDINATES OF THE THREE RELAYS?
    for i in range(0, 3):
        filter = listofThree[i]['relay'] == permanentCords['Relay']
        permX.append(float(permanentCords[filter]['XCord'].tolist()[0]))
        permY.append(float(permanentCords[filter]['YCord'].tolist()[0]))
        # print("Relay = "+str(listofThree[i]['relay'])+", X1 = "+str(x1.tolist()[0])+", Y1 = "+str(y1.tolist()[0]))

    distancesList = calculateDistancesFromRSSI(listofThree)

    distances = []
    # same = []
    for idx, element in enumerate(distancesList):
        if element in distances:
            distances.append(float(float(element) + 1))
        else:
            distances.append(element)
    # print(distances)

    #### MATH STARTS HERE ####
    A = (-2 * permX[0]) + (2 * permX[1])
    B = (-2 * permY[0]) + (2 * permY[1])
    C = (distances[0] ** 2) - (distances[1] ** 2) - (permX[0] ** 2) + (permX[1] ** 2) - (permY[0] ** 2) + (
                permY[1] ** 2)
    D = (-2 * permX[1]) + (2 * permX[2])
    E = (-2 * permY[1]) + (2 * permY[2])
    F = (distances[1] ** 2) - (distances[2] ** 2) - (permX[1] ** 2) + (permX[2] ** 2) - (permY[1] ** 2) + (
                permY[2] ** 2)
    calculatedX = 0
    calculatedY = 0
    try:
        calculatedX = ((C * E) - (F * B)) / ((E * A) - (B * D))
        calculatedY = ((C * D) - (A * F)) / ((B * D) - (A * E))
    except:
        print("Division by Zero. Resetting to Zero")
        return [0, 0]

    # print("COORDINATES ARE : ",calculatedX,",",calculatedY)
    if np.isnan(calculatedX):
        #invalidVal = invalidVal + 1
        return [0, 0]
    finalCords.append(calculatedX)
    finalCords.append(calculatedY)
    return finalCords


def performCombinations(givenList):
    count = 0
    sumX = 0
    sumY = 0
    averageList = []
    for threeComb in list(combinations(givenList, 3)):
        finalCords = calculateTrilateration(threeComb)
        count = count + 1
        sumX = sumX + finalCords[0]
        sumY = sumY + finalCords[1]
    averageX = sumX / count
    averageY = sumY / count
    averageList.append(averageX)
    averageList.append(averageY)
    return (averageList)


def calculateEuclidianDistance(x1, y1, x2, y2):
    p1 = (x1, y1)
    p2 = (x2, y2)
    mdistance = distance.euclidean(p1, p2)
    return mdistance


def generator(start, end):
    return random.randint(start, end)
    
    
def splitName(name):
  return pandas.Series(name.split(",")[0:2])




def performOperationsonEachRow(df):
    global lookupSeries
    listOfDetectedRelays = []
    cordsList = [0,0]
    for i in range(1, 24):
        if i == 22:
            continue
        if (i not in df):
            df[i] = 0
        if int(df[str(i)]) is not 0:
            thisRow = {}
            thisRow['relay'] = i
            thisRow['rssi'] = df[str(i)]
            if(float(thisRow['rssi'])<80 or float(thisRow['rssi'])>101):
                continue
            listOfDetectedRelays.append(thisRow)
    if len(listOfDetectedRelays) > 3:
        cordsList = performCombinations(listOfDetectedRelays)
    elif len(listOfDetectedRelays) is 3:
        cordsList = calculateTrilateration(listOfDetectedRelays)

    if cordsList == [0,0]:
        return
    return (str(cordsList[0])+","+str(cordsList[1]))


@app.route('/test', methods=['GET', 'POST'])
def predictRegressionValues():
    testSet = pandas.read_csv('intermediate.csv')
    testSet['cords'] = testSet.apply(performOperationsonEachRow, axis=1)
    #lookupFrame = pandas.DataFrame(lookupSeries)
    lookupFrame = testSet[['beaconId','cords']]
    lookupFrame = lookupFrame.dropna()
    lookupFrame['id'] = lookupFrame['beaconId']
    print(lookupFrame)
    lookupFrame = lookupFrame.drop(['beaconId'],axis=1)
    lookupFrame[['x', 'y']] = lookupFrame.apply(lambda x: splitName(x['cords']), axis=1)
    lookupFrame = lookupFrame.drop(['cords'],axis=1)
    lookupFrame = lookupFrame[['id','x','y']]
    lookupFrame.to_csv('lookupTable.csv')
    print('Updated : ' + str(datetime.now()))
    return("<h1>OK! 200 </h1>")




def performPivot(df):
    patternDel = "[a-zA-Z]"
    filter = df['beaconId'].str.contains(patternDel)
    df = df[~filter]
    df = df.reindex(sorted(df.columns), axis=1)
    df["beaconId"] = df["beaconId"].astype(int)
    df["relayNo"] = df["relayNo"].astype(int)
    # print(df)
    df['timeStamp'] = pandas.to_datetime(df['timeStamp'], yearfirst=True, utc=True)
    df['timeStamp'] = df['timeStamp'].dt.floor('30S')
    df = pandas.pivot_table(df, index=['timeStamp', 'beaconId'], columns=['relayNo'], values='rssiVal',
                            aggfunc=(lambda x: stats.mode(x)[0][0]))
    for i in range(1, 25):
        if (i == 22):
            continue
        if (i not in df):
            df[i] = 0
    df = df.reindex(sorted(df.columns), axis=1)
    # df = df.drop([24],axis=1)
    df = df.fillna(0.0)
    # print(df)
    df.to_csv('intermediate.csv')
    predictRegressionValues()


def twos_comp(val, bits):
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val


def performEntry(raw):
    # raw = '{"Company":"Sigma","Relay Number": "14", "Detected Nodes":"0307:b1,f703:03,1e50:0c,0020:00,0000:00,0311:00"}'
    try:
        msg = json.loads(raw)
    except ex:
        print(ex)
    relayNumber = msg['Relay Number']
    detectedNodes = msg['Detected Nodes']
    allNodes = detectedNodes.split(",")
    for node in allNodes:
        currentObject = {}
        rsSplit = node.split(":")
        beaconId = rsSplit[0]
        rssi = rsSplit[1]
        rssiVal = int(rssi, 16)
        rssiVal = abs(twos_comp(rssiVal, 8))
        currentObject['relayNo'] = relayNumber
        currentObject['beaconId'] = beaconId
        currentObject['rssiVal'] = rssiVal
        currentObject['timeStamp'] = str(datetime.now())
        # print(currentObject)
        stupidList.append(currentObject)
        # print(len(stupidList))
        # print(currentObject)
        # print('Gathering data : '+str(len(stupidList))+'%')
        if (len(stupidList) == 1000):
            print('Array Achieved. Making dataframe')
            testFrame = pandas.DataFrame(stupidList)
            # print(testFrame)
            stupidList.clear()
            performPivot(testFrame)
    return ('hello')


def on_message(client, userdata, message):
    raw = str(message.payload.decode("utf-8"))
    #print(raw)
    d = "}"
    s = [e + d for e in raw.split(d) if e]

    # print(s)

    for messages in s:
        #print(messages)
        performEntry(messages)


myArray = []
arrayOfRelays = {}


@app.route('/getcordById', methods=['GET', 'POST'])
def getCordById():
    currentFrame = pandas.read_csv('lookupTable.csv')
    data = request.get_json()
    returnArray = []

    for element in data:
        ob = {}
        # print(element)
        filter = currentFrame["id"] == element
        ob['id'] = element
        ob['x'] = generator(0, 100)
        ob['y'] = generator(0, 25)
        ob['random'] = True
        if (currentFrame[filter]['x'] is not None):
            try:
                x = currentFrame[filter]['x'].tolist()
                ob['x'] = x[0]
                y = currentFrame[filter]['y'].tolist()
                ob['y'] = y[0]
                ob['random'] = False
            # print(lookupObject)
            except:
                ob['random'] = True

        returnArray.append(ob)

    return json.dumps(returnArray)


@app.route('/getcord', methods=['GET', 'POST'])
def hello_world():
    data = request.get_json()
    retrun = ""
    myArray = []
    for element in data:
        currentElement = {}
        relayArray = element['relays']
        # df = pandas.DataFrame.from_items(relayArray,index="relay")
        # print(df)
        arrayOfRelays = {}
        for x in relayArray:
            temp = x['relayNo']
            rssiVal = x['rssiVal']
            arrayOfRelays[temp] = rssiVal
        df = pandas.Series(arrayOfRelays).to_frame()
        df = df.transpose()
        df = df.reindex(sorted(df.columns), axis=1)
        # print(df)
        myArr = df.loc[0]

        # df = df.sort_values(by=['beaconId'])
        predictions = classifier.predict([myArr])
        # print(predictions)

        datf = pandas.read_csv('tags.csv')
        filter = datf["Tag id"] == predictions[0]
        # print(datf[filter]['X-Coordinate'])
        # df.columns.values = ['relayNo','rssiVal']
        currentElement['id'] = element['_id']
        currentElement['x'] = generator(0, 100)
        currentElement['y'] = generator(0, 25)
        if (datf[filter]['x'] is not None):
            try:
                x = datf[filter]['x'].tolist()
                currentElement['x'] = x[0]
                y = datf[filter]['y'].tolist()
                currentElement['y'] = y[0]
            # print(currentElement)
            except:
                currentElement['random'] = True

        myArray.append(currentElement)
    return json.dumps(myArray)


print('Connecting to MQTT..')

broker_address = "188.166.247.94"
id = "process" + str(generator(0, 1000))
client = mqtt.Client(id)  # create new instance
client.connect(broker_address)  # connect to broker
client.subscribe("sigma")
client.on_message = on_message
client.loop_start()

if __name__ == '__main__':
    app.run("188.166.247.94", 50034)
    #app.run()