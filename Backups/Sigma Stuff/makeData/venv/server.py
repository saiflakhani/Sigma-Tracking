# Load libraries
import pandas
from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools
from flask import Flask
from flask import request
import random
import json
import paho.mqtt.client as mqtt
from datetime import datetime

print('Connecting to MQTT..')

broker_address = "188.166.247.94"
client = mqtt.Client("Processor")  # create new instance
client.connect(broker_address)  # connect to broker
client.subscribe("sigma")
stupidList = []


def performPivot(df):
    # print(df)
    # df = (df.drop(['beaconId'], axis=1).join(df['beaconId'].apply(pandas.to_numeric, errors='coerce')))
    # df = (df.drop(['rssiVal'], axis=1).join(df['rssiVal'].apply(pandas.to_numeric, errors='ignore')))
    # df = df[~df['beaconId'].str.contains("[a-zA-Z]").fillna(False)]
    # print(df)
    # patternDel = "[a-zA-Z]"
    # filter = df['beaconId'].str.contains(patternDel)
    # df = df[~filter]
    df = df.reindex(sorted(df.columns), axis=1)
    df["beaconId"] = df["beaconId"].astype(int)
    df["relayNo"] = df["relayNo"].astype(int)
    print(df)
    df['timeStamp'] = pandas.to_datetime(df['timeStamp'], yearfirst=True, utc=True)
    df['timeStamp'] = df['timeStamp'].dt.floor('1min')
    # print(df)
    df = pandas.pivot_table(df, index=['timeStamp', 'beaconId'], columns=['relayNo'], values='rssiVal', aggfunc=np.mean)


# df = df.drop([0,65,25],axis=1)
# print(df)

def twos_comp(val, bits):
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val


def on_message(client, userdata, message):
    raw = str(message.payload.decode("utf-8"))
    # print(raw)
    # d="}{"
    # s =  [e+d for e in raw.split(d) if e]
    # print(s)
    # for rawstring in s:
    try:
        msg = json.loads(raw)
    except e:
        print(e)
        return
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
        # print(currentObject)
        currentObject['timeStamp'] = str(datetime.now())
        # print(currentObject)
        stupidList.append(currentObject)
        # print(len(stupidList))
        # print(currentObject)
        # print('Gathering data : '+str(len(stupidList))+'%')
        if (len(stupidList) == 100):
            print('Array Achieved. Making dataframe')
            testFrame = pandas.DataFrame(stupidList)
            # print(testFrame)
            stupidList.clear()
            performPivot(testFrame)


client.loop_start()
client.on_message = on_message

print('Now creating Random Forest Classifier')
url = "myfile.csv"
dataset = pandas.read_csv(url)


def generator(start, end):
    return random.randint(start, end)


array = dataset.values
X = array[1:, 3:26]
Y = array[1:, 2]
Y = Y.astype('int')
validation_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

classifier = RandomForestClassifier()
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_validation)
print("Accuracy of Random Forest : ", accuracy_score(Y_validation, predictions))

app = Flask(__name__)
myArray = []
arrayOfRelays = {}


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
        print(df)
        myArr = df.loc[0]

        # df = df.sort_values(by=['beaconId'])
        predictions = classifier.predict([myArr])
        print(predictions)

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
                print(currentElement)
            except:
                currentElement['random'] = True

        myArray.append(currentElement)
    return json.dumps(myArray)


#if __name__ == '__main__':
    # app.run("188.166.247.94",50034)
    #app.run()