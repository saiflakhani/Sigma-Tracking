# Load libraries
import pandas
from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
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
from scipy import stats
import random
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import paho.mqtt.client as mqtt
from datetime import datetime
from numpy import argmax
import _pickle as cPickle
from prettytable import PrettyTable
from tabulate import tabulate
#import matplotlib.pyplot as plt
#from keras.utils import to_categorical




app = Flask(__name__)
stupidList = []    
print('Now creating Random Forest Classifier')



try:
    with open('dumped_classifier.pkl', 'rb') as fid:
        classifier = cPickle.load(fid) 
        print("Opened Classifier Successfully")                                                                             
except:
    print('Classifier not found. Creating.')
    url = "myfileMedian2min.csv"
    dataset = pandas.read_csv(url)
    print(dataset.shape)
    dataset['beaconId'] = dataset['beaconId'].astype('category')
    print (dataset.dtypes)
    array = dataset.values
    X = array[1:, 3:26]
    Y = array[1:, 2]
    Y = Y.astype('int')         
    validation_size = 0.20
    seed = 8
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)                                                                   
    classifier = RandomForestClassifier(n_estimators=50)
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_validation)
    matrix = confusion_matrix(Y_validation, predictions)
    matrix = pandas.DataFrame(matrix)
    matrix = matrix.values
    matrix = matrix[:30,:30]
    print(tabulate(matrix, tablefmt='psql'))
    #matrix = matrix[:200,:]
    #plt.matshow(matrix)
    #print(matrix)
    
    print("Accuracy of Random Forest : ",accuracy_score(Y_validation, predictions))
    #with open('dumped_classifier.pkl', 'wb') as fid:
    #    cPickle.dump(classifier, fid)


#predictions = classifier.predict(X_validation)
#print("Accuracy of Random Forest : ",accuracy_score(Y_validation, predictions))


def generator(start, end):
    return random.randint(start, end)
    
mediandf = []    
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




#eraseCount = 0
@app.route('/test', methods=['GET', 'POST'])
def predictValues():
    predSet = pandas.read_csv('intermediate.csv')
    predictionSet = predSet.values
    valX = predictionSet[:, 2:]
    mypred = classifier.predict(valX)
    print("Predictions == ")
    print(mypred)
    #class_labels1 = np.argmax(mypred, axis=1)
    #print(class_labels1)
    lookupObject = {}
    lookupArray = []
    counter = 0
    datf = pandas.read_csv('xycord.csv')
    for fingerprinted in mypred:
        lookupObject = {}
        filter = datf["Tag id"]==fingerprinted
        lookupObject['id'] = predictionSet[counter,1]
        lookupObject['x'] = generator(0, 100)
        lookupObject['y'] = generator(0, 25)
        lookupObject['random'] = True
        if(datf[filter]['X'] is not None):
            try:
                x = datf[filter]['X'].tolist()
                lookupObject['x'] = x[0]
                y = datf[filter]['Y'].tolist()
                lookupObject['y'] = y[0]
                lookupObject['random'] = False
				#print(lookupObject)
            except:
                lookupObject['random'] = True
        lookupArray.append(lookupObject)
        counter = counter+1
    lookupFrame = pandas.DataFrame(lookupArray)
    existingFrame = pandas.read_csv('lookupTable.csv')
	## NOW MERGING ##
    res = pandas.concat([existingFrame,lookupFrame])
    res.drop_duplicates(subset=['id'], inplace=True, keep='last')
    cols = [c for c in res.columns if c.lower()[:7] != 'unnamed']
    res = res[cols]
    print(res)
    print('Updated : '+str(datetime.now()))
    res.to_csv('lookupTable.csv')

@app.route('/pivotTest', methods=['GET', 'POST'])
def performPivot():
    global mediandf
    df = pandas.read_csv('beforePivot.csv')
    patternDel = "[a-zA-Z]"
    filter = df['beaconId'].str.contains(patternDel)
    df = df[~filter]
    df = df.reindex(sorted(df.columns), axis=1)
    df["beaconId"] = df["beaconId"].astype(int)
    df["relayNo"] = df["relayNo"].astype(int)
    df = df[df.beaconId < 1000]
    df = df[df.beaconId > 0]
    df = df[df.rssiVal < 95]
    df = df[df.rssiVal > 80]
    mediandf = df.groupby(['beaconId','relayNo'], as_index=False)['rssiVal'].median()
    df['withinMedian'] = df.apply(compareRSSIToMedian,axis=1)
    df = df[df.withinMedian==True]
    #top6df = df.groupby(df['beaconId'])['relayNo'].apply(lambda x: x.value_counts().head(6))
    #top6df.to_csv('top3relays.csv')
    #top6df = pandas.read_csv('top3relays.csv')
    #top6df.columns = ['beaconId','relayNo','count']
    #df = df.merge(top6df,how='inner',on=['beaconId','relayNo'])
    df['timeStamp'] = pandas.to_datetime(df['timeStamp'],yearfirst=True,utc=True)
    df['timeStamp'] = df['timeStamp'].dt.floor('30S')
    df = pandas.pivot_table(df,index=['timeStamp','beaconId'],columns=['relayNo'],values='rssiVal',aggfunc=np.median)
    #print(df)
    for i in range(1,25):
        if(i==22):
            continue
        if(i not in df):
            df[i] = 0
    df = df.reindex(sorted(df.columns), axis=1)
	#df = df.drop([24],axis=1)
    df = df.fillna(0.0)
    #datf = pandas.read_csv('xycord.csv')
    #datf.columns = ['beaconId','X-Coordinate','Y-Coordinate']
    #df = df.reset_index()
    #df = df.fillna(0)
    #df = df.merge(datf,how='inner',on='beaconId')
    df.to_csv('intermediate.csv')
    predictValues()
	
def twos_comp(val, bits):
	if (val & (1 << (bits - 1))) != 0:
		val = val - (1 << bits)
	return val
	
	
	

def performEntry(raw):
	#raw = '{"Company":"Sigma","Relay Number": "14", "Detected Nodes":"0307:b1,f703:03,1e50:0c,0020:00,0000:00,0311:00"}'
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
		rssiVal = int(rssi,16)
		rssiVal = abs(twos_comp(rssiVal,8))
		currentObject['relayNo'] = relayNumber
		currentObject['beaconId'] = beaconId
		currentObject['rssiVal'] = rssiVal
		currentObject['timeStamp'] = str(datetime.now())
		#print(currentObject)
		stupidList.append(currentObject)
    	#print(len(stupidList))
    	#print(currentObject)
		#print('Gathering data : '+str(len(stupidList)/200)+'%')
		if(len(stupidList)==5000):
			print('Array Achieved. Making dataframe')
			testFrame = pandas.DataFrame(stupidList)
			testFrame.to_csv('beforePivot.csv')
			stupidList.clear()
			performPivot()
	return('hello')



def on_message(client, userdata, message):
    	raw = str(message.payload.decode("utf-8"))
    	#print(raw)
    	d = "}"
    	s =  [e+d for e in raw.split(d) if e]
    	
    	#print(s)
    	
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
		#print(element)
		filter = currentFrame["id"]==element
		ob['id'] = element
		ob['x'] = generator(0, 100)
		ob['y'] = generator(0, 25)
		ob['random'] = True
		if(currentFrame[filter]['x'] is not None):
			try:
				x = currentFrame[filter]['x'].tolist()
				ob['x'] = x[0]
				y = currentFrame[filter]['y'].tolist()
				ob['y'] = y[0]
				ob['random'] = False
				#print(lookupObject)
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
        #df = pandas.DataFrame.from_items(relayArray,index="relay")
        #print(df)
        arrayOfRelays = {}
        for x in relayArray:
        	temp = x['relayNo']
        	rssiVal = x['rssiVal']
        	arrayOfRelays[temp] = rssiVal
        df = pandas.Series(arrayOfRelays).to_frame()
        df = df.transpose()
        df = df.reindex(sorted(df.columns), axis=1)
        #print(df)
        myArr = df.loc[0]
        
        #df = df.sort_values(by=['beaconId'])
        predictions = classifier.predict([myArr])
        #print(predictions)
        
        datf = pandas.read_csv('tags.csv')
        filter = datf["Tag id"]==predictions[0]
        #print(datf[filter]['X-Coordinate'])
        #df.columns.values = ['relayNo','rssiVal']
        currentElement['id'] = element['_id']
        currentElement['x'] = generator(0, 100)
        currentElement['y'] = generator(0, 25)
        if(datf[filter]['x'] is not None):
        	try:
        		x = datf[filter]['x'].tolist()
        		currentElement['x'] = x[0]
        		y = datf[filter]['y'].tolist()
        		currentElement['y'] = y[0]
        		#print(currentElement)
        	except:
        		currentElement['random'] = True
        
        myArray.append(currentElement)
    return json.dumps(myArray)


print('Connecting to MQTT..')

broker_address="188.166.247.94" 
id = "process"+str(generator(0,1000))
client = mqtt.Client(id) #create new instance
client.connect(broker_address) #connect to broker
client.subscribe("sigma")
client.on_message=on_message
client.loop_start()


if __name__ == '__main__':
    #app.run("188.166.247.94",50034)
    app.run()