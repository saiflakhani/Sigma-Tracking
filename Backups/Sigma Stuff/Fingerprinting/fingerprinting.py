from pymongo import MongoClient
import json
import paho.mqtt.client as mqtt
from datetime import datetime



client = MongoClient("mongodb://localhost:30107/")
db = client["Sigma"]
col = db["sigmaFingerprinting"]



def twos_comp(val, bits):
	if (val & (1 << (bits - 1))) != 0:
		val = val - (1 << bits)
	return val

def enterMongo(raw):
	try:
		msg = json.loads(raw)
	except ex:
		print(ex)
	relayNumber = msg['Relay Number']
	detectedNodes = msg['Detected Nodes']
	allNodes = detectedNodes.split(",")
	#print(allNodes)
	for node in allNodes:
		currentObject = {}
		rsSplit = node.split(":")
		beaconId = rsSplit[0]
		rssi = rsSplit[1]
		rssiVal = int(rssi,16)
		rssiVal = abs(twos_comp(rssiVal,8))
		if(rssiVal==0):
			return
		currentObject['relayNo'] = relayNumber
		currentObject['beaconId'] = beaconId
		currentObject['rssiVal'] = rssiVal
		currentObject['timeStamp'] = str(datetime.now())
		result = col.insert(currentObject)
		#print("Inserted : "+str(beaconId))


def on_message(client, userdata, message):
		currentDict = {}
		raw = str(message.payload.decode("utf-8"))
		d = "}"
		s =  [e+d for e in raw.split(d) if e]
		print(len(s))
		for messages in s:
			#print(messages)
			enterMongo(messages)

print('Connecting to MQTT..')
broker_address="188.166.247.94" 
client = mqtt.Client("FingerprintingTask3") #create new instance
client.connect(broker_address) #connect to broker
client.subscribe("sigma")
client.on_message=on_message
client.loop_forever()