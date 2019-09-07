import pandas
import math
from itertools import combinations
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import _pickle as cPickle
import numpy as np


invalidVal = 0
permanentCords = pandas.read_csv('relayCords.csv')
df = pandas.read_csv('myfileDistanced.csv')
df = df.fillna(0)
#df = df.fillinf(0)
try:
    with open('regressor_1_march.pkl', 'rb') as fid:
        regressor = cPickle.load(fid)
        print("Opened Regressor Successfully")
except:
    print("Regressor not found")
    exit(1)
#df = df.sample(frac=1).reset_index(drop=True)


def calculateDistancesFromRSSI(listOfThree):
    distances = []
    ##CONSTANTS
    measuredPower = -76.0 #-80
    n = 2.5 #2.1
    ##END CONSTANTS
    for i in range(0,3):
        rssi = float(listOfThree[i]['rssi'])
        poly_features = PolynomialFeatures(degree=3)
        rssi = [[rssi]]
        # transforms the existing features to higher degree features.
        higher = poly_features.fit_transform(rssi)
        #rssi = -1*rssi
        #print(rssi)

        mdistance = regressor.predict(higher)

        #print("Predicted Distance = ",mdistance[0][0])
        #mdistance = 10.0**((measuredPower-rssi)/(10.0*n))
        distances.append(mdistance[0][0])
    return distances
        

def calculateTrilateration(listofThree):
    global permanentCords
    global invalidVal
    permX = []
    permY = []
    finalCords = []
    ## WHAT ARE THE PERMANENT COORDINATES OF THE THREE RELAYS?
    for i in range(0,3):
        filter = listofThree[i]['relay']==permanentCords['Relay']
        permX.append(float(permanentCords[filter]['XCord'].tolist()[0]))
        permY.append(float(permanentCords[filter]['YCord'].tolist()[0]))
        #print("Relay = "+str(listofThree[i]['relay'])+", X1 = "+str(x1.tolist()[0])+", Y1 = "+str(y1.tolist()[0]))
        
    distancesList = calculateDistancesFromRSSI(listofThree)

    distances = []
    #same = []
    for idx, element in enumerate(distancesList):
        if element in distances:
            distances.append(float(float(element)+1))
        else:
            distances.append(element)
    #print(distances)
    
    #### MATH STARTS HERE ####
    A = (-2*permX[0])+(2*permX[1])
    B = (-2*permY[0])+(2*permY[1])
    C = (distances[0]**2) - (distances[1]**2) - (permX[0]**2) + (permX[1]**2) - (permY[0]**2) + (permY[1]**2)
    D = (-2*permX[1])+(2*permX[2])
    E = (-2*permY[1])+(2*permY[2])
    F = (distances[1]**2) - (distances[2]**2) - (permX[1]**2) + (permX[2]**2) - (permY[1]**2) + (permY[2]**2)
    calculatedX = 0
    calculatedY = 0
    try:
        calculatedX = ((C*E)-(F*B))/((E*A)-(B*D))
        calculatedY = ((C*D)-(A*F))/((B*D)-(A*E))
    except:
        print("Division by Zero. Resetting to Zero")
        return [0,0]

    #print("COORDINATES ARE : ",calculatedX,",",calculatedY)
    if np.isnan(calculatedX):
        invalidVal = invalidVal+1
        return [0,0]
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
        count = count+1
        sumX = sumX+finalCords[0]
        sumY = sumY+finalCords[1]
    averageX = sumX/count
    averageY = sumY/count
    averageList.append(averageX)
    averageList.append(averageY)
    return(averageList)
    
    
    
def calculateEuclidianDistance(x1,y1,x2,y2):
    p1 = (x1,y1)
    p2 = (x2,y2)
    mdistance = distance.euclidean(p1, p2)
    return mdistance  
    
    
def Average(lst): 
    return sum(lst) / len(lst) 
   
######## MAIN FUNCTION STARTS ########### 
spaces = []
rejectedCount = 0
totalCount = 0
for index, row in df.iterrows():
    totalCount = totalCount + 1
    listOfDetectedRelays = []
    for i in range(1,25):
        if i==22:
            continue
        if int(row[str(i)]) is not 0:
            thisRow = {}
            thisRow['relay'] = i
            thisRow['rssi'] = row[str(i)]
            listOfDetectedRelays.append(thisRow)
    if len(listOfDetectedRelays)>3:
        cordsList = performCombinations(listOfDetectedRelays)
        actualX = float(row['X-Coordinate'])
        actualY = float(row['Y-Coordinate'])
        if (cordsList[0] is 0 and cordsList[1] is 0) or (np.isnan(cordsList[0]) or np.isnan(cordsList[1] or np.isinf(cordsList[1]) or np.isinf(cordsList[0]))):
            continue
        space = calculateEuclidianDistance(cordsList[0],cordsList[1],actualX,actualY)
        #print("Detected Space = "+str(space)+" meters")

        if(space>22):
            rejectedCount = rejectedCount+1
            continue
        spaces.append(space)
        #totalCount = totalCount+1
        if(len(spaces)==100):
            print("AVG Space = "+str(Average(spaces)))
            print("Rejected = "+str((rejectedCount/totalCount)*100)+"%")
            print("Invalids = " + str((invalidVal/totalCount)*100)+"%")
            print("--")
            plt.plot(spaces)
            plt.show()
            spaces = []
            totalCount = 0
            rejectedCount = 0
            break
    elif len(listOfDetectedRelays) is 3:
        cordsList = calculateTrilateration(listOfDetectedRelays)
        actualX = float(row['X-Coordinate'])
        actualY = float(row['Y-Coordinate'])
        space = calculateEuclidianDistance(cordsList[0],cordsList[1],actualX,actualY)
        #print("Detected Space = " + str(space) + " meters")
        if(space>22):
            #totalCount = totalCount+1
            rejectedCount = rejectedCount+1
            continue
        if (cordsList[0] is 0 and cordsList[1] is 0) or (np.isnan(cordsList[0]) or np.isnan(cordsList[1] or np.isinf(cordsList[1]) or np.isinf(cordsList[0]))):
            continue
        spaces.append(space)
        #totalCount = totalCount+1
        if(len(spaces)==20):
            print("AVG Space = "+str(Average(spaces)))
            print("Rejected = "+str((rejectedCount/totalCount)*100)+"%")
            print("Invalids = "+str(invalidVal))
            print("--")
            spaces = []
            totalCount = 0
            rejectedCount = 0
        #print("Detected Space = "+str(space)+" meters")
    else:
        #print("Too few elements for trilateration")
        continue