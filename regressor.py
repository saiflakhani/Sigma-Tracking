import numpy
import matplotlib.pyplot as plot
import pandas
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

# Import the dataset
dataset = pandas.read_csv('myfileRegression.csv')


dataset = dataset[dataset['rssiVal']<101]
dataset['relayNo'] = dataset['relayNo'].astype('category')
x = dataset.iloc[:, 4:6].values
y = dataset.iloc[:, 7:].values

print(dataset.dtypes)

# Split the dataset into the training set and test set
# We're splitting the data in 1/3, so out of 30 rows, 20 rows will go into the training set,
# and 10 rows will go into the testing set.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 7)

linearRegressor = MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=-1)

linearRegressor.fit(xTrain, yTrain)

yPrediction = linearRegressor.predict(xTest)

#print(mean_absolute_error(yTrain,yPrediction))

print(linearRegressor.predict([[7,88],[15,91],[16,87],[4,92],[17,90],[5,93],[6,85]]))
plot.rcParams['agg.path.chunksize'] = 1000
plot.scatter(xTrain, yTrain, color = 'red')
plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
plot.title('Tracking beacons')
plot.xlabel('RSSI and Relays')
plot.ylabel('Predicted Location')
plot.show()