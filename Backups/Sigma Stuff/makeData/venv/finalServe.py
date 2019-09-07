# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import itertools

#url = "myfile.csv"
#dataset = pandas.read_csv(url)


def generator(start, end):
    return random.randint(start, end)


#array = dataset.values
#X = array[:, 3:]
#Y = array[:, 2]
#Y = Y.astype('int')
#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
#                                                                                random_state=seed)


#classifier = SVC()
#classifier.fit(X_train, Y_train)

#myArray = [[0, 87, 0, 0, 0, 0, 0, 0, 0, 0, 91, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#predictions = classifier.predict(myArray)
#print(predictions)

myArray = []


@app.route('/getcord', methods=['GET', 'POST'])
def hello_world():
    data = request.get_json()
    retrun = ""
    for element in data:
        currentElement = {}
        relayArray = element['relays']
        df = pandas.read_json(relayArray)
        print(df)
        currentElement['id'] = element['_id']
        currentElement['x'] = generator(0, 100)
        currentElement['y'] = generator(0, 25)
        myArray.append(currentElement)
    return json.dumps(myArray)


if __name__ == '__main__':
    app.run("188.166.247.94", 54003)



