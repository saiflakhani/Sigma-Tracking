import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
df = pandas.read_csv('myFileDistanced.csv')
print('Now creating Random Forest Classifier')

array = df.values
X = array[1:, 8:]
Y = array[1:, 2]
Y = Y.astype('int')
validation_size = 0.33
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_validation)
print(predictions)

print("Accuracy of Random Forest : ",accuracy_score(Y_validation, predictions))




#df['distancedBetweenPredictedandActual'] = df.apply