from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import _pickle as cPickle
from sklearn.datasets import load_boston


#boston_dataset = load_boston()
#df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
#df['MEDV'] = boston_dataset.target


#X = df['LSTAT'].values
#Y = df['MEDV'].values
#print(X.shape)
#print(Y.shape)

matplotlib.rcParams['agg.path.chunksize'] = 10000
df = pd.read_csv('distancedFile.csv')
df['rssiVal'] = df['rssiVal'].astype(int)
df['distance'] = df['distance'].astype(float)


df = df.drop(df[df.rssiVal>101].index)
df = df.drop(df[df['distance']>30.0].index)
df = df.drop(df[df['distance']<3.0].index)
print(df.shape)
df.to_csv('cleaned_1_march.csv')


array = df.values
X = array[1:, 3:5]
Y = array[1:, 10]

#X = df['rssiVal'].values
#Y = df['distance'].values
#X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=7)





def create_linear_regression_model():
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(X_train, Y_train)
    # Predict
    y_predicted = regression_model.predict(X_test)

    # model evaluation
    rmse = mean_squared_error(Y_test, y_predicted)
    r2 = r2_score(Y_test, y_predicted)
    plt.scatter(X_train, Y_train)
    plt.xlabel('X')
    plt.ylabel('Y')

    # predicted values
    plt.plot(X_test, y_predicted, color='r')
    plt.show()


def create_polynomial_regression_model(degree):
    # "Creates a polynomial regression model for the given degree"

    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Y_train)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
    r2_train = r2_score(Y_train, y_train_predicted)

    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2_test = r2_score(Y_test, y_test_predict)

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    with open('regressor_1_march.pkl', 'wb') as fid:
        cPickle.dump(poly_model, fid)
    plt.scatter(X_train,Y_train,s=10)
    plt.plot(X_train,y_train_predicted,color='red')
    plt.show()



create_polynomial_regression_model(3)
#create_linear_regression_model()