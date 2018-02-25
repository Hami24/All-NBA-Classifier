import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('allNba.csv')
x = dataset.iloc[:,2:7].values
y = dataset.iloc[:,-1].values

#Splitting into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature scaling
scaler =  StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Logistic regression model
nbaClassifier =  LogisticRegression(random_state = 0)
nbaClassifier.fit(X_train,Y_train)

#Prediction based on test set
Y_pred = nbaClassifier.predict(X_test)

#Classifier accuracy
accuracy = accuracy_score(Y_test,Y_pred)

#Testing classifier on stats from current season
currentSeasonData = pd.read_csv('allNba2018.csv')
currentSeasonTest = currentSeasonData.iloc[:,2:7].values
currentSeasonTest = scaler.fit_transform(currentSeasonTest)

#Prediction based on 2018 player stats
currentSeasonPrediction = nbaClassifier.predict(currentSeasonTest)

#Saving model for later use
pickle.dump(nbaClassifier,open('nbaClassifierModel','wb'))