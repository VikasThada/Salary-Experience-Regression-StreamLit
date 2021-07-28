import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import joblib


# Importing the dataset
dataset = pd.read_csv('data/ExpSal.csv')

# seprate feature & target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Simple Linear Regression to the Training set
lr = LinearRegression()
lr.fit(X_train, y_train)


# Predicting the Test set results
y_pred = lr.predict(X_test)

# Saving serialized model to disk

pickle.dump(lr, open('model.pkl','wb'))
#joblib.dump(lr, 'model.pkl')


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#model = joblib.load('model.pkl')

print("lr model output", lr.predict([[1.8]]))
print("Saved  model output", model.predict([[1.8]]))

