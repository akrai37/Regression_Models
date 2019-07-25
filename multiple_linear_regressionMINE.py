
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x= onehotencoder.fit_transform(x).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#avoiding dummy variables
x=x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting test set results
y_pred=regressor.predict(x_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm 
x=np.append(arr = np.ones((50,1)).astype(int) , values=x , axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_OLS.summary())

x_opt=x[:,[1,2,3,4,5]]
regressor_OLS=sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_OLS.summary())


x_opt=x[:,[2,3,4,5]]
regressor_OLS=sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_OLS.summary())

x_opt=x[:,[3]]
regressor_OLS=sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_OLS.summary())



























