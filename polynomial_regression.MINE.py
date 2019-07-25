# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting linear regresssion to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x , y)

#fitting polynomial regresssion to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly , y)

#visualizing the linear regression results
plt.scatter(x , y , color='blue')
plt.plot( x, lin_reg.predict(x) , color='red')
plt.title('truth or bluffing(linear regression)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#visualizing the polynomial regression 
x_grid=np.arange(min(x) , max(x) , 0.1)
x_grid=x_grid.reshape(len(x_grid) , 1)
plt.scatter(x , y , color='blue')
plt.plot( x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)) , color='red')
plt.title('truth or bluffing(Polynomial regression)')
plt.xlabel('position')
plt.ylabel('salary')

#predicting the new results  with linear regression
lin_reg.predict(6.5)

#predicting the new results  with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))






































