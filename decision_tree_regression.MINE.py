# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_X.fit_transform(X_train)
x_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state= 0)
regressor.fit(x,y)

# Predicting a new result
y_pred=regressor.predict(6.5)

# Visualising the Regression results
plt.scatter(x,y , color='blue')
plt.plot(x,regressor.predict(x), color='red')
plt.title('TRUTH OR BLUFFing (DecisionTree)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y , color='blue')
plt.plot(x_grid,regressor.predict(x_grid), color='red')
plt.title('TRUTH OR BLUFF (DecisionTree)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()























