# SVR

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
_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
sc_y= StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y.reshape(-1,1))

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR( kernel='rbf' )
regressor.fit(x , y)

# Predicting a new result
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(x, y , color='blue')
plt.plot(x , regressor.predict(x), color='red')
plt.title('truth or bluff (SVR)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Visualising the SVR results(for higher resolution and smoother curve)
x_grid=np.arange( min(x) ,  max(x) , 0.1)
x_grid=x_grid.reshape(len(x_grid) , 1)
plt.scatter(x, y ,color='blue')
plt.plot(x_grid , regressor.predict(x_grid), color='red')
plt.title('truth or bluff (SVR advanced)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()




















