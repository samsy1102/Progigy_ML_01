import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Loading data from a CSV file
data = pd.read_csv('train.csv')

# Displaying the first few rows of the data
print(data.head())

# Selecting features and target
features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = data['SalePrice']

# Filling missing values with the mean of the features
features.fillna(features.mean(), inplace=True)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the target values for the test set
y_pred = model.predict(X_test)

# Calculating the mean squared error and the coefficient of determination (R²)
mean_squared_error_value = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mean_squared_error_value}')
print(f'Coefficient of Determination (R²): {r2}')

# Visualizing actual prices vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
