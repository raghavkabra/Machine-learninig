## 1. Looking at the data ##

import pandas

sp500 = pandas.read_csv("sp500.csv")

sp500.head()

## 2. Cleaning the invalid rows ##

# The data is loaded into the sp500 variable.
sp500 = sp500[sp500["value"] != "."]

## 3. Finding the predictors ##

# This prints the last 10 rows -- note where the dataset ends.
#print(sp500.tail(10))
#2609 2015-06-26 2101.61

next_day = sp500["value"].iloc[1:]
print(next_day)
sp500 = sp500.iloc[:-1,:]
sp500["next_day"] = next_day.values

## 4. Converting columns to floats ##

# We can see the current types of the columns
print(sp500.dtypes)
sp500["value"] = sp500["value"].astype(float)
sp500["next_day"] = sp500["next_day"].astype(float)

## 5. Making predictions ##

# Import the linear regression class
from sklearn.linear_model import LinearRegression

# Initialize the linear regression class.
regressor = LinearRegression()

# We're using 'value' as a predictor, and making predictions for 'next_day'.
# The predictors need to be in a dataframe.
# We pass in a list when we select predictor columns from "sp500" to force pandas not to generate a series.
predictors = sp500[["value"]]
to_predict = sp500["next_day"]

# Train the linear regression model on our dataset.
regressor.fit(predictors, to_predict)

# Generate a list of predictions with our trained linear regression model
next_day_predictions = regressor.predict(predictors)
print(regressor.summary())

## 6. Measuring error ##

# The actual values are in to_predict, and the predictions are in next_day_predictions.
mse = sum((to_predict - next_day_predictions) ** 2)
mse /= len(next_day_predictions)

## 7. Overfitting ##

import numpy as np
import random

# Set a random seed to make the shuffle deterministic.
np.random.seed(1)
random.seed(1)
# Randomly shuffle the rows in our dataframe
sp500 = sp500.loc[np.random.permutation(sp500.index)]

# Select 70% of the dataset to be training data
highest_train_row = int(sp500.shape[0] * .7)
train = sp500.loc[:highest_train_row,:]

# Select 30% of the dataset to be test data.
test = sp500.loc[highest_train_row:,:]

regressor = LinearRegression()
regressor.fit(train[["value"]], train["next_day"])
predictions = regressor.predict(test[["value"]])
mse = sum((predictions - test["next_day"]) ** 2) / len(predictions)

## 8. Visualizing the fit ##

import matplotlib.pyplot as plt

# Make a scatterplot with the actual values in the training set
plt.scatter(train["value"], train["next_day"])
plt.plot(train["value"], regressor.predict(train[["value"]]))
plt.show()
plt.scatter(test["value"], test["next_day"])
plt.plot(test["value"], predictions)
plt.show()

## 9. Other error metrics ##

# The test set predictions are in the predictions variable.
import math
rmse = math.sqrt(sum((predictions - test["next_day"]) ** 2) / len(predictions))
mae = sum(abs(predictions - test["next_day"])) / len(predictions)