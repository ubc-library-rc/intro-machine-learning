---
layout: default
title: Regression
nav_order: 4
parent: Algorithms 
---
# Regression

Regression is a type of supervised machine learning algorithm used to predict a continuous numerical outcome variable based on one or more predictor variables. In regression analysis, the goal is to establish a relationship between the predictor variables and the outcome variable, which can then be used to make predictions on new data.

There are several types of regression algorithms, each with their own strengths and limitations. Here are some of the most commonly used types of regression:

Linear Regression: Linear regression is the simplest and most commonly used type of regression algorithm. It assumes that there is a linear relationship between the predictor variables and the outcome variable. The goal is to find the line of best fit that minimizes the distance between the predicted values and the actual values.

Logistic Regression: Logistic regression is used when the outcome variable is categorical. It models the relationship between the predictor variables and the probability of the outcome variable being in a particular category.

Polynomial Regression: Polynomial regression is used when the relationship between the predictor variables and the outcome variable is nonlinear. It involves fitting a polynomial curve to the data.

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("sample_data.csv")

# Split the data into inputs and outputs
X = data["input_variable"].values.reshape(-1, 1)
y = data["output_variable"].values.reshape(-1, 1)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the model's performance
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot the data and the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel("Input Variable")
plt.ylabel("Output Variable")
plt.title("Linear Regression Model")
plt.show()

# Print the model's performance metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)


### Model evaluation

Regression metrics:

Mean absolute error (MAE): the average absolute difference between the predicted and actual values.
Mean squared error (MSE): the average squared difference between the predicted and actual values.
Root mean squared error (RMSE): the square root of MSE.
R-squared (R2): the proportion of the variance in the dependent variable that is explained by the independent variables.