---
layout: default
title: Regression
nav_order: 6
parent: Algorithms 
---
# Regression

Regression is a type of supervised machine learning algorithm used to predict a continuous numerical outcome variable based on one or more predictor variables. In regression analysis, the goal is to establish a relationship between the predictor variables and the outcome variable, which can then be used to make predictions on new data. Here are some of the most commonly used types of regression:

**Linear Regression:** Linear regression is the simplest and most commonly used type of regression algorithm. It assumes that there is a linear relationship between the predictor variables and the outcome variable. The goal is to find the line of best fit that minimizes the distance between the predicted values and the actual values.

**Logistic Regression:** Logistic regression is used when the outcome variable is categorical. It models the relationship between the predictor variables and the probability of the outcome variable being in a particular category.

**Polynomial Regression:** Polynomial regression is used when the relationship between the predictor variables and the outcome variable is nonlinear. It involves fitting a polynomial curve to the data.

Input
{: .label .label-green}
```python
# Code source: Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]


# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Save the trained model
joblib.dump(regr, 'regression-model.joblib')
# model = joblib.load('regression-model.joblib')

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot the data and the regression line
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.xlabel("Input Variable")
plt.ylabel("Output Variable")
plt.title("Linear Regression Model")
plt.show()
```

## Regression Metrics:

The following quantitative measures are some of the commonly used measures used to evaluate the performance of regression models: 

* **Mean squared error (MSE):** the average squared difference between the predicted and actual values. It gives more weight to larger errors, making it sensitive to outliers. Lower MSE values indicate better model performance, with 0 being the best possible value.

* **Root mean squared error (RMSE):** the square root of MSE. Like MSE, lower RMSE values indicate better model performance.

* **Mean absolute error (MAE):** the average absolute difference between the predicted and actual values. It is less sensitive to outliers compared to MSE, as it does not square the errors. Lower MAE values indicate better model performance.

* **R-squared (R2):** the proportion of the variance in the dependent variable that is explained by the independent variables.  It ranges from 0 to 1, with higher values indicating better model performance. A value of 1 means that the model explains all the variability in the output variable, while a value of 0 means that the model explains none of the variability.

## Saving and Loading Models

**Joblib** is a Python library that is commonly used in conjunction with scikit-learn and provides tools for easy and efficient saving and loading of Python objects, including machine learning models, to and from disk. It is particularly useful when working with large datasets or complex machine learning models that take a long time to train, as it allows you to persistently store the trained models on disk and reload them later, rather than retraining them from scratch every time you need to use them.

Joblib provides an alternative to Python's built-in pickle module, but it is designed to be more efficient for dealing with large data, such as NumPy arrays, which are commonly used in machine learning applications. Joblib provides parallel processing capabilities, allowing you to save and load objects in parallel, which can significantly speed up the process. Joblib also includes features such as compression, which can reduce the size of the saved objects on disk, making it more efficient for storing and managing large numbers of machine learning models or other objects.