---
layout: default
title: Regression
nav_order: 7
parent: Algorithms 
---

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# Regression

Regression is a type of supervised machine learning algorithm used to predict a continuous numerical outcome variable based on one or more predictor variables. In regression analysis, the goal is to establish a relationship between the predictor variables (independent variables) and the outcome variables (dependent variables), which can then be used to make predictions on new data. 

Here are some of the most commonly used types of regression:

**Linear Regression:** Linear regression is the simplest and most commonly used type of regression algorithm. It assumes that there is a linear relationship between the predictor variables and the outcome variable. The goal is to find the line of best fit that minimizes the distance between the predicted values and the actual values.

In a linear regression, we assume a linear relationship between some dependent variable $$y$$ and a set of independent variables $$\textbf{x} = (x_1, x_2, \ldots, x_n)$$. Therefore, for each observation, the predicted value of the output, shown by $$\bar{y}$$, is $$\bar{y} = b_0 + b_1x_1 + \ldots + b_nx_n$$, where $$b_0$$ to $$b_n$$ are regression coefficients or the predicted weights for each independent variable. The predicted output, $$\bar{y}$$, should be as close as possible to the actual response, $$y$$, and the difference for all observations are called the residuals. To find the best weights, we minimize a measure of error, such as the sum of squared residuals, for all observations.

**Polynomial Regression:** Polynomial regression is used when the relationship between the predictor variables and the outcome variable is nonlinear. It involves fitting a polynomial curve to the data. Logarithmic regression is another method used in nonlinear problems.

## Regression Metrics:

The following quantitative measures are some of the commonly used measures used to evaluate the performance of regression models: 

* **Mean squared error (MSE):** the average squared difference between the predicted and actual values. It gives more weight to larger errors, making it sensitive to outliers. Lower MSE values indicate better model performance, with 0 being the best possible value.

* **Root mean squared error (RMSE):** the square root of MSE. Like MSE, lower RMSE values indicate better model performance.

* **Mean absolute error (MAE):** the average absolute difference between the predicted and actual values. It is less sensitive to outliers compared to MSE, as it does not square the errors. Lower MAE values indicate better model performance.

* **R-squared (R2) or coefficient of determination:** the proportion of the variance in the dependent variable that is explained by the independent variables.  It ranges from 0 to 1, with higher values indicating better model performance. A value of 1 means that the model explains all the variability in the output variable, while a value of 0 means that the model explains none of the variability.

For Regression exercises, click on the following link:

<a target="_blank" href="https://colab.research.google.com/github/ubc-library-rc/intro-machine-learning/blob/main/Examples/Regression-examples.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>