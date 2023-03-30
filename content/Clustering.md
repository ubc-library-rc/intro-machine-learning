---
layout: default
title: Classification and Clustering
nav_order: 5
parent: Algorithms 
---

Classification and clustering are two important types of machine learning techniques, but they differ in their goals and methods.

Classification aims to predict the class or category of a given input based on labeled examples or historical data. The goal is to learn a mapping function that maps input features to a discrete output class label.
Clustering aims to group similar data points or observations together in an unsupervised manner, without any predefined class labels. The goal is to discover inherent structures or patterns in the data.
Method:

Classification uses supervised learning, where the algorithm is trained on a labeled dataset to learn the relationship between input features and output class labels. The trained model can then be used to predict the class labels of new, unseen data.
Clustering uses unsupervised learning, where the algorithm groups similar data points together based on their similarity or distance from each other. The number of clusters may be predefined or learned from the data.
Output:

Classification outputs a discrete class label for each input, based on the learned mapping function. The output is interpretable and can be used for decision-making.
Clustering outputs a set of clusters or groups, which may or may not have an interpretable label. The output can be used for exploratory data analysis or as a preprocessing step for other tasks.
Evaluation:

Classification models can be evaluated using metrics such as accuracy, precision, recall, and F1-score, which measure the performance of the model in predicting the correct class labels.
Clustering models can be evaluated using metrics such as silhouette coefficient, Calinski-Harabasz index, and Davies-Bouldin index, which measure the quality of the clusters based on their compactness, separation, and similarity.

# Classification

Copy code
# Import required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from a CSV file
data = pd.read_csv('data.csv')

# Split data into training and testing sets
X = data.drop('target_variable', axis=1)  # input features
y = data['target_variable']  # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predict class labels for the test data
y_pred = lr_model.predict(X_test)

# Evaluate the model's accuracy on the test data
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
This code assumes that you have a CSV file named data.csv with your input features and target variable. You can modify the drop() function to exclude any columns that are not input features or target variables.

The code first loads the data from the CSV file and then splits it into training and testing sets using the train_test_split() function. It then trains a logistic regression model on the training data using the fit() function of the LogisticRegression class.

The model is then used to predict the class labels for the test data using the predict() function. Finally, the model's accuracy is evaluated on the test data using the accuracy_score() function from scikit-learn's metrics module.

# Clustering

Clustering is an unsupervised machine learning technique used to group similar objects together based on their characteristics. The goal of clustering is to divide a dataset into clusters in a way that objects within a cluster are similar to each other, while objects in different clusters are dissimilar.

There are several types of clustering algorithms, each with their own strengths and limitations. Here are some of the most commonly used types of clustering:

K-Means Clustering: K-means clustering is the most widely used type of clustering algorithm. It involves dividing a dataset into k clusters, where k is a user-defined parameter. The algorithm iteratively assigns data points to the nearest cluster center and recalculates the cluster center until convergence.

Hierarchical Clustering: Hierarchical clustering is a type of clustering algorithm that creates a tree-like structure of clusters, called a dendrogram. It can be divided into two types: agglomerative, which starts with each data point as its own cluster and iteratively merges the closest pairs of clusters, and divisive, which starts with all data points in one cluster and recursively splits them into smaller clusters.

Density-Based Clustering: Density-based clustering is a type of clustering algorithm that groups together data points that are close to each other in density. It can identify clusters of arbitrary shapes and sizes, and is particularly useful for datasets with noise and outliers.

Expectation-Maximization (EM) Clustering: EM clustering is a probabilistic clustering algorithm that models the data as a mixture of Gaussian distributions. It iteratively estimates the parameters of the Gaussian distributions and assigns data points to the cluster with the highest probability.

Fuzzy Clustering: Fuzzy clustering is a type of clustering algorithm that allows data points to belong to multiple clusters with different degrees of membership. It is useful when data points have ambiguous membership or when there are overlaps between clusters.

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate some sample data
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Create a KMeans clustering model with 2 clusters
model = KMeans(n_clusters=2)

# Train the model on the data
model.fit(X)

# Predict the cluster labels for the data
y_pred = model.predict(X)

# Plot the data with different colors for each cluster
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
In this example, we generate some sample data with two clusters. We then create a KMeans clustering model using scikit-learn's KMeans class, and specify that we want to cluster the data into 2 clusters using the n_clusters parameter. We then train the model on the data using the fit method, and use the predict method to predict the cluster labels for the data. Finally, we plot the data with different colors for each cluster using Matplotlib.

This code provides a simple and easy-to-understand introduction to clustering with scikit-learn, and can be easily modified to work with different datasets and clustering models.