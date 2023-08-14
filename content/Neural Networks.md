---
layout: default
title: Neural Networks
nav_order: 9
parent: Algorithms 
---
# Neural Networks

Neural networks are a type of machine learning algorithm modeled after the structure and function of the human brain. They are composed of layers of interconnected nodes, or "neurons", that process information and make predictions based on patterns in the data. Neural networks have gained popularity in recent years due to their ability to learn complex patterns in data, which makes them well-suited for tasks such as image recognition, natural language processing, and speech recognition. 

![Neural networks](https://i0.wp.com/thedatascientist.com/wp-content/uploads/2018/03/simple_neural_network_vs_deep_learning.jpg)
Image from: https://thedatascientist.com/what-deep-learning-is-and-isnt/

The architecture of a neural network can vary widely, depending on the task at hand and the characteristics of the data. Some common types of neural networks include feedforward networks, convolutional neural networks, and recurrent neural networks.

In recent years, there have been significant advancements in the field of neural networks, including the development of deep learning algorithms that allow for the creation of networks with many layers, known as deep neural networks. These networks have achieved state-of-the-art results in a wide range of tasks, including image and speech recognition, natural language processing, and game playing.

As with any machine learning algorithm, the performance of a neural network depends on the quality and quantity of the data used for training, as well as the choice of architecture and hyperparameters. Despite these challenges, neural networks continue to be a powerful tool for solving complex problems in a variety of fields.

The working of a neural network can be challenging to interpret due to several reasons:
1. Non-linearity: Neural networks are composed of multiple layers of interconnected nodes (neurons) that apply non-linear transformations to input data. These non-linearities make it difficult to understand how individual inputs contribute to the output.

2. High Dimensionality: Neural networks often operate on high-dimensional data, such as images or text. With numerous input features, it becomes challenging to visualize and comprehend how each feature influences the network's output.

3. Layer Abstraction: Neural networks consist of multiple layers, with each layer extracting increasingly abstract representations of the input data. The weights and connections between neurons are adjusted during the training process, resulting in complex internal representations that are not readily interpretable.

Scikit-learn is not primarily designed for building neural networks. However, it does have some basic neural network functionality through the MLPClassifier and MLPRegressor classes in its neural network module. Here's an example of how to use MLPClassifier to build a simple neural network for a classification task:

Input
{: .label .label-green}
```python
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
# https://thedatafrog.com/en/articles/handwritten-digit-recognition-scikit-learn/

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

digits = datasets.load_digits()

dir(digits)

print(digits.DESCR)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)

mlp.fit(X_train, y_train)

predicted = mlp.predict(X_test)

predicted = mlp.predict(X_test)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted)
print('Accuracy:', accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predicted, labels=mlp.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()
plt.show()
```

Visualizing the inner layers of a neural network is a good way to get a deeper understanding of this tool. Tools are developed to visualize the neural network in various ways and [Tensorflow Playground](https://playground.tensorflow.org/) is a powerful tool.

## Other Machine Learning Platforms

For more complex neural network tasks, you may want to consider other libraries like TensorFlow or PyTorch:

[TensorFlow](https://www.tensorflow.org/): Developed by Google, TensorFlow is one of the most widely used libraries for building and training neural networks. It offers excellent support for deep learning, including convolutional neural networks and recurrent neural networks, and provides a high level of flexibility and control over the model architecture. TensorFlow is known for its high performance and scalability, making it a good choice for large-scale production applications.

[PyTorch](https://pytorch.org/): Developed by Facebook, PyTorch is another popular library for neural networks. It is known for its ease of use and flexibility, and is especially popular among researchers and academics. PyTorch offers dynamic computational graphs, which allow for more flexible and efficient model building, and provides excellent support for deep learning, including convolutional neural networks and recurrent neural networks.

[Keras](https://keras.io/): Keras is a high-level neural network library that runs on top of TensorFlow, Theano, or Microsoft Cognitive Toolkit. It is known for its ease of use and simplicity, making it a good choice for beginners or those who want to quickly prototype and experiment with different models. Keras offers a range of pre-built models and layers.

## Generative Adversarial Networks

A **generative neural network (GAN)** is a type of neural network architecture that is used to generate new, synthetic data that is similar to some existing dataset. The key idea behind GANs is to have two neural networks, a generator and a discriminator, compete with each other in a game-like fashion.

The generator network takes random noise as input and generates a new data point, which is then evaluated by the discriminator network to determine whether it is real or fake (i.e., generated by the generator). The discriminator is trained to correctly classify the data as real or fake, while the generator is trained to generate data that is realistic enough to fool the discriminator.

Over time, as the two networks continue to play this game, the generator gets better at generating realistic data, while the discriminator gets better at distinguishing real data from fake data. The end result is a generator network that can produce new data that is similar in appearance to the original dataset.

GANs have many applications, including in image generation, text generation, and music generation, among others. They are often used in creative applications, such as generating new artwork or music, but they also have practical applications in areas such as data augmentation and data synthesis.

## Generative Pre-trained Transformers:

**Generative Pre-trained Transformer (GPT)** is a type of neural network architecture designed for natural language processing (NLP) tasks. It was introduced by OpenAI in 2018 and has been used in a wide range of NLP applications, including language generation, translation, and understanding.

The key innovation of the GPT architecture is the use of a transformer-based model that is pre-trained on a large corpus of text data, typically billions of words. This pre-training process allows the model to learn the statistical patterns and relationships in the language, enabling it to generate human-like responses to input text.

The GPT model is trained in an unsupervised manner, which means that it does not require any labeled data for training. Instead, it learns from the raw text data, using a technique called self-supervised learning. During the pre-training process, the model is trained to predict the next word in a sequence of text, based on the previous words. This task is known as language modeling, and it allows the model to learn the structure and semantics of the language.

After pre-training, the GPT model can be fine-tuned on specific NLP tasks, such as text classification or language generation. Fine-tuning involves training the model on a smaller dataset that is specific to the task, while keeping the weights of the pre-trained model fixed. This allows the model to adapt to the specific task, while leveraging the knowledge it has learned from the pre-training process.

Some of the limits of GPTs:

1. **Limited context understanding:** GPTs typically operate on a fixed-length input sequence, which limits their ability to understand long-range dependencies and contextual information beyond the input sequence.

2. **Limited domain specificity:** Pre-trained GPTs are trained on a general corpus of text and may not be optimal for specific domains or tasks. Fine-tuning on specific domains or tasks can help to address this issue, but it requires domain-specific data and additional training. The quality and quantity of the training data can greatly affect the performance of generative models. Insufficient or noisy data can lead to poor quality outputs or generate biased results.

3. **Biases in the training data:** GPTs are trained on large amounts of text data, which can contain biases and stereotypes. This can lead to biased outputs from the model, which may not be desirable in some applications.

4. **Computationally expensive:** GPTs are complex models with a large number of parameters, which require significant computational resources for training and inference.

5. **Lack of interpretability:** GPTs are often referred to as "black box" models, as it can be challenging to understand how they generate their outputs. This lack of interpretability can be problematic in applications where transparency and explainability are important.

6. **Hallucinations:** hallucination refers to the generation of text that is not grounded in the input or the real world. Hallucination occurs when the model generates text that is not based on the input context or is inconsistent with real-world knowledge. Hallucination is a common issue in generative models.

For Neural Network exercises (MLP and CNN), open the following Jupyter Notebook: 

<a target="_blank" href="https://colab.research.google.com/github/ubc-library-rc/intro-machine-learning/blob/main/Examples/NeuralNet_examples.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>