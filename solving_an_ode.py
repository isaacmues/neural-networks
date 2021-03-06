# -*- coding: utf-8 -*-
"""solving an ode.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vos2y3R0NprDTIy1UaSLkZWp-ytz77sC
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# initial condition
f0 = 1
# infinitesimal small number
inf_s = np.sqrt(np.finfo(np.float32).eps)

# Parameters
learning_rate = 0.01
training_steps = 1000
batch_size = 100
display_step = training_steps / 10

# Network Parameters
n_input = 1  # input layer number of neurons
n_hidden_1 = 32  # 1st layer number of neurons
n_hidden_2 = 32  # 2nd layer number of neurons
n_output = 1  # output layer number of neurons

weights = {
    "h1": tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    "h2": tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    "out": tf.Variable(tf.random.normal([n_hidden_2, n_output])),
}

biases = {
    "b1": tf.Variable(tf.random.normal([n_hidden_1])),
    "b2": tf.Variable(tf.random.normal([n_hidden_2])),
    "out": tf.Variable(tf.random.normal([n_output])),
}

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Create model
def multilayer_perceptron(x):
    x = np.array([[[x]]], dtype="float32")
    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.sigmoid(layer_2)
    output = tf.matmul(layer_2, weights["out"]) + biases["out"]
    return tf.nn.sigmoid(output)


# Universal Approximator
def g(x):
    return x * multilayer_perceptron(x) + f0


# Given EDO
def f(x):
    return 2 * x


# Custom loss function to approximate the derivatives
def custom_loss():
    def dNN(x):
        return (g(x + inf_s) - g(x)) / inf_s

    summation = [(dNN(x) - f(x)) ** 2 for x in np.linspace(0, 1, 11)]
    return tf.reduce_sum(tf.abs(summation))
    # return tf.sqrt(tf.reduce_mean(tf.abs(summation)))


def train_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


for i in range(training_steps):
    train_step()
    if i % display_step == 0:
        print("loss: %f " % (custom_loss()))

from matplotlib.pyplot import figure

figure(figsize=(10, 10))
# True Solution (found analitically)
def true_solution(x):
    return x ** 2 + 1


X = np.linspace(0, 1, 101)
result = [g(x).numpy()[0, 0, 0] for x in X]
# for i in X:
# result.append(f(i))
#  result.append(g(i).numpy()[0][0][0])

S = true_solution(X)

plt.plot(X, S, label="Original Function")
plt.plot(X, result, label="Neural Net Approximation")
plt.legend(loc=2, prop={"size": 20})
plt.show()
