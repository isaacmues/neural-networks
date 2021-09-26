"""
Solution of Problem 2 using TensorFlow 2 of Artificial Neural Networks for Solving
Ordinary and Partial Differential Equations by Lagaris, Likas and Fotiadis
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# This is to avoid some error flags about compilation
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def y(x):

    return np.exp(-x / 5) * np.sin(x)

def y_trial(x):

  return x * np.sin(1) * np.exp(-0.20) + x * (1 - x) * nn(x)


def custom_loss(x, y_pred):

    error = derivative(y_trial, x, 1e-3, 2)
    error += 0.2 * derivative(y_trial, x, 1e-3)
    error += y_trial(x) + 0.2 * tf.exp(-0.2 * x) * tf.cos(x)

    return tf.reduce_sum(tf.square(error))


nn = Sequential(
    [
        Dense(8, input_shape=[1], activation="tanh"),
        Dense(6, activation="tanh"),
        Dense(1, activation="tanh")
    ]
)

x = np.expand_dims(np.linspace(0, 3, 300), axis=1)

nn.compile(optimizer='adam', loss=custom_loss)
nn.fit(x, x, epochs=100)


plt.plot(x, y(x) , label="Analytic", c="black")
plt.plot(x, y_trial(x), label="Neural Network", c="red")
plt.legend()
plt.show()
