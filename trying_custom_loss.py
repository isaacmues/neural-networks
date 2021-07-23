"""
Solution of Problem 1 using TensorFlow 2 of Artificial Neural Networks for Solving
Ordinary and Partial Differential Equations by Lagaris, Likas and Fotiadis
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
import matplotlib.pyplot as plt

# This is to avoid some error flags about compilation
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

x = np.linspace(0, 1, 10)
x = np.array([x]).T


def g(x):
    nn = model(x)
    return x * nn + 1.0


def dgdx(x):
    h = 1e-6
    return (g(x + h) - g(x)) / h


def f(x):
    nn = model(x)
    a = (1 + 3 * x ** 2) / (1 + x + x ** 3)
    return x ** 3 + 2 * x + a * x ** 2 - (x + a) * nn


def custom_loss(x, y_pred):
    return tf.reduce_sum(tf.square((dgdx(x) - f(x))))


model = keras.Sequential(
    [
        layers.Dense(
            32,
            input_shape=[1],
            activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(stddev=1.0),
            bias_initializer=initializers.RandomNormal(stddev=1.0),
        ),
        layers.Dense(
            32,
            activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(stddev=1.0),
            bias_initializer=initializers.RandomNormal(stddev=1.0),
        ),
        layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(stddev=1.0),
            bias_initializer=initializers.RandomNormal(stddev=1.0),
        ),
    ]
)
model.compile(optimizer="sgd", loss=custom_loss)
model.fit(x, x, epochs=1000)

x = np.linspace(0, 1, 100)
y_true = np.exp(0.5 * x ** 2) / (1 + x + x ** 3) + x ** 2
x = np.array([x]).T
y_pred = tf.transpose(g(x)).numpy()[0]
x = x.T[0]

plt.plot(x, y_true, label="Analytic", c="black")
plt.plot(x, y_pred, label="Neural Network", c="red")
plt.legend()
plt.show()
