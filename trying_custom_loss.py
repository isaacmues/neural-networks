"""
Let's try to update the ODE solver to use Keras with a custom_loss
function

So the ODE to solve is dy/dx = 2x with y(0) = 1
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
import matplotlib.pyplot as plt
import os

# Solo para evitar las banderas de error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

x = np.linspace(0, 1, 11)
x = np.array([x]).T
h = 1e-3
y0 = 1.0

true_values = x ** 2 + 1


def g(x):
    nn = model(x)
    return x * nn + y0


def f(x):
    return 2 * x


def custom_loss():
    dydx = (g(x + h) - g(x)) / h
    return tf.reduce_sum(tf.square((dydx - f(x))))


def training_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    grads = tape.gradient(loss, model.trainable_weights)


optimizer = optimizers.SGD(1e-3)

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

print(custom_loss())
