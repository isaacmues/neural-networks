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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = np.linspace(0, 1, 11)
h = 1e-3
y0 = 1.0

def g(x):
    x = np.array([x]).T
    return x * model(x) + y0

#x = x.reshape((11,1))
def custom_loss(x,  y_pred):
    dydx = (model(x + h) - y_pred) / h
    return tf.reduce_sum(tf.square(dydx - 2 * x))

model = keras.Sequential()
model.add(layers.Dense(32, input_shape=[1], activation="sigmoid", kernel_initializer=initializers.RandomNormal(stddev=1.0), bias_initializer=initializers.RandomNormal(stddev=1.0)))
model.add(layers.Dense(32, activation="sigmoid", kernel_initializer=initializers.RandomNormal(stddev=1.0), bias_initializer=initializers.RandomNormal(stddev=1.0)))
model.add(layers.Dense(1, activation="sigmoid", kernel_initializer=initializers.RandomNormal(stddev=1.0), bias_initializer=initializers.RandomNormal(stddev=1.0)))

dydx = (g(x+h) - g(x))/h

print(tf.square(np.linspace(1, 5, 5)))
print(tf.reduce_sum(np.linspace(1, 5, 5)))
