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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

x = np.linspace(-2*np.pi, 2*np.pi, 200)

x_training = np.random.choice(x, 50)
y_training = np.sin(x_training)

model = keras.Sequential(
    [
        layers.Dense(1, input_shape=[1], activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(1, activation="tanh"),
    ]
)

x_training = np.expand_dims(x_training, axis=1)
y_training = np.expand_dims(y_training, axis=1)

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_training, y_training, epochs=25000)

plt.plot(x, np.sin(x), color="black")
plt.plot(x, model.predict(x).flatten(), color="red")

plt.show()
