"""
y = 1 - 2x + x³
"""
import os
import numpy as np
import tensorflow as tf
from scipy.misc import derivative
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# This is to avoid some error flags about compilation
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def y(x):

    return 1 - 2*x + x**3


def custom_loss(x, y_pred):

    def y_trial(x):
        y0 =1.0
        return y0 + x * nn(x)

    error = derivative(y_trial, x, 1e-4) -(- 2 + 3*x**2)

    return tf.reduce_sum(tf.square(error))
    
x = np.random.choice(np.linspace(0,2,100), 20)

x = np.expand_dims(x, axis=1)

nn = Sequential(
    [
	    Dense(16, input_shape=[1], activation="tanh"),
	    Dense(8, activation="selu"),
	    Dense(8, activation="tanh"),
	    Dense(8, activation="tanh"),
	    Dense(1, activation="selu"),
    ]
)

nn.compile(optimizer="adam", loss=custom_loss)
nn.fit(x, x, epochs=5000)

x = np.linspace(0, 2, 100) 
y_trial = 1.0 + x * nn.predict(x).flatten()

plt.plot(x, y(x), c="black")
plt.plot(x, y_trial, c="red")
plt.show()
