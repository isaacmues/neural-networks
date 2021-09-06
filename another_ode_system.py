"""
dx/dt = -x + y
dy/dt = -3x -5y

x = exp(-4t) - 3exp(-2t)
y = -3exp(-4t) + 3exp(-2t)
"""
import os
import numpy as np
import tensorflow as tf
from scipy.misc import derivative
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def x(t):
    return np.exp(-4*t) - 3*np.exp(-2*t)

def y(t):
    return 3*np.exp(-4*t) - 3*np.exp(-2*t)


def custom_loss(t, y_pred):

    def xt(t):
        x0 = -2.0
        return x0 + t * nn(t)[0]

    def yt(t):
        y0 = 0.0
        return y0 + t * nn(t)[1]

    error_x = derivative(xt, t, 1e-3) + xt(t) - yt(t)
    error_y = derivative(yt, t, 1e-3) + 3*xt(t) + 5*yt(t)

    error_x = tf.reduce_sum(tf.square(error_x))
    error_y = tf.reduce_sum(tf.square(error_y))

    return [error_x, error_y]


inputs = Input(shape=[1])

t = Dense(16, activation="tanh")(inputs)
t = Dense(8, activation="selu")(t)
t = Dense(8, activation="tanh")(t)
t = Dense(8, activation="tanh")(t)
t = Dense(8, activation="tanh")(t)
out_1 = Dense(1, activation="selu")(t)

t = Dense(16, activation="tanh")(inputs)
t = Dense(8, activation="selu")(t)
t = Dense(8, activation="tanh")(t)
t = Dense(8, activation="tanh")(t)
t = Dense(8, activation="tanh")(t)
out_2 = Dense(1, activation="selu")(t)

nn = Model(inputs=inputs, outputs=[out_1,out_2], name="ode_system")
nn.compile(optimizer="adam", loss=custom_loss)

t = np.random.choice(np.linspace(0, 2, 100), 20)
t = np.expand_dims(t, axis=1)

nn.fit(t, t, epochs=10000)

t = np.linspace(0, 2, 100)
nnx, nny = nn.predict(t)
nnx = nnx.flatten()
nny = nny.flatten()
x_trial = -2.0 + t * nnx
y_trial = t * nny

plt.plot(t, x(t), color="black")
plt.plot(t, y(t), color="black")
plt.plot(t, x_trial, color="red")
plt.plot(t, -y_trial, color="red")
plt.show()
