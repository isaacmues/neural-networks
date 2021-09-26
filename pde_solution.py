import os
import numpy as np
import tensorflow as tf
from scipy.misc import derivative
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def psi_a(p):

    x = p[:,0]
    y = p[:,1]

    return np.exp(-x) * (x + y**3)

def psi_t(p):

    x = tf.expand_dims(p[:,0], axis=1)
    y = tf.expand_dims(p[:,1], axis=1)

    A = (1 - x) * y**3 + x * (1 + y**3) * np.exp(-1.0)
    A += (1 - y) * x * (tf.exp(-x) - np.exp(-1.0))
    A += y * ((1 + x) * tf.exp(-x) - (1 - x - 2.0 * x * np.exp(-1.0)))

    return A + x * (1 - x) * y * (1 - y) * nn(p)

def custom_loss(p, y_pred):

    x = tf.expand_dims(p[:,0], axis=1)
    y = tf.expand_dims(p[:,1], axis=1)

    h = 1e-4
    hx = tf.zeros_like(p) + tf.constant([[h, 0.0]])
    hy = tf.zeros_like(p) + tf.constant([[0.0, h]])
    d2psidp2 = psi_t(p + hx) - 2.0 * psi_t(p) - psi_t(p - hx)
    d2psidp2 += psi_t(p + hy) - 2.0 * psi_t(p) - psi_t(p - hy)
    d2psidp2 /= h

    error = d2psidp2
    error -= tf.exp(x) * (x - 2.0 + y**3 + 6.0 * y)

    return tf.reduce_sum(tf.square(error))

inputs = Input(shape=[2])
p = Dense(16, activation="tanh")(inputs)
p = Dense(8, activation="tanh")(p)
p = Dense(8, activation="tanh")(p)
p = Dense(8, activation="tanh")(p)
p = Dense(8, activation="tanh")(p)
output = Dense(1, activation="selu")(p)
nn = Model(inputs=inputs, outputs=output)

points = np.linspace(0.0, 1.0, 100)
p = [[x,y] for x in points for y in points]
p = tf.constant(p, dtype=tf.float32)

nn.compile(optimizer="sgd", loss=custom_loss)
nn.fit(p, p, epochs=100)

print(psi_t(p) - psi_a(p))

