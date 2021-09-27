"""
PDE

∇²Ψ(x,y) = e⁻ˣ(x - 2 + y³ + 6y)

x ∊ [0,1]

Condiciones de frontera

Ψ(0,y) = y³
Ψ(1,y) = (1 + y³)e⁻¹
Ψ(x,0) = xe⁻ˣ
Ψ(x,1) = e⁻ˣ(x + 1)

Solución analítica

Ψₐ(x,y) = e⁻ˣ(x + y³)

Solución de prueba

Ψₜ(x,y) = A(x,y) + x(1 - x)y(1 - y)N(x,y)

donde

A(x,y) = (1 - x)y³ + x(1 - y³)e⁻¹ + (1 - y)x(e⁻ˣ - e⁻¹)
            + y[(1 + x)e⁻ˣ - (1 - x - 2xe⁻¹)]
"""
import os
import numpy as np
import tensorflow as tf
from scipy.misc import derivative
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Concatenate
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def psi_a(z):

    x = tf.reduce_sum(z * [[1.0, 0.0]], axis=1)
    y = tf.reduce_sum(z * [[0.0, 1.0]], axis=1)
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)

    return tf.exp(-x) * (x + y**3)


def psi_t(z):

    x = tf.reduce_sum(z * [[1.0, 0.0]], axis=1)
    y = tf.reduce_sum(z * [[0.0, 1.0]], axis=1)
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)

    e = np.exp(-1)
    f0 = y**3
    f1 = (y**3 + 1) * e
    g0 = x * tf.exp(-x)
    g1 = (x + 1) * tf.exp(-x)

    A = (1 - x) * f0
    A += x * f1
    A += (1 - y) * (g0 - x * e)
    A += y * (g1 - (1 - x + 2 * x * e))

    return A + x * (1 - x) * y * (1 - y) * nn(z)

def laplacian(f, z, h):

    d = f(z + [[h, 0.0]]) + f(z + [[0.0, h]]) 
    d += f(z - [[h, 0.0]]) + f(z - [[0.0, h]]) 
    d -= 4.0 * f(z)
    return d / h**2

def lhs(x,y):

    return tf.exp(-x) * (x - 2.0 + y**3 + 6.0 * y)

def custom_loss(z, y_pred):

    x = tf.reduce_sum(z * [[1.0, 0.0]], axis=1)
    y = tf.reduce_sum(z * [[0.0, 1.0]], axis=1)
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)

    error = tf.reduce_sum(tf.square(laplacian(psi_t, z, 1e-3) - lhs(x,y)))

    return error

inputs = Input(shape=[2])
p = Dense(8, activation="tanh")(inputs)
p = Dense(8, activation="tanh")(inputs)
p = Dense(8, activation="tanh")(inputs)
outputs = Dense(1, activation="selu")(p)
nn = Model(inputs=inputs, outputs=outputs)
nn.compile(optimizer="adam", loss=custom_loss)

points = np.linspace(0.0, 1.0, 40)
z = [[x,y] for x in points for y in points]
z = tf.constant(z, dtype=tf.float32)

nn.fit(z, z, epochs=200)

x = tf.reshape(z[:,0], [40,40])
y = tf.reshape(z[:,1], [40,40])
psi = tf.reshape(psi_t(z), [40, 40])
psi_2 = tf.reshape(psi_a(z), [40, 40])

fig, ax = plt.subplots()
CS = ax.contour(x, y, psi)
CS = ax.contour(x, y, psi_2)
plt.show()
