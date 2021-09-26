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

def psi_a(x,y):

    return tf.exp(-x) * (x + y**3)

def psi_t(x,y):

    e = np.exp(-1)

    f0 = y**3
    f1 = (y**3 + 1) * e
    g0 = x * tf.exp(-x)
    g1 = (x + 1) * tf.exp(-x)

    A = (1 - x) * f0
    A += x * f1
    A += (1 - y) * (g0 - x * e)
    A += y * (g1 - (1 - x + 2 * x * e))

    return A + x * (1 - x) * y * (1 - y) * nn([x,y])

def laplacian(f, x, y, h):

    d = f(x + h, y) + f(x, y + h)
    d += f(x - h, y) + f(x, y - h)
    d -= 4 * f(x, y)
    return d / h**2

def custom_loss(p, y_pred):

    x,y = p
    error = laplacian(psi_t, x, y, 1e-6)
    error -= tf.exp(-x) * (x - 2.0 + y**3 + 6.0 * y)

    return tf.reduce_sum(tf.square(error))

input_x = Input(shape=[1])
input_y = Input(shape=[1])
p = Concatenate(axis=1)([input_x, input_y])
p = Dense(16, activation="tanh")(p)
p = Dense(8, activation="tanh")(p)
p = Dense(8, activation="tanh")(p)
output = Dense(1, activation="selu")(p)
nn = Model(inputs=[input_x, input_y], outputs=output)

#x = tf.linspace(0.0, 1.0, 10)
#x = tf.expand_dims(x, axis=1)
#y = x

points = tf.linspace(0.0, 1.0, 5)
z = [[x,y] for x in points for y in points]
x,y = tf.transpose(z)
x = tf.expand_dims(x, axis=1)
y = tf.expand_dims(y, axis=1)

print(psi_a(x,y) - psi_t(x,y))
