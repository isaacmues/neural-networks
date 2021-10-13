import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from derivadas import diff
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Concatenate
from mpl_toolkits.mplot3d import Axes3D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'

def u(z):

    x = tf.reduce_sum(z * [[1.0, 0.0]], axis=1)
    t = tf.reduce_sum(z * [[0.0, 1.0]], axis=1)
    x = tf.expand_dims(x, axis=1)
    t = tf.expand_dims(t, axis=1)

    xf = (z * [[0.0, 1.0]]) + [[1.0, 0.0]]

    A = (x + 1) * tf.cos(t) - 1

    return A + (x + 1) * t * (nn(z) - nn(xf) - 2 * diff(nn, xf, hx))


def custom_loss(z, y_pred):

    error = diff(u, z, ht) - diff(u, z, hx, 2)

    return tf.reduce_sum(tf.square(error))

inputs = Input(shape=[2])
p = Dense(8, activation="tanh")(inputs)
p = Dense(8, activation="tanh")(inputs)
outputs = Dense(1, activation="swish")(p)
nn = Model(inputs=inputs, outputs=outputs)
nn.compile(optimizer="adam", loss=custom_loss)

n = 80
h = 0.00001
hx = tf.constant([[h, 1.0]])
ht = tf.constant([[1.0, h]])
x = np.linspace(-1.0, 1.0, n)
t = np.linspace(0.0, np.pi, n)
z = [[xi, ti] for xi in x for ti in t] 
z = tf.constant(z, dtype=tf.float32)

nn.fit(z, z, epochs=100)

x = tf.reshape(z[:,0], [n,n])
t = tf.reshape(z[:,1], [n,n])
sol = tf.reshape(u(z), [n,n])

plt.contourf(x, t, sol)
plt.contour(x, t, sol, colors="black")
plt.show()

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot_surface(x, t, sol)
plt.show()
