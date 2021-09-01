import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(-1, 1, 100)
true_y = 0.25 * x**2

n = 2
x_expanded = np.expand_dims(x, axis=1)
poly = PolynomialFeatures(n)
x_expanded = poly.fit_transform(x_expanded)


def custom_loss(x, y_pred):

    # Directrix coordinates
    # This could be a function
    dx, dy = x, -1.0

    # Focus coordinates
    fx, fy = 0.0, 1.0

    distance = (x - dx)**2 + (y_pred - dy)**2 - (x - fx)**2 - (y_pred - fy)**2
    distance = tf.reduce_sum(tf.square(distance))

    return distance

model = keras.Sequential(
    [
        layers.Dense(3, input_shape=(n+1,)),
        layers.Dense(1, activation="tanh")
    ]
)

model.compile(optimizer="adam", loss=custom_loss)
model.fit(x_expanded, x, epochs=1000)

plt.plot(x, true_y, c="black")
plt.plot(x, model.predict(x_expanded).reshape(np.shape(x)), c="red")
plt.show()
