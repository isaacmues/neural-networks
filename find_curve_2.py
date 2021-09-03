import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(-1, 1, 200)
a = 1.0
c = 0.5
b = np.sqrt(a**2 - c**2)

true_y = b * np.sqrt(a**2 - x**2) / a

n = 2
x_expanded = np.expand_dims(x, axis=1)
poly = PolynomialFeatures(n)
x_expanded = poly.fit_transform(x_expanded)


def custom_loss(x, y_pred):

    # Focus 1 coordinates
    f1x, f1y = c, 0.0

    # Focus 1 coordinates
    f2x, f2y = -c, 0.0

    distance = tf.sqrt((x - f1x)**2 + (y_pred - f1y)**2) + tf.sqrt((x - f2x)**2 + (y_pred - f2y)**2) - 2 * a
    distance = tf.reduce_sum(tf.square(distance))

    return distance

model = keras.Sequential(
    [
        layers.Dense(1, input_shape=(n+1,)),
        layers.Dense(32, activation="tanh"),
        layers.Dense(1, activation="tanh")
    ]
)

model.compile(optimizer="adam", loss=custom_loss)
model.fit(x_expanded, x, epochs=1000)

plt.plot(x, true_y, c="black")
plt.plot(x, model.predict(x_expanded).reshape(np.shape(x)), c="red")
plt.show()

