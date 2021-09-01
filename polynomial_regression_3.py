import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
from sklearn.preprocessing import PolynomialFeatures


def y(x):

    return tf.sin(x * np.pi)


def custom_loss(x, y_pred):

    return tf.reduce_sum(tf.square(y_pred - y(x)))

x = np.linspace(0, 1, 100)

n = 10
x_expanded = np.expand_dims(x, axis=1)
poly = PolynomialFeatures(n)
x_expanded = poly.fit_transform(x_expanded)

model = keras.Sequential(
    [
        layers.Dense(30, input_shape=(n+1,)),
        layers.Dense(30, activation="tanh"),
        layers.Dense(1, activation="tanh")
    ]
)

model.compile(optimizer="adam", loss=custom_loss)
model.fit(x_expanded, x, epochs=1000)



plt.plot(x, y(x), c="black")
plt.plot(x, model.predict(x_expanded).reshape(np.shape(x)), c="red")
plt.show()
