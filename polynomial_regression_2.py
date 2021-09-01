import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
from sklearn.preprocessing import PolynomialFeatures


trX = np.linspace(0, 1, 100)
trY = np.sin(trX *  np.pi)

n = 15
trX_expanded = np.expand_dims(trX, axis=1)
poly = PolynomialFeatures(n)
trX_expanded = poly.fit_transform(trX_expanded)

model = keras.Sequential(
    [
        layers.Dense(90, input_shape=(n+1,)),
        layers.Dense(1, activation="sigmoid")
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(trX_expanded, trY, epochs=1000)

plt.scatter(trX, trY)
plt.plot(trX, model.predict(trX_expanded), color="red")
plt.show()
