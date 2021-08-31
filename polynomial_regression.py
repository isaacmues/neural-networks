import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
from sklearn.preprocessing import PolynomialFeatures

def generate_data():

    X = np.arange(-30, 30, 1)
    y = 9*X**3 + 5*X**2 + np.random.randn(60)*1000
    return X, y


trX, trY = generate_data()
trX = trX/max(trX)
trY = trY/max(trY)

n = 3
trX_expanded = np.expand_dims(trX, axis=1)
poly = PolynomialFeatures(n)
trX_expanded = poly.fit_transform(trX_expanded)

model = keras.Sequential(
    [
        layers.Dense(1, input_shape=(n+1,)),
        layers.Dense(1)
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(trX_expanded, trY, epochs=1000)

plt.scatter(trX, trY)
plt.plot(trX, model.predict(trX_expanded), color="red")
plt.show()
