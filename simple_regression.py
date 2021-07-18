import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


m = 2.0
b = 1.0
n = 21 # number of points

# Generate a random set of data
noise = np.random.default_rng().integers(-50, 50, size=n) / 100.0
x = np.linspace(0, 5, n)
y = m * x + noise + b

# Constructing a model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=100)

predictions = model.predict(x)

plt.scatter(x, y, c='black')
plt.plot(x, predictions, c='red')
plt.title('Simple Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
