"""
Predator-prey system

	dR
	-- = 2R - 1.2RF
	dt
	
	dF
	-- = -F + 0.9RF
	dt

with initial conditions R0 = 1, F0 = 0.5

Blanchard
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"


def custom_loss(t, y_pred):
    
    h = 1e-3
    R0 = 1.0
    F0 = 0.5
    nnR, nnF = model(t)
    R = R0 + t * nnR
    F = F0 + t * nnF

    nnR, nnF = model(t + h)
    dRdt = R0 + (t + h) * nnR
    dFdt = F0 + (t + h) * nnF
    dRdt -= R
    dFdt -= F
    dRdt /= h
    dFdt /= h

    error = dRdt + dFdt - 2 * R + F + 0.3 * R * F

    return tf.reduce_sum(tf.square(error))

# Model

inputs = tf.keras.Input(shape=[1])

#For R
dense = Dense(16, activation="tanh")
x = dense(inputs)
x = Dense(32, activation="tanh")(x)
x = Dense(32, activation="tanh")(x)
x = Dense(32, activation="tanh")(x)
output_1 = Dense(1, activation="tanh")(x)

#For F
dense = Dense(32, activation="tanh")
x = dense(inputs)
x = Dense(32, activation="tanh")(x)
x = Dense(32, activation="tanh")(x)
x = Dense(32, activation="tanh")(x)
output_2 = Dense(1, activation="tanh")(x)

model = Model(inputs=inputs, outputs=[output_1, output_2], name="predator_prey")
model.compile(optimizer="adam", loss=custom_loss)

t = np.linspace(0, 5, 100)
t = np.expand_dims(t, axis=1)

model.fit(t, t, epochs=10000)

nnR, nnF = model.predict(t)

nnR = nnR.flatten()
nnF = nnF.flatten()
t = t.flatten()

R_trial = 1.0 + t * nnR
F_trial = 0.5 + t * nnF

plt.plot(t, R_trial, c="green")
plt.plot(t, F_trial, c="blue")
plt.show()
