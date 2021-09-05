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
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def custom_loss(t, y_pred):

    h = 1e-3
    R_trial, F_trial = model(t)
    dRdt, dFdt = model(t + h)
    dRdt -= R_trial
    dRdt /= h
    dFdt -= F_trial
    dFdt /= h

    error = dRdt - 2*R_trial + 1.2*R_trial*F_trial
    error += dFdt + R_trial - 0.9*R_trial*F_trial

    return tf.reduce_sum(tf.square(error))

# Model

inputs = tf.keras.Input(shape=[1])

#For R
dense = Dense(8, activation="tanh")
x = dense(inputs)
x = Dense(8, activation="tanh")(x)
output_1 = Dense(1, activation="tanh")(x)

#For F
dense = Dense(8, activation="tanh")
x = dense(inputs)
x = Dense(8, activation="tanh")(x)
output_2 = Dense(1, activation="tanh")(x)

model = Model(inputs=inputs, outputs=[output_1, output_2], name="predator_prey")
model.compile(optimizer="adam", loss=custom_loss)

t = np.linspace(0, 15, 50)
t = np.expand_dims(t, axis=1)

model.fit(t, t, epochs=10)
