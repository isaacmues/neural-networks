import numpy as np
import tensorflow as tf
from scipy.misc import derivative
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense

inputs = Input(shape=[2])
z = Dense(8, activation="relu")(inputs)
output = Dense(1)(z)
nn = Model(inputs=inputs, outputs=output)

def custom_loss(z, y_pred):

    error = z[:,1]**2 + z[:,0]**2 - nn(z) 

    return tf.reduce_sum(tf.square(error))

points = np.linspace(0.0, 1.0, 100)
z = [[x,y] for x in points for y in points]
z = np.asarray(z)


nn.compile(optimizer="adam", loss=custom_loss)
nn.fit(z, z, epochs=100)
