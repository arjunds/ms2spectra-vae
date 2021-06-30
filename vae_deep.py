import pickle
import keras
from keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import numpy as np
import math

with open('binned_complex.pkl', 'rb') as f:
    data = pickle.load(f)
    
spectra_matrix = data[0].toarray().T

x_train = spectra_matrix
# Takes ~10% of the data for validation
test_index = np.random.choice(range(len(x_train)), math.floor(len(x_train)/10), replace=False)
x_test = x_train[test_index]
x_train = np.delete(x_train, test_index, 0)
print(x_train.shape)
print(x_test.shape)

dims = x_train.shape

# This is our input data
input_data = keras.Input(shape=(dims[1],))
encoded = layers.Dense(1000, activation='relu')(input_data)
encoded = layers.Dense(500, activation='relu')(input_data)
encoded = layers.Dense(100, activation='relu')(input_data)


decoded = layers.Dense(500, activation='sigmoid')(encoded) 
decoded = layers.Dense(1000, activation='sigmoid')(decoded) 
decoded = layers.Dense(dims[1], activation='sigmoid')(decoded) 

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(dims[1],))])

log_dir = "logs/fit/DeepVAE" + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

autoencoder.fit(x_train, x_train, epochs=7000, callbacks=[tensorboard_callback], validation_data=(x_test, x_test))