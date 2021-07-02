import pickle
import keras
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import numpy as np
import math

with open('binned_gnps.pkl', 'rb') as f:
    data = pickle.load(f)
    
spectra_matrix = data[0].toarray().T

x_train = spectra_matrix
# Takes ~10% of the data for validation
test_index = np.random.choice(range(len(x_train)), math.floor(len(x_train)/10), replace=False)
x_test = x_train[test_index]
x_train = np.delete(x_train, test_index, 0)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
print(x_train.shape)
print(x_test.shape)

dims = x_train.shape

# This is our input data
input_data = keras.Input(shape=(dims[1], 1))
print(input_data.shape)
hidden_1 = Conv1D(1, (5, ), activation='relu', padding='same')(input_data)
print(hidden_1.shape)
hidden_2 = MaxPooling1D()(hidden_1)
hidden_3 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_2)
print(hidden_3.shape)
hidden_4 = MaxPooling1D()(hidden_3)
hidden_5 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_4)
print(hidden_5.shape)
encoded = MaxPooling1D((6, ))(hidden_5)
print(encoded.shape)

hidden_6 = Conv1D(1, (5, ), activation='relu', padding='same')(encoded)
print(hidden_6.shape)
hidden_7 = UpSampling1D(6)(hidden_6)
hidden_8 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_7)
print(hidden_7.shape)
hidden_9 = UpSampling1D()(hidden_8)
hidden_10 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_9)
print(hidden_10.shape)
hidden_11 = UpSampling1D()(hidden_10)
decoded = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_11)
print(decoded.shape)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(dims[1], 1))])

log_dir = "logs/fit/ConvVAE" + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

autoencoder.fit(x_train, x_train, epochs=7000, callbacks=[tensorboard_callback], validation_data=(x_test, x_test))