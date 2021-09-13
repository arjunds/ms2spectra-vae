import pickle
import keras
from keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import numpy as np
import math
import time
from keras import backend as K

epochs = 25000
tag= "GNPS_Filtered"

'''
with open('binned_filtered_0.5.pkl', 'rb') as f:
    data = pickle.load(f)
    
spectra_matrix = data[0].toarray().T

x_train = spectra_matrix
# Takes ~10% of the data for validation
test_index = np.random.choice(range(len(x_train)), math.floor(len(x_train)/10), replace=False)
x_test = x_train[test_index]
x_train = np.delete(x_train, test_index, 0)
print(x_train.shape)
print(x_test.shape)

pickle.dump((x_train,x_test, test_index),open("filtered_data.pkl", "wb"))
'''

with open("filtered_data.pkl", 'rb') as f:
    x_train, x_test, indices = pickle.load(f)

dims = x_train.shape

original_dim = dims[1]
intermediate_dim = 100
latent_dim = 20

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

# This is our input data

encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# This model maps an input to its reconstruction
outputs = decoder(encoder(inputs)[2])
autoencoder = keras.Model(inputs, outputs)

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(dims[1],))])

log_dir = "logs/fit/"+ tag + "_BasicVAE_" + str(epochs) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

autoencoder.fit(x_train, x_train, epochs=epochs, callbacks=[tensorboard_callback], validation_data=(x_test, x_test))
autoencoder.save("Filtered_GNPSv2")
