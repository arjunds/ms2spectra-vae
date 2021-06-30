import pickle
import keras
from keras import layers
from keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import numpy as np
import math

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def RSS(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    return SS_res

def TSS(y_true, y_pred):
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return SS_tot

with open('binned_data.pkl', 'rb') as f:
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

for dim in [100]:
    # This is the size of our encoded representations
    encoding_dim = dim  # changed from 32 in example to 100 for testing

    # This is our input image
    input_img = keras.Input(shape=(dims[1],)) # changed from 784 in example to matrix shape[0]
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(dims[1], activation='sigmoid')(encoded) # changed from 784 in example to matrix shape[0]

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(dims[1],))])

    # dataset = tf.data.Dataset.from_tensor_slices(spectra_matrix)
    # dataset = dataset.map(lambda x: (x, x))  # Use x_train as both inputs & targets
    # dataset = dataset.shuffle(buffer_size=1).batch(1)

    log_dir = "logs/fit/7000epochs" + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    autoencoder.fit(x_train, x_train, epochs=7000, callbacks=[tensorboard_callback], validation_data=(x_test, x_test))