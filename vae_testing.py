import pickle
import keras
from keras import layers


with open('binned_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
spectra_matrix = data[0].toarray()
dims = spectra_matrix.shape

# This is the size of our encoded representations
encoding_dim = 100  # changed from 32 in example to 100 for testing

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

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(spectra_matrix, spectra_matrix,
                epochs=50,
                batch_size=256,
                shuffle=True, validation_split=0.2)