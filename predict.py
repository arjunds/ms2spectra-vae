import keras
import pickle
import tensorflow_addons
import matplotlib.pyplot as plt
import matplotlib

with open("filtered_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("binned_filtered_0.5.pkl", "rb") as f:
    x = pickle.load(f)

scans = x[2]

x_test = data[1]
indices = data[2]

colors = []
labels = ["antimicrobial", "random"]

for index in indices:
    if scans[index].find("filtered") != -1:
        colors.append(0)
    else:
        colors.append(1)

autoencoder = keras.models.load_model("Filtered_GNPSv2", compile=False)
encoder = autoencoder.get_layer(index=1)
cmap = matplotlib.colors.ListedColormap(['orange','blue'])
x_test_encoded = encoder.predict(x_test)
scatter = plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=colors, s=10, cmap=cmap)
plt.legend(handles=scatter.legend_elements()[0], loc="upper left", labels=labels)
plt.show()
