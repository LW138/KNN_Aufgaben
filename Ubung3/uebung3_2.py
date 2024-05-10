import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Laden des MNIST-Datensatzes
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Berechnen der durchschnittlichen Bilder für jede Klasse
average_images = []
for i in range(10):
    # Auswahl der Bilder für die aktuelle Klasse (0-9) (Masking)
    idx = train_labels == i

    print(idx)
    # Berechnung des Durchschnitts
    average_image = np.mean(train_images[idx], axis=0)
    average_images.append(average_image)

# Visualisieren der Durchschnittsbilder
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes):
    ax.imshow(average_images[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Ziffer: {i}')

plt.show()