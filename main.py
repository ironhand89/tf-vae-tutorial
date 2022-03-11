import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model

# noinspection PyUnresolvedReferences
from tensorflow.keras import layers, losses
# noinspection PyUnresolvedReferences
from tensorflow.keras.datasets import fashion_mnist
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

latent_dim = 64


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D()
        ])

if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    noise = 0.2
    x_train_noisy = x_train + noise * tf.random.normal(shape=x_train.shape)
    x_test_noisy = x_test + noise * tf.random.normal(shape=x_test.shape)
    
    x_train_noisy_clipped = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy_clipped = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)



    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_train_noisy[i])
        plt.title("train_noisy")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(x_train_noisy_clipped[i])
        plt.title("train_noisy_clipped")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # print(x_train.shape)
    # print(x_test.shape)

    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    # autoencoder = load_model('/Users/franz/Privat/TUKL/MasterThesis/Tutorials/tf-vae-tutorial/ae-models')
    autoencoder.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
    # autoencoder.save('/Users/franz/Privat/TUKL/MasterThesis/Tutorials/tf-vae-tutorial/ae-models')
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
