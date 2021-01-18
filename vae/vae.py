import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


latent_dim = 16

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape):
    encoder_input = keras.Input(shape=(64,64,3))
    x = layers.Conv2D(32, activation='relu', kernel_size=4, 
strides=2)(encoder_input)  # -> 31x31x32
    x = layers.Conv2D(64, activation='relu', kernel_size=4, strides=2)(x)              
# -> 14x14x64
    x = layers.Conv2D(128, activation='relu', kernel_size=4, strides=2)(x)             
# -> 6x6x128
    x = layers.Conv2D(256, activation='relu', kernel_size=4, strides=2)(x)  
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], 
name="encoder")
    encoder.summary()
    return encoder

def build_decoder():
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(2 * 2 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((2, 2, 256))(x)
    x = layers.Conv2DTranspose(128, activation='relu', kernel_size=4, 
strides=2)(x)     # -> 6x6x128
    x = layers.Conv2DTranspose(64, activation='relu', kernel_size=4, 
strides=2)(x)      # -> 14x14x64
    x = layers.Conv2DTranspose(32, activation='relu', kernel_size=4, 
strides=2)(x)      # -> 30x30x32
    decoder_outputs = layers.Conv2DTranspose(3, activation='sigmoid', 
kernel_size=6, strides=2)(x)  # -> 64x64x3
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": 
            self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def train(digits):
    encoder = build_encoder(digits[0].shape)
    decoder = build_decoder()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(digits, epochs=30, batch_size=128)
    return vae

