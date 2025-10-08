import tensorflow as tf
import keras
from keras import ops
from keras import layers
from utils import Sampling
import numpy as np


class ParamVAE(keras.Model):
    def build_dense_decoder_param(self, latent_dim, output_dim):
        latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(64, activation='sigmoid')(latent_inputs)
        x = layers.Dense(64, activation='sigmoid')(x)
        #x = layers.Dense(16, activation='sigmoid')(x)
        #x = layers.Dense(16, activation='relu')(x)
        #x = layers.Dense(32, activation='sigmoid')(x)
        x = layers.Dense(32, activation='sigmoid')(x)
        x = layers.Dense(32, activation='sigmoid')(x)
        x = layers.Dense(16, activation='sigmoid')(x)
        x = layers.Dense(8)(x)
        #x = layers.Dense(8, activation='sigmoid')(x)
        #x = layers.Dense(8)(x)
        #x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(output_dim[0])(x)
    
        decoder = keras.Model(latent_inputs, outputs, name='dense_decoder')
        return decoder

    def __init__(self, input_dim, latent_dim, spvae, n_param=3, beta=1, wei=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_branch = spvae.encoder_branch #build_encoder_sp_branch((input_dim,1))

        self.z_mean = layers.Dense(latent_dim, name='z_mean')(self.encoder_branch.output)
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')(self.encoder_branch.output)

        self.z = Sampling()([self.z_mean, self.z_log_var])
        self.wei = np.array([1] * n_param)
        #self.wei[0] = 10.0
        if wei == None:
            self.wei = self.wei / np.sum(self.wei)
        else:
            self.wei = wei / np.sum(wei)
        self.encoder = spvae.encoder
        self.encoder.trainable = False  # Prevent training of the decoder

        self.decoder = self.build_dense_decoder_param(latent_dim,(n_param,1))
        self.decoder.trainable = True  # Prevent training of the decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta = beta
        #super().__init__(inputs=self.encoder.input, outputs=self.decoder.output, **kwargs)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def apply(self, data):
        data_in = data
        z_mean, z_log_var, z = self.encoder(data_in)
        reconstruction = self.decoder(z)
        return (z_mean, z_log_var, z, reconstruction)

    #def call(self, data):
    #    (data_in, data_out) = data[0]
    #    return self.apply(data_in)[3]

    def train_step(self, data):
        (data_in, data_out) = data[0]
        with tf.GradientTape() as tape:
            (z_mean, z_log_var, z, reconstruction) = self.apply(data_in)
            reconstruction_loss = ops.mean(
                tf.keras.backend.square(data_out*self.wei - reconstruction*self.wei)
                #tf.keras.backend.square(data_out - reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + 0*kl_loss*self.beta
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
       }

    def test_step(self, data):
        (data_in, data_out) = data[0]
        (z_mean, z_log_var, z, reconstruction) = self.apply(data_in)
        reconstruction_loss = ops.mean(
                tf.keras.backend.square(data_out*self.wei - reconstruction*self.wei)
            )
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss*self.beta
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, data):
        data_in  = data
        z_mean, z_log_var, z = self.encoder(data_in)
        reconstruction = self.decoder(z)
        return reconstruction