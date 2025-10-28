import tensorflow as tf
import keras
from keras import ops
from keras import layers
from utils import Sampling


class MonteCarloDropout(layers.Dropout):
    def call(self, inputs, training=True):
        return super().call(inputs, training=True)


class BandPassVAE(keras.Model):
    def build_dense_encoder_sed_branch(self, input_shape, dropout_rate=0.1):
        dense_input = keras.Input(shape=input_shape)
        x = layers.Dense(input_shape[0], activation='sigmoid')(dense_input)
        #x = layers.Dense(8, activation='relu')(x)
        #x = MonteCarloDropout(dropout_rate)(x)
        x = layers.Dense(16, activation='sigmoid')(x)
        #x = layers.Dense(16, activation='relu')(x)
        #x = layers.Dense(32, activation='relu')(x)
        #x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='sigmoid')(x)
        #x = MonteCarloDropout(dropout_rate)(x)
        x = MonteCarloDropout(dropout_rate)(x)
        x = layers.Dense(16, activation='sigmoid')(x)
        #x = layers.Dense(64, activation='relu')(x)
        return keras.Model(dense_input, x, name='dense_encoder_branch')

    def __init__(self, input_dim, latent_dim, spvae, beta=1, **kwargs):
        super().__init__(**kwargs)
        self.encoder_branch = self.build_dense_encoder_sed_branch((input_dim,))

        self.z_mean = layers.Dense(latent_dim, name='z_mean')(self.encoder_branch.output)
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')(self.encoder_branch.output)

        self.z = Sampling()([self.z_mean, self.z_log_var])

        self.encoder = keras.Model(self.encoder_branch.inputs, [self.z_mean, self.z_log_var, self.z], name='dense_encoder')

        self.decoder = spvae.decoder
        self.decoder.trainable = False  # Prevent training of the decoder
        self.encoder_spec = spvae.encoder
        self.encoder_spec.trainable = False
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

    def call(self, data):
        (data_in, data_out) = data[0]
        return self.apply(data_in)[3]

    def train_step(self, data):
        (data_in, data_out) = data[0]
        with tf.GradientTape() as tape:
            (z_mean, z_log_var, z, reconstruction) = self.apply(data_in)
            (z_mean_spec, z_log_var_spec, z_spec) = self.encoder_spec(data_out)

            reconstruction_loss = ops.mean(
                tf.keras.backend.mean(tf.keras.backend.square((data_out - reconstruction)/data_out))
            )

            latent_loss = ops.mean(
                tf.keras.backend.mean(tf.keras.backend.square((z - z_spec)))
            )
                        
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss*0 + latent_loss + kl_loss*self.beta
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
        print(data)
        (data_in, data_out) = data[0]
        (z_mean, z_log_var, z, reconstruction) = self.apply(data_in)
        reconstruction_loss = ops.mean(
                tf.keras.backend.mean(tf.keras.backend.square((data_out - reconstruction)/data_out))
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