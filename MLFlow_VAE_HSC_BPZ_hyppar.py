import mlflow
import mlflow.tensorflow

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import ops
from keras import layers
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.interpolate import interp1d
import glob
from scipy import interpolate

from BPZ_templ_train import run_train

expname = "mmvae_bpz_templ_v1"

mlflow.set_tracking_uri("file:/data/kirg/MMVAE/MLFlow/mlruns")
mlflow.set_experiment(expname)

beta_values = [1e-5, 3e-5, 1e-4, 3e-4, 
               1e-3, 3e-3, 1e-2, 3e-2, 
               0.1, 0.2,0.3,0.7,1.0]
epochs_values = [100]
latent_dims = [4, 8, 16, 32, 64]
batchsize_values = [512, 256, 128, 64]

for beta in beta_values:
    for epochs in epochs_values:
        for latent_dim in latent_dims:
            for batch_size in batchsize_values:

                with mlflow.start_run(nested=True):
                    mlflow.log_param("beta", beta)
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_param("latent_dim", latent_dim)
                    mlflow.log_param("batch_size", batch_size)

                    run_train(beta=beta, epochs=epochs, 
                              latent_dim=latent_dim,
                              batch_size=batch_size,
                              fig_path="/data/kirg/MMVAE/MLFlow", 
                              weight_root='/data/kirg/MMVAE/weights/'+expname, 
                              mlflow=mlflow)
