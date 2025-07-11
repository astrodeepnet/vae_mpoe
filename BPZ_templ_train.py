import mlflow
import mlflow.tensorflow
from dust_extinction.parameter_averages import CCM89
import astropy.units as u
from tensorflow.keras.callbacks import ModelCheckpoint
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

from SpectraVAE import SpectraVAE
from BandPassVAE import BandPassVAE
from ParamVAE import ParamVAE
from ParamVAEapply import ParamVAEapply

from astropy.table import Table
from gen_dataset import *
import glob
from filters import filter_data, filters
import matplotlib.pyplot as plt
import numpy as np
from validate_HSC import validate_HSC



fig_root = "/data/kirg/MMVAE/MLFlow"
weight_path = "/data/kirg/MMVAE/bpz_tmpl_1.1"
os.makedirs(weight_path, exist_ok=True)


from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()
for device in devices:
    if device.device_type == 'GPU':
        print(device)


'''
mlflow.set_tracking_uri("file:/data/kirg/MMVAE/MLFlow/mlruns")
mlflow.set_experiment("mmvae_bpztmpl")
mlflow.start_run(nested=True)


mlflow.log_param("beta", beta)
mlflow.log_param("epochs", epochs)
mlflow.log_param("latent_dim", latent_dim)
mlflow.log_param("batch", batch)
'''


def mkdataset_BPZ():
    list_sed_templ = glob.glob('/data/kirg/MMVAE/SED_BPZ/*.sed')
    print(list_sed_templ)


    spectra_list = []
    wavelengths_list = []
    param_list = []

    z_range = np.random.uniform(0, 1.5, size=16000)
    for i, fl in enumerate(list_sed_templ):
        t = Table.read(fl, format='ascii.no_header')
        wl = t['col1']
        for z in z_range:
            wavelengths_list.append(wl * (1 + z))
            spectra_list.append(t['col2'] * wl**2)
            param_list.append([z])


    n_augmentations = 10
    normalize_spectra = True


    perturbation_sigmas = [0.1] * 5

    photocalc = PhotometricCalculator(spec_range=(3900, 10200), spec_points=100, filter_data=filter_data)

    photometry, (binned_spectra, _) = photocalc.calculate_flux_and_mag(
                spectra_list, wavelengths_list, list(photocalc.filter_data.keys())
            )


    output_array = [[[entry[1] for entry in mag], params, spec]
                for mag, params, spec in zip(photometry, param_list, binned_spectra)]

    output_array = [[[float(v) for v in flux], [float(p[0])], spec]
                for flux, p, spec in output_array]

    dataset = []
    for flux_list, params, spec in output_array:
        for _ in range(n_augmentations):
            flux_arr = np.array(flux_list)
            if normalize_spectra:
                flux_arr /= np.max(flux_arr)
            perturbed = [
                val + sigma * np.random.normal(0, 1) * val
                for val, sigma in zip(flux_arr, perturbation_sigmas)
            ]
            spectrum_norm = spec / np.max(spec) if normalize_spectra else spec
            dataset.append([perturbed, params, spectrum_norm])

    integrals = np.array([entry[0] for entry in dataset])
    params = np.array([entry[1] for entry in dataset])
    spectra = np.array([entry[2] for entry in dataset])

    return integrals, params, spectra





def run_train(beta, epochs, latent_dim, batch_size, fig_path = "/data/kirg/MMVAE/MLFlow", mlflow=mlflow, 
              save_train=False, weight_root='/data/kirg/MMVAE/weights/', n_param=1,
              tbl_path='', tbl_train_name=''):
    integrals, params, spectra = mkdataset_BPZ()
    perm = np.random.permutation(len(integrals))

    # Apply the same permutation to all three
    integrals_shuffled = integrals[perm]
    params_shuffled = params[perm]
    spectra_shuffled = spectra[perm]

    # Postprocessing (params normalization already done in DatasetBuilder)
    # Normalize spectra (just in case, or omit if confident)
    spectra /= np.max(spectra, axis=1)[:, None]


    if save_train:
        tbl = Table([integrals_shuffled, params_shuffled, spectra_shuffled], names=('SED', 'Param', 'Spec'))
        tbl.write(os.path.join(tbl_path, tbl_train_name), overwrite=True)

    weight_dir = f"beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}"
    weight_path = os.path.join(weight_root, weight_dir)
    save_path = os.path.join(weight_path, 'epochs/SpectraVAE/')
    os.makedirs(save_path, exist_ok=True)


    spvae = SpectraVAE(100, latent_dim, beta=beta)
    spvae.compile(optimizer=keras.optimizers.Adam())

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_path, 'weights_epoch_{epoch:02d}.weights.h5'),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    spvae(tf.zeros((1, 100)))


    hsp=spvae.fit(spectra, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                  callbacks=[checkpoint_cb])


    save_path = os.path.join(weight_path, 'epochs/BandPassVAE/')
    os.makedirs(save_path, exist_ok=True)


    bpvae = BandPassVAE(5,latent_dim, spvae, beta=beta)
    bpvae.compile(optimizer=keras.optimizers.Adam())

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_path, 'weights_epoch_{epoch:02d}.weights.h5'),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    bpvae(tf.zeros((1, 5)))


    h = bpvae.fit((integrals, spectra), epochs=epochs, batch_size=batch_size, validation_split=0.2,
                callbacks=[checkpoint_cb])


    save_path = os.path.join(weight_path, 'epochs/ParamVAE/')
    os.makedirs(save_path, exist_ok=True)

    parvae = ParamVAE(100, latent_dim, 
                      spvae, beta=beta,
                      n_param=n_param)

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_path, 'weights_epoch_{epoch:02d}.weights.h5'),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    parvae(tf.zeros((1, 100)))


    parvae.compile(optimizer=keras.optimizers.Adam())
    h = parvae.fit((spectra, params), epochs=epochs, batch_size=batch_size, validation_split=0.2,
                 callbacks=[checkpoint_cb])

    parvaeapply = ParamVAEapply(5, latent_dim, bpvae, parvae, beta=beta)


    pnames = ['z', 't', '[Z/H]']

    s = parvaeapply(np.reshape(integrals[:80000], (80000,5)))
    p = params[:80000]

    res = s - p

    #print(s[0], p[0])
    axs = (0, 0)


    plt.clf()
    plx = p[:, axs[0]]
    ply = res[:, axs[1]]
    plt.plot(plx, ply, 'k.')
    plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$')
    filename = f"train_plot_dz_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png"
    filename = os.path.join(fig_path, filename)
    plt.savefig(filename)
    mlflow.log_artifact(filename, artifact_path="train_plots")

    plt.clf()
    plx = p[:, axs[0]]
    ply = res[:, axs[1]]
    plt.hist2d(plx, ply, bins=100)
    #plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$')
    filename = f"train_hist_dz_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png"
    filename = os.path.join(fig_path, filename)
    plt.savefig(filename)
    mlflow.log_artifact(filename, artifact_path="train_plots")

    
    filename = [f"hsc_hist_dz_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png",
               f"hsc_hist_zz_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png",
               f"hsc_plot_dz_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png",
               f"hsc_plot_dz_bias_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png",
               f"hsc_hist_dz_bias_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png",
               f"hsc_plot_dz_bias_clean_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png",
               f"hsc_hist_dz_bias_clean_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}.png"]
    fig_dir = f"fig_beta{beta}_epochs{epochs}_latent{latent_dim}_batch{batch_size}"
    fig_path = os.path.join(fig_root, fig_dir)
    
    validate_HSC(parvaeapply, filename, fig_path='/data/kirg/MMVAE/MLFlow',
                     show = [False, False, False, False, False, False, False],
                     hsc_table='DESI_DR1_HSCSSP_clean_v2.fits',
                     mlflow=mlflow)







