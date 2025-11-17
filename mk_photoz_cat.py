import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.io import fits
from astropy.table import Table, vstack
from ParamVAEapply import ParamVAEapply
from BandPassVAE import BandPassVAE
from SpectraVAE import SpectraVAE
from ParamVAE import ParamVAE
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
import numpy.ma as ma

def load_paramvaeapply(path: str, weight_file: str, input_dim: int, latent_dim: int, beta: float) -> ParamVAEapply:
    """
    Loads weights and builds a ParamVAEapply model from a saved epoch weight file.

    Parameters:
        weight_file (str): Path to the weights file (.h5).
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Latent space dimensionality.
        beta (float): Beta value for VAE loss.

    Returns:
        ParamVAEapply: A ready-to-use model with loaded weights.
    """
    # Build required submodels
    
    spvae_wf = os.path.join(os.path.join(path, 'SpectraVAE'), weight_file)
    bpvae_wf = os.path.join(os.path.join(path, 'BandPassVAE'), weight_file)
    parvae_wf = os.path.join(os.path.join(path, 'ParamVAE'), weight_file)
    print(spvae_wf, bpvae_wf, parvae_wf)
    spvae = SpectraVAE(100, latent_dim, beta=beta)
    spvae.compile(optimizer=keras.optimizers.Adam())
    spvae(tf.zeros((1, 100)))  # Force model building
    spvae.load_weights(spvae_wf)

    bpvae = BandPassVAE(input_dim, latent_dim, spvae, beta=beta)
    bpvae.compile(optimizer=keras.optimizers.Adam())
    bpvae(tf.zeros((1, input_dim)))  # Force model building
    bpvae.load_weights(bpvae_wf)

    # Attach ParamVAE
    parvae = ParamVAE(100, latent_dim, spvae, beta=beta, n_param=4)
    parvae.compile(optimizer=keras.optimizers.Adam())
    parvae(tf.zeros((1, 100)))  # Force model building
    parvae.load_weights(parvae_wf)

    return ParamVAEapply(input_dim, latent_dim, bpvae, parvae, beta=beta), parvae, bpvae, spvae


latent_dim = 64
beta = 1e-3
epochs = 150
n_param = 4
input_dim = 5

chunk_size = 100_000      # user-defined chunk size
n_smpl = 10               # number of model samples per chunk

# directory for saving intermediate chunks
chunk_save_dir = "/data/kirg/tmp/hsc_chunks"     # <--- SET THIS

os.makedirs(chunk_save_dir, exist_ok=True)

id_run = '20251010162214'
weight_path = "/data/kirg/MMVAE/cigale/" + id_run + "/"
save_path = os.path.join(weight_path, 'epochs/')
plot_output_path = os.path.join(weight_path, 'cornerplots_train_' + id_run)
weight_file = f"weights_epoch_{epochs:02d}.weights.h5"

parvaeapply, parvae, bpvae, spvae = load_paramvaeapply(
    save_path, weight_file, input_dim, latent_dim, beta
)

# ---- READ INPUT TABLE ----
fits_file = 'HSC_SSP_photcat.fits'
t_hsc_full = Table.read(fits_file)

# spatial selection
cen_ra = 39.2721
cen_dec = -3.9546
t_hsc = t_hsc_full
'''    
[
    (np.abs(t_hsc_full['ra'] - cen_ra) +
     np.abs(t_hsc_full['dec'] - cen_dec)) < 240./60.
]
'''
N = len(t_hsc)
print(f"Total rows to process: {N}")


# ---- FUNCTION TO PROCESS ONE CHUNK ----
def process_chunk(t_chunk):
    """
    Process a single chunk: extract magnitudes, convert to flux,
    normalize, run the model multiple times, and return results.
    """

    # build magnitude matrix (MaskedArray)
    mags = ma.vstack([
        t_chunk['g_kronflux_mag'],
        t_chunk['r_kronflux_mag'],
        t_chunk['i_kronflux_mag'],
        t_chunk['z_kronflux_mag'],
        t_chunk['y_kronflux_mag']
    ]).T

    # replace masked values → nan
    mags = mags.filled(np.nan)

    # convert magnitudes to flux
    band_flux = 10**(-0.4 * mags)

    # compute row-wise maximum flux
    max_flux = np.nanmax(band_flux, axis=1)

    # mask of valid rows
    mask = np.isfinite(max_flux)

    # normalize fluxes
    band_flux_norm = (band_flux[mask].T / max_flux[mask]).T
    hsc_input = band_flux_norm.reshape(-1, input_dim)

    if len(hsc_input) == 0:
        return mask, np.array([])

    # run the model multiple times
    s_all = []
    for _ in range(n_smpl):
        s = parvaeapply(hsc_input)
        s_all.append(s)

    s_all = np.array(s_all)

    # mean over samples
    s_all_mean = np.nanmean(s_all[:, :, 0], axis=0)

    return mask, s_all_mean


# ---- PROCESS AND SAVE CHUNKS ----
chunk_files = []
start = 0
chunk_id = 0

while start < N:
    end = min(start + chunk_size, N)
    print(f"Processing rows {start}:{end}")

    # extract current chunk
    t_chunk = t_hsc[start:end]
    t_chunk = t_chunk.copy()       # ensure independence
    t_chunk['z_photo'] = np.nan

    # process chunk
    mask, z_chunk = process_chunk(t_chunk)

    # fill values
    t_chunk['z_photo'][mask] = z_chunk

    # save intermediate file
    chunk_file = os.path.join(chunk_save_dir, f"chunk_{chunk_id:04d}.fits")
    t_chunk.write(chunk_file, overwrite=True)
    chunk_files.append(chunk_file)

    print(f"Saved chunk: {chunk_file}")

    start = end
    chunk_id += 1


# ---- MERGE ALL CHUNKS ----
print("Merging chunks...")
tables = [Table.read(f) for f in chunk_files]
t_out = vstack(tables, metadata_conflicts='silent')

out_file = "HSC_SSP_zphoto.fits"
t_out.write(out_file, overwrite=True)

print(f"Final output written to: {out_file}")
