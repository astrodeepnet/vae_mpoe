import os
import numpy as np
import matplotlib.pyplot as plt
#import corner
from astropy.io import fits
from astropy.table import Table, Column, vstack
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


latent_dim = 8 #64
beta = 1e-3
epochs = 150
n_param = 4
input_dim = 5

chunk_size = 500_000      # user-defined chunk size
n_smpl = 1               # number of model samples per chunk

# directory for saving intermediate chunks
chunk_save_dir = "/data/kirg/tmp/hsc_chunks_20260103185816"     # <--- SET THIS

os.makedirs(chunk_save_dir, exist_ok=True)

#id_run = '20251010162214'
id_run = '20260103185816'

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
    z_all_mean = np.nanmean(s_all[:, :, 0], axis=0)
    z_all_std = np.nanstd(s_all[:, :, 0], axis=0)

    a_all_mean = np.nanmean(s_all[:, :, 1], axis=0)
    a_all_std = np.nanstd(s_all[:, :, 1], axis=0)

    
    return mask, z_all_mean, z_all_std, a_all_mean, a_all_std, s_all[:, :, 0]


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
    t_chunk['z_photo_err'] = np.nan
    t_chunk['z_smpl'] = Column(np.full((len(t_chunk), 10), np.nan), dtype=float)

    t_chunk['age_photo'] = np.nan
    t_chunk['age_photo_err'] = np.nan

    # process chunk
    mask, z_chunk, zerr_chunk, a_chunk, aerr_chunk, z_sampled_chunck = process_chunk(t_chunk)

    # fill values
    t_chunk['z_photo'][mask] = z_chunk
    t_chunk['z_photo_err'][mask] = zerr_chunk
    t_chunk['age_photo'][mask] = a_chunk
    t_chunk['age_photo_err'][mask] = aerr_chunk
    t_chunk['z_smpl'][mask] = z_sampled_chunck.T

    # save intermediate file
    chunk_file = os.path.join(chunk_save_dir, f"chunk_{chunk_id:04d}.fits")
    t_chunk.write(chunk_file, overwrite=True)
    chunk_files.append(chunk_file)

    print(f"Saved chunk: {chunk_file}")

    start = end
    chunk_id += 1


# ---- MERGE ALL CHUNKS ----
print("Merging chunks (streaming to FITS)...")

out_file = "HSC_SSP_zphoto_20260103185816.fits"
first = True

for f in chunk_files:
    print("Processing", f)

    # Read next chunk (only this chunk in memory)
    t = Table.read(f)

    if first:
        # Write first chunk, create file
        t.write(out_file, overwrite=True)
        first = False
    else:
        # Append rows to the SAME BINTABLE HDU
        t.write(out_file, append=True)