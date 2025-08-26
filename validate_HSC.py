import numpy as np
from astropy.table import Table
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import os

def validate_HSC(parvaeapply, filenames, fig_path,
                 show = [False, False, False, False, False, False, False],
                 hsc_table='DESI_DR1_HSCSSP_clean_v2.fits',
                 mlflow = False, z_min=0.1, z_max=0.95):
    t_hsc_ = Table.read('DESI_DR1_HSCSSP_clean_v2.fits')
         
    
    hsc_int = []
    hsz_z = []
    t_hsc = t_hsc_[((t_hsc_['z'] > z_min) & (t_hsc_['z'] < z_max) & (t_hsc_['zwarn'] == 0))]
    
    for r in t_hsc:
        mags = mags = np.array([r['g_kronflux_mag'], r['r_kronflux_mag'], r['i_kronflux_mag'], r['z_kronflux_mag'], r['y_kronflux_mag']])
        bandfl = 10**(-0.4*mags)
        bandfl /= np.max(bandfl)
        hsc_int.append(bandfl)
        hsz_z.append(r['z'])
    
    n = len(hsc_int)
    pnames = ['z', 't', '[Z/H]']
    
    s = parvaeapply(np.reshape(hsc_int[:n], (n,5)))
    p = hsz_z[:n]
    
    res = s[:, 0] - p
    
    axs = (0, 0)
    
    
    plx = p[:]
    ply = res[:]
        
    plt.clf()
    plt.hist2d(plx, ply, range=[[z_min, z_max], [-1, 1]], bins=40)
    plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$')
    #filename = f"plot_dz_beta{beta}_epochs{epochs}_latent{latent_dim}.png"
    filename = os.path.join(fig_path, filenames[0])
    plt.savefig(filename)
    if show[0]: 
        plt.show()
    if mlflow:
        mlflow.log_artifact(filename, artifact_path="valid_plots")

    #mlflow.log_artifact(filename, artifact_path="plots")

    plt.clf()
    plx = p[:]
    ply = res[:]
    plt.hist2d(p[:], s[:, 0], range=[[z_min, z_max], [-0.5, z_max+0.5]], bins=40)
    plt.plot([0, 1], [0, 1], color='red')
    plt.ylim(np.min(p[:]), np.max(p[:]))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$ ' + pnames[axs[1]] + '$')
    #filename = f"plot_zz_beta{beta}_epochs{epochs}_latent{latent_dim}.png"
    filename = os.path.join(fig_path, filenames[1])
    plt.savefig(filename)
    if show[1]: 
        plt.show()
    if mlflow:
        mlflow.log_artifact(filename, artifact_path="valid_plots")

    
    #mlflow.log_artifact(filename, artifact_path="plots")
    
    t_hsc['z_photo'] = s[:, 0]

    hsc_int = []
    hsz_z = []
    t_hsc = t_hsc_[((t_hsc_['z'] > z_min) & (t_hsc_['z'] < z_max) & (t_hsc_['zwarn'] == 0) &
                    (t_hsc_['g_kronflux_mag'] > 0) &
                    (t_hsc_['r_kronflux_mag'] > 0) &
                    (t_hsc_['i_kronflux_mag'] > 0) &
                    (t_hsc_['z_kronflux_mag'] > 0) &
                    (t_hsc_['y_kronflux_mag'] > 0))]
    
    for r in t_hsc:
        mags = mags = np.array([r['g_kronflux_mag'], r['r_kronflux_mag'], r['i_kronflux_mag'], r['z_kronflux_mag'], r['y_kronflux_mag']])
        bandfl = 10**(-0.4*mags)
        bandfl /= np.max(bandfl)
        hsc_int.append(bandfl)
        hsz_z.append(r['z'])
    
    n = len(hsc_int)
    pnames = ['z', 't', '[Z/H]']
    
    s = parvaeapply(np.reshape(hsc_int[:n], (n,5)))
    p = hsz_z[:n]
    
    res = s[:, 0] - p
    print(s[0], p[0])
    axs = (0, 0)
    
    plt.clf()
    plx = p[:]
    ply = res[:]
    plt.plot(plx, ply, marker='.', color='black', linestyle='None', alpha=0.005)
    plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$')
    filename = os.path.join(fig_path, filenames[2])
    plt.savefig(filename)
    if show[2]: 
        plt.show()
    if mlflow:
        mlflow.log_artifact(filename, artifact_path="valid_plots")

    plx = np.array(p[:])
    ply = np.array(res[:])
    print(plx)
    # Set number of bins
    N = 20  # You can change this to whatever number of bins you want
    
    # Digitize plx into N bins
    bins = np.linspace(np.min(plx), np.max(plx), N + 1)
    indices = np.digitize(plx, bins)
    
    # Subtract median ply in each bin
    corrected_ply = ply
    for i in range(1, N + 1):
        bin_mask = indices == i
        if np.any(bin_mask):
            median_val = np.median(ply[bin_mask])
            corrected_ply[bin_mask] -= median_val
            
    plt.clf()
    # Plot the corrected data
    plt.plot(plx, corrected_ply, marker='.', color='black', linestyle='None', alpha=0.005)
    plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$ (median subtracted)')
    filename = os.path.join(fig_path, filenames[3])
    plt.savefig(filename)
    if show[3]: 
        plt.show()
    if mlflow:
        mlflow.log_artifact(filename, artifact_path="valid_plots")

    plt.clf()
    plt.hist2d(plx, ply, bins=40)
    plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$')
    filename = os.path.join(fig_path, filenames[4])
    plt.savefig(filename)
    if show[4]: 
        plt.show()
    
    N = 20
    
    # Digitize plx into N bins
    bins = np.linspace(np.min(plx), np.max(plx), N + 1)
    indices = np.digitize(plx, bins)
    
    # Subtract sigma-clipped mean of ply in each bin
    corrected_ply = ply.copy()
    for i in range(1, N + 1):
        bin_mask = indices == i
        if np.any(bin_mask):
            clipped = sigma_clip(ply[bin_mask], sigma=2.5, maxiters=20)
            mean_val = np.median(clipped.data[~clipped.mask])
            print(np.mean(plx[bin_mask]), mean_val)
            corrected_ply[bin_mask] -= mean_val
    
    plt.clf()
    plt.plot(plx, corrected_ply, marker='.', color='black', linestyle='None', alpha=0.005)
    plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$')
    filename = os.path.join(fig_path, filenames[5])
    plt.savefig(filename)
    if show[5]: 
        plt.show()
    if mlflow:
        mlflow.log_artifact(filename, artifact_path="valid_plots")

    plt.clf()
    plt.hist2d(plx, corrected_ply, bins=40)
    plt.ylim(-np.max(plx), np.max(plx))
    plt.xlabel('$' + pnames[axs[0]] + '$')
    plt.ylabel('$\Delta ' + pnames[axs[1]] + '$')
    plt.colorbar(label='Counts')
    filename = os.path.join(fig_path, filenames[6])
    plt.savefig(filename)
    if show[6]: 
        plt.show()
    if mlflow:
        mlflow.log_artifact(filename, artifact_path="valid_plots")



