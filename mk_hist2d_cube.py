import numpy as np
from astropy.io import fits

def make_hist_cube_and_save(ra, dec, z_photo,
                            z_min=0.0, z_max=1.0, dz=0.05,
                            ra_bins=200, dec_bins=200,
                            output_file="hist_cube.fits"):
    """
    Compute 2D histograms (ra, dec) for each z slice and save all slices
    into a single FITS 3D cube file.
    """

    # Define z slices
    z_edges = np.arange(z_min, z_max + dz, dz)
    n_z = len(z_edges) - 1

    # Compute RA/DEC bin edges once
    ra_min, ra_max = np.nanmin(ra), np.nanmax(ra)
    dec_min, dec_max = np.nanmin(dec), np.nanmax(dec)

    edges_ra = np.linspace(ra_min, ra_max, ra_bins + 1)
    edges_dec = np.linspace(dec_min, dec_max, dec_bins + 1)

    # Allocate 3D array for all histograms
    # dimensions: (z slice index, ra bin index, dec bin index)
    hist_cube = np.zeros((n_z, ra_bins, dec_bins), dtype=np.float32)

    # ---- Loop over slices ----
    for i in range(n_z):
        z1, z2 = z_edges[i], z_edges[i+1]
        print(f"Processing slice {i+1}/{n_z}: {z1:.2f}–{z2:.2f}")

        # Mask objects inside the slice
        m = (z_photo >= z1) & (z_photo < z2)

        # Compute 2D histogram
        H, _, _ = np.histogram2d(
            ra[m], dec[m],
            bins=[edges_ra, edges_dec]
        )

        # Store slice histogram in the cube
        hist_cube[i, :, :] = H

    # ---- Save to a single FITS cube ----
    hdu = fits.PrimaryHDU(hist_cube)
    hdr = hdu.header

    # Add useful metadata
    hdr['ZMIN'] = z_min
    hdr['ZMAX'] = z_max
    hdr['DZ'] = dz
    hdr['RBINS'] = ra_bins
    hdr['DBINS'] = dec_bins

    hdul = fits.HDUList([hdu])

    # Save the file
    hdul.writeto(output_file, overwrite=True)
    print(f"Saved histogram cube to: {output_file}")

    return hist_cube, edges_ra, edges_dec, z_edges

if __name__ == "__main__":
    from astropy.table import Table
    '''
    t = Table.read('HSC_SSP_zphoto.fits')
    make_hist_cube_and_save(t['ra'], t['dec'], t['z_photo'], ra_bins=500, dec_bins=500)
    '''
    t = Table.read('HSC_SSP_zphoto.fits')

    # z_smpl has shape (N, 10)
    z_smpl = t['z_smpl']
    ra = t['ra']
    dec = t['dec']
    
    # --- Expand arrays ---
    # z_flat:   N*10 redshifts
    # ra_flat:  N*10 RAs (each repeated 10×)
    # dec_flat: N*10 DECs (each repeated 10×)
    
    z_flat   = z_smpl.reshape(-1)               # shape (N*10,)
    ra_flat  = np.repeat(ra, 10)                # shape (N*10,)
    dec_flat = np.repeat(dec, 10)               # shape (N*10,)
    print(z_flat.shape)
    
    # --- Run histogram ONCE for the whole expanded sample ---
    make_hist_cube_and_save(
        ra_flat,
        dec_flat,
        z_flat,
        ra_bins=500,
        dec_bins=500
    )


