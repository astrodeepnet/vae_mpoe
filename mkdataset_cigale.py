import numpy as np
from astropy.table import Table
from gen_dataset import *
from filters import filter_data, filters

def mkdataset_cigale():
    t = Table.read('exp_cigale.fits')
    print(len(t))
    spectra_list = []
    wavelengths_list = []
    param_list = []
 
    for i, r in enumerate(t):
        z_range = np.random.uniform(0, 1.5, size=10)
        for z in z_range:
            wl_ = r['wl'] * 10 * (1 + z)
            wl = wl_[((wl_ > 3500) & (wl_ < 12000))]
            flx_ = r['flx']
            flx = flx_[((wl_ > 3500) & (wl_ < 12000))]
            wavelengths_list.append(wl)
            spectra_list.append(flx)
            param_list.append([z, r['age'], r['alpha'], r['met']])
            #print(z, r['age'], r['alpha'], r['met'])


    n_augmentations = 10
    normalize_spectra = True


    perturbation_sigmas = [0.1] * 5

    photocalc = PhotometricCalculator(spec_range=(3900, 10200), spec_points=30, filter_data=filter_data)

    photometry, (binned_spectra, wllr) = photocalc.calculate_flux_and_mag(
                spectra_list, wavelengths_list, list(photocalc.filter_data.keys())
            )


    output_array = [[[entry[1] for entry in mag], params, spec]
                for mag, params, spec in zip(photometry, param_list, binned_spectra)]

    output_array = [[[float(v) for v in flux], [float(p[0]), float(p[1]), float(p[2]), float(p[3])], spec]
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

    return integrals, params, spectra, wllr





if __name__ == "__main__":
    integrals, params, spectra = mkdataset_cigale()