import os
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import interpolate
from scipy.interpolate import interp1d
from dust_extinction.parameter_averages import CCM89
import astropy.units as u
import requests
import multiprocessing as mp
import copy

class SpectrumReader:
    def __init__(self, pattern):
        self.pattern = pattern

    def read(self):
        files = sorted(glob.glob(self.pattern))
        if not files:
            raise FileNotFoundError("No FITS files found matching pattern.")

        spectra_list, ages, mets = [], [], []

        for file in files:
            with fits.open(file) as hdul:
                flux_data = hdul[0].data
                params = hdul["ETS_PARA"].data
                wl = hdul[3].data['BFIT']
                flux_data = np.array([fl * wl**2 for fl in flux_data])

                age = params["AGE"]
                met = params["ZSTARS"]
                mask = age > 100

                spectra_list.append(flux_data[mask])
                ages.append(np.log10(age[mask]))
                mets.append(np.round(np.log10(met[mask] / 0.02), 2))

        return np.array(ages), np.array(mets), np.stack(spectra_list, axis=1), wl


class SpectrumInterpolator:
    def __init__(self, lages, lmets, flx):
        self.lages = np.unique(lages)
        self.mets = np.unique(lmets)
        self.interpolators = [
            interpolate.RectBivariateSpline(self.lages, self.mets, flx[:, :, i])
            for i in range(flx.shape[2])
        ]

    def evaluate(self, age, metallicity):
        log_age = np.log10(age)
        return np.array([interp(log_age, metallicity)[0, 0] for interp in self.interpolators])


class SpectrumProcessor:
    def __init__(self, interpolator):
        self.interpolator = interpolator
        self.emission_lines = {
            'Halpha': (6562.8, 1.0),      # Hα (reference)
            'Hbeta': (4861.3, 0.35),      # Hβ
            '[OIII]5007': (5006.8, 0.5),  # [O III] 5007
            '[NII]6583': (6583.4, 0.3),  # [N II] 6583
            '[SII]6716': (6716.4, 0.1),  # [S II] 6716
            '[SII]6731': (6730.8, 0.1),   # [S II] 6731
            '[OII]3727': (3727.1, 0.2)    # [O II] 3727 (doublet)
        }

    def calculate(self, wl, age, met, z=0.0, Av=0.0, Rv=3.1, Ha_flux_ratio=0.0, 
                  continuum_strength=0.0, continuum_slope=1.5):
        wl = np.asarray(wl)
        spec = self.interpolator.evaluate(age, met)
        wl_z = wl * (1 + z)
        wl_z_q = (wl_z * 1e-4) * u.micron

        # Apply extinction
        if Av > 0:
            ext_model = CCM89(Rv=Rv)
            inp_ext = 1.0 / wl_z_q
            inp_ext[inp_ext.value < 0.3] = 0.3 / u.micron
            inp_ext[inp_ext.value > 10] = 10.0 / u.micron
            extinction = ext_model(inp_ext) * Av
            attenuation = 10 ** (-0.4 * extinction)
            spec *= attenuation

        # Add power-law continuum if specified
        if continuum_strength > 0:
            # Power-law: F_λ ∝ λ^(-k)
            continuum = wl_z ** (-continuum_slope)

            # Normalize in 3400–3800 Å range
            norm_mask = (wl >= 3700) & (wl <= 4100)
            if np.any(norm_mask):
                norm_factor = np.mean(continuum[norm_mask])/np.mean(spec[norm_mask])
                continuum /= norm_factor  # Normalize to mean=1 in that band
                continuum *= continuum_strength  # Scale by input strength
                spec += continuum

        # Compute flux in 5600–6800 Å for line scaling
        mask = (wl >= 5600) & (wl <= 6800)
        total_flux_5600_6800 = np.trapz(spec[mask], wl_z[mask])

        # Add emission lines
        if Ha_flux_ratio > 0:
            Ha_flux = Ha_flux_ratio * total_flux_5600_6800
            emission_spec = np.zeros_like(spec)

            for line_name, (line_wl, ratio) in self.emission_lines.items():
                line_flux = Ha_flux * ratio
                line_wl_z = line_wl * (1 + z)

                sigma = 2.5 / 2.3548
                gauss = np.exp(-0.5 * ((wl_z - line_wl_z) / sigma) ** 2)
                gauss *= line_flux / (sigma * np.sqrt(2 * np.pi))

                emission_spec += gauss

            spec += emission_spec

        return spec, wl_z

class PhotometricCalculator:
    def __init__(self, spec_range, spec_points, filter_data):
        self.spec_range = spec_range
        self.spec_points = spec_points
        self.filter_data = filter_data

    def bin_spectrum(self, spec, wl):
        bins = np.linspace(*self.spec_range, self.spec_points + 1)
        mean_values, wl_values = [], []
        for i in range(len(bins) - 1):
            mask = (wl >= bins[i]) & (wl < bins[i + 1])
            bin_data = spec[mask]
            mean_values.append(bin_data.mean() if bin_data.size > 0 else (bins[i] + bins[i + 1]) / 2)
            wl_values.append(0.5 * (bins[i] + bins[i + 1]))
        return np.array(mean_values), np.array(wl_values)

    def calculate_flux_and_mag(self, spectra_list, wavelengths_list, filters):
        all_results, all_speclr, all_wllr = [], [], []
        for spectra, wavelengths in zip(spectra_list, wavelengths_list):
            speclr, wllr = self.bin_spectrum(spectra, wavelengths)
            result = []
            for filter_name in filters:
                #print(filter_data)
                fd = self.filter_data[filter_name]
                fwl, tr = fd['wl'], fd['tr']
                interp_flux = interp1d(wavelengths, spectra, kind='linear', fill_value=0, bounds_error=False)
                flux_interp = interp_flux(fwl)
                total_flux = np.trapz(flux_interp * tr, fwl) / np.trapz(tr, fwl)
                ab_mag = -2.5 * np.log10(total_flux) - 48.6 if total_flux > 0 else np.inf
                result.append([filter_name, total_flux, ab_mag])
            all_results.append(result)
            all_speclr.append(speclr)
            all_wllr.append(wllr)
        return all_results, [all_speclr, all_wllr]


import numpy as np
import multiprocessing as mp
import copy

# Global variables for multiprocessing
_global_processor = None
_global_wavelength_grid = None

def _init_worker(processor, wavelength_grid):
    global _global_processor, _global_wavelength_grid
    _global_processor = processor
    _global_wavelength_grid = wavelength_grid

def _generate_single_sample(args):
    z, age, metdex, config = args

    processor = _global_processor
    wl_grid = _global_wavelength_grid

    spectrum, wl_z = processor.calculate(wl_grid, age, metdex, z=z)
    ha_strength = 0
    Av = 0
    continuum_strength = 0
    if age < config['young_age_threshold'] and np.random.rand() < config['young_extinction_prob']:
        Av = 10 ** np.random.uniform(np.log10(config['extinction_range'][0]), np.log10(config['extinction_range'][1]))

    if age < config['young_age_threshold'] and np.random.rand() < config['eml_prob']:
        ha_strength = 10 ** np.random.uniform(np.log10(3e-3), np.log10(5e-2))
        continuum_strength = 0.8


    spectrum, _ = processor.calculate(wl_grid, age, metdex, z=z, Av=Av, 
                                      Ha_flux_ratio=ha_strength, 
                                      continuum_strength=continuum_strength,
                                      continuum_slope=0.0)

    if age > config['old_age_blend_threshold'] and np.random.rand() < config['old_blend_prob']:
        weight = np.random.rand()
        young_age = np.random.uniform(*config['young_age_range'])
        blended_spectrum, _ = processor.calculate(wl_grid, young_age, metdex, z=z)
        spectrum = (1 - weight) * spectrum + weight * blended_spectrum
        age = (1 - weight) * age + weight * young_age

    
    if config['normalize_spectra']:
        spectrum /= np.max(spectrum)

    return spectrum, wl_z, [z, age, metdex]


class DatasetBuilder:
    def __init__(self, processor, wavelength_grid, photometric_calculator):
        self.processor = processor
        self.wavelength_grid = wavelength_grid
        self.photometric_calculator = photometric_calculator

    def generate(
        self,
        n_samples: int,
        z_range=(0.0, 1.0),
        age_range=(500, 15000),
        metdex_range=(-0.5, 0.1),
        n_young_fraction=0.1,
        young_age_range=(500, 2000),
        old_age_blend_threshold=7000,
        old_blend_prob=0.3,
        young_age_threshold=3000,
        young_extinction_prob=0.5,
        eml_prob=0.9,
        extinction_range=(0.01, 0.5),
        extinction_rv=3.1,
        n_augmentations=20,
        perturbation_sigmas=None,
        normalize_spectra=True,
        n_processes=4
    ):
        if perturbation_sigmas is None:
            perturbation_sigmas = [0.01 for _ in range(5)]

        z_vals = np.random.uniform(*z_range, n_samples)
        age_vals = 10 ** np.random.uniform(np.log10(age_range[0]), np.log10(age_range[1]), n_samples)
        metdex_vals = np.random.uniform(*metdex_range, n_samples)

        config = {
            'young_age_range': young_age_range,
            'old_age_blend_threshold': old_age_blend_threshold,
            'old_blend_prob': old_blend_prob,
            'young_age_threshold': young_age_threshold,
            'young_extinction_prob': young_extinction_prob,
            'extinction_range': extinction_range,
            'extinction_rv': extinction_rv,
            'eml_prob': eml_prob,
            'normalize_spectra': normalize_spectra
        }

        args_list = [(z, age, metdex, config) for z, age, metdex in zip(z_vals, age_vals, metdex_vals)]

        with mp.Pool(processes=n_processes, initializer=_init_worker,
                     initargs=(self.processor, self.wavelength_grid)) as pool:
            results = pool.map(_generate_single_sample, args_list)

        spectra_list, wavelengths_list, param_list = zip(*results)

        # Photometry (not parallelized)
        photometry, (binned_spectra, _) = self.photometric_calculator.calculate_flux_and_mag(
            spectra_list, wavelengths_list, list(self.photometric_calculator.filter_data.keys())
        )

        output_array = [[[entry[1] for entry in mag], params, spec]
                        for mag, params, spec in zip(photometry, param_list, binned_spectra)]

        output_array = [[[float(v) for v in flux], [float(p[0]), float(p[1]), p[2]], spec]
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

        params[:, 1] /= 1e4  # normalize age

        return integrals, params, spectra



class Augmentor:
    def __init__(self, N_smpl=20, noise_level=0.01):
        self.N_smpl = N_smpl
        self.noise_level = noise_level

    def perturb(self, output_array):
        dataset_rnd = []
        for flux_list, params, lr in output_array:
            for _ in range(self.N_smpl):
                norm_flux = flux_list / np.max(flux_list)
                perturbed = [val + self.noise_level * np.random.normal(0, 1) * val for val in norm_flux]
                dataset_rnd.append([perturbed, params, lr])
        return dataset_rnd
