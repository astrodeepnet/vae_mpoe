import numpy as np
from pcigale import sed
from pcigale import sed_modules as modules
from pcigale.warehouse import SedWarehouse
from pcigale.data import SimpleDatabase
from astropy.table import Table


SED_PARAMETERS = {
    'sfhdelayed': {
        'tau_main': 1000.,
        'age_main': 8000.,
        'tau_burst': 2000.,
        'age_burst': 10.0,
        'f_burst': 0.00,
        },
    'bc03': {
        'imf': 0,
        'metallicity': 0.02,
    },
    'nebular': {
        'logU': -2.0,
        'f_esc': 0.0,
        'f_dust': 0.0,
        'emission': False,
        'line_list': """
            ArIII-713.6 &
            CII-232.4 &
            CII-232.47 &
            CII-232.54 &
            CII-232.7 &
            CII-232.8 &
            CIII-190.7 &
            CIII-190.9 &
            H-alpha &
            H-beta &
            H-delta &
            H-gamma &
            HeII-164.0 &
            Ly-alpha &
            NII-654.8 &
            NII-658.3 &
            NeIII-396.7 &
            OI-630.0 &
            OII-372.6 &
            OII-372.9 &
            OIII-495.9 &
            OIII-500.7 &
            Pa-alpha &
            Pa-beta &
            Pa-gamma &
            SII-671.6 &
            SII-673.1 &
            SIII-906.9 &
            SIII-953.1
        """,
    },
    'dustatt_modified_starburst': {
        'E_BV_lines': 0.00,
        'E_BV_factor': 1,
        'uv_bump_wavelength': 217.5,
        'uv_bump_width': 35.0,
        'uv_bump_amplitude': 0.0,
        'powerlaw_slope': 0.0,
        'Ext_law_emission_lines': 1,
        'Rv': 3.1,
    },
    'dl2014': {
        'qpah': 2.50,
        'umin': 1.5,
        'alpha': 2.0,
        'gamma': 0.02,
    },
    'redshifting': {
        'redshift': 0,
    }
}

print(SED_PARAMETERS['sfhdelayed'])

WAREHOUSE = SedWarehouse()
module_list = ['sfhdelayed', 'bc03', 'nebular', 'dustatt_modified_starburst', 'dl2014', 'redshifting']


age_tbl = []
alpha_tbl = []
met_tbl = []
wl_tbl = []
flx_tbl = []

alphas = [0.1, 0.5, 1.0, 5.0, 10]
ages = 10 ** np.linspace(np.log10(60), np.log10(13000))
mets = [0.004, 0.008, 0.02, 0.05]

for age in ages:
    for alpha in alphas:
        for met in mets:
            print(age, alpha, met)
            SED_PARAMETERS['sfhdelayed']['age_main'] = age
            SED_PARAMETERS['sfhdelayed']['tau_main'] = age * alpha
            SED_PARAMETERS['bc03']['metallicity'] = met
            
            sed = WAREHOUSE.get_sed(
                module_list = module_list,
                parameter_list = [SED_PARAMETERS[key] for key in module_list])
            age_tbl.append(age)
            alpha_tbl.append(alpha)
            met_tbl.append(met)
            wl_tbl.append(sed.wavelength_grid)
            flx_tbl.append(sed.fnu)


t = Table([age_tbl, alpha_tbl, met_tbl, wl_tbl, flx_tbl], names=('age', 'alpha', 'met', 'wl', 'flx'))
t.write('exp_cigale_noeml.fits', overwrite=True)
