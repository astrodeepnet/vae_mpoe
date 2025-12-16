from calc_kcor import calc_kcor
from astropy.cosmology import Planck18 as cosmo
import numpy as np

def rs_col_z(app_mag, mag_band, color_band, redshift, tol=1e-5, max_iter=30):
    """
    Returns observed red sequence color given apparent magnitude in any band.

    Parameters
    ----------
    app_mag : float
        Apparent magnitude in the specified band.
    mag_band : str
        Band in which the magnitude is given, e.g., 'u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'K'.
    color_band : str
        Desired color, e.g., 'u-r', 'g-r', 'g-i', 'g-z', etc.
    redshift : float
        Redshift of the galaxy.

    Returns
    -------
    float
        Observed color of the red sequence at the given redshift.
    """

    # Рассчитываем люминесцентное расстояние в парсеках
    D_L = cosmo.luminosity_distance(redshift).to('pc').value
    distance_modulus = 5 * np.log10(D_L / 10.0)
    band1, band2 = color_band.replace(' ', '').split('-')

    # Переводим наблюдаемую величину в абсолютную
    abs_mag = app_mag - distance_modulus

    # Словарь соответствия цвета и полосы для M20
    color_to_band = {
        'u - r': 'r',
        'u - i': 'i',
        'u - z': 'z',
        'g - r': 'r',
        'g - i': 'i',
        'g - z': 'z',
        'g - Y': 'Y',
        'g - J': 'J',
        'g - H': 'H',
        'g - K': 'K'
    }

    if color_band not in color_to_band:
        raise ValueError(f"Color band '{color_band}' not recognized.")

    target_band = color_to_band[color_band]

    # Сдвиг M + 20
    M20 = abs_mag + 20.0

    # Полиномы красной последовательности
    relations = {
        'u - r': lambda M20: 2.51 - 0.065*M20 - 0.005*M20**2,
        'u - i': lambda M20: 2.90 - 0.069*M20 - 0.007*M20**2,
        'u - z': lambda M20: 3.15 - 0.050*M20 - 0.014*M20**2,
        'g - r': lambda M20: 0.75 - 0.026*M20 - 0.001*M20**2,
        'g - i': lambda M20: 1.12 - 0.038*M20 - 0.003*M20**2,
        'g - z': lambda M20: 1.39 - 0.044*M20 - 0.009*M20**2,
        'g - Y': lambda M20: 1.91 - 0.067*M20 - 0.018*M20**2,
        'g - J': lambda M20: 2.01 - 0.073*M20 - 0.016*M20**2,
        'g - H': lambda M20: 2.30 - 0.094*M20 - 0.015*M20**2,
        'g - K': lambda M20: 2.00 - 0.108*M20 - 0.018*M20**2,
    }

    obs_color = relations[color_band](M20)

    for i in range(max_iter):
        # Вычисляем K-коррекции для обеих полос
        k1 = calc_kcor(band1, redshift, color_band, obs_color)
        k2 = calc_kcor(band2, redshift, color_band, obs_color)
        
        # Коррекция абсолютных величин
        M1_corr = app_mag - distance_modulus - k1
        M2_corr = M1_corr - obs_color - k2
        
        # Обновляем M20
        M20_1 = M1_corr + 20.0
        M20_2 = M2_corr + 20.0
        
        # Новое значение цвета по красной последовательности
        new_color = relations[color_band](M20_2) + k1 - k2
        
        # Проверяем сходимость
        if np.abs(new_color - obs_color) < tol:
            obs_color = new_color
            break
        
        obs_color = new_color
        
    return obs_color