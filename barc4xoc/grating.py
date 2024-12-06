
"""
This module provides auxiliary grating funcitons
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '04/JUL/2024'
__changed__ = '06/NOV/2024'


from typing import Dict

import numpy as np
from scipy.constants import c, degree, eV, h

from barc4xoc.misc import (
    fft_filtering,
    fit_exponential_decay,
    fit_power_decay,
    moving_average,
)


def align_grating(wavelength: float, line_density: float, order: int = 1,
                  condition: str = "cff", condition_value=lambda l: 0.72 + 1e-9 * l, verbose: int = 0):
    """
    Aligns a grating based on specified conditions.

    Parameters:
    - wavelength (float): Wavelength in meters at which to perform computation.
    - line_density (float): Line density of the grating at its center (line/m). 
    - order (int): Diffraction order on which the grating is aligned.
    - condition (str): Second condition for angle computations ('cff', 'omega', 'alpha', or 'deviation').
    - condition_value (float or callable): Alignment value or law. Can be a float or a function of wavelength.
    - verbose (int): Verbosity level (0: no output, 1: some output, 2: detailed output).

    Returns:
    - Dictionary containing computed values.

    Raises:
    - AssertionError: If the provided condition is unknown or violates angle laws.
    """

    if verbose:
        print("Grating alignment:")

    # Determine the condition function based on condition_value type
    if isinstance(condition_value, float):
        condition_function = lambda l: condition_value
    else:
        condition_function = condition_value

        deviation = np.nan
    alpha = np.nan
    beta = np.nan
    order_align = order
    line_density = line_density

    # Validate the condition input
    assert condition in ["omega", "cff", "alpha", "deviation"], AttributeError("Unknown condition")

    # Compute alignment based on the given condition
    if order_align != 0:
        if condition == "omega":
            omega = condition_function(wavelength)
            alpha = np.arcsin(order_align * line_density * wavelength / (2 * np.sin(omega))) + omega
            beta = np.arccos(np.cos(alpha) + order_align * wavelength * line_density)
            deviation = alpha + beta
        elif condition == "cff":
            if order_align > 0 and condition_function(wavelength) > 1:
                condition_function = lambda l: 1 / condition_function(l)
            cff = condition_function(wavelength)
            nu = order_align * wavelength * line_density
            k = 1 - cff ** 2
            X = (np.sqrt(k ** 2 + nu ** 2 * cff ** 2) - nu) / k
            alpha = np.arccos(X)
            beta = np.arccos(nu + X)
            deviation = alpha + beta
            if verbose > 1:
                print(f"\t k={k}, nu={nu}, alpha={alpha}, beta={beta}")
        elif condition == "deviation":
            deviation = condition_function(wavelength)
            alpha = np.arcsin(wavelength * order_align * line_density / (2 * np.sin(deviation / 2))) + deviation / 2
            beta = deviation - alpha
        elif condition == "alpha":
            alpha = condition_function(wavelength)
            beta = np.arccos(np.cos(alpha) + order_align * wavelength * line_density)
            deviation = alpha + beta
    else:
        if condition == "omega":
            omega = condition_function(wavelength)
            deviation = np.arcsin(1.0 * wavelength * line_density / (2 * np.sin(omega)))
        elif condition == "alpha":
            alpha = condition_function(wavelength)
            deviation = 2 * alpha
        elif condition == "cff":
            raise AttributeError("Zero order angles cannot be computed from a cff condition")
        elif condition == "deviation":
            deviation = condition_function(wavelength)
        alpha = deviation / 2
        beta = deviation / 2

    # Validate angles against grating laws
    try:
        assert np.cos(alpha) - np.cos(beta) + order_align * wavelength * line_density < 1e-6
    except AssertionError:
        raise AssertionError(f"Angles {alpha} ({np.degrees(alpha)} deg), {beta} ({np.degrees(beta)} deg) "
                             f"violate the gratings angle laws")

    # Validate conditions against computed values
    try:
        computed_dict = dict(
            cff=np.sin(beta) / np.sin(alpha),
            deviation=deviation,
            alpha=alpha,
            beta=beta,
            omega=(alpha - beta) / 2
        )
        assert computed_dict[condition] - condition_function(wavelength) < 1e-6
    except AssertionError:
        raise AssertionError(f"Angles {alpha} ({np.degrees(alpha)} deg), {beta} ({np.degrees(beta)} deg) "
                             f"violate the condition equations")

    # Print verbose output if required
    if verbose > 0:
        print(f"\t Wavelength = {wavelength:.4} m,")
        print(f"\t Groove density = {line_density / 1000} mm-1,")
        print(f"\t Alignment order = {order_align} ,")
        print(f"\t Alpha = {np.degrees(alpha):.3f} deg,")
        print(f"\t Beta = {np.degrees(beta):.3f} deg")
        print(f"\t Cff = {np.sin(beta) / np.sin(alpha):.3f}")
        print(f"\t Omega = {np.degrees((alpha - beta) / 2):.3f} deg")
        print(f"\t Deviation = {np.degrees(deviation):.3f} deg")
        print(f"\t Theta grating = {np.degrees(deviation / 2):.3f} deg")

    # Return computed values as a dictionary
    return {
        "wavelength": wavelength,
        "groove_density": line_density,
        "align_order": order_align,
        "alpha_deg": np.degrees(alpha),
        "beta_deg": np.degrees(beta),
        "alpha": alpha,
        "beta": beta,
        "deviation": deviation,
        "deviation_deg": np.degrees(deviation),
        "theta": deviation / 2,
        "omega": (alpha - beta) / 2,
        "omega_deg": np.degrees((alpha - beta) / 2),
        "cff": np.sin(beta) / np.sin(alpha)
    }


def treat_energy_resolution(resolution_dict: Dict[str, np.ndarray], mthd: int = 0, **kwargs) -> Dict:
    """
    Processes energy resolution data in `resolution_dict` by fitting it to various decay models
    and applying optional smoothing and FFT-based filtering. The fit model(s) used are determined
    by the `mthd` parameter.

    Parameters
    ----------
    resolution_dict : Dict
        Dictionary containing energy data (`'energy'` key) and resolution values to be processed.
        Resolution data entries should not include keys with `'fit'`.
    mthd : int, optional
        Specifies the processing method:
        - `0` applies all models: exponential decay (monolog), inverse power law decay (loglog),
          and FFT filtering.
        - `1` applies only exponential decay.
        - `2` applies only inverse power law decay.
        - `3` applies only FFT filtering.
    **kwargs : 
        Additional options for processing and filtering:
        - `treat` : bool, optional
            Whether to apply pre-filtering before fitting (default is True).
        - `avg` : int, optional
            Specifies the method for moving average filtering if `treat` is enabled. Defaults to 0.
        - `deltaE` : float, optional
            Step size for the moving average filter. If None, defaults to an estimate of `10` pixels.
        - `fft_deltaE` : float, optional
            Resolution cutoff for FFT filtering; defaults to 20% of the energy range if not provided.
        - `fft_sigma` : float, optional
            Standard deviation for the Gaussian roll-off in FFT filtering; defaults to `0` for a hard cutoff.

    Returns
    -------
    Dict[str, np.ndarray]
        Updated dictionary with fitted or filtered values added for the specified model(s). Each added key
        has the format `<original_key>_fit_<method>` where `<method>` represents the fitting/filtering approach used.

    Raises
    ------
    ValueError
        If `mthd` is not one of {0, 1, 2, 3}.
    """
    
    if mthd not in {0, 1, 2, 3, 4}:
        raise ValueError("Invalid method. `mthd` must be one of {0, 1, 2, 3, 4}.")
    
    treat = kwargs.get('treat', True)
    fit = kwargs.get('fit', True)
    avg = kwargs.get('avg', 0)

    fitted_results = {}
    x = resolution_dict['energy']
    
    for key in resolution_dict:
        if key != 'energy' and 'fit' not in key:
            y = resolution_dict[key]
            
            deltaE = kwargs.get('deltaE', None)
            if deltaE is None:
                deltaEpx = 10
            else:
                deltaEpx = int(np.ceil(deltaE/(x[1]-x[0])))
                if deltaEpx < 10:
                    deltaEpx = 10

            treated_signal = moving_average(y, deltaEpx, avg)
            fitted_results[f'{key}_fit_moving_avg'] = treated_signal

            if fit is True:
                if mthd in [0, 1]:

                    # print('Expontential decay (monolog)')
                    if treat:
                        fitted_signal = fit_exponential_decay(x, treated_signal)
                    else:
                        fitted_signal = fit_exponential_decay(x, y)
                    fitted_results[f'{key}_fit_monolog'] = fitted_signal

                if mthd in [0, 2]:
                    # print('Inverse power law decay (loglog)')
                    if treat:
                        fitted_signal = fit_power_decay(x, treated_signal)
                    else:
                        fitted_signal = fit_power_decay(x, y)
                    fitted_results[f'{key}_fit_loglog'] = fitted_signal

                if mthd in [0, 3]:
                    deltaE = kwargs.get('fft_deltaE', None)
                    sigma = kwargs.get('fft_sigma', 0)

                    if deltaE is None:
                        deltaE = 0.20*(x[-1]-x[0])

                    if treat:
                        fitted_signal = fft_filtering(x, treated_signal, deltaE, sigma)
                    else:
                        fitted_signal = fft_filtering(x, y, deltaE, sigma)
                    fitted_results[f'{key}_fft_filt'] = fitted_signal

    resolution_dict.update(fitted_results)
    return resolution_dict

