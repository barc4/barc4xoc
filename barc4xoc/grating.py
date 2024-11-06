
"""
This module provides...
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
from scipy.optimize import curve_fit


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


def fit_energy_resolution(resolution_dict: Dict, mthd: int = 0) -> Dict:
    """
    Fits energy resolution data in `resolution_dict` to either an exponential decay model 
    or an inverse power law decay model based on the selected `mthd` parameter.

    Parameters
    ----------
    resolution_dict : Dict[str, np.ndarray]
        Dictionary containing energy data (`'energy'` key) and resolution values to be fit. 
        Resolution data must not include keys containing `'fit'`.
    mthd : int, optional
        Fit method: `0` applies both exponential decay (monolog) and inverse power law decay (loglog),
        `1` applies only exponential decay, and `2` applies only inverse power law decay.

    Returns
    -------
    Dict[str, np.ndarray]
        Updated dictionary with fitted values added for the specified model(s).

    Raises
    ------
    ValueError
        If `mthd` is not one of {0, 1, 2}.
    """
    
    def exponential_decay(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
        """Negative exponential decay model."""
        return A * np.exp(-B * x) + C

    def power_decay(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
        """Inverse power law decay model."""
        return A * x ** (-B) + C

    if mthd not in {0, 1, 2}:
        raise ValueError("Invalid method. `mthd` must be one of {0, 1, 2}.")

    if mthd in [0, 1]:
        print('Expontential decay (monolog)')
        x = resolution_dict['energy']
        for key in resolution_dict.keys():
            if key != 'energy' and 'fit' not in key:
                y = np.log(resolution_dict[key])
                p = np.polynomial.Polynomial.fit(x, y, 1)
                coefficients = p.convert().coef
                fitted_signal = np.exp(coefficients[0])*np.exp(coefficients[1]*x)
                # popt, pcov = curve_fit(exponential_decay, 
                #                        resolution_dict['energy'], 
                #                        resolution_dict[key], 
                #                        p0=(np.exp(coefficients[0]), -coefficients[1], 0))
                # fitted_signal = exponential_decay(resolution_dict['energy'], *popt)
                resolution_dict[f'{key}_fit_monolog'] = fitted_signal
    
    if mthd in [0, 2]:
        print('Inverse power law decay (loglog)')
        x = np.log(resolution_dict['energy'])
        for key in resolution_dict.keys():
            if key != 'energy' and 'fit' not in key:
                y = np.log(resolution_dict[key])
                p = np.polynomial.Polynomial.fit(x, y, 1)
                coefficients = p.convert().coef
                fitted_signal = np.exp(coefficients[0])*(resolution_dict['energy']**(coefficients[1]))
                # popt, pcov = curve_fit(power_decay, 
                #                        resolution_dict['energy'], 
                #                        resolution_dict[key], 
                #                        p0=(np.exp(coefficients[0]), -coefficients[1], 0))
                # fitted_signal = power_decay(resolution_dict['energy'], *popt)
                resolution_dict[f'{key}_fit_loglog'] = fitted_signal

    return resolution_dict



