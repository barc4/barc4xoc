
"""
This module provides...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '07/JUL/2024'
__changed__ = '07/JUL/2024'


import numpy as np
from scipy.constants import c, degree, eV, h


def align_grating(verbose: int = 0, condition: str = "cff", condition_value=lambda l: 0.72 + 1e-9 * l,
                   wavelength: float = 1e-9, order: int = 1, line_density: float = 450e3):
    """
    Aligns a grating based on specified conditions.

    Parameters:
    - verbose (int): Verbosity level (0: no output, 1: some output, 2: detailed output).
    - condition (str): Second condition for angle computations ('cff', 'omega', 'alpha', or 'deviation').
    - condition_value (float or callable): Value of the condition. Can be a float or a function of wavelength.
    - wavelength (float): Wavelength in meters at which to perform computation.
    - order (int): Diffraction order on which the grating is aligned.
    - line_density (float): Line density of the grating at its center.

    Returns:
    - None if return_parameters is False.
    - Dictionary containing computed values if return_parameters is True.

    Raises:
    - AssertionError: If the provided condition is unknown or violates angle laws.
    """

    if verbose:
        print("Grating alignment")
    if isinstance(condition_value, float):
        condition_function = lambda l: condition_value
    else:
        condition_function = condition_value

    deviation = np.nan
    alpha = np.nan
    beta = np.nan
    order_align = order
    line_density = line_density

    assert condition in ["omega", "cff", "alpha", "deviation"], AttributeError("Unknown condition")
    if order_align != 0:
        if condition == "omega":  # cas omega constant
            omega = condition_function(wavelength)
            # The next formula is benchmarked against CARPEM :
            alpha = np.arcsin(order_align*line_density*wavelength / (2 * np.sin(omega))) + omega
            beta = np.arccos(np.cos(alpha) + order_align * wavelength * line_density)
            deviation = alpha + beta
        elif condition == "cff":
            if order_align > 0 and condition_function(wavelength) > 1:
                if isinstance(condition_value, float):
                    condition_function = lambda l: 1/condition_value
                else:
                    condition_function = lambda l: 1/condition_value(l)
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
            alpha = np.arcsin(wavelength*order_align*line_density/(2*np.sin(deviation/2)))+deviation/2
            beta = deviation - alpha
        elif condition == "alpha":
            alpha = condition_function(wavelength)
            beta = np.arccos(np.cos(alpha)+order_align*wavelength*line_density)
            deviation = alpha + beta
    else:
        if condition == "omega":
            omega = condition_value(wavelength)
            deviation = np.arcsin(1.0 * wavelength * line_density / (2 * np.sin(omega * degree)))
        elif condition == "alpha":
            alpha = condition_value(wavelength)
            deviation = 2*alpha
        elif condition == "cff":
            raise AttributeError("Zero order angles cannot be computed from a cff condition")
        elif condition == "deviation":
            deviation = condition_value(wavelength)
        alpha = deviation / 2
        beta = deviation / 2

    try:
        assert np.cos(alpha) - np.cos(beta) + order_align*wavelength*line_density < 1e-6
    except AssertionError:
        raise AssertionError(f"Angles {alpha} ({alpha / degree}deg), {beta} ({beta / degree}deg) "
                             f"violate the gratings angle laws")
    try:
        cond_dict = dict(cff=np.sin(beta) / np.sin(alpha), deviation=deviation, alpha=alpha, omega=(alpha - beta) / 2)
        assert cond_dict[condition] - condition_function(wavelength) < 1e-6  # second equation
    except AssertionError:
        raise AssertionError(f"Angles {alpha} ({alpha / degree}deg), {beta} ({beta / degree}deg) violate "
                             f"the condition equations")

    if verbose > 0:
        print(f"\t Wavelength = {wavelength} m,")
        print(f"\t Groove density = {line_density/1000} mm-1,")
        print(f"\t Alignment order = {order_align} ,")
        print(f"\t Alpha = {alpha / degree} deg,")
        print(f"\t Beta = {beta / degree} deg")
        print(f"\t Cff = {np.sin(beta) / np.sin(alpha)}")
        print(f"\t Omega = {(alpha-beta)/2/degree} deg")
        print(f"\t Deviation = {deviation / degree} deg")
        print(f"\t Theta grating = {deviation/2 / degree} deg")

    return dict(
        wavelength=wavelength,
        groove_density=line_density,
        align_order=order_align,
        alpha_deg=alpha / degree,
        beta_deg=beta / degree,
        alpha=alpha,
        beta=beta,
        deviation=deviation,
        theta=deviation / 2,
        omega=(alpha-beta)/2,
        omega_deg=(alpha-beta)/2/degree,
        cff=np.sin(beta) / np.sin(alpha)
    )