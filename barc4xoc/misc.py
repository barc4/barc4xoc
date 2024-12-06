""" 
Design and other auxiliary functions
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '09/OCT/2024'
__changed__ = '06/DEC/2024'

import warnings
from typing import Dict, Optional

import numpy as np
from scipy.constants import degree, physical_constants
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]

#***********************************************************************************
# optics
#***********************************************************************************

def coddington_equations(theta: float, **kwargs) -> Dict[str, list]:
    """
    Calculate the Coddington shape parameters (radii and focal lengths) for a mirror system based on input parameters.

    Parameters:
    - theta: float, grazing incident angle in degrees.
    - Rm: float, optional, sagittal radius of curvature (horizontal focusing direction for vertical deflecting mirror).
    - Rs: float, optional, meridional (tangential) radius of curvature (vertical focusing direction for vertical deflecting mirror).
    - fm: float, optional, sagittal focal length.
    - fs: float, optional, meridional focal length.
    - p: float, optional, object distance from the mirror.
    - q: float, optional, image distance from the mirror.

    Returns:
    - A dictionary with two keys:
      - 'theta': float, the grazing incident angle in degrees.
      - 'R': list, containing Rm (sagittal radius) and Rs (meridional radius).
      - 'f': list, containing fm (sagittal focal length) and fs (meridional focal length).

    Raises:
    - Exception: if insufficient input parameters are provided to perform the calculation.

    Notes:
    - To perform calculations, at least one of the following parameter pairs must be provided:
      (Rm, Rs), (fm, fs), or (p, q).
    - The parameter 'theta' must always be provided.
    """

    Rm = kwargs.get("Rm", None)   
    Rs = kwargs.get("Rs", None)    

    fm = kwargs.get("fm", None)
    fs = kwargs.get("fs", None)

    p = kwargs.get("p", None)
    q = kwargs.get("q", None)

    err_msg = "Please, provide either (Rm, Rs), (fm, fs) or (p, q) for calculation"
    
    if None not in [q, p]:

        pq = 2*(p*q)/(p+q)

        Rm = pq/np.sin(theta*degree)
        Rs = pq*np.sin(theta*degree)

        fm = Rm*np.sin(theta*degree)/2
        fs = Rs/np.sin(theta*degree)/2

    elif None not in [fm, fs]:
        Rm = 2*fm/np.sin(theta*degree)
        Rs = 2*fs*np.sin(theta*degree)

    elif None not in [Rm, Rs]:
        fm = Rm*np.sin(theta*degree)/2
        fs = Rs/np.sin(theta*degree)/2

    else:
        raise Exception(err_msg)

    return {"theta": theta, "R":[Rm, Rs], "f":[fm, fs]}

def lensmaker_equation():
    pass

#***********************************************************************************
# energy/wavelength conversion
#***********************************************************************************
     
def energy_wavelength(value: float, unity: str) -> float:
    """
    Converts energy to wavelength and vice versa.
    
    Parameters:
        value (float): The value of either energy or wavelength.
        unity (str): The unit of 'value'. Can be 'eV', 'meV', 'keV', 'm', 'nm', or 'A'. Case sensitive. 
        
    Returns:
        float: Converted value in meters if the input is energy, or in eV if the input is wavelength.
        
    Raises:
        ValueError: If an invalid unit is provided.
    """
    factor = 1.0
    
    # Determine the scaling factor based on the input unit
    if unity.endswith('eV') or unity.endswith('meV') or unity.endswith('keV'):
        prefix = unity[:-2]
        if prefix == "m":
            factor = 1e-3
        elif prefix == "k":
            factor = 1e3
    elif unity.endswith('m'):
        prefix = unity[:-1]
        if prefix == "n":
            factor = 1e-9
    elif unity.endswith('A'):
        factor = 1e-10
    else:
        raise ValueError("Invalid unit provided: {}".format(unity))

    return PLANCK * LIGHT / CHARGE / (value * factor)

#***********************************************************************************
# data treatment / maths
#***********************************************************************************

def fit_exponential_decay(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fits data to an exponential decay model: f(x) = A * np.exp(B * x)
    
    Parameters
    ----------
    x : np.ndarray
        The independent variable (energy).
    y : np.ndarray
        The dependent variable (resolution data).
    
    Returns
    -------
    np.ndarray
        Fitted data based on the exponential decay model.
    """
    def exponential_decay(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
        """Negative exponential decay model."""
        return A * np.exp(B * x) + C
    fx = x
    fy = np.log(y)
    p = np.polynomial.Polynomial.fit(fx, fy, 1)
    coefficients = p.convert().coef

    try:
        popt, pcov = curve_fit(exponential_decay, x, y, 
                            p0=(np.exp(coefficients[0]), coefficients[1], 0))
        fitted_signal = exponential_decay(x, *popt)
    except:
        warnings.warn("in fit_exponential_decay: scipy.optimize.curve_fit did not converge! Falling back to a simpler model", UserWarning)
        fitted_signal = np.exp(coefficients[0])*np.exp(coefficients[1]*x)

    return fitted_signal


def fit_power_decay(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fits data to an inverse power law decay model: A * x ** (B)
    
    Parameters
    ----------
    x : np.ndarray
        The independent variable (energy).
    y : np.ndarray
        The dependent variable (resolution data).
    
    Returns
    -------
    np.ndarray
        Fitted data based on the inverse power law decay model.
    """
    def power_decay(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
        """Inverse power law decay model."""
        return A * x ** (B) + C

    fx = np.log(x)
    fy = np.log(y)
    p = np.polynomial.Polynomial.fit(fx, fy, 1)
    coefficients = p.convert().coef

    try:
        popt, pcov = curve_fit(power_decay, x, y, 
                            p0=(np.exp(coefficients[0]), coefficients[1], 0))
        fitted_signal = power_decay(x, *popt)
    except:
        warnings.warn("in fit_power_decay: scipy.optimize.curve_fit did not converge! Falling back to a simpler model", UserWarning)
        fitted_signal = np.exp(coefficients[0]) * (x ** coefficients[1])

    return fitted_signal


def moving_average(y: np.ndarray, window_size: int, mthd: int = 0) -> np.ndarray:
    """
    Applies a moving average filter to smooth data.

    Parameters
    ----------
    y : np.ndarray
        The data array to be smoothed.
    window_size : int
        The size of the moving window.
    mthd : int, optional
        The method for averaging. If 0, a simple moving average is applied.
        If other, a more advanced method (e.g., uniform filter) is used.
    
    Returns
    -------
    np.ndarray
        Smoothed data array after applying the moving average filter.
    """
    if mthd == 0:
        pad_width = window_size // 2
        padded_y = np.pad(y, pad_width, mode='edge')
        cumsum_vec = np.cumsum(np.insert(padded_y, 0, 0)) 
        smoothed = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        return smoothed[:len(y)]
    else:
        uniform_filter1d(y, size=window_size, mode='nearest')


def fft_filtering(x: np.ndarray, y: np.ndarray, delta_x: float, sigma: float = 0) -> np.ndarray:
    """
    Applies an FFT-based low-pass filter to the data.

    Parameters
    ----------
    x : np.ndarray
        The independent variable array, typically representing energy.
    y : np.ndarray
        The dependent variable array, typically representing signal intensity.
    delta_x : float
        The spatial or energy resolution of the filter's cutoff frequency.
    sigma : float, optional
        Standard deviation for Gaussian roll-off near the cutoff frequency. 
        Defaults to 0 for a hard cutoff.

    Returns
    -------
    np.ndarray
        The filtered signal, transformed back to the original domain.
    """
    n = len(x)
    energy_spacing = np.mean(np.diff(x))
    freq = np.fft.fftfreq(n, d=energy_spacing)
    fft_vals = np.fft.fft(y)
    cutoff_freq = 1 / delta_x

    if sigma > 0:
        # Gaussian roll-off near the cutoff frequency
        gaussian_filter = np.exp(-0.5 * ((freq / cutoff_freq) ** 2) / sigma**2)
    else:
        # Hard cutoff (pillbox function)
        gaussian_filter = np.where(np.abs(freq) <= cutoff_freq, 1, 0)

    fft_vals *= gaussian_filter
    filtered_signal = np.fft.ifft(fft_vals).real
    
    return filtered_signal


def calculate_fwhm(profile: np.ndarray, bins: int = None, smoothing: float = 0) -> float:
    """
    Calculate the Full Width at Half Maximum (FWHM) of a 1D beam profile using UnivariateSpline.

    Args:
        profile (np.ndarray): Array representing the positions (e.g., X or Y) of the beam.
        bins (int, optional): Number of bins for the histogram. Defaults to None.
        smoothing (float, optional): Smoothing factor for UnivariateSpline. 
                                     A value of 0 means no smoothing, which will 
                                     interpolate exactly through the points. Defaults to 0.
    
    Returns:
        float: The FWHM of the beam profile in the same units as the profile input.
    """
    try:
        if bins is None:
            bins = int(np.sqrt(len(profile)))

        counts, edges = np.histogram(profile, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        max_value = max(counts)
        spline = UnivariateSpline(centers, counts - max_value / 2, s=smoothing)
        roots = spline.roots()
        peak_index = np.argmax(counts)
        peak_position = centers[peak_index]
        left_roots = roots[roots < peak_position]
        right_roots = roots[roots > peak_position]
        
        if len(left_roots) > 0 and len(right_roots) > 0:
            fwhm = right_roots[0] - left_roots[-1]
        else:
            fwhm = 0  
    except:
        fwhm = 0 

    return fwhm


def calc_averaged_psd(psd_2d, fx, fy):
    
    def _f_xy(_theta, _rho):
        x = _rho * np.cos(_theta)
        y = _rho * np.sin(_theta)
        return x, y

    xStart, xFin, nx = fx[0], fx[-1], fx.size
    yStart, yFin, ny = fy[0], fy[-1], fy.size
    x_cen = 0.5 * (xFin + xStart)
    y_cen = 0.5 * (yFin + yStart)

    range_r = [0, min(xFin - x_cen, yFin - y_cen)]
    range_theta = [0, 2 * np.pi]

    nr = int(nx * 1/2)
    ntheta = int((range_theta[1] - range_theta[0]) * 360 * 1/2/np.pi)

    X = np.linspace(xStart, xFin, nx)
    Y = np.linspace(yStart, yFin, ny)

    R = np.linspace(range_r[0], range_r[1], nr)
    THETA = np.linspace(range_theta[0], range_theta[1], ntheta)

    R_grid, THETA_grid = np.meshgrid(R, THETA, indexing='ij')
    X_grid, Y_grid = _f_xy(THETA_grid, R_grid)  # Convert polar to cartesian coordinates

    interp_func = RegularGridInterpolator((Y, X), psd_2d, bounds_error=False, fill_value=0)

    points = np.column_stack([Y_grid.ravel(), X_grid.ravel()])
    values = interp_func(points).reshape(R_grid.shape)

    psd_avg = np.mean(values, axis=1)

    return psd_avg, R