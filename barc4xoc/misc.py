""" 
Design and other  auxiliary functions
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '09/OCT/2024'
__changed__ = '09/OCT/2024'

import numpy as np
from scipy.constants import degree, physical_constants
from typing import Dict, Optional

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]

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