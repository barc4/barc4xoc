
"""
This module provides...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '26/JAN/2024'
__changed__ = '07/JUL/2024'

from copy import copy
from itertools import combinations_with_replacement
from typing import Optional, Tuple, Union

import numpy as np
import xraylib
from xoppylib.scattering_functions.xoppy_calc_f1f2 import xoppy_calc_f1f2

#***********************************************************************************
# reflectivity curves
#***********************************************************************************

def nist_compound_list() -> list:
    """
    Retrieve and returns the list of NIST compounds.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """
    return xraylib.GetCompoundDataNISTList()


def density_element(material:Union[str, int, float], verbose: bool = False) -> float:
    """
    Get the density of an element based on its atomic number or symbol.

    Args:
        material (str | int | float): The atomic symbol (str), atomic number (int), or atomic number (float).
        verbose (bool): If True, prints the density. Defaults to False.

    Returns:
        float: The density of the element.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """
    if isinstance(material, str):
        Z = xraylib.SymbolToAtomicNumber(material)
    elif isinstance(material, int):
        Z = material
    elif isinstance(input, float):
        Z = int(material)
    else:
        raise Exception("Bad input.")
    density = xraylib.ElementDensity(Z)
    if verbose:
        print("Density for %s (Z=%d): %6.3f g/cm3"%(material, Z, density))
    return density


def density_nist(material: Union[str, int], verbose: bool = False) -> float:
    """
    Get the density of a NIST compound based on its name or index.

    Args:
        material (str | int): The compound name (str) or index (int).
        verbose (bool): If True, prints the density. Defaults to False.

    Returns:
        float: The density of the compound.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """

    if isinstance(material, str):
        Zarray = xraylib.GetCompoundDataNISTByName(material)
        density = Zarray["density"]
        if verbose:
            print("Density for %s: %6.3f g/cm3"%(material, density))
    elif isinstance(material, int):
        Zarray = xraylib.GetCompoundDataNISTByIndex(material)
        density = Zarray["density"]
        if verbose:
            print("Density for NIST compound with index %d: %6.3f " % (material, density))

    return density


def get_density(material: str, verbose: bool = False) -> float:
    """
    Determine the density of a material based on its descriptor.

    Args:
        material (str): The descriptor of the material (atomic symbol, compound name, etc.).
        verbose (bool): If True, prints the density. Defaults to False.

    Returns:
        float: The density of the material.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """
        
    if not isinstance(material, str):
        raise Exception("descriptor must be a string!")
    kind = -1
    if len(material) <=2:
        Z = xraylib.SymbolToAtomicNumber(material)
        if Z > 0:
            kind = 0
    elif material in nist_compound_list():
        kind = 2
    else:
        try:
            xraylib.CompoundParser(material)
            kind = 1
        except:
            pass

    if kind == 0:
        return density_element(material, verbose)
    elif kind == 1:
        raise Exception("cannot retrieve density for a compound (%s): it must be defined by user" % material)
    elif kind == 2:
        return density_nist(material, verbose)
    else:
        raise Exception("Unknown descriptor: %s" % material)
        

def reflectivity_curve(material: str, density: float, theta: float, ei: float, ef: float, ne: int,
                       e_axis: Optional[np.ndarray] = None) -> dict:
    """ 
    Calculate the reflectivity for a given material and conditions.

    Args:
        material (str): The material's name.
        density (float): The material's density in grams per cubic centimeter (g/cm^3).
        theta (float): The angle of incidence in milliradians (mrad).
        ei (float): The initial energy in electron volts (eV).
        ef (float): The final energy in electron volts (eV).
        ne (int): The number of energy steps.
        e_axis (Optional[np.ndarray], optional): An array representing the energy axis for point-wise calculation. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "reflectivity" (np.ndarray): The reflectivity values.
            - "energy" (np.ndarray): The corresponding energy values.
            - "theta" (float): The angle of incidence.
            - "material" (str): The material's name. 
    """

    if density == -1:
        density = get_density(material, True)

    if len(material) <=2:
        mat_flag = 0
    else:
        mat_flag = 1

    if e_axis is None:
        out_dict =  xoppy_calc_f1f2(
                descriptor   = material,
                density      = density,
                MAT_FLAG     = mat_flag,
                CALCULATE    = 9,
                GRID         = 1,
                GRIDSTART    = ei,
                GRIDEND      = ef,
                GRIDN        = ne,
                THETAGRID    = 0,
                ROUGH        = 0.0,
                THETA1       = theta,
                THETA2       = 5.0,
                THETAN       = 50,
                DUMP_TO_FILE = 0,
                FILE_NAME    = "%s.dat"%material,
                material_constants_library = xraylib,
            )
        
        energy_axis = out_dict["data"][0,:]
        reflectivity = out_dict["data"][-1,:]
    else:
        energy_axis = np.array([])
        reflectivity = np.array([])
        for E in e_axis:
            out_dict = xoppy_calc_f1f2(
                descriptor=material,
                density=density,
                MAT_FLAG=mat_flag,
                CALCULATE=9,
                GRID=2,
                GRIDSTART=E,
                GRIDEND=E,
                GRIDN=1,
                THETAGRID=0,
                ROUGH=0.0,
                THETA1=theta,
                THETA2=5.0,
                THETAN=50,
                DUMP_TO_FILE=0,
                FILE_NAME="%s.dat" % material,
                material_constants_library=xraylib,
            )
            energy_axis = np.append(energy_axis, out_dict["data"][0, :])
            reflectivity = np.append(reflectivity, out_dict["data"][-1, :])

    return {
        "reflectivity": reflectivity,
        "energy": energy_axis,
        "theta": theta,
        "material": material,
    }


def reflectivity_map(material: str, density: float, thetai: float, thetaf: float,
                     ntheta: int, ei: float, ef: float, ne: int,
                     e_axis: Optional[np.ndarray] = None) -> dict:
    """
    Compute a reflectivity map for a given material over a range of angles and energies.

    Args:
        material (str): The material's name.
        density (float): The material's density in g/cm^3.
        thetai (float): The initial angle of incidence in milliradians (mrad).
        thetaf (float): The final angle of incidence in milliradians (mrad).
        ntheta (int): The number of angles between thetai and thetaf.
        ei (float): The initial energy in electron volts (eV).
        ef (float): The final energy in electron volts (eV).
        ne (int): The number of energy points between ei and ef.
        e_axis (Optional[np.ndarray], optional): An array representing the energy axis. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "reflectivity_map" (np.ndarray): The reflectivity map.
            - "energy" (np.ndarray): The energy axis.
            - "theta" (np.ndarray): The angle axis.
            - "material" (str): The material's name. 
    """

    if density == -1:
        density = get_density(material)
        
    theta = np.linspace(thetai, thetaf, ntheta)

    if e_axis is None:
        reflectivity_map = np.zeros((ntheta, ne))
    else:
        reflectivity_map = np.zeros((ntheta, len(e_axis)))

    for k, th in enumerate(theta):
        result = reflectivity_curve(material, density, th, ei, ef, ne, e_axis)
        reflectivity_map[k, :] = result["reflectivity"]
    return {
        "reflectivity": reflectivity_map,
        "energy": result["energy"],
        "theta": theta,
        "material": material,
    }


def ml_reflectivity_curve():
    #TODO: interface for xoppy and DarpanX
    pass


def ml_reflectivity_map():
    pass


def crystal_reflectivity_curve():
    #TODO:  interface for xoppy and CrystalPy
    pass


def crystal_reflectivity_map():
    pass

#***********************************************************************************
# Mirrors combination
#***********************************************************************************

def combine_mirrors(candidates: Tuple, mirrors: int, **kwargs) -> Tuple:
    """
    Combine reflectivity curves for multiple mirror coatings.

    This function calculates the combined reflectivity of multiple mirror coatings by 
    combining the reflectivity curves for each coating. It uses the `reflectivity_curve` 
    function to calculate the reflectivity for each candidate material and then combines 
    them based on the number of mirrors.

    Args:
        candidates (Tuple): A tuple of candidate materials.
        mirrors (int): The number of mirrors to combine.
        **kwargs: Additional keyword arguments, including:
            - density (array or tuple): The density of each candidate material in g/cm^3.
            - ei (float, optional): The initial energy in eV.
            - ef (float, optional): The final energy in eV.
            - ne (int, optional): The number of energy steps.
            - e_axis (Optional[np.ndarray], optional): An array representing the energy axis 
              for point-wise calculation. Defaults to None.
            - theta (float, optional): The angle of incidence in milliradians (mrad).

    Returns:
        Tuple: A tuple containing:
            - transmission (np.ndarray): The combined reflectivity values.
            - combination (np.ndarray): The corresponding material combinations.
    
    """

    density = kwargs.get('density', None)
    ei = kwargs.get('ei', None)
    ef = kwargs.get('ef', None)
    ne = kwargs.get('ne', None)
    e_axis = kwargs.get('e_axis', None)
    theta = kwargs.get('theta', None)

    if density is not None:
        elements = copy(candidates)
        candidates = []
        if e_axis is None:
            if any(energy_var is None for energy_var in (ei, ef, ne, theta)):
                raise ValueError("Please provide initial and final energies (ei and ef)\
                                 as well as number of energy points (ne) OR the energy\
                                 axis (e_axis) ALL in eV. Check if you provided the glancing\
                                 angle theta in mrad as well.")
            else:
                for dens, element in zip(elements, density):
                    candidates.append(reflectivity_curve(element, dens, theta, ei, ef, ne))
        else:
            for dens, element in zip(elements, density):
                candidates.append(reflectivity_curve(element, dens, theta, e_axis))

    labels = [coating["material"] for coating in candidates]
    reflectivity = [coating["reflectivity"] for coating in candidates]

    transmission = np.hstack([
        np.prod(np.array(combo), axis=0)[:, np.newaxis]
        for combo in combinations_with_replacement(reflectivity, mirrors)
    ])

    # Generate combinations of labels
    combination = np.array([
        tuple(labels[i] for i in combo)
        for combo in combinations_with_replacement(range(len(labels)), mirrors)
    ])

    return transmission, combination