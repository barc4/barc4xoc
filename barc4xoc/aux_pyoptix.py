""" 
This module interfaces PyOptiX.
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '22/JUL/2024'
__changed__ = '22/JUL/2024'

import numpy as np
from scipy.constants import c, eV, h

hc = h*c/eV

#***********************************************************************************
# R/W functions
#***********************************************************************************

def save_beam_to_csv(beam, filename):
    """
    Save beam data to a CSV file with predefined column headers.
    Convention: X/Xp - horizontal direction
                Y/Yp - vertical direction
                Z/Zp - longitudinal direction/along optical axis
    Parameters:
    - beam : PyOptiX.get_impacts()
        The Shadow Beam object containing the data to be saved.
    - filename : str
        The filename (including path) to save the data.

    Returns:
    - beam : dict
        Dictionary containing the beam data with headers as keys.
    """
    cols = ["Lambda", "Intensity", "X", "Y", "Z", "dX", "dY", "dZ"]
    
    data = []

    for n, hdr in enumerate(cols):
        if n == 0:
            data.append(hc/beam[hdr].values)
        else:
            data.append(beam[hdr].values)
    data = np.asarray(data).T

    # Define column headers
    headers = [
        "energy",
        "intensity",
        "X",
        "Y",
        "Z",
        "Xp",
        "Yp",
        "Zp",
    ]
    
    # Save data to file with headers
    if filename is not None:
        np.savetxt(filename, data, header=",".join(headers), fmt='%1.6e', delimiter=',', comments='')
    return {header: column for header, column in zip(headers, data.T)}


def read_beam_from_csv(filename):
    """
    Read beam data from a CSV file and return as a dictionary with headers as keys.
    Convention: X/Xp - horizontal direction
                Y/Yp - vertical direction
                Z/Zp - longitudinal direction/along optical axis
    Parameters:
    - filename : str
        The filename (including path) of the CSV file containing the data.

    Returns:
    - beam : dict
        Dictionary containing the beam data with headers as keys.
    """
    # Read headers from CSV file
    with open(filename, 'r') as file:
        headers = file.readline().strip().split(',')

    # Read data from CSV file
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Create dictionary with headers as keys and corresponding columns as values
    beam = {header: column for header, column in zip(headers, data.T)}

    return beam


def save_resolution_curve(resolution, filename):
    """
    Save resolution data to a CSV file with predefined column headers.

    Parameters:
    - resolution : dict of {str: np.ndarray}
        A dictionary where each key is a string representing the column header,
        and each value is a 1D numpy array representing the data for that column.
        All arrays must have the same length.
    - filename : str
        The path and name of the file where data will be saved.
    """
    array_length = len(next(iter(resolution.values())))
    for key, array in resolution.items():
        if len(array) != array_length:
            raise ValueError("All arrays must have the same length.")
    data = np.column_stack([resolution[key] for key in resolution])

    # Save with headers
    np.savetxt(filename, data, header=",".join(resolution.keys()), fmt='%1.6e', delimiter=',', comments='')


def read_resolution_curve(filename):
    """
    Read resolution data from a CSV file and return it as a dictionary.

    Parameters:
    - filename : str
        The path to the CSV file to read.

    Returns:
    - resolution : dict of {str: np.ndarray}
        A dictionary where each key is a column header from the CSV,
        and each value is a 1D numpy array with the column's data.
    """
    # Extract headers from the first line
    with open(filename, 'r') as file:
        headers = file.readline().strip().replace("#", "").split(',')

    # Read data from CSV file
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Create dictionary with headers as keys and corresponding columns as values
    resolution = {header: column for header, column in zip(headers, data.T)}

    return resolution