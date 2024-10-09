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

def save_beam_data_to_csv(beam, filename):
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
    np.savetxt(filename, data, header=",".join(headers), fmt='%1.6e', delimiter=',', comments='')
    return {header: column for header, column in zip(headers, data)}


def read_pyoptix_beam_from_csv(filename):
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