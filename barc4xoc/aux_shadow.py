
"""
This module provides...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '04/JUL/2024'
__changed__ = '19/JUL/2024'


import numpy as np


def save_beam_data_to_csv(beam, filename):
    """
    Save beam data to a CSV file with predefined column headers.

    Parameters:
    - beam : ShadowLib.Beam
        The Shadow Beam object containing the data to be saved.
    - filename : str
        The filename (including path) to save the data.

    Returns:
    None
    """
    # Fixed columns list
    cols = [11, 23, 24, 25, 1, 2, 3, 4, 5, 6]
    
    # Extract data from beam object
    data = np.asarray(beam._beam.getshcol(cols)).T
    
    # Define column headers
    headers = [
        "energy",
        "total_intensity",
        "total_intensity_s-pol",
        "total_intensity_p-pol",
        "X",
        "Y",
        "Z",
        "Xp",
        "Yp",
        "Zp",
    ]
    
    # Save data to file with headers
    np.savetxt(filename, data, header=",".join(headers), fmt='%1.6e', delimiter=',', comments='')


def read_shadow_beam_from_csv(filename):
    """
    Read beam data from a CSV file and return as a dictionary with headers as keys.

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