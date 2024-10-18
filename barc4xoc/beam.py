""" 
Ray tracing auxiliary functions
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '09/OCT/2024'
__changed__ = '09/OCT/2024'

import numpy as np


def read_beam_from_csv(filename):
    """
    Read beam data from a CSV file and return as a dictionary with headers as keys. See
    barc4xoc.aux_pyoptix or barc4xoc.aux_shadow for definition of csv beam file.

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