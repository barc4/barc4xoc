
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