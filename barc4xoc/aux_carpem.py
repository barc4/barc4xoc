
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
import pandas as pd


def read_carpem_efficiency(file_path):
    # Define the initial column names mapping
    col_names = ['alpha', 'energy', 'wavelength', 'beta', 'h1']
    col_positions = [0, 1, 2, 9, 10]
    
    # Read the file and locate the header line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the header line
    header_line = None
    for i, line in enumerate(lines):
        if line.startswith('#ANGLE INC.'):
            header_line = i
            break
    
    if header_line is None:
        raise ValueError("Header line not found.")
    
    # Read a sample line to determine the total number of columns
    sample_line = lines[header_line + 1]
    total_columns = len(sample_line.split())
    
    # Add positions for the columns beyond the initial 12 columns
    additional_cols = []
    for pos in range(16, total_columns, 6):
        if pos < total_columns:
            additional_cols.append(pos)
    
    col_positions.extend(additional_cols)
    
    # Create dynamic column names for 'hn'
    for n, pos in enumerate(additional_cols, start=2):
        col_names.append(f'h{n}')
    
    # Read the data into a pandas DataFrame
    data = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        skiprows=header_line + 1,
        usecols=col_positions,
        names=col_names,
        header=None
    )

    return data