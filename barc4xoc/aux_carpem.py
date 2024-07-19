
"""
This module provides...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '04/JUL/2024'
__changed__ = '19/JUL/2024'


import glob
import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd

#***********************************************************************************
# IO CARPEM
#***********************************************************************************

def load_carpem_dataset(directory_path: Union[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Loads and processes CARPEM efficiency files from a directory or a list of file paths.

    This function reads multiple CARPEM efficiency files, processes them using the
    `read_carpem_efficiency` function, and stores the resulting pandas DataFrames
    in a dictionary. The dictionary keys are the filenames, and the values are the DataFrames.

    Parameters:
    directory_path (str or list of str): A directory path containing CARPEM efficiency files
                                         or a list of specific file paths.

    Returns:
    dict[str, pd.DataFrame]: A dictionary where keys are filenames and values are pandas DataFrames
                             containing the processed CARPEM data.

    Raises:
    ValueError: If the input is neither a string nor a list of strings.

    """
    
    if isinstance(directory_path, str):
        files_list = glob.glob(directory_path)
        files_list.sort()
    elif isinstance(directory_path, list):
        files_list = directory_path
    else:
        raise ValueError("Input should be a string or a list of strings.")
    
    carpem_dict = {}
    for file in files_list:
        try:
            if os.name == 'nt':
                dict_key = file.split("\\")[-1]
            else:
                dict_key = file.split("/")[-1]

            data = read_carpem_efficiency(file)

            carpem_dict[dict_key] = data
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

    return carpem_dict


def read_carpem_efficiency(file_path: str):
    """
    Reads a CARPEM efficiency file and processes the data into a pandas DataFrame.

    CARPEM is a tool used for simulating the diffraction efficiency of gratings. 
    This function extracts specific columns from the CARPEM output file and dynamically 
    includes additional columns if they exist. The extracted columns are renamed for clarity.

    Parameters:
    file_path (str): The path to the CARPEM efficiency file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the processed data with the following columns:
        - 'alpha': Column 0 (ANGLE INC.)
        - 'energy': Column 1 (ENERGIE)
        - 'wavelength': Column 2 (LONG. Ond. A)
        - 'beta': Column 9 (Ang.So. 1)
        - 'h1': Column 11 (efficiency for Ang.So. 1)
        - 'h2', 'h3', ...: Additional columns with efficiency values for Ang.So. 1, 
                    starting from column 15 and every 6th column thereafter, if they exist.

    Raises:
    ValueError: If the header line starting with '#ANGLE INC.' is not found in the file.

    """
    
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