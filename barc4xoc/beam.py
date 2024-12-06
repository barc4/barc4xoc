""" 
Ray tracing auxiliary functions
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '09/OCT/2024'
__changed__ = '05/NOV/2024'

from typing import Dict, List

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew

from barc4xoc.misc import calculate_fwhm


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


def merge_beams(beams: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Merge multiple photon beam dictionaries into a single combined beam.

    This function takes a list of beam dictionaries, each containing various properties 
    (e.g., 'energy', 'intensity', 'X', 'Y', etc.) as numpy arrays, and merges them by 
    concatenating the arrays for each key across all beams.

    Args:
        beams (List[Dict[str, np.ndarray]]): A list of dictionaries, where each dictionary 
            represents a beam and contains keys:
            - 'energy': Photon energy values.
            - 'intensity': Beam intensity values.
            - 'X', 'Y', 'Z': Spatial coordinates of the beam.
            - 'Xp', 'Yp', 'Zp': Angular or directional components of the beam.

    Returns:
        Dict[str, np.ndarray]: A dictionary with the same keys as individual beams, where 
        each key contains a concatenated numpy array of the respective values from all beams.
    """
    merged_beam = {
        'energy': np.array([]),
        'intensity': np.array([]),
        'X': np.array([]),
        'Y': np.array([]),
        'Z': np.array([]),
        'Xp': np.array([]),
        'Yp': np.array([]),
        'Zp': np.array([]),
    }
    
    for beam in beams:
        for key in merged_beam.keys():
            merged_beam[key] = np.append(merged_beam[key], beam[key])

    return merged_beam


def get_beam_stats(beam: dict, direction: str = "both", verbose: bool = False) -> Dict:
    """
    Calculate beam statistics, including RMS beam size, divergence, focusing distance, 
    skewness, and kurtosis along specified axes for a photon beam.

    Args:
        beam (dict): A dictionary containing beam data with keys:
            - "X" (np.ndarray): Array of horizontal positions.
            - "Xp" (np.ndarray): Array of horizontal angles/divergence.
            - "Y" (np.ndarray): Array of vertical positions.
            - "Yp" (np.ndarray): Array of vertical angles/divergence.
        direction (str, optional): Direction for calculating statistics; 
            "x" for horizontal, "y" for vertical, or "both" for both axes. 
            Defaults to "both".
        verbose (bool, optional): If True, prints detailed statistics for each direction.

    Returns:
        Dict: A dictionary containing the statistics for each axis requested.
    """
    
    def objective_function(distance, spots, axis):
        return (spots[axis.capitalize()] + distance * spots[axis.capitalize()+"p"]).std()

    stats = {'X': {}, 'Y': {}}

    if direction in ['x', 'both']:
        result_x = minimize(objective_function, 0, args=(beam, 'x'))
        stats['X']['focus_distance'] = result_x.x[0]
        
        # Calculate statistics for X and Xp
        stats['X']['sigma'] = {
            'mean': np.mean(beam["X"]),
            'std': np.std(beam["X"]),
            'fwhm': calculate_fwhm(beam["X"]),
            'skew': skew(beam["X"]),
            'kurtosis': kurtosis(beam["X"])
        }
        stats['X']['sigma_p'] = {
            'mean': np.mean(beam["Xp"]),
            'std': np.std(beam["Xp"]),
            'fwhm': calculate_fwhm(beam["Xp"]),
            'skew': skew(beam["Xp"]),
            'kurtosis': kurtosis(beam["Xp"])
        }

        if verbose:
            print("\n------------------ Horizontal plane:")
            print(f"> Beam focusing along X at {result_x.x[0]:.3f} m")
            print(f">> RMS beam size: {stats['X']['sigma']['std'] * 1E6:.1f} µm")
            print(f">> Divergence: {stats['X']['sigma_p']['std'] * 1E6:.1f} µrad")
            print(f">> FWHM: {stats['X']['sigma']['fwhm'] * 1E6:.1f} µm")
            print(f">> Skewness: {stats['X']['sigma']['skew']:.3f}")
            print(f">> Kurtosis: {stats['X']['sigma']['kurtosis']:.3f}")

    if direction in ['y', 'both']:
        result_y = minimize(objective_function, 0, args=(beam, 'y'))
        stats['Y']['focus_distance'] = result_y.x[0]
        
        # Calculate statistics for Y and Yp
        stats['Y']['sigma'] = {
            'mean': np.mean(beam["Y"]),
            'std': np.std(beam["Y"]),
            'fwhm': calculate_fwhm(beam["Y"]),
            'skew': skew(beam["Y"]),
            'kurtosis': kurtosis(beam["Y"])
        }
        stats['Y']['sigma_p'] = {
            'mean': np.mean(beam["Yp"]),
            'std': np.std(beam["Yp"]),
            'fwhm': calculate_fwhm(beam["Yp"]),
            'skew': skew(beam["Yp"]),
            'kurtosis': kurtosis(beam["Yp"])
        }

        if verbose:
            print("\n------------------ Vertical plane:")
            print(f">> Beam focusing along Y at {result_y.x[0]:.3f} m")
            print(f">> RMS beam size: {stats['Y']['sigma']['std'] * 1E6:.1f} µm")
            print(f">> Divergence: {stats['Y']['sigma_p']['std'] * 1E6:.1f} µrad")
            print(f">> FWHM: {stats['Y']['sigma']['fwhm'] * 1E6:.1f} µm")
            print(f">> Skewness: {stats['Y']['sigma']['skew']:.3f}")
            print(f">> Kurtosis: {stats['Y']['sigma']['kurtosis']:.3f}")

    return stats

def initialise_beam_stats_dict(npts, group_name):

    stats = {
        f'{group_name}':
        {'X': 
         {'focus_distance': np.zeros(npts),
          'sigma': {'mean': np.zeros(npts),
                    'std': np.zeros(npts),
                    'fwhm': np.zeros(npts),
                    'skew': np.zeros(npts),
                    'kurtosis':np.zeros(npts)
                    },
          'sigma_p': {'mean': np.zeros(npts),
                      'std': np.zeros(npts),
                      'fwhm': np.zeros(npts),
                      'skew': np.zeros(npts),
                      'kurtosis': np.zeros(npts)
                     }
         },
        'Y': {'focus_distance': np.zeros(npts),
              'sigma': {'mean': np.zeros(npts),
                        'std': np.zeros(npts),
                        'fwhm': np.zeros(npts),
                        'skew': np.zeros(npts),
                        'kurtosis': np.zeros(npts)
                        },
              'sigma_p': {'mean': np.zeros(npts),
                          'std': np.zeros(npts),
                          'fwhm': np.zeros(npts),
                          'skew': np.zeros(npts),
                          'kurtosis': np.zeros(npts)
                          }
             }
        }
    }

    return stats

def save_beam_stats_to_csv(stats: Dict, scan: np.ndarray, filename: str, **kwargs) -> Dict:
    """
    Load and structure data from a CSV file into a nested dictionary format.

    Args:
        stats (Dict): A dictionary containing the statistics for a ray tracing beam.
        scan (np.ndarray): Array of scan values to include as a column.
        filename (str): Path to the CSV file containing the structured statistics data.
    """

    headers = [
        'config', 'scan',
        'X_f', 'X_mean', 'X_std', 'X_fwhm', 'X_skew', 'X_kurtosis', 
        'Xp_mean', 'Xp_std', 'Xp_fwhm', 'Xp_skew', 'Xp_kurtosis', 
        'Y_f', 'Y_mean', 'Y_std', 'Y_fwhm', 'Y_skew', 'Y_kurtosis', 
        'Yp_mean', 'Yp_std', 'Yp_fwhm', 'Yp_skew', 'Yp_kurtosis'
    ]
    

    data_rows = []
    for config, coord_data in stats.items():
        focus_distance_X = coord_data['X']['focus_distance']
        sigma_X = coord_data['X']['sigma']
        sigma_p_X = coord_data['X']['sigma_p']
        
        focus_distance_Y = coord_data['Y']['focus_distance']
        sigma_Y = coord_data['Y']['sigma']
        sigma_p_Y = coord_data['Y']['sigma_p']
        
        for i in range(len(focus_distance_X)):
            row = [
                config, scan[i],
                focus_distance_X[i], sigma_X['mean'][i], sigma_X['std'][i], sigma_X['fwhm'][i], sigma_X['skew'][i], sigma_X['kurtosis'][i],
                sigma_p_X['mean'][i], sigma_p_X['std'][i], sigma_p_X['fwhm'][i], sigma_p_X['skew'][i], sigma_p_X['kurtosis'][i],
                focus_distance_Y[i], sigma_Y['mean'][i], sigma_Y['std'][i], sigma_Y['fwhm'][i], sigma_Y['skew'][i], sigma_Y['kurtosis'][i],
                sigma_p_Y['mean'][i], sigma_p_Y['std'][i], sigma_p_Y['fwhm'][i], sigma_p_Y['skew'][i], sigma_p_Y['kurtosis'][i]
            ]
            data_rows.append(row)
    
    data = np.array(data_rows, dtype=object)
    
    np.savetxt(
        filename, 
        data, 
        header=",".join(headers), 
        fmt='%s',
        delimiter=',', 
        comments=''
    )
    
    print(f"Data saved to {filename}")


def load_stats_from_csv(filename: str) -> Dict:
    """
    Load and structure data from a CSV file into a nested dictionary format with energy values.

    Args:
        filename (str): Path to the CSV file containing the structured statistics data.

    Returns:
        Dict: Dictionary containing the structured statistics data with energy values.
    """
    stats = {}

    data = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None, names=True)

    configs = np.unique(data['config'])

    for config in configs:
        stats[config] = {
            'scan': [],  # Store energy for each config
            'X': {
                'focus_distance': [],
                'sigma': {'mean': [], 'std': [], 'fwhm': [], 'skew': [], 'kurtosis': []},
                'sigma_p': {'mean': [], 'std': [], 'fwhm': [], 'skew': [], 'kurtosis': []}
            },
            'Y': {
                'focus_distance': [],
                'sigma': {'mean': [], 'std': [], 'fwhm': [], 'skew': [], 'kurtosis': []},
                'sigma_p': {'mean': [], 'std': [], 'fwhm': [], 'skew': [], 'kurtosis': []}
            }
        }

        config_data = data[data['config'] == config]

        for row in config_data:
            stats[config]['scan'].append(float(row['scan']))
            stats[config]['X']['focus_distance'].append(float(row['X_f']))
            stats[config]['X']['sigma']['mean'].append(float(row['X_mean']))
            stats[config]['X']['sigma']['std'].append(float(row['X_std']))
            stats[config]['X']['sigma']['fwhm'].append(float(row['X_fwhm']))
            stats[config]['X']['sigma']['skew'].append(float(row['X_skew']))
            stats[config]['X']['sigma']['kurtosis'].append(float(row['X_kurtosis']))
            stats[config]['X']['sigma_p']['mean'].append(float(row['Xp_mean']))
            stats[config]['X']['sigma_p']['std'].append(float(row['Xp_std']))
            stats[config]['X']['sigma_p']['fwhm'].append(float(row['Xp_fwhm']))
            stats[config]['X']['sigma_p']['skew'].append(float(row['Xp_skew']))
            stats[config]['X']['sigma_p']['kurtosis'].append(float(row['Xp_kurtosis']))
            
            stats[config]['Y']['focus_distance'].append(float(row['Y_f']))
            stats[config]['Y']['sigma']['mean'].append(float(row['Y_mean']))
            stats[config]['Y']['sigma']['std'].append(float(row['Y_std']))
            stats[config]['Y']['sigma']['fwhm'].append(float(row['Y_fwhm']))
            stats[config]['Y']['sigma']['skew'].append(float(row['Y_skew']))
            stats[config]['Y']['sigma']['kurtosis'].append(float(row['Y_kurtosis']))
            stats[config]['Y']['sigma_p']['mean'].append(float(row['Yp_mean']))
            stats[config]['Y']['sigma_p']['std'].append(float(row['Yp_std']))
            stats[config]['Y']['sigma_p']['fwhm'].append(float(row['Yp_fwhm']))
            stats[config]['Y']['sigma_p']['skew'].append(float(row['Yp_skew']))
            stats[config]['Y']['sigma_p']['kurtosis'].append(float(row['Yp_kurtosis']))

        for axis in ['X', 'Y']:
            stats[config][axis]['focus_distance'] = np.array(stats[config][axis]['focus_distance'])
            for stat in ['mean', 'std', 'fwhm', 'skew', 'kurtosis']:
                stats[config][axis]['sigma'][stat] = np.array(stats[config][axis]['sigma'][stat])
                stats[config][axis]['sigma_p'][stat] = np.array(stats[config][axis]['sigma_p'][stat])

        stats[config]['scan'] = np.array(stats[config]['scan'])

    return stats