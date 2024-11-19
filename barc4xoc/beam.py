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
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew


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


def calculate_fwhm(profile: np.ndarray, bins: int = None, smoothing: float = 0) -> float:
    """
    Calculate the Full Width at Half Maximum (FWHM) of a 1D beam profile using UnivariateSpline.

    Args:
        profile (np.ndarray): Array representing the positions (e.g., X or Y) of the beam.
        bins (int, optional): Number of bins for the histogram. Defaults to None.
        smoothing (float, optional): Smoothing factor for UnivariateSpline. 
                                     A value of 0 means no smoothing, which will 
                                     interpolate exactly through the points. Defaults to 0.
    
    Returns:
        float: The FWHM of the beam profile in the same units as the profile input.
    """
    if bins is None:
        bins = int(np.sqrt(len(profile)))

    counts, edges = np.histogram(profile, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    max_value = max(counts)
    spline = UnivariateSpline(centers, counts - max_value / 2, s=smoothing)
    roots = spline.roots()
    peak_index = np.argmax(counts)
    peak_position = centers[peak_index]
    left_roots = roots[roots < peak_position]
    right_roots = roots[roots > peak_position]
    
    if len(left_roots) > 0 and len(right_roots) > 0:
        fwhm = right_roots[0] - left_roots[-1]
    else:
        fwhm = 0  

    return fwhm
