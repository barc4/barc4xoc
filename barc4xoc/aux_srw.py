""" 
This module interfaces SRW.
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '22/JUL/2024'
__changed__ = '01/AUG/2024'

from array import array
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import oasys_srw.srwlib as srwlib
from oasys_srw.uti_plot import *
from scipy.ndimage import zoom
# from barc4plots import PlotManager

#***********************************************************************************
# R/W functions
#***********************************************************************************


def read_srw_intensity_dat(file_name, bandwidth=1e-3, transmission=1, norm=False):

    image, mesh = srwl_uti_read_intens_ascii(file_name)
    image = np.reshape(image, (mesh.ny, mesh.nx))
    dx = (mesh.xFin - mesh.xStart)/mesh.nx * 1E3
    dy = (mesh.yFin - mesh.yStart)/mesh.ny * 1E3

    image = image*dx*dy*bandwidth*transmission/(1e-3)
    n = 1
    if norm is not False:
        if norm is True:
            image /= np.amax(image)
            n = np.amax(image)
        else:
            image /= norm
            n = norm

    x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx)
    y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny)

    return image, x, y, mesh, n


def read_srw_phase_dat(file_name, unwrap=False):

    image, mesh = srwl_uti_read_intens_ascii(file_name)
    image = np.reshape(image, (mesh.ny, mesh.nx))

    if unwrap:
        image = unwrap_phase(image, wrap_around=False)

    x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx)
    y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny)

    return image, x, y, mesh

#***********************************************************************************
# SRW auxiliary calculations
#***********************************************************************************

def orient_mirror(incident_angle:float, orientation: str, **kwargs) -> Dict:
    """
    Determine the orientation of a mirror based on the incident angle and specified orientation.

    Parameters:
    - incident_angle: float, grazing incident angle in radians.
    - orientation: str, desired orientation of the mirror ('up', 'down', 'left', 'right').
    - invert_tangential_component: bool, optional, default is False.
      Invert the tangential component if set to True.

    Returns:
    - mirror_orientation: dict, a dictionary with the components of the mirror's normal and tangential vectors:
      - 'nvx', 'nvy', 'nvz': float, components of the normal vector.
      - 'tvx', 'tvy': float, components of the tangential vector.

    Raises:
    - ValueError: if an invalid orientation is provided.
    """

    invert_tangential_component = kwargs.get("invert_tangential_component", False)
    if invert_tangential_component:
        ivt = -1
    else:
        ivt = 1
    if (orientation.lower() in ["u", "d", "l", "r", "up", "down", "left", "right"]) is False:
        raise ValueError("Not a valid orientation! Please, choose from 'up', 'down', 'left' or 'right'.")
    
    s = np.sin(incident_angle)
    c = np.cos(incident_angle)

    mirror_orientation = {
        "nvx":0,
        "nvy":0,
        "nvz":0,
        "tvx":0,
        "tvy":0,
    }

    if orientation.startswith("u"):
        mirror_orientation["nvx"] =  0
        mirror_orientation["nvy"] =  c
        mirror_orientation["nvz"] = -s
        mirror_orientation["tvx"] =  0
        mirror_orientation["tvy"] = -s*ivt
    if orientation.startswith("d"):
        mirror_orientation["nvx"] =  0
        mirror_orientation["nvy"] = -c
        mirror_orientation["nvz"] = -s
        mirror_orientation["tvx"] =  0
        mirror_orientation["tvy"] = -s*ivt
    if orientation.startswith("l"):
        mirror_orientation["nvx"] = -c
        mirror_orientation["nvy"] =  0
        mirror_orientation["nvz"] = -s
        mirror_orientation["tvx"] =  s*ivt
        mirror_orientation["tvy"] =  0
    if orientation.startswith("r"):
        mirror_orientation["nvx"] =  c
        mirror_orientation["nvy"] =  0
        mirror_orientation["nvz"] = -s
        mirror_orientation["tvx"] = -s*ivt
        mirror_orientation["tvy"] =  0

    return mirror_orientation


def wavefront_info(wft)->None:
    "Prints detailed information about the wavefront properties."

    Dx = wft.mesh.xFin - wft.mesh.xStart
    dx = Dx/(wft.mesh.nx-1)
    Dy = wft.mesh.yFin - wft.mesh.yStart
    dy = Dy/(wft.mesh.ny-1)

    print('\nWavefront information:')
    print(f'Nx = {wft.mesh.nx}, Ny = {wft.mesh.ny}')
    print(f'dx = {dx * 1E6:.4f} um, dy = {dy * 1E6:.4f} um')
    print(f'range x = {Dx * 1E3:.4f} mm, range y = {Dy * 1E3:.4f} mm')
    print(f'Rx = {wft.Rx:.8f} m, Ry = {wft.Ry:.8f} m')

def srw_quick_plot(wfr, phase=False, me=0, backend="srw"):
    
    mesh0 = deepcopy(wfr.mesh)
    arI = array('f', [0]*mesh0.nx*mesh0.ny)
    srwlib.srwl.CalcIntFromElecField(arI, wfr, 6, me, 3, mesh0.eStart, 0, 0)
    arIx = array('f', [0]*mesh0.nx)
    srwlib.srwl.CalcIntFromElecField(arIx, wfr, 6, me, 1, mesh0.eStart, 0, 0)
    arIy = array('f', [0]*mesh0.ny)
    srwlib.srwl.CalcIntFromElecField(arIy, wfr, 6, me, 2, mesh0.eStart, 0, 0)

    if phase:
        arP = array('d', [0]*mesh0.nx*mesh0.ny)
        srwlib.srwl.CalcIntFromElecField(arP, wfr, 0, 4, 3, mesh0.eStart, 0, 0)
        arPx = array('d', [0]*mesh0.nx)
        srwlib.srwl.CalcIntFromElecField(arPx, wfr, 0, 4, 1, mesh0.eStart, 0, 0)
        arPy = array('d', [0]*mesh0.ny)
        srwlib.srwl.CalcIntFromElecField(arPy, wfr, 0, 4, 2, mesh0.eStart, 0, 0)

    if backend == "srw":
        plotMesh0x = [1000*mesh0.xStart, 1000*mesh0.xFin, mesh0.nx]
        plotMesh0y = [1000*mesh0.yStart, 1000*mesh0.yFin, mesh0.ny]
        uti_plot2d1d(arI, plotMesh0x, plotMesh0y, labels=['Horizontal Position [mm]', 'Vertical Position [mm]', 'Intensity'])

        if phase:
            uti_plot2d1d(arP, plotMesh0x, plotMesh0y, labels=['Horizontal Position [mm]', 'Vertical Position [mm]', 'Phase'])

        uti_plot_show()

#***********************************************************************************
# Miscellaneous 
#***********************************************************************************

def get_rays_from_wavefront(intensity: np.ndarray, phase: np.ndarray, n_rays: int, 
    upscale_factor: int = 1, method: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Generate rays from a wavefront defined by intensity and phase maps.

    Parameters:
    - intensity: np.ndarray, the intensity array.
    - phase: np.ndarray, the phase array.
    - n_rays: int, the number of rays to generate.
    - upscale_factor: int, factor by which to upscale the arrays (default is 1, no upscaling).
    - method: int, method of upscaling. (0 = interpolation)

    Returns:
    - ray_positions: np.ndarray, positions of the sampled rays (n_rays, 2).
    - ray_directions: np.ndarray, directions of the sampled rays (n_rays, 2).
    """

    if upscale_factor > 1:
        if method == 0:
            upscaled_intensity = zoom(intensity, upscale_factor, order=1)
            upscaled_phase = zoom(phase, upscale_factor, order=1)
        else:
            raise ValueError("Upscaling method not implemented.")
    else:
        upscaled_intensity = intensity
        upscaled_phase = phase

    intensity_normalized = upscaled_intensity / np.sum(upscaled_intensity)

    positions = np.random.choice(np.arange(intensity_normalized.size),
                                 p=intensity_normalized.ravel(), 
                                 size=n_rays)
    
    y_indices, x_indices = np.unravel_index(positions, intensity_normalized.shape)

    # Calculate phase gradients to determine ray directions
    grad_y, grad_x = np.gradient(upscaled_phase)

    # Get ray directions at sampled positions
    ray_directions = np.column_stack((grad_x[y_indices, x_indices], grad_y[y_indices, x_indices]))

    # Ray positions in upscaled coordinates
    ray_positions = np.column_stack((x_indices, y_indices))

    return ray_positions, ray_directions


