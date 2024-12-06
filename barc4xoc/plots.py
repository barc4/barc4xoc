""" 
Auxiliary plot tools
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '05/NOV/2024'
__changed__ = '05/NOV/2024'

from barc4plots.barc4plots import PlotManager
import matplotlib.cm as cm

from typing import Optional, List, Dict
import numpy as np
import matplotlib.cm as cm

def plot_beam(
    beam: Dict[str, np.ndarray],
    plot_type: str,
    direction: Optional[str] = None,
    prange: Optional[List[Optional[float]]] = None,
    file_name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot different aspects of the photon beam: phase space, beam size, or beam divergence.

    This function allows plotting either the phase space, spatial distribution, or angular 
    distribution of a beam by specifying the plot type and direction (if applicable). 
    It uses `PlotManager` to generate scatter plots with optional histogram projections 
    along each axis, and users can set plot limits and file saving options.

    Args:
        beam (Dict[str, np.ndarray]): A dictionary containing beam properties, including:
            - 'X' and 'Y': Position components.
            - 'Xp' and 'Yp': Angular components.
        plot_type (str): The type of plot to generate ('phase_space', 'size', 'divergence').
            - 'phase_space' requires an additional `direction` argument ('x' or 'y').
            - 'size': Plots 'X' vs 'Y' for beam size.
            - 'divergence': Plots 'Xp' vs 'Yp' for beam divergence.
        direction (Optional[str], optional): The direction for phase space plotting ('x' or 'y').
        prange (Optional[List[Optional[float]]], optional): Plot range limits `[xmin, xmax, ymin, ymax]`.
        file_name (Optional[str], optional): File name for saving the plot.

    Returns:
        None
    """
    aspect_ratio = kwargs.get('aspect_ratio', True)

    if prange is None:
        prange = [None, None, None, None]
    
    if plot_type == "phase_space":
        if direction is None:
            raise ValueError("Direction must be specified for phase space plots ('x' or 'y').")
        direction = direction.strip().lower()
        if direction not in ['x', 'y']:
            raise ValueError("Invalid direction. Choose 'x' or 'y'.")
        d = direction.capitalize()
        axis_x = beam[f"{d}"] * 1E6
        axis_y = beam[f"{d}p"] * 1E6
        x_label = f"${d}$ [µm]"
        y_label = f"${d}_p$ [µrad]"
        color_scheme = 3

    elif plot_type == "size":
        axis_x = beam["X"] * 1E6
        axis_y = beam["Y"] * 1E6
        x_label = "$x$ [µm]"
        y_label = "$y$ [µm]"
        color_scheme = 2

    elif plot_type == "divergence":
        axis_x = beam["Xp"] * 1E6
        axis_y = beam["Yp"] * 1E6
        x_label = "$x_p$ [µrad]"
        y_label = "$y_p$ [µrad]"
        color_scheme = 2
    
    else:
        raise ValueError("Invalid plot_type. Choose 'phase_space', 'size', or 'divergence'.")

    fig = PlotManager(axis_x=axis_x, axis_y=axis_y)
    fig.additional_info(
        '', 
        x_label, 
        y_label, 
        xmin=prange[0], 
        xmax=prange[1], 
        ymin=prange[2], 
        ymax=prange[3]
    )
    
    fig.aesthetics(
        dpi=600, 
        LaTex=True, 
        AspectRatio=aspect_ratio, 
        FontsSize=None, 
        grid=True, 
        nbins=None
    )
    
    fig.info_scatter(
        ColorScheme=color_scheme,
        LineStyle='.', 
        alpha=1, 
        s=1, 
        edgeColors='face', 
        monochrome=False
    )
    
    fig.plot_scatter_hist(file_name)


def plot_tuning_curve(tc: dict, 
                      unit: str, 
                      title: str=None,
                      prange: Optional[list] = None, 
                      file_name: Optional[str] = None, 
                      even_harmonics: bool = False) -> None:
    """
    Plot the tuning curve for a given dataset.

    This function visualizes the tuning curve by plotting the flux as a function 
    of energy for different harmonics. The plot can optionally include even harmonics.

    Args:
        tc (dict): A dictionary containing the tuning curve data. It should have 
            the following keys:
            - "flux": A 2D NumPy array with shape (n_points, n_harmonics).
            - "energy": A 1D NumPy array representing energy values corresponding 
              to the flux data.
        unit (str): The label for the y-axis, representing the unit of flux.
        title (str): The name of the plot (optional)
        prange (Optional[list]): A list specifying the axis limits in the format 
            [xmin, xmax, ymin, ymax]. Defaults to None, which sets no limits.
        file_name (Optional[str]): The name of the file to save the plot. 
            If None, the plot will not be saved.
        even_harmonics (bool): If True, only even harmonics will be plotted. 
            Defaults to False, allowing both even and odd harmonics.

    Returns:
        None: This function does not return a value but generates a plot.
    """

    if prange is None:
        prange = [None, None, None, None]

    if (tc["flux"].shape[1]-1)% 2 == 0:
        nHarmMax = tc["flux"].shape[1]-2
    else:
        nHarmMax = tc["flux"].shape[1]-1

    fig = PlotManager()
    fig.additional_info(title, "energy [eV]", unit, 
                        xmin=prange[0], xmax=prange[1], 
                        ymin=prange[2], ymax=prange[3], 
                        sort_ax=False, sort_ax_lim=False).aesthetics(LaTex=True, grid=True)
    k = 0
    for nharm in range(nHarmMax):
        if (nharm + 1) % 2 == 0 and even_harmonics or (nharm + 1) % 2 != 0:
            flux = np.copy(tc["flux"][:, nharm+1])
            flux[flux==0] = np.nan
            fig.image, fig.x = flux, tc["energy"]
            if (nharm + 1) % 2 == 0:
                fig.info_1d_plot(k, f'n={(nharm + 1)}', 1, ":").plot_1d(enable=False, hold=(nharm+1 != 1))
            else:
                fig.info_1d_plot(k, f'n={(nharm + 1)}', 1, "-").plot_1d(enable=False, hold=(nharm+1 != 1))
            if not np.all(np.isnan(flux)):
                k += 1
    flux = np.copy(tc["flux"][:, 0])
    flux[flux==0] = np.nan
    fig.image = flux
    fig.info_1d_plot(-1, 'envelope', 1, "--").plot_1d(enable=True, hold=True, file_name=file_name)