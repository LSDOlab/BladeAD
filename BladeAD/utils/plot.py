import matplotlib.pyplot as plt
import numpy as np
import csdl_alpha as csdl
from typing import Union


def make_polarplot(
        data: csdl.Variable,
        radius: Union[csdl.Variable, float, int, None] = None,
        norm_hub_radius: Union[float] = 0.2,
        quantity_name: list[str] = ['quantity'], 
        plot_contours: bool = True,
        num_contours: int = 10,
        plot_min_max: bool=True,
        azimuthal_offset: Union[float, int, None] = None,
        fig_size: tuple = (10, 8),
        cmap: str='viridis',

    ):
    """Generate polar plots for quantities that vary azimuthally and radially. 

    If radius is not provided, plot between normalized hub radius and 1. 
    
    Works for multiple evaluations (i.e., num_nodes>1)

    Parameters
    ----------
    data : csdl.Variable
        Data to be plotted; must be of shape (num_nodes, num_radial, num_azimuthal)
    
    radius : Union[csdl.Variable, float, int, None], optional
        rotor radius; if None, plot between hub radius and 1, by default None
    
    norm_hub_radius : float, optional
        fraction of radius, by default 0.2
    
    quantity_name : str, optional
        name of the quantity to be plotted, by default 'quantity'
    
    plot_contours : bool, optional
        flag for plotting contours, by default True

    num_contours : int, optional
        number of contours, by default 10

    plot_min_max : bool, optional
        flag for plotting min/max values

    azimuthal_offset : Union[float, int, None], optional
        rotate the polar plot counter clockwise (in degrees), by default None
    
    fig_size : tuple, optional
        by default (10, 8)
    
    cmap : str, optional
        string for matplotlib colormaps, by default 'viridis'
    """
    csdl.check_parameter(data, "data", types=(csdl.Variable, np.ndarray))
    csdl.check_parameter(radius, "radius", types=(int, float, csdl.Variable), allow_none=True)
    # csdl.check_parameter(psi, "psi", types=(csdl.Variable, np.ndarray))
    if len(data.shape) != 3:
        raise ValueError("data must be of shape (num_nodes, num_radial, num_azimuthal)")
    if isinstance(data, csdl.Variable):
        data = data.value
    if data.shape[2] < 2:
        raise ValueError("num_azimuthal must be at least 2")
    if isinstance(radius, csdl.Variable):
        if len(radius.shape) > 1:
            raise ValueError("radius should be a scalar")
        else:
            if radius.shape[0] > 1:
                raise ValueError("radius should be a scaler")

    num_nodes, num_radial, num_azimuthal = data.shape

    if not isinstance(quantity_name, (str, list)):
        raise ValueError("quantity_name must be a string or list of strings")

    if isinstance(quantity_name, str):
        quantity_name = [quantity_name]

    if len(quantity_name) > 1:
        if len(quantity_name) != num_nodes:
            raise ValueError(f"length of quantity name list must be equal to 1 or num_nodes; received length {len(quantity_name)}")
    if len(quantity_name) == 1:
        quantity_name = quantity_name[0]

    # Calculate the number of rows and columns for subplots
    num_cols = int(np.ceil(np.sqrt(num_nodes)))
    num_rows = int(np.ceil(num_nodes / num_cols))

    # Create a figure and axes
    if num_nodes == 1:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=fig_size)
        axs = ax
    elif num_nodes ==2:
        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=fig_size)
    else:
        fig, axs = plt.subplots(num_rows, num_cols, subplot_kw={'projection': 'polar'}, figsize=fig_size)

        # Flatten the axes if it's not a 2D array
        if num_rows == 1 or num_cols == 1:
            axs = axs.flatten()

    # Plot each subplot
    for i in range(num_nodes):
        if num_nodes == 1:
            ax = axs
        elif num_nodes == 2:
            ax = axs[i]
        else:
            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col]
            ax.set_title(quantity_name)

        # Unpack the data for this node
        radial_data = data[i, :, :]

        max_idx = np.unravel_index(np.argmax(radial_data), radial_data.shape)
        min_idx = np.unravel_index(np.argmin(radial_data), radial_data.shape)

        if radius is not None:
            if isinstance(radius, csdl.Variable):
                radius = radius.value
            theta, r = np.meshgrid(np.linspace(0, 2*np.pi - 2 * np.pi/num_azimuthal, num_azimuthal), np.linspace(norm_hub_radius, radius, num_radial))
        else:
            theta, r = np.meshgrid(np.linspace(0, 2*np.pi - 2 * np.pi/num_azimuthal, num_azimuthal), np.linspace(norm_hub_radius, 1, num_radial))

        if azimuthal_offset is not None:
            theta += np.deg2rad(azimuthal_offset)

        # Plot the contour plot for this node
        mesh = ax.pcolormesh(theta, r, radial_data, cmap=cmap)
        contour_levels = np.linspace(np.min(radial_data), np.max(radial_data), num_contours)  # Adjust number of contour levels as needed
        if plot_contours:
            cs = ax.contour(theta, r, radial_data, levels=contour_levels, colors='k', linestyles='dashed')
            ax.clabel(cs, inline=True, fontsize=10)
        if plot_min_max:
            ax.plot(theta[max_idx], r[max_idx], 'ro', label='Max Thrust')
            ax.plot(theta[min_idx], r[min_idx], 'bo', label='Min Thrust')
            ax.legend()

        ax.set_theta_zero_location('E')  # Set 0 angle to the right
        ax.set_theta_direction(1)  # counter-clockwise direction

        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8)


    # Hide any remaining empty subplots
    for i in range(num_nodes, num_rows*num_cols):
        if num_nodes == 1 or num_nodes == 2:
            axs[i].axis('off')
        else:
            axs.flatten()[i].axis('off')

    # fig.tight_layout()
    plt.tight_layout()
    plt.show()



