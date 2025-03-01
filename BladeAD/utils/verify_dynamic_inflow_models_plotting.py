import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import pandas as pd


def plot_vnv_data(
    nasa_data_df : pd.DataFrame,
    nasa_C_T : float,
    blade_ad_data : list,
    num_radial : int,
    num_azimuthal : int,
    model_name : list[str],
    model_C_T : list[float]
):
    # Azimuth and normalized radial stations
    psi_r_data = nasa_data_df.iloc[:, [0, 1]].values
    theta = psi_r_data[:, 0] * np.pi/180
    r = psi_r_data[:, 1]

    # Normalized inflow ratio
    lambda_i = nasa_data_df.iloc[:, 2].values *-1
    lambda_i = lambda_i.tolist() 

    # Define a regular grid for the contour plot
    theta_grid = np.linspace(0, np.max(theta), num_azimuthal)
    r_grid = np.linspace(np.min(r), np.max(r), num_radial)
    Theta, R = np.meshgrid(theta_grid, r_grid)

    # Interpolate the data onto the regular grid
    interp_data = griddata((theta, r), lambda_i, (Theta, R), method='cubic')

    num_cols = len(blade_ad_data)

    # (num_cols x 3) polar subplots for NASA data, BladeAD data and difference
    fig, axs = plt.subplots(1, num_cols+1, subplot_kw={'projection': 'polar'}, figsize=(15, 5))
    num_contours = 10

    for i in range(num_cols+1):
        if i == 0:
            mesh_1 = axs[i].pcolormesh(Theta, R, interp_data, cmap='viridis')
            title = "NASA data" + "\n" + rf"$C_T$= {nasa_C_T:.5f}"
            axs[i].set_title(title, fontsize=10)
            contour_levels_nasa = np.linspace(np.min(lambda_i), np.max(lambda_i), num_contours)
            contour_nasa = axs[i].contour(Theta, R, interp_data, levels=contour_levels_nasa, colors='k', linestyles='dashed', extend="both")
            axs[i].clabel(contour_nasa, inline=True, fontsize=10)
    
            c_bar_1 = fig.colorbar(mesh_1, ax=axs[i], orientation='horizontal', fraction=0.046)# , pad=0.2, aspect=15)
            c_bar_1.set_label(r'Inflow ratio $\lambda_i$', size=8)

        else:
            model_data = blade_ad_data[i-1].squeeze()
            mesh_2 = axs[i].pcolormesh(Theta, R, model_data, cmap='viridis')
            title = model_name[i-1] + "\n" + rf"$C_T$= {model_C_T[i-1]:.5f}"
            axs[i].set_title(title, fontsize=10)
            contour_levels_model = np.linspace(np.min(model_data), np.max(model_data), num_contours)
            contour_model = axs[i].contour(Theta, R, model_data, levels=contour_levels_model, colors='k', linestyles='dashed', extend="both")
            axs[i].clabel(contour_model, inline=True, fontsize=10)

            c_bar_2 = fig.colorbar(mesh_2, ax=axs[i], orientation='horizontal', fraction=0.046)#,  pad=0.2, aspect=15)
            c_bar_2.set_label(r'Inflow ratio $\lambda_i$', size=8)

    plt.tight_layout()
    plt.show()
