import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import csdl_alpha as csdl
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.utils.plot import make_polarplot


recorder = csdl.Recorder(inline=True)
recorder.start()

# Discretization
num_nodes = 1
num_radial = 35
num_azimuthal = 50

num_blades = 4

radius = csdl.Variable(shape=(1, ), value=0.860552)
chord_profile = csdl.Variable(shape=(num_radial, ), value=0.06604)
twist_profile = csdl.Variable(value=np.linspace(np.deg2rad(40), 20, num_radial))

tilt_angle = np.deg2rad(5.7)
tx = np.cos(tilt_angle)
tz = np.sin(tilt_angle)
thrust_vector=csdl.Variable(value=np.array([tx, 0, -tz]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 66.7512
mesh_vel = csdl.Variable(shape=(num_nodes, 3), value=mesh_vel_np)
rpm = csdl.Variable(shape=(num_nodes,), value=2110.5)


naca_0012_airfoil_model = TwoDMLAirfoilModel(
    airfoil_name="naca_0012"
)

mesh_parameters = RotorMeshParameters(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    chord_profile=chord_profile,
    twist_profile=twist_profile, 
    radius=radius,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
)

inputs = RotorAnalysisInputs(
    rpm=rpm,
    mesh_parameters=mesh_parameters,
    mesh_velocity=mesh_vel,
)

peters_he_model = PetersHeModel(
    num_nodes=num_nodes,
    airfoil_model=naca_0012_airfoil_model,
)

outputs = peters_he_model.evaluate(inputs=inputs)

make_polarplot(outputs.sectional_thrust, plot_contours=False, azimuthal_offset=-90)

exit()

# Load data from NASA report
df = pd.read_csv("nasa_report_rotor_inflow_data.csv")
do_contours = False

# Azimuth and normalized radial stations
psi_r_data = df.iloc[:, [0, 1]].values
theta = psi_r_data[:, 0] * np.pi/180
r = psi_r_data[:, 1]

# Normalized inflow ratio
lambda_i = df.iloc[:, 5].values.tolist()

# Define a regular grid for the contour plot
num_theta = 200
num_r = 100

theta_grid = np.linspace(0, np.max(theta), num_theta)
r_grid = np.linspace(np.min(r), np.max(r), num_r)
Theta, R = np.meshgrid(theta_grid, r_grid)

# Interpolate the data onto the regular grid
interp_data = griddata((theta, r), lambda_i, (Theta, R), method='cubic')

# Plot the data
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
mesh = ax.pcolormesh(Theta, R, interp_data, cmap='viridis')

if do_contours:
    num_contours = 10
    contour_levels = np.linspace(np.min(lambda_i), np.max(lambda_i), num_contours)
    contour = ax.contour(Theta, R, interp_data, levels=contour_levels, colors='k', linestyles='dashed')
    ax.clabel(contour, inline=True, fontsize=10)


# Add a colorbar
# cbar = plt.colorbar(contour, ax=ax)
cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('Quantity of Interest')

# Set the title
ax.set_title("Axial induced inflow ratio")

# Show the plot
plt.show()
exit()

