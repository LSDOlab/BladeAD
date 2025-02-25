import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import csdl_alpha as csdl
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
# from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
from BladeAD.utils.plot import make_polarplot


recorder = csdl.Recorder(inline=True, expand_ops=True)
recorder.start()

# Discretization
num_nodes = 1
num_radial = 35
num_azimuthal = 50

num_blades = 4

radius = csdl.Variable(shape=(1, ), value=0.860552)
chord_profile = csdl.Variable(shape=(num_radial, ), value=0.06604)

root_twist = csdl.Variable(name="root_twist", shape=(1, ), value=np.deg2rad(8- 2.470588235))
tip_twist = root_twist - np.deg2rad(8)
twist_profile = csdl.linear_combination(root_twist, tip_twist, num_radial).flatten()

alpha = -3.04# -5.7 # -3.0
tilt_angle = np.deg2rad(90+alpha)
tx = np.cos(tilt_angle)
tz = np.sin(tilt_angle)
thrust_vector=csdl.Variable(value=np.array([tx, 0, -tz]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 43.8605044 # 28.50022 # 66.7512 #
mesh_vel = csdl.Variable(shape=(num_nodes, 3), value=mesh_vel_np)
rpm = csdl.Variable(shape=(num_nodes,), value=2113)


naca_0012_airfoil_model = TwoDMLAirfoilModel(
    airfoil_name="naca_0012"
)

# airfoil_model_maker = ThreeDAirfoilMLModelMaker(
#     airfoil_name="naca_0012",
#     aoa_range=np.linspace(-12, 16, 50),
#     reynolds_range=[2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 4e6],
#     mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6]
# )

# airfoil_model = airfoil_model_maker.get_airfoil_model(
#     quantities=["Cl", "Cd"],
# )

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
    theta_0=np.deg2rad(8.16), #,  9.2, 9.37)
    theta_1_c=np.deg2rad(1.52), #  0.3, 1.11),
    theta_1_s=np.deg2rad(-4.13),#  -6.8 -3.23),
    xi_0=np.deg2rad(-0.9), #-0.95, # -1.3
)

peters_he_model = PetersHeModel(
    num_nodes=num_nodes,
    airfoil_model=naca_0012_airfoil_model, #airfoil_model, #
    tip_loss=True,
    Q=6,
    M=6,
)

outputs = peters_he_model.evaluate(inputs=inputs)

Ct = outputs.thrust_coefficient

Ct_re = csdl.norm(Ct - 0.0063)
Ct_re.set_as_objective(scaler=10)

py_sim = csdl.experimental.PySimulator(
    recorder=recorder
)
# py_sim.check_totals()

print(outputs.thrust_coefficient.value)
print(outputs.total_thrust.value)


# exit()

# Load data from NASA report
# df = pd.read_csv("nasa_report_rotor_inflow_data_mu_015.csv")
# df = pd.read_csv("nasa_report_rotor_inflow_data_mu_035.csv")
df = pd.read_csv("nasa_report_rotor_inflow_data_mu_023.csv")
do_contours = False

# Azimuth and normalized radial stations
psi_r_data = df.iloc[:, [0, 1]].values
theta = psi_r_data[:, 0] * np.pi/180
r = psi_r_data[:, 1]

# Normalized inflow ratio
lambda_i = df.iloc[:, 2].values *-1
# lambda_i +=  df.iloc[:, 3].values
lambda_i = lambda_i.tolist() 

# Define a regular grid for the contour plot
num_theta = 200
num_r = 100

theta_grid = np.linspace(0, np.max(theta), num_theta)
r_grid = np.linspace(np.min(r), np.max(r), num_r)
Theta, R = np.meshgrid(theta_grid, r_grid)

# Interpolate the data onto the regular grid
interp_data = griddata((theta, r), lambda_i, (Theta, R), method='linear')

# Plot the data
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
mesh = ax.pcolormesh(Theta, R, interp_data, cmap='viridis')

if True:
    num_contours = 15
    contour_levels = np.linspace(np.min(lambda_i), np.max(lambda_i), num_contours)
    contour = ax.contour(Theta, R, interp_data, levels=contour_levels, colors='k', linestyles='dashed', extend="both")
    ax.clabel(contour, inline=True, fontsize=10)
    # ax.set_rlim(np.min(R), np.max(R))

make_polarplot(outputs.inflow, plot_contours=True, plot_min_max=False)#, vmin=np.min(interp_data), vmax=np.max(interp_data))
# make_polarplot(outputs.sectional_thrust / 4, plot_contours=True)#, azimuthal_offset=90)#, vmin=np.min(interp_data), vmax=np.max(interp_data))
print(csdl.sum(outputs.sectional_thrust, axes=(1,)).value[0, 0])
print(csdl.sum(outputs.sectional_thrust, axes=(1,)).value[0, 12])
print(csdl.sum(outputs.sectional_thrust, axes=(1,)).value[0, 25])
print(csdl.sum(outputs.sectional_thrust, axes=(1,)).value[0, 37])

# Add a colorbar
# cbar = plt.colorbar(contour, ax=ax)
cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('Quantity of Interest')

# Set the title
ax.set_title("Axial induced inflow ratio")

# Show the plot
plt.show()
exit()

