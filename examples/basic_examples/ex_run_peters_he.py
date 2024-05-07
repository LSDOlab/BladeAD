import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.utils.plot import make_polarplot
import numpy as np


recorder = csdl.Recorder(inline=True)
recorder.start()

num_nodes = 1
num_radial = 31
num_azimuthal = 50

num_blades = 2
num_cp = 5

thrust_vector = csdl.Variable(value=np.array([1 * np.sin(np.deg2rad(5)), 0, -1 * np.cos(np.deg2rad(5))]))
thrust_origin = csdl.Variable(value=np.array([0. ,0., 0.]))
chord_control_points = csdl.Variable(value=np.linspace(0.031 ,0.012, num_cp))
twist_control_points = csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_cp))

profile_parameterization = BsplineParameterization(
    num_radial=num_radial,
    num_cp=num_cp,
)
chord_profile = profile_parameterization.evaluate_radial_profile(chord_control_points)
twist_profile = profile_parameterization.evaluate_radial_profile(twist_control_points)

radius = csdl.Variable(value=0.3048 / 2)
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 10 # np.linspace(0, 10, num_nodes)
mesh_vel_np[:, 1] = 0
mesh_vel_np[:, 2] = 0
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=4058 * np.ones((num_nodes,)))

polar_parameters_1 = ZeroDAirfoilPolarParameters(
    alpha_stall_minus=-10.,
    alpha_stall_plus=15.,
    Cl_stall_minus=-1.,
    Cl_stall_plus=1.5,
    Cd_stall_minus=0.02,
    Cd_stall_plus=0.06,
    Cl_0=0.48,
    Cd_0=0.01,
    Cl_alpha=5.1566,
)

# polar_parameters_2 = ZeroDAirfoilPolarParameters(
#     alpha_stall_minus=-10.,
#     alpha_stall_plus=15.,
#     Cl_stall_minus=-1.,
#     Cl_stall_plus=1.5,
#     Cd_stall_minus=0.02,
#     Cd_stall_plus=0.06,
#     Cl_0=0.3,
#     Cd_0=0.01,
#     Cl_alpha=5.1566,
# )

airfoil_model_1 = ZeroDAirfoilModel(
    polar_parameters=polar_parameters_1,
)

# airfoil_model_2 = ZeroDAirfoilModel(
#     polar_parameters=polar_parameters_2,
# )


# ml_airfoil_model = NACA4412MLAirfoilModel()

# airfoil_model = CompositeAirfoilModel(
#     sections=[0., 0.5, 0.7, 1.],
#     airfoil_models=[airfoil_model_1, airfoil_model_1, airfoil_model_1]
# )

naca_4412_2d_model = TwoDMLAirfoilModel(
    airfoil_name="naca_4412"
)

clark_y_2d_model = TwoDMLAirfoilModel(
    airfoil_name="clark_y"
)

mh_117_2d_model = TwoDMLAirfoilModel(
    airfoil_name="mh_117"
)

airfoil_model = CompositeAirfoilModel(
    sections=[0., 0.35, 0.7, 1.],
    airfoil_models=[naca_4412_2d_model, mh_117_2d_model, clark_y_2d_model],
)


bem_mesh_parameters = RotorMeshParameters(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    chord_profile=chord_profile,
    twist_profile=twist_profile, 
    radius=radius,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
)

bem_inputs = RotorAnalysisInputs(
    rpm=rpm,
    mesh_parameters=bem_mesh_parameters,
    mesh_velocity=mesh_velocity,
)

bem_model = PetersHeModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
)

import time 

t1 =  time.time()
outputs = bem_model.evaluate(inputs=bem_inputs)
t2 = time.time()

make_polarplot(outputs.sectional_thrust, plot_contours=False, azimuthal_offset=-90)

print(outputs.total_thrust.value)

print("time", t2-t1)
# recorder.visualize_graph(trim_loops=True)
recorder.stop()

