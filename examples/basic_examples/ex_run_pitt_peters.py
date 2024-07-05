import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from BladeAD.utils.plot import make_polarplot
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()


num_nodes = 5
num_radial = 35
num_azimuthal = 50

num_blades = 2
 
# thrust_vector=csdl.Variable(value=np.array([1 * np.sin(np.deg2rad(90)), 0, -1 * np.cos(np.deg2rad(90))]))
# thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
# chord_profile=csdl.Variable(value=np.linspace(0.031 ,0.012, num_radial))
# twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_radial))
# radius = csdl.Variable(value=0.3048 / 2)
# mesh_vel_np = np.zeros((num_nodes, 3))
# mesh_vel_np[:, 0] = np.linspace(0, 10, num_nodes)
# mesh_vel_np[:, 1] = 0
# mesh_vel_np[:, 2] = 0
# mesh_velocity = csdl.Variable(value=mesh_vel_np)
# rpm = csdl.Variable(value=4400 * np.ones((num_nodes,)))

thrust_vector=csdl.Variable(value=np.array([1., 0, 0]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
# chord_profile=csdl.Variable(value=np.linspace(0.25, 0.05, num_radial))
# twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(70), np.deg2rad(30), num_radial))
# radius = csdl.Variable(value=1.5)
chord_profile=csdl.Variable(value=np.linspace(0.031 ,0.012, num_radial))
twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_radial))
radius = csdl.Variable(value=0.3048001 / 2)
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = np.linspace(0, 10.0, num_nodes) # np.linspace(0, 50.06, num_nodes)
# mesh_vel_np[:, 2] = -10.
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=4400 * np.ones((num_nodes,)))

polar_parameters_1 = ZeroDAirfoilPolarParameters(
    alpha_stall_minus=-10.,
    alpha_stall_plus=15.,
    Cl_stall_minus=-1.,
    Cl_stall_plus=1.5,
    Cd_stall_minus=0.02,
    Cd_stall_plus=0.06,
    Cl_0=0.5,
    Cd_0=0.008,
    Cl_alpha=5.1566,
)

polar_parameters_2 = ZeroDAirfoilPolarParameters(
    alpha_stall_minus=-12.,
    alpha_stall_plus=15.,
    Cl_stall_minus=-1.,
    Cl_stall_plus=1.5,
    Cd_stall_minus=0.02,
    Cd_stall_plus=0.06,
    Cl_0=0.48,
    Cd_0=0.01,
    Cl_alpha=5.1566,
)

airfoil_model_1 = ZeroDAirfoilModel(
    polar_parameters=polar_parameters_1,
)

airfoil_model_2 = ZeroDAirfoilModel(
    polar_parameters=polar_parameters_2,
)

# airfoil_model = CompositeAirfoilModel(
#     sections=[0., 0.5, 0.7, 1.],
#     airfoil_models=[airfoil_model_1, airfoil_model_1, airfoil_model_2]
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
    airfoil_models=[naca_4412_2d_model, clark_y_2d_model, mh_117_2d_model],
)


# airfoil_model = NACA4412MLAirfoilModel()


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

bem_model = PittPetersModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,#NACA4412MLAirfoilModel(),
    integration_scheme='trapezoidal',
)

import time 

outputs = bem_model.evaluate(inputs=bem_inputs)
T = outputs.total_thrust
T.set_as_constraint(lower=5.7, upper=5.7)
Q = csdl.average(outputs.total_torque)
Q.set_as_objective(scaler=10)

print("thrust BEM", T.value)
print("torque BEM", Q.value)
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder, gpu=True,
    additional_inputs=[rpm],
    additional_outputs=[T, Q]
)

jax_sim.run()

# make_polarplot(outputs.sectional_thrust, plot_contours=False)

print("thrust Pitt--Peters", outputs.total_thrust.value)
print("torque Pitt--Peters", outputs.total_torque.value)
# recorder.visualize_graph(trim_loops=True)
recorder.stop()

