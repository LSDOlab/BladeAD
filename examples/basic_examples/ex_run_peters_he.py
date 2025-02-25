"""Example Peters--He"""
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


recorder = csdl.Recorder(inline=False, debug=False)
recorder.start()

num_nodes = 10
num_radial = 35
num_azimuthal = 40

num_blades = 2
num_cp = 5

# # thrust_vector = csdl.Variable(value=np.array([1 * np.sin(np.deg2rad(5)), 0, -1 * np.cos(np.deg2rad(5))]))
# thrust_vector = csdl.Variable(value=np.array([0, 0, -1 ]))
# thrust_origin = csdl.Variable(value=np.array([0. ,0., 0.]))
# chord_control_points = csdl.Variable(value=np.linspace(0.031 ,0.012, num_cp))
# twist_control_points = csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_cp))

# chord_control_points.set_as_design_variable(lower=0.01, upper=0.05, scaler=10)
# twist_control_points.set_as_design_variable(lower=np.deg2rad(2), upper=np.deg2rad(40), scaler=10)

# profile_parameterization = BsplineParameterization(
#     num_radial=num_radial,
#     num_cp=num_cp,
# )
# chord_profile = profile_parameterization.evaluate_radial_profile(chord_control_points)
# twist_profile = profile_parameterization.evaluate_radial_profile(twist_control_points)

# radius = csdl.Variable(value=0.3048 / 2)
# mesh_vel_np = np.zeros((num_nodes, 3))
# mesh_vel_np[:, 0] = 0 # np.linspace(0, 10, num_nodes)
# mesh_vel_np[:, 1] = 0
# mesh_vel_np[:, 2] = 0
# mesh_velocity = csdl.Variable(value=mesh_vel_np)
# rpm = csdl.Variable(value=4058 * np.ones((num_nodes,)))


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

naca_4412_polar = ZeroDAirfoilPolarParameters(
    alpha_stall_minus=-14, 
    alpha_stall_plus=16,
    Cl_stall_minus=-1.15,
    Cl_stall_plus=1.7,
    Cd_stall_minus=0.028,
    Cd_stall_plus=0.041,
    Cd_0=0.008,
    Cl_0=0.5,
    Cl_alpha=6.13883351925882,
)
naca_4412_zero_d_model = ZeroDAirfoilModel(naca_4412_polar)

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

inputs = RotorAnalysisInputs(
    rpm=rpm,
    mesh_parameters=bem_mesh_parameters,
    mesh_velocity=mesh_velocity,
)

peters_he_model = PetersHeModel(
    num_nodes=num_nodes,
    airfoil_model=naca_4412_2d_model, #naca_4412_zero_d_model, #mh_117_2d_model, #airfoil_model_1,
    integration_scheme='trapezoidal',
)

import time 

t1 = time.time()
outputs = peters_he_model.evaluate(inputs=inputs)

recorder.execute()

T = outputs.total_thrust
Q = outputs.total_torque

print("thrust Peters--He", T.value)
print("torque Peters--He", Q.value)
t2 = time.time()
print(t2-t1)

t3 = time.time()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder, gpu=True,
    additional_inputs=[rpm],
    additional_outputs=[T, Q]
)

jax_sim.run()

print("thrust Peters--He", jax_sim[T])
print("torque Peters--He", jax_sim[Q])
t4 = time.time()
print(t4-t3)

exit()

torque = outputs.total_torque
torque.set_as_objective()
thrust = outputs.total_thrust
thrust.set_as_constraint(upper=5.5, lower=5.5, scaler=1/5)

# make_polarplot(outputs.sectional_thrust, plot_contours=False, azimuthal_offset=-90)

# print(thrust.value)
q1 = torque.value

# exit()

# recorder.visualize_graph(trim_loops=True)
from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# verify_derivatives_inline([FOM], [chord_control_points], 1e-6, raise_on_error=True)
recorder.stop()
# exit()
# Create a Simulator object from the Recorder object
sim = csdl.experimental.PySimulator(recorder)

from modopt import CSDLAlphaProblem
from modopt import SLSQP

prob = CSDLAlphaProblem(problem_name='rotor_blade',simulator=sim)


# Setup your preferred optimizer (here, SLSQP) with the Problem object 
# Pass in the options for your chosen optimizer
optimizer = SLSQP(prob, ftol=1e-6, maxiter=30, outputs=['x'])

# Solve your optimization problem
optimizer.solve()

# Print results of optimization
optimizer.print_results()

t = thrust.value
q2 = torque.value
print("thrust", t)
print("optimized torque", q2)
print("percent improvement", (q2 - q1)/q1 * 100)

