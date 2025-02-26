"""Blade Element Momentum (BEM) Rotor Optimization Script"""
import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.BEM.bem_model import BEMModel
import numpy as np


# Start CSDL recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# Discretization
num_nodes = 1 # Number of evaluation points
num_radial = 35 # Number of radial sections
num_azimuthal = 1 # Number of azimuthal sections (can be 1 for axisymmetric flow)

num_blades = 2

# Simple 1D airfoil model
# Specify polar parameters
polar_parameters = ZeroDAirfoilPolarParameters(
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
# Create airfoil model
airfoil_model = ZeroDAirfoilModel(
    polar_parameters=polar_parameters,
)

# Set up rotor analysis inputs
# 1) thrust vector and origin (origin is the rotor hub location and only needed for computing moments)
thrust_vector=csdl.Variable(name="thrust_vector", value=np.array([1, 0, 0])) # Thrust vector in the x-direction
thrust_origin=csdl.Variable(name="thrust_origin", value=np.array([0. ,0., 0.]))

# 2) Rotor geometry 
# Chord and twist profiles parameterized as B-splines
# B-spline control points (design variables)
num_cp = 5
chord_profile_cps = csdl.Variable(name="chord_profile_cps", value=np.linspace(0.25, 0.05, num_cp)) # in meters
twist_profile_cps = csdl.Variable(name="twist_profile_cps", value=np.linspace(np.deg2rad(50), np.deg2rad(20), num_cp)) # Twist in RADIANS
# set control points as design variables
chord_profile_cps.set_as_design_variable(lower=0.01, upper=0.35, scaler=5)
twist_profile_cps.set_as_design_variable(lower=np.deg2rad(5), upper=np.deg2rad(85), scaler=2)

# Create B-spline parameterization
profile_parameterization = BsplineParameterization(
    num_radial=num_radial,
    num_cp=num_cp,
    order=4, # Cubic B-spline
)
# Evaluate radial profiles
chord_profile = profile_parameterization.evaluate_radial_profile(chord_profile_cps)
twist_profile = profile_parameterization.evaluate_radial_profile(twist_profile_cps)

# Radius of the rotor
radius = csdl.Variable(name="radius", value=1.2)

# 3) Mesh velocity: vector of shape (num_nodes, 3) where each row is the 
# free streamvelocity vector (u, v, w) at the rotor center
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 50 # Free stream velocity in the x-direction
mesh_velocity = csdl.Variable(value=mesh_vel_np)
# Rotor speed in RPM
rpm = csdl.Variable(value=1500 * np.ones((num_nodes,)))

# 4) Assemble inputs
# mesh parameters
bem_mesh_parameters = RotorMeshParameters(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    chord_profile=chord_profile,
    twist_profile=twist_profile, 
    radius=radius,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
    norm_hub_radius=0.2,
)
# rotor analysis inputs
inputs = RotorAnalysisInputs(
    rpm=rpm,
    mesh_parameters=bem_mesh_parameters,
    mesh_velocity=mesh_velocity,
)

# Instantiate and run BEM model
bem_model = BEMModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
)
bem_outputs = bem_model.evaluate(inputs=inputs)

# Set objective (minimize torque) and constraints (thrust)
torque = bem_outputs.total_torque
torque.set_as_objective(scaler=5e-3)

thrust = bem_outputs.total_thrust
thrust.set_as_constraint(equals=2000, scaler=1e-3)

# Print scalar outputs
print("Scalar Outputs before optimization:")
print(f"Total thrust: {bem_outputs.total_thrust.value}")
print(f"Total torque: {bem_outputs.total_torque.value}")
print(f"Efficiency: {bem_outputs.efficiency.value}")
print(f"Thrust coefficient: {bem_outputs.thrust_coefficient.value}")
print(f"Torque coefficient: {bem_outputs.torque_coefficient.value}")
print("\n")

# set up optimization 
# csdl simulator
sim = csdl.experimental.PySimulator(
    recorder=recorder,
)

# Import modopt (if installed)
try:
    import modopt
except ImportError:
    raise ImportError("Please install modopt to run this example (see https://modopt.readthedocs.io/en/latest/src/getting_started.html)")

from modopt import CSDLAlphaProblem

# Setup your optimization problem
prob = CSDLAlphaProblem(problem_name='blade_optimization', simulator=sim)

# Setup your preferred optimizer (here, SLSQP) with the Problem object 
# Pass in the options for your chosen optimizer
optimizer = modopt.SLSQP(prob, solver_options={'maxiter': 200, 'ftol': 1e-5})

# Solve your optimization problem
optimizer.solve()
optimizer.print_results()

# Print scalar outputs
print("Scalar Outputs after optimization:")
print(f"Total thrust: {bem_outputs.total_thrust.value}")
print(f"Total torque: {bem_outputs.total_torque.value}")
print(f"Efficiency: {bem_outputs.efficiency.value}")
print(f"Thrust coefficient: {bem_outputs.thrust_coefficient.value}")
print(f"Torque coefficient: {bem_outputs.torque_coefficient.value}")
print("\n")

# plot optimized blade profiles
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Please install matplotlib to run this example (pip install matplotlib)")

# normalize radius
norm_rad = np.linspace(0, 1, num_radial)


# Create figure
fig, ax1 = plt.subplots()
ax1.set_xlabel('Normalized Radius')
ax1.set_ylabel('Chord (m)', color='tab:blue')
ax1.plot(norm_rad, chord_profile.value, label='Chord', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Twist (rad)', color='tab:orange')
ax2.plot(norm_rad, twist_profile.value, label='Twist', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
plt.show()
