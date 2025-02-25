"""Blade Element Momentum (BEM) Rotor Analysis Script"""
import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
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
# chord and twist profiles (linearly varying from root to tip)
chord_profile=csdl.Variable(name="chord_profile", value=np.linspace(0.25, 0.05, num_radial))
twist_profile=csdl.Variable(name="twist_profile", value=np.linspace(np.deg2rad(50), np.deg2rad(20), num_radial)) # Twist in RADIANS
# Radius of the rotor
radius = csdl.Variable(name="radius", value=1.2)

# 3) Mesh velocity: vector of shape (num_nodes, 3) where each row is the 
# free streamvelocity vector (u, v, w) at the rotor center
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 50 # Free stream velocity in the x-direction
mesh_velocity = csdl.Variable(value=mesh_vel_np)
# Rotor speed in RPM
rpm = csdl.Variable(value=2000 * np.ones((num_nodes,)))

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

# Print scalar outputs
print("Scalar Outputs:")
print(f"Total thrust: {bem_outputs.total_thrust.value}")
print(f"Total torque: {bem_outputs.total_torque.value}")
print(f"Efficiency: {bem_outputs.efficiency.value}")
print(f"Thrust coefficient: {bem_outputs.thrust_coefficient.value}")
print(f"Torque coefficient: {bem_outputs.torque_coefficient.value}")
print("\n")
