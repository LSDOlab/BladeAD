import csdl_alpha as csdl
from BladeAD.utils.var_groups import BEMInputs, BEMMeshParameters
from BladeAD.core.airfoil.custom_airfoil_polar import TestAirfoilModel, SimpleAirfoilPolar
from BladeAD.core.BEM.bem_model import BEMModel
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()


num_nodes = 1
num_radial = 50
num_azimuthal = 50

num_blades = 2

thrust_vector=csdl.Variable(value=np.array([0., 0, -1]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
# chord_profile=csdl.Variable(value=np.linspace(0.25, 0.05, num_radial))
# twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(70), np.deg2rad(30), num_radial))
# radius = csdl.Variable(value=1.5)
chord_profile=csdl.Variable(value=np.linspace(0.031 ,0.012, num_radial))
twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_radial))
radius = csdl.Variable(value=0.3048 / 2)
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 10.0 # np.linspace(0, 50.06, num_nodes)
# mesh_vel_np[:, 2] = -10.
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=4400 * np.ones((num_nodes,)))

custom_polar = SimpleAirfoilPolar(
    alpha_stall_minus=-10.,
    alpha_stall_plus=15.,
    Cl_stall_minus=-1.,
    Cl_stall_plus=1.5,
    Cd_stall_minus=0.02,
    Cd_stall_plus=0.06,
    Cl_0=0.25,
    Cd_0=0.01,
    Cl_alpha=5.1566,
)

airfoil_model = TestAirfoilModel(
    custom_polar=custom_polar,
)

bem_mesh_parameters = BEMMeshParameters(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    chord_profile=chord_profile,
    twist_profile=twist_profile, 
    radius=radius,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
)

bem_inputs = BEMInputs(
    rpm=rpm,
    mesh_parameters=bem_mesh_parameters,
    mesh_velocity=mesh_velocity,
)

bem_model = BEMModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='Riemann',
)

outputs = bem_model.evaluate(inputs=bem_inputs)

T = outputs.total_thrust.value
Q = outputs.total_torque.value

print("thrust", T)
print("torque", Q)
print("eta", outputs.efficiency.value)

recorder.stop()


