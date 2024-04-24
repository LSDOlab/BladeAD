import csdl_alpha as csdl
from BladeAD.utils.var_groups import BEMInputs, BEMMeshParameters
from BladeAD.core.airfoil.custom_airfoil_polar import TestAirfoilModel, SimpleAirfoilPolar
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()


num_nodes = 1
num_radial = 31
num_azimuthal = 30

num_blades = 2
 
thrust_vector=csdl.Variable(value=np.array([1 * np.sin(np.deg2rad(0)), 0, -1 * np.cos(np.deg2rad(0))]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
chord_profile=csdl.Variable(value=np.linspace(0.031 ,0.012, num_radial))
twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_radial))
radius = csdl.Variable(value=0.3048 / 2)
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 10
mesh_vel_np[:, 1] = 0
mesh_vel_np[:, 2] = 0
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=4058 * np.ones((num_nodes,)))


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

bem_model = PittPetersModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
)

import time 

t1 =  time.time()
outputs = bem_model.evaluate(inputs=bem_inputs)
t2 = time.time()

print("time", t2-t1)
# recorder.visualize_graph(trim_loops=True)
recorder.stop()

