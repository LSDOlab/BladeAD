import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.utils.plot import make_polarplot
import numpy as np
# from jax._src import config
# config.update("jax_platforms", "cpu")

recorder = csdl.Recorder(inline=True, debug=False, expand_ops=True)
recorder.start()

num_nodes = 1
num_radial = 25
num_azimuthal = 25

num_blades = 2
num_cp = 5

tilt_angle = np.deg2rad(10)
tx = np.cos(tilt_angle)
tz = np.sin(tilt_angle)
thrust_vector=csdl.Variable(value=np.array([tx, 0, -tz]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
# chord_profile=csdl.Variable(value=np.linspace(0.031 ,0.012, num_radial))
# twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_radial))
radius = csdl.Variable(value=0.3048001 / 2)
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = np.linspace(0.0, 10.0, num_nodes) # np.linspace(0, 50.06, num_nodes)
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=4400 * np.ones((num_nodes,)))


chord_control_points = csdl.Variable(value=np.linspace(0.031 ,0.012, num_cp))
twist_control_points = csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_cp))

chord_control_points.set_as_design_variable(lower=0.005, upper=0.05, scaler=50)
twist_control_points.set_as_design_variable(lower=np.deg2rad(2), upper=np.deg2rad(40), scaler=15)

profile_parameterization = BsplineParameterization(
    num_radial=num_radial,
    num_cp=num_cp,
)
chord_profile = profile_parameterization.evaluate_radial_profile(chord_control_points)
twist_profile = profile_parameterization.evaluate_radial_profile(twist_control_points)

# rpm.set_as_design_variable(scaler=8e-4)
# radius.set_as_design_variable(scaler=4, upper=0.25)
# mesh_velocity.set_as_design_variable()

ml_airfoil_model = NACA4412MLAirfoilModel()

two_ml_model = TwoDMLAirfoilModel(airfoil_name="naca_4412")

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

bem_inputs = RotorAnalysisInputs(
    rpm=rpm,
    mesh_parameters=bem_mesh_parameters,
    mesh_velocity=mesh_velocity,
)

bem_model = PittPetersModel(
    num_nodes=num_nodes,
    airfoil_model=ml_airfoil_model, #naca_4412_zero_d_model, #
    integration_scheme='trapezoidal',
    use_frange_in_nl_solver=False,
    # nl_solver_mode="vectorized",
)
outputs = bem_model.evaluate(inputs=bem_inputs)

thrust_stack = outputs.total_thrust
torque_stack = outputs.total_torque

import time 

torque = csdl.average(torque_stack)
torque.set_as_objective(scaler=10)
thrust = csdl.average(thrust_stack)
thrust.set_as_constraint(upper=5.5, lower=5.5, scaler=1/5)


recorder.iinline = False
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder, gpu=False,
    derivatives_kwargs= {
            "concatenate_ofs" : True # Turn off
        }
)

t4 = time.time()
jax_sim.compute_optimization_derivatives()
t5 = time.time()
jax_sim.compute_optimization_derivatives()
t6 = time.time()
print("Compile derivative function time: ", t5-t4)
print("derivative function time: ", t6-t5)

jax_sim.check_optimization_derivatives()
