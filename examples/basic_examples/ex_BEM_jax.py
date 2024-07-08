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


recorder = csdl.Recorder(inline=False, debug=False, expand_ops=False)
recorder.start()

vectorized = True
num_nodes = 20
num_radial = 35
num_azimuthal = 40

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

if vectorized:
    bem_inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_parameters=bem_mesh_parameters,
        mesh_velocity=mesh_velocity,
    )

    bem_model = BEMModel(
        num_nodes=num_nodes,
        airfoil_model=mh_117_2d_model, #airfoil_model_1,
        integration_scheme='trapezoidal',
    )
    outputs = bem_model.evaluate(inputs=bem_inputs)


else:
    bem_model = BEMModel(
        num_nodes=1,
        airfoil_model=mh_117_2d_model, #airfoil_model_1,
        integration_scheme='trapezoidal',
    )
    
    # for i in range(8):
    for i in csdl.frange(num_nodes):
        bem_inputs = RotorAnalysisInputs(
            rpm=rpm[i],
            mesh_parameters=bem_mesh_parameters,
            mesh_velocity=mesh_velocity[i, :].reshape((-1, 3)),
        )

        outputs = bem_model.evaluate(inputs=bem_inputs)


import time 


torque = csdl.average(outputs.total_torque)
torque.set_as_objective(scaler=10)
thrust = outputs.total_thrust
thrust.set_as_constraint(upper=5.5, lower=5.5, scaler=1/5)

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder, gpu=False,
)

jax_sim.check_totals()

exit()

# make_polarplot(outputs.sectional_thrust, plot_contours=False, azimuthal_offset=-90)

# print(thrust.value)
q1 = torque.value

# recorder.visualize_graph(trim_loops=True)
recorder.stop()
# exit()
# Create a Simulator object from the Recorder object
sim = csdl.experimental.PySimulator(recorder)

from modopt import CSDLAlphaProblem
from modopt import SLSQP, SNOPT, IPOPT

prob = CSDLAlphaProblem(problem_name='rotor_blade',simulator=jax_sim)


# Setup your preferred optimizer (here, SLSQP) with the Problem object 
# Pass in the options for your chosen optimizer
# optimizer = SLSQP(prob, ftol=1e-6, maxiter=200, outputs=['x'])
optimizer = IPOPT(prob, solver_options={'max_iter': 100, 'tol': 1e-5})
# optimizer = SNOPT(prob, 
#     Major_iterations=100, 
#     Major_optimality=1e-5, 
#     Major_feasibility=1e-5,
#     Major_step_limit=1.5,
#     Linesearch_tolerance=0.6,
# )

# Solve your optimization problem
optimizer.solve()

# Print results of optimization
optimizer.print_results()

recorder.execute()
t = thrust.value
q2 = torque.value
print("thrust", t)
print("optimized torque", q2)
print("percent improvement", (q2 - q1)/q1 * 100)
print("optimized radius", radius.value)
