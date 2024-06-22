import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.utils.plot import make_polarplot
import numpy as np
from BladeAD.core.BEM.bem_model import BEMModel
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

num_nodes = 1
num_radial_interp = 50
num_azimuthal = 40

num_blades = 2

radius = 0.1524
chord = np.array([0.01605, 0.02392, 0.03073, 0.03195, 0.03015, 0.02751, 0.02413, 0.01958, 0.01455, 0.01130])
twist = np.deg2rad(np.array([20.8, 32.5, 27.9, 21.5, 18.0, 14.5, 11.0, 9.5, 8.0, 6.5]))
norm_radius = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

chord_fun = interp1d(norm_radius, chord, kind="quadratic")
twist_fun = interp1d(norm_radius, twist, kind="quadratic")
x_norm_interp = np.linspace(0.1, 1., num_radial_interp)
chord_interp = chord_fun(x_norm_interp)
twist_interp = twist_fun(x_norm_interp)

quad_pitch = np.deg2rad(5)

thrust_vector=csdl.Variable(value=np.array([1 * np.sin(quad_pitch), 0, -1 * np.cos(quad_pitch)]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))

mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 1] = 10 #np.linspace(0, 10.0, num_nodes) # np.linspace(0, 50.06, num_nodes)
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=3864 * np.ones((num_nodes,)))


naca_4412_2d_model = TwoDMLAirfoilModel(
    airfoil_name="naca_4412"
)

clark_y_2d_model = TwoDMLAirfoilModel(
    airfoil_name="clark_y"
)

e63_2d_model = TwoDMLAirfoilModel(
    airfoil_name="e63",
    plot_model=False,
)

airfoil_model = CompositeAirfoilModel(
    sections=[0., 0.2, 0.8, 1.],
    airfoil_models=[naca_4412_2d_model, e63_2d_model, clark_y_2d_model],
    transition_window=8,
)


bem_mesh_parameters = RotorMeshParameters(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    chord_profile=chord_interp,
    twist_profile=twist_interp, 
    radius=radius,
    num_radial=num_radial_interp,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
    norm_hub_radius=0.0542
)

bem_inputs = RotorAnalysisInputs(
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

outputs = bem_model.evaluate(inputs=bem_inputs)


T = outputs.total_thrust.value
Q = outputs.total_torque.value



print("thrust Peters--He", T)
print("torque Peters--He", Q)
make_polarplot(outputs.sectional_thrust, plot_contours=False, azimuthal_offset=-90)

exit()

torque = outputs.total_torque
torque.set_as_objective()
thrust = outputs.total_thrust
thrust.set_as_constraint(upper=5.5, lower=5.5, scaler=1/5)


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

